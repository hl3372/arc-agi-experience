"""Configuration and frozen-parent checks for the V4.5 verification study."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import tarfile

from .config import ConfigError
from .util import content_hash, read_json
from .v44_control_supplement import load_v44_control_supplement_config


V45_PIPELINE = "arcgen-v4.5-exp-answer-aware"
V45_VERSION = "arcgen-v4.5-answer-aware-verification-4.5.0"


def _read(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"V4.5 configuration root must be a mapping: {path}")
    return value


def _resolve(root: Path, value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    path = Path(value)
    return str((path if path.is_absolute() else root / path).resolve())


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _verify_registry_or_authorized_append(
    *, root: Path, manifest: Mapping[str, Any], registry_path: Path
) -> dict[str, Any]:
    """Keep the frozen V4.4 bank immutable while permitting later commits.

    A historical baseline cannot bind a moving append-only registry by its
    current whole-file hash forever.  The V4.4 archive remains the immutable
    source of truth for the original entries; the live registry is accepted
    only when that exact frozen registry is an unchanged prefix and all later
    entries arrived through internally consistent commit sets.
    """

    declaration = manifest.get("committed_bank", {})
    expected_hash = declaration.get("registry_sha256")
    expected_count = declaration.get("task_count")
    current_bytes = registry_path.read_bytes()
    if sha256(current_bytes).hexdigest() == expected_hash:
        return {
            "mode": "exact_frozen_registry",
            "frozen_task_count": expected_count,
            "live_task_count": expected_count,
        }

    archive_relative = (manifest.get("snapshot") or {}).get("archive")
    archive = root / str(archive_relative or "")
    if not archive.is_file():
        raise ConfigError("V4.4 snapshot archive is unavailable for append-only verification")
    try:
        with tarfile.open(archive, "r:gz") as handle:
            member = handle.extractfile("committed_tasks/registry.json")
            if member is None:
                raise ConfigError("V4.4 snapshot has no frozen committed-task registry")
            frozen_bytes = member.read()
        if sha256(frozen_bytes).hexdigest() != expected_hash:
            raise ConfigError("V4.4 snapshot registry does not match its frozen hash")
        frozen = json.loads(frozen_bytes)
        current = json.loads(current_bytes)
    except (OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot verify the V4.4 frozen registry: {exc}") from exc

    frozen_entries = frozen.get("entries") if isinstance(frozen, Mapping) else None
    current_entries = current.get("entries") if isinstance(current, Mapping) else None
    frozen_sets = frozen.get("commit_sets") if isinstance(frozen, Mapping) else None
    current_sets = current.get("commit_sets") if isinstance(current, Mapping) else None
    if not isinstance(frozen_entries, list) or len(frozen_entries) != expected_count:
        raise ConfigError("V4.4 snapshot registry has an unexpected task count")
    if not isinstance(current_entries, list) or current_entries[:expected_count] != frozen_entries:
        raise ConfigError("live committed-task registry changed the frozen V4.4 entry prefix")
    if not isinstance(frozen_sets, list) or not isinstance(current_sets, list):
        raise ConfigError("committed-task registry has malformed commit sets")
    if current_sets[: len(frozen_sets)] != frozen_sets:
        raise ConfigError("live committed-task registry changed the frozen V4.4 commit sets")
    for field in (
        "schema_version",
        "registry_id",
        "frozen",
        "automatic_writes_allowed",
        "requires_explicit_user_authorization",
        "dossier_policy",
    ):
        if current.get(field) != frozen.get(field):
            raise ConfigError(
                f"live committed-task registry changed frozen policy field {field}"
            )
    declared = sum(
        item.get("entry_count", 0)
        for item in current_sets
        if isinstance(item, Mapping) and type(item.get("entry_count")) is int
    )
    if declared != len(current_entries):
        raise ConfigError("live committed-task registry commit-set counts are inconsistent")
    if len(current_entries) <= expected_count:
        raise ConfigError("live committed-task registry differs without an append-only extension")
    return {
        "mode": "verified_authorized_append_only_extension",
        "frozen_task_count": expected_count,
        "live_task_count": len(current_entries),
    }


def verify_v44_baseline(repository_root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get("baseline_id") != "arcgen-v4.4-abstraction-led-20260806":
        raise ConfigError("unexpected V4.4 frozen baseline")
    failures: list[str] = []
    checked: dict[str, str] = {}
    for section in ("critical_source_sha256", "run_report_sha256"):
        for relative, expected in manifest.get(section, {}).items():
            source = root / relative
            if not source.is_file():
                failures.append(f"missing {relative}")
                continue
            actual = _sha(source)
            checked[relative] = actual
            if actual != expected:
                failures.append(f"changed {relative}: {actual} != {expected}")
    feature_path = root / str(manifest["feature_manifest"])
    if not feature_path.is_file() or _sha(feature_path) != manifest["feature_manifest_sha256"]:
        failures.append("V4.4 feature manifest changed")
    registry = root / str(manifest["committed_bank"]["registry"])
    registry_status = None
    if not registry.is_file():
        failures.append("V4.4 committed-task registry is missing")
    else:
        registry_status = _verify_registry_or_authorized_append(
            root=root,
            manifest=manifest,
            registry_path=registry,
        )
    if failures:
        raise ConfigError("frozen V4.4 baseline changed: " + " | ".join(failures))
    return {
        "baseline_id": manifest["baseline_id"],
        "manifest_path": str(path),
        "manifest_sha256": _sha(path),
        "verified_artifact_count": len(checked) + 2,
        "committed_task_count": manifest["committed_bank"]["task_count"],
        "snapshot_archive": str((root / manifest["snapshot"]["archive"]).resolve()),
        "verification_hash": content_hash(checked),
    }


def load_v45_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read(source)
    if str(raw.get("version")) != "4.5" or raw.get("pipeline") != V45_PIPELINE:
        raise ConfigError(f"V4.5 must use version 4.5 and pipeline {V45_PIPELINE}")
    base_path = Path(str(raw.get("base_v4_4_config", "")))
    base_path = (source.parent / base_path).resolve() if not base_path.is_absolute() else base_path.resolve()
    base = load_v44_control_supplement_config(base_path)
    # The checksummed V4.4 base, rather than the location of an optional
    # override file, defines the repository whose frozen artifacts are checked.
    root = Path(base["_repository_root"]).resolve()
    baseline_path = _resolve(root, raw.get("v4_4_baseline_manifest"), "v4_4_baseline_manifest")
    baseline = verify_v44_baseline(root, baseline_path)

    experiment = raw.get("answer_aware_experiment")
    if not isinstance(experiment, Mapping):
        raise ConfigError("answer_aware_experiment settings are required")
    fixed_false = (
        "gating",
        "auto_promote",
        "production_state_mutation_allowed",
        "good_task_bank_mutation_allowed",
        "reservoir_mutation_allowed",
        "final_batch_judging_enabled",
    )
    for field in fixed_false:
        if experiment.get(field) is not False:
            raise ConfigError(f"answer_aware_experiment.{field} must be false")
    if experiment.get("enabled") is not True:
        raise ConfigError("the V4.5 experiment must be explicitly enabled")
    historical_minimum = experiment.get("historical_minimum")
    historical_maximum = experiment.get("historical_maximum")
    fresh_count = experiment.get("fresh_development_count")
    if type(historical_minimum) is not int or historical_minimum < 15:
        raise ConfigError("historical_minimum must be at least 15")
    if type(historical_maximum) is not int or not historical_minimum <= historical_maximum <= 25:
        raise ConfigError("historical_maximum must be between the minimum and 25")
    if type(fresh_count) is not int or not 1 <= fresh_count <= 8:
        raise ConfigError("fresh_development_count must be between 1 and 8")
    if experiment.get("control_role") != "blind_solver" or experiment.get("treatment_role") != "blind_solver":
        raise ConfigError("both verification arms must use the frozen blind-solver model role")
    if experiment.get("missing_negative_evidence_policy") != "downgrade_to_inconclusive_without_retry":
        raise ConfigError("V4.5 must downgrade unsupported negative claims without retry")

    config = deepcopy(base)
    config["version"] = "4.5"
    config["pipeline"] = V45_PIPELINE
    config["seed"] = raw.get("seed", base["seed"])
    if type(config["seed"]) is not int:
        raise ConfigError("seed must be an integer")
    config["experimental"] = {
        "enabled": True,
        "kind": "answer_aware_verification",
        "gating": False,
        "auto_promote": False,
    }
    config["answer_aware_experiment"] = deepcopy(dict(experiment))
    config["run"] = _merge(base["run"], raw.get("run", {}))
    if config["run"].get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("V4.5 cannot commit experimental tasks")
    if not isinstance(config["run"].get("id"), str) or not config["run"]["id"].strip():
        raise ConfigError("run.id must be non-empty")
    config["paths"] = deepcopy(base["paths"])
    for key, value in raw.get("paths", {}).items():
        config["paths"][key] = _resolve(root, value, f"paths.{key}")
    if "v4_5_historical_case_manifest" not in config["paths"]:
        raise ConfigError("paths.v4_5_historical_case_manifest is required")
    config["paths"]["v4_4_feature_manifest"] = str(
        (root / "config/v4_4_feature_manifest.json").resolve()
    )
    config["control_supplement"] = deepcopy(base["control_supplement"])
    config["control_supplement"]["maximum_development"] = int(fresh_count)

    blind = config["model"]["roles"]["blind_solver"]
    if blind != {"model": "gpt-5.6-sol", "reasoning_effort": "xhigh"}:
        raise ConfigError("V4.5 initial arms must retain Sol/xhigh")

    config["_config_path"] = str(source)
    config["_repository_root"] = str(root.resolve())
    config["_v4_4_base_config"] = str(base_path)
    config["_v4_4_baseline_verification"] = baseline
    return config


def snapshot_v45_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


__all__ = [
    "V45_PIPELINE",
    "V45_VERSION",
    "load_v45_config",
    "snapshot_v45_config",
    "verify_v44_baseline",
]
