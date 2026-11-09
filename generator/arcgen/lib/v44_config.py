"""Configuration and frozen-input checks for the V4.4 abstraction study."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json

from .config import ConfigError
from .util import content_hash
from .v43_config import load_v43_config


V44_PIPELINE = "arcgen-v4.4-exp-abstraction-led"
V44_DESIGN = "frozen_control_matched_abstraction_idea_study_v1"


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def verify_v43_control_source(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config["_repository_root"])).resolve()
    settings = config["abstraction_led"]
    source = Path(str(settings["source_v4_3_run"]))
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_dir():
        raise ConfigError(f"V4.3 frozen control source does not exist: {source}")
    expected = settings.get("source_artifact_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ConfigError("abstraction_led.source_artifact_sha256 is required")
    verified: dict[str, str] = {}
    failures: list[str] = []
    for relative, digest in expected.items():
        path = source / str(relative)
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"changed {relative}: {actual} != {digest}")
        verified[str(relative)] = actual
    if failures:
        raise ConfigError("frozen V4.3 control source changed: " + " | ".join(failures))
    return {
        "source_run": str(source),
        "artifact_sha256": verified,
        "snapshot_id": content_hash({"source": str(source), "artifacts": verified}),
    }


def load_experimental_soft_memory(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load optional, experiment-local mechanism warnings for supplemental slates.

    This is deliberately separate from the production reservoir and committed
    concept bank.  It lets a repeated experimental sample warn both blinded
    arms about already-developed procedures without turning uncommitted work
    into permanent novelty memory.
    """

    configured = config["abstraction_led"].get(
        "experimental_soft_memory_manifest"
    )
    if configured in (None, ""):
        return {
            "enabled": False,
            "manifest_path": None,
            "manifest_sha256": None,
            "entries": [],
        }
    root = Path(str(config["_repository_root"])).resolve()
    path = Path(str(configured))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(
            f"experimental soft-memory manifest does not exist: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read experimental soft-memory manifest: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ConfigError("experimental soft-memory manifest must use schema_version 1")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("experimental soft-memory manifest must contain entries")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise ConfigError(f"experimental soft-memory entry {index} must be an object")
        if any(key in item for key in ("private_arm", "experimental_arm", "source_arm")):
            raise ConfigError(
                "experimental soft-memory entries cannot expose private arm labels"
            )
        summary = " ".join(str(item.get("mechanism_summary", "")).split())
        if not summary:
            raise ConfigError(
                f"experimental soft-memory entry {index} has no mechanism_summary"
            )
        digest = content_hash(summary)
        if digest in seen:
            continue
        seen.add(digest)
        normalized.append(
            {
                "mechanism_summary": summary[:900],
                "reason": "Prior experiment developed this procedure; avoid the same minimal solution procedure, while treating its broad theme as available.",
            }
        )
    return {
        "enabled": True,
        "manifest_path": str(path),
        "manifest_sha256": sha256(path.read_bytes()).hexdigest(),
        "entries": normalized,
    }


def load_v44_config(path: str | Path) -> dict[str, Any]:
    config = load_v43_config(path)
    if config.get("v4_4_pipeline") != V44_PIPELINE:
        raise ConfigError(f"v4_4_pipeline must be {V44_PIPELINE!r}")
    if config.get("v4_4_design") != V44_DESIGN:
        raise ConfigError(f"v4_4_design must be {V44_DESIGN!r}")
    flags = config.get("v4_4_features")
    if not isinstance(flags, Mapping):
        raise ConfigError("v4_4_features must be a mapping")
    if flags.get("abstraction_led_idea_treatment") is not True:
        raise ConfigError("the isolated abstraction-led treatment must be enabled")
    if flags.get("production_promotion") is not False:
        raise ConfigError("V4.4 cannot promote itself")

    settings = config.get("abstraction_led")
    if not isinstance(settings, Mapping):
        raise ConfigError("abstraction_led must be a mapping")
    fixed = {
        "frozen_control_kernel_count": 24,
        "independent_slates": 3,
        "treatment_generation_calls_per_slate": 2,
        "ideas_per_call": 4,
        "shard_maximum_per_arm": 4,
        "global_maximum_per_arm": 8,
        "maximum_global_development": 16,
    }
    for field, expected in fixed.items():
        actual = _integer(settings.get(field), f"abstraction_led.{field}", minimum=1)
        if actual != expected:
            raise ConfigError(
                f"abstraction_led.{field} is fixed at {expected} for this study"
            )
    for field in (
        "audit_batch_size",
        "minimum_genuine_completed",
        "minimum_mechanism_clusters",
        "escalation_calls_per_arm_per_slate",
    ):
        _integer(settings.get(field), f"abstraction_led.{field}", minimum=1)
    fraction = settings.get("maximum_largest_cluster_fraction")
    if not isinstance(fraction, (int, float)) or not 0 < float(fraction) <= 1:
        raise ConfigError(
            "abstraction_led.maximum_largest_cluster_fraction must be in (0, 1]"
        )
    for field in (
        "shared_quality_gate_for_both_arms",
        "source_labels_hidden_downstream",
        "reuse_frozen_control_supply",
    ):
        if settings.get(field) is not True:
            raise ConfigError(f"abstraction_led.{field} must be true")
    for field in ("automatic_promotion", "automatic_task_commit"):
        if settings.get(field) is not False:
            raise ConfigError(f"abstraction_led.{field} must be false")
    if config["run"].get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("V4.4 experimental tasks cannot be committed automatically")
    config["abstraction_led"] = deepcopy(dict(settings))
    config["_v4_3_control_source_verification"] = verify_v43_control_source(config)
    config["_v44_experimental_soft_memory"] = load_experimental_soft_memory(config)
    return config


__all__ = [
    "V44_DESIGN",
    "V44_PIPELINE",
    "load_experimental_soft_memory",
    "load_v44_config",
    "verify_v43_control_source",
]
