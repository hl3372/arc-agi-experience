"""Configuration and frozen-parent checks for the V4.3.1 quality pilot."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError
from .util import content_hash
from .v43_config import load_v43_config


V431_PIPELINE = "arcgen-v4.3.1-episode-joint-quality"
V431_DESIGN = "frozen_v43_kernel_global_routing_and_embodiment_v1"


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def verify_v43_source(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config["_repository_root"])).resolve()
    settings = config["authoring_quality"]
    source = Path(str(settings["source_v4_3_run"]))
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_dir():
        raise ConfigError(f"V4.3 source run does not exist: {source}")
    expected = settings.get("source_artifact_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ConfigError("authoring_quality.source_artifact_sha256 is required")
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
        raise ConfigError("frozen V4.3 source changed: " + " | ".join(failures))
    return {
        "source_run": str(source),
        "artifact_sha256": verified,
        "snapshot_id": content_hash({"source": str(source), "artifacts": verified}),
    }


def load_v431_config(path: str | Path) -> dict[str, Any]:
    config = load_v43_config(path)
    if config.get("v4_3_1_pipeline") != V431_PIPELINE:
        raise ConfigError(f"v4_3_1_pipeline must be {V431_PIPELINE!r}")
    if config.get("v4_3_1_design") != V431_DESIGN:
        raise ConfigError(f"v4_3_1_design must be {V431_DESIGN!r}")
    settings = config.get("authoring_quality")
    if not isinstance(settings, Mapping):
        raise ConfigError("authoring_quality must be a mapping")
    expected_count = _integer(
        settings.get("expected_kernel_count"),
        "authoring_quality.expected_kernel_count",
        minimum=1,
    )
    selected = _integer(
        settings.get("maximum_global_selection"),
        "authoring_quality.maximum_global_selection",
        minimum=1,
    )
    realized = _integer(
        settings.get("maximum_realizations"),
        "authoring_quality.maximum_realizations",
        minimum=1,
    )
    if expected_count != 24 or selected != 8 or realized != 6 or realized > selected:
        raise ConfigError("the first V4.3.1 pilot is fixed at 24 -> 8 -> 6")
    for field in (
        "global_same_procedure_deduplication",
        "no_new_idea_calls",
        "downstream_stack_unchanged",
    ):
        if settings.get(field) is not True:
            raise ConfigError(f"authoring_quality.{field} must be true")
    for field in ("automatic_promotion", "automatic_task_commit"):
        if settings.get(field) is not False:
            raise ConfigError(f"authoring_quality.{field} must be false")
    if settings.get("frozen_source_arm") != "treatment":
        raise ConfigError("the frozen source arm must be treatment")
    config["authoring_quality"] = deepcopy(dict(settings))
    config["_v4_3_source_verification"] = verify_v43_source(config)
    return config


__all__ = [
    "V431_DESIGN",
    "V431_PIPELINE",
    "load_v431_config",
    "verify_v43_source",
]
