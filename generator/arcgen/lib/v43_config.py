"""Configuration and frozen-baseline checks for the V4.3 episode-joint study."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import tarfile

from .config import ConfigError
from .util import content_hash, read_json
from .v42_config import load_v42_config


V43_PIPELINE = "arcgen-v4.3-exp-episode-joint"
V43_DESIGN = "matched_episode_joint_idea_study_v1"
V42_BASELINE_ID = "arcgen-v4.2-logical-leap-20260805"


def _verify_registry_or_authorized_append(
    *, root: Path, manifest: Mapping[str, Any], registry_path: Path
) -> dict[str, Any]:
    """Verify V4.2's registry while allowing later authorized append-only sets.

    The V4.2 archive is the immutable source of truth for its original 93
    entries.  A live committed-task bank is expected to grow after later user
    authorizations; requiring its whole-file hash to remain frozen would make
    every later version unrunnable.  We therefore compare the frozen prefix
    byte-for-structure and reject edits, reordering, or removal of any parent
    entry or commit set.
    """

    declaration = manifest.get("committed_bank", {})
    expected_hash = declaration.get("registry_sha256")
    current_bytes = registry_path.read_bytes()
    current_hash = sha256(current_bytes).hexdigest()
    expected_count = declaration.get("task_count")
    if current_hash == expected_hash:
        return {
            "mode": "exact_frozen_registry",
            "frozen_task_count": expected_count,
            "live_task_count": expected_count,
        }

    archive_relative = (manifest.get("snapshot") or {}).get("archive")
    archive = root / str(archive_relative or "")
    if not archive.is_file():
        raise ConfigError("V4.2 snapshot archive is unavailable for append-only verification")
    try:
        with tarfile.open(archive, "r:gz") as handle:
            member = handle.extractfile("committed_tasks/registry.json")
            if member is None:
                raise ConfigError("V4.2 snapshot has no frozen committed-task registry")
            frozen_bytes = member.read()
        if sha256(frozen_bytes).hexdigest() != expected_hash:
            raise ConfigError("V4.2 snapshot registry does not match its frozen hash")
        frozen = json.loads(frozen_bytes)
        current = json.loads(current_bytes)
    except (OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot verify the V4.2 frozen registry: {exc}") from exc

    frozen_entries = frozen.get("entries") if isinstance(frozen, Mapping) else None
    current_entries = current.get("entries") if isinstance(current, Mapping) else None
    frozen_sets = frozen.get("commit_sets") if isinstance(frozen, Mapping) else None
    current_sets = current.get("commit_sets") if isinstance(current, Mapping) else None
    if not isinstance(frozen_entries, list) or len(frozen_entries) != expected_count:
        raise ConfigError("V4.2 snapshot registry has an unexpected task count")
    if not isinstance(current_entries, list) or current_entries[:expected_count] != frozen_entries:
        raise ConfigError("live committed-task registry changed the frozen V4.2 entry prefix")
    if not isinstance(frozen_sets, list) or not isinstance(current_sets, list):
        raise ConfigError("committed-task registry has malformed commit sets")
    if current_sets[: len(frozen_sets)] != frozen_sets:
        raise ConfigError("live committed-task registry changed the frozen V4.2 commit sets")
    for field in (
        "schema_version",
        "registry_id",
        "frozen",
        "automatic_writes_allowed",
        "requires_explicit_user_authorization",
        "dossier_policy",
    ):
        if current.get(field) != frozen.get(field):
            raise ConfigError(f"live committed-task registry changed frozen policy field {field}")
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


def verify_v42_baseline(
    repository_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    manifest = read_json(Path(manifest_path).resolve())
    if manifest.get("baseline_id") != V42_BASELINE_ID or manifest.get("status") != "frozen":
        raise ConfigError("unexpected or unfrozen ARCGEN V4.2 baseline manifest")

    failures: list[str] = []
    for relative, expected in manifest.get("critical_source_sha256", {}).items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"changed {relative}: {actual} != {expected}")

    registry = manifest.get("committed_bank", {})
    registry_path = root / str(registry.get("registry", ""))
    registry_verification: dict[str, Any] | None = None
    if not registry_path.is_file():
        failures.append("missing frozen committed-task registry")
    else:
        try:
            registry_verification = _verify_registry_or_authorized_append(
                root=root, manifest=manifest, registry_path=registry_path
            )
        except ConfigError as exc:
            failures.append(str(exc))

    feature_path = root / str(manifest.get("feature_manifest", ""))
    if not feature_path.is_file():
        failures.append("missing V4.2 feature manifest")
    elif sha256(feature_path.read_bytes()).hexdigest() != manifest.get(
        "feature_manifest_sha256"
    ):
        failures.append("V4.2 feature manifest differs from the frozen snapshot")

    if failures:
        raise ConfigError("frozen V4.2 baseline changed: " + " | ".join(failures))
    qualification = manifest.get("qualification", {})
    if qualification.get("all_core_one_answer_cleared") is not True:
        raise ConfigError("V4.2 baseline lacks the one-answer qualification guarantee")
    return {
        "baseline_id": manifest["baseline_id"],
        "manifest_sha256": content_hash(manifest),
        "verified_critical_file_count": len(manifest.get("critical_source_sha256", {})),
        "committed_task_count": registry.get("task_count"),
        "live_committed_task_count": (
            registry_verification.get("live_task_count")
            if registry_verification
            else registry.get("task_count")
        ),
        "registry_verification_mode": (
            registry_verification.get("mode")
            if registry_verification
            else "unavailable"
        ),
        "qualification": deepcopy(qualification),
    }


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def load_v43_config(path: str | Path) -> dict[str, Any]:
    """Load V4.3 through the frozen V4.2-compatible runtime boundary.

    The internal V4.1 experiment slot remains an implementation detail.  The
    public identity, frozen parent, prompts, schemas, state and reports are all
    versioned separately as V4.3.
    """

    config = load_v42_config(path)
    root = Path(config["_repository_root"])
    if config.get("v4_3_pipeline") != V43_PIPELINE:
        raise ConfigError(f"v4_3_pipeline must be {V43_PIPELINE!r}")
    if config.get("v4_3_design") != V43_DESIGN:
        raise ConfigError(f"v4_3_design must be {V43_DESIGN!r}")
    baseline_path = config.get("paths", {}).get("v4_2_baseline_manifest")
    if not baseline_path:
        raise ConfigError("paths.v4_2_baseline_manifest is required")
    config["_v4_2_baseline_verification"] = verify_v42_baseline(root, baseline_path)

    settings = config.get("episode_joint")
    if not isinstance(settings, Mapping):
        raise ConfigError("episode_joint must be a mapping")
    slates = _integer(settings.get("independent_slates"), "episode_joint.independent_slates", minimum=1)
    calls = _integer(
        settings.get("paired_generation_calls_per_slate"),
        "episode_joint.paired_generation_calls_per_slate",
        minimum=1,
    )
    ideas = _integer(settings.get("ideas_per_call"), "episode_joint.ideas_per_call", minimum=1)
    selected = _integer(
        settings.get("maximum_development_per_arm_per_slate"),
        "episode_joint.maximum_development_per_arm_per_slate",
        minimum=1,
    )
    if (slates, calls, ideas, selected) != (3, 2, 4, 4):
        raise ConfigError(
            "the qualification design is fixed at three slates, two paired calls, "
            "four ideas per call, and up to four developments per arm"
        )
    if calls * ideas != 8:
        raise ConfigError("each arm must offer eight kernels per independent slate")
    if settings.get("normal_generators_unchanged") is not True:
        raise ConfigError("the control concept-led generator must remain unchanged")
    if settings.get("source_labels_hidden_downstream") is not True:
        raise ConfigError("episode-joint source labels must remain hidden downstream")
    if settings.get("paired_task_variants") is not False:
        raise ConfigError("episode-joint compares independently generated tasks, not variants")
    if settings.get("automatic_promotion") is not False:
        raise ConfigError("V4.3 cannot promote itself")
    if settings.get("automatic_task_commit") is not False:
        raise ConfigError("V4.3 cannot mutate the good-task bank")
    _integer(settings.get("minimum_genuine_completed"), "episode_joint.minimum_genuine_completed", minimum=1)
    _integer(settings.get("minimum_mechanism_clusters"), "episode_joint.minimum_mechanism_clusters", minimum=1)
    _integer(settings.get("audit_batch_size"), "episode_joint.audit_batch_size", minimum=1)
    _integer(
        settings.get("escalation_calls_per_arm_per_slate"),
        "episode_joint.escalation_calls_per_arm_per_slate",
    )
    config["episode_joint"] = deepcopy(dict(settings))
    return config


__all__ = [
    "V42_BASELINE_ID",
    "V43_DESIGN",
    "V43_PIPELINE",
    "load_v43_config",
    "verify_v42_baseline",
]
