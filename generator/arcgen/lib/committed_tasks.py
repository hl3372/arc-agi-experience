"""Read-only committed generated-task registry and novelty-reference view.

Historical pipelines retain their own archives for reproducibility. New
campaigns use this registry as the sole persistent generated-task reference
source. There is intentionally no automatic append API in this module.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from .validation import validate_arc_task


class CommittedTaskRegistryError(ValueError):
    """Raised when the committed registry or one of its tasks is invalid."""


REQUIRED_ENTRY_FIELDS = {
    "registry_task_id",
    "title",
    "task_path",
    "arc_sha256",
    "source_pipeline",
    "source_run",
    "source_candidate_id",
    "search_mode",
    "intrinsic_status",
    "portfolio_status",
    "commit_set_id",
    "concept_summary",
    "intended_rule",
    "dossier_path",
    "dossier_sha256",
}

COMMITTABLE_INTRINSIC_STATUSES = {
    "core_quality",
    "reserve_quality",
    # Kept in the legacy status field for registry compatibility. This is an
    # evidence disposition, not an editorial quality grade.
    "boardless_verified",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CommittedTaskRegistryError(f"cannot read valid JSON from {path}: {exc}") from exc


def _resolve_contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise CommittedTaskRegistryError("task_path must be a non-empty string")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CommittedTaskRegistryError(
            f"committed task path escapes registry directory: {relative}"
        ) from exc
    return candidate


def load_committed_task_registry(
    path: str | Path, *, asset_root: str | Path | None = None
) -> dict[str, Any]:
    """Load and fully verify an immutable committed-task registry.

    ``asset_root`` is used only for a frozen registry snapshot whose task and
    dossier payloads remain in the original committed-task directory.  The
    default remains the registry file's parent, and every referenced path is
    still required to stay within the selected root.
    """

    registry_path = Path(path).resolve()
    committed_root = (
        Path(asset_root).resolve() if asset_root is not None else registry_path.parent
    )
    if not committed_root.is_dir():
        raise CommittedTaskRegistryError(
            f"committed asset root is missing: {committed_root}"
        )
    raw = _read_json(registry_path)
    if not isinstance(raw, dict):
        raise CommittedTaskRegistryError("committed registry must be a JSON object")
    if raw.get("schema_version") != 1:
        raise CommittedTaskRegistryError("unsupported committed registry schema")
    if raw.get("frozen") is not True:
        raise CommittedTaskRegistryError("committed registry must be frozen")
    if raw.get("automatic_writes_allowed") is not False:
        raise CommittedTaskRegistryError("automatic registry writes must be disabled")
    if raw.get("requires_explicit_user_authorization") is not True:
        raise CommittedTaskRegistryError(
            "registry must require explicit user authorization for commits"
        )
    dossier_policy = raw.get("dossier_policy")
    if not isinstance(dossier_policy, dict) or dossier_policy.get(
        "required_for_every_committed_task"
    ) is not True:
        raise CommittedTaskRegistryError(
            "registry must require an author-only dossier for every task"
        )
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise CommittedTaskRegistryError("registry entries must be an array")

    commit_sets = raw.get("commit_sets")
    if not isinstance(commit_sets, list) or not commit_sets:
        raise CommittedTaskRegistryError("registry requires at least one commit set")
    commit_set_ids = {
        item.get("commit_set_id")
        for item in commit_sets
        if isinstance(item, dict) and isinstance(item.get("commit_set_id"), str)
    }
    declared_count = sum(
        item.get("entry_count", 0)
        for item in commit_sets
        if isinstance(item, dict) and type(item.get("entry_count")) is int
    )
    if declared_count != len(entries):
        raise CommittedTaskRegistryError(
            f"commit-set entry count {declared_count} does not match {len(entries)} entries"
        )

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    verified: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        location = f"entries[{index}]"
        if not isinstance(entry, dict):
            raise CommittedTaskRegistryError(f"{location} must be an object")
        missing = sorted(REQUIRED_ENTRY_FIELDS - set(entry))
        if missing:
            raise CommittedTaskRegistryError(
                f"{location} is missing required fields: {', '.join(missing)}"
            )
        for field in REQUIRED_ENTRY_FIELDS:
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise CommittedTaskRegistryError(
                    f"{location}.{field} must be a non-empty string"
                )
        if entry["intrinsic_status"] not in COMMITTABLE_INTRINSIC_STATUSES:
            raise CommittedTaskRegistryError(
                f"{location} is not committable good-task quality: "
                f"{entry['intrinsic_status']}"
            )
        if entry["intrinsic_status"] == "boardless_verified" and (
            entry.get("editorial_mode") != "bypass"
            or entry.get("quality_grade") != "unassigned"
            or entry.get("boardless_disposition") != "verified_candidate"
            or "source_intrinsic_status" not in entry
            or entry["source_intrinsic_status"] is not None
            or entry.get("answer_mode") != "single_answer"
        ):
            raise CommittedTaskRegistryError(
                f"{location} has inconsistent boardless admission metadata"
            )
        if entry["portfolio_status"] != "include_now":
            raise CommittedTaskRegistryError(
                f"{location} is not headline-included: {entry['portfolio_status']}"
            )
        if entry["commit_set_id"] not in commit_set_ids:
            raise CommittedTaskRegistryError(
                f"{location} names unknown commit set {entry['commit_set_id']}"
            )
        task_id = entry["registry_task_id"]
        if task_id in seen_ids:
            raise CommittedTaskRegistryError(f"duplicate registry_task_id: {task_id}")
        seen_ids.add(task_id)
        relative_path = entry["task_path"]
        if relative_path in seen_paths:
            raise CommittedTaskRegistryError(f"duplicate task_path: {relative_path}")
        seen_paths.add(relative_path)
        task_path = _resolve_contained(committed_root, relative_path)
        if not task_path.is_file():
            raise CommittedTaskRegistryError(f"committed task is missing: {task_path}")
        actual_hash = sha256(task_path.read_bytes()).hexdigest()
        if actual_hash != entry["arc_sha256"]:
            raise CommittedTaskRegistryError(
                f"hash mismatch for {task_id}: {actual_hash} != {entry['arc_sha256']}"
            )
        if actual_hash in seen_hashes:
            raise CommittedTaskRegistryError(
                f"duplicate committed ARC payload hash: {actual_hash}"
            )
        seen_hashes.add(actual_hash)
        task = _read_json(task_path)
        validation = validate_arc_task(task, require_test_outputs=True)
        if not validation.valid:
            raise CommittedTaskRegistryError(
                f"invalid committed ARC task {task_id}: {'; '.join(validation.errors)}"
            )
        try:
            from .committed_dossiers import verify_dossier_manifest

            dossier = verify_dossier_manifest(
                committed_root=committed_root,
                entry=entry,
            )
        except (ValueError, OSError) as exc:
            raise CommittedTaskRegistryError(
                f"invalid committed dossier {task_id}: {exc}"
            ) from exc
        item = deepcopy(entry)
        item["absolute_task_path"] = str(task_path)
        item["absolute_dossier_path"] = str(
            _resolve_contained(committed_root, entry["dossier_path"])
        )
        item["dossier_artifact_count"] = len(dossier["files"])
        item["train_count"] = validation.train_count
        item["test_count"] = validation.test_count
        verified.append(item)

    result = deepcopy(raw)
    result["entries"] = verified
    result["registry_path"] = str(registry_path)
    result["registry_asset_root"] = str(committed_root)
    return result


def committed_generated_novelty_references(
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the only persistent generated references allowed for new work."""

    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise CommittedTaskRegistryError("verified registry has no entries array")
    references: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise CommittedTaskRegistryError("verified registry contains a malformed entry")
        references.append(
            {
                "reference_id": f"committed:{entry['registry_task_id']}",
                "source_kind": "committed_generated_task",
                "registry_task_id": entry["registry_task_id"],
                "title": entry["title"],
                "rule_description": entry["intended_rule"],
                "concept_summary": entry["concept_summary"],
                "combined_text": (
                    f"{entry['concept_summary']}\n{entry['intended_rule']}"
                ),
                "task_path": entry["absolute_task_path"],
                "arc_sha256": entry["arc_sha256"],
            }
        )
    return references


def assert_generated_references_are_committed(
    references: Iterable[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> None:
    """Reject a persistent generated reference not backed by a commit."""

    allowed = {
        f"committed:{entry['registry_task_id']}"
        for entry in registry.get("entries", [])
        if isinstance(entry, Mapping)
    }
    for reference in references:
        if reference.get("source_kind") != "committed_generated_task":
            raise CommittedTaskRegistryError(
                "persistent generated reference is not marked committed_generated_task"
            )
        if reference.get("reference_id") not in allowed:
            raise CommittedTaskRegistryError(
                f"uncommitted generated reference: {reference.get('reference_id')!r}"
            )


def automatic_commit_forbidden() -> None:
    """Fail loudly if orchestration attempts an automatic persistent commit."""

    raise CommittedTaskRegistryError(
        "automatic generated-task commits are disabled; explicit user authorization is required"
    )


__all__ = [
    "COMMITTABLE_INTRINSIC_STATUSES",
    "CommittedTaskRegistryError",
    "assert_generated_references_are_committed",
    "automatic_commit_forbidden",
    "committed_generated_novelty_references",
    "load_committed_task_registry",
]
