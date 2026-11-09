"""Build and verify self-contained authoring dossiers for committed tasks.

The canonical ARC payload remains in ``committed_tasks/tasks``.  A dossier is
author-only material: it contains the original authoring record, the complete
candidate review record (including hidden answers), relevant raw model-call
artifacts, and immutable run-context snapshots.  Dossiers must never be used
as solver-facing task exports.

There is deliberately no automatic hook from generation into this module.
Materializing or refreshing dossiers requires the explicit authorization flag,
just like adding a task to the committed good-task registry.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .util import atomic_write_json, read_json
from .v4_diagnostics import complete_task


DOSSIER_SCHEMA_VERSION = 1
CONTEXT_FILES = (
    "config.snapshot.json",
    "config.snapshot.yaml",
    "implementation_manifest.json",
    "prompts/manifest.json",
    "factory.input_manifest.json",
    "factory.supply_design.private.json",
    "factory.requested.yaml",
    "factory.downstream_runtime.json",
    "factory.review_policy.json",
    "factory.review.prepared.json",
    "factory.review.report.json",
)


class CommittedDossierError(ValueError):
    """Raised when a committed-task dossier cannot be built or verified."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CommittedDossierError(f"path escapes repository root: {relative}") from exc
    return candidate


def _source_authoring_path(source_run: Path, candidate_id: str) -> Path:
    direct = source_run / "tasks" / f"{candidate_id}.authoring.json"
    if direct.is_file():
        return direct
    matches = sorted((source_run / "tasks").glob(f"**/{candidate_id}.authoring.json"))
    if len(matches) != 1:
        raise CommittedDossierError(
            f"expected one authoring record for {candidate_id} in {source_run}, found {len(matches)}"
        )
    return matches[0]


def _candidate_record(source_run: Path, candidate_id: str) -> dict[str, Any]:
    state_path = source_run / "state.json"
    state = read_json(state_path)
    records = state.get("records") if isinstance(state, Mapping) else None
    if not isinstance(records, Mapping):
        records = state.get("semantic_records") if isinstance(state, Mapping) else None
    record = records.get(candidate_id) if isinstance(records, Mapping) else None
    if not isinstance(record, dict):
        raise CommittedDossierError(
            f"source state has no complete candidate record for {candidate_id}: {state_path}"
        )
    return deepcopy(record)


def _authoring_task(authoring: Mapping[str, Any]) -> Any:
    task = authoring.get("task")
    if task is not None:
        return task
    realization = authoring.get("realization")
    if isinstance(realization, Mapping):
        if (
            isinstance(realization.get("task"), Mapping)
            and isinstance(realization.get("authoring_metadata"), Mapping)
            and realization["authoring_metadata"].get("expected_test_outputs")
        ):
            return complete_task(realization)
        return realization.get("task")
    return None


def _complete_authoring_task(authoring: Mapping[str, Any]) -> Any:
    """Recombine the solver-facing task with its author-only expected outputs."""

    task = _authoring_task(authoring)
    if not isinstance(task, Mapping):
        return task
    complete = deepcopy(dict(task))
    tests = complete.get("test")
    if not isinstance(tests, list):
        return complete
    if all(isinstance(pair, Mapping) and "output" in pair for pair in tests):
        return complete
    metadata = authoring.get("authoring_metadata")
    outputs = metadata.get("expected_test_outputs") if isinstance(metadata, Mapping) else None
    if not isinstance(outputs, list) or len(outputs) != len(tests):
        return complete
    for pair, output in zip(tests, outputs):
        if isinstance(pair, dict):
            pair["output"] = deepcopy(output)
    return complete


def _directory_mentions_candidate(directory: Path, candidate: bytes) -> bool:
    if candidate in directory.name.encode("utf-8", errors="ignore"):
        return True
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    if candidate in chunk:
                        return True
        except OSError:
            continue
    return False


def _relevant_call_directories(source_run: Path, candidate_id: str) -> list[Path]:
    root = source_run / "model_calls"
    if not root.is_dir():
        return []
    candidate = candidate_id.encode("utf-8")
    call_directories: list[Path] = []
    for role in sorted(path for path in root.iterdir() if path.is_dir()):
        for call in sorted(path for path in role.iterdir() if path.is_dir()):
            if _directory_mentions_candidate(call, candidate):
                call_directories.append(call)
    return call_directories


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _manifest_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json" and path.parent == root:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return files


def _build_one(
    *,
    repository_root: Path,
    committed_root: Path,
    entry: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    registry_task_id = str(entry["registry_task_id"])
    candidate_id = str(entry["source_candidate_id"])
    source_run = _contained(repository_root, str(entry["source_run"]))
    task_path = _contained(committed_root, str(entry["task_path"]))
    if not source_run.is_dir():
        raise CommittedDossierError(f"source run is missing for {registry_task_id}: {source_run}")
    if not task_path.is_file():
        raise CommittedDossierError(f"committed task is missing for {registry_task_id}: {task_path}")

    authoring_path = _source_authoring_path(source_run, candidate_id)
    authoring = read_json(authoring_path)
    committed_task = read_json(task_path)
    review = _candidate_record(source_run, candidate_id)
    review_task = review.get("complete_task")
    if review_task is not None and review_task != committed_task:
        raise CommittedDossierError(
            f"candidate review task differs from committed ARC payload for {registry_task_id}"
        )
    original_authoring_task = _complete_authoring_task(authoring)
    authoring_differs_after_repair = original_authoring_task != committed_task
    if authoring_differs_after_repair and not (
        review_task == committed_task and int(review.get("repair_count", 0)) > 0
    ):
        raise CommittedDossierError(
            "authoring task differs from committed ARC payload without verified "
            f"repair lineage for {registry_task_id}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise CommittedDossierError(f"temporary dossier directory is not empty: {destination}")
    atomic_write_json(destination / "authoring.json", authoring)
    atomic_write_json(destination / "review.json", review)

    copied_context: list[dict[str, str]] = []
    for relative in CONTEXT_FILES:
        source = source_run / relative
        if not source.is_file():
            continue
        target = destination / "run_context" / relative
        _copy_file(source, target)
        copied_context.append(
            {
                "source_path": source.relative_to(repository_root).as_posix(),
                "stored_path": target.relative_to(destination).as_posix(),
            }
        )

    copied_calls: list[dict[str, Any]] = []
    model_calls_root = source_run / "model_calls"
    for call in _relevant_call_directories(source_run, candidate_id):
        relative = call.relative_to(model_calls_root)
        target = destination / "model_calls" / relative
        shutil.copytree(call, target)
        copied_calls.append(
            {
                "role": relative.parts[0],
                "call_id": relative.parts[1] if len(relative.parts) > 1 else relative.name,
                "source_path": call.relative_to(repository_root).as_posix(),
                "stored_path": target.relative_to(destination).as_posix(),
            }
        )

    provenance = {
        "schema_version": DOSSIER_SCHEMA_VERSION,
        "visibility": "author_only_contains_hidden_test_output_and_rule",
        "registry_entry_at_packaging": {
            key: value
            for key, value in entry.items()
            if key not in {"dossier_path", "dossier_sha256"}
        },
        "source_run": source_run.relative_to(repository_root).as_posix(),
        "source_candidate_id": candidate_id,
        "source_authoring": {
            "path": authoring_path.relative_to(repository_root).as_posix(),
            "sha256": _file_sha256(authoring_path),
            "differs_from_final_after_verified_repair": authoring_differs_after_repair,
        },
        "source_state": {
            "path": (source_run / "state.json").relative_to(repository_root).as_posix(),
            "sha256": _file_sha256(source_run / "state.json"),
        },
        "source_report": (
            {
                "path": (source_run / "report.json").relative_to(repository_root).as_posix(),
                "sha256": _file_sha256(source_run / "report.json"),
            }
            if (source_run / "report.json").is_file()
            else None
        ),
        "copied_run_context": copied_context,
        "copied_model_calls": copied_calls,
        "task_payload": {
            "path": task_path.relative_to(repository_root).as_posix(),
            "sha256": _file_sha256(task_path),
        },
    }
    atomic_write_json(destination / "provenance.json", provenance)

    manifest = {
        "schema_version": DOSSIER_SCHEMA_VERSION,
        "registry_task_id": registry_task_id,
        "visibility": "author_only",
        "contains_hidden_test_output": True,
        "contains_intended_rule": True,
        "source_run": str(entry["source_run"]),
        "source_candidate_id": candidate_id,
        "arc_sha256": str(entry["arc_sha256"]),
        "artifact_policy": {
            "authoring_record": "verbatim structured source artifact",
            "review_record": "verbatim candidate record extracted from source state",
            "repair_lineage": "If a verified repair changed the original realization, authoring.json preserves the original and review.json preserves the final repaired task and review history.",
            "model_calls": "exact copies of every call directory containing the candidate ID",
            "run_context": "exact immutable configuration, prompt-manifest, and implementation snapshots when available",
        },
        "files": _manifest_files(destination),
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


def materialize_committed_dossiers(
    registry_path: str | Path,
    *,
    authorize_user_decision: bool,
    registry_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Package registry entries and bind each dossier manifest by hash.

    ``registry_task_ids`` is used by explicit maintenance commits to package
    only newly authorized entries.  Existing hash-bound dossiers are left
    byte-for-byte intact.  Omitting it retains the historical full-refresh
    behavior.
    """

    if not authorize_user_decision:
        raise CommittedDossierError(
            "dossier materialization requires explicit user authorization"
        )
    registry_path = Path(registry_path).resolve()
    committed_root = registry_path.parent
    repository_root = committed_root.parent
    registry = read_json(registry_path)
    entries = registry.get("entries") if isinstance(registry, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise CommittedDossierError("committed registry has no entries")

    dossiers_root = committed_root / "dossiers"
    dossiers_root.mkdir(parents=True, exist_ok=True)
    built: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise CommittedDossierError("committed registry contains a malformed entry")
        registry_task_id = str(entry["registry_task_id"])
        if registry_task_ids is not None and registry_task_id not in registry_task_ids:
            continue
        temporary = Path(tempfile.mkdtemp(prefix=f".{registry_task_id}.", dir=dossiers_root))
        try:
            manifest = _build_one(
                repository_root=repository_root,
                committed_root=committed_root,
                entry=entry,
                destination=temporary,
            )
            target = dossiers_root / registry_task_id
            previous = dossiers_root / f".{registry_task_id}.previous"
            if previous.exists():
                shutil.rmtree(previous)
            if target.exists():
                target.replace(previous)
            temporary.replace(target)
            if previous.exists():
                shutil.rmtree(previous)
            manifest_path = target / "manifest.json"
            entry["dossier_path"] = manifest_path.relative_to(committed_root).as_posix()
            entry["dossier_sha256"] = _file_sha256(manifest_path)
            built.append(
                {
                    "registry_task_id": registry_task_id,
                    "dossier_path": entry["dossier_path"],
                    "dossier_sha256": entry["dossier_sha256"],
                    "artifact_count": len(manifest["files"]),
                }
            )
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    registry["dossier_policy"] = {
        "schema_version": DOSSIER_SCHEMA_VERSION,
        "required_for_every_committed_task": True,
        "visibility": "author_only_contains_hidden_answers",
        "raw_artifact_rule": "Relevant candidate model-call directories are copied verbatim into each dossier.",
        "solver_export_rule": "Dossier content must never enter solver-facing ARC exports.",
        "authorization": (
            "explicit user-authorized maintenance operations; each authorization "
            "is retained in the corresponding registry commit set"
        ),
    }
    atomic_write_json(registry_path, registry)
    return {
        "registry_path": str(registry_path),
        "registry_task_count": len(entries),
        "materialized_task_count": len(built),
        "dossiers": built,
    }


def verify_dossier_manifest(
    *,
    committed_root: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one registry-bound dossier and every artifact hash it declares."""

    relative = entry.get("dossier_path")
    expected_hash = entry.get("dossier_sha256")
    if not isinstance(relative, str) or not relative.strip():
        raise CommittedDossierError("committed entry has no dossier_path")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise CommittedDossierError("committed entry has no valid dossier_sha256")
    manifest_path = _contained(committed_root, relative)
    if not manifest_path.is_file():
        raise CommittedDossierError(f"dossier manifest is missing: {manifest_path}")
    actual_hash = _file_sha256(manifest_path)
    if actual_hash != expected_hash:
        raise CommittedDossierError(
            f"dossier manifest hash mismatch: {actual_hash} != {expected_hash}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != DOSSIER_SCHEMA_VERSION:
        raise CommittedDossierError("unsupported dossier schema")
    if manifest.get("registry_task_id") != entry.get("registry_task_id"):
        raise CommittedDossierError("dossier is bound to a different registry task")
    if manifest.get("arc_sha256") != entry.get("arc_sha256"):
        raise CommittedDossierError("dossier is bound to a different ARC payload")
    if manifest.get("visibility") != "author_only" or manifest.get("contains_hidden_test_output") is not True:
        raise CommittedDossierError("dossier visibility declaration is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise CommittedDossierError("dossier manifest has no artifact files")
    root = manifest_path.parent
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise CommittedDossierError("dossier file entry is malformed")
        path_text = item.get("path")
        if not isinstance(path_text, str) or path_text in seen:
            raise CommittedDossierError("dossier file path is missing or duplicated")
        seen.add(path_text)
        path = _contained(root, path_text)
        if not path.is_file() or _file_sha256(path) != item.get("sha256"):
            raise CommittedDossierError(f"dossier artifact failed hash verification: {path_text}")
    required = {"authoring.json", "review.json", "provenance.json"}
    if not required.issubset(seen):
        raise CommittedDossierError(
            f"dossier lacks required artifacts: {', '.join(sorted(required - seen))}"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="committed_tasks/registry.json")
    parser.add_argument("--authorize-user-decision", action="store_true")
    args = parser.parse_args(argv)
    result = materialize_committed_dossiers(
        args.registry,
        authorize_user_decision=args.authorize_user_decision,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())


__all__ = [
    "CommittedDossierError",
    "DOSSIER_SCHEMA_VERSION",
    "materialize_committed_dossiers",
    "verify_dossier_manifest",
]
