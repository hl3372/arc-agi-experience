"""Maintenance-only explicit commits to the generated good-task bank."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from .factory_contracts import is_canonical_plan, display_value
from .committed_dossiers import materialize_committed_dossiers
from .committed_tasks import (
    COMMITTABLE_INTRINSIC_STATUSES,
    load_committed_task_registry,
)
from .lean_v3_rendering import build_contact_sheet, render_episode
from .util import atomic_write_json, atomic_write_text, read_json
from .v41_palette import palette_equivalent_hash
from .validation import validate_arc_task


class AuthorizedCommitError(ValueError):
    pass


def _portfolio_commit_override_allowed(
    record: Mapping[str, Any], request: Mapping[str, Any]
) -> bool:
    """Validate an explicitly authorized non-headline core-task commit.

    Crowding holds retain the historical override behavior.  A confirmed
    same-procedure sibling needs the stronger, separately recorded override;
    this is intentionally unavailable to normal generation orchestration.
    """

    placement = record.get("portfolio_placement") or record.get(
        "portfolio_placement_status"
    )
    if placement == "include_now":
        return False
    if placement == "same_procedure_duplicate":
        if request.get("authorize_same_procedure_override") is not True:
            raise AuthorizedCommitError(
                "same-procedure duplicate requires an explicit same-procedure override"
            )
        return True
    if placement != "hold_crowded" or request.get("authorize_portfolio_override") is not True:
        raise AuthorizedCommitError(
            f"{record.get('task_id', request.get('source_candidate_id'))} is not include-now"
        )
    intrinsic_status = _intrinsic_status(record)
    if intrinsic_status == "reserve_quality":
        if request.get("authorize_reserve_quality") is not True:
            raise AuthorizedCommitError(
                "reserve-quality commit requires an explicit reserve-quality override"
            )
        judgments = record.get("editorial_judgments")
        if not isinstance(judgments, list) or len(judgments) != 2:
            raise AuthorizedCommitError(
                "reserve-quality override requires both editorial judgments"
            )
        recommendations = {
            judgment.get("intrinsic_recommendation")
            for judgment in judgments
            if isinstance(judgment, Mapping)
        }
        if (
            len(recommendations) == 0
            or not recommendations.issubset({"core_quality", "reserve_quality"})
            or "reserve_quality" not in recommendations
            or any(
                not isinstance(judgment, Mapping)
                or judgment.get("same_minimal_procedure") is not False
                for judgment in judgments
            )
        ):
            raise AuthorizedCommitError(
                "reserve-quality override requires two nonduplicate core-or-reserve judgments"
            )
        return True
    if request.get("authorize_all_core_override") is True:
        return True
    judgments = record.get("editorial_judgments")
    if not isinstance(judgments, list) or len(judgments) != 2:
        raise AuthorizedCommitError("crowding override requires both editorial judgments")
    if any(
        judgment.get("same_minimal_procedure") is not False
        or judgment.get("intrinsic_recommendation") != "core_quality"
        or judgment.get("portfolio_recommendation") != "hold_crowded"
        for judgment in judgments
    ):
        raise AuthorizedCommitError(
            "crowding override cannot admit a same-procedure duplicate or non-core task"
        )
    return True


def _record_collection(state: Mapping[str, Any]) -> Mapping[str, Any]:
    records = state.get("semantic_records")
    if isinstance(records, Mapping):
        return records
    records = state.get("records")
    if isinstance(records, Mapping):
        return records
    raise AuthorizedCommitError("source run has no records or semantic_records")


def _intrinsic_status(record: Mapping[str, Any]) -> Any:
    return (
        record.get("intrinsic_quality")
        or record.get("intrinsic_quality_status")
        or record.get("status")
    )


def _source_portfolio_status(record: Mapping[str, Any]) -> str:
    value = record.get("portfolio_placement") or record.get(
        "portfolio_placement_status"
    )
    return str(value or "include_now")


def _concept_summary(record: Mapping[str, Any]) -> str:
    kernel = record.get("kernel")
    if isinstance(kernel, Mapping) and isinstance(kernel.get("central_relation"), str):
        return str(kernel["central_relation"])
    if isinstance(record.get("central_relation"), str):
        return str(record["central_relation"])
    raise AuthorizedCommitError("source record has no central relation")


def _intended_rule(record: Mapping[str, Any]) -> str:
    plan = record.get("concrete_plan")
    if is_canonical_plan(plan):
        return str(display_value(plan["public_contract"]["semantic_core"]["central_rule"], plan))
    if isinstance(plan, Mapping):
        semantic = plan.get("semantic_core")
        if isinstance(semantic, Mapping) and isinstance(semantic.get("central_rule"), str):
            return str(semantic["central_rule"])
    architecture = record.get("architecture")
    if isinstance(architecture, Mapping) and isinstance(
        architecture.get("intended_rule"), str
    ):
        return str(architecture["intended_rule"])
    realization = record.get("realization")
    if isinstance(realization, Mapping):
        metadata = realization.get("authoring_metadata")
        if isinstance(metadata, Mapping) and isinstance(
            metadata.get("intended_rule"), str
        ):
            return str(metadata["intended_rule"])
        if isinstance(realization.get("intended_rule"), str):
            return str(realization["intended_rule"])
    raise AuthorizedCommitError("source record has no intended rule")


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _repository_root(manifest_path: Path) -> Path:
    for parent in manifest_path.parents:
        if (parent / "committed_tasks" / "registry.json").is_file():
            return parent
    raise AuthorizedCommitError("cannot locate repository root from commit manifest")


def _write_catalog(committed_root: Path, registry: Mapping[str, Any]) -> None:
    lines = [
        "# Committed good-task catalog",
        "",
        "Commit sets: "
        + ", ".join(f"`{item['commit_set_id']}`" for item in registry["commit_sets"]),
        f"Count: {len(registry['entries'])}",
        "Automatic additions: disabled",
        "",
    ]
    index = 1
    for commit_set in registry["commit_sets"]:
        lines.extend(
            [
                f"## {commit_set['commit_set_id']}",
                "",
                commit_set.get("notes", ""),
                "",
            ]
        )
        for entry in registry["entries"]:
            if entry["commit_set_id"] != commit_set["commit_set_id"]:
                continue
            lines.append(
                f"{index}. [{entry['title']}]({entry['task_path']}) · "
                f"[author dossier]({entry['dossier_path']})"
            )
            index += 1
        lines.append("")
    lines.extend(
        [
            "Every entry has an exact ARC payload plus a hash-bound author-only dossier containing its rule, plan, realization witness, review record, and relevant raw model-call artifacts.",
            "",
        ]
    )
    atomic_write_text(committed_root / "CATALOG.md", "\n".join(lines))


def _write_checksums(committed_root: Path, registry: Mapping[str, Any]) -> None:
    rows = []
    for entry in registry["entries"]:
        rows.extend(
            [
                f"{entry['arc_sha256']}  {entry['task_path']}",
                f"{entry['dossier_sha256']}  {entry['dossier_path']}",
            ]
        )
    atomic_write_text(committed_root / "SHA256SUMS", "\n".join(sorted(rows)) + "\n")


def _render_bank(committed_root: Path, registry: Mapping[str, Any]) -> None:
    destination = committed_root / "renders"
    rows = []
    images = []
    readme = [
        "# Committed good-task renders",
        "",
        "Each render contains the complete episode, expected test output, and intended rule.",
        "",
        f"Count: {len(registry['entries'])}",
        "",
        "## Tasks",
        "",
    ]
    for index, entry in enumerate(registry["entries"], 1):
        task = read_json(Path(entry["absolute_task_path"]))
        filename = f"{index:02d}__{entry['registry_task_id']}.png"
        image = render_episode(
            task,
            title=entry["title"],
            subtitle=f"{entry['registry_task_id']} · {entry['search_mode']}",
            rule=entry["intended_rule"],
            destination=destination / filename,
        )
        images.append((entry["registry_task_id"], image))
        rows.append(
            {
                "index": index,
                "registry_task_id": entry["registry_task_id"],
                "title": entry["title"],
                "source_task": entry["task_path"],
                "render": filename,
                "arc_sha256": entry["arc_sha256"],
                "intended_rule_included": True,
            }
        )
        readme.append(
            f"{index}. [{entry['title']}]({filename}) — `{entry['registry_task_id']}`"
        )
    build_contact_sheet(images, destination / "contact_sheet.png")
    atomic_write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "source_registry": "../registry.json",
            "registry_id": registry["registry_id"],
            "task_count": len(rows),
            "contact_sheet": "contact_sheet.png",
            "intended_rules_included": True,
            "tasks": rows,
        },
    )
    readme.extend(["", "## Contact sheet", "", "[Open the complete contact sheet](contact_sheet.png).", ""])
    atomic_write_text(destination / "README.md", "\n".join(readme))


def commit_authorized_manifest(
    manifest_path: str | Path, *, authorize_user_decision: bool
) -> dict[str, Any]:
    if not authorize_user_decision:
        raise AuthorizedCommitError("explicit user authorization is required")
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise AuthorizedCommitError("unsupported commit manifest schema")
    root = _repository_root(manifest_path)
    committed_root = root / "committed_tasks"
    registry_path = committed_root / "registry.json"
    before = load_committed_task_registry(registry_path)
    raw = read_json(registry_path)
    commit_set_id = manifest["commit_set_id"]
    requested_ids = {item["registry_task_id"] for item in manifest["entries"]}
    existing_ids = {
        item["registry_task_id"]
        for item in raw["entries"]
        if item["commit_set_id"] == commit_set_id
    }
    if existing_ids:
        if existing_ids != requested_ids:
            raise AuthorizedCommitError("existing commit set differs from request")
        return {
            "idempotent": True,
            "commit_set_id": commit_set_id,
            "registry_count": len(raw["entries"]),
        }
    new_entries = []
    written_tasks: list[Path] = []
    source_cache: dict[str, Mapping[str, Any]] = {}
    existing_palette_hashes = {
        palette_equivalent_hash(read_json(Path(entry["absolute_task_path"])))
        for entry in before["entries"]
    }
    try:
        for request in manifest["entries"]:
            source_run_text = request.get("source_run", manifest.get("source_run"))
            if not isinstance(source_run_text, str) or not source_run_text:
                raise AuthorizedCommitError("commit entry has no source_run")
            if source_run_text not in source_cache:
                source_run = (root / source_run_text).resolve()
                source_cache[source_run_text] = _record_collection(
                    read_json(source_run / "state.json")
                )
            records = source_cache[source_run_text]
            candidate_id = request["source_candidate_id"]
            record = records.get(candidate_id)
            if not isinstance(record, Mapping):
                raise AuthorizedCommitError(f"missing source candidate {candidate_id}")
            intrinsic_status = _intrinsic_status(record)
            if intrinsic_status not in COMMITTABLE_INTRINSIC_STATUSES:
                raise AuthorizedCommitError(
                    f"{candidate_id} is not committable core-or-reserve quality"
                )
            declared_status = request.get("intrinsic_status", intrinsic_status)
            if declared_status != intrinsic_status:
                raise AuthorizedCommitError(
                    f"{candidate_id} intrinsic status differs from commit request"
                )
            if (
                intrinsic_status == "reserve_quality"
                and request.get("authorize_reserve_quality") is not True
            ):
                raise AuthorizedCommitError(
                    f"{candidate_id} reserve-quality commit lacks explicit authorization"
                )
            portfolio_override = _portfolio_commit_override_allowed(record, request)
            task = deepcopy(record.get("complete_task"))
            result = validate_arc_task(task, require_test_outputs=True)
            if not result.valid:
                raise AuthorizedCommitError(
                    f"invalid source task {candidate_id}: {'; '.join(result.errors)}"
                )
            if palette_equivalent_hash(task) in existing_palette_hashes:
                raise AuthorizedCommitError(
                    f"{candidate_id} is an exact or palette-equivalent committed duplicate"
                )
            existing_palette_hashes.add(palette_equivalent_hash(task))
            registry_task_id = request["registry_task_id"]
            relative = f"tasks/{registry_task_id}.json"
            target = committed_root / relative
            if target.exists():
                raise AuthorizedCommitError(f"target already exists: {target}")
            atomic_write_json(target, task)
            written_tasks.append(target)
            new_entries.append(
                {
                    "registry_task_id": registry_task_id,
                    "title": request["title"],
                    "task_path": relative,
                    "arc_sha256": _hash(target),
                    "source_pipeline": request.get(
                        "source_pipeline", manifest.get("source_pipeline", "unknown")
                    ),
                    "source_run": source_run_text,
                    "source_candidate_id": candidate_id,
                    "search_mode": record["search_mode"],
                    "intrinsic_status": intrinsic_status,
                    "portfolio_status": "include_now",
                    "source_portfolio_status": _source_portfolio_status(record),
                    "user_authorized_portfolio_override": portfolio_override,
                    "user_authorized_same_procedure_override": request.get(
                        "authorize_same_procedure_override"
                    )
                    is True,
                    "commit_set_id": commit_set_id,
                    "concept_summary": _concept_summary(record),
                    "intended_rule": _intended_rule(record),
                    "dossier_path": f"dossiers/{registry_task_id}/manifest.json",
                    "dossier_sha256": "pending-explicit-authorized-materialization",
                }
            )
    except BaseException:
        for path in written_tasks:
            path.unlink(missing_ok=True)
        raise
    old_registry = deepcopy(raw)
    try:
        raw["commit_sets"].append(
            {
                "commit_set_id": commit_set_id,
                "committed_on": manifest["committed_on"],
                "authorization": manifest["authorization_record"],
                "entry_count": len(new_entries),
                "notes": manifest["notes"],
            }
        )
        raw["entries"].extend(new_entries)
        atomic_write_json(registry_path, raw)
        materialize_committed_dossiers(
            registry_path,
            authorize_user_decision=True,
            registry_task_ids={item["registry_task_id"] for item in new_entries},
        )
        verified = load_committed_task_registry(registry_path)
    except BaseException:
        atomic_write_json(registry_path, old_registry)
        for path in written_tasks:
            path.unlink(missing_ok=True)
        for entry in new_entries:
            dossier = committed_root / "dossiers" / entry["registry_task_id"]
            if dossier.exists():
                shutil.rmtree(dossier)
        raise
    _write_checksums(committed_root, verified)
    _write_catalog(committed_root, verified)
    _render_bank(committed_root, verified)
    return {
        "idempotent": False,
        "commit_set_id": commit_set_id,
        "added": [item["registry_task_id"] for item in new_entries],
        "previous_count": len(before["entries"]),
        "registry_count": len(verified["entries"]),
        "dossiers_include_full_authoring_and_relevant_raw_calls": True,
        "authorization_record": manifest["authorization_record"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--authorize-user-decision", action="store_true")
    args = parser.parse_args(argv)
    result = commit_authorized_manifest(
        args.manifest,
        authorize_user_decision=args.authorize_user_decision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuthorizedCommitError",
    "_portfolio_commit_override_allowed",
    "commit_authorized_manifest",
]
