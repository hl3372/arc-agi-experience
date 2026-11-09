"""Separate legacy and moving lean concept archives."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence
import fcntl
import json
import time

from .data import TaskRecord
from .util import atomic_write_json, content_hash, read_json


ARCHIVE_SCHEMA_VERSION = 1


class ConceptArchiveError(ValueError):
    pass


def _relation(record: dict[str, Any]) -> str | None:
    for value in (
        record.get("central_relation"),
        record.get("task", {}).get("authored_response", {})
        .get("authoring_metadata", {})
        .get("intended_rule"),
        record.get("intended_rule"),
    ):
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return None


def _archive_entry(
    *,
    reference_id: str,
    source_kind: str,
    source_run: str,
    candidate_id: str,
    central_relation: str,
    intended_rule: str | None,
    status: str,
    portfolio_note: str = "",
) -> dict[str, Any]:
    return {
        "reference_id": reference_id,
        "source_kind": source_kind,
        "source_run": source_run,
        "candidate_id": candidate_id,
        "central_relation": central_relation,
        "intended_rule": intended_rule or "",
        "status": status,
        "portfolio_note": portfolio_note,
        "concept_hash": content_hash(central_relation.casefold()),
    }


def build_legacy_archive(
    runs_dir: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Freeze strong historical generated concepts without carrying tasks forward."""

    root = Path(runs_dir)
    entries: dict[str, dict[str, Any]] = {}

    # The campaign portfolio is the authoritative original moving list.  It
    # includes accepted representatives and high-quality duplicates that were
    # deliberately retained for cross-batch avoidance.
    campaign_records = root / "arcgen-scale70-20260725" / "portfolio" / "records"
    for path in sorted(campaign_records.glob("*.json")):
        value = read_json(path)
        if not isinstance(value, dict):
            continue
        relation = _relation(value)
        if not relation:
            continue
        portfolio = next(
            (
                item
                for item in read_json(root / "arcgen-scale70-20260725" / "state.json").get(
                    "portfolio", []
                )
                if item.get("portfolio_id") == path.stem
            ),
            {},
        )
        status = str(portfolio.get("portfolio_status", value.get("status", "historical")))
        source_kind = "strong_duplicate" if status == "duplicate" else "core_or_reserve"
        intended = (
            value.get("task", {})
            .get("authored_response", {})
            .get("authoring_metadata", {})
            .get("intended_rule")
        )
        entries[f"legacy:campaign:{path.stem}"] = _archive_entry(
            reference_id=f"legacy:campaign:{path.stem}",
            source_kind=source_kind,
            source_run=str(portfolio.get("source_run", "")),
            candidate_id=str(value.get("candidate_id", path.stem)),
            central_relation=relation,
            intended_rule=intended if isinstance(intended, str) else None,
            status=status,
            portfolio_note=str(portfolio.get("deduplication_reason", "")),
        )

    # Include qualifying records from completed historical runs that may not
    # have reached the campaign portfolio.
    for state_path in sorted(root.glob("mvp-*/state.json")):
        try:
            state = read_json(state_path)
        except ValueError:
            continue
        if not isinstance(state, dict):
            continue
        for candidate in state.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            status = str(candidate.get("status", ""))
            rejections = candidate.get("rejections", [])
            duplicate_only = bool(rejections) and all(
                str(item.get("category", "")).casefold()
                in {"duplicate", "duplication", "novelty"}
                for item in rejections
                if isinstance(item, dict)
            )
            novelty = (
                candidate.get("stage_outcomes", {})
                .get("conceptual_novelty", {})
                .get("novelty_score", -1)
            )
            strong = status in {"accepted", "valid_reserve"} or (
                duplicate_only and isinstance(novelty, int) and novelty >= 3
            )
            if not strong:
                continue
            relation = _relation(candidate)
            if not relation:
                continue
            run_id = state_path.parent.name
            candidate_id = str(candidate.get("candidate_id", "unknown"))
            reference_id = f"legacy:{run_id}:{candidate_id}"
            if any(
                entry["concept_hash"] == content_hash(relation.casefold())
                for entry in entries.values()
            ):
                continue
            intended = (
                candidate.get("task", {})
                .get("authored_response", {})
                .get("authoring_metadata", {})
                .get("intended_rule")
            )
            entries[reference_id] = _archive_entry(
                reference_id=reference_id,
                source_kind=(
                    "strong_duplicate" if duplicate_only else "core_or_reserve"
                ),
                source_run=str(state_path.parent.resolve()),
                candidate_id=candidate_id,
                central_relation=relation,
                intended_rule=intended if isinstance(intended, str) else None,
                status=status,
            )

    archive = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "archive_id": "legacy-generated-concepts-v1-frozen",
        "frozen": True,
        "created_from": str(root.resolve()),
        "entries": sorted(entries.values(), key=lambda item: item["reference_id"]),
    }
    destination_path = Path(destination)
    if destination_path.exists():
        existing = load_archive(destination_path, expected_frozen=True)
        if existing != archive:
            raise ConceptArchiveError(
                f"frozen legacy concept archive differs from reproducible source: {destination_path}"
            )
    else:
        atomic_write_json(destination_path, archive)
    return archive


def load_archive(path: str | Path, *, expected_frozen: bool | None = None) -> dict[str, Any]:
    source = Path(path)
    value = read_json(source)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != ARCHIVE_SCHEMA_VERSION
        or not isinstance(value.get("archive_id"), str)
        or type(value.get("frozen")) is not bool
        or not isinstance(value.get("entries"), list)
    ):
        raise ConceptArchiveError(f"malformed concept archive: {source}")
    if expected_frozen is not None and value["frozen"] != expected_frozen:
        raise ConceptArchiveError(
            f"concept archive frozen flag is wrong for this role: {source}"
        )
    ids: set[str] = set()
    for index, entry in enumerate(value["entries"]):
        if not isinstance(entry, dict):
            raise ConceptArchiveError(f"{source}: entries[{index}] must be an object")
        required = {
            "reference_id",
            "source_kind",
            "source_run",
            "candidate_id",
            "central_relation",
            "intended_rule",
            "status",
            "portfolio_note",
            "concept_hash",
        }
        if set(entry) != required:
            raise ConceptArchiveError(
                f"{source}: entries[{index}] has wrong fields"
            )
        reference_id = entry["reference_id"]
        if not isinstance(reference_id, str) or reference_id in ids:
            raise ConceptArchiveError(
                f"{source}: entries[{index}] has invalid/duplicate reference_id"
            )
        ids.add(reference_id)
        relation = entry["central_relation"]
        if (
            not isinstance(relation, str)
            or not relation.strip()
            or entry["concept_hash"] != content_hash(relation.casefold())
        ):
            raise ConceptArchiveError(
                f"{source}: entries[{index}] has invalid relation/hash"
            )
    return value


def initialize_lean_archive(path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        return load_archive(destination, expected_frozen=False)
    value = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "archive_id": "lean-authoring-v1-moving-concepts",
        "frozen": False,
        "entries": [],
    }
    atomic_write_json(destination, value)
    return value


def append_lean_concepts(
    path: str | Path,
    entries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically append unique run concepts under an inter-process lock."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        archive = initialize_lean_archive(destination)
        by_id = {item["reference_id"]: item for item in archive["entries"]}
        by_hash = {item["concept_hash"] for item in archive["entries"]}
        for raw in entries:
            relation = " ".join(str(raw["central_relation"]).split())
            concept_hash = content_hash(relation.casefold())
            reference_id = str(raw["reference_id"])
            entry = _archive_entry(
                reference_id=reference_id,
                source_kind=str(raw["source_kind"]),
                source_run=str(raw["source_run"]),
                candidate_id=str(raw["candidate_id"]),
                central_relation=relation,
                intended_rule=(
                    str(raw.get("intended_rule", ""))
                    if raw.get("intended_rule")
                    else None
                ),
                status=str(raw["status"]),
                portfolio_note=str(raw.get("portfolio_note", "")),
            )
            if reference_id in by_id:
                if by_id[reference_id] != entry:
                    raise ConceptArchiveError(
                        f"moving archive entry changed: {reference_id}"
                    )
                continue
            if concept_hash in by_hash:
                continue
            by_id[reference_id] = entry
            by_hash.add(concept_hash)
        archive["entries"] = sorted(by_id.values(), key=lambda item: item["reference_id"])
        archive["updated_at_epoch_seconds"] = round(time.time(), 3)
        atomic_write_json(destination, archive)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return archive


def reference_records(
    bank: Iterable[TaskRecord],
    legacy_archive: dict[str, Any],
    lean_archive: dict[str, Any],
    *,
    excluded_real_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_real_ids or set()
    references = [
        {
            "reference_id": f"real:{record.task_id}",
            "source": "real_task",
            "central_relation": record.rule_description,
            "summary": record.rule_description,
            "status": "real",
        }
        for record in bank
        if record.task_id not in excluded
    ]
    for archive in (legacy_archive, lean_archive):
        for entry in archive["entries"]:
            references.append(
                {
                    "reference_id": entry["reference_id"],
                    "source": "previous_generated",
                    "central_relation": entry["central_relation"],
                    "summary": entry["intended_rule"] or entry["central_relation"],
                    "status": entry["status"],
                }
            )
    return references


def portfolio_digest(
    legacy_archive: dict[str, Any],
    lean_archive: dict[str, Any],
    *,
    maximum_entries: int = 24,
) -> str:
    """Build a prose-only digest without imposing a taxonomy."""

    combined = [*legacy_archive["entries"], *lean_archive["entries"]]
    if not combined:
        return "No previously generated task concepts are recorded."
    selected = combined[-maximum_entries:]
    lines = [
        f"- {item['central_relation'][:360]}"
        + (f" Portfolio note: {item['portfolio_note'][:180]}" if item["portfolio_note"] else "")
        for item in selected
    ]
    return (
        "Recent/high-value generated mechanisms to avoid repeating. This is a "
        "memory list, not a menu or taxonomy:\n" + "\n".join(lines)
    )


__all__ = [
    "ConceptArchiveError",
    "append_lean_concepts",
    "build_legacy_archive",
    "initialize_lean_archive",
    "load_archive",
    "portfolio_digest",
    "reference_records",
]
