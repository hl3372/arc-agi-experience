"""Independent moving concept archive and in-flight reservations for Lean V2."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence
import fcntl
import time

from .concept_archive import load_archive
from .data import TaskRecord
from .util import atomic_write_json, content_hash, read_json


V2_ARCHIVE_SCHEMA_VERSION = 2
V2_ARCHIVE_ID = "lean-authoring-v2-moving-concepts"


class LeanV2ArchiveError(ValueError):
    pass


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _concept_hash(text: str) -> str:
    return content_hash(_normalized(text).casefold())


def _empty_archive() -> dict[str, Any]:
    return {
        "schema_version": V2_ARCHIVE_SCHEMA_VERSION,
        "archive_id": V2_ARCHIVE_ID,
        "frozen": False,
        "entries": [],
        "active_reservations": [],
        "reservation_history": [],
    }


def _validate_entry(entry: Any, path: str) -> None:
    if not isinstance(entry, dict):
        raise LeanV2ArchiveError(f"{path} must be an object")
    required = {
        "reference_id",
        "source_kind",
        "source_run",
        "batch_id",
        "candidate_id",
        "central_relation",
        "intended_rule",
        "minimal_inference_procedure",
        "evidence_structure",
        "representation_summary",
        "test_generalization",
        "coverage_properties",
        "background_color",
        "status",
        "portfolio_note",
        "concept_hash",
    }
    if set(entry) != required:
        raise LeanV2ArchiveError(f"{path} has the wrong fields")
    relation = entry["central_relation"]
    if (
        not isinstance(relation, str)
        or not relation.strip()
        or entry["concept_hash"] != _concept_hash(relation)
    ):
        raise LeanV2ArchiveError(f"{path} has an invalid relation/hash")
    if entry["background_color"] is not None and (
        type(entry["background_color"]) is not int
        or not 0 <= entry["background_color"] <= 9
    ):
        raise LeanV2ArchiveError(f"{path}.background_color is invalid")
    properties = entry["coverage_properties"]
    if (
        not isinstance(properties, list)
        or any(
            value not in {"episode_level_inference", "supported_test_leap"}
            for value in properties
        )
        or len(properties) != len(set(properties))
    ):
        raise LeanV2ArchiveError(f"{path}.coverage_properties is invalid")


def _validate_reservation(item: Any, path: str, *, active: bool) -> None:
    if not isinstance(item, dict):
        raise LeanV2ArchiveError(f"{path} must be an object")
    common = {
        "reservation_id",
        "run_id",
        "batch_id",
        "candidate_id",
        "central_relation",
        "mechanism_summary",
        "concept_hash",
        "created_at_epoch_seconds",
    }
    expected = common if active else common | {
        "resolved_at_epoch_seconds",
        "outcome",
        "resolution_note",
    }
    if set(item) != expected:
        raise LeanV2ArchiveError(f"{path} has the wrong fields")
    if item["concept_hash"] != _concept_hash(item["central_relation"]):
        raise LeanV2ArchiveError(f"{path} has an invalid concept hash")


def load_v2_archive(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = read_json(source)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != V2_ARCHIVE_SCHEMA_VERSION
        or value.get("archive_id") != V2_ARCHIVE_ID
        or value.get("frozen") is not False
        or not isinstance(value.get("entries"), list)
        or not isinstance(value.get("active_reservations"), list)
        or not isinstance(value.get("reservation_history"), list)
    ):
        raise LeanV2ArchiveError(f"malformed Lean V2 concept archive: {source}")
    entry_ids: set[str] = set()
    for index, entry in enumerate(value["entries"]):
        _validate_entry(entry, f"{source}: entries[{index}]")
        reference_id = entry["reference_id"]
        if reference_id in entry_ids:
            raise LeanV2ArchiveError(f"{source}: duplicate reference_id {reference_id}")
        entry_ids.add(reference_id)
    reservation_ids: set[str] = set()
    for index, item in enumerate(value["active_reservations"]):
        _validate_reservation(
            item, f"{source}: active_reservations[{index}]", active=True
        )
        if item["reservation_id"] in reservation_ids:
            raise LeanV2ArchiveError(f"{source}: duplicate active reservation")
        reservation_ids.add(item["reservation_id"])
    for index, item in enumerate(value["reservation_history"]):
        _validate_reservation(
            item, f"{source}: reservation_history[{index}]", active=False
        )
    return value


def initialize_v2_archive(path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        return load_v2_archive(destination)
    value = _empty_archive()
    atomic_write_json(destination, value)
    return value


def _locked_update(
    path: str | Path,
    mutator: Any,
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        archive = initialize_v2_archive(destination)
        mutator(archive)
        archive["updated_at_epoch_seconds"] = round(time.time(), 3)
        atomic_write_json(destination, archive)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return archive


def reserve_concepts(
    path: str | Path,
    *,
    run_id: str,
    batch_id: str,
    selections: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Add idempotent reservations and reject exact in-flight concept reuse."""

    created_at = round(time.time(), 3)

    def mutate(archive: dict[str, Any]) -> None:
        active_by_id = {
            item["reservation_id"]: item
            for item in archive["active_reservations"]
        }
        occupied_hashes = {
            item["concept_hash"]: item["reservation_id"]
            for item in archive["active_reservations"]
        }
        occupied_hashes.update(
            {
                item["concept_hash"]: item["reference_id"]
                for item in archive["entries"]
            }
        )
        for selection in selections:
            candidate_id = str(selection["candidate_id"])
            relation = _normalized(str(selection["central_relation"]))
            reservation_id = f"reservation:{run_id}:{batch_id}:{candidate_id}"
            item = {
                "reservation_id": reservation_id,
                "run_id": run_id,
                "batch_id": batch_id,
                "candidate_id": candidate_id,
                "central_relation": relation,
                "mechanism_summary": _normalized(
                    str(selection.get("mechanism_summary") or relation)
                )[:700],
                "concept_hash": _concept_hash(relation),
                "created_at_epoch_seconds": created_at,
            }
            if reservation_id in active_by_id:
                if active_by_id[reservation_id] != item:
                    # The timestamp is intentionally stable on idempotent
                    # resume; compare everything except its original value.
                    existing = active_by_id[reservation_id]
                    if {
                        key: value
                        for key, value in existing.items()
                        if key != "created_at_epoch_seconds"
                    } != {
                        key: value
                        for key, value in item.items()
                        if key != "created_at_epoch_seconds"
                    }:
                        raise LeanV2ArchiveError(
                            f"reservation changed during resume: {reservation_id}"
                        )
                continue
            conflict = occupied_hashes.get(item["concept_hash"])
            if conflict:
                raise LeanV2ArchiveError(
                    f"concept already finalized or reserved by {conflict}"
                )
            archive["active_reservations"].append(item)
            active_by_id[reservation_id] = item
            occupied_hashes[item["concept_hash"]] = reservation_id
        archive["active_reservations"].sort(
            key=lambda item: item["reservation_id"]
        )

    return _locked_update(path, mutate)


def make_archive_entry(
    record: dict[str, Any],
    *,
    run_id: str,
    run_dir: str,
) -> dict[str, Any]:
    architecture = record.get("architecture", {})
    selector = record.get("selector_assessment", {})
    complete_task = record.get("complete_task")
    coverage: list[str] = []
    technical = record.get("technical_results", [])
    if technical:
        assessment = technical[-1].get("response", {}).get(
            "coverage_property_assessment", {}
        )
        if assessment.get("status") == "genuine":
            if assessment.get("property") == "both":
                coverage = ["episode_level_inference", "supported_test_leap"]
            elif assessment.get("property") in {
                "episode_level_inference",
                "supported_test_leap",
            }:
                coverage = [assessment["property"]]
    relation = _normalized(str(record["central_relation"]))
    metadata = record.get("realization", {}).get("authoring_metadata", {})
    return {
        "reference_id": f"lean-v2:{run_id}:{record['candidate_id']}",
        "source_kind": str(record.get("archive_source_kind") or record["status"]),
        "source_run": run_dir,
        "batch_id": str(record["batch_id"]),
        "candidate_id": str(record["candidate_id"]),
        "central_relation": relation,
        "intended_rule": _normalized(str(architecture.get("intended_rule", ""))),
        "minimal_inference_procedure": _normalized(
            str(
                architecture.get("simplest_plausible_shortcut", "")
                or selector.get("reason", "")
            )
        ),
        "evidence_structure": _normalized(
            " | ".join(
                str(item.get("what_it_teaches", ""))
                for item in architecture.get("training_pair_plan", [])
            )
        ),
        "representation_summary": _normalized(
            str(architecture.get("realization_plan", ""))
        ),
        "test_generalization": _normalized(
            str(architecture.get("test_plan", {}).get("what_is_new", ""))
        ),
        "coverage_properties": coverage,
        "background_color": (
            record.get("palette", {}).get("final_background_color")
            if complete_task is not None
            else None
        ),
        "status": str(record["status"]),
        "portfolio_note": _normalized(
            str(
                record.get("final_reason", "")
                or selector.get("reason", "")
            )
        ),
        "concept_hash": _concept_hash(relation),
    }


def resolve_reservation(
    path: str | Path,
    *,
    reservation_id: str,
    outcome: str,
    resolution_note: str,
    archive_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one reservation and optionally retain the concept permanently."""

    resolved_at = round(time.time(), 3)

    def mutate(archive: dict[str, Any]) -> None:
        active = {
            item["reservation_id"]: item
            for item in archive["active_reservations"]
        }
        if reservation_id not in active:
            prior = next(
                (
                    item
                    for item in archive["reservation_history"]
                    if item["reservation_id"] == reservation_id
                ),
                None,
            )
            if prior is None:
                raise LeanV2ArchiveError(
                    f"unknown reservation: {reservation_id}"
                )
            if prior["outcome"] != outcome:
                raise LeanV2ArchiveError(
                    f"reservation outcome changed: {reservation_id}"
                )
        else:
            item = active[reservation_id]
            archive["active_reservations"] = [
                value
                for value in archive["active_reservations"]
                if value["reservation_id"] != reservation_id
            ]
            archive["reservation_history"].append(
                {
                    **item,
                    "resolved_at_epoch_seconds": resolved_at,
                    "outcome": outcome,
                    "resolution_note": _normalized(resolution_note),
                }
            )
            archive["reservation_history"].sort(
                key=lambda value: (
                    value["resolved_at_epoch_seconds"],
                    value["reservation_id"],
                )
            )

        if archive_entry is not None:
            _validate_entry(archive_entry, "archive_entry")
            by_id = {
                item["reference_id"]: item for item in archive["entries"]
            }
            existing = by_id.get(archive_entry["reference_id"])
            if existing is not None and existing != archive_entry:
                raise LeanV2ArchiveError(
                    f"archive entry changed: {archive_entry['reference_id']}"
                )
            if existing is None and not any(
                item["concept_hash"] == archive_entry["concept_hash"]
                for item in archive["entries"]
            ):
                archive["entries"].append(deepcopy(archive_entry))
                archive["entries"].sort(key=lambda item: item["reference_id"])

    return _locked_update(path, mutate)


def append_unreserved_entry(
    path: str | Path, entry: dict[str, Any]
) -> dict[str, Any]:
    """Retain a strong selector-stage duplicate that had no reservation."""

    _validate_entry(entry, "entry")

    def mutate(archive: dict[str, Any]) -> None:
        existing = {
            item["reference_id"]: item for item in archive["entries"]
        }
        if entry["reference_id"] in existing:
            if existing[entry["reference_id"]] != entry:
                raise LeanV2ArchiveError(
                    f"archive entry changed: {entry['reference_id']}"
                )
            return
        if any(
            item["concept_hash"] == entry["concept_hash"]
            for item in archive["entries"]
        ):
            return
        archive["entries"].append(deepcopy(entry))
        archive["entries"].sort(key=lambda item: item["reference_id"])

    return _locked_update(path, mutate)


def reservation_digest(archive: dict[str, Any]) -> str:
    active = archive.get("active_reservations", [])
    if not active:
        return "No task mechanisms are currently reserved in flight."
    return (
        "In-flight task mechanisms to avoid developing close siblings of:\n"
        + "\n".join(
            f"- {item['mechanism_summary'][:500]}" for item in active
        )
    )


def reference_records_v2(
    bank: Iterable[TaskRecord],
    legacy_archive: dict[str, Any],
    v1_archive: dict[str, Any],
    v2_archive: dict[str, Any],
    *,
    excluded_real_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_real_ids or set()
    result = [
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
    for archive in (legacy_archive, v1_archive):
        for entry in archive["entries"]:
            result.append(
                {
                    "reference_id": entry["reference_id"],
                    "source": "previous_generated",
                    "central_relation": entry["central_relation"],
                    "summary": entry["intended_rule"] or entry["central_relation"],
                    "status": entry["status"],
                }
            )
    for entry in v2_archive["entries"]:
        result.append(
            {
                "reference_id": entry["reference_id"],
                "source": "previous_generated",
                "central_relation": entry["central_relation"],
                "summary": entry["intended_rule"] or entry["central_relation"],
                "status": entry["status"],
            }
        )
    for item in v2_archive["active_reservations"]:
        result.append(
            {
                "reference_id": item["reservation_id"],
                "source": "in_flight_generated",
                "central_relation": item["central_relation"],
                "summary": item["mechanism_summary"],
                "status": "reserved",
            }
        )
    return result


def portfolio_digest_v2(
    legacy_archive: dict[str, Any],
    v1_archive: dict[str, Any],
    v2_archive: dict[str, Any],
    *,
    maximum_entries: int = 24,
) -> str:
    combined = [
        *legacy_archive["entries"],
        *v1_archive["entries"],
        *v2_archive["entries"],
    ]
    lines = [
        f"- {item['central_relation'][:360]}"
        + (
            f" Portfolio note: {item.get('portfolio_note', '')[:180]}"
            if item.get("portfolio_note")
            else ""
        )
        for item in combined[-maximum_entries:]
    ]
    memory = (
        "No finalized generated concepts are recorded."
        if not lines
        else (
            "Recent/high-value generated mechanisms to avoid repeating. This is "
            "memory, not a menu or taxonomy:\n" + "\n".join(lines)
        )
    )
    coverage = Counter(
        property_name
        for item in v2_archive["entries"]
        if item["status"] == "core"
        for property_name in item["coverage_properties"]
    )
    background = Counter(
        str(item["background_color"])
        for item in v2_archive["entries"]
        if item["status"] in {"core", "reserve", "ambiguous_reserve"}
        and item["background_color"] is not None
    )
    return (
        memory
        + "\n\n"
        + reservation_digest(v2_archive)
        + "\n\nLean V2 verified coverage counts: "
        + str(dict(coverage))
        + ". Retained realized background counts: "
        + str(dict(background))
        + ". These distributions are portfolio context, not acceptance quotas."
    )


def load_source_archives(
    legacy_path: str | Path,
    v1_path: str | Path,
    v2_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_archive(legacy_path, expected_frozen=True),
        load_archive(v1_path, expected_frozen=False),
        initialize_v2_archive(v2_path),
    )


__all__ = [
    "LeanV2ArchiveError",
    "V2_ARCHIVE_ID",
    "V2_ARCHIVE_SCHEMA_VERSION",
    "append_unreserved_entry",
    "initialize_v2_archive",
    "load_source_archives",
    "load_v2_archive",
    "make_archive_entry",
    "portfolio_digest_v2",
    "reference_records_v2",
    "reservation_digest",
    "reserve_concepts",
    "resolve_reservation",
]
