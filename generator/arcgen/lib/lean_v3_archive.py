"""Tiered concept memory for Lean V3 without mutating V1 or V2 archives."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import fcntl
import time

from .concept_archive import load_archive
from .data import TaskRecord
from .lean_v2_archive import load_v2_archive
from .util import atomic_write_json, content_hash, read_json


V3_ARCHIVE_SCHEMA_VERSION = 3
V3_ARCHIVE_ID = "lean-authoring-v3-moving-concepts"
TIER_NAMES = (
    "hard_duplicate_memory",
    "soft_mechanism_memory",
    "recoverable_work",
)
ENTRY_FIELDS = {
    "reference_id",
    "source_kind",
    "source_run",
    "candidate_id",
    "central_relation",
    "intended_rule",
    "status",
    "portfolio_note",
    "concept_hash",
    "origin",
    "classification_reason",
}


class LeanV3ArchiveError(ValueError):
    pass


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _hash(text: str) -> str:
    return content_hash(_normalize(text).casefold())


def _empty_archive() -> dict[str, Any]:
    return {
        "schema_version": V3_ARCHIVE_SCHEMA_VERSION,
        "archive_id": V3_ARCHIVE_ID,
        "frozen": False,
        "tiers": {tier: [] for tier in TIER_NAMES},
    }


def _normalized_entry(
    raw: Mapping[str, Any],
    *,
    origin: str,
    classification_reason: str,
) -> dict[str, Any]:
    relation = _normalize(str(raw["central_relation"]))
    return {
        "reference_id": str(raw["reference_id"]),
        "source_kind": str(raw.get("source_kind", "unknown")),
        "source_run": str(raw.get("source_run", "")),
        "candidate_id": str(raw.get("candidate_id", "")),
        "central_relation": relation,
        "intended_rule": str(raw.get("intended_rule") or ""),
        "status": str(raw.get("status", "")),
        "portfolio_note": str(raw.get("portfolio_note") or ""),
        "concept_hash": _hash(relation),
        "origin": origin,
        "classification_reason": classification_reason,
    }


def _validate_entry(entry: Any, path: str) -> None:
    if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS:
        raise LeanV3ArchiveError(f"{path} has the wrong fields")
    if (
        not isinstance(entry["central_relation"], str)
        or not entry["central_relation"].strip()
        or entry["concept_hash"] != _hash(entry["central_relation"])
    ):
        raise LeanV3ArchiveError(f"{path} has an invalid relation/hash")


def load_v3_archive(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = read_json(source)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != V3_ARCHIVE_SCHEMA_VERSION
        or value.get("archive_id") != V3_ARCHIVE_ID
        or value.get("frozen") is not False
        or not isinstance(value.get("tiers"), dict)
        or set(value["tiers"]) != set(TIER_NAMES)
    ):
        raise LeanV3ArchiveError(f"malformed Lean V3 concept archive: {source}")
    ids: set[str] = set()
    for tier in TIER_NAMES:
        if not isinstance(value["tiers"][tier], list):
            raise LeanV3ArchiveError(f"{source}: {tier} must be a list")
        for index, entry in enumerate(value["tiers"][tier]):
            _validate_entry(entry, f"{source}: {tier}[{index}]")
            if entry["reference_id"] in ids:
                raise LeanV3ArchiveError(
                    f"{source}: duplicate reference_id {entry['reference_id']}"
                )
            ids.add(entry["reference_id"])
    return value


def initialize_v3_archive(path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        return load_v3_archive(destination)
    value = _empty_archive()
    atomic_write_json(destination, value)
    return value


def _external_tier(
    origin: str, entry: Mapping[str, Any]
) -> tuple[str, str]:
    source_kind = str(entry.get("source_kind", ""))
    status = str(entry.get("status", ""))
    note = str(entry.get("portfolio_note", "")).casefold()

    if origin == "legacy_v1":
        return (
            "hard_duplicate_memory",
            "the frozen legacy list contains a completed task or confirmed duplicate",
        )
    if origin == "lean_v1":
        if status in {"core", "reserve"}:
            return (
                "hard_duplicate_memory",
                "completed successful V1 task; only a same-procedure match is hard",
            )
        if source_kind == "strong_duplicate":
            return (
                "hard_duplicate_memory",
                "V1 explicitly recorded this as a confirmed strong duplicate",
            )
        return (
            "soft_mechanism_memory",
            "V1 ambiguity or unrealized architecture is comparison context, not spent space",
        )

    if origin != "lean_v2":
        raise LeanV3ArchiveError(f"unknown external archive origin {origin}")

    infrastructure_markers = (
        "timed out",
        "timeout",
        "failed after configured",
        "response failed",
        "transport",
        "parse",
        "parser",
        "schema",
        "infrastructure",
        "interrupted",
        "not reached",
    )
    if source_kind == "strong_interrupted_pre_editorial" or any(
        marker in note for marker in infrastructure_markers
    ):
        return (
            "recoverable_work",
            "V2 lost this work operationally; it is retryable and not a novelty boundary",
        )
    if status in {"core", "reserve"}:
        return (
            "hard_duplicate_memory",
            "completed V2 task; only a confirmed same-procedure match is hard",
        )
    return (
        "soft_mechanism_memory",
        "V2 pilot similarity and crowding labels are retained only as soft warnings",
    )


def build_tiered_archive_view(
    *,
    legacy_path: str | Path,
    v1_path: str | Path,
    v2_path: str | Path,
    v3_path: str | Path,
) -> dict[str, Any]:
    """Combine immutable historical lists with the separate V3 moving list."""

    legacy = load_archive(legacy_path, expected_frozen=True)
    v1 = load_archive(v1_path, expected_frozen=False)
    v2 = load_v2_archive(v2_path)
    v3 = initialize_v3_archive(v3_path)
    result = {
        "schema_version": V3_ARCHIVE_SCHEMA_VERSION,
        "archive_id": "lean-authoring-v3-tiered-selection-view",
        "frozen": True,
        "source_hashes": {
            "legacy_v1": content_hash(legacy),
            "lean_v1": content_hash(v1),
            "lean_v2": content_hash(v2),
            "lean_v3": content_hash(v3),
        },
        "tiers": {tier: [] for tier in TIER_NAMES},
    }
    for origin, archive in (
        ("legacy_v1", legacy),
        ("lean_v1", v1),
        ("lean_v2", v2),
    ):
        for raw in archive["entries"]:
            tier, reason = _external_tier(origin, raw)
            result["tiers"][tier].append(
                _normalized_entry(
                    raw,
                    origin=origin,
                    classification_reason=reason,
                )
            )
    for tier in TIER_NAMES:
        result["tiers"][tier].extend(deepcopy(v3["tiers"][tier]))
        result["tiers"][tier].sort(key=lambda item: item["reference_id"])
    return result


def append_v3_concepts(
    path: str | Path,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Append V3-only entries atomically while retaining explicit tier semantics."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        archive = initialize_v3_archive(destination)
        by_id = {
            item["reference_id"]: (tier, item)
            for tier in TIER_NAMES
            for item in archive["tiers"][tier]
        }
        by_hash = {
            item["concept_hash"]
            for tier in TIER_NAMES
            for item in archive["tiers"][tier]
        }
        for raw in entries:
            tier = str(raw["tier"])
            if tier not in TIER_NAMES:
                raise LeanV3ArchiveError(f"unknown archive tier {tier}")
            entry = _normalized_entry(
                raw,
                origin="lean_v3",
                classification_reason=str(raw["classification_reason"]),
            )
            existing = by_id.get(entry["reference_id"])
            if existing:
                if existing != (tier, entry):
                    raise LeanV3ArchiveError(
                        f"moving archive entry changed: {entry['reference_id']}"
                    )
                continue
            if entry["concept_hash"] in by_hash:
                continue
            archive["tiers"][tier].append(entry)
            archive["tiers"][tier].sort(key=lambda item: item["reference_id"])
            by_id[entry["reference_id"]] = (tier, entry)
            by_hash.add(entry["concept_hash"])
        archive["updated_at_epoch_seconds"] = round(time.time(), 3)
        atomic_write_json(destination, archive)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return archive


def reference_records(
    bank: Iterable[TaskRecord],
    archive_view: Mapping[str, Any],
    *,
    excluded_real_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_real_ids or set()
    references = [
        {
            "reference_id": f"real:{record.task_id}",
            "source": "real_task",
            "archive_tier": "real_task",
            "central_relation": record.rule_description,
            "summary": record.rule_description,
            "status": "real",
        }
        for record in bank
        if record.task_id not in excluded
    ]
    generated: dict[str, dict[str, Any]] = {}
    for tier in ("hard_duplicate_memory", "soft_mechanism_memory"):
        for entry in archive_view["tiers"][tier]:
            item = {
                "reference_id": entry["reference_id"],
                "source": "previous_generated",
                "archive_tier": tier,
                "central_relation": entry["central_relation"],
                "summary": entry["intended_rule"] or entry["central_relation"],
                "status": entry["status"],
            }
            # Collapse only exact textual concepts. A hard memory wins over a
            # soft warning; no semantic threshold is applied in code.
            key = entry["concept_hash"]
            if key not in generated or tier == "hard_duplicate_memory":
                generated[key] = item
    references.extend(generated.values())
    return references


def tiered_portfolio_digest(
    archive_view: Mapping[str, Any],
    *,
    maximum_hard: int = 16,
    maximum_soft: int = 8,
) -> str:
    hard = archive_view["tiers"]["hard_duplicate_memory"][-maximum_hard:]
    soft = archive_view["tiers"]["soft_mechanism_memory"][-maximum_soft:]
    lines = [
        "Hard duplicate memory. Reject only a match to the same minimal "
        "solution procedure; these are not banned themes:"
    ]
    lines.extend(f"- {item['central_relation'][:360]}" for item in hard)
    lines.append("")
    lines.append(
        "Soft mechanism memory. Use as comparison warnings only; broad areas "
        "remain available:"
    )
    lines.extend(f"- {item['central_relation'][:360]}" for item in soft)
    if not hard and not soft:
        return "No generated-task concepts are recorded."
    return "\n".join(lines)


def archive_tier_report(archive_view: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy": {
            "hard_duplicate_memory": (
                "Hard only when a candidate shares the same minimal solution "
                "procedure or is an exact duplicate."
            ),
            "soft_mechanism_memory": (
                "Warnings for themes, metaphors, unrealized ideas, and mild "
                "crowding; never an automatic rejection."
            ),
            "recoverable_work": (
                "Operationally lost work; omitted from novelty retrieval and "
                "eligible for a future offline retry."
            ),
        },
        "counts": {
            tier: len(archive_view["tiers"][tier]) for tier in TIER_NAMES
        },
        "origin_counts": {
            tier: dict(
                sorted(
                    Counter(
                        item["origin"] for item in archive_view["tiers"][tier]
                    ).items()
                )
            )
            for tier in TIER_NAMES
        },
        "entries": deepcopy(archive_view["tiers"]),
        "source_hashes": deepcopy(archive_view.get("source_hashes", {})),
    }


__all__ = [
    "LeanV3ArchiveError",
    "TIER_NAMES",
    "append_v3_concepts",
    "archive_tier_report",
    "build_tiered_archive_view",
    "initialize_v3_archive",
    "load_v3_archive",
    "reference_records",
    "tiered_portfolio_digest",
]
