"""Attempt-complete accounting for the repaired E2 qualification.

The original E2 replay consumed only the aggregate timing attached to a
successful logical call.  That aggregate describes the final successful
attempt, so it understates retried calls and cannot describe a terminally
failed call.  ``ModelInvoker`` already persists one timing file per physical
attempt; this module treats those immutable files as authoritative.

This module is additive.  It deliberately does not change the historical E2
runtime or any frozen run artifact.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any

from .util import read_json

PIPELINE = "arcgen-e2-attempt-ledger"
VERSION = "2.0"

POST_SELECTION_ROLES = frozenset(
    {
        "concrete_planner",
        "task_realizer",
        "technical_verifier",
        "blind_solver",
        "ambiguity_adjudicator",
        "targeted_repair",
    }
)

ATTEMPT_TIMING_RE = re.compile(r"^attempt-(\d+)\.timing\.json$")
EMBEDDED_CANDIDATE_RE = re.compile(r"k[0-9a-f]{12}")


class E2AttemptLedgerError(RuntimeError):
    """Raised when physical-attempt coverage is incomplete or contradictory."""


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _number(value: Any, *, field: str, call_id: str) -> float:
    if not isinstance(value, (int, float)):
        raise E2AttemptLedgerError(f"{call_id} attempt lacks numeric {field}")
    return float(value)


def candidate_for_call(call_id: str, candidate_ids: Sequence[str]) -> str | None:
    """Resolve the candidate without depending on provider/private arm labels."""

    prefix_matches = [
        candidate for candidate in candidate_ids if call_id.startswith(f"{candidate}-")
    ]
    if len(prefix_matches) > 1:
        raise E2AttemptLedgerError(f"ambiguous candidate prefix for {call_id}")
    if prefix_matches:
        return prefix_matches[0]
    embedded = sorted(set(EMBEDDED_CANDIDATE_RE.findall(call_id)))
    if len(embedded) > 1:
        raise E2AttemptLedgerError(f"multiple embedded candidates in {call_id}")
    return embedded[0] if embedded else None


def phase_for_call(role: str, call_id: str) -> str:
    if role == "concrete_planner":
        return "planning"
    if role == "task_realizer":
        return "realization"
    if role == "technical_verifier" and "ambiguity-repair" not in call_id:
        return "technical"
    if role == "targeted_repair" and "mechanical" in call_id:
        return "technical"
    return "blind_and_conditional"


def _candidate_log_directories(row: Mapping[str, Any], *, run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if isinstance(row.get("log_dir"), str):
        paths.append(Path(str(row["log_dir"])))
    stage = row.get("stage")
    call_id = row.get("call_id")
    if isinstance(stage, str) and isinstance(call_id, str):
        paths.append(run_dir / "model_calls" / stage / call_id)
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_dir():
            result.append(path)
    return result


def _attempt_paths(directory: Path) -> list[Path]:
    """Return base and archived-session attempt timings without duplicates."""

    paths = []
    for path in directory.rglob("attempt-*.timing.json"):
        if ATTEMPT_TIMING_RE.match(path.name):
            paths.append(path)
    return sorted(paths, key=lambda path: str(path.relative_to(directory)))


def _read_attempts(
    row: Mapping[str, Any], *, run_dir: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    call_id = str(row.get("call_id") or "")
    directories = _candidate_log_directories(row, run_dir=run_dir)
    if not directories:
        return [], [f"{call_id}: model-call directory is missing"]
    attempts: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_hashes: set[tuple[str, str]] = set()
    for directory in directories:
        by_parent: dict[Path, list[int]] = {}
        for path in _attempt_paths(directory):
            match = ATTEMPT_TIMING_RE.match(path.name)
            assert match is not None
            local_index = int(match.group(1))
            by_parent.setdefault(path.parent, []).append(local_index)
            identity = (str(path.resolve()), _sha(path))
            if identity in seen_hashes:
                continue
            seen_hashes.add(identity)
            try:
                timing = read_json(path)
            except Exception as exc:  # noqa: BLE001 - integrity report retains detail
                errors.append(f"{call_id}: unreadable {path}: {exc}")
                continue
            if not isinstance(timing, Mapping):
                errors.append(f"{call_id}: timing is not an object: {path}")
                continue
            try:
                queued = _number(
                    timing.get("queued_at_epoch_seconds"),
                    field="queued_at_epoch_seconds",
                    call_id=call_id,
                )
                completed = _number(
                    timing.get("completed_at_epoch_seconds"),
                    field="completed_at_epoch_seconds",
                    call_id=call_id,
                )
            except E2AttemptLedgerError as exc:
                errors.append(str(exc))
                continue
            if completed < queued:
                errors.append(f"{call_id}: attempt completed before it queued: {path}")
                continue
            acquired_raw = timing.get("slot_acquired_at_epoch_seconds")
            acquired = (
                float(acquired_raw) if isinstance(acquired_raw, (int, float)) else None
            )
            finished_raw = timing.get("model_finished_at_epoch_seconds")
            model_finished = (
                float(finished_raw) if isinstance(finished_raw, (int, float)) else None
            )
            attempt = {
                "local_attempt_index": local_index,
                "timing_path": str(path.resolve()),
                "timing_sha256": _sha(path),
                "session_directory": str(path.parent.resolve()),
                "queued_at_epoch_seconds": queued,
                "slot_acquired_at_epoch_seconds": acquired,
                "model_finished_at_epoch_seconds": model_finished,
                "parse_started_at_epoch_seconds": timing.get(
                    "parse_started_at_epoch_seconds"
                ),
                "completed_at_epoch_seconds": completed,
                "queue_wait_seconds": float(timing.get("queue_wait_seconds") or 0),
                "model_execution_seconds": float(
                    timing.get("model_execution_seconds") or 0
                ),
                "parse_validation_seconds": float(
                    timing.get("parse_validation_seconds") or 0
                ),
                "total_seconds": float(timing.get("total_seconds") or 0),
                "outcome": str(timing.get("outcome") or "success"),
            }
            failure_path = path.with_name(
                path.name.replace(".timing.json", ".failure.log")
            )
            if failure_path.is_file():
                attempt["failure_log"] = {
                    "path": str(failure_path.resolve()),
                    "sha256": _sha(failure_path),
                }
            attempts.append(attempt)
        for parent, indices in by_parent.items():
            expected = list(range(1, max(indices) + 1)) if indices else []
            if sorted(set(indices)) != expected or len(indices) != len(set(indices)):
                errors.append(
                    f"{call_id}: non-contiguous attempt numbering under {parent}"
                )
    attempts.sort(
        key=lambda item: (
            float(item["queued_at_epoch_seconds"]),
            str(item["timing_path"]),
        )
    )
    for index, attempt in enumerate(attempts, start=1):
        attempt["physical_attempt_index"] = index
    for previous, current in pairwise(attempts):
        if current["queued_at_epoch_seconds"] < previous["queued_at_epoch_seconds"]:
            errors.append(f"{call_id}: physical attempt timestamps are not ordered")
    return attempts, errors


def _logical_call(
    key: str,
    raw: Mapping[str, Any],
    *,
    run_dir: Path,
    candidate_ids: Sequence[str],
    slate_by_candidate: Mapping[str, int],
) -> tuple[dict[str, Any], list[str]]:
    row = deepcopy(dict(raw))
    call_id = str(row.get("call_id") or "")
    role = str(row.get("role") or "")
    candidate = candidate_for_call(call_id, candidate_ids)
    errors: list[str] = []
    if candidate is None:
        errors.append(f"{call_id}: could not bind post-selection call to a candidate")
    attempts, attempt_errors = _read_attempts(row, run_dir=run_dir)
    errors.extend(attempt_errors)
    cache_source = row.get("cache_source")
    if not attempts:
        if cache_source:
            errors.append(
                f"{call_id}: cache-only logical call has no live attempt timing"
            )
        else:
            errors.append(f"{call_id}: no physical attempt timing was found")

    status = str(row.get("status") or "")
    if status not in {"complete", "failed"}:
        errors.append(f"{call_id}: logical call is not terminal ({status})")
    if attempts:
        if status == "complete" and attempts[-1]["outcome"] != "success":
            errors.append(f"{call_id}: completed logical call lacks a final success")
        if status == "failed" and any(
            attempt["outcome"] == "success" for attempt in attempts
        ):
            errors.append(
                f"{call_id}: failed logical call contains a successful attempt"
            )
        declared_attempts = row.get("attempts")
        if isinstance(declared_attempts, int) and declared_attempts != len(attempts):
            errors.append(
                f"{call_id}: ledger declares {declared_attempts} attempts, found {len(attempts)}"
            )

    first_queued = attempts[0]["queued_at_epoch_seconds"] if attempts else None
    last_completed = attempts[-1]["completed_at_epoch_seconds"] if attempts else None
    inter_attempt_gap = 0.0
    for previous, current in pairwise(attempts):
        inter_attempt_gap += max(
            0.0,
            float(current["queued_at_epoch_seconds"])
            - float(previous["completed_at_epoch_seconds"]),
        )
    logical = {
        "ledger_key": key,
        "call_id": call_id,
        "candidate_id": candidate,
        "slate_index": slate_by_candidate.get(candidate) if candidate else None,
        "role": role,
        "stage": row.get("stage"),
        "phase": phase_for_call(role, call_id),
        "status": status,
        "model": row.get("model"),
        "reasoning_effort": row.get("reasoning_effort"),
        "cache_source": cache_source,
        "physical_attempt_count": len(attempts),
        "retry_attempt_count": max(0, len(attempts) - 1),
        "failed_attempt_count": sum(
            attempt["outcome"] != "success" for attempt in attempts
        ),
        "first_queued_at_epoch_seconds": first_queued,
        "last_completed_at_epoch_seconds": last_completed,
        "logical_envelope_seconds": (
            round(float(last_completed) - float(first_queued), 6)
            if first_queued is not None and last_completed is not None
            else None
        ),
        "active_model_seconds": round(
            sum(float(attempt["model_execution_seconds"]) for attempt in attempts),
            6,
        ),
        "model_slot_queue_wait_seconds": round(
            sum(float(attempt["queue_wait_seconds"]) for attempt in attempts), 6
        ),
        "parse_validation_seconds": round(
            sum(float(attempt["parse_validation_seconds"]) for attempt in attempts),
            6,
        ),
        "inter_attempt_gap_seconds": round(inter_attempt_gap, 6),
        "attempts": attempts,
        "coverage_complete": not errors,
        "coverage_errors": errors,
    }
    return logical, errors


def build_attempt_ledger(state: Mapping[str, Any], *, run_dir: Path) -> dict[str, Any]:
    """Build a fail-closed ledger over every substantive post-selection call."""

    semantic = state.get("semantic_records") or state.get("records") or {}
    if not isinstance(semantic, Mapping):
        raise E2AttemptLedgerError("state.semantic_records must be a mapping")
    candidate_ids = sorted(str(candidate) for candidate in semantic)
    slate_by_candidate = {
        str(candidate): int(record.get("slate_index") or 0)
        for candidate, record in semantic.items()
        if isinstance(record, Mapping)
    }
    source = state.get("call_ledger") or {}
    if not isinstance(source, Mapping):
        raise E2AttemptLedgerError("state.call_ledger must be a mapping")
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    for key, raw in source.items():
        if not isinstance(raw, Mapping):
            continue
        if raw.get("role") not in POST_SELECTION_ROLES:
            continue
        if raw.get("substantive_experimental_call") is False:
            continue
        logical, call_errors = _logical_call(
            str(key),
            raw,
            run_dir=run_dir,
            candidate_ids=candidate_ids,
            slate_by_candidate=slate_by_candidate,
        )
        calls.append(logical)
        errors.extend(call_errors)
    calls.sort(
        key=lambda call: (
            float(call["first_queued_at_epoch_seconds"] or float("inf")),
            str(call["call_id"]),
        )
    )
    attempts = [attempt for call in calls for attempt in call["attempts"]]
    intervals = [
        (
            float(attempt["slot_acquired_at_epoch_seconds"]),
            float(
                attempt["model_finished_at_epoch_seconds"]
                or attempt["completed_at_epoch_seconds"]
            ),
        )
        for attempt in attempts
        if attempt["slot_acquired_at_epoch_seconds"] is not None
    ]
    events = sorted(
        [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals],
        key=lambda item: (item[0], item[1]),
    )
    active = 0
    peak = 0
    area = 0.0
    previous = events[0][0] if events else 0.0
    for moment, delta in events:
        area += active * (moment - previous)
        active += delta
        peak = max(peak, active)
        previous = moment
    span = (
        max(end for _, end in intervals) - min(start for start, _ in intervals)
        if intervals
        else 0.0
    )
    return {
        "schema_version": 2,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_directory": str(run_dir.resolve()),
        "logical_calls": calls,
        "summary": {
            "logical_call_count": len(calls),
            "completed_logical_call_count": sum(
                call["status"] == "complete" for call in calls
            ),
            "failed_logical_call_count": sum(
                call["status"] == "failed" for call in calls
            ),
            "physical_attempt_count": len(attempts),
            "retry_attempt_count": sum(
                int(call["retry_attempt_count"]) for call in calls
            ),
            "failed_attempt_count": sum(
                attempt["outcome"] != "success" for attempt in attempts
            ),
            "calls_by_role": dict(
                sorted(Counter(str(call["role"]) for call in calls).items())
            ),
            "attempts_by_role": dict(
                sorted(
                    Counter(
                        str(call["role"])
                        for call in calls
                        for _attempt in call["attempts"]
                    ).items()
                )
            ),
            "active_model_seconds": round(
                sum(float(call["active_model_seconds"]) for call in calls), 6
            ),
            "model_slot_queue_wait_seconds": round(
                sum(float(call["model_slot_queue_wait_seconds"]) for call in calls),
                6,
            ),
            "parse_validation_seconds": round(
                sum(float(call["parse_validation_seconds"]) for call in calls), 6
            ),
            "measured_peak_concurrent_attempts": peak,
            "measured_effective_attempt_concurrency": (
                round(area / span, 6) if span else 0.0
            ),
            "measured_attempt_span_seconds": round(span, 6),
            "coverage_complete": not errors,
            "coverage_error_count": len(errors),
        },
        "coverage_errors": errors,
    }


__all__ = [
    "POST_SELECTION_ROLES",
    "E2AttemptLedgerError",
    "build_attempt_ledger",
    "candidate_for_call",
    "phase_for_call",
]
