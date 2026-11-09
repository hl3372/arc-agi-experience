"""Persistent candidate-level dependency scheduler for the E2 experiment.

The scheduler knows nothing about ARC semantics or model prompts.  Callers
provide a pure ``next_stage`` view over their persisted semantic records and a
``run_stage`` function that performs exactly one candidate stage.  This keeps
the scheduling experiment isolated from the frozen authoring policy.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import atomic_write_json, content_hash, read_json

PIPELINE = "arcgen-e2-candidate-streaming-scheduler"
VERSION = "1.0"
STAGES = ("planning", "realization", "technical", "blind")


class E2SchedulerError(RuntimeError):
    """Raised when dependency or resume invariants are violated."""


@dataclass(frozen=True)
class StageResult:
    candidate_id: str
    stage: str
    error: str | None


class PersistentCandidateScheduler:
    """Run independent candidate chains under one global worker ceiling."""

    def __init__(
        self,
        *,
        state_path: Path,
        candidate_ids: Sequence[str],
        workers: int,
        next_stage: Callable[[str], str | None],
        run_stage: Callable[[str, str], None],
        max_downstream_before_new_plan: int = 2,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if workers < 1:
            raise E2SchedulerError("workers must be positive")
        if max_downstream_before_new_plan < 1:
            raise E2SchedulerError("starvation limit must be positive")
        ids = [str(item) for item in candidate_ids]
        if not ids or len(ids) != len(set(ids)):
            raise E2SchedulerError("candidate IDs must be non-empty and unique")
        self.state_path = state_path
        self.candidate_ids = tuple(sorted(ids))
        self.workers = int(workers)
        self.next_stage = next_stage
        self.run_stage = run_stage
        self.max_downstream_before_new_plan = int(max_downstream_before_new_plan)
        self.clock = clock
        self.lock = threading.RLock()
        self._event_serial = 0
        self.state = self._load_or_initialize()
        self._event_serial = max(
            (int(event.get("serial") or 0) for event in self.state.get("events", [])),
            default=0,
        )

    def _load_or_initialize(self) -> dict[str, Any]:
        fingerprint = content_hash(
            {
                "pipeline": PIPELINE,
                "version": VERSION,
                "candidate_ids": self.candidate_ids,
                "workers": self.workers,
                "max_downstream_before_new_plan": self.max_downstream_before_new_plan,
            }
        )
        if self.state_path.is_file():
            state = read_json(self.state_path)
            if state.get("fingerprint") != fingerprint:
                raise E2SchedulerError("scheduler state belongs to different inputs")
            if state.get("completed") is True:
                return state
            # A process can die after persisting dispatch but before it records
            # completion.  Requeue the exact stage; the model wrapper is
            # responsible for validating/reusing a completed call artifact.
            for candidate in state.get("candidates", {}).values():
                active = candidate.get("active_stage")
                if active is not None:
                    candidate["active_stage"] = None
                    candidate["recovery_count"] = int(
                        candidate.get("recovery_count") or 0
                    ) + 1
                    candidate.setdefault("recoveries", []).append(
                        {
                            "stage": active,
                            "at_epoch_seconds": round(self.clock(), 6),
                        }
                    )
            atomic_write_json(self.state_path, state)
            return state
        now = round(self.clock(), 6)
        state = {
            "schema_version": 1,
            "pipeline": PIPELINE,
            "version": VERSION,
            "fingerprint": fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "workers": self.workers,
            "max_downstream_before_new_plan": self.max_downstream_before_new_plan,
            "started_at_epoch_seconds": now,
            "completed": False,
            "events": [],
            "candidates": {
                candidate_id: {
                    "active_stage": None,
                    "ready_stage": None,
                    "ready_at_epoch_seconds": None,
                    "completed_stage_count": 0,
                    "stage_attempts": {},
                    "recovery_count": 0,
                    "recoveries": [],
                }
                for candidate_id in self.candidate_ids
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.state_path, state)
        return state

    def _save(self) -> None:
        with self.lock:
            atomic_write_json(self.state_path, self.state)

    def _event(self, kind: str, candidate_id: str, stage: str, **extra: Any) -> None:
        with self.lock:
            self._event_serial += 1
            self.state["events"].append(
                {
                    "serial": self._event_serial,
                    "kind": kind,
                    "candidate_id": candidate_id,
                    "stage": stage,
                    "at_epoch_seconds": round(self.clock(), 6),
                    **deepcopy(extra),
                }
            )
            self._save()

    def _refresh_ready(self, running_ids: set[str]) -> None:
        with self.lock:
            for candidate_id in self.candidate_ids:
                if candidate_id in running_ids:
                    continue
                row = self.state["candidates"][candidate_id]
                stage = self.next_stage(candidate_id)
                if stage is not None and stage not in STAGES:
                    raise E2SchedulerError(
                        f"unknown next stage {stage!r} for {candidate_id}"
                    )
                if row.get("ready_stage") != stage:
                    row["ready_stage"] = stage
                    row["ready_at_epoch_seconds"] = (
                        round(self.clock(), 6) if stage is not None else None
                    )
            self._save()

    def _pick_ready(self, downstream_since_plan: int) -> tuple[str, str] | None:
        ready = []
        for candidate_id in self.candidate_ids:
            row = self.state["candidates"][candidate_id]
            stage = row.get("ready_stage")
            if stage is None or row.get("active_stage") is not None:
                continue
            ready.append(
                (
                    candidate_id,
                    str(stage),
                    float(row.get("ready_at_epoch_seconds") or 0),
                )
            )
        if not ready:
            return None
        plans = [item for item in ready if item[1] == "planning"]
        if plans and downstream_since_plan >= self.max_downstream_before_new_plan:
            candidate, stage, _ = min(plans, key=lambda item: (item[2], item[0]))
            return candidate, stage
        # Deeper unblocked work has priority.  Within one stage, the oldest
        # ready candidate wins, with candidate ID as a stable tie breaker.
        candidate, stage, _ = min(
            ready,
            key=lambda item: (-STAGES.index(item[1]), item[2], item[0]),
        )
        return candidate, stage

    def _execute(self, candidate_id: str, stage: str) -> StageResult:
        try:
            self.run_stage(candidate_id, stage)
            return StageResult(candidate_id, stage, None)
        except Exception as exc:  # noqa: BLE001 - stage records retain the error
            return StageResult(candidate_id, stage, str(exc))

    def run(self) -> dict[str, Any]:
        if self.state.get("completed") is True:
            return deepcopy(self.state)
        running: dict[Future[StageResult], tuple[str, str]] = {}
        downstream_since_plan = 0
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while True:
                running_ids = {candidate for candidate, _stage in running.values()}
                self._refresh_ready(running_ids)
                while len(running) < self.workers:
                    picked = self._pick_ready(downstream_since_plan)
                    if picked is None:
                        break
                    candidate_id, stage = picked
                    with self.lock:
                        row = self.state["candidates"][candidate_id]
                        ready_at = float(row["ready_at_epoch_seconds"])
                        submitted = round(self.clock(), 6)
                        row["active_stage"] = stage
                        row["ready_stage"] = None
                        row["ready_at_epoch_seconds"] = None
                        row["stage_attempts"][stage] = int(
                            row["stage_attempts"].get(stage, 0)
                        ) + 1
                        self._save()
                    self._event(
                        "stage_dispatched",
                        candidate_id,
                        stage,
                        ready_at_epoch_seconds=ready_at,
                        scheduler_queue_wait_seconds=round(submitted - ready_at, 6),
                    )
                    future = executor.submit(self._execute, candidate_id, stage)
                    running[future] = (candidate_id, stage)
                    if stage == "planning":
                        downstream_since_plan = 0
                    else:
                        downstream_since_plan += 1
                if not running:
                    unresolved = [
                        candidate_id
                        for candidate_id in self.candidate_ids
                        if self.next_stage(candidate_id) is not None
                    ]
                    if unresolved:
                        raise E2SchedulerError(
                            "scheduler has unresolved candidates but no runnable work: "
                            + ", ".join(unresolved)
                        )
                    break
                done, _pending = wait(running, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: running[item]):
                    candidate_id, stage = running.pop(future)
                    result = future.result()
                    with self.lock:
                        row = self.state["candidates"][candidate_id]
                        if row.get("active_stage") != stage:
                            raise E2SchedulerError(
                                f"active-stage mismatch for {candidate_id}/{stage}"
                            )
                        row["active_stage"] = None
                        row["completed_stage_count"] = int(
                            row.get("completed_stage_count") or 0
                        ) + (0 if result.error else 1)
                        self._save()
                    self._event(
                        "stage_failed" if result.error else "stage_completed",
                        candidate_id,
                        stage,
                        error=result.error,
                    )
                    if result.error:
                        # An unexpected exception outside the semantic stage
                        # wrapper must stop the coordinator. Re-dispatching the
                        # unchanged semantic state here would create an
                        # unbounded substantive retry loop. The persisted event
                        # and active-stage recovery record make a later,
                        # explicitly supervised resume deterministic.
                        with self.lock:
                            self.state["fatal_stage_error"] = {
                                "candidate_id": candidate_id,
                                "stage": stage,
                                "error": result.error,
                                "at_epoch_seconds": round(self.clock(), 6),
                            }
                            self._save()
                        raise E2SchedulerError(
                            f"candidate stage failed unexpectedly: {candidate_id}/{stage}: "
                            f"{result.error}"
                        )
        with self.lock:
            self.state["completed"] = True
            self.state["completed_at_epoch_seconds"] = round(self.clock(), 6)
            self.state["terminal_next_stage"] = {
                candidate_id: self.next_stage(candidate_id)
                for candidate_id in self.candidate_ids
            }
            if any(self.state["terminal_next_stage"].values()):
                raise E2SchedulerError("scheduler completed with runnable candidates")
            self._save()
        return deepcopy(self.state)


def validate_dependency_events(state: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one dispatch/completion order per candidate and dependency order."""

    stage_index = {stage: index for index, stage in enumerate(STAGES)}
    dispatched: CounterLike = {}
    completed: CounterLike = {}
    last_completed: dict[str, int] = {}
    errors = []
    for event in sorted(state.get("events", []), key=lambda row: int(row["serial"])):
        candidate = str(event["candidate_id"])
        stage = str(event["stage"])
        if stage not in stage_index:
            errors.append(f"unknown stage {stage}")
            continue
        key = (candidate, stage)
        if event["kind"] == "stage_dispatched":
            dispatched[key] = dispatched.get(key, 0) + 1
            prior = last_completed.get(candidate, -1)
            if stage_index[stage] > prior + 1:
                errors.append(f"{candidate} skipped a dependency before {stage}")
        elif event["kind"] in {"stage_completed", "stage_failed"}:
            completed[key] = completed.get(key, 0) + 1
            if completed[key] > dispatched.get(key, 0):
                errors.append(f"{candidate}/{stage} completed without dispatch")
            if event["kind"] == "stage_completed":
                last_completed[candidate] = max(
                    last_completed.get(candidate, -1), stage_index[stage]
                )
    return {
        "valid": not errors,
        "errors": errors,
        "dispatch_count": sum(dispatched.values()),
        "completion_or_failure_count": sum(completed.values()),
    }


# A small explicit alias keeps static type checkers readable without importing
# Counter solely for a two-key integer mapping.
CounterLike = dict[tuple[str, str], int]


__all__ = [
    "STAGES",
    "E2SchedulerError",
    "PersistentCandidateScheduler",
    "StageResult",
    "validate_dependency_events",
]
