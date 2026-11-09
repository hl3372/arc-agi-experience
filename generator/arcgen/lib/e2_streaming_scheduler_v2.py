"""Crash-reconciling candidate scheduler for the repaired E2 experiment.

This is an additive replacement for the historical E2 scheduler.  It persists
priority state, invocation IDs, coordinator execution sessions, and an atomic
dispatch/terminal event journal.  Resume reconciles the scheduler journal with
the already-persisted semantic state before it dispatches more work.
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
VERSION = "2.0"
STAGES = ("planning", "realization", "technical", "blind")
TERMINAL_EVENTS = frozenset(
    {
        "stage_completed",
        "stage_failed",
        "stage_interrupted",
        "stage_recovered_completed",
        "stage_recovered_terminal",
        "stage_substep_completed",
    }
)


class E2SchedulerV2Error(RuntimeError):
    """Raised when persistence, dependencies, or resume invariants fail."""


@dataclass(frozen=True)
class StageResult:
    candidate_id: str
    stage: str
    invocation_id: str
    error: str | None
    semantic_stage_complete: bool


class PersistentCandidateSchedulerV2:
    """Advance independent candidate chains under one global worker ceiling."""

    def __init__(
        self,
        *,
        state_path: Path,
        candidate_ids: Sequence[str],
        workers: int,
        next_stage: Callable[[str], str | None],
        run_stage: Callable[[str, str], bool | None],
        can_dispatch: Callable[[str, str], bool] | None = None,
        reconcile_active: Callable[[str, str], str] | None = None,
        max_downstream_before_new_plan: int = 2,
        clock: Callable[[], float] = time.time,
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        if workers < 1:
            raise E2SchedulerV2Error("workers must be positive")
        if max_downstream_before_new_plan < 1:
            raise E2SchedulerV2Error("starvation limit must be positive")
        if heartbeat_interval_seconds <= 0:
            raise E2SchedulerV2Error("heartbeat interval must be positive")
        ids = [str(candidate) for candidate in candidate_ids]
        if not ids or len(ids) != len(set(ids)):
            raise E2SchedulerV2Error("candidate IDs must be non-empty and unique")
        self.state_path = state_path
        self.candidate_ids = tuple(sorted(ids))
        self.workers = int(workers)
        self.next_stage = next_stage
        self.run_stage = run_stage
        self.can_dispatch = can_dispatch or (lambda _candidate, _stage: True)
        self.reconcile_active = reconcile_active
        self.max_downstream_before_new_plan = int(max_downstream_before_new_plan)
        self.clock = clock
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self.state = self._load_or_initialize()

    def _fingerprint(self) -> str:
        return content_hash(
            {
                "pipeline": PIPELINE,
                "version": VERSION,
                "candidate_ids": self.candidate_ids,
                "workers": self.workers,
                "max_downstream_before_new_plan": self.max_downstream_before_new_plan,
            }
        )

    def _now(self) -> float:
        return round(float(self.clock()), 6)

    @staticmethod
    def _stage_index(stage: str) -> int:
        try:
            return STAGES.index(stage)
        except ValueError as exc:
            raise E2SchedulerV2Error(f"unknown stage {stage!r}") from exc

    def _append_event_locked(
        self,
        kind: str,
        candidate_id: str,
        stage: str,
        *,
        invocation_id: str,
        at: float | None = None,
        **extra: Any,
    ) -> None:
        serial = int(self.state.get("event_serial") or 0) + 1
        self.state["event_serial"] = serial
        self.state.setdefault("events", []).append(
            {
                "serial": serial,
                "kind": kind,
                "candidate_id": candidate_id,
                "stage": stage,
                "invocation_id": invocation_id,
                "at_epoch_seconds": self._now() if at is None else round(at, 6),
                **deepcopy(extra),
            }
        )

    def _write_locked(self, *, heartbeat: bool = True) -> None:
        if heartbeat:
            now = self._now()
            self.state["last_heartbeat_at_epoch_seconds"] = now
            sessions = self.state.get("execution_sessions") or []
            if sessions and sessions[-1].get("ended_at_epoch_seconds") is None:
                sessions[-1]["last_heartbeat_at_epoch_seconds"] = now
        atomic_write_json(self.state_path, self.state)

    def _new_state(self) -> dict[str, Any]:
        now = self._now()
        return {
            "schema_version": 2,
            "pipeline": PIPELINE,
            "version": VERSION,
            "fingerprint": self._fingerprint(),
            "candidate_ids": list(self.candidate_ids),
            "workers": self.workers,
            "max_downstream_before_new_plan": self.max_downstream_before_new_plan,
            "started_at_epoch_seconds": now,
            "completed": False,
            "event_serial": 0,
            "events": [],
            "downstream_since_plan": 0,
            "execution_sessions": [
                {
                    "session_index": 1,
                    "started_at_epoch_seconds": now,
                    "last_heartbeat_at_epoch_seconds": now,
                    "ended_at_epoch_seconds": None,
                    "outcome": "running",
                }
            ],
            "offline_gaps": [],
            "last_heartbeat_at_epoch_seconds": now,
            "candidates": {
                candidate: {
                    "active_stage": None,
                    "active_invocation_id": None,
                    "ready_stage": None,
                    "ready_at_epoch_seconds": None,
                    "completed_stage_count": 0,
                    "stage_dispatch_counts": {},
                    "recovery_count": 0,
                    "recoveries": [],
                }
                for candidate in self.candidate_ids
            },
        }

    def _open_resume_session_locked(self) -> None:
        now = self._now()
        sessions = self.state.setdefault("execution_sessions", [])
        if sessions and sessions[-1].get("ended_at_epoch_seconds") is None:
            heartbeat = float(
                sessions[-1].get("last_heartbeat_at_epoch_seconds")
                or self.state.get("last_heartbeat_at_epoch_seconds")
                or sessions[-1]["started_at_epoch_seconds"]
            )
            sessions[-1]["ended_at_epoch_seconds"] = heartbeat
            sessions[-1]["outcome"] = "interrupted"
            self.state.setdefault("offline_gaps", []).append(
                {
                    "after_session_index": sessions[-1]["session_index"],
                    "from_epoch_seconds": heartbeat,
                    "to_epoch_seconds": now,
                    "seconds": round(max(0.0, now - heartbeat), 6),
                }
            )
        sessions.append(
            {
                "session_index": len(sessions) + 1,
                "started_at_epoch_seconds": now,
                "last_heartbeat_at_epoch_seconds": now,
                "ended_at_epoch_seconds": None,
                "outcome": "running",
            }
        )

    def _reconcile_active_locked(self) -> None:
        now = self._now()
        sessions = self.state.get("execution_sessions") or []
        prior_active_boundary = (
            float(sessions[-2]["ended_at_epoch_seconds"])
            if len(sessions) >= 2
            and isinstance(sessions[-2].get("ended_at_epoch_seconds"), (int, float))
            else float(self.state.get("last_heartbeat_at_epoch_seconds") or now)
        )
        for candidate in self.candidate_ids:
            row = self.state["candidates"][candidate]
            active = row.get("active_stage")
            if active is None:
                continue
            invocation_id = str(row.get("active_invocation_id") or "")
            if not invocation_id:
                raise E2SchedulerV2Error(
                    f"active stage lacks invocation ID: {candidate}/{active}"
                )
            current = self.next_stage(candidate)
            if current is not None and current not in STAGES:
                raise E2SchedulerV2Error(
                    f"unknown semantic next stage {current!r} for {candidate}"
                )
            explicit = (
                self.reconcile_active(candidate, str(active))
                if self.reconcile_active is not None
                else None
            )
            if explicit not in {None, "requeue", "completed", "substep", "terminal"}:
                raise E2SchedulerV2Error(
                    f"invalid resume reconciliation {explicit!r} for {candidate}/{active}"
                )
            if explicit == "requeue" or (explicit is None and current == active):
                kind = "stage_interrupted"
                outcome = "requeue_same_semantic_stage"
            elif explicit == "substep":
                kind = "stage_substep_completed"
                outcome = "semantic_substep_already_persisted"
            elif explicit == "terminal" or (explicit is None and current is None):
                kind = "stage_recovered_terminal"
                outcome = "semantic_state_already_terminal"
                row["completed_stage_count"] = (
                    int(row.get("completed_stage_count") or 0) + 1
                )
            elif explicit == "completed" or (
                explicit is None
                and current is not None
                and self._stage_index(current) > self._stage_index(str(active))
            ):
                kind = "stage_recovered_completed"
                outcome = "semantic_state_already_advanced"
                row["completed_stage_count"] = (
                    int(row.get("completed_stage_count") or 0) + 1
                )
            else:
                raise E2SchedulerV2Error(
                    f"semantic state moved backward during resume: {candidate}: "
                    f"active={active}, current={current}"
                )
            self._append_event_locked(
                kind,
                candidate,
                str(active),
                invocation_id=invocation_id,
                # Never charge machine-offline time to the active stage.  The
                # exact semantic completion instant is unknowable after a
                # coordinator crash, so any recovered trace is explicitly
                # disqualified from primary timing below.
                at=prior_active_boundary,
                recovery_outcome=outcome,
                reconciled_at_epoch_seconds=now,
                completion_time_uncertain=True,
            )
            self.state["recovery_timing_uncertain"] = True
            row["active_stage"] = None
            row["active_invocation_id"] = None
            row["ready_stage"] = None
            row["ready_at_epoch_seconds"] = None
            row["recovery_count"] = int(row.get("recovery_count") or 0) + 1
            row.setdefault("recoveries", []).append(
                {
                    "stage": active,
                    "invocation_id": invocation_id,
                    "at_epoch_seconds": now,
                    "outcome": outcome,
                }
            )

    def _load_or_initialize(self) -> dict[str, Any]:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_path.is_file():
            state = self._new_state()
            atomic_write_json(self.state_path, state)
            return state
        state = read_json(self.state_path)
        if state.get("fingerprint") != self._fingerprint():
            raise E2SchedulerV2Error("scheduler state belongs to different inputs")
        if state.get("completed") is True:
            return state
        if state.get("fatal_stage_error"):
            raise E2SchedulerV2Error(
                "scheduler has an unresolved fatal wrapper error; explicit code/state "
                "repair is required before a new qualification"
            )
        self.state = state
        with self.lock:
            self._open_resume_session_locked()
            self._reconcile_active_locked()
            self._write_locked()
        return self.state

    def _close_session_locked(self, outcome: str) -> None:
        sessions = self.state.get("execution_sessions") or []
        if not sessions or sessions[-1].get("ended_at_epoch_seconds") is not None:
            return
        now = self._now()
        sessions[-1]["last_heartbeat_at_epoch_seconds"] = now
        sessions[-1]["ended_at_epoch_seconds"] = now
        sessions[-1]["outcome"] = outcome

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_interval_seconds):
            with self.lock:
                if self.state.get("completed") is True:
                    return
                self._write_locked()

    def _refresh_ready_locked(self, running_ids: set[str]) -> None:
        for candidate in self.candidate_ids:
            if candidate in running_ids:
                continue
            row = self.state["candidates"][candidate]
            stage = self.next_stage(candidate)
            if stage is not None and stage not in STAGES:
                raise E2SchedulerV2Error(
                    f"unknown next stage {stage!r} for {candidate}"
                )
            if row.get("ready_stage") != stage:
                row["ready_stage"] = stage
                row["ready_at_epoch_seconds"] = self._now() if stage else None

    def _pick_ready_locked(self) -> tuple[str, str] | None:
        ready = []
        for candidate in self.candidate_ids:
            row = self.state["candidates"][candidate]
            stage = row.get("ready_stage")
            if stage is None or row.get("active_stage") is not None:
                continue
            if not self.can_dispatch(candidate, str(stage)):
                continue
            ready.append(
                (
                    candidate,
                    str(stage),
                    float(row.get("ready_at_epoch_seconds") or 0),
                )
            )
        if not ready:
            return None
        plans = [item for item in ready if item[1] == "planning"]
        if plans and int(self.state["downstream_since_plan"]) >= (
            self.max_downstream_before_new_plan
        ):
            candidate, stage, _ready = min(plans, key=lambda item: (item[2], item[0]))
            return candidate, stage
        candidate, stage, _ready = min(
            ready,
            key=lambda item: (-self._stage_index(item[1]), item[2], item[0]),
        )
        return candidate, stage

    def _dispatch_locked(self, candidate: str, stage: str) -> str:
        row = self.state["candidates"][candidate]
        count = int(row["stage_dispatch_counts"].get(stage, 0)) + 1
        invocation_id = f"{candidate}-{stage}-invocation-{count:03d}"
        ready_at = float(row["ready_at_epoch_seconds"])
        now = self._now()
        row["stage_dispatch_counts"][stage] = count
        row["active_stage"] = stage
        row["active_invocation_id"] = invocation_id
        row["ready_stage"] = None
        row["ready_at_epoch_seconds"] = None
        if stage == "planning":
            self.state["downstream_since_plan"] = 0
        else:
            self.state["downstream_since_plan"] = (
                int(self.state["downstream_since_plan"]) + 1
            )
        self._append_event_locked(
            "stage_dispatched",
            candidate,
            stage,
            invocation_id=invocation_id,
            at=now,
            ready_at_epoch_seconds=ready_at,
            scheduler_queue_wait_seconds=round(now - ready_at, 6),
        )
        self._write_locked()
        return invocation_id

    def _execute(self, candidate: str, stage: str, invocation_id: str) -> StageResult:
        try:
            completed = self.run_stage(candidate, stage)
            return StageResult(
                candidate,
                stage,
                invocation_id,
                None,
                True if completed is None else bool(completed),
            )
        except Exception as exc:  # noqa: BLE001 - fatal event retains exact error
            return StageResult(candidate, stage, invocation_id, str(exc), False)

    def _complete_locked(self, result: StageResult) -> None:
        row = self.state["candidates"][result.candidate_id]
        if (
            row.get("active_stage") != result.stage
            or row.get("active_invocation_id") != result.invocation_id
        ):
            raise E2SchedulerV2Error(
                f"active invocation mismatch for {result.candidate_id}/{result.stage}"
            )
        row["active_stage"] = None
        row["active_invocation_id"] = None
        if result.error is None and result.semantic_stage_complete:
            row["completed_stage_count"] = (
                int(row.get("completed_stage_count") or 0) + 1
            )
        self._append_event_locked(
            (
                "stage_failed"
                if result.error
                else (
                    "stage_completed"
                    if result.semantic_stage_complete
                    else "stage_substep_completed"
                )
            ),
            result.candidate_id,
            result.stage,
            invocation_id=result.invocation_id,
            error=result.error,
        )
        if result.error:
            self.state["fatal_stage_error"] = {
                "candidate_id": result.candidate_id,
                "stage": result.stage,
                "invocation_id": result.invocation_id,
                "error": result.error,
                "at_epoch_seconds": self._now(),
            }
        self._write_locked()

    def run(self) -> dict[str, Any]:
        if self.state.get("completed") is True:
            return deepcopy(self.state)
        heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="e2-v2-heartbeat", daemon=True
        )
        heartbeat.start()
        running: dict[Future[StageResult], tuple[str, str, str]] = {}
        outcome = "failed"
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                while True:
                    running_ids = {
                        candidate for candidate, _stage, _invocation in running.values()
                    }
                    with self.lock:
                        self._refresh_ready_locked(running_ids)
                        self._write_locked()
                        while len(running) < self.workers:
                            picked = self._pick_ready_locked()
                            if picked is None:
                                break
                            candidate, stage = picked
                            invocation_id = self._dispatch_locked(candidate, stage)
                            future = executor.submit(
                                self._execute, candidate, stage, invocation_id
                            )
                            running[future] = (candidate, stage, invocation_id)
                    if not running:
                        unresolved = [
                            candidate
                            for candidate in self.candidate_ids
                            if self.next_stage(candidate) is not None
                        ]
                        if unresolved:
                            raise E2SchedulerV2Error(
                                "unresolved candidates but no runnable work: "
                                + ", ".join(unresolved)
                            )
                        break
                    done, _pending = wait(running, return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda item: running[item]):
                        running.pop(future)
                        result = future.result()
                        with self.lock:
                            self._complete_locked(result)
                        if result.error:
                            raise E2SchedulerV2Error(
                                f"candidate stage failed unexpectedly: "
                                f"{result.candidate_id}/{result.stage}: {result.error}"
                            )
            with self.lock:
                terminal = {
                    candidate: self.next_stage(candidate)
                    for candidate in self.candidate_ids
                }
                if any(terminal.values()):
                    raise E2SchedulerV2Error(
                        "scheduler completed with runnable candidates"
                    )
                self.state["terminal_next_stage"] = terminal
                self.state["completed"] = True
                self.state["completed_at_epoch_seconds"] = self._now()
                self._close_session_locked("completed")
                self.state["active_coordinator_seconds"] = round(
                    sum(
                        max(
                            0.0,
                            float(session["ended_at_epoch_seconds"])
                            - float(session["started_at_epoch_seconds"]),
                        )
                        for session in self.state.get("execution_sessions", [])
                        if isinstance(
                            session.get("ended_at_epoch_seconds"), (int, float)
                        )
                    ),
                    6,
                )
                self.state["offline_seconds"] = round(
                    sum(
                        float(gap.get("seconds") or 0)
                        for gap in self.state.get("offline_gaps", [])
                    ),
                    6,
                )
                self.state["primary_timing_qualified"] = not bool(
                    self.state.get("recovery_timing_uncertain")
                )
                self._write_locked()
            outcome = "completed"
            return deepcopy(self.state)
        finally:
            self._heartbeat_stop.set()
            heartbeat.join(timeout=max(1.0, self.heartbeat_interval_seconds * 2))
            if outcome != "completed":
                with self.lock:
                    self._close_session_locked("failed")
                    self._write_locked()


def validate_dependency_events_v2(state: Mapping[str, Any]) -> dict[str, Any]:
    """Require one terminal event per dispatch and no skipped semantic stage."""

    active: dict[str, tuple[str, str]] = {}
    dispatched_ids: set[str] = set()
    semantic_completed: dict[str, set[str]] = {}
    errors: list[str] = []
    dispatch_count = 0
    terminal_count = 0
    for event in sorted(
        state.get("events", []), key=lambda row: int(row.get("serial") or 0)
    ):
        kind = str(event.get("kind") or "")
        candidate = str(event.get("candidate_id") or "")
        stage = str(event.get("stage") or "")
        invocation = str(event.get("invocation_id") or "")
        if stage not in STAGES:
            errors.append(f"unknown stage {stage}")
            continue
        if kind == "stage_dispatched":
            dispatch_count += 1
            if not invocation or invocation in dispatched_ids:
                errors.append(f"duplicate/missing invocation ID {invocation}")
            dispatched_ids.add(invocation)
            if candidate in active:
                errors.append(f"{candidate} dispatched while another stage is active")
            prior = semantic_completed.setdefault(candidate, set())
            index = STAGES.index(stage)
            if index and STAGES[index - 1] not in prior:
                errors.append(f"{candidate} skipped dependency before {stage}")
            if stage in prior:
                errors.append(
                    f"{candidate}/{stage} dispatched after semantic completion"
                )
            active[candidate] = (stage, invocation)
            continue
        if kind not in TERMINAL_EVENTS:
            continue
        terminal_count += 1
        if active.get(candidate) != (stage, invocation):
            errors.append(f"{candidate}/{stage}/{invocation} terminal mismatch")
            continue
        del active[candidate]
        if kind in {
            "stage_completed",
            "stage_recovered_completed",
            "stage_recovered_terminal",
        }:
            completed = semantic_completed.setdefault(candidate, set())
            if stage in completed:
                errors.append(f"{candidate}/{stage} completed semantically twice")
            completed.add(stage)
    if active:
        errors.extend(
            f"{candidate}/{stage}/{invocation} lacks terminal event"
            for candidate, (stage, invocation) in active.items()
        )
    candidates = state.get("candidates") or {}
    for candidate, row in candidates.items():
        if row.get("active_stage") is not None:
            errors.append(f"{candidate} remains active in persisted state")
    if state.get("completed") is True and any(
        value is not None for value in (state.get("terminal_next_stage") or {}).values()
    ):
        errors.append("completed scheduler retains runnable semantic stages")
    return {
        "valid": not errors,
        "errors": errors,
        "dispatch_count": dispatch_count,
        "terminal_event_count": terminal_count,
        "semantic_completion_count": sum(map(len, semantic_completed.values())),
        "unique_invocation_count": len(dispatched_ids),
    }


__all__ = [
    "STAGES",
    "E2SchedulerV2Error",
    "PersistentCandidateSchedulerV2",
    "StageResult",
    "validate_dependency_events_v2",
]
