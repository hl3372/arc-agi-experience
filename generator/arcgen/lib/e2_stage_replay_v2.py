"""Stage-inclusive replay for the repaired E2 scheduling experiment.

The headline E2 comparison must use the same unit on both sides.  A scheduler
stage begins when the coordinator dispatches it and ends after every model
attempt, parser, deterministic check, semantic-state write, and terminal
failure decision belonging to that stage.  Replaying those exact durations
keeps retries and wrapper work in both the streamed and barrier schedules.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from typing import Any

from .util import content_hash

PIPELINE = "arcgen-e2-stage-replay"
VERSION = "2.0"
STAGES = ("planning", "realization", "technical", "blind")
TERMINAL_EVENT_KINDS = frozenset(
    {
        "stage_completed",
        "stage_failed",
        "stage_recovered_completed",
        "stage_recovered_terminal",
        "stage_interrupted",
        "stage_substep_completed",
    }
)


class E2StageReplayError(RuntimeError):
    """Raised when a stage trace cannot support an exact replay."""


def _stage_index(stage: str) -> int:
    try:
        return STAGES.index(stage)
    except ValueError as exc:
        raise E2StageReplayError(f"unknown E2 stage {stage!r}") from exc


def build_stage_ledger(
    scheduler_state: Mapping[str, Any], semantic_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Parse scheduler events and bind every stage to its original slate."""

    events = scheduler_state.get("events")
    if not isinstance(events, list):
        raise E2StageReplayError("scheduler state has no event list")
    records = semantic_state.get("semantic_records") or semantic_state.get("records")
    if not isinstance(records, Mapping):
        raise E2StageReplayError("semantic state has no semantic_records")
    metadata = {
        str(candidate): {
            "slate_index": int(record.get("slate_index") or 0),
            "development_priority": int(record.get("source_development_priority") or 0),
        }
        for candidate, record in records.items()
        if isinstance(record, Mapping)
    }
    active: dict[tuple[str, str], list[dict[str, Any]]] = {}
    invocations: list[dict[str, Any]] = []
    errors: list[str] = []
    dispatch_serial = 0
    for raw in sorted(events, key=lambda row: int(row.get("serial") or 0)):
        kind = str(raw.get("kind") or "")
        candidate = str(raw.get("candidate_id") or "")
        stage = str(raw.get("stage") or "")
        if candidate not in metadata:
            errors.append(f"event references unknown candidate {candidate}")
            continue
        if stage not in STAGES:
            errors.append(f"event references unknown stage {stage}")
            continue
        key = (candidate, stage)
        if kind == "stage_dispatched":
            dispatch_serial += 1
            invocation_id = str(
                raw.get("invocation_id")
                or f"historical-{candidate}-{stage}-{dispatch_serial}"
            )
            active.setdefault(key, []).append(
                {
                    "candidate_id": candidate,
                    "stage": stage,
                    "invocation_id": invocation_id,
                    "dispatch_serial": int(raw.get("serial") or dispatch_serial),
                    "dispatched_at_epoch_seconds": float(raw["at_epoch_seconds"]),
                    "ready_at_epoch_seconds": raw.get("ready_at_epoch_seconds"),
                    "scheduler_queue_wait_seconds": float(
                        raw.get("scheduler_queue_wait_seconds") or 0
                    ),
                    "recovered": False,
                }
            )
            continue
        if kind not in TERMINAL_EVENT_KINDS:
            continue
        pending = active.get(key) or []
        if not pending:
            errors.append(f"{candidate}/{stage} has {kind} without dispatch")
            continue
        requested_id = raw.get("invocation_id")
        selected_index = None
        if requested_id is not None:
            for index, invocation in enumerate(pending):
                if invocation["invocation_id"] == requested_id:
                    selected_index = index
                    break
        if selected_index is None:
            selected_index = 0
        invocation = pending.pop(selected_index)
        completed = float(raw["at_epoch_seconds"])
        if completed < float(invocation["dispatched_at_epoch_seconds"]):
            errors.append(f"{candidate}/{stage} completed before dispatch")
            continue
        invocation.update(
            {
                "terminal_event_kind": kind,
                "completed_at_epoch_seconds": completed,
                "duration_seconds": round(
                    completed - float(invocation["dispatched_at_epoch_seconds"]),
                    6,
                ),
                "error": raw.get("error"),
                "recovered": kind.startswith("stage_recovered"),
            }
        )
        invocations.append(invocation)
    for (candidate, stage), pending in active.items():
        for _invocation in pending:
            errors.append(f"{candidate}/{stage} dispatch lacks terminal event")

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for invocation in invocations:
        by_key.setdefault(
            (str(invocation["candidate_id"]), str(invocation["stage"])), []
        ).append(invocation)
    stages: list[dict[str, Any]] = []
    for (candidate, stage), values in sorted(
        by_key.items(), key=lambda item: min(row["dispatch_serial"] for row in item[1])
    ):
        values.sort(key=lambda row: int(row["dispatch_serial"]))
        semantic_terminal = [
            row
            for row in values
            if row["terminal_event_kind"]
            in {
                "stage_completed",
                "stage_failed",
                "stage_recovered_completed",
                "stage_recovered_terminal",
            }
        ]
        if len(semantic_terminal) != 1:
            errors.append(
                f"{candidate}/{stage} must have exactly one semantic terminal event"
            )
        stages.append(
            {
                "candidate_id": candidate,
                "slate_index": metadata[candidate]["slate_index"],
                "development_priority": metadata[candidate]["development_priority"],
                "stage": stage,
                "stage_index": _stage_index(stage),
                "original_dispatch_serial": min(
                    int(row["dispatch_serial"]) for row in values
                ),
                "duration_seconds": round(
                    sum(float(row["duration_seconds"]) for row in values), 6
                ),
                "scheduler_queue_wait_seconds": round(
                    sum(float(row["scheduler_queue_wait_seconds"]) for row in values),
                    6,
                ),
                "invocations": deepcopy(values),
                "recovery_count": sum(bool(row["recovered"]) for row in values),
                "terminal_event_kind": (
                    semantic_terminal[0]["terminal_event_kind"]
                    if len(semantic_terminal) == 1
                    else None
                ),
            }
        )

    stage_sets: dict[str, list[int]] = {}
    for row in stages:
        stage_sets.setdefault(str(row["candidate_id"]), []).append(
            int(row["stage_index"])
        )
    for candidate, indices in stage_sets.items():
        ordered = sorted(indices)
        if ordered != list(range(max(ordered) + 1)) or len(indices) != len(
            set(indices)
        ):
            errors.append(f"{candidate} has skipped or duplicate semantic stages")

    started = scheduler_state.get("started_at_epoch_seconds")
    completed = scheduler_state.get("completed_at_epoch_seconds")
    inclusive_wall = (
        round(float(completed) - float(started), 6)
        if isinstance(started, (int, float))
        and isinstance(completed, (int, float))
        and completed >= started
        else None
    )
    active_coordinator = scheduler_state.get("active_coordinator_seconds")
    observed_wall = (
        round(float(active_coordinator), 6)
        if isinstance(active_coordinator, (int, float))
        else inclusive_wall
    )
    active_stage_seconds = round(
        sum(float(row["duration_seconds"]) for row in stages), 6
    )
    return {
        "schema_version": 2,
        "pipeline": PIPELINE,
        "version": VERSION,
        "coverage_complete": not errors,
        "coverage_errors": errors,
        "worker_count": int(scheduler_state.get("workers") or 0),
        "max_downstream_before_new_plan": int(
            scheduler_state.get("max_downstream_before_new_plan") or 0
        ),
        "observed_scheduler_wall_seconds": observed_wall,
        "inclusive_restart_span_seconds": inclusive_wall,
        "offline_seconds": float(scheduler_state.get("offline_seconds") or 0),
        "recovery_count": sum(bool(row["recovered"]) for row in invocations),
        "primary_timing_qualified": bool(
            scheduler_state.get("primary_timing_qualified", True)
        )
        and not bool(scheduler_state.get("recovery_timing_uncertain")),
        "active_stage_seconds": active_stage_seconds,
        "observed_stage_effective_concurrency": (
            round(active_stage_seconds / observed_wall, 6) if observed_wall else None
        ),
        "observed_worker_utilization": (
            round(
                active_stage_seconds
                / (int(scheduler_state.get("workers") or 0) * observed_wall),
                6,
            )
            if observed_wall and int(scheduler_state.get("workers") or 0)
            else None
        ),
        "stage_count": len(stages),
        "stages_by_name": dict(
            sorted(Counter(str(row["stage"]) for row in stages).items())
        ),
        "stages": stages,
        "work_identity_sha256": content_hash(
            [
                {
                    "candidate_id": row["candidate_id"],
                    "slate_index": row["slate_index"],
                    "stage": row["stage"],
                    "duration_seconds": row["duration_seconds"],
                    "terminal_event_kind": row["terminal_event_kind"],
                }
                for row in stages
            ]
        ),
    }


def reconcile_attempts_with_stages(
    stage_ledger: Mapping[str, Any], attempt_ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove every logical call lies inside one candidate-stage invocation."""

    lookup = {
        (str(row["candidate_id"]), str(row["stage"])): row
        for row in stage_ledger.get("stages", [])
    }
    phase_to_stage = {
        "planning": "planning",
        "realization": "realization",
        "technical": "technical",
        "blind_and_conditional": "blind",
    }
    errors: list[str] = []
    bindings = []
    for call in attempt_ledger.get("logical_calls", []):
        candidate = str(call.get("candidate_id") or "")
        stage = phase_to_stage.get(str(call.get("phase") or ""))
        node = lookup.get((candidate, str(stage))) if stage else None
        if node is None:
            errors.append(f"{call.get('call_id')}: no matching scheduler stage")
            continue
        first = call.get("first_queued_at_epoch_seconds")
        last = call.get("last_completed_at_epoch_seconds")
        invocations = node.get("invocations") or []
        inside = any(
            isinstance(first, (int, float))
            and isinstance(last, (int, float))
            and float(first) >= float(invocation["dispatched_at_epoch_seconds"]) - 0.1
            and float(last) <= float(invocation["completed_at_epoch_seconds"]) + 0.1
            for invocation in invocations
        )
        if not inside:
            errors.append(
                f"{call.get('call_id')}: attempt envelope falls outside {candidate}/{stage}"
            )
        bindings.append(
            {
                "call_id": call.get("call_id"),
                "candidate_id": candidate,
                "scheduler_stage": stage,
                "inside_stage_interval": inside,
            }
        )
    return {
        "passed": not errors,
        "errors": errors,
        "binding_count": len(bindings),
        "bindings": bindings,
    }


def _execution_nodes(stages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand semantic stages into retry/substep-complete execution nodes.

    A second dispatch of a technical or blind stage is not necessarily a retry:
    after ``stage_substep_completed`` it is conditional resolution work.  Keeping
    that distinction is essential because the five-call budget is allocated only
    after all primaries in a slate have reported their needs.
    """

    nodes: list[dict[str, Any]] = []
    for stage_row in stages:
        invocations = stage_row.get("invocations")
        if not isinstance(invocations, list) or not invocations:
            invocations = [
                {
                    "invocation_id": (
                        f"synthetic-{stage_row['candidate_id']}-{stage_row['stage']}"
                    ),
                    "dispatch_serial": stage_row["original_dispatch_serial"],
                    "duration_seconds": stage_row["duration_seconds"],
                    "terminal_event_kind": stage_row.get("terminal_event_kind")
                    or "stage_completed",
                }
            ]
        substep = 0
        for invocation_index, invocation in enumerate(invocations):
            node_id = str(invocation.get("invocation_id") or "")
            if not node_id:
                raise E2StageReplayError("stage invocation lacks an invocation ID")
            nodes.append(
                {
                    "node_id": node_id,
                    "candidate_id": str(stage_row["candidate_id"]),
                    "slate_index": int(stage_row["slate_index"]),
                    "development_priority": int(stage_row["development_priority"]),
                    "stage": str(stage_row["stage"]),
                    "stage_index": int(stage_row["stage_index"]),
                    "invocation_index": invocation_index,
                    "semantic_substep_index": substep,
                    "original_dispatch_serial": int(
                        invocation.get("dispatch_serial")
                        or stage_row["original_dispatch_serial"]
                    ),
                    "duration_seconds": float(invocation["duration_seconds"]),
                    "terminal_event_kind": str(
                        invocation.get("terminal_event_kind") or "stage_completed"
                    ),
                }
            )
            if invocation.get("terminal_event_kind") == "stage_substep_completed":
                substep += 1
    identities = [str(row["node_id"]) for row in nodes]
    if len(identities) != len(set(identities)):
        raise E2StageReplayError("execution-node IDs are not unique")
    return nodes


def _terminal_node(
    rows: Sequence[Mapping[str, Any]], *, substep: int | None = None
) -> Mapping[str, Any]:
    eligible = [
        row
        for row in rows
        if substep is None or int(row["semantic_substep_index"]) == substep
    ]
    if not eligible:
        raise E2StageReplayError("execution substep has no terminal node")
    return max(eligible, key=lambda row: int(row["invocation_index"]))


def _dependency_graph(
    stages: Sequence[Mapping[str, Any]], *, global_barriers: bool
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    nodes = _execution_nodes(stages)
    by_candidate_stage: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for node in nodes:
        by_candidate_stage.setdefault(
            (str(node["candidate_id"]), str(node["stage"])), []
        ).append(node)
    for rows in by_candidate_stage.values():
        rows.sort(key=lambda row: int(row["invocation_index"]))

    dependencies: dict[str, set[str]] = {str(node["node_id"]): set() for node in nodes}
    for (candidate, stage), rows in by_candidate_stage.items():
        del candidate, stage
        for prior, current in pairwise(rows):
            dependencies[str(current["node_id"])].add(str(prior["node_id"]))

    # Ordinary candidate-local stage dependencies.
    candidates = sorted({str(node["candidate_id"]) for node in nodes})
    for candidate in candidates:
        previous: Mapping[str, Any] | None = None
        for stage in STAGES:
            rows = by_candidate_stage.get((candidate, stage), [])
            if not rows:
                continue
            if previous is not None:
                dependencies[str(rows[0]["node_id"])].add(str(previous["node_id"]))
            previous = _terminal_node(rows)

    if global_barriers:
        # Traditional orchestration finishes all work, including retries and
        # conditional substeps, in a stage before opening the next stage.
        for later_index in range(1, len(STAGES)):
            prior_ids = {
                str(node["node_id"])
                for node in nodes
                if int(node["stage_index"]) < later_index
            }
            for node in nodes:
                if int(node["stage_index"]) == later_index:
                    dependencies[str(node["node_id"])].update(prior_ids)
    # Conditional allocation is a legitimate per-slate synchronization in
    # both schedules. Primary completion order cannot determine who consumes
    # the finite budget; resolution candidates execute in frozen development
    # priority after all eligible primaries are known.
    slates = sorted({int(node["slate_index"]) for node in nodes})
    for slate in slates:
        for stage in ("technical", "blind"):
            stage_rows = [
                node
                for node in nodes
                if int(node["slate_index"]) == slate and str(node["stage"]) == stage
            ]
            if not stage_rows:
                continue
            primary_terminal_ids: set[str] = set()
            grouped: dict[str, list[dict[str, Any]]] = {}
            for node in stage_rows:
                grouped.setdefault(str(node["candidate_id"]), []).append(node)
            for rows in grouped.values():
                primary_terminal_ids.add(
                    str(_terminal_node(rows, substep=0)["node_id"])
                )
            resolution_candidates = sorted(
                (
                    candidate
                    for candidate, rows in grouped.items()
                    if any(int(row["semantic_substep_index"]) > 0 for row in rows)
                ),
                key=lambda candidate: (
                    int(grouped[candidate][0]["development_priority"]),
                    candidate,
                ),
            )
            prior_resolution_terminal: str | None = None
            for candidate in resolution_candidates:
                resolution_rows = [
                    row
                    for row in grouped[candidate]
                    if int(row["semantic_substep_index"]) > 0
                ]
                first = min(
                    resolution_rows, key=lambda row: int(row["invocation_index"])
                )
                dependencies[str(first["node_id"])].update(primary_terminal_ids)
                if stage == "blind":
                    # Blind resolution eligibility is not frozen until all
                    # technical work in the slate has terminated.
                    dependencies[str(first["node_id"])].update(
                        str(_terminal_node(rows)["node_id"])
                        for (
                            _candidate_key,
                            stage_key,
                        ), rows in by_candidate_stage.items()
                        if stage_key == "technical"
                        and int(rows[0]["slate_index"]) == slate
                    )
                if prior_resolution_terminal is not None:
                    dependencies[str(first["node_id"])].add(prior_resolution_terminal)
                prior_resolution_terminal = str(
                    _terminal_node(resolution_rows)["node_id"]
                )
    return nodes, dependencies


def _schedule_graph(
    nodes: Sequence[Mapping[str, Any]],
    dependencies: Mapping[str, set[str]],
    *,
    workers: int,
    max_downstream_before_new_plan: int,
) -> dict[str, Any]:
    if workers < 1 or max_downstream_before_new_plan < 1:
        raise E2StageReplayError("invalid scheduler limits")
    by_id = {str(node["node_id"]): deepcopy(dict(node)) for node in nodes}
    completed: set[str] = set()
    dispatched: set[str] = set()
    running: list[tuple[float, int, str, int]] = []
    available_slots = list(range(workers))
    heapq.heapify(available_slots)
    now = 0.0
    serial = 0
    downstream_since_plan = 0
    timeline: list[dict[str, Any]] = []
    while len(completed) < len(by_id):
        while available_slots:
            ready = [
                row
                for node_id, row in by_id.items()
                if node_id not in completed
                and node_id not in dispatched
                and dependencies[node_id].issubset(completed)
            ]
            if not ready:
                break
            plans = [row for row in ready if row["stage"] == "planning"]
            if plans and downstream_since_plan >= max_downstream_before_new_plan:
                selected = min(
                    plans,
                    key=lambda row: (
                        int(row["original_dispatch_serial"]),
                        str(row["candidate_id"]),
                    ),
                )
            else:
                selected = min(
                    ready,
                    key=lambda row: (
                        -int(row["stage_index"]),
                        int(row["original_dispatch_serial"]),
                        int(row["development_priority"]),
                        str(row["candidate_id"]),
                    ),
                )
            node_id = str(selected["node_id"])
            slot = heapq.heappop(available_slots)
            finish = now + float(selected["duration_seconds"])
            serial += 1
            heapq.heappush(running, (finish, serial, node_id, slot))
            dispatched.add(node_id)
            if selected["stage"] == "planning":
                downstream_since_plan = 0
            else:
                downstream_since_plan += 1
            timeline.append(
                {
                    "node_id": node_id,
                    "candidate_id": selected["candidate_id"],
                    "slate_index": selected["slate_index"],
                    "stage": selected["stage"],
                    "semantic_substep_index": selected["semantic_substep_index"],
                    "worker_slot": slot,
                    "start_seconds": round(now, 6),
                    "finish_seconds": round(finish, 6),
                }
            )
        if not running:
            remaining = sorted(set(by_id) - completed)
            raise E2StageReplayError(
                "execution replay deadlocked: " + ", ".join(remaining)
            )
        finish = running[0][0]
        now = finish
        finishing = []
        while running and abs(running[0][0] - finish) < 1e-9:
            finishing.append(heapq.heappop(running))
        for _finish, _serial, node_id, slot in finishing:
            completed.add(node_id)
            heapq.heappush(available_slots, slot)
    active = sum(float(row["duration_seconds"]) for row in nodes)
    return {
        "makespan_seconds": round(now, 6),
        "active_stage_seconds": round(active, 6),
        "effective_concurrency": round(active / now, 6) if now else 0.0,
        "worker_utilization": round(active / (workers * now), 6) if now else 0.0,
        "execution_node_count": len(nodes),
        "timeline": timeline,
    }


def simulate_barrier(
    stages: Sequence[Mapping[str, Any]], *, workers: int
) -> dict[str, Any]:
    """Replay traditional global stage barriers over the exact stage work."""
    nodes, dependencies = _dependency_graph(stages, global_barriers=True)
    result = _schedule_graph(
        nodes,
        dependencies,
        workers=workers,
        max_downstream_before_new_plan=1,
    )
    result["phases"] = [
        {
            "stage": stage,
            "execution_node_count": sum(row["stage"] == stage for row in nodes),
        }
        for stage in STAGES
    ]
    return result


def simulate_streaming(
    stages: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    max_downstream_before_new_plan: int,
) -> dict[str, Any]:
    """Replay the exact E2 priority and starvation policy."""

    nodes, dependencies = _dependency_graph(stages, global_barriers=False)
    return _schedule_graph(
        nodes,
        dependencies,
        workers=workers,
        max_downstream_before_new_plan=max_downstream_before_new_plan,
    )


def replay_pooled_and_independent(stage_ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Keep pooled continuous-factory and isolated-slate estimates distinct."""

    if stage_ledger.get("coverage_complete") is not True:
        raise E2StageReplayError("stage coverage is incomplete")
    stages = list(stage_ledger["stages"])
    workers = int(stage_ledger["worker_count"])
    starvation = int(stage_ledger["max_downstream_before_new_plan"])
    pooled_barrier = simulate_barrier(stages, workers=workers)
    pooled_streaming = simulate_streaming(
        stages,
        workers=workers,
        max_downstream_before_new_plan=starvation,
    )
    slate_rows = []
    for slate in sorted({int(row["slate_index"]) for row in stages}):
        rows = [row for row in stages if int(row["slate_index"]) == slate]
        barrier = simulate_barrier(rows, workers=workers)
        streaming = simulate_streaming(
            rows,
            workers=workers,
            max_downstream_before_new_plan=starvation,
        )
        slate_rows.append(
            {
                "slate_index": slate,
                "candidate_count": len({str(row["candidate_id"]) for row in rows}),
                "stage_count": len(rows),
                "barrier_makespan_seconds": barrier["makespan_seconds"],
                "streaming_makespan_seconds": streaming["makespan_seconds"],
                "estimated_reduction_fraction": (
                    round(
                        1
                        - float(streaming["makespan_seconds"])
                        / float(barrier["makespan_seconds"]),
                        6,
                    )
                    if barrier["makespan_seconds"]
                    else 0.0
                ),
            }
        )
    independent_barrier = sum(
        float(row["barrier_makespan_seconds"]) for row in slate_rows
    )
    independent_streaming = sum(
        float(row["streaming_makespan_seconds"]) for row in slate_rows
    )
    return {
        "schema_version": 2,
        "work_identity_sha256": stage_ledger["work_identity_sha256"],
        "pooled_continuous_factory": {
            "observed_streaming_wall_seconds": stage_ledger[
                "observed_scheduler_wall_seconds"
            ],
            "replayed_streaming": pooled_streaming,
            "barrier_counterfactual": pooled_barrier,
            "observed_reduction_fraction": round(
                1
                - float(stage_ledger["observed_scheduler_wall_seconds"])
                / float(pooled_barrier["makespan_seconds"]),
                6,
            ),
            "replay_reduction_fraction": round(
                1
                - float(pooled_streaming["makespan_seconds"])
                / float(pooled_barrier["makespan_seconds"]),
                6,
            ),
            "isolated_e2_adoption_eligible": False,
        },
        "independent_slates": {
            "slates": slate_rows,
            "aggregate_barrier_makespan_seconds": round(independent_barrier, 6),
            "aggregate_streaming_makespan_seconds": round(independent_streaming, 6),
            "aggregate_estimated_reduction_fraction": (
                round(1 - independent_streaming / independent_barrier, 6)
                if independent_barrier
                else 0.0
            ),
            "observed_live_independent_slates": False,
            "retrospective_simulation_only": True,
        },
    }


__all__ = [
    "STAGES",
    "E2StageReplayError",
    "build_stage_ledger",
    "reconcile_attempts_with_stages",
    "replay_pooled_and_independent",
    "simulate_barrier",
    "simulate_streaming",
]
