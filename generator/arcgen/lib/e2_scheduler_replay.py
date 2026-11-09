"""No-call replay of stage barriers versus candidate-level streaming.

The replay consumes completed V4 semantic state/call ledgers, removes editorial
work, and schedules the exact recorded post-selection calls in two ways:

* the current phase barriers (plan, realize, technical, blind/conditional);
* one per-candidate dependency chain under the same worker limit.

Prompts, responses, durations, failures, and terminal outcomes are not changed.
This is a directional E2 timing estimate, not a live adoption result.
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from .util import atomic_write_json, atomic_write_text, read_json

PIPELINE = "arcgen-e2-candidate-streaming-replay"
VERSION = "1.0"
PHASES = ("planning", "realization", "technical", "blind_and_conditional")
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


class E2ReplayError(RuntimeError):
    pass


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _candidate_for(call_id: str, candidates: Sequence[str]) -> str | None:
    matches = [candidate for candidate in candidates if call_id.startswith(f"{candidate}-")]
    if len(matches) > 1:
        raise E2ReplayError(f"ambiguous candidate prefix for {call_id}")
    if matches:
        return matches[0]
    # V4.6 outer arm states retain provider call ledgers without embedding the
    # semantic records.  Their content-addressed kernel/task IDs are carried in
    # every downstream call ID, sometimes after a provider prefix.
    embedded = sorted(set(re.findall(r"k[0-9a-f]{12}", call_id)))
    if len(embedded) > 1:
        raise E2ReplayError(f"multiple embedded candidate IDs for {call_id}")
    return embedded[0] if embedded else None


def _duration(call: Mapping[str, Any]) -> float:
    timing = call.get("timing") or {}
    value = timing.get("total_seconds")
    if not isinstance(value, (int, float)) or value < 0:
        raise E2ReplayError(f"completed call lacks non-negative timing: {call.get('call_id')}")
    return float(value)


def _queued(call: Mapping[str, Any]) -> float:
    value = (call.get("timing") or {}).get("queued_at_epoch_seconds")
    if not isinstance(value, (int, float)):
        raise E2ReplayError(f"completed call lacks queue timestamp: {call.get('call_id')}")
    return float(value)


def _calls(
    state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    candidates = sorted(str(key) for key in state.get("semantic_records", {}))
    rows: list[dict[str, Any]] = []
    unbound: list[str] = []
    unmeasured: list[str] = []
    for raw in state.get("call_ledger", {}).values():
        if raw.get("role") not in POST_SELECTION_ROLES or raw.get("status") not in {
            "complete",
            "failed",
        }:
            continue
        if raw.get("substantive_experimental_call") is False:
            continue
        call = deepcopy(dict(raw))
        candidate = _candidate_for(str(call.get("call_id", "")), candidates)
        if candidate is None:
            unbound.append(str(call.get("call_id")))
            continue
        timing = call.get("timing") or {}
        if (
            not isinstance(timing.get("total_seconds"), (int, float))
            or not isinstance(timing.get("queued_at_epoch_seconds"), (int, float))
        ):
            # Terminal historical failures and content-addressed cache hits
            # can lack attempt timing. Exclude them explicitly rather than
            # inventing a duration or silently treating them as zero-cost.
            unmeasured.append(str(call.get("call_id")))
            continue
        call["candidate_id"] = candidate
        if candidate not in candidates:
            candidates.append(candidate)
        call["duration_seconds"] = _duration(call)
        call["queued_at"] = _queued(call)
        rows.append(call)
    if unbound:
        raise E2ReplayError(
            "non-editorial calls could not be bound to candidates: " + ", ".join(unbound)
        )
    if not rows:
        raise E2ReplayError(
            "state contains no timed completed post-selection calls"
        )
    for row in rows:
        role = row.get("role")
        call_id = str(row.get("call_id", ""))
        if role == "concrete_planner":
            phase = "planning"
        elif role == "task_realizer":
            phase = "realization"
        elif role == "technical_verifier" and "ambiguity-repair" not in call_id or role == "targeted_repair" and "mechanical" in call_id:
            phase = "technical"
        else:
            phase = "blind_and_conditional"
        row["phase"] = phase
    return rows, sorted(candidates), sorted(unmeasured)


def _chains(calls: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in calls:
        result.setdefault(str(row["candidate_id"]), []).append(deepcopy(dict(row)))
    for values in result.values():
        values.sort(key=lambda row: (float(row["queued_at"]), str(row["call_id"])))
    return result


def _schedule_chains(
    chains: Mapping[str, Sequence[Mapping[str, Any]]], workers: int
) -> dict[str, Any]:
    if workers < 1:
        raise E2ReplayError("worker count must be positive")
    # Deeper candidate work wins over untouched upstream work.  This models
    # the stated E2 behavior: when a plan finishes, its realization may launch
    # immediately rather than sitting behind every not-yet-started plan.
    ready: list[tuple[int, float, str, int]] = []
    for candidate, values in chains.items():
        if values:
            first = values[0]
            heapq.heappush(ready, (0, float(first["queued_at"]), candidate, 0))
    running: list[tuple[float, int, str, int]] = []
    serial = 0
    now = 0.0
    timeline = []
    while ready or running:
        while ready and len(running) < workers:
            _depth_priority, _original_priority, candidate, index = heapq.heappop(
                ready
            )
            row = chains[candidate][index]
            start = now
            finish = start + float(row["duration_seconds"])
            serial += 1
            heapq.heappush(running, (finish, serial, candidate, index))
            timeline.append(
                {
                    "candidate_id": candidate,
                    "call_id": row["call_id"],
                    "phase": row["phase"],
                    "start_seconds": round(start, 6),
                    "finish_seconds": round(finish, 6),
                }
            )
        if not running:
            raise E2ReplayError("scheduler deadlocked")
        finish = running[0][0]
        now = finish
        completed = []
        while running and abs(running[0][0] - finish) < 1e-9:
            completed.append(heapq.heappop(running))
        for _end, _serial, candidate, index in completed:
            next_index = index + 1
            if next_index < len(chains[candidate]):
                next_row = chains[candidate][next_index]
                heapq.heappush(
                    ready,
                    (
                        -next_index,
                        float(next_row["queued_at"]),
                        candidate,
                        next_index,
                    ),
                )
    compute = sum(
        float(row["duration_seconds"])
        for values in chains.values()
        for row in values
    )
    return {
        "makespan_seconds": round(now, 6),
        "active_model_seconds": round(compute, 6),
        "effective_concurrency": round(compute / now, 6) if now else 0.0,
        "worker_utilization": round(compute / (workers * now), 6) if now else 0.0,
        "timeline": timeline,
    }


def _barrier_schedule(calls: Sequence[Mapping[str, Any]], workers: int) -> dict[str, Any]:
    offset = 0.0
    timeline = []
    phase_summaries = []
    for phase in PHASES:
        values = [row for row in calls if row["phase"] == phase]
        scheduled = _schedule_chains(_chains(values), workers) if values else {
            "makespan_seconds": 0.0,
            "active_model_seconds": 0.0,
            "effective_concurrency": 0.0,
            "worker_utilization": 0.0,
            "timeline": [],
        }
        for row in scheduled["timeline"]:
            row = deepcopy(row)
            row["start_seconds"] = round(row["start_seconds"] + offset, 6)
            row["finish_seconds"] = round(row["finish_seconds"] + offset, 6)
            timeline.append(row)
        phase_summaries.append(
            {
                "phase": phase,
                "call_count": len(values),
                "makespan_seconds": scheduled["makespan_seconds"],
                "active_model_seconds": scheduled["active_model_seconds"],
            }
        )
        offset += float(scheduled["makespan_seconds"])
    compute = sum(float(row["duration_seconds"]) for row in calls)
    return {
        "makespan_seconds": round(offset, 6),
        "active_model_seconds": round(compute, 6),
        "effective_concurrency": round(compute / offset, 6) if offset else 0.0,
        "worker_utilization": round(compute / (workers * offset), 6) if offset else 0.0,
        "phases": phase_summaries,
        "timeline": timeline,
    }


def replay_state(path: Path, workers: int) -> dict[str, Any]:
    state = read_json(path)
    calls, candidates, unmeasured = _calls(state)
    barrier = _barrier_schedule(calls, workers)
    streaming = _schedule_chains(_chains(calls), workers)
    measured_start = min(
        float((row.get("timing") or {}).get("slot_acquired_at_epoch_seconds"))
        for row in calls
    )
    measured_end = max(
        float((row.get("timing") or {}).get("completed_at_epoch_seconds"))
        for row in calls
    )
    improvement = (
        1 - streaming["makespan_seconds"] / barrier["makespan_seconds"]
        if barrier["makespan_seconds"]
        else 0.0
    )
    return {
        "state": {"path": str(path.resolve()), "sha256": _sha(path)},
        "candidate_count": len(candidates),
        "call_count": len(calls),
        "unmeasured_terminal_call_ids": unmeasured,
        "calls_by_phase": dict(sorted(Counter(row["phase"] for row in calls).items())),
        "calls_by_role": dict(sorted(Counter(str(row["role"]) for row in calls).items())),
        "transport_retry_calls": sum(int(row.get("retry_calls") or 0) for row in calls),
        "timing_has_retry_understatement": any(int(row.get("retry_calls") or 0) for row in calls),
        "measured_elapsed_seconds": round(measured_end - measured_start, 6),
        "barrier_replay": barrier,
        "candidate_streaming_replay": streaming,
        "estimated_wall_reduction_fraction": round(improvement, 6),
        "estimated_speedup": round(
            barrier["makespan_seconds"] / streaming["makespan_seconds"], 6
        )
        if streaming["makespan_seconds"]
        else None,
        "measured_calls_replayed_identically": True,
        "all_semantic_calls_timed": not unmeasured,
        "quality_dispositions_reused_not_recomputed": True,
    }


def build_report(states: Sequence[Path], workers: int, output: Path) -> dict[str, Any]:
    rows = [replay_state(path.resolve(), workers) for path in states]
    barrier = sum(row["barrier_replay"]["makespan_seconds"] for row in rows)
    streaming = sum(
        row["candidate_streaming_replay"]["makespan_seconds"] for row in rows
    )
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "completed": True,
        "worker_count": workers,
        "source_count": len(rows),
        "sources": rows,
        "aggregate": {
            "candidate_count": sum(row["candidate_count"] for row in rows),
            "call_count": sum(row["call_count"] for row in rows),
            "unmeasured_terminal_call_count": sum(
                len(row["unmeasured_terminal_call_ids"]) for row in rows
            ),
            "barrier_makespan_seconds": round(barrier, 6),
            "streaming_makespan_seconds": round(streaming, 6),
            "estimated_wall_reduction_fraction": round(1 - streaming / barrier, 6)
            if barrier
            else 0.0,
            "estimated_speedup": round(barrier / streaming, 6) if streaming else None,
            "retry_affected_source_count": sum(
                row["timing_has_retry_understatement"] for row in rows
            ),
        },
        "interpretation": {
            "replay_only": True,
            "no_model_calls": True,
            "no_prompt_or_disposition_change": True,
            "live_streaming_qualification_still_required": True,
            "conditional_budget_ordering_requires_deterministic_priority": True,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "report.json", report)
    lines = [
        "# E2 candidate-level streaming replay",
        "",
        f"- Sources: {len(rows)}",
        f"- Candidates: {report['aggregate']['candidate_count']}",
        f"- Timed recorded calls replayed exactly: {report['aggregate']['call_count']}",
        f"- Historical terminal calls without timing: {report['aggregate']['unmeasured_terminal_call_count']}",
        f"- Barrier replay: {barrier / 60:.2f} minutes",
        f"- Candidate streaming replay: {streaming / 60:.2f} minutes",
        f"- Estimated reduction: {report['aggregate']['estimated_wall_reduction_fraction']:.1%}",
        f"- Estimated speedup: {report['aggregate']['estimated_speedup']:.2f}x",
        "",
        "This is a no-call directional replay. A live crash/resume qualification remains required before adoption.",
        "",
    ]
    atomic_write_text(output / "report.md", "\n".join(lines))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        value = build_report(
            [Path(item) for item in args.state], args.workers, Path(args.output)
        )
        print(json.dumps(value, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
