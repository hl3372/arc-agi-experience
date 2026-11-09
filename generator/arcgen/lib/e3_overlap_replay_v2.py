"""Attempt-complete E3 replay over a fresh E2 v2 qualification.

The replay combines the exact selection and downstream calls from one run,
including retries and terminal failures, under a shared eight-slot scheduler.
It compares the current qualification order, a fully serial slate factory,
and cross-slate selection/development overlap.  No model call is made and no
semantic artifact is changed.
"""

from __future__ import annotations

import argparse
import heapq
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .e2_attempt_ledger_v2 import _read_attempts
from .util import atomic_write_json, atomic_write_text, read_json

PIPELINE = "arcgen-e3-selection-development-overlap-replay-v2"
VERSION = "2.0"


class E3ReplayV2Error(RuntimeError):
    """Raised when exact-call E3 timing cannot be reconstructed safely."""


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Node:
    node_id: str
    duration: float
    kind: str
    slate: int
    depth: int
    role: str
    provider: str
    source_call_id: str
    candidate_id: str | None = None


def _logical_duration(
    row: Mapping[str, Any], *, run_dir: Path, label: str
) -> tuple[float, dict[str, Any]]:
    attempts, errors = _read_attempts(row, run_dir=run_dir)
    if errors or not attempts:
        raise E3ReplayV2Error(
            f"{label} lacks attempt-complete timing: {errors or 'no attempts'}"
        )
    if str(row.get("status")) not in {"complete", "failed"}:
        raise E3ReplayV2Error(f"{label} is not terminal")
    first = float(attempts[0]["queued_at_epoch_seconds"])
    last = float(attempts[-1]["completed_at_epoch_seconds"])
    if last < first:
        raise E3ReplayV2Error(f"{label} has a negative attempt envelope")
    replay_service = 0.0
    observed_queue = 0.0
    for attempt in attempts:
        acquired = attempt.get("slot_acquired_at_epoch_seconds")
        completed = attempt.get("completed_at_epoch_seconds")
        if not isinstance(acquired, (int, float)) or not isinstance(
            completed, (int, float)
        ):
            raise E3ReplayV2Error(
                f"{label} has an attempt without measured slot service"
            )
        replay_service += float(completed) - float(acquired)
        observed_queue += float(attempt.get("queue_wait_seconds") or 0)
    return replay_service, {
        "logical_call_id": str(row.get("call_id") or label),
        "status": str(row.get("status")),
        "physical_attempt_count": len(attempts),
        "retry_attempt_count": max(0, len(attempts) - 1),
        "failed_attempt_count": sum(
            item.get("outcome") != "success" for item in attempts
        ),
        "logical_envelope_seconds": round(last - first, 6),
        "replay_slot_service_seconds": round(replay_service, 6),
        "original_model_slot_queue_wait_seconds": round(observed_queue, 6),
        "attempt_timing_sources": [
            {"path": item["timing_path"], "sha256": item["timing_sha256"]}
            for item in attempts
        ],
    }


def _selection_ledger(slate: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    ref = slate.get("selected_slate_manifest") or {}
    selected_path = Path(str(ref.get("path") or ""))
    if not selected_path.is_file() or _sha(selected_path) != ref.get("sha256"):
        raise E3ReplayV2Error("selected-slate manifest is missing or changed")
    selected = read_json(selected_path)
    refs = {
        str(item.get("source_id")): item
        for item in selected.get("source_artifacts", [])
        if isinstance(item, Mapping)
    }
    ledger_ref = refs.get("pre-authoring-call-ledger") or {}
    path = Path(str(ledger_ref.get("path") or ""))
    if not path.is_file() or _sha(path) != ledger_ref.get("sha256"):
        raise E3ReplayV2Error("pre-authoring call ledger is missing or changed")
    value = read_json(path)
    ledger = value.get("call_ledger")
    if not isinstance(ledger, Mapping):
        raise E3ReplayV2Error("pre-authoring artifact has no call ledger")
    return path, dict(ledger)


def _supply_slot(call_id: str) -> str | None:
    if "cycle-" not in call_id or not call_id.endswith("-supply"):
        return None
    suffix = call_id.split("cycle-", 1)[1]
    cycle, mode = suffix.split("-", 1)
    return f"{cycle}:{mode.removesuffix('-supply')}"


def _selection_nodes(
    selection: Mapping[str, Any], *, production_equivalent: bool
) -> tuple[
    dict[str, Node],
    dict[str, set[str]],
    dict[int, str],
    dict[int, list[str]],
    list[dict[str, Any]],
    dict[str, int],
]:
    nodes: dict[str, Node] = {}
    deps: dict[str, set[str]] = {}
    global_by_slate: dict[int, str] = {}
    supply_by_slate: dict[int, list[str]] = {}
    provenance = []
    totals: Counter[str] = Counter()
    prior_global: str | None = None
    for slate in sorted(selection.get("slates") or [], key=lambda row: int(row["slate_index"])):
        index = int(slate["slate_index"])
        ledger_path, ledger = _selection_ledger(slate)
        supply = [
            (str(key), row)
            for key, row in ledger.items()
            if isinstance(row, Mapping) and row.get("role") == "idea_search"
        ]
        selectors = [
            (str(key), row)
            for key, row in ledger.items()
            if isinstance(row, Mapping) and row.get("role") == "batch_selector"
        ]
        by_slot: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
        for key, row in supply:
            slot = _supply_slot(str(row.get("call_id") or key))
            if slot is None:
                raise E3ReplayV2Error(f"unrecognized supply call: {key}")
            by_slot[slot].append((key, row))
        kept: list[tuple[str, Mapping[str, Any]]] = []
        omitted: list[str] = []
        for slot, rows in sorted(by_slot.items()):
            if production_equivalent and len(rows) == 2:
                claude = [item for item in rows if item[1].get("provider") == "claude"]
                codex = [item for item in rows if item[1].get("provider") == "codex"]
                if len(claude) != 1 or len(codex) != 1:
                    raise E3ReplayV2Error(f"malformed paired supply slot: {slot}")
                kept.extend(claude)
                omitted.append(codex[0][0])
            else:
                kept.extend(rows)
        supply_ids = []
        call_evidence = []
        for key, row in kept:
            duration, evidence = _logical_duration(
                row, run_dir=ledger_path.parent, label=key
            )
            node_id = f"s{index}:supply:{sha256(key.encode()).hexdigest()[:12]}"
            provider = str(row.get("provider") or "codex")
            nodes[node_id] = Node(
                node_id,
                duration,
                "selection",
                index,
                0,
                "idea_search",
                provider,
                key,
            )
            deps[node_id] = {prior_global} if prior_global else set()
            supply_ids.append(node_id)
            call_evidence.append(evidence)
            totals["logical_calls"] += 1
            totals["physical_attempts"] += evidence["physical_attempt_count"]
            totals["retries"] += evidence["retry_attempt_count"]
            totals["failed_attempts"] += evidence["failed_attempt_count"]
        shards = [(key, row) for key, row in selectors if "shard-selector" in key]
        globals_ = [(key, row) for key, row in selectors if "global-selector" in key]
        if not shards or len(globals_) != 1:
            raise E3ReplayV2Error(f"slate {index} selector graph is incomplete")
        shard_ids = []
        for key, row in sorted(shards):
            duration, evidence = _logical_duration(
                row, run_dir=ledger_path.parent, label=key
            )
            node_id = f"s{index}:shard:{sha256(key.encode()).hexdigest()[:12]}"
            nodes[node_id] = Node(
                node_id,
                duration,
                "selection",
                index,
                1,
                "batch_selector_shard",
                "codex",
                key,
            )
            deps[node_id] = set(supply_ids)
            shard_ids.append(node_id)
            call_evidence.append(evidence)
            totals["logical_calls"] += 1
            totals["physical_attempts"] += evidence["physical_attempt_count"]
            totals["retries"] += evidence["retry_attempt_count"]
            totals["failed_attempts"] += evidence["failed_attempt_count"]
        key, row = globals_[0]
        duration, evidence = _logical_duration(
            row, run_dir=ledger_path.parent, label=key
        )
        global_id = f"s{index}:global:{sha256(key.encode()).hexdigest()[:12]}"
        nodes[global_id] = Node(
            global_id,
            duration,
            "selection",
            index,
            2,
            "batch_selector_global",
            "codex",
            key,
        )
        deps[global_id] = set(shard_ids)
        global_by_slate[index] = global_id
        supply_by_slate[index] = supply_ids
        prior_global = global_id
        call_evidence.append(evidence)
        totals["logical_calls"] += 1
        totals["physical_attempts"] += evidence["physical_attempt_count"]
        totals["retries"] += evidence["retry_attempt_count"]
        totals["failed_attempts"] += evidence["failed_attempt_count"]
        provenance.append(
            {
                "slate_index": index,
                "ledger": {"path": str(ledger_path.resolve()), "sha256": _sha(ledger_path)},
                "recorded_supply_calls": len(supply),
                "replayed_supply_calls": len(kept),
                "omitted_paired_sol_control_call_ids": omitted,
                "selector_calls": len(shards) + 1,
                "attempt_evidence": call_evidence,
            }
        )
    return (
        nodes,
        deps,
        global_by_slate,
        supply_by_slate,
        provenance,
        dict(totals),
    )


def _downstream_nodes(
    live: Mapping[str, Any], global_by_slate: Mapping[int, str]
) -> tuple[
    dict[str, Node],
    dict[str, set[str]],
    dict[int, list[str]],
    dict[int, list[str]],
    dict[str, int],
]:
    calls = (
        (live.get("scheduler") or {}).get("attempt_accounting") or {}
    ).get("logical_calls") or []
    if not calls or any(row.get("coverage_complete") is not True for row in calls):
        raise E3ReplayV2Error("downstream attempt ledger is incomplete")
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    totals: Counter[str] = Counter()
    for row in calls:
        candidate = row.get("candidate_id")
        if not isinstance(candidate, str) or not candidate:
            raise E3ReplayV2Error("downstream call lacks candidate binding")
        by_candidate[candidate].append(row)
        totals["logical_calls"] += 1
        totals["physical_attempts"] += int(row.get("physical_attempt_count") or 0)
        totals["retries"] += int(row.get("retry_attempt_count") or 0)
        totals["failed_attempts"] += int(row.get("failed_attempt_count") or 0)
    nodes: dict[str, Node] = {}
    deps: dict[str, set[str]] = {}
    first_by_slate: dict[int, list[str]] = defaultdict(list)
    terminal_by_slate: dict[int, list[str]] = defaultdict(list)
    for candidate, rows in sorted(by_candidate.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row.get("first_queued_at_epoch_seconds") or 0),
                str(row.get("call_id")),
            ),
        )
        slate = int(ordered[0].get("slate_index") or 0)
        if slate not in global_by_slate or any(
            int(row.get("slate_index") or 0) != slate for row in ordered
        ):
            raise E3ReplayV2Error(f"candidate {candidate} has inconsistent slate binding")
        prior: str | None = None
        for depth, row in enumerate(ordered):
            attempts = row.get("attempts") or []
            if not attempts:
                raise E3ReplayV2Error(
                    f"downstream call has no physical attempts: {row.get('call_id')}"
                )
            duration = 0.0
            for attempt in attempts:
                acquired = attempt.get("slot_acquired_at_epoch_seconds")
                completed = attempt.get("completed_at_epoch_seconds")
                if not isinstance(acquired, (int, float)) or not isinstance(
                    completed, (int, float)
                ):
                    raise E3ReplayV2Error(
                        f"downstream attempt lacks slot service: {row.get('call_id')}"
                    )
                duration += float(completed) - float(acquired)
            if duration < 0:
                raise E3ReplayV2Error(
                    f"downstream call has negative service: {row.get('call_id')}"
                )
            node_id = f"d:{candidate}:{depth:02d}"
            nodes[node_id] = Node(
                node_id,
                float(duration),
                "development",
                slate,
                depth,
                str(row.get("role")),
                "codex",
                str(row.get("call_id")),
                candidate,
            )
            deps[node_id] = {prior} if prior else {global_by_slate[slate]}
            if prior is None:
                first_by_slate[slate].append(node_id)
            prior = node_id
        assert prior is not None
        terminal_by_slate[slate].append(prior)
    return nodes, deps, dict(first_by_slate), dict(terminal_by_slate), dict(totals)


def _schedule(
    nodes: Mapping[str, Node],
    deps: Mapping[str, set[str]],
    *,
    workers: int,
    selection_cap: int,
    codex_selection_cap: int = 4,
    claude_selection_cap: int = 4,
) -> dict[str, Any]:
    if workers != 8 or not 1 <= selection_cap <= workers:
        raise E3ReplayV2Error("E3 v2 requires eight workers and a valid selection cap")
    completed: set[str] = set()
    dispatched: set[str] = set()
    running: list[tuple[float, int, str]] = []
    timeline = []
    now = 0.0
    serial = 0
    while len(completed) < len(nodes):
        ready = [
            node
            for node_id, node in nodes.items()
            if node_id not in completed
            and node_id not in dispatched
            and deps.get(node_id, set()) <= completed
        ]
        while ready and len(running) < workers:
            running_nodes = [nodes[node_id] for _finish, _serial, node_id in running]
            selection_running = sum(node.kind == "selection" for node in running_nodes)
            codex_selection_running = sum(
                node.kind == "selection" and node.provider == "codex"
                for node in running_nodes
            )
            claude_selection_running = sum(
                node.kind == "selection" and node.provider == "claude"
                for node in running_nodes
            )
            development_exists = any(node.kind == "development" for node in ready) or any(
                node.kind == "development" for node in running_nodes
            )
            eligible = []
            for node in ready:
                if node.kind == "selection":
                    if development_exists and selection_running >= selection_cap:
                        continue
                    if node.provider == "codex" and codex_selection_running >= codex_selection_cap:
                        continue
                    if node.provider == "claude" and claude_selection_running >= claude_selection_cap:
                        continue
                eligible.append(node)
            if not eligible:
                break
            selection_ready = [node for node in eligible if node.kind == "selection"]
            if selection_ready and (
                not development_exists or selection_running < selection_cap
            ):
                node = min(
                    selection_ready,
                    key=lambda item: (item.slate, item.depth, item.provider, item.node_id),
                )
            else:
                node = min(
                    eligible,
                    key=lambda item: (
                        0 if item.kind == "development" else 1,
                        -item.depth,
                        item.slate,
                        item.node_id,
                    ),
                )
            ready.remove(node)
            dispatched.add(node.node_id)
            serial += 1
            finish = now + node.duration
            heapq.heappush(running, (finish, serial, node.node_id))
            timeline.append(
                {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "provider": node.provider,
                    "slate_index": node.slate,
                    "role": node.role,
                    "candidate_id": node.candidate_id,
                    "start_seconds": round(now, 6),
                    "finish_seconds": round(finish, 6),
                }
            )
        if not running:
            unresolved = sorted(set(nodes) - completed)
            raise E3ReplayV2Error(f"combined replay deadlocked: {unresolved[:4]}")
        now = running[0][0]
        while running and abs(running[0][0] - now) < 1e-9:
            _finish, _serial, node_id = heapq.heappop(running)
            completed.add(node_id)
    active = sum(node.duration for node in nodes.values())
    return {
        "makespan_seconds_before_boards": round(now, 6),
        "active_call_envelope_seconds": round(active, 6),
        "effective_concurrency": round(active / now, 6) if now else 0.0,
        "worker_utilization": round(active / (workers * now), 6) if now else 0.0,
        "call_count": len(nodes),
        "calls_by_kind": dict(sorted(Counter(node.kind for node in nodes.values()).items())),
        "calls_by_provider": dict(sorted(Counter(node.provider for node in nodes.values()).items())),
        "timeline": timeline,
    }


def _board_tail(live_run: Path) -> dict[str, Any]:
    state_path = live_run / "state.json"
    state = read_json(state_path)
    ledger = state.get("call_ledger") or {}
    intervals = []
    evidence = []
    for key, row in ledger.items():
        if not isinstance(row, Mapping) or row.get("role") != "editorial_board":
            continue
        duration, logical = _logical_duration(row, run_dir=live_run, label=str(key))
        attempts, _ = _read_attempts(row, run_dir=live_run)
        intervals.append(
            (
                float(attempts[0]["queued_at_epoch_seconds"]),
                float(attempts[-1]["completed_at_epoch_seconds"]),
            )
        )
        evidence.append({**logical, "duration_seconds": round(duration, 6)})
    if not intervals:
        raise E3ReplayV2Error("completed E2 run has no timed editorial calls")
    start = min(item[0] for item in intervals)
    finish = max(item[1] for item in intervals)
    return {
        "wall_seconds": round(finish - start, 6),
        "logical_calls": len(evidence),
        "physical_attempts": sum(item["physical_attempt_count"] for item in evidence),
        "retry_attempts": sum(item["retry_attempt_count"] for item in evidence),
        "calls": evidence,
        "modeled_as_identical_post-development_tail": True,
    }


def _with_board_tail(result: Mapping[str, Any], board: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(result)
    value["editorial_tail_seconds"] = float(board["wall_seconds"])
    value["end_to_end_seconds"] = round(
        float(result["makespan_seconds_before_boards"]) + float(board["wall_seconds"]),
        6,
    )
    return value


def replay(
    *, selection_manifest: Path, live_run: Path, selection_caps: Sequence[int]
) -> dict[str, Any]:
    selection_path = selection_manifest.resolve()
    run = live_run.resolve()
    selection = read_json(selection_path)
    live = read_json(run / "report.json")
    if (
        selection.get("completed") is not True
        or selection.get("selection_only") is not True
        or live.get("pipeline") != "arcgen-e2-live-candidate-streaming-v2"
        or live.get("completed") is not True
    ):
        raise E3ReplayV2Error("E3 requires a completed selection cohort and E2 v2 run")
    if int((live.get("summary") or {}).get("selected") or 0) != int(
        (selection.get("funnel") or {}).get("selected_candidates") or 0
    ):
        raise E3ReplayV2Error("selection and downstream candidate counts differ")
    board = _board_tail(run)
    variants = {}
    for name, production_equivalent in (
        ("qualification_actual_15_call_supply", False),
        ("production_equivalent_12_call_supply", True),
    ):
        (
            selection_nodes,
            base_selection_deps,
            globals_by_slate,
            supply_by_slate,
            provenance,
            selection_totals,
        ) = _selection_nodes(selection, production_equivalent=production_equivalent)
        (
            development_nodes,
            base_development_deps,
            first_by_slate,
            terminal_by_slate,
            downstream_totals,
        ) = _downstream_nodes(live, globals_by_slate)
        nodes = {**selection_nodes, **development_nodes}
        slates = sorted(globals_by_slate)

        # The current E2 qualification freezes all selections, then executes
        # downstream slates independently and sequentially.
        qualification_deps = {key: set(value) for key, value in base_selection_deps.items()}
        qualification_deps.update(
            {key: set(value) for key, value in base_development_deps.items()}
        )
        final_global = globals_by_slate[slates[-1]]
        for slate in slates:
            for node_id in first_by_slate.get(slate, []):
                qualification_deps[node_id] = {final_global}
                if slate != slates[0]:
                    qualification_deps[node_id] |= set(
                        terminal_by_slate.get(slates[slates.index(slate) - 1], [])
                    )

        # Fully serial production baseline: select and finish one slate before
        # supplying/selecting the next.
        serial_deps = {key: set(value) for key, value in base_selection_deps.items()}
        serial_deps.update({key: set(value) for key, value in base_development_deps.items()})
        for position, slate in enumerate(slates[1:], start=1):
            previous = slates[position - 1]
            for node_id in supply_by_slate[slate]:
                serial_deps[node_id] = set(terminal_by_slate.get(previous, []))

        qualification = _with_board_tail(
            _schedule(nodes, qualification_deps, workers=8, selection_cap=8), board
        )
        serial = _with_board_tail(
            _schedule(nodes, serial_deps, workers=8, selection_cap=8), board
        )
        overlap = {}
        for cap in sorted({int(item) for item in selection_caps}):
            result = _with_board_tail(
                _schedule(nodes, {**base_selection_deps, **base_development_deps}, workers=8, selection_cap=cap),
                board,
            )
            result["selection_concurrency_cap_during_development"] = cap
            result["reduction_vs_current_qualification_fraction"] = round(
                1 - result["end_to_end_seconds"] / qualification["end_to_end_seconds"], 6
            )
            result["reduction_vs_fully_serial_slate_fraction"] = round(
                1 - result["end_to_end_seconds"] / serial["end_to_end_seconds"], 6
            )
            overlap[str(cap)] = result
        variants[name] = {
            "production_equivalent_supply": production_equivalent,
            "selection_provenance": provenance,
            "attempt_accounting": {
                "selection": selection_totals,
                "downstream": downstream_totals,
                "editorial": {
                    "logical_calls": board["logical_calls"],
                    "physical_attempts": board["physical_attempts"],
                    "retries": board["retry_attempts"],
                },
            },
            "current_e2_qualification_order": qualification,
            "fully_serial_slate_factory": serial,
            "overlap_scenarios": overlap,
        }
    return {
        "schema_version": 2,
        "pipeline": PIPELINE,
        "version": VERSION,
        "completed": True,
        "workers": 8,
        "selection_manifest": {"path": str(selection_path), "sha256": _sha(selection_path)},
        "live_e2_report": {"path": str(run / 'report.json'), "sha256": _sha(run / 'report.json')},
        "editorial_tail": board,
        "variants": variants,
        "interpretation": {
            "no_model_calls": True,
            "all_attempts_including_retries_used": True,
            "terminal_failed_calls_retained": True,
            "original_queue_wait_removed_before_rescheduling": True,
            "logical_retries_modeled_as_contiguous_attempt_service": True,
            "shared_eight_worker_limit_modeled": True,
            "provider_local_selection_caps_modeled": True,
            "boards_retained_as_identical_tail": True,
            "live_e3_qualification_still_required": True,
            "automatic_promotion_or_commit": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--live-run", required=True)
    parser.add_argument("--selection-cap", action="append", type=int, default=[])
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = replay(
            selection_manifest=Path(args.selection_manifest),
            live_run=Path(args.live_run),
            selection_caps=args.selection_cap or [4, 5, 6, 7, 8],
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / "report.json", value)
        rows = ["# E3 v2 attempt-complete overlap replay", ""]
        production = value["variants"]["production_equivalent_12_call_supply"]
        for cap, result in production["overlap_scenarios"].items():
            rows.append(
                f"- Shared selection cap {cap}: {result['reduction_vs_current_qualification_fraction']:.1%} vs current E2 order; "
                f"{result['reduction_vs_fully_serial_slate_fraction']:.1%} vs a serial slate factory."
            )
        rows.extend(
            [
                "",
                "Every retry and terminal failed attempt is retained. Boards are included as the same measured tail in all scenarios.",
                "",
                "This is a zero-call replay, not E3 adoption evidence. No task was committed and no production promotion occurred.",
                "",
            ]
        )
        atomic_write_text(output / "report.md", "\n".join(rows))
        print(output / "report.json")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI fails closed
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["E3ReplayV2Error", "Node", "_schedule", "replay"]
