"""Persistent bounded idea reservoir, tiered memory, and recoverable leases.

The reservoir is operational state, not a programmatic task generator.  It
stores model-authored kernels and controls when they may be compared or
developed.  Exact hashes prevent accidental literal reuse; semantic duplicate
decisions remain the selectors' responsibility.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping, Sequence
import fcntl
import time

from .util import atomic_write_json, content_hash, read_json


RESERVOIR_SCHEMA_VERSION = 1
RESERVOIR_ID = "arcgen-v4-persistent-idea-reservoir"
PROPOSAL_STATES = {"proposed", "in_flight", "accepted", "rejected"}
MEMORY_TIERS = {
    "hard_duplicate",
    "soft_mechanism",
    "failure_mode",
    "recoverable_work",
}
TERMINAL_RESERVATION_OUTCOMES = {
    "accepted",
    "rejected",
    "aborted",
    "deferred",
    "recovered_expired",
    "recovered_interrupted",
}


class V4ReservoirError(ValueError):
    pass


def normalize_concept(text: str) -> str:
    return " ".join(str(text).split())


def concept_hash(text: str) -> str:
    return content_hash(normalize_concept(text).casefold())


def _empty() -> dict[str, Any]:
    return {
        "schema_version": RESERVOIR_SCHEMA_VERSION,
        "reservoir_id": RESERVOIR_ID,
        "proposals": [],
        "active_reservations": [],
        "reservation_history": [],
        "rejection_memory": [],
        "frozen_windows": [],
        "supply_history": [],
        "lifecycle_events": [],
    }


def _validate_proposal(item: Any, path: str) -> None:
    if not isinstance(item, dict):
        raise V4ReservoirError(f"{path} must be an object")
    required = {
        "proposal_id", "kernel", "search_mode", "state", "concept_hash",
        "created_at_epoch_seconds", "supply_run_id", "supply_cycle_id",
        "calibration_task_ids", "calibration_descriptions",
        "calibration_version", "prompt_version", "prompt_hash", "model",
        "reasoning_effort", "source_bank_snapshot_id", "embedding",
        "novelty_status", "staleness", "lifecycle_history",
        "selection_attempts", "final_disposition",
    }
    missing = required - set(item)
    if missing:
        raise V4ReservoirError(f"{path} is missing {sorted(missing)}")
    if item["state"] not in PROPOSAL_STATES:
        raise V4ReservoirError(f"{path}.state is invalid")
    kernel = item["kernel"]
    if not isinstance(kernel, dict) or kernel.get("candidate_id") != item["proposal_id"]:
        raise V4ReservoirError(f"{path}.kernel has the wrong candidate ID")
    relation = kernel.get("central_relation")
    if not isinstance(relation, str) or not relation.strip() or item["concept_hash"] != concept_hash(relation):
        raise V4ReservoirError(f"{path}.concept_hash is invalid")
    if item["search_mode"] not in {"open", "concept_led", "composition_led"}:
        raise V4ReservoirError(f"{path}.search_mode is invalid")
    if kernel.get("search_mode") != item["search_mode"]:
        raise V4ReservoirError(f"{path}.kernel changed search mode")
    if not isinstance(item["lifecycle_history"], list) or not item["lifecycle_history"]:
        raise V4ReservoirError(f"{path}.lifecycle_history must be nonempty")
    if not isinstance(item["selection_attempts"], list):
        raise V4ReservoirError(f"{path}.selection_attempts must be an array")


def _validate_reservation(item: Any, path: str, *, active: bool) -> None:
    if not isinstance(item, dict):
        raise V4ReservoirError(f"{path} must be an object")
    required = {
        "reservation_id", "proposal_id", "concept_hash", "mechanism_summary",
        "owner_run_id", "owner_slate_id", "owner_candidate_id",
        "created_at_epoch_seconds", "heartbeat_at_epoch_seconds",
        "lease_seconds",
    }
    if not active:
        required |= {
            "resolved_at_epoch_seconds", "outcome", "resolution_note",
        }
    if set(item) != required:
        raise V4ReservoirError(f"{path} has the wrong reservation fields")
    if not active and item["outcome"] not in TERMINAL_RESERVATION_OUTCOMES:
        raise V4ReservoirError(f"{path}.outcome is invalid")


def _validate_memory(item: Any, path: str) -> None:
    required = {
        "memory_id", "tier", "proposal_id", "concept_hash",
        "mechanism_summary", "reason", "source_run_id", "created_at_epoch_seconds",
        "active",
    }
    if not isinstance(item, dict) or set(item) != required:
        raise V4ReservoirError(f"{path} has the wrong fields")
    if item["tier"] not in MEMORY_TIERS:
        raise V4ReservoirError(f"{path}.tier is invalid")


def validate_reservoir(value: Any, *, path: str = "reservoir") -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != RESERVOIR_SCHEMA_VERSION
        or value.get("reservoir_id") != RESERVOIR_ID
    ):
        raise V4ReservoirError(f"{path} has an unsupported schema")
    for field in (
        "proposals", "active_reservations", "reservation_history",
        "rejection_memory", "frozen_windows", "supply_history", "lifecycle_events",
    ):
        if not isinstance(value.get(field), list):
            raise V4ReservoirError(f"{path}.{field} must be an array")
    proposal_ids: set[str] = set()
    for index, item in enumerate(value["proposals"]):
        _validate_proposal(item, f"{path}.proposals[{index}]")
        if item["proposal_id"] in proposal_ids:
            raise V4ReservoirError(f"{path} contains duplicate proposal IDs")
        proposal_ids.add(item["proposal_id"])
    reservation_ids: set[str] = set()
    reserved_proposals: set[str] = set()
    for index, item in enumerate(value["active_reservations"]):
        _validate_reservation(item, f"{path}.active_reservations[{index}]", active=True)
        if item["reservation_id"] in reservation_ids or item["proposal_id"] in reserved_proposals:
            raise V4ReservoirError(f"{path} contains duplicate active reservations")
        if item["proposal_id"] not in proposal_ids:
            raise V4ReservoirError(f"{path} reserves an unknown proposal")
        reservation_ids.add(item["reservation_id"])
        reserved_proposals.add(item["proposal_id"])
    for index, item in enumerate(value["reservation_history"]):
        _validate_reservation(item, f"{path}.reservation_history[{index}]", active=False)
    for index, item in enumerate(value["rejection_memory"]):
        _validate_memory(item, f"{path}.rejection_memory[{index}]")
    for item in value["proposals"]:
        if (item["state"] == "in_flight") != (item["proposal_id"] in reserved_proposals):
            raise V4ReservoirError(
                f"{path} proposal {item['proposal_id']} in-flight/reservation state disagrees"
            )
    return value


def initialize_reservoir(path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        return load_reservoir(destination)
    value = _empty()
    atomic_write_json(destination, value)
    return value


def load_reservoir(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return validate_reservoir(read_json(source), path=str(source))


def _locked_update(path: str | Path, mutator: Any) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = initialize_reservoir(destination)
        mutator(state)
        state["updated_at_epoch_seconds"] = round(time.time(), 3)
        validate_reservoir(state)
        atomic_write_json(destination, state)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return state


def active_count(value: Mapping[str, Any]) -> int:
    return sum(item["state"] in {"proposed", "in_flight"} for item in value["proposals"])


def reservoir_status(value: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    states = Counter(item["state"] for item in value["proposals"])
    tiers = Counter(item["tier"] for item in value["rejection_memory"] if item.get("active", True))
    active = active_count(value)
    low = int(config["low_watermark"])
    high = int(config["high_watermark"])
    hard = int(config["hard_capacity"])
    return {
        "proposal_state_counts": dict(sorted(states.items())),
        "active_capacity_used": active,
        "active_capacity_remaining": max(0, hard - active),
        "low_watermark": low,
        "high_watermark": high,
        "hard_capacity": hard,
        "routine_replenishment": (
            "trigger" if active < low else ("pause" if active >= high else "idle")
        ),
        "active_reservation_count": len(value["active_reservations"]),
        "active_rejection_memory_by_tier": dict(sorted(tiers.items())),
        "frozen_window_count": len(value["frozen_windows"]),
    }


def replenishment_decision(value: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    return str(reservoir_status(value, config)["routine_replenishment"])


def make_proposal(
    kernel: Mapping[str, Any],
    *,
    supply_run_id: str,
    supply_cycle_id: str,
    calibration_task_ids: Sequence[str],
    calibration_descriptions: Sequence[str],
    calibration_version: str,
    prompt_version: str,
    prompt_hash: str,
    model: str,
    reasoning_effort: str,
    source_bank_snapshot_id: str,
    embedding: Mapping[str, Any],
    created_at_epoch_seconds: float | None = None,
) -> dict[str, Any]:
    proposal_id = str(kernel["candidate_id"])
    created = round(created_at_epoch_seconds if created_at_epoch_seconds is not None else time.time(), 3)
    relation = str(kernel["central_relation"])
    return {
        "proposal_id": proposal_id,
        "kernel": deepcopy(dict(kernel)),
        "search_mode": str(kernel["search_mode"]),
        "state": "proposed",
        "concept_hash": concept_hash(relation),
        "created_at_epoch_seconds": created,
        "supply_run_id": supply_run_id,
        "supply_cycle_id": supply_cycle_id,
        "calibration_task_ids": list(calibration_task_ids),
        "calibration_descriptions": list(calibration_descriptions),
        "calibration_version": calibration_version,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "source_bank_snapshot_id": source_bank_snapshot_id,
        "embedding": deepcopy(dict(embedding)),
        "novelty_status": {
            "status": "pending_latest_selector_review",
            "checked_at_epoch_seconds": None,
            "bank_snapshot_id": source_bank_snapshot_id,
            "reservation_snapshot_id": None,
            "reason": "new proposal awaits a frozen-window comparison",
        },
        "staleness": {"status": "current", "reasons": [], "checked_at_epoch_seconds": created},
        "lifecycle_history": [
            {"from": None, "to": "proposed", "at_epoch_seconds": created, "reason": "model-authored reservoir proposal"}
        ],
        "selection_attempts": [],
        "final_disposition": None,
    }


def _memory(
    proposal: Mapping[str, Any],
    *,
    tier: str,
    reason: str,
    source_run_id: str,
    now: float,
    active: bool = True,
) -> dict[str, Any]:
    if tier not in MEMORY_TIERS:
        raise V4ReservoirError(f"unknown rejection-memory tier {tier}")
    return {
        "memory_id": f"memory:{proposal['proposal_id']}:{tier}:{content_hash(reason)[:12]}",
        "tier": tier,
        "proposal_id": proposal["proposal_id"],
        "concept_hash": proposal["concept_hash"],
        "mechanism_summary": normalize_concept(proposal["kernel"]["central_relation"])[:700],
        "reason": normalize_concept(reason),
        "source_run_id": source_run_id,
        "created_at_epoch_seconds": round(now, 3),
        "active": active,
    }


def append_supply(
    path: str | Path,
    proposals: Sequence[Mapping[str, Any]],
    *,
    hard_capacity: int,
    supply_run_id: str,
    supply_cycle_id: str,
) -> dict[str, Any]:
    now = time.time()

    def mutate(state: dict[str, Any]) -> None:
        by_id = {item["proposal_id"]: item for item in state["proposals"]}
        occupied_hashes = {
            item["concept_hash"]: item["proposal_id"]
            for item in state["proposals"]
            if item["state"] in {"proposed", "in_flight", "accepted"}
        }
        active = active_count(state)
        added: list[str] = []
        exact_duplicates: list[str] = []
        for raw in proposals:
            item = deepcopy(dict(raw))
            _validate_proposal(item, "new_proposal")
            proposal_id = item["proposal_id"]
            if proposal_id in by_id:
                if by_id[proposal_id] != item:
                    raise V4ReservoirError(f"proposal ID changed: {proposal_id}")
                continue
            conflict = occupied_hashes.get(item["concept_hash"])
            if conflict is not None:
                item["state"] = "rejected"
                item["final_disposition"] = {
                    "outcome": "rejected",
                    "memory_tier": "hard_duplicate",
                    "reason": f"exact normalized relation duplicates {conflict}",
                    "at_epoch_seconds": round(now, 3),
                }
                item["lifecycle_history"].append(
                    {"from": "proposed", "to": "rejected", "at_epoch_seconds": round(now, 3), "reason": item["final_disposition"]["reason"]}
                )
                state["rejection_memory"].append(
                    _memory(item, tier="hard_duplicate", reason=item["final_disposition"]["reason"], source_run_id=supply_run_id, now=now)
                )
                exact_duplicates.append(proposal_id)
            else:
                if active >= hard_capacity:
                    raise V4ReservoirError("reservoir hard active-capacity would be exceeded")
                active += 1
                occupied_hashes[item["concept_hash"]] = proposal_id
                added.append(proposal_id)
            state["proposals"].append(item)
            by_id[proposal_id] = item
        state["proposals"].sort(key=lambda item: (item["created_at_epoch_seconds"], item["proposal_id"]))
        event = {
            "supply_run_id": supply_run_id,
            "supply_cycle_id": supply_cycle_id,
            "at_epoch_seconds": round(now, 3),
            "proposal_ids_added": added,
            "exact_duplicates_rejected": exact_duplicates,
        }
        if not any(
            row.get("supply_run_id") == supply_run_id and row.get("supply_cycle_id") == supply_cycle_id
            for row in state["supply_history"]
        ):
            state["supply_history"].append(event)

    return _locked_update(path, mutate)


def _transition(
    proposal: dict[str, Any],
    target: str,
    *,
    reason: str,
    now: float,
    allow_recovery: bool = False,
    allow_explicit_accept: bool = False,
) -> None:
    source = proposal["state"]
    allowed = {
        "proposed": {"in_flight", "rejected"} | ({"accepted"} if allow_explicit_accept else set()),
        "in_flight": {"accepted", "rejected"} | ({"proposed"} if allow_recovery else set()),
        "rejected": {"proposed"} if allow_recovery else set(),
        "accepted": set(),
    }
    if target not in allowed[source]:
        raise V4ReservoirError(f"invalid proposal transition {source} -> {target}")
    proposal["state"] = target
    proposal["lifecycle_history"].append(
        {"from": source, "to": target, "at_epoch_seconds": round(now, 3), "reason": normalize_concept(reason)}
    )


def _current_reservation_snapshot(state: Mapping[str, Any]) -> str:
    return content_hash(
        [
            {
                "reservation_id": item["reservation_id"],
                "concept_hash": item["concept_hash"],
                "mechanism_summary": item["mechanism_summary"],
            }
            for item in sorted(state["active_reservations"], key=lambda row: row["reservation_id"])
        ]
    )


def freeze_eligible_window(
    path: str | Path,
    *,
    window_id: str,
    window_size: int,
    shard_size: int,
    seed: int,
    current_bank_snapshot_id: str,
    current_prompt_hashes: Mapping[str, str],
    calibration_version: str,
    accepted_concept_hashes: Iterable[str],
) -> dict[str, Any]:
    frozen: dict[str, Any] = {}
    now = time.time()
    accepted_hashes = set(accepted_concept_hashes)

    def mutate(state: dict[str, Any]) -> None:
        nonlocal frozen
        existing = next((item for item in state["frozen_windows"] if item["window_id"] == window_id), None)
        if existing is not None:
            frozen = deepcopy(existing)
            return
        active_reservations = {item["concept_hash"]: item for item in state["active_reservations"]}
        eligible: list[dict[str, Any]] = []
        for proposal in state["proposals"]:
            if proposal["state"] != "proposed":
                continue
            reasons = []
            if proposal["source_bank_snapshot_id"] != current_bank_snapshot_id:
                reasons.append("accepted task-bank snapshot changed")
            if proposal["prompt_hash"] != current_prompt_hashes.get(proposal["search_mode"]):
                reasons.append("idea prompt changed")
            if proposal["calibration_version"] != calibration_version:
                reasons.append("calibration configuration changed")
            proposal["staleness"] = {
                "status": "stale" if reasons else "current",
                "reasons": reasons,
                "checked_at_epoch_seconds": round(now, 3),
            }
            conflict_reason = None
            if proposal["concept_hash"] in accepted_hashes:
                conflict_reason = "exact normalized mechanism already exists in the latest accepted bank"
            elif proposal["concept_hash"] in active_reservations:
                conflict_reason = (
                    "exact normalized mechanism is reserved by "
                    + active_reservations[proposal["concept_hash"]]["reservation_id"]
                )
            if conflict_reason:
                _transition(proposal, "rejected", reason=conflict_reason, now=now)
                proposal["final_disposition"] = {
                    "outcome": "rejected", "memory_tier": "hard_duplicate",
                    "reason": conflict_reason, "at_epoch_seconds": round(now, 3),
                }
                state["rejection_memory"].append(
                    _memory(proposal, tier="hard_duplicate", reason=conflict_reason, source_run_id=window_id, now=now)
                )
                continue
            if reasons:
                proposal["staleness"] = {
                    "status": "revalidated",
                    "reasons": reasons,
                    "checked_at_epoch_seconds": round(now, 3),
                    "disposition": "eligible after latest exact-bank/reservation recheck; semantic comparison remains mandatory in selectors",
                }
                proposal["lifecycle_history"].append(
                    {"from": "proposed", "to": "proposed", "at_epoch_seconds": round(now, 3), "reason": "stale proposal revalidated against latest exact bank and reservations"}
                )
            proposal["novelty_status"] = {
                "status": "pending_frozen_window_selector_review",
                "checked_at_epoch_seconds": round(now, 3),
                "bank_snapshot_id": current_bank_snapshot_id,
                "reservation_snapshot_id": _current_reservation_snapshot(state),
                "reason": "exact conflicts cleared; semantic retrieval and selector review required",
            }
            eligible.append(proposal)
        if len(eligible) < window_size:
            raise V4ReservoirError(
                f"reservoir has {len(eligible)} eligible proposals; {window_size} are required"
            )

        # Complete balanced supply cycles are preferred as a supply-stability
        # policy.  This does not constrain what selectors may develop.
        cycles: dict[str, list[dict[str, Any]]] = {}
        for item in eligible:
            cycles.setdefault(item["supply_cycle_id"], []).append(item)
        selected: list[dict[str, Any]] = []
        for cycle_id in sorted(
            cycles,
            key=lambda key: (
                min(item["created_at_epoch_seconds"] for item in cycles[key]), key
            ),
        ):
            rows = sorted(cycles[cycle_id], key=lambda item: item["proposal_id"])
            counts = Counter(item["search_mode"] for item in rows)
            if len(rows) == 12 and counts == Counter({"open": 4, "concept_led": 4, "composition_led": 4}) and len(selected) + 12 <= window_size:
                selected.extend(rows)
            if len(selected) == window_size:
                break
        if len(selected) < window_size:
            remainder = [item for item in eligible if item["proposal_id"] not in {row["proposal_id"] for row in selected}]
            Random(seed).shuffle(remainder)
            selected.extend(remainder[: window_size - len(selected)])
        ids = [item["proposal_id"] for item in selected]
        shards = [ids[index:index + shard_size] for index in range(0, len(ids), shard_size)]
        frozen = {
            "window_id": window_id,
            "created_at_epoch_seconds": round(now, 3),
            "seed": seed,
            "window_size": window_size,
            "shard_size": shard_size,
            "proposal_ids": ids,
            "shards": shards,
            "accepted_bank_snapshot_id": current_bank_snapshot_id,
            "reservation_snapshot_id": _current_reservation_snapshot(state),
            "rejection_memory_snapshot_id": content_hash(state["rejection_memory"]),
            "reservoir_snapshot_id": content_hash(
                {
                    "proposals": [(item["proposal_id"], item["state"], item["concept_hash"]) for item in state["proposals"]],
                    "reservations": state["active_reservations"],
                    "memory": state["rejection_memory"],
                }
            ),
        }
        state["frozen_windows"].append(deepcopy(frozen))
        for proposal in selected:
            proposal["selection_attempts"].append(
                {"window_id": window_id, "stage": "eligible_window", "decision": "included", "reason": "deterministic frozen eligible window", "at_epoch_seconds": round(now, 3)}
            )

    _locked_update(path, mutate)
    return frozen


def reserve_slate(
    path: str | Path,
    *,
    window_id: str,
    owner_run_id: str,
    slate_id: str,
    selections: Sequence[Mapping[str, Any]],
    lease_seconds: int,
    now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else now
    result: dict[str, Any] = {}

    def mutate(state: dict[str, Any]) -> None:
        nonlocal result
        window = next((item for item in state["frozen_windows"] if item["window_id"] == window_id), None)
        if window is None:
            raise V4ReservoirError(f"unknown frozen window {window_id}")
        by_id = {item["proposal_id"]: item for item in state["proposals"]}
        requested = [str(item["proposal_id"]) for item in selections]
        if len(requested) != len(set(requested)) or not set(requested) <= set(window["proposal_ids"]):
            raise V4ReservoirError("selected slate must be a unique subset of the frozen window")
        existing_for_slate = [
            item for item in state["active_reservations"]
            if item["owner_run_id"] == owner_run_id and item["owner_slate_id"] == slate_id
        ]
        if existing_for_slate:
            if {item["proposal_id"] for item in existing_for_slate} != set(requested):
                raise V4ReservoirError("reserved slate changed during resume")
            result = {"reservations": deepcopy(existing_for_slate), "reserved_atomically": True, "resumed": True}
            return
        occupied = {item["concept_hash"]: item["reservation_id"] for item in state["active_reservations"]}
        hashes: set[str] = set()
        for proposal_id in requested:
            proposal = by_id.get(proposal_id)
            if proposal is None or proposal["state"] != "proposed":
                raise V4ReservoirError(f"proposal is no longer available: {proposal_id}")
            if proposal["concept_hash"] in occupied:
                raise V4ReservoirError(f"proposal conflicts with {occupied[proposal['concept_hash']]}")
            if proposal["concept_hash"] in hashes:
                raise V4ReservoirError("selected slate contains exact mechanism siblings")
            hashes.add(proposal["concept_hash"])
        reservations = []
        for selection in selections:
            proposal = by_id[str(selection["proposal_id"])]
            _transition(proposal, "in_flight", reason=f"selected and atomically reserved by {slate_id}", now=timestamp)
            reservation = {
                "reservation_id": f"reservation:{owner_run_id}:{slate_id}:{proposal['proposal_id']}",
                "proposal_id": proposal["proposal_id"],
                "concept_hash": proposal["concept_hash"],
                "mechanism_summary": normalize_concept(str(selection.get("mechanism_summary") or proposal["kernel"]["central_relation"]))[:700],
                "owner_run_id": owner_run_id,
                "owner_slate_id": slate_id,
                "owner_candidate_id": str(selection.get("candidate_id") or proposal["proposal_id"]),
                "created_at_epoch_seconds": round(timestamp, 3),
                "heartbeat_at_epoch_seconds": round(timestamp, 3),
                "lease_seconds": lease_seconds,
            }
            state["active_reservations"].append(reservation)
            reservations.append(reservation)
        state["active_reservations"].sort(key=lambda item: item["reservation_id"])
        state["lifecycle_events"].append(
            {"event": "whole_slate_reserved", "owner_run_id": owner_run_id, "slate_id": slate_id, "proposal_ids": requested, "at_epoch_seconds": round(timestamp, 3)}
        )
        result = {"reservations": deepcopy(reservations), "reserved_atomically": True, "resumed": False}

    _locked_update(path, mutate)
    return result


def recover_interrupted_slate(
    path: str | Path,
    *,
    window_id: str,
    owner_run_id: str,
    slate_id: str,
    selections: Sequence[Mapping[str, Any]],
    lease_seconds: int,
    reason: str,
    now: float | None = None,
) -> dict[str, Any]:
    """Re-lease the same frozen slate after an explicitly audited interruption.

    Only recoverable operational dispositions may be reopened.  Semantic
    rejections and accepted proposals are never eligible.  The operation is
    atomic, retires old leases into history, and reserves the complete slate
    before any candidate can resume.
    """

    timestamp = time.time() if now is None else now
    explanation = normalize_concept(reason)
    if not explanation:
        raise V4ReservoirError("interrupted-slate recovery requires a reason")
    result: dict[str, Any] = {}

    def mutate(state: dict[str, Any]) -> None:
        nonlocal result
        window = next(
            (item for item in state["frozen_windows"] if item["window_id"] == window_id),
            None,
        )
        if window is None:
            raise V4ReservoirError(f"unknown frozen window {window_id}")
        requested = [str(item["proposal_id"]) for item in selections]
        if len(requested) != len(set(requested)) or not set(requested) <= set(window["proposal_ids"]):
            raise V4ReservoirError("recovered slate must be a unique subset of its frozen window")
        by_id = {item["proposal_id"]: item for item in state["proposals"]}
        requested_set = set(requested)

        foreign = [
            item for item in state["active_reservations"]
            if item["proposal_id"] in requested_set and item["owner_run_id"] != owner_run_id
        ]
        if foreign:
            raise V4ReservoirError("interrupted recovery cannot preempt another run's lease")

        for proposal_id in requested:
            proposal = by_id[proposal_id]
            if proposal["state"] == "accepted":
                raise V4ReservoirError(f"accepted proposal cannot be recovered: {proposal_id}")
            if proposal["state"] == "rejected":
                final = proposal.get("final_disposition") or {}
                if final.get("memory_tier") != "recoverable_work":
                    raise V4ReservoirError(
                        f"semantic rejection cannot be recovered as interrupted work: {proposal_id}"
                    )

        # Retire every surviving lease for this slate before atomically
        # constructing its replacement.  No candidate resumes in between.
        for proposal_id in requested:
            proposal = by_id[proposal_id]
            reservation = next(
                (item for item in state["active_reservations"] if item["proposal_id"] == proposal_id),
                None,
            )
            if reservation is not None:
                _resolve_active_reservation(
                    state, proposal_id, outcome="recovered_interrupted",
                    note=explanation, now=timestamp,
                )
            if proposal["state"] == "in_flight":
                _transition(
                    proposal, "proposed", reason=explanation,
                    now=timestamp, allow_recovery=True,
                )
            elif proposal["state"] == "rejected":
                _transition(
                    proposal, "proposed", reason=explanation,
                    now=timestamp, allow_recovery=True,
                )
            proposal["final_disposition"] = None
            for memory in state["rejection_memory"]:
                if (
                    memory["proposal_id"] == proposal_id
                    and memory["tier"] == "recoverable_work"
                    and memory["source_run_id"] == owner_run_id
                    and memory["active"]
                ):
                    memory["active"] = False

        occupied = {
            item["concept_hash"]: item["reservation_id"]
            for item in state["active_reservations"]
        }
        hashes: set[str] = set()
        reservations = []
        for selection in selections:
            proposal = by_id[str(selection["proposal_id"])]
            if proposal["concept_hash"] in occupied or proposal["concept_hash"] in hashes:
                raise V4ReservoirError("recovered slate conflicts with another active mechanism")
            hashes.add(proposal["concept_hash"])
            _transition(
                proposal, "in_flight",
                reason=f"explicit interrupted-run recovery for {slate_id}",
                now=timestamp,
            )
            generation = 1 + sum(
                item["proposal_id"] == proposal["proposal_id"]
                for item in state["reservation_history"]
            )
            reservation = {
                "reservation_id": (
                    f"reservation:{owner_run_id}:{slate_id}:{proposal['proposal_id']}:recovery-{generation}"
                ),
                "proposal_id": proposal["proposal_id"],
                "concept_hash": proposal["concept_hash"],
                "mechanism_summary": normalize_concept(
                    str(selection.get("mechanism_summary") or proposal["kernel"]["central_relation"])
                )[:700],
                "owner_run_id": owner_run_id,
                "owner_slate_id": slate_id,
                "owner_candidate_id": str(
                    selection.get("candidate_id") or proposal["proposal_id"]
                ),
                "created_at_epoch_seconds": round(timestamp, 3),
                "heartbeat_at_epoch_seconds": round(timestamp, 3),
                "lease_seconds": lease_seconds,
            }
            state["active_reservations"].append(reservation)
            reservations.append(reservation)
        state["active_reservations"].sort(key=lambda item: item["reservation_id"])
        state["lifecycle_events"].append(
            {
                "event": "whole_slate_recovered_after_interruption",
                "owner_run_id": owner_run_id,
                "slate_id": slate_id,
                "proposal_ids": requested,
                "reason": explanation,
                "at_epoch_seconds": round(timestamp, 3),
            }
        )
        result = {
            "reservations": deepcopy(reservations),
            "reserved_atomically": True,
            "recovered_after_interruption": True,
        }

    _locked_update(path, mutate)
    return result


def heartbeat_leases(path: str | Path, *, owner_run_id: str, now: float | None = None) -> dict[str, Any]:
    timestamp = time.time() if now is None else now

    def mutate(state: dict[str, Any]) -> None:
        for item in state["active_reservations"]:
            if item["owner_run_id"] == owner_run_id:
                item["heartbeat_at_epoch_seconds"] = round(timestamp, 3)

    return _locked_update(path, mutate)


def _resolve_active_reservation(
    state: dict[str, Any], proposal_id: str, *, outcome: str, note: str, now: float
) -> dict[str, Any] | None:
    reservation = next((item for item in state["active_reservations"] if item["proposal_id"] == proposal_id), None)
    if reservation is None:
        return None
    state["active_reservations"] = [item for item in state["active_reservations"] if item["proposal_id"] != proposal_id]
    history = {
        **reservation,
        "resolved_at_epoch_seconds": round(now, 3),
        "outcome": outcome,
        "resolution_note": normalize_concept(note),
    }
    state["reservation_history"].append(history)
    return history


def resolve_proposal(
    path: str | Path,
    *,
    proposal_id: str,
    outcome: str,
    reason: str,
    source_run_id: str,
    memory_tier: str | None = None,
    committed_registry_task_id: str | None = None,
    explicit_commit_verified: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else now

    def mutate(state: dict[str, Any]) -> None:
        proposal = next((item for item in state["proposals"] if item["proposal_id"] == proposal_id), None)
        if proposal is None:
            raise V4ReservoirError(f"unknown proposal {proposal_id}")
        if outcome == "accepted":
            if not explicit_commit_verified or not committed_registry_task_id:
                raise V4ReservoirError("accepted state requires a verified explicit committed-task entry")
            if proposal["state"] == "accepted":
                existing = proposal.get("final_disposition") or {}
                if existing.get("registry_task_id") != committed_registry_task_id:
                    raise V4ReservoirError("accepted proposal is bound to a different registry task")
                return
            reservation = next(
                (item for item in state["active_reservations"] if item["proposal_id"] == proposal_id),
                None,
            )
            if reservation is not None and reservation["owner_run_id"] != source_run_id:
                raise V4ReservoirError("explicit acceptance cannot preempt another run's reservation")
            _transition(
                proposal, "accepted", reason=reason, now=timestamp,
                allow_explicit_accept=True,
            )
            proposal["final_disposition"] = {
                "outcome": "accepted", "registry_task_id": committed_registry_task_id,
                "reason": normalize_concept(reason), "at_epoch_seconds": round(timestamp, 3),
            }
            _resolve_active_reservation(state, proposal_id, outcome="accepted", note=reason, now=timestamp)
            # Acceptance is strong novelty memory, represented both by the
            # committed bank and this auditable local record.
            state["rejection_memory"].append(
                _memory(proposal, tier="hard_duplicate", reason=f"accepted as {committed_registry_task_id}: {reason}", source_run_id=source_run_id, now=timestamp)
            )
            return
        if outcome != "rejected" or memory_tier not in MEMORY_TIERS:
            raise V4ReservoirError("non-accepted disposition must be rejected with an explicit memory tier")
        _transition(proposal, "rejected", reason=reason, now=timestamp)
        proposal["final_disposition"] = {
            "outcome": "rejected", "memory_tier": memory_tier,
            "reason": normalize_concept(reason), "at_epoch_seconds": round(timestamp, 3),
        }
        _resolve_active_reservation(
            state, proposal_id,
            outcome="aborted" if memory_tier == "soft_mechanism" else "rejected",
            note=reason, now=timestamp,
        )
        state["rejection_memory"].append(
            _memory(proposal, tier=memory_tier, reason=reason, source_run_id=source_run_id, now=timestamp)
        )

    return _locked_update(path, mutate)


def recover_expired_leases(
    path: str | Path,
    *,
    active_owner_run_ids: Iterable[str],
    now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else now
    active_owners = set(active_owner_run_ids)

    def mutate(state: dict[str, Any]) -> None:
        by_id = {item["proposal_id"]: item for item in state["proposals"]}
        expired = [
            item for item in state["active_reservations"]
            if item["owner_run_id"] not in active_owners
            and timestamp - float(item["heartbeat_at_epoch_seconds"]) > int(item["lease_seconds"])
        ]
        for reservation in expired:
            proposal = by_id[reservation["proposal_id"]]
            reason = f"expired abandoned lease recovered from {reservation['owner_run_id']}"
            _resolve_active_reservation(state, proposal["proposal_id"], outcome="recovered_expired", note=reason, now=timestamp)
            _transition(proposal, "proposed", reason=reason, now=timestamp, allow_recovery=True)
            proposal["final_disposition"] = None
            state["rejection_memory"].append(
                _memory(proposal, tier="recoverable_work", reason=reason, source_run_id=reservation["owner_run_id"], now=timestamp, active=False)
            )

    return _locked_update(path, mutate)


def mark_selection_decisions(
    path: str | Path,
    *,
    window_id: str,
    selector_id: str,
    assessments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    timestamp = time.time()

    def mutate(state: dict[str, Any]) -> None:
        by_id = {item["proposal_id"]: item for item in state["proposals"]}
        for assessment in assessments:
            proposal = by_id[str(assessment["candidate_id"])]
            proposal["selection_attempts"].append(
                {
                    "window_id": window_id,
                    "stage": selector_id,
                    "decision": assessment["decision"],
                    "reason": normalize_concept(str(assessment["reason"])),
                    "at_epoch_seconds": round(timestamp, 3),
                }
            )
            proposal["novelty_status"] = {
                **proposal["novelty_status"],
                "status": "selected_for_development" if assessment["decision"] == "select" else "selector_rejected",
                "reason": normalize_concept(str(assessment["reason"])),
            }

    return _locked_update(path, mutate)


def memory_digest(value: Mapping[str, Any], *, limit_per_tier: int = 30) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tier in sorted(MEMORY_TIERS):
        rows = [item for item in value["rejection_memory"] if item["tier"] == tier and item.get("active", True)]
        result[tier] = [
            {"mechanism_summary": item["mechanism_summary"], "reason": item["reason"]}
            for item in rows[-limit_per_tier:]
        ]
    return result


def reservation_digest(value: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "reservation_id": item["reservation_id"],
            "mechanism_summary": item["mechanism_summary"],
        }
        for item in sorted(value["active_reservations"], key=lambda row: row["reservation_id"])
    ]


__all__ = [
    "MEMORY_TIERS", "PROPOSAL_STATES", "RESERVOIR_ID", "V4ReservoirError",
    "active_count", "append_supply", "concept_hash", "freeze_eligible_window",
    "heartbeat_leases", "initialize_reservoir", "load_reservoir", "make_proposal",
    "mark_selection_decisions", "memory_digest", "normalize_concept",
    "recover_expired_leases", "recover_interrupted_slate",
    "replenishment_decision", "reservation_digest",
    "reserve_slate", "reservoir_status", "resolve_proposal", "validate_reservoir",
]
