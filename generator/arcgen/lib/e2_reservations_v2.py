"""Run-local reservation handoff and lease lifecycle for E2 v2."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from .util import atomic_write_json, content_hash, read_json

PIPELINE = "arcgen-e2-run-local-reservations"
VERSION = "2.0"


class E2ReservationError(RuntimeError):
    """Raised when a selected-slate handoff or reservation changes."""


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validated_reference(raw: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(raw.get("path") or ""))
    digest = raw.get("sha256")
    if not path.is_file() or not isinstance(digest, str) or _sha(path) != digest:
        raise E2ReservationError(f"{label} is missing or changed")
    return path


def validate_recoverable_handoff(
    selection_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove every selected proposal was atomically reserved before handoff."""

    selected_ids = sorted(
        str(candidate)
        for slate in selection_manifest.get("slates", [])
        for candidate in slate.get("selected_candidate_ids", [])
    )
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise E2ReservationError(
            "selection manifest has missing or duplicate candidates"
        )
    resolution_ref = selection_manifest.get("lease_terminal_resolution")
    if not isinstance(resolution_ref, Mapping):
        raise E2ReservationError("selection manifest lacks lease handoff")
    resolution_path = _validated_reference(
        resolution_ref, label="lease terminal resolution"
    )
    resolution = read_json(resolution_path)
    if (
        resolution.get("resolution") != "recoverable_downstream_handoff"
        or resolution.get("semantic_rejection") is not False
        or resolution.get("production_state_mutated") is not False
        or resolution.get("remaining_owned_active_leases") != []
        or sorted(map(str, resolution.get("resolved_proposal_ids") or []))
        != selected_ids
    ):
        raise E2ReservationError("selection lease handoff is incomplete")

    source_rows = []
    reserved_ids = []
    for slate in selection_manifest.get("slates", []):
        reservation_ref = slate.get("cross_slate_reservation")
        if not isinstance(reservation_ref, Mapping):
            raise E2ReservationError("selected slate lacks reservation artifact")
        reservation_path = _validated_reference(
            reservation_ref,
            label=f"slate {slate.get('slate_index')} reservation",
        )
        value = read_json(reservation_path)
        reservations = (value.get("reservations") or {}).get("reservations") or []
        if (value.get("reservations") or {}).get("reserved_atomically") is not True:
            raise E2ReservationError("source slate was not reserved atomically")
        slate_selected = sorted(map(str, slate.get("selected_candidate_ids") or []))
        slate_reserved = sorted(
            map(str, (row.get("proposal_id") for row in reservations))
        )
        if slate_selected != slate_reserved:
            raise E2ReservationError("source reservation does not match selected slate")
        reserved_ids.extend(slate_reserved)
        source_rows.append(
            {
                "slate_index": int(slate["slate_index"]),
                "path": str(reservation_path.resolve()),
                "sha256": _sha(reservation_path),
                "proposal_ids": slate_reserved,
                "reservations": deepcopy(reservations),
            }
        )
    if sorted(reserved_ids) != selected_ids:
        raise E2ReservationError("source reservation set differs from handoff set")
    return {
        "schema_version": 2,
        "validated": True,
        "selected_proposal_ids": selected_ids,
        "source_reservations": source_rows,
        "terminal_resolution": {
            "path": str(resolution_path.resolve()),
            "sha256": _sha(resolution_path),
        },
        "handoff_hash": content_hash(
            {
                "selected": selected_ids,
                "sources": [row["sha256"] for row in source_rows],
                "resolution": _sha(resolution_path),
            }
        ),
    }


class RunLocalReservationLeases:
    """Atomically acquire the complete experimental slate before planning."""

    def __init__(
        self,
        *,
        path: Path,
        owner_run_id: str,
        selected: Sequence[Mapping[str, Any]],
        handoff: Mapping[str, Any],
        lease_seconds: int = 259_200,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if lease_seconds < 60:
            raise E2ReservationError("reservation lease must be at least 60 seconds")
        self.path = path
        self.owner_run_id = str(owner_run_id)
        self.clock = clock
        self.lease_seconds = int(lease_seconds)
        self.selected = [deepcopy(dict(row)) for row in selected]
        ids = [str(row["source_candidate_id"]) for row in self.selected]
        if len(ids) != len(set(ids)) or sorted(ids) != sorted(
            map(str, handoff.get("selected_proposal_ids") or [])
        ):
            raise E2ReservationError("runtime selected set differs from source handoff")
        self.fingerprint = content_hash(
            {
                "pipeline": PIPELINE,
                "version": VERSION,
                "owner_run_id": self.owner_run_id,
                "selected": sorted(ids),
                "handoff_hash": handoff.get("handoff_hash"),
                "lease_seconds": self.lease_seconds,
            }
        )
        self.state = self._load_or_acquire(handoff)

    def _now(self) -> float:
        return round(float(self.clock()), 6)

    def _load_or_acquire(self, handoff: Mapping[str, Any]) -> dict[str, Any]:
        now = self._now()
        if self.path.is_file():
            state = read_json(self.path)
            if state.get("fingerprint") != self.fingerprint:
                raise E2ReservationError("reservation ledger belongs to another run")
            recovered = False
            for lease in state.get("leases", {}).values():
                if lease.get("state") != "active":
                    continue
                if float(lease.get("expires_at_epoch_seconds") or 0) < now:
                    lease["recovery_count"] = int(lease.get("recovery_count") or 0) + 1
                    lease["heartbeat_at_epoch_seconds"] = now
                    lease["expires_at_epoch_seconds"] = now + self.lease_seconds
                    recovered = True
            if recovered:
                state.setdefault("history", []).append(
                    {"kind": "expired_leases_recovered", "at_epoch_seconds": now}
                )
                atomic_write_json(self.path, state)
            return state
        leases = {}
        for row in self.selected:
            source = str(row["source_candidate_id"])
            leases[source] = {
                "reservation_id": f"reservation:{self.owner_run_id}:{source}",
                "source_candidate_id": source,
                "runtime_candidate_id": str(row.get("task_id") or source),
                "slate_index": int(row["slate_index"]),
                "development_priority": int(row["source_development_priority"]),
                "owner_run_id": self.owner_run_id,
                "state": "active",
                "created_at_epoch_seconds": now,
                "heartbeat_at_epoch_seconds": now,
                "expires_at_epoch_seconds": now + self.lease_seconds,
                "recovery_count": 0,
                "terminal_disposition": None,
            }
        state = {
            "schema_version": 2,
            "pipeline": PIPELINE,
            "version": VERSION,
            "fingerprint": self.fingerprint,
            "owner_run_id": self.owner_run_id,
            "handoff_hash": handoff.get("handoff_hash"),
            "lease_seconds": self.lease_seconds,
            "reserved_atomically": True,
            "production_state_mutated": False,
            "leases": leases,
            "history": [
                {
                    "kind": "complete_selected_set_acquired",
                    "at_epoch_seconds": now,
                    "count": len(leases),
                }
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, state)
        return state

    def heartbeat(self) -> None:
        now = self._now()
        for lease in self.state["leases"].values():
            if lease["state"] == "active":
                lease["heartbeat_at_epoch_seconds"] = now
                lease["expires_at_epoch_seconds"] = now + self.lease_seconds
        self.state["history"].append({"kind": "heartbeat", "at_epoch_seconds": now})
        atomic_write_json(self.path, self.state)

    def assert_active(
        self,
        source_candidate_ids: Sequence[str],
        *,
        require_complete_set: bool = False,
    ) -> None:
        now = self._now()
        requested = set(map(str, source_candidate_ids))
        if require_complete_set and requested != set(self.state["leases"]):
            raise E2ReservationError(
                "planning set differs from acquired reservation set"
            )
        if not requested.issubset(self.state["leases"]):
            raise E2ReservationError("planning set contains an unreserved candidate")
        bad = [
            source
            for source, lease in self.state["leases"].items()
            if lease.get("state") != "active"
            or float(lease.get("expires_at_epoch_seconds") or 0) < now
        ]
        if bad:
            raise E2ReservationError(
                "planning attempted without active leases: " + ", ".join(sorted(bad))
            )

    def assert_all_active(self, source_candidate_ids: Sequence[str]) -> None:
        self.assert_active(source_candidate_ids, require_complete_set=True)

    def resolve(self, source_candidate_id: str, disposition: str) -> None:
        source = str(source_candidate_id)
        if source not in self.state["leases"]:
            raise E2ReservationError(f"unknown reservation {source}")
        lease = self.state["leases"][source]
        if lease["state"] == "terminal":
            if lease["terminal_disposition"] != disposition:
                raise E2ReservationError("terminal reservation disposition changed")
            return
        now = self._now()
        lease["state"] = "terminal"
        lease["terminal_disposition"] = str(disposition)
        lease["resolved_at_epoch_seconds"] = now
        self.state["history"].append(
            {
                "kind": "lease_resolved",
                "source_candidate_id": source,
                "disposition": str(disposition),
                "at_epoch_seconds": now,
            }
        )
        atomic_write_json(self.path, self.state)


__all__ = [
    "E2ReservationError",
    "RunLocalReservationLeases",
    "validate_recoverable_handoff",
]
