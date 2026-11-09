"""Persistent 8+4 provider-call limiter for the pre-E20 scheduler study.

The downstream lane owns eight slots.  Four upstream-only slots prepare idea
supply and selection.  Downstream slots may be lent upstream while no
downstream slate is ready.  The reverse is permitted only after the scheduler
declares that no upstream work remains for this pool (``close_upstream``,
v1 continuous supply since 2026-10-11): downstream may then use every slot.
This module is additive: the frozen E18.1 shared-eight limiter remains untouched.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from .util import atomic_write_json, content_hash, read_json


class DedicatedPoolError(RuntimeError):
    """Raised when the experimental worker partition loses an invariant."""


class _BoundLimiter:
    def __init__(self, pool: "DedicatedCallPool", category: str) -> None:
        self.pool = pool
        self.category = category

    def __enter__(self) -> Self:
        self.pool.acquire(self.category)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.pool.release(self.category)


class DedicatedCallPool:
    """Durable asymmetric limiter with a configured global hard ceiling (qualified default 8+4=12)."""

    CATEGORIES = frozenset({"selection_codex", "selection_claude", "downstream"})
    UPSTREAM = frozenset({"selection_codex", "selection_claude"})

    def __init__(
        self,
        path: str | Path,
        *,
        downstream_workers: int = 8,
        upstream_workers: int = 4,
        global_ceiling: int = 12,
        codex_cap: int = 12,
        claude_cap: int = 4,
        allow_resize: bool = False,
        ramp_gap_seconds: float = 0.0,
        ramp_window_seconds: float = 0.0,
    ) -> None:
        if (
            type(downstream_workers) is not int or type(upstream_workers) is not int
            or downstream_workers < 1 or upstream_workers < 1
            or downstream_workers + upstream_workers != global_ceiling
        ):
            raise DedicatedPoolError("pool shape must be positive downstream+upstream=global")
        self.path = Path(path)
        self.downstream_workers = downstream_workers
        self.upstream_workers = upstream_workers
        self.workers = global_ceiling
        self.provider_caps = {
            "selection_codex": codex_cap,
            "selection_claude": claude_cap,
        }
        self.condition = threading.Condition(threading.RLock())
        self.local = threading.local()
        # Start-up ramp (2026-10-15, after the v1-b16 429 waves, which all came at a run's start): for the first
        # ramp_window_seconds of a session, acquisitions are spaced at least ramp_gap_seconds apart. It paces this
        # process only, is not part of the pool's fingerprint, and is off by default.
        self.ramp_gap_seconds = float(ramp_gap_seconds)
        self.ramp_window_seconds = float(ramp_window_seconds)
        self._session_started = time.time()
        self._last_acquired = 0.0
        borrowing_policy = "ready-slate-or-downstream-waiter-priority-v3"
        fingerprint = content_hash(
            {
                "downstream_workers": downstream_workers,
                "upstream_workers": upstream_workers,
                "global_ceiling": global_ceiling,
                "provider_caps": self.provider_caps,
                "borrowing_direction": borrowing_policy,
            }
        )
        if self.path.is_file():
            self.state = read_json(self.path)
            if self.state.get("fingerprint") != fingerprint:
                if not allow_resize or self.state.get("borrowing_direction") != borrowing_policy:
                    raise DedicatedPoolError("dedicated pool configuration changed on resume")
                # A resumed batch may run with more or fewer workers (a recorded config amendment); the
                # borrowing policy and every past event stay as they were.
                self.state.setdefault("events", []).append(
                    {
                        "kind": "resume_resized",
                        "at_epoch_seconds": round(time.time(), 6),
                        "from": {k: self.state.get(k) for k in ("downstream_workers", "upstream_workers", "global_ceiling")},
                        "to": {"downstream_workers": downstream_workers, "upstream_workers": upstream_workers,
                               "global_ceiling": global_ceiling},
                    }
                )
                self.state.update(fingerprint=fingerprint, downstream_workers=downstream_workers,
                                  upstream_workers=upstream_workers, global_ceiling=global_ceiling,
                                  provider_caps=deepcopy(self.provider_caps))
            abandoned = deepcopy(self.state.get("active_calls") or {})
            if abandoned:
                self.state.setdefault("events", []).append(
                    {
                        "kind": "resume_abandoned_active_calls",
                        "at_epoch_seconds": round(time.time(), 6),
                        "calls": abandoned,
                    }
                )
            self.state["active_calls"] = {}
            self.state["active_by_category"] = {
                category: 0 for category in self.CATEGORIES
            }
            # Waiters belong to the dead process just like active calls.  A
            # stale downstream waiter must not suppress safe upstream lending.
            self.state["waiting_by_category"] = {
                category: 0 for category in self.CATEGORIES
            }
            self.state["downstream_ready"] = False
            # A new session re-declares upstream closure itself; absent for pools that never closed it.
            self.state.pop("upstream_closed", None)
            self.state.setdefault("execution_sessions", []).append(
                {
                    "session": len(self.state.get("execution_sessions") or []) + 1,
                    "started_at_epoch_seconds": round(time.time(), 6),
                    "outcome": "running",
                }
            )
        else:
            self.state = {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "downstream_workers": downstream_workers,
                "upstream_workers": upstream_workers,
                "global_ceiling": global_ceiling,
                "provider_caps": deepcopy(self.provider_caps),
                "borrowing_direction": borrowing_policy,
                "downstream_ready": False,
                "active_calls": {},
                "active_by_category": {
                    category: 0 for category in self.CATEGORIES
                },
                "waiting_by_category": {
                    category: 0 for category in self.CATEGORIES
                },
                "peak_active": 0,
                "peak_downstream": 0,
                "peak_upstream": 0,
                "peak_borrowed_upstream": 0,
                "events": [],
                "execution_sessions": [
                    {
                        "session": 1,
                        "started_at_epoch_seconds": round(time.time(), 6),
                        "outcome": "running",
                    }
                ],
            }
        self._validate_locked()
        self._save_locked()

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self.state)

    def _counts(self) -> tuple[int, int, int]:
        active = self.state["active_by_category"]
        downstream = int(active["downstream"])
        upstream = sum(int(active[category]) for category in self.UPSTREAM)
        return downstream, upstream, downstream + upstream

    def _validate_locked(self) -> None:
        if self.state.get("schema_version") != 1:
            raise DedicatedPoolError("unsupported dedicated pool state")
        downstream, upstream, total = self._counts()
        if downstream > self._downstream_limit():
            raise DedicatedPoolError("downstream exceeded its eight-slot lane")
        if total > self.workers:
            raise DedicatedPoolError("provider calls exceeded the global ceiling")
        if self.state.get("downstream_ready") and upstream > self.upstream_workers:
            # Borrowed calls already running are explicitly allowed to drain.
            borrowed = [
                row
                for row in self.state.get("active_calls", {}).values()
                if row.get("borrowed") is True
            ]
            if len(borrowed) < upstream - self.upstream_workers:
                raise DedicatedPoolError("unmarked upstream borrowing survived priority return")

    def bound(self, category: str) -> _BoundLimiter:
        if category not in self.CATEGORIES:
            raise DedicatedPoolError(f"unknown dedicated-pool category: {category}")
        return _BoundLimiter(self, category)

    @contextmanager
    def call_context(self, category: str, call_id: str):
        prior = getattr(self.local, "call", None)
        self.local.call = {"category": category, "call_id": call_id}
        try:
            yield
        finally:
            self.local.call = prior

    def _call_key(self, category: str) -> str:
        context = getattr(self.local, "call", None) or {}
        call_id = str(context.get("call_id") or f"thread-{threading.get_ident()}")
        return f"{category}:{call_id}:{threading.get_ident()}"

    def set_downstream_ready(self, ready: bool, *, reason: str) -> None:
        """Return borrowed capacity as soon as executable downstream work exists."""

        with self.condition:
            wanted = bool(ready)
            if bool(self.state.get("downstream_ready")) != wanted:
                self.state["downstream_ready"] = wanted
                self.state["events"].append(
                    {
                        "kind": "downstream_priority_changed",
                        "ready": wanted,
                        "reason": str(reason),
                        "at_epoch_seconds": round(time.time(), 6),
                    }
                )
                self._save_locked()
                self.condition.notify_all()

    def _downstream_limit(self) -> int:
        return self.workers if self.state.get("upstream_closed") else self.downstream_workers

    def close_upstream(self, *, reason: str) -> None:
        """No upstream call will come again this session: downstream may take the idle upstream slots."""

        with self.condition:
            if self.state.get("upstream_closed"):
                return
            self.state["upstream_closed"] = True
            self.state["events"].append(
                {
                    "kind": "upstream_closed",
                    "reason": str(reason),
                    "at_epoch_seconds": round(time.time(), 6),
                }
            )
            self._save_locked()
            self.condition.notify_all()

    def _upstream_active(self) -> int:
        return sum(
            int(self.state["active_by_category"][category])
            for category in self.UPSTREAM
        )

    def _can_acquire(self, category: str) -> tuple[bool, bool]:
        downstream, upstream, total = self._counts()
        if total >= self.workers:
            return False, False
        if category == "downstream":
            return downstream < self._downstream_limit(), False
        if int(self.state["active_by_category"][category]) >= int(
            self.provider_caps[category]
        ):
            return False, False
        if upstream < self.upstream_workers:
            return True, False
        downstream_waiting = int(self.state["waiting_by_category"]["downstream"])
        # The approved intervention lends downstream capacity only while no
        # selected slate is ready or active.  Publication therefore closes
        # admission for *new* loans even before the first downstream provider
        # request is queued.  Existing loans are never preempted; they drain
        # normally while the dedicated four upstream slots remain usable.
        may_borrow = (
            not bool(self.state.get("downstream_ready"))
            and not downstream_waiting
        )
        return may_borrow, may_borrow

    def _ramp_pause(self) -> float:
        now = time.time()
        if self.ramp_gap_seconds <= 0 or now - self._session_started >= self.ramp_window_seconds:
            return 0.0
        return self._last_acquired + self.ramp_gap_seconds - now

    def acquire(self, category: str) -> None:
        if category not in self.CATEGORIES:
            raise DedicatedPoolError(f"unknown dedicated-pool category: {category}")
        key = self._call_key(category)
        queued = round(time.time(), 6)
        with self.condition:
            self.state["waiting_by_category"][category] = (
                int(self.state["waiting_by_category"][category]) + 1
            )
            self.state["events"].append(
                {
                    "kind": "queued",
                    "category": category,
                    "call_key": key,
                    "at_epoch_seconds": queued,
                }
            )
            self._save_locked()
            borrowed = False
            while True:
                allowed, borrowed = self._can_acquire(category)
                if allowed:
                    pause = self._ramp_pause()
                    if pause <= 0:
                        break
                    self.condition.wait(timeout=pause)
                    continue
                self.condition.wait(timeout=5.0)
            self._last_acquired = time.time()
            acquired = round(self._last_acquired, 6)
            self.state["waiting_by_category"][category] = max(
                0, int(self.state["waiting_by_category"][category]) - 1
            )
            self.state["active_by_category"][category] = (
                int(self.state["active_by_category"][category]) + 1
            )
            self.state["active_calls"][key] = {
                "category": category,
                "queued_at_epoch_seconds": queued,
                "acquired_at_epoch_seconds": acquired,
                "borrowed": borrowed,
            }
            if category == "downstream" and int(self.state["active_by_category"][category]) > self.downstream_workers:
                self.state["active_calls"][key]["upstream_slot"] = True  # only after close_upstream
            downstream, upstream, total = self._counts()
            self.state["peak_active"] = max(int(self.state["peak_active"]), total)
            self.state["peak_downstream"] = max(
                int(self.state["peak_downstream"]), downstream
            )
            self.state["peak_upstream"] = max(
                int(self.state["peak_upstream"]), upstream
            )
            self.state["peak_borrowed_upstream"] = max(
                int(self.state["peak_borrowed_upstream"]),
                max(0, upstream - self.upstream_workers),
            )
            self.state["events"].append(
                {
                    "kind": "acquired",
                    "category": category,
                    "call_key": key,
                    "borrowed": borrowed,
                    "at_epoch_seconds": acquired,
                    "active_total": total,
                    "active_downstream": downstream,
                    "active_upstream": upstream,
                }
            )
            self._validate_locked()
            self._save_locked()

    def release(self, category: str) -> None:
        key = self._call_key(category)
        with self.condition:
            active = self.state["active_calls"].pop(key, None)
            if not isinstance(active, dict):
                raise DedicatedPoolError(f"released unknown provider call: {key}")
            self.state["active_by_category"][category] = max(
                0, int(self.state["active_by_category"][category]) - 1
            )
            released = round(time.time(), 6)
            downstream, upstream, total = self._counts()
            self.state["events"].append(
                {
                    "kind": "released",
                    "category": category,
                    "call_key": key,
                    "borrowed": bool(active.get("borrowed")),
                    "at_epoch_seconds": released,
                    "active_total": total,
                    "active_downstream": downstream,
                    "active_upstream": upstream,
                    "held_seconds": round(
                        released - float(active["acquired_at_epoch_seconds"]), 6
                    ),
                }
            )
            self._validate_locked()
            self._save_locked()
            self.condition.notify_all()

    def heartbeat(self) -> None:
        with self.condition:
            self.state["last_heartbeat_at_epoch_seconds"] = round(time.time(), 6)
            self._save_locked()

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return deepcopy(self.state)

    def finish(self) -> dict[str, Any]:
        with self.condition:
            if self.state["active_calls"]:
                raise DedicatedPoolError("cannot finish with active provider calls")
            self.state["execution_sessions"][-1]["outcome"] = "complete"
            self.state["execution_sessions"][-1]["ended_at_epoch_seconds"] = round(
                time.time(), 6
            )
            self.state["completed"] = True
            self._validate_locked()
            self._save_locked()
            return deepcopy(self.state)


__all__ = ["DedicatedCallPool", "DedicatedPoolError"]
