"""Live E3 selection/development-overlap qualification.

This additive coordinator runs the frozen minority-Opus idea supplier and
source-blind selectors while already-selected candidates advance through the
unchanged V4.8/E2 downstream stack.  Every substantive provider call shares
one eight-slot limiter.  The run is experimental: it cannot mutate production
state, commit tasks, or promote a scheduling policy.

The E8 deterministic plan checker is recorded prospectively, but is strictly
non-gating.  Every flagged plan continues through the ordinary downstream
path so false suppression can be measured rather than assumed away.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from .config import ConfigError
from .e1_selection_cohort import (
    E1SelectionCohortError,
    load_cohort_config,
    run_selection_cohort,
)
from .e2_attempt_ledger_v2 import build_attempt_ledger
from .e2_live_streaming import (
    IdempotentSlateBudget,
    _assert_downstream_prompt_blinding,
    _blind_one,
    _load_or_freeze_post_blind,
    _materialize,
    _next_stage,
    _plan_one,
    _realize_one,
    _safe_summary,
    _technical_one,
)
from .e2_live_streaming_v2 import (
    _aggregate_scheduler_report,
    _AttemptPreservingInvoker,
    _blind_primary,
    _can_dispatch,
    _reconcile_active,
    _technical_primary,
)
from .e2_streaming_scheduler_v2 import (
    PersistentCandidateSchedulerV2,
    validate_dependency_events_v2,
)
from .e3_overlap_replay_v2 import (
    _board_tail,
    _downstream_nodes,
    _schedule,
    _selection_nodes,
    _with_board_tail,
)
from .e8_plan_contradictions import check_plan
from .util import atomic_write_json, content_hash, read_json
from .v41_experiments import V41Runtime
from .v41_integration import (
    _assert_hashes_unchanged,
    _committed_distribution,
    _palette_target,
    _protected_hashes,
)
from .v41_palette import schedule_backgrounds
from .v41_test_challenge import _board as _editorial_boards
from .v44_abstraction_led import _install_extensions as _install_v44_extensions
from .v44_scale_config import load_v44_scale_config
from .v48_evidence_scale_paired import _install_evidence_extension

PIPELINE = "arcgen-e3-live-selection-development-overlap"
VERSION = "1.0"
EXPECTED_WINNER_ID = (
    "arcgen-v4.8-evidence-preserving-scale-experimental-winner-20260812"
)


class E3LiveError(RuntimeError):
    """Raised when E3 loses isolation, persistence, or timing provenance."""


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"E3 configuration must be a mapping: {path}")
    return value


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _validated_ref(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must freeze path and sha256")
    path = _resolve(root, value.get("path"), f"{label}.path")
    digest = value.get("sha256")
    if not path.is_file() or not isinstance(digest, str) or _sha(path) != digest:
        raise ConfigError(f"{label} is missing or changed")
    return path


def _write_immutable(path: Path, value: Any, *, label: str) -> None:
    if path.is_file():
        if read_json(path) != value:
            raise E3LiveError(f"immutable {label} changed: {path}")
        return
    atomic_write_json(path, value)


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read_yaml_or_json(source)
    if raw.get("pipeline") != PIPELINE or str(raw.get("version")) != VERSION:
        raise ConfigError(f"E3 configuration must use {PIPELINE} {VERSION}")
    root = source.parent.parent if source.parent.name == "config" else source.parent
    base_path = _validated_ref(root, raw.get("base_scale_config"), "base_scale_config")
    winner_path = _validated_ref(
        root, raw.get("experimental_winner_manifest"), "experimental_winner_manifest"
    )
    selection_path = _validated_ref(
        root, raw.get("selection_config"), "selection_config"
    )
    winner = read_json(winner_path)
    if (
        winner.get("baseline_id") != EXPECTED_WINNER_ID
        or winner.get("status") != "frozen_experimental_candidate_continue_testing"
        or (winner.get("production_policy") or {}).get(
            "production_promotion_authorized"
        )
        is not False
    ):
        raise ConfigError("E3 must use the frozen, unpromoted V4.8 winner")
    selection = load_cohort_config(selection_path)
    if (
        selection.get("producer_kind") != "minority_opus_supply"
        or selection.get("runner_kind") != "v4_6_minority_opus_supply_sol_downstream"
        or selection.get("functional_scale", {}).get("enabled") is not False
    ):
        raise ConfigError(
            "E3 requires the source-blind 75/25 supplier with no auxiliary routes"
        )
    slate_count = int(selection["cohort"]["slate_count"])
    if not 3 <= slate_count <= 6:
        raise ConfigError("E3 live qualification requires three to six slates")
    scheduler = raw.get("scheduler")
    run = raw.get("run")
    if not isinstance(scheduler, Mapping) or scheduler.get("enabled") is not True:
        raise ConfigError("scheduler.enabled must be true")
    exact = {
        "global_workers": 8,
        "selection_cap_during_development": 6,
        "codex_selection_cap": 4,
        "claude_selection_cap": 4,
        "escalation_calls_per_slate": 5,
    }
    for field, expected in exact.items():
        if scheduler.get(field) != expected:
            raise ConfigError(f"scheduler.{field} must be {expected}")
    for field in (
        "candidate_level_streaming",
        "cross_slate_selection_overlap",
        "shared_global_call_limiter",
        "whole_slate_reserved_before_planning",
        "reservations_visible_to_later_selectors",
        "unchanged_editorial_boards",
        "attempt_complete_accounting",
        "e8_shadow_non_gating",
    ):
        if scheduler.get(field) is not True:
            raise ConfigError(f"scheduler.{field} must be true")
    if (
        not isinstance(run, Mapping)
        or not isinstance(run.get("id"), str)
        or run.get("mock") not in {True, False}
    ):
        raise ConfigError("run.id and run.mock are required")
    for field in (
        "automatic_task_commit",
        "production_state_mutation_allowed",
        "production_promotion_allowed",
    ):
        if run.get(field) is not False:
            raise ConfigError(f"run.{field} must be false")
    output = _resolve(root, raw.get("output_directory"), "output_directory")
    if output.name != run["id"]:
        raise ConfigError("output directory basename must equal run.id")

    config = deepcopy(load_v44_scale_config(base_path))
    if Path(config["_repository_root"]).resolve() != root.resolve():
        raise ConfigError("base configuration resolved another repository root")
    config["seed"] = int(raw.get("seed"))
    config["model"] = deepcopy(config["model"])
    config["model"]["concurrency"] = 8
    config["functional_scale"] = deepcopy(config["functional_scale"])
    config["functional_scale"]["escalation_calls_per_slate"] = 5
    config["run"] = {
        **deepcopy(config["run"]),
        "id": str(run["id"]),
        "mock": bool(run["mock"]),
        "automatically_commit_generated_tasks": False,
    }
    config["paths"] = deepcopy(config["paths"])
    config["paths"]["runs"] = str(output.parent)
    config["scheduler"] = deepcopy(dict(scheduler))
    config["_e3_config_path"] = str(source)
    config["_e3_output_directory"] = str(output)
    config["_e3_base_config_path"] = str(base_path)
    config["_e3_winner_manifest_path"] = str(winner_path)
    config["_e3_selection_config_path"] = str(selection_path)
    config["_e3_selection_config"] = selection
    return config


class _BoundLimiter:
    def __init__(self, pool: SharedCallPool, category: str) -> None:
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


class SharedCallPool:
    """One persistent, work-conserving eight-slot pool for every provider."""

    CATEGORIES = frozenset({"selection_codex", "selection_claude", "downstream"})

    def __init__(
        self,
        path: Path,
        *,
        workers: int,
        selection_cap: int,
        codex_cap: int,
        claude_cap: int,
    ) -> None:
        self.path = path
        self.workers = workers
        self.selection_cap = selection_cap
        self.provider_caps = {
            "selection_codex": codex_cap,
            "selection_claude": claude_cap,
        }
        self.condition = threading.Condition(threading.RLock())
        self.local = threading.local()
        fingerprint = content_hash(
            {
                "workers": workers,
                "selection_cap": selection_cap,
                "provider_caps": self.provider_caps,
            }
        )
        if path.is_file():
            self.state = read_json(path)
            if self.state.get("fingerprint") != fingerprint:
                raise E3LiveError("shared limiter configuration changed on resume")
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
            self.state["active_by_category"] = {key: 0 for key in self.CATEGORIES}
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
                "workers": workers,
                "selection_cap_during_development": selection_cap,
                "provider_caps": self.provider_caps,
                "active_calls": {},
                "active_by_category": {key: 0 for key in self.CATEGORIES},
                "waiting_by_category": {key: 0 for key in self.CATEGORIES},
                "peak_active": 0,
                "events": [],
                "execution_sessions": [
                    {
                        "session": 1,
                        "started_at_epoch_seconds": round(time.time(), 6),
                        "outcome": "running",
                    }
                ],
            }
        self._save_locked()

    def _save_locked(self) -> None:
        atomic_write_json(self.path, self.state)

    def bound(self, category: str) -> _BoundLimiter:
        if category not in self.CATEGORIES:
            raise E3LiveError(f"unknown shared-limiter category: {category}")
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

    def _selection_active(self) -> int:
        active = self.state["active_by_category"]
        return int(active["selection_codex"]) + int(active["selection_claude"])

    def _can_acquire(self, category: str) -> bool:
        active = self.state["active_by_category"]
        total = sum(int(value) for value in active.values())
        if total >= self.workers:
            return False
        if category == "downstream":
            # Keep one opportunity for a waiting future slate if selection has
            # been completely starved, while otherwise preferring deep work.
            selection_waiting = int(
                self.state["waiting_by_category"]["selection_codex"]
            ) + int(self.state["waiting_by_category"]["selection_claude"])
            return not (
                selection_waiting
                and self._selection_active() == 0
                and total >= self.workers - 1
            )
        if int(active[category]) >= int(self.provider_caps[category]):
            return False
        downstream_present = bool(
            int(active["downstream"])
            or int(self.state["waiting_by_category"]["downstream"])
        )
        cap = self.selection_cap if downstream_present else self.workers
        return self._selection_active() < cap

    def acquire(self, category: str) -> None:
        if category not in self.CATEGORIES:
            raise E3LiveError(f"unknown limiter category {category}")
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
            while not self._can_acquire(category):
                self.condition.wait(timeout=5.0)
            acquired = round(time.time(), 6)
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
            }
            active = sum(
                int(value) for value in self.state["active_by_category"].values()
            )
            self.state["peak_active"] = max(int(self.state["peak_active"]), active)
            self.state["events"].append(
                {
                    "kind": "acquired",
                    "category": category,
                    "call_key": key,
                    "at_epoch_seconds": acquired,
                    "active_total": active,
                    "active_by_category": deepcopy(self.state["active_by_category"]),
                }
            )
            self._save_locked()

    def release(self, category: str) -> None:
        key = self._call_key(category)
        with self.condition:
            active = self.state["active_calls"].pop(key, None)
            if active is None:
                raise E3LiveError(f"shared limiter released an unknown call: {key}")
            self.state["active_by_category"][category] = max(
                0, int(self.state["active_by_category"][category]) - 1
            )
            released = round(time.time(), 6)
            self.state["events"].append(
                {
                    "kind": "released",
                    "category": category,
                    "call_key": key,
                    "at_epoch_seconds": released,
                    "active_total": sum(
                        int(value)
                        for value in self.state["active_by_category"].values()
                    ),
                    "held_seconds": round(
                        released - float(active["acquired_at_epoch_seconds"]), 6
                    ),
                }
            )
            self._save_locked()
            self.condition.notify_all()

    def heartbeat(self) -> None:
        with self.condition:
            self.state["last_heartbeat_at_epoch_seconds"] = round(time.time(), 6)
            self._save_locked()

    def finish(self) -> dict[str, Any]:
        with self.condition:
            if self.state["active_calls"]:
                raise E3LiveError("cannot finish shared limiter with active calls")
            self.state["execution_sessions"][-1]["outcome"] = "complete"
            self.state["execution_sessions"][-1]["ended_at_epoch_seconds"] = round(
                time.time(), 6
            )
            self.state["completed"] = True
            self._save_locked()
            return deepcopy(self.state)


class _LimiterInvoker:
    """Bind an existing provider wrapper to one shared limiter category."""

    def __init__(self, inner: Any, pool: SharedCallPool, category: str) -> None:
        self.inner = inner
        self.pool = pool
        self.category = category
        # Both the Codex and Claude wrappers use this context manager around
        # the actual provider process. Replacing only this experimental
        # instance leaves every frozen/default wrapper untouched.
        self.inner._semaphore = pool.bound(category)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def invoke(self, *, stage: str, call_id: str, **kwargs: Any) -> Any:
        with self.pool.call_context(self.category, f"{stage}/{call_id}"):
            return self.inner.invoke(stage=stage, call_id=call_id, **kwargs)


class ReservationLedger:
    """Run-local leases that remain active through semantic disposition."""

    def __init__(self, path: Path, *, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.lock = threading.RLock()
        self.state = (
            read_json(path)
            if path.is_file()
            else {
                "schema_version": 1,
                "run_id": run_id,
                "leases": {},
                "history": [],
            }
        )
        if self.state.get("run_id") != run_id:
            raise E3LiveError("reservation ledger belongs to another run")

    def _save(self) -> None:
        atomic_write_json(self.path, self.state)

    def add_slate(self, slate: int, checkpoint: Mapping[str, Any]) -> None:
        reservation = checkpoint.get("context_reservation") or {}
        evidence = reservation.get("reservations") or {}
        reserved = {
            str(row.get("proposal_id")): row
            for row in evidence.get("reservations") or []
            if isinstance(row, Mapping)
        }
        now = round(time.time(), 3)
        with self.lock:
            for row in checkpoint.get("source_rows") or []:
                source = str(row["candidate_id"])
                if source not in reserved:
                    raise E3LiveError(
                        "selected candidate lacks atomic reservation evidence"
                    )
                prior = self.state["leases"].get(source)
                value = {
                    "source_candidate_id": source,
                    "slate_index": slate,
                    "state": "active",
                    "owner_run_id": self.run_id,
                    "source_reservation": deepcopy(reserved[source]),
                    "reserved_at_epoch_seconds": now,
                    "last_heartbeat_at_epoch_seconds": now,
                    "terminal_disposition": None,
                }
                if prior is None:
                    self.state["leases"][source] = value
                    self.state["history"].append(
                        {"event": "reserved", "source_candidate_id": source, "at": now}
                    )
                elif (
                    int(prior["slate_index"]) != slate
                    or prior["source_reservation"] != value["source_reservation"]
                ):
                    raise E3LiveError("reservation changed on resume")
            self._save()

    def heartbeat(self) -> None:
        now = round(time.time(), 3)
        with self.lock:
            for row in self.state["leases"].values():
                if row["state"] == "active":
                    row["last_heartbeat_at_epoch_seconds"] = now
            self._save()

    def resolve_slate(
        self, slate: int, records: Mapping[str, Mapping[str, Any]]
    ) -> None:
        by_source = {
            str(row["source_candidate_id"]): row
            for row in records.values()
            if int(row["slate_index"]) == slate
        }
        now = round(time.time(), 3)
        with self.lock:
            for source, record in by_source.items():
                lease = self.state["leases"].get(source)
                if not lease:
                    raise E3LiveError("terminal candidate lacks reservation lease")
                lease["state"] = "terminal"
                lease["terminal_disposition"] = str(record.get("status") or "terminal")
                lease["terminal_at_epoch_seconds"] = now
                self.state["history"].append(
                    {
                        "event": "terminal",
                        "source_candidate_id": source,
                        "disposition": lease["terminal_disposition"],
                        "at": now,
                    }
                )
            self._save()


def _selected_from_checkpoint(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in checkpoint.get("source_rows") or []:
        rows.append(
            {
                "source_candidate_id": str(raw["candidate_id"]),
                "slate_index": int(raw["slate_index"]),
                "source_development_priority": int(raw["selector_global_priority"]),
                "search_mode": str(raw["search_mode"]),
                "private_kernel_author_provider": str(
                    raw["private_kernel_author_provider"]
                ),
                "kernel": deepcopy(raw["kernel"]),
                "editorial_brief": deepcopy(raw["editorial_brief"]),
            }
        )
    return rows


def _task_id(seed: int, source: str) -> str:
    return content_hash({"e3_seed": seed, "source_candidate_id": source})[:12]


def _add_records(
    runtime: V41Runtime, selected: Sequence[Mapping[str, Any]], *, seed: int
) -> list[str]:
    task_ids = []
    with runtime.lock:
        records = runtime.state.setdefault("semantic_records", {})
        for row in selected:
            source = str(row["source_candidate_id"])
            task_id = _task_id(seed, source)
            task_ids.append(task_id)
            value = {
                "task_id": task_id,
                "source_candidate_id": source,
                "lineage_id": content_hash({"source_candidate_id": source}),
                "arm": "single",
                "private_route": "evidence_preserving_scale",
                "slate_index": int(row["slate_index"]),
                "source_development_priority": int(row["source_development_priority"]),
                "search_mode": str(row["search_mode"]),
                "private_kernel_author_provider": str(
                    row["private_kernel_author_provider"]
                ),
                "kernel": deepcopy(row["kernel"]),
                "editorial_brief": deepcopy(row["editorial_brief"]),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
            prior = records.get(task_id)
            if prior is None:
                records[task_id] = value
            else:
                for field in (
                    "source_candidate_id",
                    "lineage_id",
                    "slate_index",
                    "source_development_priority",
                    "search_mode",
                    "private_kernel_author_provider",
                    "kernel",
                    "editorial_brief",
                ):
                    if prior.get(field) != value[field]:
                        raise E3LiveError(
                            f"dynamic selected record changed: {task_id}:{field}"
                        )
        runtime.state["e3_scheduler"] = {
            "selection_development_overlap": True,
            "global_workers": 8,
            "editorial_barrier_retained": True,
            "e8_shadow_non_gating": True,
        }
        runtime.save()
    if len(task_ids) != len(set(task_ids)):
        raise E3LiveError("anonymous task identity collision")
    return sorted(task_ids)


def _palette_assignments_dynamic(
    runtime: V41Runtime, selected: Sequence[Mapping[str, Any]], *, slate: int
) -> dict[str, int | None]:
    path = runtime.run_dir / "palette_assignments.dynamic.json"
    target, reference = _palette_target(runtime.config)
    committed = _committed_distribution(runtime.config)
    value = (
        read_json(path)
        if path.is_file()
        else {
            "schema_version": 1,
            "assignment_order": "selected-slate reservation order",
            "assignments": {},
            "by_source_candidate": {},
            "slates": {},
            "target_distribution": {str(key): item for key, item in target.items()},
            "real_reference": reference,
            "committed_distribution": committed,
        }
    )
    source_ids = sorted(str(row["source_candidate_id"]) for row in selected)
    existing = value["by_source_candidate"]
    missing = [source for source in source_ids if source not in existing]
    if missing:
        counts = {int(key): int(item) for key, item in committed["counts"].items()}
        for color in existing.values():
            if color is not None:
                counts[int(color)] = counts.get(int(color), 0) + 1
        scheduled = schedule_backgrounds(
            missing,
            seed=int(runtime.config["seed"]) + slate * 104729,
            current_counts=counts,
            target_distribution=target,
            eligible={source: True for source in missing},
        )
        existing.update(scheduled)
    task_assignments = {
        _task_id(int(runtime.config["seed"]), source): existing[source]
        for source in source_ids
    }
    for task_id, color in task_assignments.items():
        prior = value["assignments"].get(task_id)
        if prior is not None and prior != color:
            raise E3LiveError("palette assignment changed on resume")
        value["assignments"][task_id] = color
    value["slates"][str(slate)] = {
        "source_candidate_ids": source_ids,
        "task_ids": sorted(task_assignments),
    }
    atomic_write_json(path, value)
    return {str(key): item for key, item in value["assignments"].items()}


def _record_e8_shadow(runtime: V41Runtime, task_id: str) -> None:
    try:
        record = deepcopy(runtime.state["semantic_records"][task_id])
        plan = record.get("concrete_plan")
        result = (
            check_plan(plan)
            if isinstance(plan, Mapping)
            else {
                "checked": False,
                "reason": "no ready concrete plan",
                "issues": [],
                "deterministic_abort_recommended": False,
            }
        )
        row = {
            "task_id": task_id,
            "checked_after_planning_before_realization": True,
            "non_gating": True,
            "would_abort": result["deterministic_abort_recommended"],
            "result": result,
            "recorded_at_epoch_seconds": round(time.time(), 3),
        }
    except Exception as exc:  # noqa: BLE001 - shadow failure is non-gating
        row = {
            "task_id": task_id,
            "checked_after_planning_before_realization": True,
            "non_gating": True,
            "would_abort": False,
            "audit_operational_failure": True,
            "error": str(exc),
            "recorded_at_epoch_seconds": round(time.time(), 3),
        }
    with runtime.lock:
        prior = runtime.state.setdefault("e8_prospective_shadow", {}).get(task_id)
        if prior is not None and prior != row:
            # Timestamp is observational; preserve the first completed shadow
            # rather than rewriting it during scheduler resume.
            return
        runtime.state["e8_prospective_shadow"][task_id] = row
        runtime.save()


def _run_slate(
    runtime: V41Runtime,
    *,
    slate: int,
    task_ids: Sequence[str],
    assignments: Mapping[str, int | None],
) -> dict[str, Any]:
    budget = IdempotentSlateBudget(runtime, slate, 5)

    def run_stage(task_id: str, stage: str) -> bool:
        if stage == "planning":
            _plan_one(runtime, task_id)
            _record_e8_shadow(runtime, task_id)
            return True
        if stage == "realization":
            _realize_one(runtime, task_id, assignments)
            return True
        record = runtime.state["semantic_records"][task_id]
        if stage == "technical":
            if not record.get("technical_results"):
                return _technical_primary(runtime, task_id)
            _technical_one(runtime, task_id, budget)
            return True
        if stage == "blind":
            if not record.get("blind_results"):
                return _blind_primary(runtime, task_id)
            _blind_one(runtime, task_id, budget)
            return True
        raise E3LiveError(f"unknown candidate stage: {stage}")

    def next_stage(task_id: str) -> str | None:
        with runtime.lock:
            return _next_stage(runtime, task_id)

    def can_dispatch(task_id: str, stage: str) -> bool:
        # The E2 helper scans every record in this slate. A later selected
        # slate may be registered concurrently, so hold the runtime lock while
        # iterating the shared semantic-record mapping.
        with runtime.lock:
            return _can_dispatch(runtime, task_id, stage)

    def reconcile(task_id: str, stage: str) -> str:
        with runtime.lock:
            return _reconcile_active(runtime, task_id, stage)

    scheduler = PersistentCandidateSchedulerV2(
        state_path=runtime.run_dir
        / "candidate_dependency_states"
        / f"slate-{slate:02d}.json",
        candidate_ids=task_ids,
        workers=8,
        next_stage=next_stage,
        can_dispatch=can_dispatch,
        run_stage=run_stage,
        reconcile_active=reconcile,
        max_downstream_before_new_plan=2,
    )
    state = scheduler.run()
    validation = validate_dependency_events_v2(state)
    if not validation["valid"]:
        raise E3LiveError(
            f"slate {slate} candidate dependency graph failed: {validation['errors']}"
        )
    return state


def _implementation_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["_repository_root"])
    relatives = (
        "arcgen.lib/e3_live_overlap.py",
        "arcgen.lib/e1_selection_cohort.py",
        "arcgen.lib/e2_live_streaming.py",
        "arcgen.lib/e2_live_streaming_v2.py",
        "arcgen.lib/e2_streaming_scheduler_v2.py",
        "arcgen.lib/e2_attempt_ledger_v2.py",
        "arcgen.lib/e3_overlap_replay_v2.py",
        "arcgen.lib/e8_plan_contradictions.py",
        "arcgen.lib/v41_experiments.py",
        "arcgen.lib/v41_supported_departure.py",
        "arcgen.lib/v41_test_challenge.py",
        "arcgen.lib/v48_evidence_scale_paired.py",
    )
    return {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "files": {relative: _sha(root / relative) for relative in relatives},
        "config": {
            "path": config["_e3_config_path"],
            "sha256": _sha(config["_e3_config_path"]),
        },
        "selection_config": {
            "path": config["_e3_selection_config_path"],
            "sha256": _sha(config["_e3_selection_config_path"]),
        },
        "one_variable": "overlap later-slate selection with unchanged candidate development",
        "e8_shadow_non_gating": True,
    }


def _checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    return value if value.get("selection_complete") is True else None


def _e8_report(runtime: V41Runtime) -> dict[str, Any]:
    rows = []
    shadows = runtime.state.get("e8_prospective_shadow") or {}
    for task_id, record in sorted(runtime.state["semantic_records"].items()):
        shadow = deepcopy(shadows.get(task_id) or {})
        flagged = shadow.get("would_abort") is True
        late_failure = record.get("status") in {
            "realization_failure",
            "realization_operational_failure",
            "rejected_technical",
            "technical_operational_failure",
        }
        good = record.get("intrinsic_quality") in {"core_quality", "reserve_quality"}
        rows.append(
            {
                "task_id": task_id,
                "slate_index": record["slate_index"],
                "would_abort": flagged,
                "audit_operational_failure": shadow.get(
                    "audit_operational_failure", False
                ),
                "control_terminal_status": record.get("status"),
                "control_intrinsic_quality": record.get("intrinsic_quality"),
                "observed_late_failure": late_failure,
                "would_suppress_core_or_reserve": bool(flagged and good),
                "shadow": shadow,
            }
        )
    flagged = [row for row in rows if row["would_abort"]]
    late = [row for row in rows if row["observed_late_failure"]]
    return {
        "schema_version": 1,
        "pipeline": "arcgen-e8-prospective-shadow-inside-e3",
        "completed": True,
        "model_calls": 0,
        "non_gating": True,
        "automatic_abort_enabled": False,
        "summary": {
            "ready_or_aborted_plan_count": len(rows),
            "flagged_count": len(flagged),
            "observed_late_failure_count": len(late),
            "flagged_late_failure_count": sum(
                row["observed_late_failure"] for row in flagged
            ),
            "core_or_reserve_false_suppression_count": sum(
                row["would_suppress_core_or_reserve"] for row in rows
            ),
            "audit_operational_failure_count": sum(
                row["audit_operational_failure"] for row in rows
            ),
        },
        "cases": rows,
        "automatic_promotion_or_commit": False,
    }


def _attempt_span(*roots: Path) -> dict[str, Any]:
    rows = []
    seen: set[Path] = set()
    for root in roots:
        for path in root.rglob("attempt-*.timing.json"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            value = read_json(path)
            queued = value.get("queued_at_epoch_seconds")
            completed = value.get("completed_at_epoch_seconds")
            if isinstance(queued, (int, float)) and isinstance(completed, (int, float)):
                rows.append(
                    {
                        "path": str(resolved),
                        "sha256": _sha(path),
                        "queued": float(queued),
                        "completed": float(completed),
                        "outcome": str(value.get("outcome") or "success"),
                    }
                )
    if not rows:
        raise E3LiveError("E3 run has no physical attempt timings")
    return {
        "physical_attempt_count": len(rows),
        "failed_attempt_count": sum(row["outcome"] != "success" for row in rows),
        "first_queued_at_epoch_seconds": min(row["queued"] for row in rows),
        "last_completed_at_epoch_seconds": max(row["completed"] for row in rows),
        "call_span_seconds": round(
            max(row["completed"] for row in rows) - min(row["queued"] for row in rows),
            6,
        ),
        "sources": [{"path": row["path"], "sha256": row["sha256"]} for row in rows],
    }


def _timing_analysis(
    *,
    selection_manifest: Mapping[str, Any],
    runtime: V41Runtime,
    selection_root: Path,
    selection_cap: int,
) -> dict[str, Any]:
    attempts = build_attempt_ledger(runtime.state, run_dir=runtime.run_dir)
    board = _board_tail(runtime.run_dir)
    variants = {}
    for name, production_equivalent in (
        ("qualification_actual_15_call_supply", False),
        ("production_equivalent_12_call_supply", True),
    ):
        (
            selection_nodes,
            selection_deps,
            globals_by_slate,
            supply_by_slate,
            provenance,
            selection_totals,
        ) = _selection_nodes(
            selection_manifest, production_equivalent=production_equivalent
        )
        development_nodes, development_deps, _first, terminal, downstream_totals = (
            _downstream_nodes(
                {"scheduler": {"attempt_accounting": attempts}}, globals_by_slate
            )
        )
        nodes = {**selection_nodes, **development_nodes}
        base_deps = {**selection_deps, **development_deps}
        slates = sorted(globals_by_slate)
        serial_deps = {key: set(value) for key, value in base_deps.items()}
        for position, slate in enumerate(slates[1:], start=1):
            previous = slates[position - 1]
            for node_id in supply_by_slate[slate]:
                serial_deps[node_id] = set(terminal.get(previous, []))
        serial = _with_board_tail(
            _schedule(nodes, serial_deps, workers=8, selection_cap=8), board
        )
        modeled_overlap = _with_board_tail(
            _schedule(
                nodes,
                base_deps,
                workers=8,
                selection_cap=selection_cap,
                codex_selection_cap=4,
                claude_selection_cap=4,
            ),
            board,
        )
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
            "fully_serial_slate_factory": serial,
            "modeled_overlap": modeled_overlap,
            "modeled_reduction_fraction": round(
                1
                - modeled_overlap["end_to_end_seconds"] / serial["end_to_end_seconds"],
                6,
            ),
        }
    measured = _attempt_span(selection_root, runtime.run_dir)
    actual = variants["qualification_actual_15_call_supply"]
    measured["reduction_vs_exact_call_serial_counterfactual_fraction"] = round(
        1
        - measured["call_span_seconds"]
        / actual["fully_serial_slate_factory"]["end_to_end_seconds"],
        6,
    )
    measured["difference_from_modeled_overlap_seconds"] = round(
        measured["call_span_seconds"] - actual["modeled_overlap"]["end_to_end_seconds"],
        6,
    )
    return {
        "schema_version": 1,
        "measured_live_overlap": measured,
        "variants": variants,
        "board_tail": board,
        "attempt_complete": attempts["summary"]["coverage_complete"] is True,
    }


def validate(config: Mapping[str, Any]) -> dict[str, Any]:
    selection = config["_e3_selection_config"]
    return {
        "valid": True,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "slate_count": int(selection["cohort"]["slate_count"]),
        "global_workers": 8,
        "selection_cap_during_development": 6,
        "minority_opus_supply": True,
        "all_downstream_roles_sol": True,
        "editorial_boards_retained": True,
        "e8_shadow_non_gating": True,
        "automatic_task_commit": False,
        "production_state_mutation_allowed": False,
    }


def run(config: Mapping[str, Any], *, authorize_model_calls: bool) -> dict[str, Any]:
    if not authorize_model_calls and not config["run"]["mock"]:
        raise E3LiveError("live E3 requires --authorize-model-calls")
    output = Path(config["_e3_output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.is_file():
        report = read_json(report_path)
        if report.get("completed") is not True:
            raise E3LiveError("existing E3 report is incomplete")
        return report
    started = time.time()
    protected = _protected_hashes(config)
    snapshot = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "config": {
            key: deepcopy(value)
            for key, value in config.items()
            if not key.startswith("_")
        },
        "config_path": config["_e3_config_path"],
        "config_sha256": _sha(config["_e3_config_path"]),
        "protected_production_state": protected,
        "automatic_task_commit": False,
        "production_state_mutation_allowed": False,
    }
    _write_immutable(
        output / "e3_config.snapshot.json", snapshot, label="E3 config snapshot"
    )
    _write_immutable(
        output / "e3_implementation_manifest.json",
        _implementation_manifest(config),
        label="E3 implementation manifest",
    )
    pool = SharedCallPool(
        output / "shared_call_limiter.json",
        workers=8,
        selection_cap=6,
        codex_cap=4,
        claude_cap=4,
    )
    runtime = V41Runtime(
        deepcopy(dict(config)),
        {
            "schema_version": 1,
            "pipeline": PIPELINE,
            "scheduling_only_change": True,
            "automatic_task_commit": False,
            "e8_shadow_non_gating": True,
        },
    )
    runtime.invoker = _AttemptPreservingInvoker(
        _LimiterInvoker(runtime.invoker, pool, "downstream"), runtime.run_dir
    )
    _install_v44_extensions(runtime)
    _install_evidence_extension(runtime)
    reservations = ReservationLedger(
        output / "in_flight_reservations.json", run_id=str(config["run"]["id"])
    )
    selection_config = config["_e3_selection_config"]
    selection_root = Path(selection_config["_output_directory"])
    slate_count = int(selection_config["cohort"]["slate_count"])

    # Patch only the E3 process-local factory symbol. Every newly constructed
    # selection runtime receives the same pool; no frozen module is edited.
    from . import e1_selection_cohort as selection_module

    original_runtime_factory = selection_module._v46_runtime

    def shared_runtime_factory(*args: Any, **kwargs: Any) -> Any:
        value = original_runtime_factory(*args, **kwargs)
        value.codex = _LimiterInvoker(value.codex, pool, "selection_codex")
        value.claude = _LimiterInvoker(value.claude, pool, "selection_claude")
        return value

    selection_module._v46_runtime = shared_runtime_factory  # type: ignore[attr-defined]
    scheduler_states: dict[int, dict[str, Any]] = {}
    registered: set[int] = set()
    futures: dict[int, Future[dict[str, Any]]] = {}
    try:
        with (
            ThreadPoolExecutor(max_workers=1) as selection_executor,
            ThreadPoolExecutor(max_workers=slate_count) as downstream_executor,
        ):
            selection_future = selection_executor.submit(
                run_selection_cohort,
                selection_config,
                authorize_model_calls=authorize_model_calls,
            )
            while True:
                for slate in range(1, slate_count + 1):
                    if slate in registered:
                        continue
                    checkpoint_path = (
                        selection_root
                        / "selection_slates"
                        / f"slate-{slate:02d}"
                        / "selection_checkpoint.private.json"
                    )
                    checkpoint = _checkpoint(checkpoint_path)
                    if checkpoint is None:
                        continue
                    selected = _selected_from_checkpoint(checkpoint)
                    if not selected:
                        raise E3LiveError(f"selection slate {slate} is empty")
                    reservations.add_slate(slate, checkpoint)
                    task_ids = _add_records(runtime, selected, seed=int(config["seed"]))
                    assignments = _palette_assignments_dynamic(
                        runtime, selected, slate=slate
                    )
                    registered.add(slate)
                    futures[slate] = downstream_executor.submit(
                        _run_slate,
                        runtime,
                        slate=slate,
                        task_ids=task_ids,
                        assignments=assignments,
                    )
                for slate, future in list(futures.items()):
                    if slate in scheduler_states or not future.done():
                        continue
                    scheduler_states[slate] = future.result()
                    reservations.resolve_slate(slate, runtime.state["semantic_records"])
                reservations.heartbeat()
                pool.heartbeat()
                if selection_future.done():
                    selection_manifest = selection_future.result()
                    if len(registered) != slate_count:
                        missing = sorted(set(range(1, slate_count + 1)) - registered)
                        if missing:
                            raise E3LiveError(
                                f"selection completed without checkpoints: {missing}"
                            )
                    if len(scheduler_states) == slate_count:
                        break
                time.sleep(1.0)
    finally:
        selection_module._v46_runtime = original_runtime_factory  # type: ignore[attr-defined]

    if selection_manifest.get("completed") is not True:
        raise E3LiveError("selection cohort did not complete")
    if int(
        (selection_manifest.get("funnel") or {}).get("selected_candidates") or 0
    ) != len(runtime.state["semantic_records"]):
        raise E3LiveError("selection and downstream candidate sets differ")
    post_blind = _load_or_freeze_post_blind(runtime)
    if not (runtime.run_dir / "editorial_board_results.json").is_file():
        _editorial_boards(runtime)
    prompt_blinding = _assert_downstream_prompt_blinding(runtime)
    if not prompt_blinding["passed"]:
        raise E3LiveError("private E3/source labels leaked into downstream prompts")
    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda row: (row["slate_index"], row["source_development_priority"]),
    )
    tasks = _materialize(runtime)
    summary = _safe_summary(records)
    ordered_states = [scheduler_states[index] for index in sorted(scheduler_states)]
    candidate_integrity = _aggregate_scheduler_report(runtime, ordered_states)
    if not candidate_integrity["live_timing_qualified"]:
        raise E3LiveError("candidate graph or attempt accounting is incomplete")
    e8 = _e8_report(runtime)
    atomic_write_json(output / "e8_prospective_shadow_report.json", e8)
    timing = _timing_analysis(
        selection_manifest=selection_manifest,
        runtime=runtime,
        selection_root=selection_root,
        selection_cap=6,
    )
    pool_state = pool.finish()
    if int(pool_state.get("peak_active") or 0) > 8:
        raise E3LiveError("shared provider concurrency exceeded eight")
    reservations.heartbeat()
    if any(row["state"] != "terminal" for row in reservations.state["leases"].values()):
        raise E3LiveError("one or more E3 reservations are not terminal")
    _assert_hashes_unchanged(protected, _protected_hashes(config))
    measured_reduction = timing["measured_live_overlap"][
        "reduction_vs_exact_call_serial_counterfactual_fraction"
    ]
    recommendation = (
        "candidate_for_adoption"
        if measured_reduction >= 0.20
        and timing["attempt_complete"]
        and prompt_blinding["passed"]
        else "continue_testing"
    )
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "completed": True,
        "summary": summary,
        "tasks": tasks,
        "selection_manifest": {
            "path": str((selection_root / "selected_cohort_manifest.json").resolve()),
            "sha256": _sha(selection_root / "selected_cohort_manifest.json"),
        },
        "post_blind_freeze": {
            "path": str((runtime.run_dir / "post_blind.private.json").resolve()),
            "sha256": _sha(runtime.run_dir / "post_blind.private.json"),
            "record_count": len(post_blind["records"]),
        },
        "candidate_graph_integrity": candidate_integrity,
        "shared_call_limiter": {
            "path": str((output / "shared_call_limiter.json").resolve()),
            "sha256": _sha(output / "shared_call_limiter.json"),
            "peak_active": pool_state["peak_active"],
            "global_worker_limit": 8,
        },
        "reservation_ledger": {
            "path": str(reservations.path.resolve()),
            "sha256": _sha(reservations.path),
            "all_terminal": True,
        },
        "timing": timing,
        "e8_prospective_shadow": {
            "path": str((output / "e8_prospective_shadow_report.json").resolve()),
            "sha256": _sha(output / "e8_prospective_shadow_report.json"),
            "summary": e8["summary"],
            "non_gating": True,
        },
        "prompt_blinding": prompt_blinding,
        "editorial_boards_retained": True,
        "recommendation": recommendation,
        "automatic_task_commit_occurred": False,
        "production_state_mutated": False,
        "production_promotion_occurred": False,
        "coordinator_wall_seconds": round(time.time() - started, 3),
    }
    markdown = "\n".join(
        [
            "# E3 live selection/development-overlap qualification",
            "",
            f"Selected: {summary['selected']}",
            f"Realized: {summary['realized']}",
            f"Technical pass: {summary['technical_pass']}",
            f"Blind clear: {summary['blind_clear']}",
            f"Core / reserve / ambiguous: {summary['core']} / {summary['reserve']} / {summary['ambiguous']}",
            f"Measured call-span reduction vs exact-call serial slate factory: {measured_reduction:.1%}",
            f"Shared peak substantive calls: {pool_state['peak_active']}/8",
            f"E8 shadow flags: {e8['summary']['flagged_count']} (non-gating)",
            f"Recommendation: `{recommendation}`",
            "",
            "Boards were retained. E8 did not abort any task. No task was committed and no production policy was promoted.",
            "",
        ]
    )
    finished = runtime.finish(report, markdown)
    # ``runtime.finish`` writes report.json in the same output directory.
    if finished.get("completed") is not True:
        raise E3LiveError("E3 runtime failed to finalize")
    return finished


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "report"))
    parser.add_argument("--config")
    parser.add_argument("--run")
    parser.add_argument("--authorize-model-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            if not args.run:
                raise E3LiveError("--run is required")
            print(json.dumps(read_json(Path(args.run) / "report.json"), indent=2))
            return 0
        if not args.config:
            raise E3LiveError("--config is required")
        config = load_config(args.config)
        value = (
            validate(config)
            if args.command == "validate"
            else run(config, authorize_model_calls=args.authorize_model_calls)
        )
        print(json.dumps(value, indent=2))
        return 0
    except (ConfigError, E1SelectionCohortError, E3LiveError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PIPELINE",
    "VERSION",
    "E3LiveError",
    "SharedCallPool",
    "_add_records",
    "_selected_from_checkpoint",
    "load_config",
    "run",
    "validate",
]
