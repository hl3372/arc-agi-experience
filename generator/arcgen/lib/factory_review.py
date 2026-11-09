"""Bounded, resumable comparative reviews, separate from candidate execution.

Only frozen packages and allowed editorial fields cross this boundary. Reuse
the existing board/audit payloads and judgment policy, not the forensic replay
coordinator. Freeze one finite completed cohort before any review dispatch;
stable call IDs and ordinary successful-call caching handle resume.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
import time
from typing import Any

from .e18_evidence_gate import evidence_is_clear
from .e18_pre_e20_deferred import (
    _audit_call, _audit_specs, _board_call, _board_updates, freeze_board_packages,
)
from .factory_evidence import apply_final_evidence
from .factory_policy import validate_review_policy
from .factory_review_policy import board_binding, evidence_hash
from .util import atomic_write_json, content_hash, read_json


PROTECTED_FIELDS = (
    "task_id", "complete_task", "concrete_plan", "realization", "answer_evidence",
    "technical_results", "blind_results",
)


def _protected_records(records: Mapping[str, Any]) -> str:
    return content_hash({key: {field: row.get(field) for field in PROTECTED_FIELDS}
                         for key, row in sorted(records.items())})


def _packageable(record: Mapping[str, Any]) -> bool:
    task = record.get("complete_task")
    realization = record.get("realization")
    metadata = realization.get("authoring_metadata") if isinstance(realization, Mapping) else None
    return (isinstance(task, Mapping) and isinstance(task.get("train"), list)
            and isinstance(task.get("test"), list) and isinstance(metadata, Mapping)
            and isinstance(metadata.get("intended_rule"), str) and bool(metadata["intended_rule"].strip())
            and isinstance(record.get("concrete_plan"), Mapping))


class FactoryReviews:
    def __init__(self, runtime: Any, *, maximum_groups: int = 4, audit_workers: int = 8,
                 group_size: int = 6, maximum_wait_seconds: float = 300,
                 portfolio_digest: Mapping[str, Any] | None = None, review_policy: Mapping[str, Any] | None = None) -> None:
        if group_size != 6 or maximum_groups != 4 or audit_workers != 8:
            raise ValueError("review policy requires legacy six-task groups, four groups and eight audit workers")
        self.review_policy = validate_review_policy(review_policy)
        self.runtime, self.group_size = runtime, group_size
        self.portfolio_digest = deepcopy(dict(portfolio_digest)) if portfolio_digest is not None else None
        self.maximum_groups, self.maximum_wait = maximum_groups, maximum_wait_seconds
        self.root = runtime.run_dir / "reviews"
        self.path = self.root / "queue.json"
        self.state = read_json(self.path) if self.path.is_file() else {
            "schema_version": 1, "waiting": [], "seen": [], "batches": [],
        }
        self.state.setdefault("skipped_diagnostics", {})
        self.board_executor = ThreadPoolExecutor(max_workers=maximum_groups * (1 if self.review_policy else 2))
        self.audit_executor = ThreadPoolExecutor(max_workers=audit_workers)
        self.active: dict[str, Any] = {}

    def save(self) -> None:
        atomic_write_json(self.path, self.state)

    def _view(self, batch: dict[str, Any], *, board_only: bool = False) -> Any:
        # A frozen view for pure historical package constructors. It is not a
        # runtime, cannot invoke models, and never aliases mutable candidates.
        return SimpleNamespace(
            config=self.runtime.config, run_dir=self.root / batch["id"],
            state={"semantic_records": {key: deepcopy(row) for key, row in sorted(batch["records"].items())
                   if not board_only or (row.get("status") == "blind_pass" and evidence_is_clear(row))}},
        )

    def _prepare(self, task_ids: list[str]) -> dict[str, Any]:
        # Canonical input order precedes the unchanged seeded legacy helpers.
        # In particular, audit peers must not inherit candidate arrival order.
        task_ids = sorted(task_ids)
        batch_id = "group-" + content_hash(task_ids)[:14]
        directory = self.root / batch_id
        with self.runtime.lock:
            records = {key: deepcopy(self.runtime.state["semantic_records"][key]) for key in task_ids}
        prepared_path = directory / "prepared.json"
        input_path = directory / "inputs.private.json"
        if prepared_path.is_file():
            prepared = read_json(prepared_path)
            if (prepared.get("id") != batch_id or prepared.get("task_ids") != task_ids
                    or _protected_records(prepared["records"]) != _protected_records(records)
                    or prepared.get("prepared_hash") != content_hash({key: value for key, value in prepared.items() if key != "prepared_hash"})):
                raise RuntimeError("prepared review membership or episode changed")
            if prepared.get("review_policy") != self.review_policy:
                raise RuntimeError("frozen review policy changed")
            if self.review_policy and any(evidence_hash(prepared["records"][key]) != evidence_hash(row) for key, row in records.items()):
                raise RuntimeError("prepared review final evidence changed")
            if self.portfolio_digest is not None and any(
                spec["values"]["PORTFOLIO_DIGEST"] != self.portfolio_digest
                for group in prepared["board_groups"] for spec in group["calls"]
            ):
                raise RuntimeError("frozen portfolio context changed")
            return prepared
        if input_path.is_file():
            batch = read_json(input_path)
            if batch.get("task_ids") != task_ids or _protected_records(batch["records"]) != _protected_records(records):
                raise RuntimeError("partially prepared review membership or episode changed")
        else:
            batch = {"id": batch_id, "task_ids": task_ids, "records": records, "status": "prepared",
                     "prepared_at": time.time(), "results": {}, "failures": {}}
            atomic_write_json(input_path, batch)
        view = self._view(batch)
        board_view = self._view(batch, board_only=True)
        options = {"board_indices": (1,)} if self.review_policy else {}
        if self.portfolio_digest is not None:
            options["portfolio_digest"] = self.portfolio_digest
        board_package = freeze_board_packages(board_view, **options)
        if self.review_policy:
            batch["review_policy"] = deepcopy(self.review_policy)
        batch["board_groups"] = board_package["groups"]
        # The legacy package file is the reproducible construction template.
        # prepared.json below freezes the actual uniquely named factory calls.
        for group in batch["board_groups"]:
            for spec in group["calls"]:
                call_id = batch_id + "-" + spec["call_id"]
                spec["call_id"] = call_id
                spec["values"]["BOARD_ID"] = call_id
        audits = [] if self.review_policy else _audit_specs(view)
        # Old audit IDs are ordinal per invocation. Prefix them here at the
        # package boundary (including expected output IDs), never monkeypatch
        # runtime.call or model-visible authoring prompts.
        for spec in audits:
            call_id = batch_id + "-" + spec["call_id"]
            spec["call_id"] = call_id
            spec["values"]["AUDIT_ID"] = call_id
            spec["expected"]["audit_id"] = call_id
        batch["audit_specs"] = audits
        # Complete standalone package is the inexpensive recovery checkpoint.
        batch["prepared_hash"] = content_hash(batch)
        atomic_write_json(view.run_dir / "prepared.json", batch)
        return batch

    def _specs(self, batch: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return [("board", spec) for group in batch["board_groups"] for spec in group["calls"]] + [
            ("audit", spec) for spec in batch["audit_specs"]
        ]

    def _dispatch(self, batch: dict[str, Any]) -> bool:
        changed = batch["status"] != "running"

        def submit(kind: str, spec: dict[str, Any]) -> None:
            nonlocal changed
            key = spec["call_id"]
            executor = self.board_executor if kind == "board" else self.audit_executor
            function = _board_call if kind == "board" else _audit_call
            directory = self.root / batch["id"] / kind
            self.active[key] = (batch["id"], kind, executor.submit(function, self.runtime, directory, spec))
            changed = True

        terminal = batch["results"].keys() | batch["failures"].keys()
        active_groups = {
            (other["id"], group["group_index"])
            for other in self.state["batches"] for group in other["board_groups"]
            if any(spec["call_id"] in self.active for spec in group["calls"])
        }
        for group in batch["board_groups"]:
            pending = [spec for spec in group["calls"] if spec["call_id"] not in terminal]
            group_key = (batch["id"], group["group_index"])
            if not pending or (group_key not in active_groups and len(active_groups) >= self.maximum_groups):
                continue
            active_groups.add(group_key)
            for spec in pending:
                if spec["call_id"] not in self.active:
                    submit("board", spec)
        audit_active = sum(kind == "audit" for _, kind, _ in self.active.values())
        for spec in batch["audit_specs"]:
            if audit_active >= 8:
                break
            if spec["call_id"] not in terminal and spec["call_id"] not in self.active:
                submit("audit", spec)
                audit_active += 1
        batch["status"] = "running"
        return changed

    def _complete(self, batch: dict[str, Any]) -> None:
        updates = {}
        for group in batch["board_groups"]:
            if any(row["call_id"] in batch["failures"] for row in group["calls"]):
                continue
            by_key = {(row["group_index"], row["board_index"]): batch["results"][row["call_id"]]
                      for row in group["calls"]}
            if self.review_policy:
                result = by_key[(group["group_index"], 1)]
                updates.update({key: board_binding(self.review_policy, batch, group, result, key)
                                for key in group["task_ids"]})
            else:
                updates.update(_board_updates(group, by_key))
        board_task_ids = {key for group in batch["board_groups"] for key in group["task_ids"]}
        with self.runtime.lock:
            # Validate the entire cohort before applying any editorial update.
            for task_id, frozen in batch["records"].items():
                current = self.runtime.state["semantic_records"][task_id]
                if self.review_policy and evidence_hash(current) != evidence_hash(frozen):
                    raise RuntimeError("reviewed final evidence changed: " + task_id)
                for name in PROTECTED_FIELDS:
                    if current.get(name) != frozen.get(name):
                        raise RuntimeError(f"reviewed episode changed: {task_id}:{name}")
            for task_id in sorted(batch["records"]):
                current = self.runtime.state["semantic_records"][task_id]
                if task_id in updates:
                    if not evidence_is_clear(current):
                        raise RuntimeError("editorial update cannot clear held evidence")
                    current.update(updates[task_id], status="editorial_complete")
                elif task_id in board_task_ids:
                    current.update(status="editorial_operational_failure", editorial_failure="required board call failed")
                current["review_batch_id"] = batch["id"]
                if self.review_policy:
                    current["mechanism_audit_status"] = "not_requested"
                    current["audit_feature_prevalence"] = None
                current["mechanism_audits"] = {
                    result["family"]: next(item for item in result["response"]["judgments"] if item["task_id"] == task_id)
                    for spec in batch["audit_specs"]
                    if spec["call_id"] in batch["results"] and task_id in spec["task_ids"]
                    for result in [batch["results"][spec["call_id"]]]
                }
                # Audit failures affect research completeness, never task quality.
                current["mechanism_audit_failures"] = [spec["call_id"] for spec in batch["audit_specs"]
                    if spec["call_id"] in batch["failures"] and task_id in spec["task_ids"]]
            self.runtime.save()
        batch["status"] = "complete_with_failures" if batch["failures"] else "complete"
        batch["completed_at"] = time.time()
        atomic_write_json(self.root / batch["id"] / "report.json", batch)

    def _stage_active(self, task_id: str, record: Mapping[str, Any]) -> bool:
        slate = record.get("slate_index")
        if slate is None:
            return False
        path = self.runtime.run_dir / "candidate_dependency_states" / f"slate-{int(slate):02d}.json"
        if not path.is_file():
            return False
        row = read_json(path).get("candidates", {}).get(task_id, {})
        return row.get("active_stage") is not None

    def tick(self, *, final: bool = False) -> bool:
        """Freeze the final cohort, then run bounded reviews without regrouping."""
        if not final:
            return False
        changed = False
        with self.runtime.lock:
            candidates = sorted(self.runtime.state["semantic_records"])
            if "cohort_task_ids" not in self.state:
                if any(self._stage_active(key, self.runtime.state["semantic_records"][key]) for key in candidates):
                    return False
                self.state["cohort_task_ids"] = candidates
                changed = True
            elif self.state["cohort_task_ids"] != candidates:
                raise RuntimeError("final review cohort membership changed; use a new run")
        for task_id in candidates:
            if task_id in self.state["seen"]:
                continue
            apply_final_evidence(self.runtime, task_id)
            with self.runtime.lock:
                current = self.runtime.state["semantic_records"][task_id]
                if _packageable(current):
                    kind = "editorial" if current.get("status") == "blind_pass" and evidence_is_clear(current) else "audit_only"
                    self.state["waiting"].append({"task_id": task_id, "kind": kind})
                else:
                    reason = "final_realization_missing_review_package_fields"
                    self.state["skipped_diagnostics"][task_id] = {"reason": reason, "candidate_status": current.get("status")}
                    current["review_diagnostic"] = {"status": "not_reviewable", "reason": reason}
                    self.runtime.save()
            self.state["seen"].append(task_id)
            changed = True
        if changed:
            self.save()  # Freeze full membership before package construction.
            changed = False
        if self.state["waiting"]:
            batch = self._prepare([row["task_id"] for row in self.state["waiting"]])
            self.state["batches"].append(batch)
            self.state["waiting"] = []
            self.save()  # Persist the actual packages before the first call.
        by_id = {row["id"]: row for row in self.state["batches"]}
        for key, (batch_id, kind, future) in sorted(self.active.items()):
            if not future.done():
                continue
            batch = by_id[batch_id]
            try:
                batch["results"][key] = future.result()
            except Exception as exc:
                batch["failures"][key] = {"kind": kind, "error": str(exc), "error_type": type(exc).__name__}
            del self.active[key]
            changed = True
        for batch in self.state["batches"]:
            if batch["status"].startswith("complete"):
                continue
            wanted = {spec["call_id"] for _, spec in self._specs(batch)}
            if wanted <= batch["results"].keys() | batch["failures"].keys():
                self._complete(batch)
                changed = True
            else:
                changed = self._dispatch(batch) or changed
        if changed:
            self.save()
        unfinished = any(not row["status"].startswith("complete") for row in self.state["batches"])
        return not self.active and not self.state["waiting"] and not unfinished

    def close(self) -> None:
        self.board_executor.shutdown(wait=True)
        self.audit_executor.shutdown(wait=True)
