"""Compact isolated factory entry point: freeze, execute, prepare, verify, finish.

Historical experiments remain readable and runnable in their original modules.
This coordinator owns only new study directories and never publishes tasks.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

from .e18_artifacts import (
    attach_curriculum_statuses, materialize_terminal_artifacts, render_tasks,
    strict_technical_clear, verify_terminal_artifact_bundle, verify_render_artifacts,
)
from .e18_candidate_runtime import add_records, finalize_reservoir_for_slate, palette_assignments, run_slate
from .e18_evidence_gate import evidence_is_clear
from .e18_pre_e20_pool import DedicatedCallPool
from .factory_config import (
    load_factory_config, resolved_config_manifest, snapshot_explicit_files,
    verify_factory_references, verify_snapshot,
)
from .factory_evidence import apply_final_evidence, recover_terminal_evidence_metadata
from .factory_metrics import cohort_metrics
from .factory_review import FactoryReviews
from .factory_policy import pipeline_policy, review_policy
from .factory_review_policy import verify_recorded_review
from .factory_window import FullWindowSupply, window_bindings
from .factory_runtime import FactoryRuntime
from .factory_contracts import load_contract_bindings
from .factory_supply import FactorySupply
from .util import atomic_write_json, atomic_write_text, content_hash, read_json


@contextmanager
def run_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("this run already has an active coordinator") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def freeze_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """One bounded source/protected-file snapshot, not a historical run scan."""
    root = Path(config["_output_directory"])
    manifest_path = root / "input_manifest.json"
    identity = content_hash(resolved_config_manifest(config))
    if manifest_path.is_file():
        value = read_json(manifest_path)
        if value["config_identity"] != identity:
            raise RuntimeError("run configuration changed; start a new run")
        verify_snapshot(root / "frozen_source", verify_sources=True)
        verify_snapshot(root / "protected_state", verify_sources=True)
        return value
    references = verify_factory_references(config)
    repository = Path(config["_repository_root"])
    explicit = {"config/requested.yaml": config["_config_path"]}
    explicit.update({"references/" + name + Path(row["path"]).suffix: Path(row["path"])
                     for name, row in references.items()})
    # Freeze the imported local implementation closure once. This includes
    # reused schemas/parsers and recipes, not 80,000 historical run artifacts.
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source and Path(source).is_relative_to(repository / "arcgen.lib") and str(source).endswith(".py"):
            explicit[str(Path(source).relative_to(repository))] = Path(source)
    for prompt_path in (repository / "prompts").glob("*.md"):
        explicit[str(prompt_path.relative_to(repository))] = prompt_path
    for name in ("v4_1_control.yaml", "v4_4_functional_scale_fresh_integrated.yaml"):
        explicit["config/" + name] = repository / "config" / name
    source = snapshot_explicit_files(explicit, root / "frozen_source")
    protected = snapshot_explicit_files(
        {name + ".json": config["_paths"][name] for name in (
            "historical_registry", "active_registry", "quarantine_registry", "basic_valid_registry", "production_reservoir")},
        root / "protected_state",
    )
    value = {"schema_version": 1, "run_id": config["run"]["id"], "config_identity": identity,
             "frozen_at": time.time(), "source": source, "protected": protected,
             "config": resolved_config_manifest(config), "automatic_commit": False,
             "technical_boundary_correction_in_both_profiles": True}
    atomic_write_json(manifest_path, value)
    return value


def interval_union(intervals: list[tuple[float, float]]) -> float:
    end, total = 0.0, 0.0
    for start, finish in sorted(intervals):
        if finish >= start:
            total += max(0.0, finish - max(start, end))
            end = max(end, finish)
    return total


def call_accounting(runtimes: list[Any]) -> dict[str, Any]:
    rows, attempts = [], []
    for runtime in runtimes:
        rows.extend(deepcopy(list(runtime.state.get("call_ledger", {}).values())))
        for logical in runtime.invoker.value.get("calls", {}).values():
            directory = runtime.calls_dir / logical["stage"] / logical["call_id"]
            for attempt in logical.get("attempts", []):
                ordinal = int(attempt["attempt"])
                archived = directory / "physical_attempts" / f"attempt-{ordinal:02d}" / "attempt-1.timing.json"
                current = directory / "attempt-1.timing.json"
                path = archived if archived.is_file() else current if ordinal == len(logical["attempts"]) else None
                timing = read_json(path) if path and path.is_file() else None
                attempts.append({**attempt, "logical_id": logical.get("call_id"), "stage": logical.get("stage"),
                                 "namespace": runtime.run_dir.name, "timing": timing})
    role_counts = Counter(row["purpose"] for row in rows)
    active_seconds = 0.0
    queue_seconds = 0.0
    intervals = []
    for row in attempts:
        start, end = row.get("started_at_epoch_seconds"), row.get("completed_at_epoch_seconds")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            intervals.append((start, end))
        timing = row.get("timing") or {}
        active_seconds += float(timing.get("model_execution_seconds", 0) or 0)
        queue_seconds += float(timing.get("queue_wait_seconds", 0) or 0)
    # Provider timing is persisted separately from call wrapper spans. Do not
    # mislabel queue-inclusive journal duration as active model time.
    return {"logical_calls": len(rows), "logical_calls_by_purpose": dict(role_counts),
            "physical_attempts": len(attempts), "retries": sum(max(0, int(row.get("attempts", 0)) - 1) for row in rows),
            "active_model_seconds": active_seconds, "queue_wait_seconds": queue_seconds,
            "model_minutes": active_seconds / 60,
            "provider_token_usage": None,
            "provider_token_usage_note": "The reused Codex CLI success path does not expose billed token accounting. No token savings are inferred from elapsed time.",
            "attempts_without_transport_timing": sum(row["timing"] is None and row.get("cache_source") is None for row in attempts),
            "attempt_span_union_seconds": interval_union(intervals),
            "calls": rows, "attempts": attempts}


class FactoryCoordinator:
    def __init__(self, config: dict[str, Any], manifest: dict[str, Any]) -> None:
        self.config, self.manifest = config, manifest
        self.root = Path(config["_output_directory"])
        self.path = self.root / "factory_state.json"
        self.pool = DedicatedCallPool(self.root / "worker_pool.json")
        self.pipeline_policy = pipeline_policy(config)
        upstream_options = {"stage_bindings": window_bindings()} if self.pipeline_policy else {}
        contract = load_contract_bindings(config)
        contract_options = {"contract_bindings": contract} if contract is not None else {}
        self.upstream = FactoryRuntime(config, manifest, self.pool, namespace="upstream", lane="selection_codex", **upstream_options, **contract_options)
        self.downstream = FactoryRuntime(config, manifest, self.pool, **contract_options)
        supply_type = FullWindowSupply if self.pipeline_policy else FactorySupply
        self.supply = supply_type(config, self.upstream, context_guard=self.upstream.context_guard)
        self.reviews = FactoryReviews(self.downstream, review_policy=review_policy(config))
        self.state = read_json(self.path) if self.path.is_file() else {
            "schema_version": 1, "run_id": config["run"]["id"], "status": "prepared",
            "slates": {}, "sessions": [], "ready_queue_peak": 0,
        }

    def save(self) -> None:
        atomic_write_json(self.path, self.state)

    def _produce(self, slate: int) -> dict[str, Any]:
        supply = self.supply.generate_slate(slate)
        if not supply.get("selection_eligible"):
            return {"slate_index": slate, "source_rows": [], "deferred_rows": [],
                    "terminal_status": "insufficient_supply", "supply_failures": supply.get("supply_failures", [])}
        return self.supply.select_slate(supply, slate)

    def execute(self) -> dict[str, Any]:
        self.supply.freeze_supply_design()
        # Complete the reused stage helper's interrupted metadata-only save
        # before a terminal candidate can be admitted to review. Successful
        # model responses are reused; this does not retry or rejudge a task.
        recover_terminal_evidence_metadata(self.downstream)
        self.state["status"] = "running"
        self.state["sessions"].append({"started_at": time.time(), "status": "running"})
        self.save()
        producer = None
        consumers = {}
        maximum_active = self.config["worker_policy"]["active_downstream_slates"]
        try:
            with ThreadPoolExecutor(max_workers=1) as upstream_executor, ThreadPoolExecutor(max_workers=maximum_active) as downstream_executor:
                while True:
                    if producer and producer[1].done():
                        slate, future = producer
                        result = future.result()
                        self.state["slates"][str(slate)] = {"status": "ready", "selection": result, "ready_at": time.time()}
                        producer = None
                        self.save()
                    for slate, future in list(consumers.items()):
                        if not future.done():
                            continue
                        future.result()
                        row = self.state["slates"][str(slate)]
                        for task_id in row["task_ids"]:
                            apply_final_evidence(self.downstream, task_id)
                        finalize_reservoir_for_slate(self.downstream, self.supply.ledger, row["task_ids"])
                        row.update(status="downstream_complete", completed_at=time.time())
                        del consumers[slate]
                        self.save()
                    ready = [int(key) for key, value in self.state["slates"].items() if value["status"] == "ready"]
                    self.state["ready_queue_peak"] = max(self.state["ready_queue_peak"], len(ready))
                    if len(ready) > self.config["worker_policy"]["ready_queue_hard_maximum"]:
                        raise RuntimeError("ready slate queue exceeded its hard bound")
                    # An interrupted active slate is resumable work, not a
                    # waiting-ready queue entry. Restore it before new work.
                    resumable = sorted(int(key) for key, value in self.state["slates"].items()
                                       if value["status"] == "active" and int(key) not in consumers)
                    for slate in (resumable + sorted(ready))[:maximum_active - len(consumers)]:
                        row = self.state["slates"][str(slate)]
                        selected = row["selection"]["source_rows"]
                        task_ids = add_records(self.config, self.downstream, selected) if selected else []
                        assignments = palette_assignments(self.config, self.downstream, selected, slate=slate) if selected else {}
                        row.update(status="active", task_ids=task_ids, started_at=row.get("started_at", time.time()))
                        self.save()
                        if task_ids:
                            consumers[slate] = downstream_executor.submit(run_slate, self.config, self.downstream,
                                slate=slate, task_ids=task_ids, assignments=assignments)
                        else:
                            row.update(status="downstream_complete", completed_at=time.time())
                            self.save()
                    ready_count = sum(row["status"] == "ready" for row in self.state["slates"].values())
                    remaining = [slate for slate in range(1, self.config["run"]["slates"] + 1)
                                 if str(slate) not in self.state["slates"]]
                    # One producer prepares the next complete frozen slate;
                    # twelve concurrent calls borrow idle capacity only through
                    # the shared 8+4 provider pool.
                    if not producer and remaining and ready_count < self.config["worker_policy"]["ready_queue_target"]:
                        slate = min(remaining)
                        producer = (slate, upstream_executor.submit(self._produce, slate))
                    self.pool.set_downstream_ready(bool(consumers or ready_count or self.reviews.active), reason="factory-coordinator")
                    finished = not producer and not consumers and not remaining and all(row["status"] == "downstream_complete" for row in self.state["slates"].values())
                    reviewed = self.reviews.tick(final=finished)
                    if finished and reviewed:
                        break
                    time.sleep(0.5)
            self.pool.finish()
            self.state["sessions"][-1].update(status="complete", ended_at=time.time())
            self.state["status"] = "awaiting_terminal_verification"
            self.save()
            return self.prepare_and_finish()
        except BaseException as exc:
            self.state["status"] = "incomplete"
            self.state["last_error"] = {"type": type(exc).__name__, "message": str(exc)}
            self.state["sessions"][-1].update(status="interrupted", ended_at=time.time())
            self.save()
            raise
        finally:
            self.reviews.close()

    def prepare_and_finish(self) -> dict[str, Any]:
        if self.pipeline_policy:
            with self.downstream.lock:
                for record in self.downstream.state["semantic_records"].values():
                    record.update(mechanism_audit_status="not_requested", audit_feature_prevalence=None)
                self.downstream.save()
        attach_curriculum_statuses(self.downstream, review_policy=review_policy(self.config))
        artifacts = materialize_terminal_artifacts(self.config, self.downstream)
        renders = render_tasks(self.config, self.downstream)
        records = self.downstream.state["semantic_records"]
        accounting = call_accounting([self.upstream, self.downstream])
        atomic_write_json(self.root / "call_timing_ledger.json", accounting)
        rows = [{"task_id": key, "status": row["status"], "source": row["private_source"],
                 "route": row["private_route"], "intrinsic_quality": row.get("intrinsic_quality"),
                 "portfolio_placement": row.get("portfolio_placement"), "curriculum_status": row.get("curriculum_status"),
                 "evidence": row.get("final_episode_evidence"), "realized": bool(row.get("complete_task")),
                 "technical_clear": strict_technical_clear(row), "evidence_clear": evidence_is_clear(row),
                 "repair_count": row.get("repair_count", 0)} for key, row in sorted(records.items())]
        counts = {"selected": len(rows), "realized": sum(row["realized"] for row in rows),
                  "technical_clear": sum(row["technical_clear"] for row in rows),
                  "evidence_clear": sum(row["evidence_clear"] for row in rows),
                  "core": sum(row["intrinsic_quality"] == "core_quality" for row in rows),
                  "reserve": sum(row["intrinsic_quality"] == "reserve_quality" for row in rows),
                  "basic_valid": sum(row["curriculum_status"] == "basic_valid" for row in rows)}
        report = {"schema_version": 1, "run_id": self.config["run"]["id"], "status": "prepared",
                  "model_profile": self.config["model_profile"], "counts": counts, "tasks": rows,
                  "descriptive_metrics": cohort_metrics(records),
                  "calls": {key: value for key, value in accounting.items() if key not in {"calls", "attempts"}},
                  "slates": self.state["slates"], "worker_pool": self.pool.snapshot(),
                  "review_batches": [{"id": row["id"], "status": row["status"], "failures": row["failures"]} for row in self.reviews.state["batches"]],
                  "wall_sessions": self.state["sessions"], "artifacts": artifacts, "renders": renders,
                  "tasks_committed": 0, "production_promoted": False, "protected_state_unchanged": None}
        if self.pipeline_policy:
            report["pipeline_policy"] = deepcopy(self.pipeline_policy)
            report["omitted_stages"] = ["shard_selection", "native_global_selection", "editorial_board_2", "mechanism_audits"]
            report["mechanism_audit_status"] = "not_requested"
            report["audit_feature_prevalence"] = None
        atomic_write_json(self.root / "prepared_report.json", report)
        verification = verify_prepared(self.config, report, records)
        report.update(status="complete", terminal_verification=verification, protected_state_unchanged=True)
        atomic_write_json(self.root / "report.json", report)
        atomic_write_text(self.root / "report.md", render_report(report))
        # The authoritative complete marker is written LAST, after independent
        # verification. An interruption before this point simply verifies again.
        atomic_write_json(self.root / "complete.json", {"report_hash": content_hash(report), "verified_at": time.time(), "tasks_committed": 0})
        self.state["status"] = "complete"
        self.save()
        return report


def verify_prepared(config: dict[str, Any], report: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_output_directory"])
    verify_snapshot(root / "protected_state", verify_sources=True)
    verify_snapshot(root / "frozen_source", verify_sources=True)
    if report["tasks_committed"] or report["production_promoted"]:
        raise RuntimeError("experimental factory cannot publish tasks or promote itself")
    for row in report["artifacts"]["rows"]:
        verify_terminal_artifact_bundle(config, row, records[row["task_id"]])
    artifact_ids = [row["task_id"] for row in report["artifacts"]["rows"]]
    if len(artifact_ids) != len(set(artifact_ids)) or set(artifact_ids) != set(records):
        raise RuntimeError("prepared artifact set differs from terminal candidates")
    verify_render_artifacts(config, report["renders"], SimpleNamespace(state={"semantic_records": records}))
    pool = read_json(root / "worker_pool.json")
    if pool["active_calls"] or pool["peak_active"] > 12 or pool["peak_downstream"] > 8:
        raise RuntimeError("nonterminal or invalid provider pool")
    reservoir = read_json(root / "reservoir.json")
    if reservoir.get("reservations"):
        raise RuntimeError("in-flight reservations remain at terminal verification")
    queue = read_json(root / "downstream" / "reviews" / "queue.json")
    if queue["waiting"] or any(not row["status"].startswith("complete") for row in queue["batches"]):
        raise RuntimeError("editorial queue has unfinished work")
    for namespace in ("upstream", "downstream"):
        journal = read_json(root / namespace / "attempt_journal.json")
        if any(row.get("status") not in {"complete", "failed"} or any(
            attempt.get("outcome") == "started" for attempt in row.get("attempts", [])
        ) for row in journal.get("calls", {}).values()):
            raise RuntimeError("attempt journal has unfinished logical/provider calls")
    policy = review_policy(config)
    review_runtime = SimpleNamespace(state=read_json(root / "downstream/state.json")) if policy else None
    for row in records.values():
        if policy and row.get("status") == "editorial_complete":
            verify_recorded_review(config, review_runtime, row)
        if evidence_is_clear(row) and row.get("final_episode_evidence", {}).get("status") not in {"evidence_clear", "evidence_clear_after_review"}:
            raise RuntimeError("unbound clear evidence in prepared output")
    return {"verified": True, "candidate_bundles": len(records), "protected_files_unchanged": True,
            "no_live_reservations": True, "no_unfinished_review_groups": True}


def render_report(report: dict[str, Any]) -> str:
    lines = [f"# Factory qualification — {report['run_id']}", "", f"Status: {report['status']}", "",
             "Experimental only. No tasks committed and no production promotion.", "", "## Outcomes", ""]
    lines.extend(f"- {key}: {value}" for key, value in report["counts"].items())
    lines += ["", "Detailed task records, episode-bound evidence, calls/timings, frozen source,",
              "reservoir, queue and worker accounting are retained alongside report.json.", "",
              "This run alone does not establish Astra superiority or optimal role efforts."]
    return "\n".join(lines) + "\n"


def run(config: dict[str, Any], *, authorize_model_calls: bool = False) -> dict[str, Any]:
    if not config["run"]["mock"] and not authorize_model_calls:
        raise RuntimeError("live runs require --authorize-model-calls")
    if "contract_profile" in config:
        load_contract_bindings(config)  # Include selected contract modules in the frozen import closure.
    root = Path(config["_output_directory"])
    with run_lock(root):
        if (root / "complete.json").is_file():
            freeze_inputs(config)  # Identity/source check only; no runtime construction.
            report = read_json(root / "report.json")
            if read_json(root / "complete.json")["report_hash"] != content_hash(report):
                raise RuntimeError("completed report changed")
            records = read_json(root / "downstream" / "state.json")["semantic_records"]
            verify_prepared(config, report, records)
            return report
        manifest = freeze_inputs(config)
        return FactoryCoordinator(config, manifest).execute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "status", "report"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--authorize-model-calls", action="store_true")
    args = parser.parse_args(argv)
    config = load_factory_config(args.config)
    root = Path(config["_output_directory"])
    if args.command == "validate":
        verify_factory_references(config)
        load_contract_bindings(config)
        value = {"valid": True, "run_id": config["run"]["id"], "profile": config["model_profile"], "calls_made": 0}
    elif args.command == "run":
        report = run(config, authorize_model_calls=args.authorize_model_calls)
        value = {"run_id": report["run_id"], "status": report["status"], "counts": report["counts"]}
    else:
        path = root / ("report.json" if args.command == "report" else "factory_state.json")
        value = read_json(path) if path.is_file() else {"status": "not_started", "run_id": config["run"]["id"]}
        if args.command == "status" and "slates" in value:
            value = {"run_id": value["run_id"], "status": value["status"], "slates": {key: row["status"] for key, row in value["slates"].items()}, "last_error": value.get("last_error")}
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
