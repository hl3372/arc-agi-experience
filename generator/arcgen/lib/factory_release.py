"""Run the factory, then optionally publish through the existing commit transaction.

The generation coordinator stays isolated. Publication has its own explicit
authorization and receipt; it never rewrites the source report or evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import factory
from .committed_tasks import load_committed_task_registry
from .e18_artifacts import verify_terminal_artifact_bundle
from .e18_commit import (
    collect_commit_eligibility, commit_transaction, load_basic_valid_registry,
    prepare_commit, _verify_prepared,
)
from .e18_evidence_gate import evidence_is_clear
from .factory_config import load_factory_config, resolved_config_manifest, verify_snapshot
from .factory_evidence import project_final_evidence
from .util import atomic_write_json, content_hash, read_json
from .factory_policy import review_policy
from .factory_review_policy import verify_recorded_review


def status(config: dict[str, Any]) -> dict[str, Any]:
    """Cheap monitoring: do not construct runtimes or rehash today's bank."""
    root = Path(config["_output_directory"])

    def saved(relative: str) -> dict[str, Any]:
        path = root / relative
        return read_json(path) if path.is_file() else {}

    state = saved("factory_state.json")
    downstream = saved("downstream/state.json")
    records = downstream.get("semantic_records", {})
    publication = saved("publication_report.json")
    report = saved("report.json")
    marker = saved("complete.json")
    completed = (report.get("status") == "complete"
                 and report.get("terminal_verification", {}).get("verified") is True
                 and marker.get("report_hash") == content_hash(report))
    return {
        "run_id": config["run"]["id"], "account": config["run"]["account"],
        "status": "complete" if completed else state.get("status", "not_started"),
        "last_error": state.get("last_error"),
        "slates": {key: row["status"] for key, row in state.get("slates", {}).items()},
        "candidate_states": dict(Counter(row["status"] for row in records.values())),
        "completed_logical_calls": sum(row.get("status") == "complete" for namespace in ("upstream", "downstream")
                                       for row in saved(namespace + "/state.json").get("call_ledger", {}).values()),
        "active_model_calls": len(saved("worker_pool.json").get("active_calls", {})),
        "final_counts": report.get("counts"),
        "publication_status": publication.get("status", "not_published"),
        "tasks_committed": publication.get("verification", {}).get("tasks_committed", 0),
    }


def _terminal(config: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Read-only verification, also valid after the authorized bank has grown."""
    root = Path(config["_output_directory"])
    marker = read_json(root / "complete.json")
    report = read_json(root / "report.json")
    manifest = read_json(root / "input_manifest.json")
    if (report.get("status") != "complete"
            or not report.get("terminal_verification", {}).get("verified")
            or marker.get("report_hash") != content_hash(report)
            or manifest["config_identity"] != content_hash(resolved_config_manifest(config))):
        raise RuntimeError("factory result is not a verified, unchanged terminal run")
    verify_snapshot(root / "frozen_source")
    verify_snapshot(root / "protected_state")  # Frozen inputs, not today's bank.
    runtime = SimpleNamespace(run_dir=root / "downstream",
                              state=read_json(root / "downstream/state.json"))
    records = runtime.state["semantic_records"]
    artifacts = report["artifacts"]["rows"]
    if len(artifacts) != len(records) or {row["task_id"] for row in artifacts} != set(records):
        raise RuntimeError("factory terminal candidate membership changed")
    calls = runtime.state.get("call_ledger", {})
    calls = list(calls.values()) if isinstance(calls, dict) else calls
    for row in artifacts:
        key = row["task_id"]
        verify_terminal_artifact_bundle(config, row, records[key])
        if review_policy(config) and records[key].get("status") == "editorial_complete":
            verify_recorded_review(config, runtime, records[key])
        if evidence_is_clear(records[key]) and project_final_evidence(key, records[key], calls)["status"] not in {
            "evidence_clear", "evidence_clear_after_review",
        }:
            raise RuntimeError(f"publication lacks final-version evidence: {key}")
    return report, runtime


def _publication_config(config: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    value["commit"] = {
        "authorization_record": authorization["authorization_record"],
        "commit_set_id": config["run"]["id"],
        "committed_on": authorization["committed_on"],
        "source_pipeline": "arcgen-factory",
        "authorize_same_procedure_override": authorization["allow_nonexact_siblings"],
    }
    return value


def _verify_publication(config: dict[str, Any], runtime: Any, prepared: dict[str, Any],
                        result: dict[str, Any]) -> dict[str, Any]:
    """Check this commit's entries/dossiers, not an obsolete total bank count."""
    _verify_prepared(config, runtime, prepared)
    installed = Path(result["installed_commit_manifest"])
    if sha256(installed.read_bytes()).hexdigest() != result["installed_commit_manifest_sha256"]:
        raise RuntimeError("installed publication manifest changed")
    expected = {key for key in prepared["main_task_ids"] + prepared["basic_task_ids"]}
    found: set[str] = set()
    for registry in (
        load_committed_task_registry(config["_paths"]["historical_registry"]),
        load_basic_valid_registry(config["_paths"]["basic_valid_registry"]),
    ):
        for entry in registry["entries"]:
            if entry.get("commit_set_id") != config["commit"]["commit_set_id"]:
                continue
            key = entry["source_candidate_id"]
            if key not in expected or key in found:
                raise RuntimeError("publication membership differs from prepared candidates")
            if read_json(Path(entry["absolute_task_path"])) != runtime.state["semantic_records"][key]["complete_task"]:
                raise RuntimeError(f"published payload differs from verified candidate: {key}")
            found.add(key)
    if found != expected:
        raise RuntimeError("published tasks or dossiers are missing")
    return {"verified": True, "task_ids": sorted(found), "tasks_committed": len(found)}


def preview(config: dict[str, Any]) -> dict[str, Any]:
    """Save eligibility locally; no model calls or shared-state changes."""
    with factory.run_lock(Path(config["_output_directory"])):
        _, runtime = _terminal(config)
        root = Path(config["_output_directory"])
        if (root / "commit_transaction/prepared.json").is_file():
            return read_json(root / "commit_eligibility.json")
        authorization = {"authorization_record": "preview only; no publication authorized",
                         "committed_on": "preview", "allow_nonexact_siblings": False}
        return collect_commit_eligibility(_publication_config(config, authorization), runtime)


def publish(config: dict[str, Any], *, authorize_user_decision: bool = False,
            authorization_record: str = "", allow_nonexact_siblings: bool = False) -> dict[str, Any]:
    if not authorize_user_decision or not authorization_record.strip():
        raise RuntimeError("publication requires explicit user authorization and its record")
    root = Path(config["_output_directory"])
    with factory.run_lock(root):
        report, runtime = _terminal(config)
        path = root / "publication_authorization.json"
        authorization = read_json(path) if path.is_file() else {
            "authorization_record": authorization_record,
            "committed_on": datetime.now(timezone.utc).date().isoformat(),
            "allow_nonexact_siblings": allow_nonexact_siblings,
            "source_report_hash": content_hash(report),
        }
        if (authorization["authorization_record"] != authorization_record
                or authorization["allow_nonexact_siblings"] != allow_nonexact_siblings
                or authorization["source_report_hash"] != content_hash(report)):
            raise RuntimeError("saved publication authorization changed")
        value = _publication_config(config, authorization)
        journal_path = root / "commit_transaction/journal.json"
        if not journal_path.is_file():
            # Before any mutation, rerun the inexpensive normal terminal checks.
            factory.verify_prepared(config, report, runtime.state["semantic_records"])
        if not path.is_file():
            atomic_write_json(path, authorization)
        prepared_path = root / "commit_transaction/prepared.json"
        if prepared_path.is_file():
            prepared = read_json(prepared_path)
            _verify_prepared(value, runtime, prepared)
        else:
            eligibility = collect_commit_eligibility(value, runtime)
            prepared = prepare_commit(value, runtime, eligibility)
        journal = read_json(journal_path) if journal_path.is_file() else {}
        # A completed transaction is checked against its own entries. Unrelated
        # later commits must not make resume fail or replay successful calls.
        result = journal["result"] if journal.get("status") == "complete" else commit_transaction(value, runtime, prepared)
        verification = _verify_publication(value, runtime, prepared, result)
        receipt = {"schema_version": 1, "status": "complete", "run_id": config["run"]["id"],
                   "source_report_hash": content_hash(report), "authorization": authorization,
                   "verification": verification, "transaction_result": result,
                   "production_promoted": False, "model_calls_made": 0}
        if review_policy(config):
            receipt["review_policy"] = review_policy(config)
        receipt_path = root / "publication_report.json"
        if receipt_path.is_file() and read_json(receipt_path) != receipt:
            raise RuntimeError("saved publication receipt changed")
        if not receipt_path.is_file():
            atomic_write_json(receipt_path, receipt)
        return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "preview", "publish"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--authorize-model-calls", action="store_true")
    parser.add_argument("--publish-eligible", action="store_true")
    parser.add_argument("--authorize-user-decision", action="store_true")
    parser.add_argument("--authorization-record", default="")
    parser.add_argument("--allow-nonexact-siblings", action="store_true")
    args = parser.parse_args(argv)
    config = load_factory_config(args.config)
    publishing = args.command == "publish" or args.publish_eligible
    if publishing and (not args.authorize_user_decision or not args.authorization_record.strip()):
        parser.error("publication requires --authorize-user-decision and --authorization-record")
    if args.publish_eligible and args.command != "run":
        parser.error("--publish-eligible belongs to the run command")
    if args.command == "run":
        root = Path(config["_output_directory"])
        if (root / "complete.json").is_file():
            with factory.run_lock(root):
                report, _ = _terminal(config)
        else:
            report = factory.run(config, authorize_model_calls=args.authorize_model_calls)
        result = {"run_id": report["run_id"], "status": report["status"], "counts": report["counts"]}
    elif args.command == "preview":
        result = preview(config)
    elif args.command == "status":
        result = status(config)
    if publishing:
        result = publish(config, authorize_user_decision=args.authorize_user_decision,
                         authorization_record=args.authorization_record,
                         allow_nonexact_siblings=args.allow_nonexact_siblings)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
