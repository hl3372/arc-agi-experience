"""Opt-in, one-purpose Astra effort screens over immutable saved packets.

Prepare one role, then explicitly run it; no command launches all role panels.
This runner collects target-role observations only. It does not develop ideas,
re-verify repairs, choose an effort, publish tasks or count prior calls as free
independent trials. Those separate evaluations remain visible requirements.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .e18_pre_e20_pool import DedicatedCallPool
from .factory import call_accounting, freeze_inputs, run_lock
from .factory_config import load_factory_config, resolved_config_manifest, snapshot_explicit_files, verify_snapshot
from .factory_effort_inputs import ACTIVE_PURPOSES, verify_effort_inputs
from .factory_runtime import FactoryRuntime, UPSTREAM_PURPOSES, build_stage_bindings
from .model import WRAPPER_VERSION
from .util import atomic_write_json, atomic_write_text, content_hash, read_json


EFFORTS = ("high", "xhigh", "max")
ARM = "role-effort-screen"


class FactoryEffortError(RuntimeError):
    """An effort screen is unprepared, incompatible or inconsistent."""


def _purpose(purpose: str) -> None:
    if purpose not in ACTIVE_PURPOSES:
        raise FactoryEffortError(f"not an active substantive purpose: {purpose}")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise FactoryEffortError(f"unsafe effort artifact path: {name}")
    return path


def _immutable(path: Path, value: Any) -> None:
    if path.exists():
        if not path.is_file() or read_json(path) != value:
            raise FactoryEffortError(f"refusing to replace frozen effort input: {path}")
    else:
        atomic_write_json(path, value)


def _config(config: Mapping[str, Any], purpose: str) -> dict[str, Any]:
    _purpose(purpose)
    if config["model_profile"]["model"] != "gpt-6-astra":
        raise FactoryEffortError("effort screens require the Astra reference profile")
    result = deepcopy(dict(config))
    root = Path(config["_output_directory"]) / "effort" / purpose
    result["_output_directory"] = str(root)
    result["_paths"].update(output_directory=root, cache=root / "cache")
    result["paths"].update(output_directory=str(root), cache=str(root / "cache"))
    result["run"]["replicate"] = "effort-" + purpose
    result["_run_identity"]["replicate"] = result["run"]["replicate"]
    return result


def runtime_profile(config: Mapping[str, Any], assignment: Mapping[str, Any]) -> dict[str, Any]:
    """Change only the measured purpose's effort, plus explicit run isolation."""
    result = deepcopy(dict(config))
    purpose, effort = assignment["purpose"], assignment["effort"]
    _purpose(purpose)
    if effort not in EFFORTS:
        raise FactoryEffortError("the frozen initial screen permits high/xhigh/max only")
    account = assignment["account"]
    if account != config["run"]["account"]:
        raise FactoryEffortError("a runtime profile keeps its configured Codex account")
    result["run"].update(account=account, replicate=assignment["namespace"])
    result["model_profile"]["efforts"][purpose] = effort
    result["role_profile"][purpose]["reasoning_effort"] = effort
    result["_profile_hash"] = content_hash(result["model_profile"])
    result["_run_identity"].update(account=account, replicate=assignment["namespace"], profile_hash=result["_profile_hash"])
    cache = Path(result["_output_directory"]) / "cache" / assignment["namespace"]
    result["_paths"]["cache"], result["paths"]["cache"] = cache, str(cache)
    return result


def assign_trials(inputs: Sequence[Mapping[str, Any]], *, purpose: str, seed: int,
                  accounts: Sequence[str] = ("main",)) -> list[dict[str, Any]]:
    """Seeded Latin-order blocks; paired efforts share their block's account.

    Board pairs stay together. Preregistered repeats switch account when two
    are configured and always receive an independent per-input cache namespace.
    """
    _purpose(purpose)
    if not accounts or len(accounts) > 2 or len(set(accounts)) != len(accounts):
        raise FactoryEffortError("choose one or two distinct accounts")
    rows = [deepcopy(dict(row)) for row in inputs if row["purpose"] == purpose]
    if not rows or len({row["input_id"] for row in rows}) != len(rows):
        raise FactoryEffortError("a nonempty unique frozen panel is required for this purpose")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("group_id") or row["input_id"], []).append(row)
    order = sorted(groups, key=lambda key: content_hash({"seed": seed, "purpose": purpose, "input": key}))
    assignments = []
    for repeat in (0, 1):
        for position, group_id in enumerate(order):
            group = sorted(groups[group_id], key=lambda row: (row.get("board_index", 0), row["input_id"]))
            if purpose == "editorial_board" and (len(group) != 2 or {row.get("board_index") for row in group} != {1, 2}):
                raise FactoryEffortError("each editorial input group requires both independent board packets")
            if len({bool(row.get("repeat_selected")) for row in group}) != 1:
                raise FactoryEffortError("both board packets must share preregistered repeat membership")
            if repeat and not group[0].get("repeat_selected"):
                continue
            offset = (position + repeat + seed) % len(EFFORTS)
            for effort in EFFORTS[offset:] + EFFORTS[:offset]:
                for row in group:
                    identity = content_hash({"purpose": purpose, "input_id": row["input_id"]})[:16]
                    namespace = f"effort-{purpose}-{identity}-{effort}-r{repeat}"
                    assignments.append({
                        "assignment_id": namespace, "namespace": namespace, "input_id": row["input_id"],
                        "purpose": purpose, "effort": effort, "model": "gpt-6-astra",
                        "account": accounts[(position + repeat) % len(accounts)], "repeat": repeat,
                        "stage": row["stage"], "call_id": row["call_id"], "parent_ids": row["parent_ids"],
                        "order": len(assignments) + 1,
                    })
    return assignments


def _check_contract(inputs_root: Path, row: Mapping[str, Any], bindings: Mapping[str, Any]) -> None:
    binding = bindings.get(row["stage"])
    if binding is None or row["purpose"] not in binding.purposes:
        raise FactoryEffortError("frozen effort purpose/stage is not registered")
    schema = read_json(_inside(inputs_root, row["schema_path"]))
    if schema != binding.schema or content_hash(schema) != row["schema_hash"]:
        raise FactoryEffortError("frozen response schema differs from the current registered contract")
    if _digest(_inside(inputs_root, row["schema_path"])) != row["schema_sha256"]:
        raise FactoryEffortError("frozen schema artifact changed")
    if _digest(_inside(inputs_root, row["prompt_path"])) != row["prompt_sha256"]:
        raise FactoryEffortError("frozen rendered prompt changed")


def matching_reference(config: Mapping[str, Any], inputs_root: Path,
                       packet: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any] | None:
    """Credit only an exact, successful, same-account base observation.

    This never fills a repeat, creates a provider-call record or supplies new
    timing. The input freezer preserves and validates the original artifacts.
    Missing or incompatible provenance simply earns no credit.
    """
    source = packet.get("source", {})
    if (trial["repeat"] or not source.get("prior_attempt_succeeded")
            or source.get("account") != trial["account"]
            or source.get("model") != trial["model"] or source.get("effort") != trial["effort"]
            or not packet.get("response_path") or not packet.get("request_path")):
        return None
    request = read_json(_inside(inputs_root, packet["request_path"]))
    expected_model = {"provider": config["transport"]["provider"],
                      "name": trial["model"], "reasoning_effort": trial["effort"],
                      "timeout_seconds": config["transport"]["timeout_seconds"], "retries": 0}
    if (request.get("wrapper_version") != WRAPPER_VERSION or request.get("model") != expected_model
            or request.get("stage") != trial["stage"] or request.get("call_id") != trial["call_id"]
            or request.get("parent_ids") != trial["parent_ids"]
            or request.get("schema_hash") != packet["schema_hash"]
            or request.get("prompt_hash") != content_hash(_inside(inputs_root, packet["prompt_path"]).read_bytes())
            or request.get("prompt_template_version") != packet["prompt_template_version"]
            or request.get("prompt_template_hash") != packet["prompt_template_hash"]):
        return None
    path = _inside(inputs_root, packet["response_path"])
    if _digest(path) != source.get("response_sha256"):
        raise FactoryEffortError("saved reference response differs from its frozen provenance")
    response = read_json(path)
    build_stage_bindings()[trial["stage"]].parser(response)
    _validate_response(packet, response)
    return {"status": "reused_reference", "response_hash": content_hash(response),
            "response_path": str(path), "source_account": source["account"],
            "source_request_sha256": source["request_sha256"],
            "structural_outcome": describe_response(response),
            "new_provider_calls": 0, "independent_repeat": False,
            "timing_comparability": "historical_noncontemporaneous_reference_not_new_latency",
            "source_call": {key: source.get(key) for key in ("run_directory", "stage", "call_id")}}


def prepare_effort(config: Mapping[str, Any], inputs_directory: str | Path, *, purpose: str,
                   accounts: Sequence[str] = ("main",),
                   reuse_matching_references: bool = False) -> dict[str, Any]:
    """Freeze exactly one role screen without constructing an invoker."""
    resolved = _config(config, purpose)
    root, inputs_root = Path(resolved["_output_directory"]), Path(inputs_directory).resolve()
    with run_lock(root):
        manifest = verify_effort_inputs(inputs_root)
        selected = [row for row in manifest["inputs"] if row["purpose"] == purpose]
        bindings = build_stage_bindings()
        for row in selected:
            _check_contract(inputs_root, row, bindings)
        assignments = assign_trials(selected, purpose=purpose, seed=int(config["seed"]), accounts=accounts)
        packets = {row["input_id"]: row for row in selected}
        references = {}
        if reuse_matching_references:
            for trial in assignments:
                reference = matching_reference(resolved, inputs_root, packets[trial["input_id"]], trial)
                if reference is not None:
                    references[trial["assignment_id"]] = reference
        frozen = freeze_inputs(resolved)
        if config["run"]["mock_responses"]:
            directory = root / "mock_source"
            if directory.exists():
                verify_snapshot(directory, verify_sources=True)
            else:
                snapshot_explicit_files({"responses.json": Path(config["run"]["mock_responses"])}, directory)
        prepared = {
            "schema_version": 1, "purpose": purpose, "inputs_directory": str(inputs_root),
            "inputs_manifest_hash": manifest["manifest_hash"], "input_manifest": manifest,
            "factory_input_manifest_hash": content_hash(frozen),
            "config_identity": content_hash(resolved_config_manifest(resolved)),
            "accounts": list(accounts), "efforts": list(EFFORTS), "assignments": assignments,
            "target_observations": len(assignments),
            "target_logical_calls": len(assignments) - len(references), "prior_call_credit": len(references),
            "reuse_matching_references": reuse_matching_references, "references": references,
            "automatic_effort_adoption": False, "automatic_task_commit": False,
        }
        prepared["content_hash"] = content_hash(prepared)
        _immutable(root / "prepared.json", prepared)
        if not (root / "call_ledger.private.json").exists():
            _immutable(root / "call_ledger.private.json", {"prepared_hash": prepared["content_hash"], "calls": references})
        return _read_prepared(resolved)


def _read_prepared(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_output_directory"])
    if not (root / "prepared.json").is_file():
        raise FactoryEffortError("prepare this purpose before running it")
    prepared = read_json(root / "prepared.json")
    if prepared["content_hash"] != content_hash({key: value for key, value in prepared.items() if key != "content_hash"}):
        raise FactoryEffortError("frozen effort assignments changed")
    if prepared["config_identity"] != content_hash(resolved_config_manifest(config)):
        raise FactoryEffortError("effort reference configuration changed; use a new run")
    # These paths already exist: reuse the bounded explicit snapshot verifier.
    if content_hash(freeze_inputs(config)) != prepared["factory_input_manifest_hash"]:
        raise FactoryEffortError("factory source/protected snapshot identity changed")
    manifest = verify_effort_inputs(Path(prepared["inputs_directory"]))
    if manifest != prepared["input_manifest"] or manifest["manifest_hash"] != prepared["inputs_manifest_hash"]:
        raise FactoryEffortError("effort input manifest changed")
    if (root / "mock_source").exists():
        verify_snapshot(root / "mock_source", verify_sources=True)
    bindings = build_stage_bindings()
    for row in manifest["inputs"]:
        if row["purpose"] == prepared["purpose"]:
            _check_contract(Path(prepared["inputs_directory"]), row, bindings)
    return prepared


def _validate_response(row: Mapping[str, Any], response: dict[str, Any]) -> None:
    for key, expected in row.get("expected", {}).items():
        if response.get(key) != expected:
            raise FactoryEffortError(f"response has wrong {key}")
    member = row.get("response_membership")
    if member:
        ids = [item[member["id_field"]] for item in response[member["field"]]]
        if len(ids) != len(set(ids)) or set(ids) != set(member["ids"]):
            raise FactoryEffortError("response membership differs from the frozen packet")


def describe_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Report structural/model-reported outcomes, never independent quality."""
    scalars = ("status", "decision", "property_status", "scale_decision")
    lists = ("ideas", "selected", "judgments", "candidate_assessments", "derived_test_outputs",
             "predicted_test_outputs", "issues", "deviations", "changes")
    return {"model_reported": {key: response[key] for key in scalars if key in response},
            "structural_counts": {key: len(response[key]) for key in lists if isinstance(response.get(key), list)},
            "independent_quality_assessment": "not_performed"}


def _saved_runtimes(root: Path, assignments: Sequence[Mapping[str, Any]]) -> list[Any]:
    result = []
    for row in assignments:
        directory = root / row["namespace"]
        if (directory / "state.json").is_file() and (directory / "attempt_journal.json").is_file():
            result.append(SimpleNamespace(run_dir=directory, calls_dir=directory / "model_calls",
                          state=read_json(directory / "state.json"),
                          invoker=SimpleNamespace(value=read_json(directory / "attempt_journal.json"))))
    return result


def _verify_ledger(config: dict[str, Any], prepared: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["_output_directory"])
    ledger = read_json(root / "call_ledger.private.json")
    if ledger["prepared_hash"] != prepared["content_hash"]:
        raise FactoryEffortError("effort ledger belongs to another preparation")
    assignments = {row["assignment_id"]: row for row in prepared["assignments"]}
    inputs = {row["input_id"]: row for row in prepared["input_manifest"]["inputs"]}
    # A branch checkpoints each result before its next call. Reconcile a crash
    # before the aggregate ledger update without constructing completed runtimes.
    for name in assignments:
        checkpoint = root / "trial_results" / f"{name}.json"
        if checkpoint.is_file():
            value = read_json(checkpoint)
            prior = ledger["calls"].get(name)
            if prior and prior["status"] == "complete" and prior != value:
                raise FactoryEffortError("successful trial checkpoint differs from the ledger")
            ledger["calls"][name] = value
    if set(ledger["calls"]) - assignments.keys():
        raise FactoryEffortError("unscheduled effort call in ledger")
    for name, reference in prepared.get("references", {}).items():
        if ledger["calls"].get(name) != reference:
            raise FactoryEffortError("prepared reference cannot be dropped or replaced by a new call")
    bindings = build_stage_bindings()
    for name, saved in ledger["calls"].items():
        if saved["status"] == "reused_reference":
            trial, packet = assignments[name], inputs[assignments[name]["input_id"]]
            expected = matching_reference(config, Path(prepared["inputs_directory"]), packet, trial)
            if saved != prepared.get("references", {}).get(name) or saved != expected:
                raise FactoryEffortError("reused reference is no longer the exact prepared observation")
            continue
        if saved["status"] != "complete":
            continue
        trial, packet = assignments[name], inputs[assignments[name]["input_id"]]
        directory = root / trial["namespace"]
        call_dir = directory / "model_calls" / trial["stage"] / trial["call_id"]
        required = {str((call_dir / filename).relative_to(root)) for filename in
                    ("request.json", "prompt.md", "result.json", "raw_response.txt", "parsed_response.json")}
        required.update(f"{trial['namespace']}/{filename}" for filename in
                        ("state.json", "attempt_journal.json", "transport/manifest.json", "transport/codex-account"))
        if not required.issubset(saved["artifact_hashes"]):
            raise FactoryEffortError("completed effort call lacks required artifacts")
        for path, digest in saved["artifact_hashes"].items():
            if _digest(_inside(root, path)) != digest:
                raise FactoryEffortError(f"completed effort artifact changed: {path}")
        response = read_json(call_dir / "parsed_response.json")
        bindings[trial["stage"]].parser(response)
        _validate_response(packet, response)
        request = read_json(call_dir / "request.json")
        logical = read_json(directory / "state.json")["call_ledger"][f"{ARM}/{trial['stage']}/{trial['call_id']}"]
        journal = read_json(directory / "attempt_journal.json")["calls"][f"{trial['stage']}/{trial['call_id']}"]
        transport = read_json(directory / "transport/manifest.json")
        if (request["model"]["name"] != trial["model"] or request["model"]["reasoning_effort"] != trial["effort"]
                or request["schema_hash"] != packet["schema_hash"] or request["parent_ids"] != trial["parent_ids"]
                or _digest(call_dir / "prompt.md") != packet["prompt_sha256"]
                or logical["status"] != "complete" or logical["account"] != trial["account"]
                or logical["response_content_hash"] != saved["response_hash"]
                or content_hash(response) != saved["response_hash"]
                or transport["account"] != trial["account"] or transport["codex_home"] != logical["codex_home"]
                or journal["status"] != "complete" or not journal["attempts"]
                or len(journal["attempts"]) > 2 or any(row["outcome"] == "started" for row in journal["attempts"])):
            raise FactoryEffortError("completed effort response lost its request/account/attempt binding")
    return ledger


def _execute(config: dict[str, Any], prepared: Mapping[str, Any], trial: Mapping[str, Any],
             packet: Mapping[str, Any], pool: DedicatedCallPool) -> dict[str, Any]:
    root = Path(config["_output_directory"])
    profile = runtime_profile(config, trial)
    cache = Path(profile["_paths"]["cache"])
    if not (root / trial["namespace"] / "state.json").exists() and cache.exists() and any(cache.rglob("*.json")):
        raise FactoryEffortError("fresh independent namespace already contains a response cache")
    runtime = FactoryRuntime(profile, {"effort_prepared_hash": prepared["content_hash"],
                             "frozen_input_id": packet["input_id"], "frozen_input": packet}, pool,
                             namespace=trial["namespace"],
                             lane="selection_codex" if trial["purpose"] in UPSTREAM_PURPOSES else "downstream")
    response = runtime.invoke_rendered(
        purpose=trial["purpose"], stage=trial["stage"], call_id=trial["call_id"],
        prompt=_inside(Path(prepared["inputs_directory"]), packet["prompt_path"]).read_bytes().decode("utf-8"),
        prompt_template_version=packet["prompt_template_version"], prompt_template_hash=packet["prompt_template_hash"],
        parent_ids=trial["parent_ids"], response_validator=lambda value: _validate_response(packet, value), arm_id=ARM,
    )
    marker = read_json(response.log_dir / "result.json")
    if marker.get("cache_source") == "global":
        raise FactoryEffortError("independent effort observation used a global response cache")
    files = [response.log_dir / filename for filename in
             ("request.json", "prompt.md", "result.json", "raw_response.txt", "parsed_response.json")]
    files += [runtime.run_dir / filename for filename in
              ("state.json", "attempt_journal.json", "runtime_manifest.json", "transport/manifest.json", "transport/codex-account")]
    files += sorted(response.log_dir.glob("*.log"))
    files += sorted((response.log_dir / "transport_observations").glob("*.log"))
    return {"status": "complete", "response_hash": content_hash(response.value),
            "response_path": str((response.log_dir / "parsed_response.json").relative_to(root)),
            "initial_cache_source": marker.get("cache_source"), "structural_outcome": describe_response(response.value),
            "artifact_hashes": {str(path.relative_to(root)): _digest(path) for path in files}}


def _report(config: dict[str, Any], prepared: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["_output_directory"])
    ledger = _verify_ledger(config, prepared)
    assignments = prepared["assignments"]
    cells = {}
    runtimes = _saved_runtimes(root, assignments)
    for effort in EFFORTS:
        trials = [row for row in assignments if row["effort"] == effort]
        done = [row for row in trials if ledger["calls"].get(row["assignment_id"], {}).get("status") == "complete"]
        cells[effort] = {"scheduled": len(trials), "completed": len(done),
                         "failed": sum(ledger["calls"].get(row["assignment_id"], {}).get("status") == "failed" for row in trials),
                         "independent_repeat_completions": sum(row["repeat"] == 1 for row in done)}
        namespaces = {row["namespace"] for row in trials}
        accounting = call_accounting([runtime for runtime in runtimes if runtime.run_dir.name in namespaces])
        cells[effort]["accounting"] = {key: value for key, value in accounting.items() if key not in {"calls", "attempts"}}
    complete = sum(row["completed"] for row in cells.values())
    failed = sum(row["failed"] for row in cells.values())
    reused = len(prepared.get("references", {}))
    for effort in EFFORTS:
        cells[effort]["reused_references"] = sum(row["effort"] == effort and row["assignment_id"] in prepared.get("references", {})
                                                for row in assignments)
        cells[effort]["available_observations"] = cells[effort]["completed"] + cells[effort]["reused_references"]
    status = ("screen_complete_requires_independent_evaluation" if complete + reused == len(assignments) else
              "screen_terminal_with_failures" if complete + reused + failed == len(assignments) else
              "partial" if complete or failed else "prepared")
    accounting = call_accounting(runtimes)
    repeats = []
    for trial in assignments:
        if trial["repeat"] != 1:
            continue
        base = next(row for row in assignments if row["input_id"] == trial["input_id"] and row["effort"] == trial["effort"] and row["repeat"] == 0)
        base_result, repeat_result = ledger["calls"].get(base["assignment_id"], {}), ledger["calls"].get(trial["assignment_id"], {})
        both_available = base_result.get("status") in {"complete", "reused_reference"} and repeat_result.get("status") == "complete"
        repeats.append({"input_id": trial["input_id"], "effort": trial["effort"],
                        "both_complete": both_available,
                        "base_reused_reference": base_result.get("status") == "reused_reference",
                        "parsed_response_identical": (base_result["response_hash"] == repeat_result["response_hash"])
                            if both_available else None,
                        "meaning": "descriptive repeat agreement, not independent correctness"})
    pools = {}
    events = []
    for account in prepared["accounts"]:
        path = root / "worker_pools" / f"{account}.json"
        if not path.is_file():
            continue
        pool = read_json(path)
        if pool["peak_active"] > 12 or pool["peak_downstream"] > 8 or pool["peak_upstream"] > 12:
            raise FactoryEffortError("account worker pool exceeded its frozen bound")
        pools[account] = {key: pool[key] for key in
                          ("peak_active", "peak_downstream", "peak_upstream", "peak_borrowed_upstream", "active_calls", "waiting_by_category")}
        for ordinal, event in enumerate(pool["events"]):
            if event["kind"] in {"acquired", "released", "resume_abandoned_active_calls"}:
                events.append((event["at_epoch_seconds"], account, ordinal, event.get("active_total", 0)))
    active, peak = {}, 0
    for _, account, _, count in sorted(events):
        active[account] = count
        peak = max(peak, sum(active.values()))
    drained = all(not row["active_calls"] and not any(row["waiting_by_category"].values()) for row in pools.values())
    if not drained:
        status = "incomplete_active_pool"
    return {"schema_version": 1, "purpose": prepared["purpose"], "status": status,
            "prepared_hash": prepared["content_hash"], "cells": cells,
            "coverage": prepared["input_manifest"]["coverage"], "target_observations": len(assignments),
            "target_logical_calls": len(assignments) - reused, "available_observations": complete + reused,
            "completed_target_calls": complete, "call_accounting": accounting,
            "repeat_observations": repeats,
            "account_pools": pools, "combined_observed_provider_peak": peak,
            "maximum_account_campaigns": 2, "account_pools_drained": drained,
            "observations": [{**row, **ledger["calls"].get(row["assignment_id"], {"status": "pending"})} for row in assignments],
            "missing_evaluation": ["common_independent_quality_review", "held_out_confirmation",
                                   "role_specific_downstream_or_reverification_outcomes"],
            "recommendation": "no_automatic_effort_decision", "reference_effort": config["model_profile"]["efforts"][prepared["purpose"]],
            "prior_call_credit": reused, "automatic_effort_adoption": False, "automatic_task_commit": False,
            "mock": config["run"]["mock"], "protected_state_unchanged": True,
            "transport_evidence_note": "Requested bindings and saved CLI artifacts are retained; mock results are not account-access evidence. No token savings or model absence-of-tools claim is inferred."}


def _save_report(root: Path, report: Mapping[str, Any]) -> None:
    path = root / "report.json"
    if not path.is_file() or read_json(path) != report:
        atomic_write_json(path, report)
    note = (f"# Effort screen: {report['purpose']}\n\nStatus: {report['status']}. "
            f"Completed {report['completed_target_calls']}/{report['target_logical_calls']} target calls.\n\n"
            "These are structural and model-reported observations, not an independently graded effort winner. "
            "Common independent evaluation, role-specific downstream outcomes and held-out confirmation remain missing. "
            f"Reused {report['prior_call_credit']} exact same-account base references; they are not new calls or fresh timing. "
            "No automatic effort adoption or task-bank mutation.\n")
    if not (root / "report.md").is_file() or (root / "report.md").read_text(encoding="utf-8") != note:
        atomic_write_text(root / "report.md", note)


def run_effort(config: Mapping[str, Any], *, purpose: str, authorize_model_calls: bool = False) -> dict[str, Any]:
    """Run one prepared purpose, with two bounded concurrent account campaigns."""
    resolved = _config(config, purpose)
    root = Path(resolved["_output_directory"])
    with run_lock(root):
        prepared = _read_prepared(resolved)
        ledger = _verify_ledger(resolved, prepared)
        pending = [row for row in prepared["assignments"] if ledger["calls"].get(row["assignment_id"], {}).get("status") not in {"complete", "reused_reference"}]
        if pending and not config["run"]["mock"] and not authorize_model_calls:
            raise FactoryEffortError("live effort calls require --authorize-model-calls for this purpose")
        if pending:
            pools = {account: DedicatedCallPool(root / "worker_pools" / f"{account}.json")
                     for account in prepared["accounts"]}
            packets = {row["input_id"]: row for row in prepared["input_manifest"]["inputs"]}
            ledger_lock = threading.Lock()

            def branch(account: str) -> None:
                trials = [row for row in pending if row["account"] == account]
                workers = 12 if purpose in UPSTREAM_PURPOSES else 8
                # Counterbalanced submission order is frozen; completion order
                # does not impose a barrier on unrelated trials or later work.
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(_execute, resolved, prepared, trial,
                               packets[trial["input_id"]], pools[account]): trial for trial in trials}
                    for future in as_completed(futures):
                        trial = futures[future]
                        try:
                            value = future.result()
                        except Exception as exc:
                            value = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
                        atomic_write_json(root / "trial_results" / f"{trial['assignment_id']}.json", value)
                        with ledger_lock:
                            ledger["calls"][trial["assignment_id"]] = value
                            atomic_write_json(root / "call_ledger.private.json", ledger)

            try:
                with ThreadPoolExecutor(max_workers=len(prepared["accounts"])) as executor:
                    futures = [executor.submit(branch, account) for account in prepared["accounts"]]
                    for future in as_completed(futures):
                        future.result()
            finally:
                for pool in pools.values():
                    pool.set_downstream_ready(False, reason="effort-purpose-drained")
                    pool.finish()
        _read_prepared(resolved)  # Protected state and frozen source stay unchanged.
        report = _report(resolved, prepared)
        _save_report(root, report)
        return report


def effort_status(config: Mapping[str, Any], *, purpose: str | None = None) -> dict[str, Any]:
    """Read saved progress; missing roles are not mistaken for completed work."""
    if purpose is not None:
        resolved = _config(config, purpose)
        if not (Path(resolved["_output_directory"]) / "prepared.json").is_file():
            return {"purpose": purpose, "status": "missing_input_panel", "completed_target_calls": 0}
        return _report(resolved, _read_prepared(resolved))
    roles = {name: effort_status(config, purpose=name) for name in ACTIVE_PURPOSES}
    return {"status": "role_screens_not_a_validated_profile", "roles": roles,
            "missing_purposes": [name for name, row in roles.items() if row["status"] == "missing_input_panel"],
            "automatic_effort_adoption": False, "automatic_task_commit": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "report"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--purpose", choices=tuple(ACTIVE_PURPOSES))
    parser.add_argument("--accounts", nargs="+", default=["main"])
    parser.add_argument("--reuse-matching-references", action="store_true")
    parser.add_argument("--authorize-model-calls", action="store_true")
    args = parser.parse_args(argv)
    if args.command in {"prepare", "run"} and not args.purpose:
        parser.error("prepare/run requires one explicit --purpose; there is no launch-all command")
    if args.command == "prepare" and args.inputs is None:
        parser.error("prepare requires --inputs")
    try:
        config = load_factory_config(args.config)
        if args.command == "prepare":
            value = prepare_effort(config, args.inputs, purpose=args.purpose, accounts=args.accounts,
                                   reuse_matching_references=args.reuse_matching_references)
        elif args.command == "run":
            value = run_effort(config, purpose=args.purpose, authorize_model_calls=args.authorize_model_calls)
        else:
            value = effort_status(config, purpose=args.purpose)
            if args.command == "report":
                reports = [value] if args.purpose else list(value["roles"].values())
                for report in reports:
                    if "prepared_hash" in report:
                        _save_report(Path(_config(config, report["purpose"])["_output_directory"]), report)
        print(json.dumps(value, indent=2))
        return 1 if args.command == "run" and value["completed_target_calls"] != value["target_logical_calls"] else 0
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
