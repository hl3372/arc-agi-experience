"""Completion-order-invariant parallel boards and audits for pre-E20."""

from __future__ import annotations

import time
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any

from .factory_contracts import display_value
from .lean_v3_pipeline import classify_v3_editorial_results
from .util import atomic_write_json, content_hash, read_json
from .v41_experiments import _anonymous_id, _portfolio_digest
from .v41_supported_departure import SPECIAL_LABEL_FRAGMENTS
from .v41_test_challenge import _anchors
from .v42_logical_leap import _private_label_leaks
from .v4_integrated_validation import MECHANISM_AUDITOR_FORBIDDEN_LABELS


class PreE20DeferredError(RuntimeError):
    """Raised on package drift or an incomplete qualified review call set."""


def _immutable(path: Path, value: Any, label: str) -> None:
    if path.is_file():
        if read_json(path) != value:
            raise PreE20DeferredError(f"immutable {label} changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _protected_state(runtime: Any) -> str:
    fields = (
        "complete_task",
        "concrete_plan",
        "realization",
        "technical_results",
        "blind_results",
        "answer_evidence",
        "boardless_disposition",
    )
    return content_hash(
        {
            task_id: {key: record.get(key) for key in fields}
            for task_id, record in sorted(runtime.state["semantic_records"].items())
        }
    )


def _audit_protected_state(runtime: Any) -> str:
    """Audits are forbidden from changing any task or disposition field."""

    return content_hash(
        {
            task_id: deepcopy(record)
            for task_id, record in sorted(runtime.state["semantic_records"].items())
        }
    )


def _call_attempt_span(runtime: Any, call_ids: Sequence[str]) -> float:
    starts: list[float] = []
    ends: list[float] = []
    wanted = set(call_ids)
    journal_path = runtime.run_dir / "attempt_journal.json"
    if not wanted:
        return 0.0
    if not journal_path.is_file():
        raise PreE20DeferredError("deferred timing lacks the physical-attempt journal")
    journal = read_json(journal_path)
    found: set[str] = set()
    for row in (journal.get("calls") or {}).values():
        call_id = str(row.get("call_id") or "")
        if call_id not in wanted:
            continue
        found.add(call_id)
        attempts = list(row.get("attempts") or [])
        if not attempts:
            raise PreE20DeferredError(
                f"deferred call has no physical attempt timing: {call_id}"
            )
        for attempt in attempts:
            start = attempt.get("started_at_epoch_seconds")
            end = attempt.get("completed_at_epoch_seconds")
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or float(end) < float(start)
                or attempt.get("outcome") == "started"
            ):
                raise PreE20DeferredError(
                    f"deferred physical attempt is nonterminal: {call_id}"
                )
            starts.append(float(start))
            ends.append(float(end))
    if found != wanted:
        raise PreE20DeferredError(
            "deferred timing journal does not contain the exact call set"
        )
    return max(0.0, max(ends) - min(starts))


def _append_dispatch_session(directory: Path, value: Mapping[str, Any]) -> None:
    """Persist non-semantic observation data without making order immutable."""

    path = directory / "dispatch_sessions.json"
    prior = read_json(path) if path.is_file() else {"schema_version": 1, "sessions": []}
    rows = list(prior.get("sessions") or [])
    rows.append({"session_index": len(rows) + 1, **deepcopy(dict(value))})
    atomic_write_json(path, {"schema_version": 1, "sessions": rows})


def freeze_board_packages(runtime: Any, *, portfolio_digest: Mapping[str, Any] | None = None, board_indices: tuple[int, ...] = (1, 2)) -> dict[str, Any]:
    if board_indices not in {(1, 2), (1,)}:
        raise ValueError("board packages require the legacy pair or preselected board 1")
    directory = runtime.run_dir / "pre_e20_parallel_boards"
    path = directory / "packages.json"
    records = runtime.state["semantic_records"]
    eligible = sorted(
        [
            task_id
            for task_id, record in records.items()
            if record.get("status") in {"blind_pass", "editorial_complete"}
        ],
        key=lambda task_id: content_hash(
            {"seed": runtime.config["seed"], "task": task_id}
        ),
    )
    anchors = _anchors(runtime.config)
    # Paired studies may reuse the already frozen board context. Ordinary
    # production still resolves and verifies its current registry as before.
    portfolio = deepcopy(dict(portfolio_digest)) if portfolio_digest is not None else _portfolio_digest(runtime.config)
    groups = []
    for group_index in range(0, len(eligible), 6):
        group = eligible[group_index : group_index + 6]
        ordinal = group_index // 6 + 1
        calls = []
        for board_index in board_indices:
            rng = Random(int(runtime.config["seed"]) + ordinal * 131 + board_index * 19)
            used: set[str] = set()
            rows = [
                {
                    "source": f"anchor:{anchor.task_id}",
                    "rule": anchor.rule_description,
                    "episode": {
                        "train": list(anchor.train),
                        "test": list(anchor.test),
                    },
                }
                for anchor in anchors
            ] + [
                {
                    "source": f"generated:{task_id}",
                    "rule": display_value(
                        records[task_id]["realization"]["authoring_metadata"]["intended_rule"],
                        records[task_id].get("concrete_plan"),
                    ),
                    "episode": records[task_id]["complete_task"],
                }
                for task_id in group
            ]
            rng.shuffle(rows)
            mapping: dict[str, str] = {}
            package = []
            for row in rows:
                opaque = _anonymous_id(rng, used)
                mapping[opaque] = str(row["source"])
                package.append(
                    {
                        "task_id": opaque,
                        "rule_summary": row["rule"],
                        "episode": row["episode"],
                    }
                )
            board_id = "board-" + content_hash(
                {"group": ordinal, "board": board_index, "ids": sorted(mapping)}
            )[:14]
            calls.append(
                {
                    "group_index": ordinal,
                    "board_index": board_index,
                    "call_id": board_id,
                    "group": group,
                    "mapping": mapping,
                    "values": {
                        "BLINDED_TASKS": package,
                        "PORTFOLIO_DIGEST": portfolio,
                        "BOARD_ID": board_id,
                    },
                    "parent_ids": group,
                }
            )
        groups.append({"group_index": ordinal, "task_ids": group, "calls": calls})
    value = {
        "schema_version": 1,
        "eligible_task_ids": eligible,
        "group_size": 6,
        "boards_per_group": len(board_indices),
        "fixed_anchors_included": True,
        "groups": groups,
        "protected_state_hash": _protected_state(runtime),
    }
    _immutable(path, value, "parallel board packages")
    return value


def _board_call(runtime: Any, directory: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    call_id = str(spec["call_id"])
    response_path = directory / "responses" / f"{call_id}.json"
    if response_path.is_file():
        response = read_json(response_path)
    else:

        def validate(value: dict[str, Any]) -> None:
            returned = [item["task_id"] for item in value["judgments"]]
            expected = set(spec["mapping"])
            if len(returned) != len(set(returned)) or set(returned) != expected:
                raise PreE20DeferredError("board did not judge every opaque task")

        response = runtime.call(
            prompt_name="editorial_board",
            call_id=call_id,
            values=spec["values"],
            parent_ids=spec["parent_ids"],
            role="editorial_board",
            arm_id="blinded-single-editorial",
            expected={"board_id": call_id},
            validator=validate,
            forbidden_prompt_fragments=(
                *SPECIAL_LABEL_FRAGMENTS,
                "test challenge",
                "test-only refinement",
            ),
        )
        _immutable(response_path, response, f"board response {call_id}")
    returned = [item["task_id"] for item in response["judgments"]]
    if len(returned) != len(set(returned)) or set(returned) != set(spec["mapping"]):
        raise PreE20DeferredError("persisted board response changed membership")
    return {
        "group_index": int(spec["group_index"]),
        "board_index": int(spec["board_index"]),
        "group": list(spec["group"]),
        "mapping": deepcopy(dict(spec["mapping"])),
        "response": response,
    }


def _board_updates(
    group: Mapping[str, Any],
    by_key: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Derive one group's editorial update in deterministic board order."""

    ordinal = int(group["group_index"])
    updates = {}
    for task_id in group["task_ids"]:
        judgments = []
        for board_index in (1, 2):
            result = by_key[(ordinal, board_index)]
            opaque = next(
                key
                for key, source in result["mapping"].items()
                if source == f"generated:{task_id}"
            )
            judgment = next(
                item
                for item in result["response"]["judgments"]
                if item["task_id"] == opaque
            )
            judgments.append({**deepcopy(judgment), "task_id": task_id})
        intrinsic, placement, intrinsic_reason, placement_reason = (
            classify_v3_editorial_results(judgments)
        )
        updates[str(task_id)] = {
            "editorial_judgments": judgments,
            "intrinsic_quality": intrinsic,
            "portfolio_placement": placement,
            "editorial_reason": intrinsic_reason + "; " + placement_reason,
        }
    return updates


def _verify_completed_boards(
    runtime: Any,
    directory: Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    call_workers: int,
) -> dict[str, Any]:
    """Reconstruct a terminal board aggregate without provider dispatch."""

    specs = [call for group in manifest["groups"] for call in group["calls"]]
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for spec in specs:
        response_path = directory / "responses" / f"{spec['call_id']}.json"
        if not response_path.is_file():
            raise PreE20DeferredError("terminal board report lacks a call response")
        result = _board_call(runtime, directory, spec)
        by_key[(int(spec["group_index"]), int(spec["board_index"]))] = result
    all_results = [by_key[key] for key in sorted(by_key)]
    aggregate_path = runtime.run_dir / "editorial_board_results.json"
    if not aggregate_path.is_file() or read_json(aggregate_path) != all_results:
        raise PreE20DeferredError("terminal board aggregate changed")
    for group in manifest["groups"]:
        ordinal = int(group["group_index"])
        updates = _board_updates(group, by_key)
        for task_id, update in updates.items():
            record = runtime.state["semantic_records"].get(task_id) or {}
            if record.get("status") != "editorial_complete" or any(
                record.get(key) != value for key, value in update.items()
            ):
                raise PreE20DeferredError("terminal editorial task state changed")
        applied = directory / "applied" / f"group-{ordinal:02d}.json"
        wanted = {
            "schema_version": 1,
            "group_index": ordinal,
            "task_ids": list(group["task_ids"]),
            "updates_hash": content_hash(updates),
            "both_boards_required": True,
        }
        if not applied.is_file() or read_json(applied) != wanted:
            raise PreE20DeferredError("terminal board application ledger changed")
    if _protected_state(runtime) != manifest["protected_state_hash"]:
        raise PreE20DeferredError("terminal boards changed protected task state")
    fixed = {
        "completed": True,
        "path": str(aggregate_path.resolve()),
        "board_call_count": len(specs),
        "group_count": len(manifest["groups"]),
        "maximum_call_workers": call_workers,
        "maximum_concurrent_groups": 4,
        "deterministic_apply_order": True,
        "wall_seconds": round(
            _call_attempt_span(runtime, [spec["call_id"] for spec in specs]), 6
        ),
        "wall_seconds_source": "persisted_call_attempt_span",
    }
    if any(report.get(key) != value for key, value in fixed.items()):
        raise PreE20DeferredError("terminal board report changed")
    return deepcopy(dict(report))


def run_parallel_boards(runtime: Any, *, call_workers: int = 8) -> dict[str, Any]:
    if call_workers != 8:
        raise PreE20DeferredError("parallel board dispatcher requires eight call workers")
    directory = runtime.run_dir / "pre_e20_parallel_boards"
    report_path = directory / "report.json"
    if report_path.is_file():
        manifest = freeze_board_packages(runtime)
        return _verify_completed_boards(
            runtime,
            directory,
            manifest,
            read_json(report_path),
            call_workers=call_workers,
        )
    started = time.monotonic()
    manifest = freeze_board_packages(runtime)
    specs = [call for group in manifest["groups"] for call in group["calls"]]
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    completion: list[dict[str, Any]] = []
    completion_lock = threading.Lock()
    def run_group(group: Mapping[str, Any]) -> list[dict[str, Any]]:
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(_board_call, runtime, directory, spec): spec
                for spec in group["calls"]
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                with completion_lock:
                    completion.append(
                        {
                            "completion_rank": len(completion) + 1,
                            "group_index": int(result["group_index"]),
                            "board_index": int(result["board_index"]),
                            "call_id": str(futures[future]["call_id"]),
                        }
                    )
        return results

    # Four group futures, each owning exactly two board-call futures, enforce
    # both the four-group and eight-call ceilings.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_group, group): group for group in manifest["groups"]
        }
        for future in as_completed(futures):
            for result in future.result():
                key = (int(result["group_index"]), int(result["board_index"]))
                by_key[key] = result
    _append_dispatch_session(
        directory,
        {
            "kind": "board_dispatch",
            "completion_order": completion,
            "orchestration_wall_seconds": round(time.monotonic() - started, 6),
        },
    )
    expected = {
        (int(spec["group_index"]), int(spec["board_index"])) for spec in specs
    }
    if set(by_key) != expected:
        raise PreE20DeferredError("parallel board call set is incomplete")

    all_results = [by_key[key] for key in sorted(by_key)]
    for group in manifest["groups"]:
        ordinal = int(group["group_index"])
        updates = _board_updates(group, by_key)
        with runtime.lock:
            now = round(time.time(), 3)
            for task_id, update in updates.items():
                record = runtime.state["semantic_records"][task_id]
                if record.get("status") == "editorial_complete":
                    if any(record.get(key) != value for key, value in update.items()):
                        raise PreE20DeferredError("editorial aggregate changed on resume")
                    continue
                record.update(deepcopy(update))
                record["status"] = "editorial_complete"
                record.setdefault("stage_history", []).append(
                    {
                        "stage": "editorial",
                        "at_epoch_seconds": now,
                        "status": "editorial_complete",
                    }
                )
            runtime.save()
        _immutable(
            directory / "applied" / f"group-{ordinal:02d}.json",
            {
                "schema_version": 1,
                "group_index": ordinal,
                "task_ids": list(group["task_ids"]),
                "updates_hash": content_hash(updates),
                "both_boards_required": True,
            },
            f"board group {ordinal} application",
        )
    atomic_write_json(runtime.run_dir / "editorial_board_results.json", all_results)
    if _protected_state(runtime) != manifest["protected_state_hash"]:
        raise PreE20DeferredError("boards changed pre-editorial protected fields")
    result = {
        "completed": True,
        "path": str((runtime.run_dir / "editorial_board_results.json").resolve()),
        "board_call_count": len(specs),
        "group_count": len(manifest["groups"]),
        "maximum_call_workers": call_workers,
        "maximum_concurrent_groups": 4,
        "deterministic_apply_order": True,
        "wall_seconds": round(
            _call_attempt_span(runtime, [spec["call_id"] for spec in specs]), 6
        ),
        "wall_seconds_source": "persisted_call_attempt_span",
        "orchestration_wall_seconds": round(time.monotonic() - started, 6),
    }
    atomic_write_json(report_path, result)
    return result


def _audit_specs(runtime: Any) -> list[dict[str, Any]]:
    completed = [
        deepcopy(record)
        for record in runtime.state["semantic_records"].values()
        if record.get("complete_task")
    ]
    families = (
        (
            "logical_leap",
            52883,
            int(runtime.config["logical_leap"]["audit_batch_size"]),
            "logical-leap-audit",
            "logical_leap_auditor",
            "blinded-logical-leap-audit",
        ),
        (
            "structural_transfer",
            46271,
            int(runtime.config["structural_transfer"]["audit_batch_size"]),
            "structural-transfer-audit",
            "structural_transfer_auditor",
            "blinded-structural-audit",
        ),
        (
            "joint_and_abstraction",
            99181,
            int(runtime.config["integrated_qualification"]["audit_batch_size"]),
            "integrated-joint-abstraction",
            "mechanism_auditor",
            "blinded-integrated-mechanism-audit",
        ),
    )
    specs = []
    for family_order, row in enumerate(families, 1):
        family, seed_offset, batch_size, prefix, prompt, arm = row
        records = deepcopy(completed)
        Random(int(runtime.config["seed"]) + seed_offset).shuffle(records)
        for offset in range(0, len(records), batch_size):
            chunk = records[offset : offset + batch_size]
            batch_index = offset // batch_size + 1
            audit_id = f"{prefix}-{batch_index:02d}"
            packages = [
                {
                    "task_id": record["task_id"],
                    "episode": record["complete_task"],
                    "concise_rule": record["realization"]["authoring_metadata"][
                        "intended_rule"
                    ],
                    "concrete_plan": record["concrete_plan"],
                }
                for record in chunk
            ]
            if family == "logical_leap" and any(
                _private_label_leaks(package) for package in packages
            ):
                raise PreE20DeferredError("private routing label entered logical audit")
            values = {"AUDIT_ID": audit_id, "TASK_PACKAGES": packages}
            expected: dict[str, Any] = {"audit_id": audit_id}
            forbidden: Sequence[str] = ()
            if family == "structural_transfer":
                forbidden = (
                    "single-authoring",
                    "optional suggestion",
                    "experimental arm",
                    "nominated candidate",
                )
            elif family == "joint_and_abstraction":
                values = {
                    "REQUESTED_PROPERTIES": [
                        "episode_joint_inference",
                        "abstraction_led",
                    ],
                    "TASK_PACKAGES": packages,
                    "AUDIT_ID": audit_id,
                }
                expected.update(
                    {
                        "requested_properties": [
                            "episode_joint_inference",
                            "abstraction_led",
                        ],
                        "non_gating_confirmation": True,
                    }
                )
                forbidden = MECHANISM_AUDITOR_FORBIDDEN_LABELS
            specs.append(
                {
                    "family": family,
                    "family_order": family_order,
                    "batch_index": batch_index,
                    "call_id": audit_id,
                    "prompt_name": prompt,
                    "arm_id": arm,
                    "values": values,
                    "expected": expected,
                    "forbidden": list(forbidden),
                    "parent_ids": [item["task_id"] for item in packages],
                    "task_ids": [item["task_id"] for item in packages],
                }
            )
    return specs


def freeze_audit_packages(runtime: Any) -> dict[str, Any]:
    path = runtime.run_dir / "pre_e20_parallel_audits" / "packages.json"
    value = {
        "schema_version": 1,
        "non_gating": True,
        "all_families_frozen_before_dispatch": True,
        "protected_state_hash": _audit_protected_state(runtime),
        "calls": _audit_specs(runtime),
    }
    _immutable(path, value, "parallel audit packages")
    return value


def _audit_call(runtime: Any, directory: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    response_path = directory / "responses" / f"{spec['call_id']}.json"
    if response_path.is_file():
        response = read_json(response_path)
    else:

        def validate(value: dict[str, Any]) -> None:
            returned = [item["task_id"] for item in value["judgments"]]
            expected_ids = set(spec["task_ids"])
            if len(returned) != len(set(returned)) or set(returned) != expected_ids:
                raise PreE20DeferredError("auditor omitted or duplicated tasks")

        response = runtime.call(
            prompt_name=str(spec["prompt_name"]),
            call_id=str(spec["call_id"]),
            values=spec["values"],
            parent_ids=spec["parent_ids"],
            role="editorial_board",
            arm_id=str(spec["arm_id"]),
            expected=spec["expected"],
            validator=validate,
            forbidden_prompt_fragments=spec["forbidden"],
        )
        _immutable(response_path, response, f"audit response {spec['call_id']}")
    returned = [item["task_id"] for item in response["judgments"]]
    if len(returned) != len(set(returned)) or set(returned) != set(spec["task_ids"]):
        raise PreE20DeferredError("persisted audit response changed membership")
    return {**deepcopy(dict(spec)), "response": response}


def _audit_families(
    ordered: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for family in ("logical_leap", "structural_transfer", "joint_and_abstraction"):
        rows = [row for row in ordered if row["family"] == family]
        calls = [deepcopy(dict(row["response"])) for row in rows]
        judgments = [item for response in calls for item in response["judgments"]]
        families[family] = {"calls": calls, "judgments": judgments}
    return families


def _audit_artifacts(
    families: Mapping[str, Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    family_artifacts = {
        "logical_leap_audit.json": {
            "route_hidden": True,
            "source_hidden": True,
            "non_gating": True,
            **deepcopy(dict(families["logical_leap"])),
        },
        "structural_transfer_audit.json": {
            "arm_hidden": True,
            "nomination_hidden": True,
            "non_gating": True,
            **deepcopy(dict(families["structural_transfer"])),
        },
        "joint_abstraction_audit.json": {
            "source_and_route_hidden": True,
            "non_gating": True,
            **deepcopy(dict(families["joint_and_abstraction"])),
        },
    }
    integrated = {
        "schema_version": 1,
        "non_gating": True,
        "route_and_source_hidden": True,
        "logical_leap": deepcopy(families["logical_leap"]["judgments"]),
        "structural_transfer": deepcopy(
            families["structural_transfer"]["judgments"]
        ),
        "joint_and_abstraction": deepcopy(
            families["joint_and_abstraction"]["judgments"]
        ),
        "completed": not failures,
        "operational_failures": deepcopy(list(failures)),
    }
    return integrated, family_artifacts


def _verify_completed_audits(
    runtime: Any,
    directory: Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    workers: int,
) -> dict[str, Any]:
    """Reconstruct terminal non-gating audits without provider dispatch."""

    failures = list(report.get("operational_failures") or [])
    if any(row.get("error_type") == "PreE20DeferredError" for row in failures):
        raise PreE20DeferredError("terminal audit report retained an integrity failure")
    failure_ids = [str(row.get("call_id") or "") for row in failures]
    if len(failure_ids) != len(set(failure_ids)):
        raise PreE20DeferredError("terminal audit report duplicated failures")
    expected_ids = {str(spec["call_id"]) for spec in manifest["calls"]}
    if not set(failure_ids) <= expected_ids:
        raise PreE20DeferredError("terminal audit report names an unknown failure")
    ordered = []
    for spec in manifest["calls"]:
        call_id = str(spec["call_id"])
        response_path = directory / "responses" / f"{call_id}.json"
        if call_id in failure_ids:
            if response_path.is_file():
                raise PreE20DeferredError(
                    "terminal audit failure also has a successful response"
                )
            continue
        if not response_path.is_file():
            raise PreE20DeferredError("terminal audit report lacks a call response")
        ordered.append(_audit_call(runtime, directory, spec))
    ordered.sort(key=lambda row: (row["family_order"], row["batch_index"]))
    families = _audit_families(ordered)
    integrated, family_artifacts = _audit_artifacts(families, failures)
    for filename, wanted in family_artifacts.items():
        path = runtime.run_dir / filename
        if not path.is_file() or read_json(path) != wanted:
            raise PreE20DeferredError(f"terminal audit artifact changed: {filename}")
    integrated_path = runtime.run_dir / "integrated_mechanism_audits.json"
    if not integrated_path.is_file() or read_json(integrated_path) != integrated:
        raise PreE20DeferredError("terminal integrated audit aggregate changed")
    if _audit_protected_state(runtime) != manifest["protected_state_hash"]:
        raise PreE20DeferredError("terminal audits changed protected task state")
    fixed = {
        "schema_version": 1,
        "completed": not failures,
        "non_gating": True,
        "call_count": len(ordered),
        "family_call_counts": {
            family: len(value["calls"]) for family, value in families.items()
        },
        "maximum_workers": workers,
        "deterministic_aggregation_order": True,
        "wall_seconds": round(
            _call_attempt_span(
                runtime, [str(spec["call_id"]) for spec in manifest["calls"]]
            ),
            6,
        ),
        "wall_seconds_source": "persisted_call_attempt_span",
        "operational_failures": failures,
    }
    if any(report.get(key) != value for key, value in fixed.items()):
        raise PreE20DeferredError("terminal audit report changed")
    return integrated


def run_parallel_audits(runtime: Any, *, workers: int = 8) -> dict[str, Any]:
    if workers != 8:
        raise PreE20DeferredError("parallel audit dispatcher requires eight workers")
    directory = runtime.run_dir / "pre_e20_parallel_audits"
    report_path = directory / "report.json"
    if report_path.is_file():
        manifest = freeze_audit_packages(runtime)
        return _verify_completed_audits(
            runtime,
            directory,
            manifest,
            read_json(report_path),
            workers=workers,
        )
    started = time.monotonic()
    manifest = freeze_audit_packages(runtime)
    results = {}
    completion = []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_audit_call, runtime, directory, spec): spec
            for spec in manifest["calls"]
        }
        for rank, future in enumerate(as_completed(futures), 1):
            spec = futures[future]
            try:
                result = future.result()
                results[str(spec["call_id"])] = result
                completion.append(
                    {
                        "completion_rank": rank,
                        "family": spec["family"],
                        "batch_index": spec["batch_index"],
                        "call_id": spec["call_id"],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - audits remain non-gating
                failures.append(
                    {
                        "family": spec["family"],
                        "family_order": int(spec["family_order"]),
                        "batch_index": spec["batch_index"],
                        "call_id": spec["call_id"],
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )
    failures.sort(
        key=lambda row: (
            int(row["family_order"]),
            int(row["batch_index"]),
            str(row["call_id"]),
        )
    )
    _append_dispatch_session(
        directory,
        {
            "kind": "audit_dispatch",
            "non_gating": True,
            "completion_order": completion,
            "failures": failures,
            "orchestration_wall_seconds": round(time.monotonic() - started, 6),
        },
    )
    if any(row["error_type"] == "PreE20DeferredError" for row in failures):
        raise PreE20DeferredError(
            "a persisted audit package/response failed integrity validation"
        )

    ordered = sorted(
        results.values(), key=lambda row: (row["family_order"], row["batch_index"])
    )
    families = _audit_families(ordered)
    value, family_artifacts = _audit_artifacts(families, failures)
    for filename, artifact in family_artifacts.items():
        atomic_write_json(runtime.run_dir / filename, artifact)
    atomic_write_json(runtime.run_dir / "integrated_mechanism_audits.json", value)
    if _audit_protected_state(runtime) != manifest["protected_state_hash"]:
        raise PreE20DeferredError("non-gating audits changed protected task state")
    atomic_write_json(
        report_path,
        {
            "schema_version": 1,
            "completed": not failures,
            "non_gating": True,
            "call_count": len(ordered),
            "family_call_counts": {
                family: len(value["calls"]) for family, value in families.items()
            },
            "maximum_workers": workers,
            "deterministic_aggregation_order": True,
            "wall_seconds": round(
                _call_attempt_span(
                    runtime, [str(spec["call_id"]) for spec in manifest["calls"]]
                ),
                6,
            ),
            "wall_seconds_source": "persisted_call_attempt_span",
            "orchestration_wall_seconds": round(time.monotonic() - started, 6),
            "operational_failures": failures,
        },
    )
    return value


__all__ = [
    "PreE20DeferredError",
    "freeze_audit_packages",
    "freeze_board_packages",
    "run_parallel_audits",
    "run_parallel_boards",
]
