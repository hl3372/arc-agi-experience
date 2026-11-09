"""Explicitly authorized, transactional E18 task commitment.

Core and reserve tasks enter the existing accepted registry.  ``basic_valid``
tasks retain rejected intrinsic provenance and enter a separate curriculum
registry that is never used as positive generator context.
"""

from __future__ import annotations

import fcntl
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from .factory_contracts import display_value
from .active_bank import filter_active_registry
from .factory_policy import review_policy
from .factory_review_policy import required_boards, verify_recorded_review, verify_dossier_calls
from .basic_valid import assess_basic_valid
from .authorized_commits import (
    _render_bank,
    _write_catalog,
    _write_checksums,
    commit_authorized_manifest,
)
from .committed_dossiers import (
    materialize_committed_dossiers,
    verify_dossier_manifest,
)
from .committed_tasks import load_committed_task_registry
from .e18_artifacts import strict_technical_clear, verify_terminal_artifact_bundle
from .e18_evidence_gate import evidence_is_clear
from .lean_v3_rendering import build_contact_sheet, render_episode
from .util import atomic_write_json, atomic_write_text, content_hash, read_json
from .v41_palette import palette_equivalent_hash
from .v47_baseline_core_commit import _copy_calls, _title
from .validation import validate_arc_task

COMMIT_SET_ID = "e18-production-candidate-20260820"
BASIC_REGISTRY_ID = "arcgen-committed-basic-valid-v1"
LEGACY_COMMITTED_ON = "2026-08-20"


class E18CommitError(RuntimeError):
    """Raised when eligibility, staging, or the transaction is unsafe."""


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _root(config: Mapping[str, Any]) -> Path:
    return Path(config["_repository_root"])


def _output(config: Mapping[str, Any]) -> Path:
    return Path(config["_output_directory"])


def _commit_identity(config: Mapping[str, Any]) -> dict[str, str]:
    commit = config["commit"]
    commit_set_id = str(commit.get("commit_set_id") or COMMIT_SET_ID)
    committed_on = str(commit.get("committed_on") or LEGACY_COMMITTED_ON)
    manifest_filename = str(
        commit.get("installed_manifest_filename") or f"{commit_set_id}.json"
    )
    if (
        not commit_set_id
        or not committed_on
        or not manifest_filename.endswith(".json")
        or Path(manifest_filename).name != manifest_filename
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in commit_set_id)
    ):
        raise E18CommitError("invalid E18 commit identity")
    return {
        "commit_set_id": commit_set_id,
        "committed_on": committed_on,
        "installed_manifest_filename": manifest_filename,
    }


@contextmanager
def _committed_bank_lock(config: Mapping[str, Any]):
    """Serialize the complete E18 snapshot/mutation/verification transaction."""

    committed = Path(config["_paths"]["historical_registry"]).parent
    lock_path = committed / ".e18-commit.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _two_boards(record: Mapping[str, Any]) -> bool:
    return required_boards(record)


def _same_procedure_marked(record: Mapping[str, Any]) -> bool:
    """Return whether terminal editorial evidence marks a procedure sibling."""

    judgments = record.get("editorial_judgments") or []
    return record.get("portfolio_placement") == "same_procedure_duplicate" or any(
        isinstance(item, Mapping)
        and (
            item.get("same_minimal_procedure") is True
            or item.get("portfolio_recommendation") == "same_procedure_duplicate"
        )
        for item in judgments
    )


def _base_eligibility(
    record: Mapping[str, Any], *, allow_same_procedure: bool = False, review_policy=None
) -> list[str]:
    failures: list[str] = []
    validation = validate_arc_task(
        record.get("complete_task"), require_test_outputs=True
    )
    if not validation.valid:
        failures.append("invalid_arc_payload")
    if (record.get("deterministic_diagnostics") or {}).get("hard_valid") is not True:
        failures.append("deterministic_hard_invalid")
    if (record.get("plan_fidelity") or {}).get("passed") is not True:
        failures.append("plan_fidelity_failed")
    if not strict_technical_clear(record):
        failures.append("technical_not_exact_clear")
    if not evidence_is_clear(record):
        failures.append("answer_evidence_not_clear")
    if record.get("status") != "editorial_complete":
        failures.append("editorial_not_complete")
    if not required_boards(record, review_policy):
        failures.append("required_actual_editorial_board_missing" if review_policy else "two_editorial_boards_missing")
    if _same_procedure_marked(record) and not allow_same_procedure:
        failures.append("same_procedure_duplicate")
    terminal = record.get("terminal_artifact")
    if not isinstance(terminal, Mapping):
        failures.append("terminal_artifact_missing")
    elif not Path(str(terminal.get("artifact_manifest") or "")).is_file():
        failures.append("terminal_artifact_manifest_missing")
    return failures


def collect_commit_eligibility(
    config: Mapping[str, Any], runtime: Any
) -> dict[str, Any]:
    policy = review_policy(config)
    allow_same_procedure = (
        config.get("commit", {}).get("authorize_same_procedure_override") is True
    )
    registry = load_committed_task_registry(config["_paths"]["historical_registry"])
    existing = {
        palette_equivalent_hash(read_json(Path(item["absolute_task_path"])))
        for item in registry["entries"]
    }
    basic_path = Path(config["_paths"]["basic_valid_registry"])
    if basic_path.is_file():
        basic = load_basic_valid_registry(basic_path)
        existing.update(
            palette_equivalent_hash(read_json(Path(item["absolute_task_path"])))
            for item in basic["entries"]
        )
    rows = []
    provisional: list[dict[str, Any]] = []
    for task_id, record in sorted(
        runtime.state["semantic_records"].items(),
        key=lambda item: (
            int(item[1].get("slate_index") or 0),
            int(item[1].get("source_development_priority") or 999),
            item[0],
        ),
    ):
        failures = _base_eligibility(
            record, allow_same_procedure=allow_same_procedure, review_policy=policy
        )
        if policy and not failures:
            try:
                verify_recorded_review(config, runtime, record)
            except (ValueError, KeyError, OSError) as exc:
                failures.append("recorded_review_evidence_invalid: " + str(exc))
        category = None
        intrinsic = record.get("intrinsic_quality")
        curriculum = record.get("curriculum_status")
        if not failures and intrinsic in {"core_quality", "reserve_quality"}:
            allowed_portfolio = {"include_now", "hold_crowded"}
            if allow_same_procedure:
                allowed_portfolio.add("same_procedure_duplicate")
            if record.get("portfolio_placement") not in allowed_portfolio:
                failures.append("portfolio_not_committable")
            else:
                category = "core_reserve"
        elif not failures and curriculum == "basic_valid":
            if policy and assess_basic_valid(record, review_policy=policy)["curriculum_status"] != "basic_valid":
                failures.append("basic_valid_strict_policy_failed")
            elif intrinsic != "rejected_quality":
                failures.append("basic_valid_intrinsic_provenance_changed")
            else:
                category = "basic_valid"
        elif not failures:
            failures.append("quality_status_not_authorized")

        task = record.get("complete_task")
        palette_hash = (
            palette_equivalent_hash(task) if isinstance(task, Mapping) else None
        )
        if not failures and palette_hash in existing:
            failures.append("already_represented_exact_or_palette_equivalent")
        row = {
            "task_id": task_id,
            "category": category,
            "intrinsic_quality": intrinsic,
            "portfolio_placement": record.get("portfolio_placement"),
            "curriculum_status": curriculum,
            "eligible": not failures,
            "exclusion_reasons": failures,
            "exact_task_hash": content_hash(task)
            if isinstance(task, Mapping)
            else None,
            "palette_equivalent_hash": palette_hash,
            "slate_index": record.get("slate_index"),
            "source_development_priority": record.get("source_development_priority"),
        }
        rows.append(row)
        if not failures:
            provisional.append({"row": row, "record": record})

    # A same payload may appear twice inside the run even if the boards missed
    # the relationship.  Keep one best editorial representative and exclude
    # every sibling from automatic commitment.
    by_palette: dict[str, list[dict[str, Any]]] = {}
    for item in provisional:
        by_palette.setdefault(str(item["row"]["palette_equivalent_hash"]), []).append(
            item
        )
    for members in by_palette.values():
        if len(members) < 2:
            continue
        ordered = sorted(
            members,
            key=lambda item: (
                item["record"].get("intrinsic_quality") != "core_quality",
                item["record"].get("portfolio_placement") != "include_now",
                int(item["record"].get("slate_index") or 0),
                int(item["record"].get("source_development_priority") or 999),
                str(item["row"]["task_id"]),
            ),
        )
        for duplicate in ordered[1:]:
            duplicate["row"]["eligible"] = False
            duplicate["row"]["exclusion_reasons"].append(
                f"within_run_palette_duplicate_of:{ordered[0]['row']['task_id']}"
            )

    result = {
        "schema_version": 1,
        "authorization_record": config["commit"]["authorization_record"],
        "rows": rows,
        "eligible_core_reserve": [
            row["task_id"]
            for row in rows
            if row["eligible"] and row["category"] == "core_reserve"
        ],
        "eligible_basic_valid": [
            row["task_id"]
            for row in rows
            if row["eligible"] and row["category"] == "basic_valid"
        ],
        "automatic_feature_promotion": False,
    }
    if policy:
        result["review_policy"] = policy
    result["counts"] = {
        "core_reserve": len(result["eligible_core_reserve"]),
        "basic_valid": len(result["eligible_basic_valid"]),
        "excluded": sum(not row["eligible"] for row in rows),
    }
    atomic_write_json(_output(config) / "commit_eligibility.json", result)
    return result


def _copy_context(config: Mapping[str, Any], destination: Path) -> None:
    outer = _output(config)
    downstream = outer / "downstream"
    for source, relative in (
        (outer / "config.snapshot.yaml", "config.snapshot.yaml"),
        (outer / "implementation_manifest.json", "implementation_manifest.json"),
        (downstream / "config.snapshot.json", "downstream.config.snapshot.json"),
        (downstream / "prompts" / "manifest.json", "prompts/manifest.json"),
        (outer / "frozen_input_manifest.json", "frozen_input_manifest.json"),
        (outer / "input_manifest.json", "factory.input_manifest.json"),
        (outer / "frozen_supply_design.private.json", "factory.supply_design.private.json"),
        (outer / "frozen_source" / "config" / "requested.yaml", "factory.requested.yaml"),
        (downstream / "runtime_manifest.json", "factory.downstream_runtime.json"),
    ):
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _staging_manifest(destination: Path, task_id: str) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        if path.name == "staging_manifest.json" and path.parent == destination:
            continue
        rows.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
            }
        )
    value = {
        "schema_version": 1,
        "task_id": task_id,
        "files": rows,
    }
    value["manifest_content_hash"] = content_hash(value)
    return value


def _verify_staging(destination: Path, task_id: str) -> dict[str, Any]:
    manifest_path = destination / "staging_manifest.json"
    if not manifest_path.is_file():
        raise E18CommitError(f"staging manifest is missing: {task_id}")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("task_id") != task_id:
        raise E18CommitError(f"staging manifest identity changed: {task_id}")
    unhashed = deepcopy(manifest)
    declared = unhashed.pop("manifest_content_hash", None)
    if declared != content_hash(unhashed):
        raise E18CommitError(f"staging manifest content hash changed: {task_id}")
    expected_paths: set[str] = set()
    root = destination.resolve()
    for row in manifest.get("files") or []:
        if not isinstance(row, Mapping):
            raise E18CommitError(f"malformed staging artifact: {task_id}")
        relative = str(row.get("path") or "")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise E18CommitError(f"staging artifact escapes root: {task_id}") from exc
        if relative in expected_paths:
            raise E18CommitError(f"duplicate staging artifact: {task_id}/{relative}")
        expected_paths.add(relative)
        if (
            not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or _sha(path) != row.get("sha256")
        ):
            raise E18CommitError(f"staging artifact changed: {task_id}/{relative}")
    actual_paths = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != expected_paths:
        raise E18CommitError(f"staging artifact set changed: {task_id}")
    return {
        "valid": True,
        "file_count": len(expected_paths),
        "manifest_sha256": _sha(manifest_path),
    }


def _stage_candidate(
    config: Mapping[str, Any], runtime: Any, task_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _root(config)
    record = deepcopy(runtime.state["semantic_records"][task_id])
    policy = review_policy(config)
    minimum_calls = 7 if policy else 8
    if policy:
        failures = _base_eligibility(record, review_policy=policy, allow_same_procedure=config["commit"].get("authorize_same_procedure_override") is True)
        if failures:
            raise E18CommitError("candidate policy eligibility failed: " + ", ".join(failures))
        verify_recorded_review(config, runtime, record)
        if record.get("intrinsic_quality") == "rejected_quality" and assess_basic_valid(record, review_policy=policy)["curriculum_status"] != "basic_valid":
            raise E18CommitError("basic_valid strict classification failed before staging")
    terminal = record.get("terminal_artifact")
    if not isinstance(terminal, Mapping):
        raise E18CommitError(f"candidate lacks terminal artifact row: {task_id}")
    try:
        verify_terminal_artifact_bundle(config, terminal, record)
    except Exception as exc:
        raise E18CommitError(
            f"terminal artifact verification failed before staging: {task_id}: {exc}"
        ) from exc
    relative = Path("runs") / f"{config['run']['id']}-commit-staging" / task_id
    destination = root / relative
    if destination.exists():
        _verify_staging(destination, task_id)
        saved = read_json(destination / "state.json")
        existing = (saved.get("semantic_records") or {}).get(task_id)
        if not isinstance(existing, Mapping):
            raise E18CommitError(f"staged candidate state is malformed: {task_id}")
        comparable = deepcopy(dict(existing))
        provenance = comparable.pop("commit_source_provenance", None)
        if comparable != record:
            raise E18CommitError(f"staged candidate changed on resume: {task_id}")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("terminal_artifact_manifest_sha256")
            != terminal.get("artifact_manifest_sha256")
            or int(provenance.get("copied_model_call_directories") or 0) < minimum_calls
        ):
            raise E18CommitError(f"staged provenance changed on resume: {task_id}")
        record = deepcopy(dict(existing))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{task_id}.building.", dir=destination.parent)
        )
        try:
            (temporary / "tasks").mkdir(parents=True, exist_ok=True)
            source_authoring = runtime.run_dir / "tasks" / f"{task_id}.authoring.json"
            if not source_authoring.is_file():
                raise E18CommitError(
                    f"canonical authoring file is missing: {task_id}"
                )
            shutil.copy2(
                source_authoring, temporary / "tasks" / f"{task_id}.authoring.json"
            )
            needles = list(
                dict.fromkeys(
                    value
                    for value in (
                        task_id,
                        str(record.get("source_candidate_id") or ""),
                        str(record.get("lineage_id") or ""),
                    )
                    if value
                )
            )
            copied = _copy_calls(
                roots=[("e18", _output(config))],
                needles=needles,
                destination=temporary / "model_calls",
                candidate_id=task_id,
            )
            if copied < minimum_calls:
                raise E18CommitError(
                    "candidate dossier would omit relevant authoring calls: "
                    f"{task_id}/{copied}"
                )
            _copy_context(config, temporary)
            if policy:
                verify_dossier_calls(config, runtime, record, temporary)
                batch = record["editorial_review_binding"]["batch_id"]
                for name in ("prepared.json", "report.json"):
                    shutil.copy2(_output(config) / "downstream/reviews" / batch / name, temporary / ("factory.review." + name))
                atomic_write_json(temporary / "factory.review_policy.json", policy)
            record["commit_source_provenance"] = {
                "schema_version": 1,
                "source_run": str(_output(config).relative_to(root)),
                "terminal_artifact_manifest": terminal["artifact_manifest"],
                "terminal_artifact_manifest_sha256": terminal[
                    "artifact_manifest_sha256"
                ],
                "copied_model_call_directories": copied,
                "call_binding_needles": needles,
                "corrected_e19_evidence_gate": True,
                "automatic_commit_explicitly_authorized": True,
            }
            atomic_write_json(
                temporary / "state.json",
                {
                    "schema_version": 1,
                    "pipeline": "arcgen-e18-commit-staging",
                    "run_id": task_id,
                    "completed": True,
                    "semantic_records": {task_id: record},
                    "automatic_task_commit": True,
                    "authorization_record": config["commit"]["authorization_record"],
                },
            )
            atomic_write_json(
                temporary / "report.json",
                {
                    "schema_version": 1,
                    "pipeline": "arcgen-e18-commit-staging",
                    "completed": True,
                    "task_id": task_id,
                    "intrinsic_quality": record.get("intrinsic_quality"),
                    "curriculum_status": record.get("curriculum_status"),
                    "source_portfolio_placement": record.get("portfolio_placement"),
                    "answer_evidence": record.get("answer_evidence"),
                    "copied_model_call_directories": copied,
                },
            )
            atomic_write_json(
                temporary / "staging_manifest.json",
                _staging_manifest(temporary, task_id),
            )
            _verify_staging(temporary, task_id)
            temporary.replace(destination)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    if policy:
        verify_dossier_calls(config, runtime, record, destination)
    registry_id = f"e18-production__{task_id}"
    same_procedure_override = (
        config.get("commit", {}).get("authorize_same_procedure_override") is True
        and _same_procedure_marked(record)
    )
    entry = {
        "source_run": relative.as_posix(),
        "source_pipeline": config["commit"].get("source_pipeline", "arcgen-e18-production-candidate"),
        "source_candidate_id": task_id,
        "registry_task_id": registry_id,
        "title": _title(record, task_id),
        "intrinsic_status": record.get("intrinsic_quality"),
        "authorize_reserve_quality": record.get("intrinsic_quality")
        == "reserve_quality",
        "authorize_portfolio_override": record.get("portfolio_placement")
        == "hold_crowded",
        "authorize_all_core_override": record.get("portfolio_placement")
        == "hold_crowded",
        "authorize_same_procedure_override": same_procedure_override,
        "qualification_task_hash": content_hash(record["complete_task"]),
    }
    return entry, record


def prepare_commit(
    config: Mapping[str, Any], runtime: Any, eligibility: Mapping[str, Any]
) -> dict[str, Any]:
    identity = _commit_identity(config)
    destination = _output(config) / "commit_transaction" / "prepared.json"
    if destination.is_file():
        value = read_json(destination)
        expected_main = list(eligibility["eligible_core_reserve"])
        expected_basic = list(eligibility["eligible_basic_valid"])
        if (
            value.get("main_task_ids") != expected_main
            or value.get("basic_task_ids") != expected_basic
            or value.get("eligibility_content_hash") != content_hash(eligibility)
        ):
            raise E18CommitError("prepared commit changed on resume")
        _verify_prepared(config, runtime, value)
        return value
    main_entries = []
    staged: dict[str, dict[str, Any]] = {}
    all_ids = [
        *eligibility["eligible_core_reserve"],
        *eligibility["eligible_basic_valid"],
    ]
    for task_id in all_ids:
        entry, record = _stage_candidate(config, runtime, task_id)
        staged[task_id] = record
        if task_id in eligibility["eligible_core_reserve"]:
            main_entries.append(entry)
    same_procedure_note = (
        " Same-procedure siblings whose ARC payloads are not exact or globally "
        "palette-equivalent are included under the user's explicit override; "
        "their source portfolio judgments remain preserved."
        if config.get("commit", {}).get("authorize_same_procedure_override") is True
        else ""
    )
    source_label = (
        "E18 " if config["commit"].get("source_pipeline", "arcgen-e18-production-candidate")
        == "arcgen-e18-production-candidate" else ""
    )
    main_manifest = {
        "schema_version": 1,
        "commit_set_id": identity["commit_set_id"],
        "committed_on": identity["committed_on"],
        "authorization_record": config["commit"]["authorization_record"],
        "notes": (
            f"Every nonduplicate {source_label}core-quality and reserve-quality task that "
            "passed exact mechanical, technical, dominant-answer evidence, "
            + ("one actual policy-bound board, " if review_policy(config) else "two-board, ") +
            "and terminal-artifact gates. Crowding holds are included under the "
            "user's explicit all-valid/reserve/core authorization."
            + same_procedure_note
            + " No feature is promoted."
        ),
        "source_pipeline": config["commit"].get("source_pipeline", "arcgen-e18-production-candidate"),
        "source_run": str(_output(config).relative_to(_root(config))),
        "entries": main_entries,
    }
    if review_policy(config):
        main_manifest["review_policy"] = review_policy(config)
    # Keep the authorization request inside the isolated run until the
    # transaction is known-good.  Writing it into ``committed_tasks`` during
    # preparation would itself be an unauthorized pre-terminal mutation.
    manifest_path = _output(config) / "commit_transaction" / "authorized_manifest.json"
    if manifest_path.is_file() and read_json(manifest_path) != main_manifest:
        raise E18CommitError("E18 authorized commit manifest changed")
    if not manifest_path.is_file():
        atomic_write_json(manifest_path, main_manifest)
    value = {
        "schema_version": 1,
        "main_manifest": str(manifest_path.resolve()),
        "main_manifest_sha256": _sha(manifest_path),
        "main_task_ids": list(eligibility["eligible_core_reserve"]),
        "basic_task_ids": list(eligibility["eligible_basic_valid"]),
        "staging_runs": {
            task_id: f"runs/{config['run']['id']}-commit-staging/{task_id}"
            for task_id in all_ids
        },
        "authorization_record": config["commit"]["authorization_record"],
        "eligibility_content_hash": content_hash(eligibility),
    }
    if review_policy(config):
        value["review_policy"] = review_policy(config)
    value["prepared_content_hash"] = content_hash(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, value)
    _verify_prepared(config, runtime, value)
    return value


def _verify_prepared(
    config: Mapping[str, Any], runtime: Any, prepared: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate the immutable handoff immediately before bank mutation."""

    identity = _commit_identity(config)
    if prepared.get("review_policy") != review_policy(config):
        raise E18CommitError("prepared required-board policy changed")

    unhashed = deepcopy(dict(prepared))
    declared = unhashed.pop("prepared_content_hash", None)
    if declared != content_hash(unhashed):
        raise E18CommitError("prepared commit content hash changed")
    if prepared.get("authorization_record") != config["commit"]["authorization_record"]:
        raise E18CommitError("prepared commit authorization changed")
    eligibility_path = _output(config) / "commit_eligibility.json"
    if (
        not eligibility_path.is_file()
        or content_hash(read_json(eligibility_path))
        != prepared.get("eligibility_content_hash")
    ):
        raise E18CommitError("prepared eligibility decision changed")
    main_ids = list(prepared.get("main_task_ids") or [])
    basic_ids = list(prepared.get("basic_task_ids") or [])
    all_ids = [*main_ids, *basic_ids]
    if len(all_ids) != len(set(all_ids)):
        raise E18CommitError("prepared commit repeats a candidate")
    semantic_ids = set(runtime.state.get("semantic_records", {}))
    if not set(all_ids) <= semantic_ids:
        raise E18CommitError("prepared commit refers to a missing semantic record")
    manifest_path = Path(str(prepared.get("main_manifest") or "")).resolve()
    expected_manifest = (
        _output(config) / "commit_transaction" / "authorized_manifest.json"
    ).resolve()
    if (
        manifest_path != expected_manifest
        or not manifest_path.is_file()
        or _sha(manifest_path) != prepared.get("main_manifest_sha256")
    ):
        raise E18CommitError("prepared authorized manifest changed")
    manifest = read_json(manifest_path)
    if (
        manifest.get("commit_set_id") != identity["commit_set_id"]
        or manifest.get("authorization_record")
        != config["commit"]["authorization_record"]
    ):
        raise E18CommitError("prepared authorized manifest policy changed")
    if manifest.get("review_policy") != review_policy(config):
        raise E18CommitError("authorized required-board policy changed")
    requested = {
        str(row.get("source_candidate_id")): row
        for row in manifest.get("entries") or []
        if isinstance(row, Mapping)
    }
    if set(requested) != set(main_ids) or len(requested) != len(main_ids):
        raise E18CommitError("prepared main-task manifest membership changed")
    staging = prepared.get("staging_runs")
    if not isinstance(staging, Mapping) or set(staging) != set(all_ids):
        raise E18CommitError("prepared staging-run membership changed")
    checks = []
    for task_id in all_ids:
        expected_relative = f"runs/{config['run']['id']}-commit-staging/{task_id}"
        if staging.get(task_id) != expected_relative:
            raise E18CommitError(f"prepared staging path changed: {task_id}")
        entry, staged_record = _stage_candidate(config, runtime, task_id)
        stage_path = _root(config) / expected_relative
        stage_check = _verify_staging(stage_path, task_id)
        if task_id in requested:
            if entry != requested[task_id]:
                raise E18CommitError(f"prepared commit entry changed: {task_id}")
        elif staged_record.get("curriculum_status") != "basic_valid":
            raise E18CommitError(
                f"prepared curriculum candidate changed status: {task_id}"
            )
        if entry.get("qualification_task_hash") != content_hash(
            staged_record.get("complete_task")
        ):
            raise E18CommitError(f"prepared task hash changed: {task_id}")
        checks.append({"task_id": task_id, **stage_check})
    return {"valid": True, "candidate_count": len(checks), "staging": checks}


def _new_basic_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "registry_id": BASIC_REGISTRY_ID,
        "frozen": True,
        "automatic_writes_allowed": False,
        "requires_explicit_user_authorization": True,
        "excluded_from_positive_generation_context": True,
        "curriculum_status_is_not_intrinsic_quality": True,
        "dossier_policy": {
            "schema_version": 1,
            "required_for_every_committed_task": True,
            "visibility": "author_only_contains_hidden_answers",
        },
        "commit_sets": [],
        "entries": [],
    }


def load_basic_valid_registry(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = read_json(source) if source.is_file() else _new_basic_registry()
    if (
        value.get("schema_version") != 1
        or value.get("registry_id") != BASIC_REGISTRY_ID
        or value.get("frozen") is not True
        or value.get("automatic_writes_allowed") is not False
        or value.get("requires_explicit_user_authorization") is not True
        or value.get("excluded_from_positive_generation_context") is not True
    ):
        raise E18CommitError("invalid basic-valid curriculum registry policy")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise E18CommitError("basic-valid registry entries must be a list")
    root = source.parent
    ids: set[str] = set()
    payloads: set[str] = set()
    verified = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise E18CommitError("malformed basic-valid registry entry")
        task_id = str(entry.get("registry_task_id") or "")
        if not task_id or task_id in ids:
            raise E18CommitError("duplicate basic-valid registry ID")
        ids.add(task_id)
        if (
            entry.get("curriculum_status") != "basic_valid"
            or entry.get("intrinsic_status") != "rejected_quality"
        ):
            raise E18CommitError("basic-valid entry changed quality provenance")
        task_path = (root / str(entry.get("task_path"))).resolve()
        if not task_path.is_file() or _sha(task_path) != entry.get("arc_sha256"):
            raise E18CommitError("basic-valid task file is missing or changed")
        validation = validate_arc_task(read_json(task_path), require_test_outputs=True)
        if not validation.valid:
            raise E18CommitError("basic-valid registry contains invalid ARC task")
        if entry["arc_sha256"] in payloads:
            raise E18CommitError("basic-valid registry repeats exact payload")
        payloads.add(entry["arc_sha256"])
        verify_dossier_manifest(committed_root=root, entry=entry)
        row = deepcopy(dict(entry))
        row["absolute_task_path"] = str(task_path)
        verified.append(row)
    declared = sum(int(item.get("entry_count") or 0) for item in value["commit_sets"])
    if declared != len(entries):
        raise E18CommitError("basic-valid commit-set counts differ from entries")
    result = deepcopy(value)
    result["entries"] = verified
    result["registry_path"] = str(source)
    return result


def _commit_basic(
    config: Mapping[str, Any], runtime: Any, task_ids: Sequence[str]
) -> dict[str, Any]:
    identity = _commit_identity(config)
    path = Path(config["_paths"]["basic_valid_registry"])
    current = load_basic_valid_registry(path)
    raw = read_json(path) if path.is_file() else _new_basic_registry()
    existing_set = {
        item["registry_task_id"]
        for item in raw["entries"]
        if item.get("commit_set_id") == identity["commit_set_id"]
    }
    requested_set = {f"e18-basic__{task_id}" for task_id in task_ids}
    existing_commit_rows = [
        item
        for item in raw["commit_sets"]
        if item.get("commit_set_id") == identity["commit_set_id"]
    ]
    if len(existing_commit_rows) > 1:
        raise E18CommitError("basic-valid commit set is duplicated")
    if existing_commit_rows:
        if existing_set != requested_set:
            raise E18CommitError("partial or changed basic-valid commit set")
        if int(existing_commit_rows[0].get("entry_count") or 0) != len(existing_set):
            raise E18CommitError("basic-valid commit-set count changed")
        return {"idempotent": True, "added": sorted(existing_set)}
    if existing_set:
        raise E18CommitError("basic-valid entries exist without their commit set")
    committed_root = path.parent
    new_entries = []
    for task_id in task_ids:
        record = runtime.state["semantic_records"][task_id]
        registry_id = f"e18-basic__{task_id}"
        target = committed_root / "basic_valid_tasks" / f"{registry_id}.json"
        if target.exists():
            raise E18CommitError(f"unexpected basic-valid target exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, record["complete_task"])
        new_entries.append(
            {
                "registry_task_id": registry_id,
                "title": _title(record, task_id),
                "task_path": target.relative_to(committed_root).as_posix(),
                "arc_sha256": _sha(target),
                "source_pipeline": config["commit"].get("source_pipeline", "arcgen-e18-production-candidate"),
                "source_run": f"runs/{config['run']['id']}-commit-staging/{task_id}",
                "source_candidate_id": task_id,
                "search_mode": record["search_mode"],
                "intrinsic_status": "rejected_quality",
                "curriculum_status": "basic_valid",
                "portfolio_status": str(
                    record.get("portfolio_placement") or "not_applicable"
                ),
                "commit_set_id": identity["commit_set_id"],
                "concept_summary": str(record["kernel"]["central_relation"]),
                "intended_rule": str(display_value(
                    record["realization"]["authoring_metadata"]["intended_rule"],
                    record.get("concrete_plan"),
                )),
                "dossier_path": f"dossiers/{registry_id}/manifest.json",
                "dossier_sha256": "pending-explicit-authorized-materialization",
                "excluded_from_positive_generation_context": True,
            }
        )
    raw["commit_sets"].append(
        {
            "commit_set_id": identity["commit_set_id"],
            "committed_on": identity["committed_on"],
            "authorization": config["commit"]["authorization_record"],
            "entry_count": len(new_entries),
            "notes": (
                "Mechanically exact, routinely blind-solved E18 tasks rejected only "
                "for editorial thinness/familiarity. Intrinsic rejected-quality "
                "provenance is preserved and these tasks are excluded from positive bank context."
            ),
        }
    )
    raw["entries"].extend(new_entries)
    atomic_write_json(path, raw)
    if new_entries:
        materialize_committed_dossiers(
            path,
            authorize_user_decision=True,
            registry_task_ids={item["registry_task_id"] for item in new_entries},
        )
    verified = load_basic_valid_registry(path)
    return {
        "idempotent": False,
        "added": [item["registry_task_id"] for item in new_entries],
        "previous_count": len(current["entries"]),
        "registry_count": len(verified["entries"]),
    }


def _write_basic_catalog(config: Mapping[str, Any]) -> None:
    path = Path(config["_paths"]["basic_valid_registry"])
    registry = load_basic_valid_registry(path)
    root = path.parent
    lines = [
        "# Committed basic-valid curriculum catalog",
        "",
        f"Count: {len(registry['entries'])}",
        "",
        "These tasks are valid/easy curriculum tasks, not core or reserve quality, and are excluded from positive bank-conditioned idea generation.",
        "",
    ]
    images = []
    render_root = root / "basic_valid_renders"
    render_root.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(registry["entries"], 1):
        lines.append(f"{index}. {entry['title']} — `{entry['registry_task_id']}`")
        task = read_json(Path(entry["absolute_task_path"]))
        image = render_episode(
            task,
            title=entry["title"],
            subtitle="basic_valid · curriculum registry",
            rule=entry["intended_rule"],
            destination=render_root / f"{index:03d}__{entry['registry_task_id']}.png",
        )
        images.append((entry["registry_task_id"], image))
    if images:
        build_contact_sheet(images, render_root / "contact_sheet.png")
    atomic_write_text(root / "BASIC_VALID_CATALOG.md", "\n".join(lines) + "\n")
    atomic_write_text(
        root / "BASIC_VALID_SHA256SUMS",
        "\n".join(
            sorted(
                f"{item['arc_sha256']}  {item['task_path']}"
                for item in registry["entries"]
            )
        )
        + ("\n" if registry["entries"] else ""),
    )


def _backup_paths(config: Mapping[str, Any], destination: Path) -> dict[str, bool]:
    root = Path(config["_paths"]["historical_registry"]).parent
    identity = _commit_identity(config)
    names = (
        "registry.json",
        "active_registry.json",
        "basic_valid_registry.json",
        "CATALOG.md",
        "SHA256SUMS",
        "BASIC_VALID_CATALOG.md",
        "BASIC_VALID_SHA256SUMS",
        f"commit_manifests/{identity['installed_manifest_filename']}",
    )
    present = {}
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = root / name
        present[name] = source.exists()
        if source.is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ("renders", "basic_valid_renders"):
        source = root / name
        present[name] = source.exists()
        if source.is_dir():
            # These renders are regenerated in place by the historical bank
            # writer.  Hard-linking the backup would let a direct overwrite
            # corrupt both the live file and its rollback copy.
            shutil.copytree(source, destination / name, copy_function=shutil.copy2)
    atomic_write_json(destination / "presence.json", present)
    return present


def _backup_manifest(backup: Path) -> dict[str, Any]:
    rows = [
        {
            "path": path.relative_to(backup).as_posix(),
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(backup.rglob("*"))
        if path.is_file() and path.name != "backup_manifest.json"
    ]
    value = {"schema_version": 1, "files": rows}
    value["manifest_content_hash"] = content_hash(value)
    return value


def _verify_backup(backup: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = backup / "backup_manifest.json"
    if not manifest_path.is_file():
        raise E18CommitError("commit rollback backup manifest is missing")
    if not expected_sha256 or _sha(manifest_path) != expected_sha256:
        raise E18CommitError("commit rollback backup manifest changed")
    manifest = read_json(manifest_path)
    unhashed = deepcopy(manifest)
    declared = unhashed.pop("manifest_content_hash", None)
    if manifest.get("schema_version") != 1 or declared != content_hash(unhashed):
        raise E18CommitError("commit rollback backup manifest is malformed")
    expected_paths: set[str] = set()
    for row in manifest.get("files") or []:
        if not isinstance(row, Mapping):
            raise E18CommitError("commit rollback backup row is malformed")
        relative = str(row.get("path") or "")
        path = (backup / relative).resolve()
        try:
            path.relative_to(backup.resolve())
        except ValueError as exc:
            raise E18CommitError("commit rollback backup path escapes root") from exc
        if relative in expected_paths:
            raise E18CommitError("commit rollback backup repeats a path")
        expected_paths.add(relative)
        if (
            not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or _sha(path) != row.get("sha256")
        ):
            raise E18CommitError(f"commit rollback backup changed: {relative}")
    actual_paths = {
        path.relative_to(backup).as_posix()
        for path in backup.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != expected_paths:
        raise E18CommitError("commit rollback backup file set changed")
    return {"valid": True, "file_count": len(expected_paths)}


def _verify_main_derived_artifacts(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = load_committed_task_registry(config["_paths"]["historical_registry"])
    root = Path(config["_paths"]["historical_registry"]).parent
    checksums = root / "SHA256SUMS"
    catalog = root / "CATALOG.md"
    render_root = root / "renders"
    render_manifest = render_root / "manifest.json"
    expected_checksums = "\n".join(
        sorted(
            row
            for entry in registry["entries"]
            for row in (
                f"{entry['arc_sha256']}  {entry['task_path']}",
                f"{entry['dossier_sha256']}  {entry['dossier_path']}",
            )
        )
    ) + "\n"
    catalog_lines = [
        "# Committed good-task catalog",
        "",
        "Commit sets: "
        + ", ".join(
            f"`{item['commit_set_id']}`" for item in registry["commit_sets"]
        ),
        f"Count: {len(registry['entries'])}",
        "Automatic additions: disabled",
        "",
    ]
    catalog_index = 1
    for commit_set in registry["commit_sets"]:
        catalog_lines.extend(
            [
                f"## {commit_set['commit_set_id']}",
                "",
                commit_set.get("notes", ""),
                "",
            ]
        )
        for entry in registry["entries"]:
            if entry["commit_set_id"] != commit_set["commit_set_id"]:
                continue
            catalog_lines.append(
                f"{catalog_index}. [{entry['title']}]({entry['task_path']}) · "
                f"[author dossier]({entry['dossier_path']})"
            )
            catalog_index += 1
        catalog_lines.append("")
    catalog_lines.extend(
        [
            "Every entry has an exact ARC payload plus a hash-bound author-only dossier containing its rule, plan, realization witness, review record, and relevant raw model-call artifacts.",
            "",
        ]
    )
    expected_catalog = "\n".join(catalog_lines)
    expected_render_rows = [
        {
            "index": index,
            "registry_task_id": entry["registry_task_id"],
            "title": entry["title"],
            "source_task": entry["task_path"],
            "render": f"{index:02d}__{entry['registry_task_id']}.png",
            "arc_sha256": entry["arc_sha256"],
            "intended_rule_included": True,
        }
        for index, entry in enumerate(registry["entries"], 1)
    ]
    expected_render_manifest = {
        "schema_version": 1,
        "source_registry": "../registry.json",
        "registry_id": registry["registry_id"],
        "task_count": len(expected_render_rows),
        "contact_sheet": "contact_sheet.png",
        "intended_rules_included": True,
        "tasks": expected_render_rows,
    }
    readme = [
        "# Committed good-task renders",
        "",
        "Each render contains the complete episode, expected test output, and intended rule.",
        "",
        f"Count: {len(registry['entries'])}",
        "",
        "## Tasks",
        "",
    ]
    readme.extend(
        f"{index}. [{entry['title']}]({index:02d}__{entry['registry_task_id']}.png) — `{entry['registry_task_id']}`"
        for index, entry in enumerate(registry["entries"], 1)
    )
    readme.extend(
        [
            "",
            "## Contact sheet",
            "",
            "[Open the complete contact sheet](contact_sheet.png).",
            "",
        ]
    )
    expected_readme = "\n".join(readme)
    rendered = read_json(render_manifest) if render_manifest.is_file() else {}
    render_rows = rendered.get("tasks") if isinstance(rendered, Mapping) else None
    rendered_ids = {
        str(item.get("registry_task_id"))
        for item in (render_rows if isinstance(render_rows, list) else [])
        if isinstance(item, Mapping)
    }
    expected_ids = {str(item["registry_task_id"]) for item in registry["entries"]}
    expected_render_files = {
        "manifest.json",
        "README.md",
        "contact_sheet.png",
        *(str(item["render"]) for item in expected_render_rows),
    }
    actual_render_files = (
        {
            path.relative_to(render_root).as_posix()
            for path in render_root.rglob("*")
            if path.is_file()
        }
        if render_root.is_dir()
        else set()
    )
    renders_complete = (
        isinstance(render_rows, list)
        and rendered == expected_render_manifest
        and rendered_ids == expected_ids
        and len(rendered_ids) == len(render_rows)
        and all(
            (render_root / str(item.get("render") or "")).is_file()
            for item in render_rows
            if isinstance(item, Mapping)
        )
        and (render_root / "contact_sheet.png").is_file()
        and (render_root / "README.md").is_file()
        and (render_root / "README.md").read_text(encoding="utf-8")
        == expected_readme
        and actual_render_files == expected_render_files
    )
    render_file_inventory = [
        {
            "path": relative,
            "sha256": _sha(render_root / relative),
            "bytes": (render_root / relative).stat().st_size,
        }
        for relative in sorted(expected_render_files)
        if (render_root / relative).is_file()
    ]
    value = {
        "catalog_exact": catalog.is_file()
        and catalog.read_text(encoding="utf-8") == expected_catalog,
        "checksums_exact": checksums.is_file()
        and checksums.read_text(encoding="utf-8") == expected_checksums,
        "renders_complete": renders_complete,
        "render_tree_content_hash": content_hash(render_file_inventory),
        "render_file_count": len(render_file_inventory),
        "registry_count": len(registry["entries"]),
    }
    value["valid"] = all(
        value[key]
        for key in ("catalog_exact", "checksums_exact", "renders_complete")
    )
    return value


def _reconcile_main_derived_artifacts(
    config: Mapping[str, Any], *, regenerate: bool
) -> dict[str, Any]:
    if regenerate:
        registry = load_committed_task_registry(
            config["_paths"]["historical_registry"]
        )
        root = Path(config["_paths"]["historical_registry"]).parent
        _write_checksums(root, registry)
        _write_catalog(root, registry)
        _render_bank(root, registry)
    audit = _verify_main_derived_artifacts(config)
    if not audit["valid"]:
        raise E18CommitError("committed main-bank derived artifacts are incomplete")
    return audit


def _verify_basic_derived_artifacts(config: Mapping[str, Any]) -> dict[str, Any]:
    registry_path = Path(config["_paths"]["basic_valid_registry"])
    registry = load_basic_valid_registry(registry_path)
    root = registry_path.parent
    catalog = root / "BASIC_VALID_CATALOG.md"
    checksums = root / "BASIC_VALID_SHA256SUMS"
    expected_checksums = "\n".join(
        sorted(
            f"{item['arc_sha256']}  {item['task_path']}"
            for item in registry["entries"]
        )
    ) + ("\n" if registry["entries"] else "")
    catalog_lines = [
        "# Committed basic-valid curriculum catalog",
        "",
        f"Count: {len(registry['entries'])}",
        "",
        "These tasks are valid/easy curriculum tasks, not core or reserve quality, and are excluded from positive bank-conditioned idea generation.",
        "",
    ]
    catalog_lines.extend(
        f"{index}. {entry['title']} — `{entry['registry_task_id']}`"
        for index, entry in enumerate(registry["entries"], 1)
    )
    expected_catalog = "\n".join(catalog_lines) + "\n"
    render_root = root / "basic_valid_renders"
    expected_render_files = {
        f"{index:03d}__{entry['registry_task_id']}.png"
        for index, entry in enumerate(registry["entries"], 1)
    }
    if registry["entries"]:
        expected_render_files.add("contact_sheet.png")
    actual_render_files = (
        {
            path.relative_to(render_root).as_posix()
            for path in render_root.rglob("*")
            if path.is_file()
        }
        if render_root.is_dir()
        else set()
    )
    renders_complete = actual_render_files == expected_render_files
    render_file_inventory = [
        {
            "path": relative,
            "sha256": _sha(render_root / relative),
            "bytes": (render_root / relative).stat().st_size,
        }
        for relative in sorted(expected_render_files)
        if (render_root / relative).is_file()
    ]
    value = {
        "catalog_exact": catalog.is_file()
        and catalog.read_text(encoding="utf-8") == expected_catalog,
        "checksums_exact": checksums.is_file()
        and checksums.read_text(encoding="utf-8") == expected_checksums,
        "renders_complete": renders_complete,
        "render_tree_content_hash": content_hash(render_file_inventory),
        "render_file_count": len(render_file_inventory),
        "registry_count": len(registry["entries"]),
    }
    value["valid"] = all(
        value[key]
        for key in ("catalog_exact", "checksums_exact", "renders_complete")
    )
    return value


def _rollback(
    config: Mapping[str, Any], backup: Path, prepared: Mapping[str, Any]
) -> None:
    committed = Path(config["_paths"]["historical_registry"]).parent
    present = read_json(backup / "presence.json")
    for name, existed in present.items():
        target = committed / name
        saved = backup / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        if existed:
            if saved.is_dir():
                shutil.copytree(saved, target, copy_function=shutil.copy2)
            elif saved.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(saved, target)
    for task_id in prepared.get("main_task_ids", []):
        registry_id = f"e18-production__{task_id}"
        (committed / "tasks" / f"{registry_id}.json").unlink(missing_ok=True)
        shutil.rmtree(committed / "dossiers" / registry_id, ignore_errors=True)
    for task_id in prepared.get("basic_task_ids", []):
        registry_id = f"e18-basic__{task_id}"
        (committed / "basic_valid_tasks" / f"{registry_id}.json").unlink(
            missing_ok=True
        )
        shutil.rmtree(committed / "dossiers" / registry_id, ignore_errors=True)


def _commit_transaction_locked(
    config: Mapping[str, Any], runtime: Any, prepared: Mapping[str, Any]
) -> dict[str, Any]:
    identity = _commit_identity(config)
    _verify_prepared(config, runtime, prepared)
    directory = _output(config) / "commit_transaction"
    journal_path = directory / "journal.json"
    if journal_path.is_file():
        journal = read_json(journal_path)
        if (
            journal.get("schema_version") != 1
            or journal.get("authorization_record")
            != config["commit"]["authorization_record"]
            or journal.get("prepared_sha256") != _sha(directory / "prepared.json")
            or not isinstance(journal.get("events"), list)
        ):
            raise E18CommitError("commit transaction journal changed")
        if journal.get("status") == "complete":
            result = journal.get("result")
            if not isinstance(result, Mapping):
                raise E18CommitError("completed commit journal lacks a result")
            result_path = directory / "result.json"
            if result_path.is_file() and read_json(result_path) != result:
                raise E18CommitError("completed commit result differs from journal")
            if not result_path.is_file():
                atomic_write_json(result_path, result)
            historical = load_committed_task_registry(
                config["_paths"]["historical_registry"]
            )
            active = load_committed_task_registry(config["_paths"]["active_registry"])
            basic = load_basic_valid_registry(config["_paths"]["basic_valid_registry"])
            if (
                len(historical["entries"]) != result.get("historical_registry_count")
                or len(active["entries"]) != result.get("active_registry_count")
                or len(basic["entries"]) != result.get("basic_valid_registry_count")
            ):
                raise E18CommitError("completed commit registry counts changed")
            installed = Path(str(result.get("installed_commit_manifest") or ""))
            if (
                not installed.is_file()
                or _sha(installed) != result.get("installed_commit_manifest_sha256")
            ):
                raise E18CommitError("completed commit manifest changed")
            if prepared.get("main_task_ids"):
                main_derived = _verify_main_derived_artifacts(config)
                expected_main_derived = (
                    (result.get("derived_artifacts") or {}).get("main")
                    if isinstance(result.get("derived_artifacts"), Mapping)
                    else None
                )
                if (
                    not main_derived["valid"]
                    or expected_main_derived != main_derived
                ):
                    raise E18CommitError("completed main-bank artifacts changed")
            basic_derived = _verify_basic_derived_artifacts(config)
            expected_basic_derived = (
                (result.get("derived_artifacts") or {}).get("basic_valid")
                if isinstance(result.get("derived_artifacts"), Mapping)
                else None
            )
            if (
                not basic_derived["valid"]
                or expected_basic_derived != basic_derived
            ):
                raise E18CommitError("completed basic-bank artifacts changed")
            return deepcopy(dict(result))
        if journal.get("status") == "rolled_back":
            raise E18CommitError("the E18 commit transaction was rolled back")
        if journal.get("status") == "rollback_failed":
            raise E18CommitError(
                "the E18 commit transaction requires manual rollback recovery"
            )
        if journal.get("status") not in {
            "prepared",
            "installing_core_reserve",
            "installing_basic_valid",
            "rebuilding_active_registry",
        }:
            raise E18CommitError("commit transaction journal status changed")
        backup = Path(str(journal.get("backup") or ""))
        if backup != (directory / "backup").resolve():
            raise E18CommitError("commit transaction backup path changed")
        _verify_backup(backup, journal.get("backup_manifest_sha256"))
    else:
        backup = directory / "backup"
        # A crash while copying the pre-state must not leave a half-backup
        # that either blocks resume or is mistaken for rollback authority.
        # With no journal, this directory can only be an orphan from that
        # pre-mutation window, so rebuild it atomically.
        if backup.exists():
            shutil.rmtree(backup)
        temporary_backup = Path(
            tempfile.mkdtemp(prefix=".backup.building.", dir=directory)
        )
        try:
            _backup_paths(config, temporary_backup)
            atomic_write_json(
                temporary_backup / "backup_manifest.json",
                _backup_manifest(temporary_backup),
            )
            temporary_backup.replace(backup)
        except BaseException:
            if temporary_backup.exists():
                shutil.rmtree(temporary_backup)
            raise
        journal = {
            "schema_version": 1,
            "status": "prepared",
            "authorization_record": config["commit"]["authorization_record"],
            "prepared_sha256": _sha(directory / "prepared.json"),
            "backup": str(backup.resolve()),
            "backup_manifest_sha256": _sha(backup / "backup_manifest.json"),
            "events": [],
        }
        atomic_write_json(journal_path, journal)
        _verify_backup(backup, journal["backup_manifest_sha256"])

    def event(status: str, detail: Mapping[str, Any]) -> None:
        journal["status"] = status
        journal["events"].append(
            {
                "status": status,
                "at_epoch_seconds": round(time.time(), 3),
                "detail": deepcopy(dict(detail)),
            }
        )
        atomic_write_json(journal_path, journal)

    quarantine = Path(config["_paths"]["quarantine_registry"])
    quarantine_before = _sha(quarantine)
    reservoir = Path(config["_paths"]["production_reservoir"])
    reservoir_before = _sha(reservoir)
    try:
        event("installing_core_reserve", {})
        if prepared["main_task_ids"]:
            main = commit_authorized_manifest(
                prepared["main_manifest"], authorize_user_decision=True
            )
        else:
            main = {
                "idempotent": True,
                "added": [],
                "registry_count": len(
                    load_committed_task_registry(
                        config["_paths"]["historical_registry"]
                    )["entries"]
                ),
            }
        main_derived = (
            _reconcile_main_derived_artifacts(
                config, regenerate=main.get("idempotent") is True
            )
            if prepared["main_task_ids"]
            else {"valid": True, "not_applicable": True}
        )
        event("installing_basic_valid", {"core_reserve": main})
        basic = _commit_basic(config, runtime, prepared["basic_task_ids"])
        event("rebuilding_active_registry", {"basic_valid": basic})
        historical_raw = read_json(Path(config["_paths"]["historical_registry"]))
        active = filter_active_registry(
            historical_raw, bank=Path(config["_paths"]["historical_registry"]).parent
        )
        atomic_write_json(Path(config["_paths"]["active_registry"]), active)
        historical = load_committed_task_registry(
            config["_paths"]["historical_registry"]
        )
        active_verified = load_committed_task_registry(
            config["_paths"]["active_registry"]
        )
        basic_verified = load_basic_valid_registry(
            config["_paths"]["basic_valid_registry"]
        )
        if _sha(quarantine) != quarantine_before:
            raise E18CommitError("commit transaction changed quarantine registry")
        if _sha(reservoir) != reservoir_before:
            raise E18CommitError("commit transaction changed production reservoir")
        _write_basic_catalog(config)
        basic_derived = _verify_basic_derived_artifacts(config)
        if not basic_derived["valid"]:
            raise E18CommitError("committed basic-bank derived artifacts are incomplete")
        installed_manifest = (
            _root(config)
            / "committed_tasks/commit_manifests"
            / identity["installed_manifest_filename"]
        )
        source_manifest = Path(prepared["main_manifest"])
        if installed_manifest.is_file():
            if installed_manifest.read_bytes() != source_manifest.read_bytes():
                raise E18CommitError("installed E18 commit manifest differs")
        else:
            shutil.copy2(source_manifest, installed_manifest)
        result = {
            "schema_version": 1,
            "commit_identity": deepcopy(identity),
            "core_reserve": main,
            "basic_valid": basic,
            "historical_registry_count": len(historical["entries"]),
            "active_registry_count": len(active_verified["entries"]),
            "basic_valid_registry_count": len(basic_verified["entries"]),
            "quarantine_registry_unchanged": True,
            "production_reservoir_unchanged": True,
            "installed_commit_manifest": str(installed_manifest.resolve()),
            "installed_commit_manifest_sha256": _sha(installed_manifest),
            "derived_artifacts": {
                "main": main_derived,
                "basic_valid": basic_derived,
            },
            "automatic_feature_promotion": False,
            "authorization_record": config["commit"]["authorization_record"],
        }
        journal["result"] = result
        event("complete", result)
        atomic_write_json(directory / "result.json", result)
        return result
    except BaseException as exc:
        try:
            _rollback(config, Path(journal["backup"]), prepared)
        except BaseException as rollback_exc:
            journal["rollback_error"] = str(rollback_exc)
            event(
                "rollback_failed",
                {"error": str(exc), "rollback_error": str(rollback_exc)},
            )
            raise E18CommitError(
                "E18 commit failed and rollback also failed; manual recovery required"
            ) from rollback_exc
        journal["rollback_error"] = None
        event("rolled_back", {"error": str(exc)})
        raise


def commit_transaction(
    config: Mapping[str, Any], runtime: Any, prepared: Mapping[str, Any]
) -> dict[str, Any]:
    with _committed_bank_lock(config):
        return _commit_transaction_locked(config, runtime, prepared)


__all__ = [
    "BASIC_REGISTRY_ID",
    "COMMIT_SET_ID",
    "E18CommitError",
    "collect_commit_eligibility",
    "commit_transaction",
    "load_basic_valid_registry",
    "prepare_commit",
]
