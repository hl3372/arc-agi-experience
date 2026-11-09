"""Deferred qualification and canonical terminal artifacts for E18."""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from .factory_contracts import plan_fidelity, display_value
from .basic_valid import assess_basic_valid
from .e18_evidence_gate import evidence_is_clear
from .lean_v3_rendering import build_contact_sheet, render_episode
from .util import atomic_write_json, atomic_write_text, content_hash, read_json
from .v4_diagnostics import check_plan_fidelity, diagnose_arc_task_v4
from .v41_palette import palette_equivalent_hash
from .validation import validate_arc_task


class E18ArtifactError(RuntimeError):
    """Raised when terminal state and exported artifacts disagree."""


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _contained(root: Path, relative: str, label: str) -> Path:
    """Resolve a manifest path while forbidding traversal outside ``root``."""

    if not isinstance(relative, str) or not relative.strip():
        raise E18ArtifactError(f"{label} has no path")
    base = root.resolve()
    path = (base / relative).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise E18ArtifactError(f"{label} escapes its artifact root") from exc
    return path


def _immutable(path: Path, value: Any, label: str) -> None:
    if path.is_file():
        if read_json(path) != value:
            raise E18ArtifactError(f"immutable {label} changed: {path}")
        return
    atomic_write_json(path, value)


def _empty_list(value: Any) -> bool:
    return isinstance(value, list) and not value


def strict_technical_clear(record: Mapping[str, Any]) -> bool:
    results = record.get("technical_results")
    if not isinstance(results, list) or not results:
        return False
    terminal = results[-1]
    response = terminal.get("response") if isinstance(terminal, Mapping) else None
    if not isinstance(response, Mapping):
        return False
    return (
        terminal.get("passed") is True
        and terminal.get("hidden_output_match") is True
        and response.get("status") == "pass"
        and response.get("mechanically_correct") is True
        and response.get("test_supported") is True
        and response.get("one_answer_supported") is True
        and response.get("plan_fidelity") != "fail"
        and response.get("evidence_sufficiency") == "sufficient"
        and response.get("pair_economy") != "fail"
        and response.get("composition_preserved") is True
        and _empty_list(response.get("arbitrary_exceptions"))
        and _empty_list(response.get("unsupported_test_semantics"))
        and _empty_list(response.get("unapproved_realizer_decisions"))
        and not list(terminal.get("unresolved_pair_economy_flags") or [])
    )


def boardless_disposition(record: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    task = record.get("complete_task")
    validation = validate_arc_task(task, require_test_outputs=True)
    if not validation.valid:
        reasons.append("mechanically_invalid")
    if (record.get("deterministic_diagnostics") or {}).get("hard_valid") is not True:
        reasons.append("deterministic_hard_invalid")
    if (record.get("plan_fidelity") or {}).get("passed") is not True:
        reasons.append("plan_fidelity_failed")
    if not strict_technical_clear(record):
        reasons.append("technical_not_exact_clear")
    if not evidence_is_clear(record):
        reasons.append("answer_evidence_not_clear")
    novelty = record.get("novelty") or {}
    if novelty.get("confirmed_same_procedure_duplicate") is True:
        reasons.append("confirmed_same_procedure_duplicate")
    return {
        "boardless_disposition": "verified_candidate"
        if not reasons
        else "not_verified",
        "neutral_not_intrinsic_quality": True,
        "reasons": reasons,
        "arc_validation_errors": list(validation.errors),
        "complete_task_hash": content_hash(task) if isinstance(task, Mapping) else None,
    }


def freeze_boardless(config: Mapping[str, Any], runtime: Any) -> dict[str, Any]:
    path = Path(config["_output_directory"]) / "boardless_dispositions.frozen.json"
    if path.is_file():
        value = read_json(path)
        for row in value.get("candidates", []):
            record = runtime.state["semantic_records"].get(row.get("task_id"))
            if not isinstance(record, Mapping):
                raise E18ArtifactError("boardless freeze refers to a missing candidate")
            if record.get("boardless_disposition") != row.get("boardless_disposition"):
                raise E18ArtifactError("persisted boardless disposition changed")
            if list(record.get("boardless_reasons") or []) != list(
                row.get("reasons") or []
            ):
                raise E18ArtifactError("persisted boardless reasons changed")
        return value
    rows = []
    for task_id, record in sorted(runtime.state.get("semantic_records", {}).items()):
        decision = boardless_disposition(record)
        rows.append(
            {
                "task_id": task_id,
                "source_candidate_id": record["source_candidate_id"],
                "slate_index": record["slate_index"],
                **decision,
            }
        )
    value = {
        "schema_version": 1,
        "pipeline": "arcgen-e18-production-candidate",
        "frozen_before_editorial_boards_and_audits": True,
        "verified_candidate_is_not_core_quality": True,
        "candidates": rows,
        "semantic_state_hash": content_hash(runtime.state.get("semantic_records", {})),
    }
    _immutable(path, value, "E18 boardless disposition")
    with runtime.lock:
        for row in rows:
            record = runtime.state["semantic_records"][row["task_id"]]
            prior = record.get("boardless_disposition")
            if prior is not None and prior != row["boardless_disposition"]:
                raise E18ArtifactError("boardless disposition changed on resume")
            record["boardless_disposition"] = row["boardless_disposition"]
            record["boardless_reasons"] = list(row["reasons"])
        runtime.save()
    return value


def run_deferred_boards(runtime: Any) -> dict[str, Any]:
    """Run the exact two fixed-anchor boards without changing pre-board evidence."""

    from .v41_test_challenge import _board

    before = {
        task_id: {
            key: deepcopy(record.get(key))
            for key in (
                "complete_task",
                "concrete_plan",
                "realization",
                "technical_results",
                "blind_results",
                "answer_evidence",
                "boardless_disposition",
            )
        }
        for task_id, record in runtime.state["semantic_records"].items()
    }
    try:
        _board(runtime)
    finally:
        for task_id, frozen in before.items():
            current = runtime.state["semantic_records"][task_id]
            for key, expected in frozen.items():
                if current.get(key) != expected:
                    raise E18ArtifactError(
                        f"deferred board changed frozen field {task_id}:{key}"
                    )
    path = runtime.run_dir / "editorial_board_results.json"
    return {
        "completed": path.is_file(),
        "path": str(path.resolve()),
        "sha256": _sha(path) if path.is_file() else None,
        "board_call_count": sum(
            row.get("role") == "editorial_board"
            for row in runtime.state.get("call_ledger", {}).values()
        ),
    }


def run_mechanism_audits(runtime: Any) -> dict[str, Any]:
    """Run the saved arm-blind, non-gating open-ended mechanism auditors."""

    from .v4_integrated_validation import _feature_audits

    before = content_hash(
        {
            task_id: {
                key: record.get(key)
                for key in (
                    "complete_task",
                    "concrete_plan",
                    "realization",
                    "technical_results",
                    "blind_results",
                    "answer_evidence",
                    "intrinsic_quality",
                    "portfolio_placement",
                )
            }
            for task_id, record in runtime.state["semantic_records"].items()
        }
    )
    try:
        result = _feature_audits(runtime)
    finally:
        after = content_hash(
            {
                task_id: {
                    key: record.get(key)
                    for key in (
                        "complete_task",
                        "concrete_plan",
                        "realization",
                        "technical_results",
                        "blind_results",
                        "answer_evidence",
                        "intrinsic_quality",
                        "portfolio_placement",
                    )
                }
                for task_id, record in runtime.state["semantic_records"].items()
            }
        )
        if before != after:
            raise E18ArtifactError(
                "non-gating mechanism audit changed task disposition"
            )
    return result


def attach_curriculum_statuses(runtime: Any, *, review_policy=None) -> dict[str, Any]:
    rows = []
    with runtime.lock:
        for task_id, record in sorted(runtime.state["semantic_records"].items()):
            if evidence_is_clear(record):
                value = assess_basic_valid(record, review_policy=review_policy)
            else:
                value = {
                    "curriculum_status": "not_basic_valid",
                    "intrinsic_quality_preserved": record.get("intrinsic_quality"),
                    "routine_solvability_proxy": False,
                    "reasons": ["answer evidence is not terminally clear"],
                }
            record["curriculum_assessment"] = deepcopy(value)
            record["curriculum_status"] = value["curriculum_status"]
            rows.append({"task_id": task_id, **deepcopy(value)})
        runtime.save()
    result = {
        "schema_version": 1,
        "parallel_to_intrinsic_quality": True,
        "basic_valid_is_not_reserve": True,
        "rows": rows,
        "counts": dict(
            sorted(Counter(row["curriculum_status"] for row in rows).items())
        ),
    }
    atomic_write_json(runtime.run_dir / "curriculum_status_report.json", result)
    return result


def _public_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train": deepcopy(task["train"]),
        "test": [{"input": deepcopy(pair["input"])} for pair in task["test"]],
    }


def _normalized_palette(record: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(record.get("palette_treatment") or {})
    task = record.get("complete_task")
    if isinstance(task, Mapping):
        value["final_exact_task_hash"] = content_hash(task)
        value["palette_equivalent_identity_hash"] = palette_equivalent_hash(task)
        value["normalized_from_terminal_semantic_state"] = True
    return value


def _write_candidate_bundle(
    config: Mapping[str, Any], runtime: Any, task_id: str, record: Mapping[str, Any]
) -> dict[str, Any]:
    directory = Path(config["_output_directory"]) / "candidates" / task_id
    directory.mkdir(parents=True, exist_ok=True)
    revision = int(record.get("repair_count") or 0)
    files: dict[str, Path] = {}

    def store(name: str, value: Any) -> None:
        path = directory / name
        _immutable(path, value, f"candidate {task_id} {name}")
        files[name] = path

    store(
        "lineage.json",
        {
            "schema_version": 1,
            "task_id": task_id,
            "source_candidate_id": record["source_candidate_id"],
            "lineage_id": record["lineage_id"],
            "slate_index": record["slate_index"],
            "private_source": record["private_source"],
            "private_conditioned": record["private_conditioned"],
            "private_route": record["private_route"],
            "search_mode": record["search_mode"],
        },
    )
    store("kernel.json", deepcopy(record["kernel"]))
    store("selector_brief.json", deepcopy(record["editorial_brief"]))
    store(
        "disposition.json",
        {
            "status": record.get("status"),
            "boardless_disposition": record.get("boardless_disposition"),
            "intrinsic_quality": record.get("intrinsic_quality"),
            "portfolio_placement": record.get("portfolio_placement"),
            "curriculum_status": record.get("curriculum_status"),
            "answer_evidence": deepcopy(record.get("answer_evidence")),
            "same_procedure_duplicate": record.get("portfolio_placement")
            == "same_procedure_duplicate",
        },
    )

    task = record.get("complete_task")
    canonical_files: list[dict[str, Any]] = []
    if isinstance(task, Mapping):
        validation = validate_arc_task(task, require_test_outputs=True)
        if not validation.valid:
            raise E18ArtifactError(
                f"terminal task is invalid {task_id}: {'; '.join(validation.errors)}"
            )
        realization = record.get("realization")
        plan = record.get("concrete_plan")
        if not isinstance(realization, Mapping) or not isinstance(plan, Mapping):
            raise E18ArtifactError("completed task lacks plan or realization")
        if plan_fidelity(plan, realization).get("passed") is not True:
            raise E18ArtifactError("terminal plan and realization are not faithful")
        store(f"plan.r{revision}.json", deepcopy(plan))
        store(f"realization.r{revision}.json", deepcopy(realization))
        store(f"task.private.r{revision}.json", deepcopy(task))
        public = _public_task(task)
        if any("output" in pair for pair in public["test"]):
            raise E18ArtifactError("hidden output entered public terminal task")
        store(f"task.public.r{revision}.json", public)
        store(
            f"diagnostics.r{revision}.json",
            {
                "deterministic": deepcopy(record.get("deterministic_diagnostics")),
                "technical_public": deepcopy(record.get("technical_diagnostics")),
                "plan_fidelity": deepcopy(record.get("plan_fidelity")),
            },
        )
        store(f"technical.r{revision}.json", deepcopy(record.get("technical_results")))
        store(f"blind.r{revision}.json", deepcopy(record.get("blind_results")))
        store(
            f"evidence_resolution.r{revision}.json",
            {
                "answer_evidence": deepcopy(record.get("answer_evidence")),
                "history": deepcopy(record.get("answer_evidence_history") or []),
                "ambiguity_adjudication": deepcopy(
                    record.get("ambiguity_adjudication")
                ),
                "repair_count": revision,
            },
        )
        store(f"palette.r{revision}.json", _normalized_palette(record))
        # Direct canonical files are the paths consumed by the existing
        # authorized dossier materializer.  Initial realization files under
        # tasks/single remain untouched as immutable pre-repair history.
        canonical = runtime.run_dir / "tasks"
        canonical.mkdir(parents=True, exist_ok=True)
        atomic_write_json(canonical / f"{task_id}.arc.json", task)
        atomic_write_json(
            canonical / f"{task_id}.authoring.json",
            {
                "schema_version": 1,
                "candidate_id": task_id,
                "task": deepcopy(task),
                "concrete_plan": deepcopy(plan),
                "realization": deepcopy(realization),
                "palette_treatment": _normalized_palette(record),
                "terminal_revision": revision,
            },
        )
        output = Path(config["_output_directory"]).resolve()
        for kind, path in (
            ("task", canonical / f"{task_id}.arc.json"),
            ("authoring", canonical / f"{task_id}.authoring.json"),
        ):
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(output).as_posix()
            except ValueError as exc:
                raise E18ArtifactError(
                    f"canonical {kind} file escaped the E18 output root"
                ) from exc
            canonical_files.append(
                {
                    "kind": kind,
                    "path": relative,
                    "sha256": _sha(resolved),
                    "bytes": resolved.stat().st_size,
                }
            )
        initial_task = runtime.run_dir / "tasks" / "single" / f"{task_id}.arc.json"
        initial_authoring = (
            runtime.run_dir / "tasks" / "single" / f"{task_id}.authoring.json"
        )
        history = []
        for source in (initial_task, initial_authoring):
            if source.is_file():
                destination = directory / "pre_repair_history" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_file():
                    shutil.copy2(source, destination)
                files[destination.relative_to(directory).as_posix()] = destination
                history.append(
                    {
                        "path": destination.relative_to(directory).as_posix(),
                        "sha256": _sha(destination),
                    }
                )
        store(
            "revision_history.json", {"terminal_revision": revision, "files": history}
        )

    manifest = {
        "schema_version": 2,
        "task_id": task_id,
        "terminal_revision": revision,
        "terminal_semantic_record_hash": content_hash(
            {key: value for key, value in record.items() if key != "terminal_artifact"}
        ),
        "files": [
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(files.values())
        ],
        # These are intentionally outside the per-candidate directory: the
        # established authorized-commit/dossier machinery consumes these
        # canonical paths.  Binding them here prevents a stale pre-repair file
        # from being committed after the candidate bundle itself passed.
        "canonical_files": canonical_files,
    }
    manifest["manifest_content_hash"] = content_hash(manifest)
    _immutable(
        directory / "artifact_manifest.json", manifest, f"candidate {task_id} manifest"
    )
    return {
        "task_id": task_id,
        "artifact_manifest": str((directory / "artifact_manifest.json").resolve()),
        "artifact_manifest_sha256": _sha(directory / "artifact_manifest.json"),
        "terminal_revision": revision,
        "complete_task_hash": content_hash(task) if isinstance(task, Mapping) else None,
        "palette_equivalent_hash": (
            palette_equivalent_hash(task) if isinstance(task, Mapping) else None
        ),
        "canonical_files": deepcopy(canonical_files),
    }


def verify_terminal_artifact_bundle(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every terminal byte and its binding to semantic state.

    The row binds the manifest, the manifest binds both its private candidate
    bundle and the canonical task/authoring files consumed by commitment, and
    (when supplied) ``record`` binds all of those artifacts to terminal state.
    """

    output = Path(config["_output_directory"]).resolve()
    task_id = str(row.get("task_id") or "")
    if not task_id:
        raise E18ArtifactError("terminal artifact row has no task_id")
    expected_manifest = (output / "candidates" / task_id / "artifact_manifest.json").resolve()
    manifest_path = Path(str(row.get("artifact_manifest") or "")).resolve()
    if manifest_path != expected_manifest or not manifest_path.is_file():
        raise E18ArtifactError(f"terminal manifest path is invalid: {task_id}")
    if _sha(manifest_path) != row.get("artifact_manifest_sha256"):
        raise E18ArtifactError(f"terminal manifest hash changed: {task_id}")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 2 or manifest.get("task_id") != task_id:
        raise E18ArtifactError(f"terminal manifest identity changed: {task_id}")
    declared_content_hash = manifest.get("manifest_content_hash")
    unhashed = deepcopy(manifest)
    unhashed.pop("manifest_content_hash", None)
    if declared_content_hash != content_hash(unhashed):
        raise E18ArtifactError(f"terminal manifest content hash changed: {task_id}")
    if manifest.get("terminal_revision") != row.get("terminal_revision"):
        raise E18ArtifactError(f"terminal revision changed: {task_id}")

    candidate_root = manifest_path.parent
    seen: set[str] = set()
    for item in manifest.get("files") or []:
        if not isinstance(item, Mapping):
            raise E18ArtifactError(f"malformed candidate artifact entry: {task_id}")
        relative = str(item.get("path") or "")
        if relative in seen:
            raise E18ArtifactError(f"duplicate candidate artifact path: {task_id}")
        seen.add(relative)
        path = _contained(candidate_root, relative, f"candidate {task_id} artifact")
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or _sha(path) != item.get("sha256")
        ):
            raise E18ArtifactError(f"candidate artifact changed: {task_id}/{relative}")

    canonical = manifest.get("canonical_files")
    if not isinstance(canonical, list):
        raise E18ArtifactError(f"canonical artifact list is missing: {task_id}")
    if deepcopy(canonical) != list(row.get("canonical_files") or []):
        raise E18ArtifactError(f"canonical artifact row differs: {task_id}")
    by_kind: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for item in canonical:
        if not isinstance(item, Mapping):
            raise E18ArtifactError(f"malformed canonical artifact entry: {task_id}")
        kind = str(item.get("kind") or "")
        if kind in by_kind or kind not in {"task", "authoring"}:
            raise E18ArtifactError(f"invalid canonical artifact kind: {task_id}/{kind}")
        path = _contained(output, str(item.get("path") or ""), f"canonical {kind}")
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or _sha(path) != item.get("sha256")
        ):
            raise E18ArtifactError(f"canonical artifact changed: {task_id}/{kind}")
        by_kind[kind] = (path, item)

    completed = row.get("complete_task_hash") is not None
    if completed and set(by_kind) != {"task", "authoring"}:
        raise E18ArtifactError(f"completed task lacks canonical artifacts: {task_id}")
    if not completed and by_kind:
        raise E18ArtifactError(f"incomplete task unexpectedly has canonical artifacts: {task_id}")
    if completed:
        task = read_json(by_kind["task"][0])
        authoring = read_json(by_kind["authoring"][0])
        if content_hash(task) != row.get("complete_task_hash"):
            raise E18ArtifactError(f"canonical task hash differs: {task_id}")
        if palette_equivalent_hash(task) != row.get("palette_equivalent_hash"):
            raise E18ArtifactError(f"canonical palette identity differs: {task_id}")
        if not isinstance(authoring, Mapping) or authoring.get("task") != task:
            raise E18ArtifactError(f"canonical authoring task differs: {task_id}")
        if authoring.get("terminal_revision") != row.get("terminal_revision"):
            raise E18ArtifactError(f"canonical authoring revision differs: {task_id}")
        if record is not None:
            if record.get("complete_task") != task:
                raise E18ArtifactError(f"terminal semantic task differs: {task_id}")
            if authoring.get("concrete_plan") != record.get("concrete_plan"):
                raise E18ArtifactError(f"terminal semantic plan differs: {task_id}")
            if authoring.get("realization") != record.get("realization"):
                raise E18ArtifactError(f"terminal semantic realization differs: {task_id}")
            if int(record.get("repair_count") or 0) != row.get("terminal_revision"):
                raise E18ArtifactError(f"terminal semantic revision differs: {task_id}")
            stored = record.get("terminal_artifact")
            if stored is not None and dict(stored) != dict(row):
                raise E18ArtifactError(f"terminal artifact state row differs: {task_id}")
    return {
        "task_id": task_id,
        "valid": True,
        "manifest_sha256": _sha(manifest_path),
        "candidate_file_count": len(seen),
        "canonical_file_count": len(by_kind),
    }


def materialize_terminal_artifacts(
    config: Mapping[str, Any], runtime: Any
) -> dict[str, Any]:
    rows = []
    seen_exact: dict[str, str] = {}
    seen_palette: dict[str, str] = {}
    for task_id, record in sorted(runtime.state["semantic_records"].items()):
        row = _write_candidate_bundle(config, runtime, task_id, record)
        exact = row.get("complete_task_hash")
        palette = row.get("palette_equivalent_hash")
        row["within_run_exact_duplicate_of"] = (
            seen_exact.get(str(exact)) if exact else None
        )
        row["within_run_palette_duplicate_of"] = (
            seen_palette.get(str(palette)) if palette else None
        )
        if exact:
            seen_exact.setdefault(str(exact), task_id)
        if palette:
            seen_palette.setdefault(str(palette), task_id)
        rows.append(row)
    report = {
        "schema_version": 2,
        "candidate_count": len(rows),
        "completed_payload_count": sum(
            row["complete_task_hash"] is not None for row in rows
        ),
        "all_terminal_artifacts_reconciled": True,
        "rows": rows,
    }
    atomic_write_json(
        Path(config["_output_directory"]) / "terminal_artifact_audit.json", report
    )
    with runtime.lock:
        by_id = {row["task_id"]: row for row in rows}
        for task_id, row in by_id.items():
            runtime.state["semantic_records"][task_id]["terminal_artifact"] = deepcopy(
                row
            )
        runtime.save()
    return report


def render_tasks(config: Mapping[str, Any], runtime: Any) -> dict[str, Any]:
    directory = Path(config["_output_directory"]) / "renders"
    directory.mkdir(parents=True, exist_ok=True)
    images = []
    rows = []
    for task_id, record in sorted(runtime.state["semantic_records"].items()):
        task = record.get("complete_task")
        if not isinstance(task, Mapping):
            continue
        rule = str(
            ((record.get("realization") or {}).get("authoring_metadata") or {}).get(
                "intended_rule"
            )
            or ""
        )
        rule = display_value(rule, record.get("concrete_plan"))
        path = directory / f"{task_id}.png"
        image = render_episode(
            task,
            title=task_id,
            subtitle=(
                f"{record.get('intrinsic_quality', 'unreviewed')} · "
                f"{record.get('portfolio_placement', 'not_applicable')}"
            ),
            rule=rule,
            destination=path,
        )
        images.append((task_id, image))
        rows.append(
            {
                "task_id": task_id,
                "path": str(path.resolve()),
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
                "complete_task_hash": content_hash(task),
                "grid_metrics": diagnose_arc_task_v4(task)["canvas_summary"],
            }
        )
    contact = directory / "contact_sheet.png"
    if images:
        build_contact_sheet(images, contact)
    report = {
        "schema_version": 1,
        "task_count": len(rows),
        "contact_sheet": str(contact.resolve()) if contact.is_file() else None,
        "contact_sheet_sha256": _sha(contact) if contact.is_file() else None,
        "contact_sheet_bytes": contact.stat().st_size if contact.is_file() else None,
        "tasks": rows,
    }
    report["manifest_content_hash"] = content_hash(report)
    atomic_write_json(directory / "manifest.json", report)
    atomic_write_text(
        directory / "README.md",
        "# E18 realized tasks\n\nEvery PNG includes the intended rule for author review.\n",
    )
    return report


def verify_render_artifacts(
    config: Mapping[str, Any], renders: Mapping[str, Any], runtime: Any
) -> dict[str, Any]:
    """Bind rendered review artifacts to the same terminal task payloads."""

    output = Path(config["_output_directory"]).resolve()
    manifest_path = output / "renders" / "manifest.json"
    if not manifest_path.is_file() or read_json(manifest_path) != dict(renders):
        raise E18ArtifactError("render manifest differs from the audited result")
    unhashed = deepcopy(dict(renders))
    declared = unhashed.pop("manifest_content_hash", None)
    if declared != content_hash(unhashed):
        raise E18ArtifactError("render manifest content hash changed")
    expected_ids = {
        task_id
        for task_id, record in runtime.state.get("semantic_records", {}).items()
        if isinstance(record.get("complete_task"), Mapping)
    }
    rows = renders.get("tasks")
    if not isinstance(rows, list) or len(rows) != renders.get("task_count"):
        raise E18ArtifactError("render task count differs")
    actual_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise E18ArtifactError("malformed render entry")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in actual_ids:
            raise E18ArtifactError("duplicate or missing render task ID")
        actual_ids.add(task_id)
        expected_path = (output / "renders" / f"{task_id}.png").resolve()
        path = Path(str(row.get("path") or "")).resolve()
        record = runtime.state["semantic_records"].get(task_id)
        if (
            path != expected_path
            or not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or _sha(path) != row.get("sha256")
            or not isinstance(record, Mapping)
            or content_hash(record.get("complete_task"))
            != row.get("complete_task_hash")
        ):
            raise E18ArtifactError(f"render changed or is misbound: {task_id}")
    if actual_ids != expected_ids:
        raise E18ArtifactError("render set differs from terminal completed-task set")
    contact_value = renders.get("contact_sheet")
    if rows:
        expected_contact = (output / "renders" / "contact_sheet.png").resolve()
        contact = Path(str(contact_value or "")).resolve()
        if (
            contact != expected_contact
            or not contact.is_file()
            or contact.stat().st_size != renders.get("contact_sheet_bytes")
            or _sha(contact) != renders.get("contact_sheet_sha256")
        ):
            raise E18ArtifactError("render contact sheet changed")
    elif contact_value is not None:
        raise E18ArtifactError("empty render set unexpectedly has a contact sheet")
    return {
        "valid": True,
        "task_count": len(actual_ids),
        "manifest_sha256": _sha(manifest_path),
        "contact_sheet_verified": bool(rows),
    }


__all__ = [
    "E18ArtifactError",
    "attach_curriculum_statuses",
    "boardless_disposition",
    "freeze_boardless",
    "materialize_terminal_artifacts",
    "render_tasks",
    "run_deferred_boards",
    "run_mechanism_audits",
    "strict_technical_clear",
    "verify_render_artifacts",
    "verify_terminal_artifact_bundle",
]
