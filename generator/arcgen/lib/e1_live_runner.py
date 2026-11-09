"""Live post-selection authoring adapter for the E1 boardless experiment.

The E1 disposition policy lives in :mod:`e1_boardless_qualification`.  This
module supplies the missing live path: it takes a source-blind selected slate,
runs the already-qualified planner/realizer/technical/blind functions, stops
before editorial review, normalizes their evidence, and freezes the neutral
E1 dispositions.  Editorial boards are dispatched only by the separate
``shadow`` command after that freeze exists.

No prompt text is copied here.  Prompt templates, schemas, model routes, and
anchors are loaded by the selected baseline's existing runtime.  The adapter
also never mutates the production reservoir, accepted bank, or novelty memory;
all selected kernels are supplied through an immutable manifest and all run
artifacts are written below the E1 output directory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence

from .e1_boardless_qualification import (
    E1QualificationError,
    NormalizedSelectedBaselineAdapter,
    attach_shadow_results,
    effective_blind_serious,
    freeze_boardless_dispositions,
    load_e1_config,
)
from .util import atomic_write_json, content_hash, read_json
from .v4_pipeline import v4_call_accounting


PIPELINE = "arcgen-e1-live-boardless-authoring"
VERSION = "1.1"

V44_ORDINARY = "v4_4_all_sol_ordinary"
V45_SCALE = "v4_5_functional_scale"
V46_SUPPLY = "v4_6_minority_opus_supply_sol_downstream"
COMPOSITE = "minority_opus_supply_optional_functional_scale"
RUNNER_KINDS = frozenset({V44_ORDINARY, V45_SCALE, V46_SUPPLY, COMPOSITE})
MINORITY_OPUS_SUPPLY_POLICY = {
    "policy": "minority_opus_idea_supply",
    "selectable_idea_call_count": 12,
    "opus_selectable_idea_call_count": 3,
    "opus_fraction": 0.25,
    "source_balancing_enabled": False,
    "source_labels_hidden_from_selectors": True,
}

REQUIRED_PROMPT_CONTRACT = (
    "concrete_planner",
    "v44_quality_plan",
    "task_realizer",
    "technical_verifier",
    "blind_solver",
    "ambiguity_adjudicator",
    "targeted_repair",
    "editorial_board",
)
REQUIRED_SCHEMA_CONTRACT = (
    "v4_concrete_plan",
    "v4_task_realizer",
    "v4_technical_verifier",
    "v4_blind_solver",
    "v4_ambiguity_adjudicator",
    "v4_targeted_repair",
    "v4_editorial_board",
)
REQUIRED_IMPLEMENTATION_BASENAMES = frozenset(
    {
        "v41_experiments.py",
        "v41_schemas.py",
        "v41_supported_departure.py",
        "v4_integrated_validation.py",
        "v44_functional_scale.py",
    }
)


class E1LiveError(RuntimeError):
    """Raised when live E1 isolation or source fidelity is violated."""


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E1LiveError(f"expected a mapping: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _file_ref(value: Mapping[str, Any], *, root: Path, field: str) -> Path:
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str) or len(digest) != 64:
        raise E1LiveError(f"{field} must contain path and SHA-256")
    path = _resolve(root, path_value)
    if not path.is_file() or _sha(path) != digest:
        raise E1LiveError(f"{field} is missing or changed: {path}")
    return path


def _assert_false(mapping: Mapping[str, Any], field: str) -> None:
    if mapping.get(field) is not False:
        raise E1LiveError(f"{field} must be false")


def _runtime_contract(
    baseline: NormalizedSelectedBaselineAdapter, *, kind: str
) -> dict[str, Any]:
    execution = baseline.value["baseline"].get("live_execution")
    if not isinstance(execution, Mapping):
        raise E1LiveError("selected baseline must freeze baseline.live_execution")
    if execution.get("backend") != "v41_integrated_post_selection_v1":
        raise E1LiveError(
            "only the exact V4.1 integrated post-selection backend is supported"
        )
    contract = execution.get("runtime_contract")
    if not isinstance(contract, Mapping):
        raise E1LiveError("baseline.live_execution.runtime_contract is required")
    digest = contract.get("resolved_adapter_config_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise E1LiveError("runtime contract must freeze the resolved adapter config hash")
    if type(contract.get("model_concurrency")) is not int or contract["model_concurrency"] < 1:
        raise E1LiveError("runtime contract must freeze positive model_concurrency")
    if contract.get("escalation_calls_per_slate") != 5:
        raise E1LiveError("runtime contract must retain the exact five-call escalation ceiling")
    prompts = contract.get("prompts")
    required_prompts = set(REQUIRED_PROMPT_CONTRACT)
    if kind in {V45_SCALE, COMPOSITE}:
        required_prompts.add("functional_scale_addendum")
    if not isinstance(prompts, Mapping) or not required_prompts <= set(prompts):
        raise E1LiveError(
            "runtime contract omits required prompt bindings: "
            + ", ".join(sorted(required_prompts - set(prompts or {})))
        )
    top_prompts = baseline.value["prompts"]
    for name in sorted(required_prompts):
        item = prompts[name]
        if not isinstance(item, Mapping):
            raise E1LiveError(f"runtime prompt contract {name} must be a mapping")
        _file_ref(item, root=baseline.base, field=f"runtime_contract.prompts.{name}")
        if not all(isinstance(item.get(key), str) and item[key] for key in ("version", "stage")):
            raise E1LiveError(f"runtime prompt {name} must freeze path/version/stage/hash")
        top = top_prompts.get(name)
        if not isinstance(top, Mapping) or (
            top.get("sha256"), top.get("version")
        ) != (item.get("sha256"), item.get("version")):
            raise E1LiveError(f"top-level and runtime prompt contracts disagree for {name}")
    schemas = contract.get("schemas")
    required_schemas = set(REQUIRED_SCHEMA_CONTRACT)
    if kind in {V45_SCALE, COMPOSITE}:
        required_schemas.add("v44_functional_scale_plan")
    if not isinstance(schemas, Mapping) or not required_schemas <= set(schemas):
        raise E1LiveError(
            "runtime contract omits required schemas: "
            + ", ".join(sorted(required_schemas - set(schemas or {})))
        )
    for stage in sorted(required_schemas):
        item = schemas[stage]
        if not isinstance(item, Mapping) or item.get("stage") != stage:
            raise E1LiveError(f"runtime schema {stage} must freeze its exact stage")
        _file_ref(item, root=baseline.base, field=f"runtime_contract.schemas.{stage}")
    parser = contract.get("parser")
    if not isinstance(parser, Mapping):
        raise E1LiveError("runtime contract must freeze the structured-output parser")
    if (
        parser.get("module") != "arcgen.lib.v41_schemas"
        or parser.get("callable") != "parse_stage_output"
    ):
        raise E1LiveError("runtime contract names an unsupported structured-output parser")
    _file_ref(parser, root=baseline.base, field="runtime_contract.parser")
    implementations = contract.get("implementation_files")
    if not isinstance(implementations, list) or not implementations:
        raise E1LiveError("runtime contract must freeze implementation_files")
    implementation_basenames: set[str] = set()
    for index, item in enumerate(implementations):
        if not isinstance(item, Mapping):
            raise E1LiveError("runtime implementation file reference must be a mapping")
        implementation_basenames.add(
            _file_ref(
                item,
                root=baseline.base,
                field=f"runtime_contract.implementation_files[{index}]",
            ).name
        )
    missing_implementations = REQUIRED_IMPLEMENTATION_BASENAMES - implementation_basenames
    if missing_implementations:
        raise E1LiveError(
            "runtime contract omits exact implementation sources: "
            + ", ".join(sorted(missing_implementations))
        )
    return deepcopy(dict(contract))


def load_live_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = _read_yaml_or_json(source)
    if value.get("pipeline") != PIPELINE or str(value.get("version")) != VERSION:
        raise E1LiveError(f"live config must target {PIPELINE} {VERSION}")
    if value.get("template") is not False or value.get("enabled") is not True:
        raise E1LiveError("create an explicitly enabled run config from the live template")
    root = source.parent.parent if source.parent.name == "config" else source.parent
    run = value.get("run")
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise E1LiveError("run.id is required")
    for field in (
        "automatically_commit_generated_tasks",
        "production_state_mutation_allowed",
        "routine_editorial_boards_enabled",
        "auxiliary_opus_routes_enabled",
    ):
        _assert_false(run, field)
    if type(run.get("mock")) is not bool:
        raise E1LiveError("run.mock must be boolean")
    kind = value.get("runner_kind")
    if kind not in RUNNER_KINDS:
        raise E1LiveError(f"unsupported runner_kind: {kind!r}")
    for name in ("selected_baseline_manifest", "selected_slate_manifest"):
        ref = value.get(name)
        if not isinstance(ref, Mapping):
            raise E1LiveError(f"{name} file reference is required")
        value[f"_{name}_path"] = str(_file_ref(ref, root=root, field=name))
    output = value.get("output_directory")
    if not isinstance(output, str) or not output:
        raise E1LiveError("output_directory is required")
    value["_output_directory"] = str(_resolve(root, output))
    value["_repository_root"] = str(root)
    value["_config_path"] = str(source)
    baseline = NormalizedSelectedBaselineAdapter(value["_selected_baseline_manifest_path"])
    baseline_kind = baseline.value["baseline"].get("kind")
    if baseline_kind != kind:
        raise E1LiveError(
            f"live runner kind {kind!r} differs from selected baseline {baseline_kind!r}"
        )
    optional_supply = baseline.value["baseline"].get("optional_supply", {})
    if kind in {V46_SUPPLY, COMPOSITE}:
        if (
            optional_supply.get("enabled") is not True
            or optional_supply.get("auxiliary_routes_enabled") is not False
        ):
            raise E1LiveError(
                "minority Opus supply must be frozen and auxiliary Opus routes disabled"
            )
    elif optional_supply.get("enabled") is True:
        raise E1LiveError("all-Sol runner kind cannot enable optional Opus supply")
    execution = baseline.value["baseline"].get("live_execution")
    if not isinstance(execution, Mapping):
        raise E1LiveError("selected baseline must freeze baseline.live_execution")
    adapter_ref = execution.get("adapter_config")
    if not isinstance(adapter_ref, Mapping):
        raise E1LiveError("baseline.live_execution.adapter_config is required")
    value["_adapter_config_path"] = str(
        _file_ref(adapter_ref, root=baseline.base, field="baseline.live_execution.adapter_config")
    )
    value["_runtime_contract"] = _runtime_contract(baseline, kind=str(kind))
    if "model_concurrency" in value:
        raise E1LiveError(
            "live configs may not override selected-baseline concurrency"
        )
    _validate_selected_slate(
        Path(value["_selected_slate_manifest_path"]), kind=kind
    )
    return value


def _validate_selected_slate(path: Path, *, kind: str) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != 1 or value.get("selection_completed") is not True:
        raise E1LiveError("selected slate must be a completed schema-v1 selection")
    if value.get("source_labels_hidden_downstream") is not True:
        raise E1LiveError("selected slate must hide source labels downstream")
    if value.get("runner_kind") != kind:
        raise E1LiveError("selected slate runner_kind differs from the selected baseline")
    source_artifacts = value.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise E1LiveError("selected slate must freeze at least one source artifact")
    source_ids: set[str] = set()
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for index, artifact in enumerate(source_artifacts):
        if not isinstance(artifact, Mapping):
            raise E1LiveError("selected source artifact must be a mapping")
        source_id = artifact.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in source_ids:
            raise E1LiveError("selected source artifact IDs must be unique")
        source_ids.add(source_id)
        artifact_paths[source_id] = _file_ref(
            artifact, root=path.parent, field=f"source_artifacts[{index}]"
        )
        artifact_hashes[source_id] = str(artifact["sha256"])
    source_hashes = value.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or any(
        not isinstance(key, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        for key, digest in source_hashes.items()
    ):
        raise E1LiveError("selected slate source_hashes must contain SHA-256 digests")
    for artifact in source_artifacts:
        if source_hashes.get(artifact["source_id"]) != artifact["sha256"]:
            raise E1LiveError(
                f"selected slate source hash disagrees for {artifact['source_id']}"
            )
    attestation = value.get("supply_attestation")
    if kind in {V46_SUPPLY, COMPOSITE}:
        if not isinstance(attestation, Mapping):
            raise E1LiveError("minority Opus supply requires a frozen supply attestation")
        if any(
            attestation.get(key) != expected
            for key, expected in MINORITY_OPUS_SUPPLY_POLICY.items()
        ):
            raise E1LiveError("minority Opus supply attestation differs from 12/3 source-blind policy")
        source_id = attestation.get("supply_manifest_source_id")
        digest = attestation.get("supply_manifest_sha256")
        if (
            not isinstance(source_id, str)
            or source_id not in artifact_paths
            or digest != artifact_hashes[source_id]
        ):
            raise E1LiveError("minority Opus attestation is not tied to a frozen source artifact")
        manifest = read_json(artifact_paths[source_id])
        manifest_attestation = (
            manifest.get("supply_attestation")
            if isinstance(manifest, Mapping)
            else None
        )
        if not isinstance(manifest_attestation, Mapping):
            manifest_attestation = manifest
        if not isinstance(manifest_attestation, Mapping) or any(
            manifest_attestation.get(key) != expected
            for key, expected in MINORITY_OPUS_SUPPLY_POLICY.items()
        ):
            raise E1LiveError("frozen supply artifact does not substantiate its attestation")
    elif attestation not in (None, {}):
        raise E1LiveError("all-Sol runner kinds cannot carry minority Opus attestation")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise E1LiveError("selected slate candidates must be non-empty")
    seen: set[str] = set()
    providers: list[str] = []
    scale_count = 0
    sibling_map: dict[str, set[str]] = {}
    scale_by_slate: dict[int, set[str]] = {}
    candidate_ids_by_slate: dict[int, set[str]] = {}
    for row in candidates:
        if not isinstance(row, Mapping):
            raise E1LiveError("selected candidate must be a mapping")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise E1LiveError("selected candidate IDs must be unique")
        seen.add(candidate_id)
        if row.get("private_kernel_author_provider") not in {"codex", "claude"}:
            raise E1LiveError(f"{candidate_id} has an invalid private source provider")
        providers.append(str(row["private_kernel_author_provider"]))
        if type(row.get("optional_functional_scale")) is not bool:
            raise E1LiveError(f"{candidate_id} must freeze optional_functional_scale")
        scale_count += int(row["optional_functional_scale"])
        for field in (
            "slate_index",
            "search_mode",
            "kernel",
            "editorial_brief",
            "selector_global_priority",
        ):
            if field not in row:
                raise E1LiveError(f"{candidate_id} is missing {field}")
        if type(row["slate_index"]) is not int or row["slate_index"] < 1:
            raise E1LiveError(f"{candidate_id} has an invalid slate_index")
        if (
            type(row["selector_global_priority"]) is not int
            or row["selector_global_priority"] < 1
        ):
            raise E1LiveError(f"{candidate_id} has an invalid selector priority")
        if not isinstance(row["kernel"], Mapping) or not isinstance(
            row["editorial_brief"], Mapping
        ):
            raise E1LiveError(f"{candidate_id} kernel/editorial brief must be mappings")
        if row["kernel"].get("candidate_id") != candidate_id:
            raise E1LiveError(f"{candidate_id} kernel ID differs")
        private_keys = {
            "private_kernel_author_provider",
            "optional_functional_scale",
            "experimental_arm",
            "runner_kind",
        }
        for label, package in (
            ("kernel", row["kernel"]),
            ("editorial_brief", row["editorial_brief"]),
        ):
            stack = [package]
            while stack:
                current = stack.pop()
                if isinstance(current, Mapping):
                    leaked = private_keys & {str(key) for key in current}
                    if leaked:
                        raise E1LiveError(
                            f"{candidate_id} {label} leaks private routing keys: {sorted(leaked)}"
                        )
                    stack.extend(current.values())
                elif isinstance(current, list):
                    stack.extend(current)
            serialized = json.dumps(package, ensure_ascii=False).casefold()
            leaked_fragments = [
                fragment
                for fragment in (
                    "private source",
                    "kernel author provider",
                    "experimental arm",
                    "control arm",
                    "treatment arm",
                    "opus-authored",
                    "claude-authored",
                    "functional-scale assignment",
                    "functional scale experiment",
                )
                if fragment in serialized
            ]
            if leaked_fragments:
                raise E1LiveError(
                    f"{candidate_id} {label} leaks private routing labels: "
                    f"{leaked_fragments}"
                )
        novelty = row.get("novelty")
        if not isinstance(novelty, Mapping) or novelty.get("selector_archive_checked") is not True:
            raise E1LiveError(f"{candidate_id} lacks selector/archive evidence")
        if type(novelty.get("confirmed_same_procedure_duplicate")) is not bool:
            raise E1LiveError(
                f"{candidate_id} must explicitly freeze same-procedure duplicate status"
            )
        sibling_ids = novelty.get("within_window_sibling_ids", [])
        if not isinstance(sibling_ids, list) or any(
            not isinstance(item, str) or not item for item in sibling_ids
        ):
            raise E1LiveError(f"{candidate_id} has malformed within-window siblings")
        if candidate_id in sibling_ids or len(sibling_ids) != len(set(sibling_ids)):
            raise E1LiveError(f"{candidate_id} has invalid within-window sibling identity")
        sibling_map[candidate_id] = set(sibling_ids)
        representative = novelty.get("selected_as_sibling_representative")
        if type(representative) is not bool:
            raise E1LiveError(
                f"{candidate_id} must freeze selected_as_sibling_representative"
            )
        if representative != bool(sibling_ids):
            raise E1LiveError(
                f"{candidate_id} sibling-representative flag contradicts its sibling list"
            )
        if row["optional_functional_scale"]:
            slate = int(row["slate_index"])
            scale_by_slate.setdefault(slate, set()).add(candidate_id)
        candidate_ids_by_slate.setdefault(int(row["slate_index"]), set()).add(
            candidate_id
        )
    opus = sum(provider == "claude" for provider in providers)
    if kind in {V44_ORDINARY, V45_SCALE} and opus:
        raise E1LiveError(f"{kind} cannot contain Opus-authored kernels")
    if kind in {V44_ORDINARY, V46_SUPPLY} and scale_count:
        raise E1LiveError(f"{kind} cannot assign functional-scale planning")
    cohort_scale_provenance = _validate_cohort_scale_assignment(
        artifact_paths=artifact_paths,
        kind=kind,
        candidate_ids_by_slate=candidate_ids_by_slate,
        scale_ids_by_slate=scale_by_slate,
    )
    if kind == V45_SCALE and scale_count == 0 and not cohort_scale_provenance:
        raise E1LiveError(
            "V4.5 zero-invitation slate requires a frozen cohort-level "
            "functional-scale assignment"
        )
    if any(len(ids) > 1 for ids in scale_by_slate.values()):
        raise E1LiveError("functional scale is limited to one invitation per slate")
    mutual = sorted(
        (left, right)
        for left, siblings in sibling_map.items()
        for right in siblings
        if left < right and right in sibling_map and left in sibling_map[right]
    )
    if mutual:
        raise E1LiveError(
            "mutually identified same-procedure siblings were both selected: "
            + ", ".join(f"{left}/{right}" for left, right in mutual)
        )
    accounting = value.get("pre_authoring_accounting", {})
    if not isinstance(accounting, Mapping):
        raise E1LiveError("pre_authoring_accounting must be a mapping")
    if accounting.get("calls_by_role", {}).get("editorial_board", 0):
        raise E1LiveError("pre-authoring accounting cannot contain editorial boards")
    return value


def _validate_cohort_scale_assignment(
    *,
    artifact_paths: Mapping[str, Path],
    kind: str,
    candidate_ids_by_slate: Mapping[int, set[str]],
    scale_ids_by_slate: Mapping[int, set[str]],
) -> bool:
    """Validate optional cohort-level scale provenance.

    Functional-scale invitations are assigned over an entire frozen cohort,
    rather than rounded independently in every small slate.  Consequently a
    valid V4.5 slate may contain no invitation.  That exception is allowed
    only when the selected manifest hashes the exact cohort assignment and the
    assignment proves that this slate's empty assignment is intentional.
    """

    source_id = "cohort-functional-scale-assignment"
    path = artifact_paths.get(source_id)
    if path is None:
        return False
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise E1LiveError("cohort functional-scale assignment must be a mapping")
    if (
        value.get("schema_version") != 1
        or value.get("runner_kind") != kind
        or value.get("assignment_scope") != "entire_frozen_cohort"
        or value.get("maximum_per_slate") != 1
        or value.get("assignment_did_not_affect_selection_rank") is not True
        or value.get("planner_may_decline") is not True
    ):
        raise E1LiveError("cohort functional-scale assignment contract differs")
    selected_count = value.get("selected_candidate_count")
    assigned_count = value.get("assigned_count")
    target_fraction = value.get("target_selected_fraction")
    if (
        type(selected_count) is not int
        or selected_count < sum(len(ids) for ids in candidate_ids_by_slate.values())
        or type(assigned_count) is not int
        or assigned_count < 0
        or isinstance(target_fraction, bool)
        or not isinstance(target_fraction, (int, float))
        or not 0 <= float(target_fraction) <= 0.20
    ):
        raise E1LiveError("cohort functional-scale assignment has invalid totals")
    raw_by_slate = value.get("assigned_by_slate")
    if not isinstance(raw_by_slate, Mapping) or not raw_by_slate:
        raise E1LiveError("cohort functional-scale assignment lacks assigned_by_slate")
    assigned_by_slate: dict[int, set[str]] = {}
    all_assigned: set[str] = set()
    for raw_slate, raw_ids in raw_by_slate.items():
        try:
            slate = int(raw_slate)
        except (TypeError, ValueError) as exc:
            raise E1LiveError(
                "cohort functional-scale assignment has an invalid slate ID"
            ) from exc
        if slate < 1 or str(slate) != str(raw_slate):
            raise E1LiveError(
                "cohort functional-scale assignment has an invalid slate ID"
            )
        if not isinstance(raw_ids, list) or any(
            not isinstance(candidate_id, str) or not candidate_id
            for candidate_id in raw_ids
        ):
            raise E1LiveError(
                "cohort functional-scale assignment has malformed candidate IDs"
            )
        ids = set(raw_ids)
        if len(ids) != len(raw_ids) or len(ids) > 1 or ids & all_assigned:
            raise E1LiveError(
                "cohort functional-scale assignment violates identity or slate limits"
            )
        assigned_by_slate[slate] = ids
        all_assigned |= ids
    if assigned_count != len(all_assigned) or selected_count < assigned_count:
        raise E1LiveError("cohort functional-scale assignment totals disagree")
    for slate, candidate_ids in candidate_ids_by_slate.items():
        if slate not in assigned_by_slate:
            raise E1LiveError(
                f"cohort functional-scale assignment omits selected slate {slate}"
            )
        expected = scale_ids_by_slate.get(slate, set())
        actual = assigned_by_slate[slate]
        if actual != expected or not actual <= candidate_ids:
            raise E1LiveError(
                f"cohort functional-scale assignment differs for slate {slate}"
            )
    # A functional-scale runner with an empty local assignment is meaningful
    # only if the feature actually fired elsewhere in the frozen cohort.
    if kind == V45_SCALE and not any(scale_ids_by_slate.values()) and assigned_count < 1:
        raise E1LiveError(
            "V4.5 cohort functional-scale assignment contains no invitation"
        )
    return True


@dataclass
class LiveExecutionResult:
    records: dict[str, dict[str, Any]]
    technical: dict[str, dict[str, Any]]
    blind: dict[str, dict[str, Any]]
    call_ledger: dict[str, Any]
    runtime: Any = None
    runtime_contract_verification: dict[str, Any] | None = None


class LiveBackend(Protocol):
    def execute(
        self,
        config: Mapping[str, Any],
        selected: Mapping[str, Any],
        *,
        authorize_model_calls: bool,
    ) -> LiveExecutionResult: ...

    def shadow(
        self,
        config: Mapping[str, Any],
        selected: Mapping[str, Any],
        clear_ids: Sequence[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


def _runtime_selected_rows(selected: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "private_kernel_author_provider": row["private_kernel_author_provider"],
            "slate_index": int(row["slate_index"]),
            "search_mode": row["search_mode"],
            "kernel": deepcopy(row["kernel"]),
            "editorial_brief": deepcopy(row["editorial_brief"]),
            "selector_global_priority": int(row["selector_global_priority"]),
        }
        for row in selected["candidates"]
    ]


def _write_immutable_json(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    """Write once, or prove the existing bytes encode the same value."""

    if path.exists():
        if read_json(path) != value:
            raise E1LiveError(f"{label} changed during immutable resume: {path}")
        return
    atomic_write_json(path, value)


def _editorial_calls(ledger: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(key)
        for key, item in ledger.items()
        if isinstance(item, Mapping)
        and (
            item.get("role") == "editorial_board"
            or item.get("stage") == "v4_editorial_board"
        )
    )


def _assert_pre_board_records(records: Mapping[str, Mapping[str, Any]]) -> None:
    forbidden = {
        "editorial_judgments",
        "intrinsic_quality",
        "portfolio_placement",
        "editorial_reason",
    }
    for task_id, record in records.items():
        leaked = forbidden & set(record)
        if leaked or record.get("status") == "editorial_complete":
            raise E1LiveError(
                f"{task_id} crossed the pre-board boundary: {sorted(leaked)}"
            )


def _callable_chain(value: Any, *, _seen: set[int] | None = None) -> set[str]:
    """Describe a parser and any callable wrappers captured by its closure."""

    seen = _seen if _seen is not None else set()
    if not callable(value) or id(value) in seen:
        return set()
    seen.add(id(value))
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
    result = {f"{module}.{name}"}
    for cell in getattr(value, "__closure__", None) or ():
        try:
            captured = cell.cell_contents
        except ValueError:
            continue
        result.update(_callable_chain(captured, _seen=seen))
    return result


def _assert_scale_planning_contracts(
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reapply the frozen functional-scale response invariants after resume."""

    from .v44_abstraction_led import _normalize_palette_claims
    from .v44_functional_scale import _private_leaks, parse_scale_plan

    for task_id, record in records.items():
        if record.get("private_route") != "functional_scale":
            continue
        result = record.get("planning_result")
        if not isinstance(result, Mapping):
            # Operational failures contain no response to validate.
            if record.get("status") == "planning_operational_failure":
                continue
            raise E1LiveError(f"{task_id} lacks its functional-scale planning result")
        try:
            parsed = parse_scale_plan(result)
        except Exception as exc:
            raise E1LiveError(
                f"{task_id} violates the exact functional-scale response schema: {exc}"
            ) from exc
        if parsed.get("candidate_id") != task_id:
            raise E1LiveError(f"{task_id} functional-scale wrapper changed candidate ID")
        plan = parsed.get("concrete_plan")
        decision = parsed.get("scale_decision")
        if plan is None:
            if decision != "abort":
                raise E1LiveError(f"{task_id} missing plan is not an explicit scale abort")
            continue
        if plan.get("candidate_id") != task_id:
            raise E1LiveError(f"{task_id} functional-scale nested plan changed candidate ID")
        if _normalize_palette_claims(deepcopy(plan)) != plan:
            raise E1LiveError(f"{task_id} functional-scale plan was not palette-normalized")
        leaks = _private_leaks(plan)
        if leaks:
            raise E1LiveError(
                f"{task_id} functional-scale plan leaked private labels: {leaks}"
            )
        if plan.get("status") == "abort" or decision == "abort":
            raise E1LiveError(f"{task_id} functional-scale abort fields disagree")


def _assert_runtime_contract(config: Mapping[str, Any], runtime: Any) -> None:
    """Verify exact prompt, schema, route, anchor, and concurrency identity."""

    baseline = NormalizedSelectedBaselineAdapter(
        config["_selected_baseline_manifest_path"]
    )
    contract = config["_runtime_contract"]
    for name, frozen in contract["prompts"].items():
        active = runtime.prompts.entries.get(name)
        if not isinstance(active, Mapping):
            raise E1LiveError(f"runtime did not install frozen prompt {name}")
        source_path = _resolve(baseline.base, str(frozen["path"]))
        observed = {
            "filename": active.get("filename"),
            "version": active.get("version"),
            "stage": active.get("stage"),
            "sha256": active.get("hash"),
        }
        expected = {
            "filename": source_path.name,
            "version": frozen.get("version"),
            "stage": frozen.get("stage"),
            "sha256": frozen.get("sha256"),
        }
        if observed != expected:
            raise E1LiveError(f"runtime prompt binding differs for {name}")
    for stage, frozen in contract["schemas"].items():
        active_path = runtime.run_dir / "schemas" / f"{stage}.schema.json"
        if not active_path.is_file() or _sha(active_path) != frozen["sha256"]:
            raise E1LiveError(f"runtime structured-output schema differs for {stage}")
    parser_chain = _callable_chain(runtime.invoker.output_parser)
    expected_parser = (
        f"{contract['parser']['module']}.{contract['parser']['callable']}"
    )
    if expected_parser not in parser_chain:
        raise E1LiveError("runtime structured-output parser binding changed")
    for role in (
        "concrete_planner",
        "task_realizer",
        "technical_verifier",
        "blind_solver",
        "ambiguity_adjudicator",
        "targeted_repair",
        "editorial_board",
    ):
        frozen = baseline.value["model_routes"].get(role)
        active = runtime.config["model"]["roles"].get(role)
        if not isinstance(frozen, Mapping) or not isinstance(active, Mapping):
            raise E1LiveError(f"selected baseline does not freeze model route {role}")
        if (
            frozen.get("provider"),
            frozen.get("model"),
            frozen.get("reasoning_effort"),
        ) != (
            "codex",
            active.get("model"),
            active.get("reasoning_effort"),
        ):
            raise E1LiveError(f"runtime model route differs for {role}")
    if int(runtime.config["model"]["concurrency"]) != int(contract["model_concurrency"]):
        raise E1LiveError("runtime concurrency changed")
    if int(runtime.config["integrated_qualification"]["escalation_calls_per_slate"]) != 5:
        raise E1LiveError("runtime escalation ceiling changed")
    anchor_path = Path(runtime.config["paths"]["fixed_anchor_manifest"])
    anchor_ref = baseline.value["frozen_inputs"]["fixed_anchor_manifest"]
    if not anchor_path.is_file() or _sha(anchor_path) != anchor_ref["sha256"]:
        raise E1LiveError("runtime fixed-anchor manifest differs")
    anchors = read_json(anchor_path).get("anchors")
    if (
        not isinstance(anchors, list)
        or len(anchors) != 6
        or len({item.get("task_id") for item in anchors if isinstance(item, Mapping)}) != 6
    ):
        raise E1LiveError("runtime fixed-anchor manifest does not contain six unique anchors")


def _observed_runtime_contract(
    config: Mapping[str, Any], runtime: Any
) -> dict[str, Any]:
    from .v41_config import snapshot_v41_config

    contract = config["_runtime_contract"]
    prompts = {
        name: {
            "filename": runtime.prompts.entries[name]["filename"],
            "version": runtime.prompts.entries[name]["version"],
            "stage": runtime.prompts.entries[name]["stage"],
            "sha256": runtime.prompts.entries[name]["hash"],
        }
        for name in sorted(contract["prompts"])
    }
    schemas = {
        stage: _sha(runtime.run_dir / "schemas" / f"{stage}.schema.json")
        for stage in sorted(contract["schemas"])
    }
    return {
        "verified": True,
        "frozen_contract_sha256": content_hash(contract),
        "resolved_execution_config_sha256": content_hash(
            snapshot_v41_config(runtime.config)
        ),
        "resolved_execution_config_snapshot": {
            "path": str(runtime.run_dir / "config.snapshot.json"),
            "sha256": _sha(runtime.run_dir / "config.snapshot.json"),
        },
        "model_concurrency": int(runtime.config["model"]["concurrency"]),
        "escalation_calls_per_slate": int(
            runtime.config["integrated_qualification"]["escalation_calls_per_slate"]
        ),
        "prompts": prompts,
        "schemas": schemas,
        "parser_chain": sorted(_callable_chain(runtime.invoker.output_parser)),
        "parser_source_sha256": contract["parser"]["sha256"],
        "implementation_manifest": {
            "path": str(runtime.run_dir / "implementation_manifest.json"),
            "sha256": _sha(runtime.run_dir / "implementation_manifest.json"),
        },
    }


def _validate_private_assignment(
    selected: Mapping[str, Any], assignment: Mapping[str, Any]
) -> None:
    rows = assignment.get("assignments")
    if not isinstance(rows, list) or len(rows) != len(selected["candidates"]):
        raise E1LiveError("private assignment does not cover the selected slate")
    by_id = {str(item["candidate_id"]): item for item in selected["candidates"]}
    seen_source: set[str] = set()
    seen_opaque: set[str] = set()
    scale_by_slate: dict[int, int] = {}
    for row in rows:
        source = str(row.get("source_candidate_id"))
        opaque = str(row.get("opaque_task_id"))
        if source not in by_id or source in seen_source or not opaque or opaque in seen_opaque:
            raise E1LiveError("private assignment IDs are incomplete or duplicated")
        seen_source.add(source)
        seen_opaque.add(opaque)
        expected = "functional_scale" if by_id[source]["optional_functional_scale"] else "ordinary_current"
        if row.get("private_route") != expected:
            raise E1LiveError(f"private assignment changed the frozen route for {source}")
        if row.get("private_source") != by_id[source]["private_kernel_author_provider"]:
            raise E1LiveError(f"private assignment changed the source provider for {source}")
        if expected == "functional_scale":
            slate = int(row["slate_index"])
            scale_by_slate[slate] = scale_by_slate.get(slate, 0) + 1
    if any(count > 1 for count in scale_by_slate.values()):
        raise E1LiveError("functional scale exceeds the exact one-invitation-per-slate ceiling")


def _record_technical(record: Mapping[str, Any]) -> dict[str, Any]:
    results = record.get("technical_results")
    if not isinstance(results, list) or not results:
        return {
            "operational_failure": (
                True if record.get("status") == "technical_operational_failure" else None
            ),
            "passed": None,
            "hidden_output_match": None,
            "unresolved_pair_economy_flags": [],
            "response": {},
        }
    latest = deepcopy(results[-1])
    response = latest.get("response")
    if not isinstance(response, Mapping):
        raise E1LiveError(f"{record.get('task_id')} has a malformed technical result")
    expected_pass = (
        response.get("status") == "pass"
        and response.get("mechanically_correct") is True
        and response.get("test_supported") is True
        and response.get("one_answer_supported") is True
        and response.get("plan_fidelity") != "fail"
        and response.get("evidence_sufficiency") == "sufficient"
        and response.get("pair_economy") != "fail"
        and response.get("composition_preserved") is True
        and not response.get("arbitrary_exceptions")
        and not response.get("unsupported_test_semantics")
        and not response.get("unapproved_realizer_decisions")
        and latest.get("hidden_output_match") is True
        and not latest.get("unresolved_pair_economy_flags")
    )
    if type(latest.get("passed")) is not bool or latest.get("passed") != expected_pass:
        raise E1LiveError(
            f"{record.get('task_id')} technical wrapper contradicts its response"
        )
    return {
        **latest,
        "operational_failure": False,
        "history": deepcopy(results),
    }


def _record_blind(record: Mapping[str, Any]) -> dict[str, Any]:
    results = record.get("blind_results")
    if not isinstance(results, list) or not results:
        return {
            "operational_failure": (
                True if record.get("status") == "blind_operational_failure" else None
            ),
            "serious_different_output_alternative": None,
            "effective_serious_different_output_alternative": None,
            "effective_resolution_kind": "not_completed",
            "results": [],
        }
    primary = results[0].get("serious_different_output_alternative")
    latest = results[-1].get("serious_different_output_alternative")
    if type(primary) is not bool or type(latest) is not bool:
        raise E1LiveError(f"{record.get('task_id')} blind result lacks an explicit serious flag")
    adjudication = record.get("ambiguity_adjudication")
    status = record.get("status")
    repair_count = int(record.get("repair_count", 0))
    if status == "ambiguous":
        effective: bool | None = True
        resolution = "unresolved_ambiguity"
    elif status != "blind_pass":
        effective = None
        resolution = "nonterminal_blind_state"
    elif repair_count and latest is False:
        effective = False
        resolution = "targeted_repair_revalidated"
    elif isinstance(adjudication, Mapping):
        decision = adjudication.get("decision")
        different = adjudication.get("comparably_simple_different_output_rule")
        if decision == "clear" and different is False:
            effective = False
            resolution = "adjudicated_clear"
        elif decision in {"repairable", "serious_ambiguity"} or different is True:
            effective = True
            resolution = "adjudicated_unresolved"
        else:
            effective = None
            resolution = "malformed_or_inconclusive_adjudication"
    elif latest is False:
        effective = False
        resolution = "blind_clear"
    else:
        effective = True
        resolution = "blind_serious_alternative"
    return {
        "operational_failure": False,
        "response": deepcopy(results[-1].get("response", {})),
        "results": deepcopy(results),
        "primary_serious_different_output_alternative": primary,
        "serious_different_output_alternative": effective,
        "effective_serious_different_output_alternative": effective,
        "effective_resolution_kind": resolution,
        "ambiguity_adjudication": deepcopy(adjudication),
        "record_status": status,
        "repair_count": repair_count,
    }


class V41IntegratedPostSelectionBackend:
    """Exact adapter over the qualified V4.1 post-selection execution stack."""

    def _resolved_source_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        from .v41_config import snapshot_v41_config
        from .v44_scale_fresh_integrated import load_fresh_scale_integrated_config

        resolved = load_fresh_scale_integrated_config(config["_adapter_config_path"])
        contract = config["_runtime_contract"]
        observed = content_hash(snapshot_v41_config(resolved))
        if observed != contract["resolved_adapter_config_sha256"]:
            raise E1LiveError(
                "resolved V4.1 adapter configuration differs from the frozen contract"
            )
        if int(resolved["model"]["concurrency"]) != int(contract["model_concurrency"]):
            raise E1LiveError("V4.1 concurrency differs from the frozen contract")
        ceiling = int(
            resolved["integrated_qualification"]["escalation_calls_per_slate"]
        )
        if ceiling != int(contract["escalation_calls_per_slate"]):
            raise E1LiveError("V4.1 escalation ceiling differs from the frozen contract")
        return resolved

    def _runtime(
        self,
        config: Mapping[str, Any],
        *,
        purpose: str,
        source_manifest: Mapping[str, Any],
    ) -> Any:
        from .v41_experiments import V41Runtime
        from .v4_integrated_validation import _install_extensions

        resolved = deepcopy(self._resolved_source_config(config))
        resolved["paths"] = deepcopy(resolved["paths"])
        resolved["paths"]["runs"] = str(
            Path(config["_output_directory"]) / f"{purpose}_runtime"
        )
        resolved["run"] = {
            **deepcopy(resolved["run"]),
            "id": purpose,
            "mock": bool(config["run"]["mock"]),
            "automatically_commit_generated_tasks": False,
        }
        runtime = V41Runtime(resolved, source_manifest)
        _install_extensions(runtime)
        _assert_runtime_contract(config, runtime)
        return runtime

    @staticmethod
    def _selected_rows(selected: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "source_candidate_id": str(row["candidate_id"]),
                "slate_index": int(row["slate_index"]),
                "source_development_priority": int(row["selector_global_priority"]),
                "kernel": deepcopy(row["kernel"]),
                "editorial_brief": deepcopy(row["editorial_brief"]),
                "private_source": str(row["private_kernel_author_provider"]),
            }
            for row in selected["candidates"]
        ]

    @staticmethod
    def _assignment(
        selected: Mapping[str, Any], aliases: Mapping[str, str], seed: int
    ) -> dict[str, Any]:
        rows = []
        for row in selected["candidates"]:
            source_id = str(row["candidate_id"])
            rows.append(
                {
                    "source_candidate_id": source_id,
                    "opaque_task_id": aliases[source_id],
                    "slate_index": int(row["slate_index"]),
                    "development_priority": int(row["selector_global_priority"]),
                    "search_mode": str(row["search_mode"]),
                    "private_source": str(row["private_kernel_author_provider"]),
                    "private_route": (
                        "functional_scale"
                        if row["optional_functional_scale"]
                        else "ordinary_current"
                    ),
                }
            )
        return {
            "schema_version": 1,
            "assignment_seed": seed,
            "selection_completed_before_assignment": True,
            "source_and_route_hidden_downstream": True,
            "one_deliberate_intervention_per_candidate": True,
            "assignments": rows,
        }

    def execute(
        self,
        config: Mapping[str, Any],
        selected: Mapping[str, Any],
        *,
        authorize_model_calls: bool,
    ) -> LiveExecutionResult:
        if not authorize_model_calls and not config["run"]["mock"]:
            raise E1LiveError("real live authoring requires --authorize-model-calls")
        from .v4_integrated_validation import (
            _aliases,
            _budgets,
            _initialize_records,
            _plan_records,
            _realize_records,
        )
        from .v41_supported_departure import _run_blind, _run_technical

        baseline = NormalizedSelectedBaselineAdapter(
            config["_selected_baseline_manifest_path"]
        )
        runtime = self._runtime(
            config,
            purpose="authoring",
            source_manifest={
                "schema_version": 1,
                "selected_baseline_manifest_sha256": baseline.sha256,
                "selected_slate_manifest_sha256": _sha(
                    config["_selected_slate_manifest_path"]
                ),
                "post_selection_only": True,
            },
        )
        if _editorial_calls(runtime.state.get("call_ledger", {})):
            raise E1LiveError("authoring runtime already contains an editorial call")
        rows = self._selected_rows(selected)
        aliases = _aliases(rows, int(runtime.config["seed"]))
        assignment = self._assignment(selected, aliases, int(runtime.config["seed"]))
        _validate_private_assignment(selected, assignment)
        _write_immutable_json(
            runtime.run_dir / "e1_route_assignment.private.json",
            assignment,
            label="E1 private route assignment",
        )
        _initialize_records(runtime, rows, assignment)
        _plan_records(runtime)
        _assert_scale_planning_contracts(runtime.state["semantic_records"])
        _realize_records(runtime)
        budgets = _budgets(runtime)
        _run_technical(runtime, budgets)
        _run_blind(runtime, budgets)
        if _editorial_calls(runtime.state.get("call_ledger", {})):
            raise E1LiveError("routine editorial review crossed the post-blind boundary")
        records = deepcopy(runtime.state["semantic_records"])
        _assert_pre_board_records(records)
        return LiveExecutionResult(
            records=records,
            technical={
                task_id: _record_technical(record)
                for task_id, record in records.items()
            },
            blind={
                task_id: _record_blind(record) for task_id, record in records.items()
            },
            call_ledger=deepcopy(runtime.state.get("call_ledger", {})),
            runtime=runtime,
            runtime_contract_verification=_observed_runtime_contract(config, runtime),
        )

    def shadow(
        self,
        config: Mapping[str, Any],
        selected: Mapping[str, Any],
        clear_ids: Sequence[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from .v41_test_challenge import _board

        if len(clear_ids) != len(set(clear_ids)):
            raise E1LiveError("shadow clear queue contains duplicate IDs")
        raw = read_json(
            Path(config["_output_directory"]) / "raw_post_blind.private.json"
        )
        if set(clear_ids) - set(raw["records"]):
            raise E1LiveError("shadow clear queue is not a subset of the frozen checkpoint")
        baseline = NormalizedSelectedBaselineAdapter(
            config["_selected_baseline_manifest_path"]
        )
        runtime = self._runtime(
            config,
            purpose="shadow",
            source_manifest={
                "schema_version": 1,
                "selected_baseline_manifest_sha256": baseline.sha256,
                "boardless_freeze_sha256": _sha(
                    Path(config["_output_directory"])
                    / "qualification"
                    / "boardless_dispositions.frozen.json"
                ),
                "exact_clear_ids": sorted(clear_ids),
            },
        )
        if runtime.state.get("semantic_records"):
            if set(runtime.state["semantic_records"]) != set(clear_ids):
                raise E1LiveError("resumed shadow runtime targets a different clear queue")
        else:
            records = {
                task_id: deepcopy(raw["records"][task_id]) for task_id in clear_ids
            }
            for task_id, record in records.items():
                record["task_id"] = task_id
                record["status"] = "blind_pass"
                for key in (
                    "editorial_judgments",
                    "intrinsic_quality",
                    "portfolio_placement",
                    "editorial_reason",
                ):
                    record.pop(key, None)
            runtime.state["semantic_records"] = records
            runtime.save()
        if clear_ids:
            _board(runtime)
        board_ledger = deepcopy(runtime.state.get("call_ledger", {}))
        if any(item.get("role") != "editorial_board" for item in board_ledger.values()):
            raise E1LiveError("shadow dispatcher made a non-editorial call")
        task_results = {}
        for task_id in clear_ids:
            record = runtime.state["semantic_records"][task_id]
            judgments = deepcopy(record.get("editorial_judgments", []))
            task_results[task_id] = {
                "operational_failure": len(judgments) != 2,
                "judgments": judgments if len(judgments) == 2 else [],
                "intrinsic_quality": record.get("intrinsic_quality"),
                "portfolio_placement": record.get("portfolio_placement"),
                "intrinsic_reason": record.get("editorial_reason"),
                "portfolio_reason": record.get("editorial_reason"),
            }
        return {"task_results": task_results}, _normalize_accounting(
            v4_call_accounting(board_ledger, run_dir=runtime.run_dir)
        )


def _normalize_accounting(value: Mapping[str, Any] | None) -> dict[str, Any]:
    source = value or {}
    return {
        "substantive_logical_calls": int(source.get("substantive_logical_calls", 0)),
        "transport_retry_calls": int(source.get("transport_retry_calls", 0)),
        "cache_hits": int(source.get("cache_hits", 0)),
        "active_model_seconds": float(source.get("active_model_seconds", 0)),
        "queue_wait_seconds": float(
            source.get("queue_wait_seconds", source.get("model_slot_queue_wait_seconds", 0))
        ),
        "input_tokens": int(source.get("input_tokens", 0)),
        "output_tokens": int(source.get("output_tokens", 0)),
        "calls_by_role": {
            str(key): int(item) for key, item in source.get("calls_by_role", {}).items()
        },
        "completed_substantive_calls": int(source.get("completed_substantive_calls", 0)),
        "failed_substantive_calls": int(source.get("failed_substantive_calls", 0)),
        "escalation_calls": int(source.get("escalation_calls", 0)),
        "recorded_model_attempts": int(source.get("recorded_model_attempts", 0)),
        "measured_peak_concurrent_calls": int(
            source.get("measured_peak_concurrent_calls", 0)
        ),
        "measured_effective_concurrency": float(
            source.get("measured_effective_concurrency", 0)
        ),
        "timed_call_count": int(source.get("timed_call_count", 0)),
    }


def _sum_accounting(*rows: Mapping[str, Any]) -> dict[str, Any]:
    result = _normalize_accounting({})
    for row in rows:
        normalized = _normalize_accounting(row)
        for field in (
            "substantive_logical_calls",
            "transport_retry_calls",
            "cache_hits",
            "active_model_seconds",
            "queue_wait_seconds",
            "input_tokens",
            "output_tokens",
            "completed_substantive_calls",
            "failed_substantive_calls",
            "escalation_calls",
            "recorded_model_attempts",
            "timed_call_count",
        ):
            result[field] += normalized[field]
        for role, count in normalized["calls_by_role"].items():
            result["calls_by_role"][role] = result["calls_by_role"].get(role, 0) + count
    result["calls_by_role"] = dict(sorted(result["calls_by_role"].items()))
    result["measured_peak_concurrent_calls"] = max(
        (_normalize_accounting(row)["measured_peak_concurrent_calls"] for row in rows),
        default=0,
    )
    # Effective concurrency cannot be added.  This weighted summary is
    # explicitly descriptive; per-stage/per-run values remain in the split.
    timed = sum(_normalize_accounting(row)["timed_call_count"] for row in rows)
    result["measured_effective_concurrency"] = (
        round(
            sum(
                _normalize_accounting(row)["measured_effective_concurrency"]
                * _normalize_accounting(row)["timed_call_count"]
                for row in rows
            )
            / timed,
            6,
        )
        if timed
        else 0.0
    )
    return result


def _normalized_candidate(
    selected: Mapping[str, Any],
    record: Mapping[str, Any],
    technical: Mapping[str, Any] | None,
    blind: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidate_id = str(record.get("task_id") or record.get("candidate_id"))
    source_candidate_id = str(record.get("source_candidate_id") or candidate_id)
    selection = next(
        item
        for item in selected["candidates"]
        if item["candidate_id"] == source_candidate_id
    )
    realized = record.get("complete_task") is not None
    diagnostics = record.get("deterministic_diagnostics") or {}
    fidelity = record.get("plan_fidelity") or {}
    tech = deepcopy(dict(technical or {}))
    blind_value = deepcopy(dict(blind or {}))
    effective = effective_blind_serious(blind_value)
    if blind_value:
        primary = blind_value.get("primary_serious_different_output_alternative")
        if type(primary) is not bool:
            primary = blind_value.get("serious_different_output_alternative")
        blind_value["primary_serious_different_output_alternative"] = primary
        blind_value["serious_different_output_alternative"] = effective
        blind_value["effective_serious_different_output_alternative"] = effective
        blind_value["effective_adjudicated_result_preserved"] = (
            blind_value.get("ambiguity_adjudication") is not None
            or int(blind_value.get("repair_count", 0)) > 0
        )
    response = tech.get("response") if isinstance(tech.get("response"), Mapping) else {}
    novelty = selection["novelty"]
    evidence = {
        "deterministic": {
            "valid": realized and not diagnostics.get("errors"),
            "hard_valid": realized and diagnostics.get("hard_valid") is True,
            "diagnostics": deepcopy(diagnostics),
        },
        "realization": {
            "operational_failure": (
                record.get("status") == "realization_operational_failure"
                if not realized
                else False
            ),
            "status": "realized" if realized else record.get("status"),
        },
        "plan_fidelity": {"passed": fidelity.get("passed") is True, "result": deepcopy(fidelity)},
        "technical": {
            "operational_failure": tech.get("operational_failure"),
            "passed": tech.get("passed"),
            "hidden_output_match": tech.get("hidden_output_match"),
            "unresolved_pair_economy_flags": deepcopy(
                tech.get("unresolved_pair_economy_flags", [])
            ),
            "response": deepcopy(dict(response)),
        },
        "blind": blind_value,
        "novelty": {
            "selector_archive_checked": novelty.get("selector_archive_checked") is True,
            "confirmed_same_procedure_duplicate": novelty.get(
                "confirmed_same_procedure_duplicate"
            ) is True,
            "theme_is_common": novelty.get("theme_is_common"),
            "within_window_sibling_ids": deepcopy(
                novelty.get("within_window_sibling_ids", [])
            ),
            "selected_as_sibling_representative": novelty.get(
                "selected_as_sibling_representative"
            ),
        },
    }
    return {
        "candidate_id": candidate_id,
        "source_id": "fresh-live-authoring",
        "lineage_id": selection.get("lineage_id", source_candidate_id),
        "search_mode": selection["search_mode"],
        "evidence": evidence,
    }


def _protected_refs(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    try:
        baseline = NormalizedSelectedBaselineAdapter(
            config["_selected_baseline_manifest_path"]
        )
    except E1QualificationError as exc:
        raise E1LiveError(
            f"protected selected-baseline state changed: {exc}"
        ) from exc
    refs = {
        "selected_baseline_manifest": {
            "path": str(baseline.path),
            "sha256": baseline.sha256,
        },
        "selected_slate_manifest": {
            "path": config["_selected_slate_manifest_path"],
            "sha256": _sha(config["_selected_slate_manifest_path"]),
        },
        "adapter_config": {
            "path": config["_adapter_config_path"],
            "sha256": _sha(config["_adapter_config_path"]),
        },
        "e1_implementation:live_runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha(Path(__file__).resolve()),
        },
        "e1_implementation:boardless_qualification": {
            "path": str(
                Path(__file__).resolve().with_name("e1_boardless_qualification.py")
            ),
            "sha256": _sha(
                Path(__file__).resolve().with_name("e1_boardless_qualification.py")
            ),
        },
    }
    for name, ref in baseline.value["frozen_inputs"].items():
        refs[f"frozen_input:{name}"] = {
            "path": str(_resolve(baseline.base, ref["path"])),
            "sha256": ref["sha256"],
        }
    contract = config["_runtime_contract"]
    for name, ref in contract["prompts"].items():
        refs[f"prompt:{name}"] = {
            "path": str(_resolve(baseline.base, ref["path"])),
            "sha256": ref["sha256"],
        }
    for stage, ref in contract["schemas"].items():
        refs[f"schema:{stage}"] = {
            "path": str(_resolve(baseline.base, ref["path"])),
            "sha256": ref["sha256"],
        }
    for index, ref in enumerate(contract["implementation_files"]):
        refs[f"implementation:{index}"] = {
            "path": str(_resolve(baseline.base, ref["path"])),
            "sha256": ref["sha256"],
        }
    return refs


def _assert_protected(refs: Mapping[str, Mapping[str, str]]) -> None:
    changed = [name for name, ref in refs.items() if _sha(ref["path"]) != ref["sha256"]]
    if changed:
        raise E1LiveError("protected selected-baseline state changed: " + ", ".join(changed))


def _e1_config(config: Mapping[str, Any], evidence_path: Path) -> Path:
    baseline = NormalizedSelectedBaselineAdapter(config["_selected_baseline_manifest_path"])
    output = Path(config["_output_directory"])
    editorial_hash = baseline.value["prompts"]["editorial_board"]["sha256"]
    value = {
        "pipeline": "arcgen-e1-boardless-qualification",
        "version": "1.0",
        "template": False,
        "enabled": True,
        "selected_baseline_manifest": str(baseline.path),
        "selected_baseline_manifest_sha256": baseline.sha256,
        "evidence_manifest": str(evidence_path),
        "evidence_manifest_sha256": _sha(evidence_path),
        "output_directory": str(output / "qualification"),
        "run": {
            "id": config["run"]["id"],
            "automatically_commit_generated_tasks": False,
            "production_state_mutation_allowed": False,
            "shadow_editorial_gating": False,
        },
        "shadow_policy": {
            "gating": False,
            "reviewer_count": 2,
            "defer_until_all_dispositions_frozen": True,
            "generated_group_minimum": 4,
            "generated_group_maximum": 6,
            "editorial_prompt_sha256": editorial_hash,
        },
    }
    path = output / "e1_qualification.config.json"
    if path.exists() and read_json(path) != value:
        raise E1LiveError("derived E1 config changed during resume")
    if not path.exists():
        atomic_write_json(path, value)
    return path


def _validate_raw_checkpoint(
    raw: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> None:
    if raw.get("schema_version") != 2:
        raise E1LiveError("raw post-blind checkpoint must use schema version 2")
    expected = {
        "runner_kind": config["runner_kind"],
        "selected_baseline_manifest_sha256": _sha(
            config["_selected_baseline_manifest_path"]
        ),
        "selected_slate_manifest_sha256": _sha(
            config["_selected_slate_manifest_path"]
        ),
        "live_config_sha256": _sha(config["_config_path"]),
        "runtime_contract_sha256": content_hash(config["_runtime_contract"]),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise E1LiveError(f"raw post-blind checkpoint changed {key}")
    for key in ("records", "technical", "blind", "call_ledger"):
        if not isinstance(raw.get(key), Mapping):
            raise E1LiveError(f"raw post-blind checkpoint lacks {key}")
    verification = raw.get("runtime_contract_verification")
    contract = config["_runtime_contract"]
    if not isinstance(verification, Mapping) or verification.get("verified") is not True:
        raise E1LiveError("raw checkpoint lacks exact runtime-contract verification")
    if verification.get("frozen_contract_sha256") != content_hash(contract):
        raise E1LiveError("raw checkpoint runtime contract hash changed")
    if verification.get("model_concurrency") != contract["model_concurrency"]:
        raise E1LiveError("raw checkpoint runtime concurrency changed")
    if (
        verification.get("escalation_calls_per_slate")
        != contract["escalation_calls_per_slate"]
    ):
        raise E1LiveError("raw checkpoint escalation policy changed")
    observed_prompts = verification.get("prompts")
    if not isinstance(observed_prompts, Mapping) or set(observed_prompts) != set(
        contract["prompts"]
    ):
        raise E1LiveError("raw checkpoint prompt bindings are incomplete")
    baseline_base = NormalizedSelectedBaselineAdapter(
        config["_selected_baseline_manifest_path"]
    ).base
    for name, frozen in contract["prompts"].items():
        observed = observed_prompts[name]
        if not isinstance(observed, Mapping) or {
            "filename": observed.get("filename"),
            "version": observed.get("version"),
            "stage": observed.get("stage"),
            "sha256": observed.get("sha256"),
        } != {
            "filename": _resolve(
                baseline_base,
                frozen["path"],
            ).name,
            "version": frozen["version"],
            "stage": frozen["stage"],
            "sha256": frozen["sha256"],
        }:
            raise E1LiveError(f"raw checkpoint prompt binding changed for {name}")
    observed_schemas = verification.get("schemas")
    expected_schemas = {
        stage: frozen["sha256"] for stage, frozen in contract["schemas"].items()
    }
    if observed_schemas != expected_schemas:
        raise E1LiveError("raw checkpoint schema bindings changed")
    expected_parser = f"{contract['parser']['module']}.{contract['parser']['callable']}"
    if (
        verification.get("parser_source_sha256") != contract["parser"]["sha256"]
        or expected_parser not in (verification.get("parser_chain") or [])
    ):
        raise E1LiveError("raw checkpoint parser binding changed")
    implementation = verification.get("implementation_manifest")
    if not isinstance(implementation, Mapping):
        raise E1LiveError("raw checkpoint lacks its runtime implementation manifest")
    _file_ref(
        implementation,
        root=Path(config["_repository_root"]),
        field="runtime_contract_verification.implementation_manifest",
    )
    resolved_hash = verification.get("resolved_execution_config_sha256")
    if not isinstance(resolved_hash, str) or len(resolved_hash) != 64:
        raise E1LiveError("raw checkpoint lacks its resolved execution-config hash")
    resolved_ref = verification.get("resolved_execution_config_snapshot")
    if not isinstance(resolved_ref, Mapping):
        raise E1LiveError("raw checkpoint lacks its resolved execution-config snapshot")
    resolved_path = _file_ref(
        resolved_ref,
        root=Path(config["_repository_root"]),
        field="runtime_contract_verification.resolved_execution_config_snapshot",
    )
    if content_hash(read_json(resolved_path)) != resolved_hash:
        raise E1LiveError("raw checkpoint resolved execution-config hash is inconsistent")
    if raw.get("post_blind_stop") is not True:
        raise E1LiveError("raw checkpoint does not freeze the post-blind stop")
    record_ids = set(raw["records"])
    if set(raw["technical"]) != record_ids or set(raw["blind"]) != record_ids:
        raise E1LiveError("raw technical/blind maps do not exactly match record IDs")
    source_ids = {
        str(record.get("source_candidate_id") or record.get("candidate_id"))
        for record in raw["records"].values()
    }
    selected_ids = {str(item["candidate_id"]) for item in selected["candidates"]}
    if source_ids != selected_ids:
        raise E1LiveError("raw checkpoint candidate set differs from the selected slate")
    if raw.get("editorial_calls_present") is not False or _editorial_calls(
        raw["call_ledger"]
    ):
        raise E1LiveError("raw checkpoint contains pre-freeze editorial calls")
    _assert_pre_board_records(raw["records"])
    for task_id, record in raw["records"].items():
        if record.get("complete_task") is None:
            continue
        if record.get("task_id", task_id) != task_id:
            raise E1LiveError(f"raw record key/task ID mismatch for {task_id}")
        normalized_tech = _record_technical(record)
        normalized_blind = _record_blind(record)
        if raw["technical"][task_id] != normalized_tech:
            raise E1LiveError(f"raw technical projection differs for {task_id}")
        if raw["blind"][task_id] != normalized_blind:
            raise E1LiveError(f"raw blind projection differs for {task_id}")


def _resume_result(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(config["_output_directory"])
    raw_path = output / "raw_post_blind.private.json"
    evidence_path = output / "normalized_pre_board_evidence.json"
    freeze_path = output / "qualification" / "boardless_dispositions.frozen.json"
    source_path = output / "post_blind_source_report.json"
    required = {
        "raw_post_blind_sha256": raw_path,
        "evidence_manifest_sha256": evidence_path,
        "boardless_freeze_sha256": freeze_path,
        "source_report_sha256": source_path,
    }
    for field, path in required.items():
        if not path.is_file() or state.get(field) != _sha(path):
            raise E1LiveError(f"completed resume failed immutable {field} validation")
    raw = read_json(raw_path)
    _validate_raw_checkpoint(raw, config=config, selected=selected)
    frozen = read_json(freeze_path)
    expected_ids = {
        task_id
        for task_id, record in raw["records"].items()
        if record.get("complete_task") is not None
    }
    if {item["candidate_id"] for item in frozen.get("candidates", [])} != expected_ids:
        raise E1LiveError("frozen disposition candidate set differs from raw checkpoint")
    evidence = read_json(evidence_path)
    by_id = {item["candidate_id"]: item for item in evidence.get("candidates", [])}
    if set(by_id) != expected_ids:
        raise E1LiveError("evidence candidate set differs from raw checkpoint")
    for task_id in expected_ids:
        task_ref = by_id[task_id]["task"]
        task_path = Path(task_ref["path"])
        if not task_path.is_file() or _sha(task_path) != task_ref["sha256"]:
            raise E1LiveError(f"task export changed during resume: {task_id}")
        if read_json(task_path) != raw["records"][task_id]["complete_task"]:
            raise E1LiveError(f"task export differs from record: {task_id}")
    return {
        "run_id": config["run"]["id"],
        "runner_kind": config["runner_kind"],
        "post_blind_candidate_count": len(expected_ids),
        "verified_candidate_count": sum(
            item["boardless_disposition"] == "verified_candidate"
            for item in frozen["candidates"]
        ),
        "evidence_manifest": str(evidence_path),
        "boardless_freeze": str(freeze_path),
        "shadow_deferred": not bool(state.get("shadow_dispatched")),
        "automatic_commit": False,
        "production_state_mutated": False,
    }


def run_live_authoring(
    config: Mapping[str, Any],
    *,
    authorize_model_calls: bool,
    backend: LiveBackend | None = None,
) -> dict[str, Any]:
    output = Path(config["_output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "live_state.json"
    protected = _protected_refs(config)
    selected_path = Path(config["_selected_slate_manifest_path"])
    selected = _validate_selected_slate(selected_path, kind=str(config["runner_kind"]))
    selected_ids = sorted(str(item["candidate_id"]) for item in selected["candidates"])
    snapshot = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "runner_kind": config["runner_kind"],
        "config_sha256": _sha(config["_config_path"]),
        "config_path": str(Path(config["_config_path"]).resolve()),
        "protected_refs": protected,
        "selected_slate_sha256": _sha(selected_path),
        "selected_candidate_ids": selected_ids,
        "runtime_contract_sha256": content_hash(config["_runtime_contract"]),
        "model_concurrency": config["_runtime_contract"]["model_concurrency"],
        "escalation_calls_per_slate": config["_runtime_contract"][
            "escalation_calls_per_slate"
        ],
        "routine_editorial_boards_enabled": False,
        "automatic_commit": False,
        "production_state_mutation_allowed": False,
    }
    snapshot_path = output / "live_config.snapshot.json"
    _write_immutable_json(snapshot_path, snapshot, label="live input snapshot")
    state = read_json(state_path) if state_path.exists() else {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "run_id": config["run"]["id"],
        "stage": "initialized",
        "completed": False,
        "started_at_epoch_seconds": round(time.time(), 3),
        "runner_kind": config["runner_kind"],
        "selected_candidate_ids": selected_ids,
        "selected_slate_sha256": _sha(selected_path),
        "live_config_sha256": _sha(config["_config_path"]),
    }
    for field, expected in (
        ("pipeline", PIPELINE),
        ("run_id", config["run"]["id"]),
        ("runner_kind", config["runner_kind"]),
        ("selected_candidate_ids", selected_ids),
        ("selected_slate_sha256", _sha(selected_path)),
        ("live_config_sha256", _sha(config["_config_path"])),
    ):
        if state.get(field) != expected:
            raise E1LiveError(f"live state differs on immutable field {field}")
    if not state_path.exists():
        atomic_write_json(state_path, state)
    existing_evidence = output / "normalized_pre_board_evidence.json"
    existing_freeze = output / "qualification" / "boardless_dispositions.frozen.json"
    if state.get("completed"):
        _assert_protected(protected)
        return _resume_result(config, state, selected)
    raw_path = output / "raw_post_blind.private.json"
    driver = backend or V41IntegratedPostSelectionBackend()
    if raw_path.exists():
        if state.get("raw_post_blind_sha256") != _sha(raw_path):
            raise E1LiveError("raw checkpoint hash differs from live state")
        raw = read_json(raw_path)
        _validate_raw_checkpoint(raw, config=config, selected=selected)
    else:
        result = driver.execute(
            config, selected, authorize_model_calls=authorize_model_calls
        )
        raw = {
            "schema_version": 2,
            "runner_kind": config["runner_kind"],
            "selected_baseline_manifest_sha256": _sha(
                config["_selected_baseline_manifest_path"]
            ),
            "selected_slate_sha256": _sha(selected_path),
            "selected_slate_manifest_sha256": _sha(selected_path),
            "live_config_sha256": _sha(config["_config_path"]),
            "runtime_contract_sha256": content_hash(config["_runtime_contract"]),
            "records": result.records,
            "technical": result.technical,
            "blind": result.blind,
            "call_ledger": result.call_ledger,
            "runtime_run_directory": (
                str(result.runtime.run_dir) if result.runtime is not None else str(output)
            ),
            "runtime_contract_verification": result.runtime_contract_verification,
            "editorial_calls_present": bool(_editorial_calls(result.call_ledger)),
            "post_blind_stop": True,
        }
        _validate_raw_checkpoint(raw, config=config, selected=selected)
        _write_immutable_json(raw_path, raw, label="raw post-blind checkpoint")
        state["stage"] = "post_blind_complete"
        state["raw_post_blind_sha256"] = _sha(raw_path)
        atomic_write_json(state_path, state)
    rows = []
    tasks_dir = output / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    for candidate_id, record in sorted(raw["records"].items()):
        task_path = tasks_dir / f"{candidate_id}.arc.json"
        if record.get("complete_task") is not None:
            _write_immutable_json(
                task_path,
                record["complete_task"],
                label=f"task export {candidate_id}",
            )
            if read_json(task_path) != record["complete_task"]:
                raise E1LiveError(f"task export differs from record: {candidate_id}")
        item = _normalized_candidate(
            selected,
            record,
            raw["technical"].get(candidate_id),
            raw["blind"].get(candidate_id),
        )
        if task_path.exists():
            item["task"] = {"path": str(task_path), "sha256": _sha(task_path)}
        else:
            # The evidence schema requires a frozen task even for an upstream
            # failure.  Such records are retained in the source report but do
            # not enter the disposition manifest.
            continue
        rows.append(item)
    runtime_accounting = _normalize_accounting(
        v4_call_accounting(
            raw.get("call_ledger", {}),
            run_dir=Path(raw.get("runtime_run_directory", output)),
        )
    )
    pre = _normalize_accounting(selected.get("pre_authoring_accounting"))
    production = _sum_accounting(pre, runtime_accounting)
    if production["calls_by_role"].get("editorial_board", 0):
        raise E1LiveError("simulated production accounting contains board calls")
    source_report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "run_id": config["run"]["id"],
        "runner_kind": config["runner_kind"],
        "selected_baseline_manifest_sha256": _sha(
            config["_selected_baseline_manifest_path"]
        ),
        "selected_slate_manifest_sha256": _sha(selected_path),
        "post_blind_stop": True,
        "routine_editorial_calls": 0,
        "candidate_ids": sorted(raw["records"]),
        "normalized_evidence_candidate_ids": [item["candidate_id"] for item in rows],
        "source_hashes": deepcopy(selected.get("source_hashes", {})),
        "source_artifacts": deepcopy(selected["source_artifacts"]),
        "upstream_failures": [
            {
                "candidate_id": candidate_id,
                "plan_status": record.get("status"),
                "realization_status": record.get("realization_status"),
                "reason": record.get("failure_reason") or record.get("abort_reason"),
            }
            for candidate_id, record in sorted(raw["records"].items())
            if record.get("complete_task") is None
        ],
        "accounting": production,
        "accounting_split": {
            "pre_authoring": pre,
            "post_selection_authoring_through_blind": runtime_accounting,
            "simulated_production_combined": production,
            "shadow_editorial_excluded": True,
        },
        "runtime_contract_sha256": content_hash(config["_runtime_contract"]),
        "runtime_contract_verification": deepcopy(
            raw["runtime_contract_verification"]
        ),
        "escalation_usage": deepcopy(
            (
                read_json(
                    Path(raw["runtime_run_directory"]) / "state.json"
                ).get("integrated_escalation_usage", {})
                if (Path(raw["runtime_run_directory"]) / "state.json").is_file()
                else {}
            )
        ),
    }
    source_path = output / "post_blind_source_report.json"
    _write_immutable_json(source_path, source_report, label="post-blind source report")
    evidence = {
        "schema_version": 1,
        "selected_baseline_manifest_sha256": _sha(
            config["_selected_baseline_manifest_path"]
        ),
        "sources": [
            {
                "source_id": "fresh-live-authoring",
                "path": str(source_path),
                "sha256": _sha(source_path),
                "kind": "fresh_post_blind",
            }
        ],
        "candidates": rows,
        "accounting": {
            "simulated_production": production,
            "independent_quality_panel": _normalize_accounting({}),
        },
    }
    evidence_path = output / "normalized_pre_board_evidence.json"
    _write_immutable_json(evidence_path, evidence, label="normalized evidence")
    _assert_protected(protected)
    e1_path = _e1_config(config, evidence_path)
    frozen = freeze_boardless_dispositions(load_e1_config(e1_path))
    state.update(
        {
            "stage": "boardless_dispositions_frozen",
            "completed": True,
            "completed_at_epoch_seconds": round(time.time(), 3),
            "evidence_manifest_sha256": _sha(evidence_path),
            "source_report_sha256": _sha(source_path),
            "boardless_freeze_sha256": _sha(
                output / "qualification" / "boardless_dispositions.frozen.json"
            ),
            "shadow_dispatched": False,
            "production_state_mutated": False,
            "automatic_commit": False,
        }
    )
    atomic_write_json(state_path, state)
    return {
        "run_id": config["run"]["id"],
        "runner_kind": config["runner_kind"],
        "post_blind_candidate_count": len(rows),
        "verified_candidate_count": sum(
            item["boardless_disposition"] == "verified_candidate"
            for item in frozen["candidates"]
        ),
        "evidence_manifest": str(evidence_path),
        "boardless_freeze": str(
            output / "qualification" / "boardless_dispositions.frozen.json"
        ),
        "shadow_deferred": True,
        "automatic_commit": False,
        "production_state_mutated": False,
    }


def dispatch_shadow_boards(
    config: Mapping[str, Any],
    *,
    authorize_model_calls: bool,
    backend: LiveBackend | None = None,
) -> dict[str, Any]:
    if not authorize_model_calls and not config["run"]["mock"]:
        raise E1LiveError("real shadow boards require --authorize-model-calls")
    output = Path(config["_output_directory"])
    freeze_path = output / "qualification" / "boardless_dispositions.frozen.json"
    if not freeze_path.is_file():
        raise E1LiveError("freeze boardless dispositions before dispatching shadow boards")
    state_path = output / "live_state.json"
    state = read_json(state_path)
    selected = _validate_selected_slate(
        Path(config["_selected_slate_manifest_path"]), kind=str(config["runner_kind"])
    )
    _assert_protected(_protected_refs(config))
    if state.get("completed") is not True:
        raise E1LiveError("live authoring must be completely frozen before shadow dispatch")
    _resume_result(config, state, selected)
    attached_path = output / "qualification" / "shadow_editorial.non_gating.json"
    if attached_path.is_file():
        if state.get("shadow_attachment_sha256") not in {None, _sha(attached_path)}:
            raise E1LiveError("attached shadow measurement changed during resume")
        if not state.get("shadow_dispatched"):
            state["shadow_dispatched"] = True
            state["shadow_recovered_after_attach"] = True
            state["shadow_attachment_sha256"] = _sha(attached_path)
            atomic_write_json(state_path, state)
        return read_json(attached_path)
    shadow_path = output / "shadow_editorial_results.private.json"
    if shadow_path.is_file():
        if state.get("shadow_results_sha256") not in {None, _sha(shadow_path)}:
            raise E1LiveError("shadow result changed during resume")
        attached = attach_shadow_results(
            load_e1_config(output / "e1_qualification.config.json"), shadow_path
        )
        state.update(
            {
                "shadow_dispatched": True,
                "shadow_results_sha256": _sha(shadow_path),
                "shadow_is_non_gating": True,
                "shadow_recovered_from_frozen_result": True,
                "shadow_attachment_sha256": _sha(attached_path),
            }
        )
        atomic_write_json(state_path, state)
        return attached
    frozen = read_json(freeze_path)
    clear_ids = [
        item["candidate_id"]
        for item in frozen["candidates"]
        if item["boardless_disposition"] == "verified_candidate"
    ]
    driver = backend or V41IntegratedPostSelectionBackend()
    board_value, accounting = driver.shadow(config, selected, clear_ids)
    task_results = board_value.get("task_results", {})
    results = []
    now = round(time.time(), 3)
    for candidate_id in clear_ids:
        item = task_results.get(candidate_id)
        if not isinstance(item, Mapping) or item.get("operational_failure"):
            results.append(
                {
                    "candidate_id": candidate_id,
                    "operational_failure": True,
                    "judgments": [],
                    "completed_at_epoch_seconds": now,
                }
            )
            continue
        judgments = []
        for index, judgment in enumerate(item.get("judgments", []), 1):
            normalized_judgment = {
                **deepcopy(dict(judgment)),
                "reviewer_id": f"board-{index}",
            }
            # The frozen board schema has no serious_issue field.  Preserve it
            # only if an exact future schema explicitly returns it; absence is
            # not evidence that no hard defect exists.
            if "serious_issue" in judgment:
                normalized_judgment["serious_issue"] = judgment["serious_issue"]
            judgments.append(normalized_judgment)
        results.append(
            {
                "candidate_id": candidate_id,
                "operational_failure": len(judgments) != 2,
                "judgments": judgments if len(judgments) == 2 else [],
                "intrinsic_quality": item.get("intrinsic_quality"),
                "portfolio_placement": item.get("portfolio_placement"),
                "reasons": [item.get("intrinsic_reason"), item.get("portfolio_reason")],
                "completed_at_epoch_seconds": now,
            }
        )
    baseline = NormalizedSelectedBaselineAdapter(config["_selected_baseline_manifest_path"])
    shadow = {
        "schema_version": 1,
        "boardless_freeze_sha256": _sha(freeze_path),
        "policy": {
            "reviewer_count": 2,
            "fixed_anchor_manifest_sha256": baseline.value["frozen_inputs"][
                "fixed_anchor_manifest"
            ]["sha256"],
            "editorial_prompt_sha256": baseline.value["prompts"]["editorial_board"][
                "sha256"
            ],
            "generated_group_minimum": 4,
            "generated_group_maximum": 6,
            "reviewer_routes": [
                {
                    "reviewer_id": f"board-{index}",
                    **deepcopy(baseline.value["model_routes"]["editorial_board"]),
                }
                for index in (1, 2)
            ],
        },
        "results": results,
        "accounting": _normalize_accounting(accounting),
    }
    _write_immutable_json(shadow_path, shadow, label="shadow editorial result")
    attached = attach_shadow_results(
        load_e1_config(output / "e1_qualification.config.json"), shadow_path
    )
    state.update(
        {
            "shadow_dispatched": True,
            "shadow_results_sha256": _sha(shadow_path),
            "shadow_attachment_sha256": _sha(attached_path),
            "shadow_is_non_gating": True,
        }
    )
    atomic_write_json(state_path, state)
    return attached


def validate_live(config: Mapping[str, Any]) -> dict[str, Any]:
    selected = _validate_selected_slate(
        Path(config["_selected_slate_manifest_path"]), kind=str(config["runner_kind"])
    )
    baseline = NormalizedSelectedBaselineAdapter(config["_selected_baseline_manifest_path"])
    resolved = V41IntegratedPostSelectionBackend()._resolved_source_config(config)
    _assert_protected(_protected_refs(config))
    return {
        "pipeline": PIPELINE,
        "version": VERSION,
        "runner_kind": config["runner_kind"],
        "selected_baseline_id": baseline.baseline_id,
        "selected_candidate_count": len(selected["candidates"]),
        "source_provider_counts": {
            provider: sum(
                item["private_kernel_author_provider"] == provider
                for item in selected["candidates"]
            )
            for provider in ("codex", "claude")
        },
        "functional_scale_invitation_count": sum(
            item["optional_functional_scale"] for item in selected["candidates"]
        ),
        "exact_backend": "v41_integrated_post_selection_v1",
        "model_concurrency": int(resolved["model"]["concurrency"]),
        "escalation_calls_per_slate": int(
            resolved["integrated_qualification"]["escalation_calls_per_slate"]
        ),
        "runtime_contract_sha256": content_hash(config["_runtime_contract"]),
        "normal_path_stops_post_blind": True,
        "shadow_dispatch_is_separate": True,
        "automatic_commit": False,
        "production_state_mutation_allowed": False,
    }


def normalize_selected_slate(
    source: str | Path,
    output: str | Path,
    *,
    runner_kind: str,
    route_assignment: str | Path | None = None,
    accounting: str | Path | None = None,
    functional_scale_candidate_ids: Sequence[str] = (),
    supply_manifest: str | Path | None = None,
    supply_attestation: Mapping[str, Any] | None = None,
    cohort_scale_assignment: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize an existing selector result without rerunning selection.

    Two existing formats are supported: V4.5/V4.6 generator-replication
    ``selected_kernel_manifest.private.json`` files and integrated V4.4/V4.5
    ``selected_kernel_manifest.private.json`` files.  The optional route file
    is used only to recover a previously frozen functional-scale assignment.
    """

    if runner_kind not in RUNNER_KINDS:
        raise E1LiveError(f"unsupported runner_kind: {runner_kind!r}")
    source_path = Path(source).resolve()
    destination = Path(output).resolve()
    value = read_json(source_path)
    routes: dict[str, str] = {}
    source_artifacts = [
        {
            "source_id": "selected-kernel-manifest",
            "path": str(source_path),
            "sha256": _sha(source_path),
        }
    ]
    if route_assignment is not None:
        route_path = Path(route_assignment).resolve()
        route_value = read_json(route_path)
        assignments = route_value.get("assignments")
        if not isinstance(assignments, list) or any(
            not isinstance(item, Mapping) for item in assignments
        ):
            raise E1LiveError("route assignment has no assignments list")
        route_ids = [str(item.get("source_candidate_id")) for item in assignments]
        if len(route_ids) != len(set(route_ids)) or any(
            item.get("private_route") not in {"ordinary_current", "functional_scale"}
            for item in assignments
        ):
            raise E1LiveError(
                "route assignment contains duplicate IDs or a non-E1 downstream route"
            )
        routes = {
            str(item["source_candidate_id"]): str(item["private_route"])
            for item in assignments
        }
        source_artifacts.append(
            {
                "source_id": "private-route-assignment",
                "path": str(route_path),
                "sha256": _sha(route_path),
            }
        )
    explicit_scale = {str(item) for item in functional_scale_candidate_ids}
    candidates: list[dict[str, Any]] = []
    if isinstance(value.get("selected"), list):
        # V4.5/V4.6 generator-replication selector output.
        for row in value["selected"]:
            assessment = row.get("selector_assessment") or {}
            same = [str(item) for item in (assessment.get("same_minimal_procedure_with") or [])]
            candidate_id = str(row["candidate_id"])
            explicit_hard_evidence = any(
                assessment.get(field)
                for field in (
                    "duplicate_of_accepted_task_id",
                    "duplicate_of_inflight_candidate_id",
                    "hard_duplicate_memory_id",
                )
            )
            explicit_duplicate_disposition = assessment.get("disposition") in {
                "same_procedure_duplicate",
                "hard_duplicate",
                "reject_duplicate",
            }
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "lineage_id": str(row.get("lineage_id") or candidate_id),
                    "private_kernel_author_provider": str(
                        row["private_kernel_author_provider"]
                    ),
                    "optional_functional_scale": candidate_id in explicit_scale
                    or routes.get(candidate_id) == "functional_scale",
                    "slate_index": int(row["slate_index"]),
                    "search_mode": str(row["search_mode"]),
                    "kernel": deepcopy(row["kernel"]),
                    "editorial_brief": deepcopy(row["editorial_brief"]),
                    "selector_global_priority": int(row["selector_global_priority"]),
                    "novelty": {
                        "selector_archive_checked": True,
                        # A selected winner may mention the sibling it beat.
                        # That makes it the retained representative, not the
                        # duplicate being rejected.
                        "confirmed_same_procedure_duplicate": bool(
                            assessment.get("confirmed_same_procedure_duplicate") is True
                            or explicit_hard_evidence
                            or explicit_duplicate_disposition
                        ),
                        "within_window_sibling_ids": same,
                        "selected_as_sibling_representative": bool(same),
                        "theme_is_common": assessment.get("theme_is_common"),
                    },
                }
            )
    elif isinstance(value.get("development_sample"), list):
        # Integrated V4.4/V4.5 selector output.
        for row in value["development_sample"]:
            candidate_id = str(row["source_candidate_id"])
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "lineage_id": str(row.get("lineage_id") or candidate_id),
                    "private_kernel_author_provider": "codex",
                    "optional_functional_scale": candidate_id in explicit_scale
                    or routes.get(candidate_id) == "functional_scale",
                    "slate_index": int(row["slate_index"]),
                    "search_mode": str(row["kernel"]["search_mode"]),
                    "kernel": deepcopy(row["kernel"]),
                    "editorial_brief": deepcopy(row["editorial_brief"]),
                    "selector_global_priority": int(row["source_development_priority"]),
                    "novelty": {
                        "selector_archive_checked": True,
                        "confirmed_same_procedure_duplicate": False,
                        "within_window_sibling_ids": [],
                        "selected_as_sibling_representative": False,
                        "theme_is_common": None,
                    },
                }
            )
    else:
        raise E1LiveError("unrecognized selected-kernel manifest format")
    candidate_ids = {item["candidate_id"] for item in candidates}
    unknown_scale = explicit_scale - candidate_ids
    if unknown_scale:
        raise E1LiveError(
            "functional-scale assignment names unselected candidates: "
            + ", ".join(sorted(unknown_scale))
        )
    if routes and candidate_ids - set(routes):
        missing = candidate_ids - set(routes)
        raise E1LiveError(
            "route assignment omits selected candidates: " + ", ".join(sorted(missing))
        )
    accounting_value = (
        _normalize_accounting(read_json(Path(accounting).resolve()))
        if accounting is not None
        else _normalize_accounting({})
    )
    if accounting is not None:
        accounting_path = Path(accounting).resolve()
        source_artifacts.append(
            {
                "source_id": "pre-authoring-accounting",
                "path": str(accounting_path),
                "sha256": _sha(accounting_path),
            }
        )
    if cohort_scale_assignment is not None:
        cohort_scale_path = Path(cohort_scale_assignment).resolve()
        if not cohort_scale_path.is_file():
            raise E1LiveError("cohort functional-scale assignment does not exist")
        source_artifacts.append(
            {
                "source_id": "cohort-functional-scale-assignment",
                "path": str(cohort_scale_path),
                "sha256": _sha(cohort_scale_path),
            }
        )
    normalized_attestation = None
    if (supply_manifest is None) != (supply_attestation is None):
        raise E1LiveError(
            "supply_manifest and supply_attestation must be supplied together"
        )
    if supply_manifest is not None and supply_attestation is not None:
        supply_path = Path(supply_manifest).resolve()
        if not supply_path.is_file():
            raise E1LiveError("minority Opus supply manifest does not exist")
        source_id = str(
            supply_attestation.get("supply_manifest_source_id")
            or "minority-opus-supply-manifest"
        )
        if not source_id or source_id in {item["source_id"] for item in source_artifacts}:
            raise E1LiveError("minority Opus supply manifest source ID is duplicated")
        supply_digest = _sha(supply_path)
        source_artifacts.append(
            {
                "source_id": source_id,
                "path": str(supply_path),
                "sha256": supply_digest,
            }
        )
        normalized_attestation = {
            **deepcopy(dict(supply_attestation)),
            "supply_manifest_source_id": source_id,
            "supply_manifest_sha256": supply_digest,
        }
    normalized = {
        "schema_version": 1,
        "selection_completed": True,
        "source_labels_hidden_downstream": True,
        "runner_kind": runner_kind,
        "source_artifacts": source_artifacts,
        "source_hashes": {
            item["source_id"]: item["sha256"] for item in source_artifacts
        },
        "pre_authoring_accounting": accounting_value,
        "candidates": candidates,
    }
    if normalized_attestation is not None:
        normalized["supply_attestation"] = normalized_attestation
    _validate_selected_slate_value_path = destination.parent / (
        "." + destination.name + ".validation.tmp"
    )
    # Validate the exact serialized form, then remove the temporary through a
    # simple unlink; no production source is touched.
    atomic_write_json(_validate_selected_slate_value_path, normalized)
    try:
        _validate_selected_slate(
            _validate_selected_slate_value_path, kind=runner_kind
        )
    finally:
        _validate_selected_slate_value_path.unlink(missing_ok=True)
    if destination.exists() and read_json(destination) != normalized:
        raise E1LiveError("a different normalized selected slate already exists")
    if not destination.exists():
        atomic_write_json(destination, normalized)
    return normalized


def aggregate_live_runs(
    run_directories: Sequence[str | Path], output: str | Path
) -> dict[str, Any]:
    """Deterministically aggregate immutable, completed E1 live slates."""

    if not run_directories:
        raise E1LiveError("aggregate requires at least one completed live run")
    directories = sorted({Path(item).resolve() for item in run_directories}, key=str)
    if len(directories) != len(run_directories):
        raise E1LiveError("aggregate input contains a duplicate run directory")
    rows = []
    run_ids: set[str] = set()
    selected_hashes: set[str] = set()
    candidate_ids: set[str] = set()
    lineage_ids: set[str] = set()
    baseline_hash: str | None = None
    runner_kind: str | None = None
    accounting_rows = []
    for directory in directories:
        required = (
            "live_state.json",
            "live_config.snapshot.json",
            "raw_post_blind.private.json",
            "post_blind_source_report.json",
            "normalized_pre_board_evidence.json",
            "qualification/boardless_dispositions.frozen.json",
        )
        if any(not (directory / item).is_file() for item in required):
            raise E1LiveError(f"aggregate input is incomplete: {directory}")
        state = read_json(directory / "live_state.json")
        snapshot = read_json(directory / "live_config.snapshot.json")
        if state.get("completed") is not True:
            raise E1LiveError(f"aggregate input is not completed: {directory}")
        for field, relative in (
            ("raw_post_blind_sha256", "raw_post_blind.private.json"),
            ("source_report_sha256", "post_blind_source_report.json"),
            ("evidence_manifest_sha256", "normalized_pre_board_evidence.json"),
            (
                "boardless_freeze_sha256",
                "qualification/boardless_dispositions.frozen.json",
            ),
        ):
            if state.get(field) != _sha(directory / relative):
                raise E1LiveError(f"aggregate input failed {field}: {directory}")
        _assert_protected(snapshot["protected_refs"])
        run_id = str(snapshot["run_id"])
        if run_id in run_ids:
            raise E1LiveError(f"aggregate run_id is duplicated: {run_id}")
        run_ids.add(run_id)
        selected_hash = str(snapshot["selected_slate_sha256"])
        if selected_hash in selected_hashes:
            raise E1LiveError("aggregate reuses the same frozen selected slate")
        selected_hashes.add(selected_hash)
        current_baseline = snapshot["protected_refs"]["selected_baseline_manifest"][
            "sha256"
        ]
        current_kind = str(snapshot["runner_kind"])
        if baseline_hash is None:
            baseline_hash = current_baseline
            runner_kind = current_kind
        elif (current_baseline, current_kind) != (baseline_hash, runner_kind):
            raise E1LiveError("aggregate inputs do not share one selected baseline/kind")
        raw = read_json(directory / "raw_post_blind.private.json")
        source = read_json(directory / "post_blind_source_report.json")
        evidence = read_json(directory / "normalized_pre_board_evidence.json")
        frozen = read_json(
            directory / "qualification" / "boardless_dispositions.frozen.json"
        )
        if raw.get("editorial_calls_present") is not False or _editorial_calls(
            raw.get("call_ledger", {})
        ):
            raise E1LiveError("aggregate input contains routine editorial calls")
        if raw.get("post_blind_stop") is not True:
            raise E1LiveError("aggregate input did not stop at the post-blind boundary")
        by_evidence = {item["candidate_id"]: item for item in evidence["candidates"]}
        by_frozen = {item["candidate_id"]: item for item in frozen["candidates"]}
        expected_task_ids = {
            task_id
            for task_id, record in raw.get("records", {}).items()
            if record.get("complete_task") is not None
        }
        if set(by_evidence) != set(by_frozen) or set(by_evidence) != expected_task_ids:
            raise E1LiveError("aggregate raw/evidence/freeze candidate sets differ")
        tasks = []
        for task_id in sorted(by_evidence):
            task_ref = by_evidence[task_id].get("task")
            if not isinstance(task_ref, Mapping):
                raise E1LiveError(f"aggregate task reference is missing for {task_id}")
            task_path = Path(str(task_ref.get("path")))
            if (
                not task_path.is_file()
                or _sha(task_path) != task_ref.get("sha256")
                or read_json(task_path) != raw["records"][task_id]["complete_task"]
            ):
                raise E1LiveError(f"aggregate task export differs from record: {task_id}")
            lineage = str(by_evidence[task_id].get("lineage_id") or task_id)
            if task_id in candidate_ids or lineage in lineage_ids:
                raise E1LiveError(
                    "aggregate contains duplicated candidate or lineage identity"
                )
            candidate_ids.add(task_id)
            lineage_ids.add(lineage)
            tasks.append(
                {
                    "candidate_id": task_id,
                    "lineage_id": lineage,
                    "source_run_id": run_id,
                    "boardless_disposition": by_frozen[task_id][
                        "boardless_disposition"
                    ],
                    "task": deepcopy(by_evidence[task_id]["task"]),
                }
            )
        accounting = _normalize_accounting(source["accounting"])
        accounting_rows.append(accounting)
        rows.append(
            {
                "run_id": run_id,
                "path": str(directory),
                "snapshot_sha256": _sha(directory / "live_config.snapshot.json"),
                "selected_slate_sha256": selected_hash,
                "source_report_sha256": _sha(
                    directory / "post_blind_source_report.json"
                ),
                "task_count": len(tasks),
                "verified_candidate_count": sum(
                    item["boardless_disposition"] == "verified_candidate"
                    for item in tasks
                ),
                "shadow_attached": (
                    directory
                    / "qualification"
                    / "shadow_editorial.non_gating.json"
                ).is_file(),
                "tasks": tasks,
                "accounting": accounting,
            }
        )
    rows.sort(key=lambda item: item["run_id"])
    tasks = sorted(
        [task for row in rows for task in row["tasks"]],
        key=lambda item: (item["source_run_id"], item["candidate_id"]),
    )
    report = {
        "schema_version": 1,
        "pipeline": "arcgen-e1-live-multi-slate-aggregate",
        "selected_baseline_manifest_sha256": baseline_hash,
        "runner_kind": runner_kind,
        "source_runs": rows,
        "funnel": {
            "run_count": len(rows),
            "candidate_count": len(tasks),
            "verified_candidate_count": sum(
                item["boardless_disposition"] == "verified_candidate"
                for item in tasks
            ),
            "not_verified_candidate_count": sum(
                item["boardless_disposition"] != "verified_candidate"
                for item in tasks
            ),
            "shadow_attached_run_count": sum(item["shadow_attached"] for item in rows),
        },
        "tasks": tasks,
        "accounting": {
            "simulated_production": _sum_accounting(*accounting_rows),
            "per_run_preserved": True,
            "shadow_editorial_excluded": True,
        },
        "deterministic_input_order": [item["run_id"] for item in rows],
        "automatic_commit_occurred": False,
        "production_state_mutated": False,
    }
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "aggregate_report.json"
    _write_immutable_json(report_path, report, label="multi-slate aggregate")
    lines = [
        "# E1 live multi-slate aggregate",
        "",
        f"- Runs: {report['funnel']['run_count']}",
        f"- Candidates: {report['funnel']['candidate_count']}",
        f"- Neutral verified candidates: {report['funnel']['verified_candidate_count']}",
        "- Routine editorial calls included: 0",
        "- Automatic commits: 0",
        "",
    ]
    markdown = destination / "aggregate_report.md"
    text = "\n".join(lines)
    if markdown.exists() and markdown.read_text(encoding="utf-8") != text:
        raise E1LiveError("multi-slate aggregate markdown changed during resume")
    if not markdown.exists():
        from .util import atomic_write_text

        atomic_write_text(markdown, text)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E1 live post-selection adapter")
    parser.add_argument(
        "command", choices=("normalize-selection", "validate", "run", "shadow", "aggregate")
    )
    parser.add_argument("--config")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--runner-kind", choices=sorted(RUNNER_KINDS))
    parser.add_argument("--route-assignment")
    parser.add_argument("--accounting")
    parser.add_argument("--supply-manifest")
    parser.add_argument("--supply-attestation")
    parser.add_argument("--cohort-functional-scale-assignment")
    parser.add_argument("--functional-scale-candidate", action="append", default=[])
    parser.add_argument("--authorize-model-calls", action="store_true")
    parser.add_argument("--run-directory", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "aggregate":
            if not args.output or not args.run_directory:
                raise E1LiveError(
                    "aggregate requires --output and at least one --run-directory"
                )
            value = aggregate_live_runs(args.run_directory, args.output)
            print(json.dumps(value, indent=2, sort_keys=True))
            return 0
        if args.command == "normalize-selection":
            if not args.source or not args.output or not args.runner_kind:
                raise E1LiveError(
                    "normalize-selection requires --source, --output, and --runner-kind"
                )
            value = normalize_selected_slate(
                args.source,
                args.output,
                runner_kind=args.runner_kind,
                route_assignment=args.route_assignment,
                accounting=args.accounting,
                functional_scale_candidate_ids=args.functional_scale_candidate,
                supply_manifest=args.supply_manifest,
                supply_attestation=(
                    (
                        lambda value: value.get("supply_attestation", value)
                    )(read_json(Path(args.supply_attestation).resolve()))
                    if args.supply_attestation
                    else None
                ),
                cohort_scale_assignment=args.cohort_functional_scale_assignment,
            )
            print(json.dumps(value, indent=2, sort_keys=True))
            return 0
        if not args.config:
            raise E1LiveError(f"{args.command} requires --config")
        config = load_live_config(args.config)
        if args.command == "validate":
            value = validate_live(config)
        elif args.command == "run":
            value = run_live_authoring(
                config, authorize_model_calls=args.authorize_model_calls
            )
        else:
            value = dispatch_shadow_boards(
                config, authorize_model_calls=args.authorize_model_calls
            )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except (E1LiveError, E1QualificationError) as exc:
        print(f"E1 live error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPOSITE",
    "E1LiveError",
    "V41IntegratedPostSelectionBackend",
    "LiveExecutionResult",
    "PIPELINE",
    "RUNNER_KINDS",
    "V44_ORDINARY",
    "V45_SCALE",
    "V46_SUPPLY",
    "dispatch_shadow_boards",
    "aggregate_live_runs",
    "load_live_config",
    "normalize_selected_slate",
    "run_live_authoring",
    "validate_live",
]
