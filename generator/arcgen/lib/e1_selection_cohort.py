"""Selection-only cohort producer for the E1 boardless qualification.

This additive adapter deliberately stops at the selected-slate boundary.  It
does not contain a planner, realizer, verifier, solver, or editorial-board
call.  Instead it invokes the already-qualified supply and selector entry
points and serializes their results into the immutable schema consumed by
``e1_live_runner``.

Two source controls are supported:

``minority_opus_supply``
    Reuses the V4.6 paired-supply function and its source-blind Sol/max shard
    and global selectors.  Only the hybrid 48-kernel window is selectable.
    The three paired Sol control calls that V4.6 necessarily makes are kept in
    the actual accounting as explicit qualification overhead.

``all_sol``
    Reuses :class:`V4ReservoirProducer` and :class:`V4ControlPipeline` against
    an isolated selection reservoir.  No Claude/Opus route is instantiated.

Production registry and reservoir files are read and hash-guarded, never
written.  A run-local reservoir carries reservations from one selected slate
to the next so later selectors can avoid mechanisms already in flight.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import ConfigError
from .e1_live_runner import (
    COMPOSITE,
    V44_ORDINARY,
    V45_SCALE,
    V46_SUPPLY,
    _validate_selected_slate,
    normalize_selected_slate,
)
from .util import atomic_write_json, read_json
from .v4_config import load_v4_config
from .v4_pipeline import (
    V4ControlPipeline,
    V4ReservoirProducer,
    _DataContext,
    v4_call_accounting,
)
from .v4_reservoir import (
    append_supply,
    freeze_eligible_window,
    heartbeat_leases,
    load_reservoir,
    recover_expired_leases,
    reserve_slate,
    resolve_proposal,
    validate_reservoir,
)
from .v46_hybrid_pipeline import (
    _runtime as _v46_runtime,
)
from .v46_hybrid_pipeline import (
    build_arm_reservoir,
    generate_paired_supply,
    load_hybrid_config,
    select_arm,
)

PIPELINE = "arcgen-e1-selection-only-cohort"
VERSION = "1.1"
PRODUCER_KINDS = frozenset({"minority_opus_supply", "all_sol"})
MAXIMUM_COHORT_SLATES = 8

# This is deliberately a narrow implementation closure rather than the whole
# repository.  It freezes every module whose functions this adapter invokes,
# plus the normalizer that defines its handoff schema.  A resumed cohort must
# continue against these exact bytes.
IMPLEMENTATION_DEPENDENCIES = (
    "claude_code.py",
    "committed_tasks.py",
    "config.py",
    "data.py",
    "e1_selection_cohort.py",
    "e1_live_runner.py",
    "lean_pipeline.py",
    "model.py",
    "v4_config.py",
    "v4_diagnostics.py",
    "v4_pipeline.py",
    "v4_reservoir.py",
    "v4_schemas.py",
    "v41_experiments.py",
    "v41_integration.py",
    "v41_palette.py",
    "v43_episode_joint.py",
    "v431_episode_joint_quality.py",
    "v44_scale_fresh_integrated.py",
    "v45_claude_config.py",
    "v45_claude_cross_eval.py",
    "v45_claude_generator_replication.py",
    "v45_config.py",
    "v46_hybrid_pipeline.py",
    "embeddings.py",
    "retrieval.py",
    "util.py",
    "validation.py",
)


class E1SelectionCohortError(RuntimeError):
    """Raised when the selection-only or isolation contract is violated."""


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"E1 selection cohort config must be a mapping: {path}")
    return value


def _resolve(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _write_immutable(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    if path.exists():
        if read_json(path) != value:
            raise E1SelectionCohortError(f"{label} changed during resume: {path}")
        return
    atomic_write_json(path, value)


def load_cohort_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read_yaml_or_json(source)
    if raw.get("pipeline") != PIPELINE or str(raw.get("version")) != VERSION:
        raise ConfigError(f"selection cohort must use {PIPELINE} {VERSION}")
    if raw.get("template") is not False or raw.get("enabled") is not True:
        raise ConfigError(
            "create an explicitly enabled cohort config from the template"
        )
    root = source.parent.parent if source.parent.name == "config" else source.parent
    kind = raw.get("producer_kind")
    if kind not in PRODUCER_KINDS:
        raise ConfigError(f"producer_kind must be one of {sorted(PRODUCER_KINDS)}")
    runner_kind = raw.get("runner_kind")
    if runner_kind not in {V44_ORDINARY, V45_SCALE, V46_SUPPLY, COMPOSITE}:
        raise ConfigError("runner_kind must freeze one supported E1 baseline kind")
    source_ref = raw.get("source_config")
    if not isinstance(source_ref, Mapping):
        raise ConfigError("source_config must freeze path and sha256")
    source_path = _resolve(root, source_ref.get("path"), "source_config.path")
    digest = source_ref.get("sha256")
    if (
        not source_path.is_file()
        or not isinstance(digest, str)
        or _sha(source_path) != digest
    ):
        raise ConfigError("source_config is missing or changed")
    run = raw.get("run")
    if (
        not isinstance(run, Mapping)
        or not isinstance(run.get("id"), str)
        or not run["id"]
    ):
        raise ConfigError("run.id is required")
    if type(run.get("mock")) is not bool:
        raise ConfigError("run.mock must be boolean")
    for field in (
        "selection_only",
        "stop_before_planning",
        "source_labels_hidden_from_selectors",
        "whole_slate_reserved_before_authoring",
    ):
        if run.get(field) is not True:
            raise ConfigError(f"run.{field} must be true")
    for field in (
        "production_state_mutation_allowed",
        "automatic_task_commit",
        "auxiliary_opus_routes_enabled",
    ):
        if run.get(field) is not False:
            raise ConfigError(f"run.{field} must be false")
    cohort = raw.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ConfigError("cohort settings are required")
    exact = {
        "eligible_window_size": 48,
        "shard_size": 12,
        "maximum_selected_per_slate": 8,
    }
    for field, expected in exact.items():
        if cohort.get(field) != expected:
            raise ConfigError(f"cohort.{field} must be {expected}")
    if (
        type(cohort.get("slate_count")) is not int
        or not 1 <= cohort["slate_count"] <= MAXIMUM_COHORT_SLATES
    ):
        raise ConfigError(
            f"cohort.slate_count must be between 1 and {MAXIMUM_COHORT_SLATES}"
        )
    if type(cohort.get("seed_stride")) is not int or cohort["seed_stride"] < 1:
        raise ConfigError("cohort.seed_stride must be positive")
    seed = raw.get("seed")
    if type(seed) is not int:
        raise ConfigError("seed must be an integer")
    scale = raw.get("functional_scale")
    if not isinstance(scale, Mapping) or type(scale.get("enabled")) is not bool:
        raise ConfigError("functional_scale.enabled must be boolean")
    fraction = scale.get("target_selected_fraction")
    if type(fraction) not in {int, float} or not 0 <= float(fraction) <= 0.20:
        raise ConfigError(
            "functional_scale.target_selected_fraction must be in [0, 0.20]"
        )
    if scale.get("maximum_per_slate") != 1:
        raise ConfigError("functional_scale.maximum_per_slate must remain one")
    if scale.get("assignment_stage") != "after_selection":
        raise ConfigError("functional scale must be assigned after selection")
    if scale.get("planner_may_decline") is not True:
        raise ConfigError("functional_scale.planner_may_decline must be true")
    expected_runner = {
        ("minority_opus_supply", False): V46_SUPPLY,
        ("minority_opus_supply", True): COMPOSITE,
        ("all_sol", False): V44_ORDINARY,
        ("all_sol", True): V45_SCALE,
    }[(str(kind), bool(scale["enabled"]))]
    if runner_kind != expected_runner:
        raise ConfigError(
            "runner_kind must be frozen cohort-wide and agree with producer/scale "
            f"policy ({expected_runner})"
        )
    output = _resolve(root, raw.get("output_directory"), "output_directory")

    # Load and validate the exact parent now; the resulting mapping is never
    # serialized as a replacement for its frozen source.
    parent = (
        load_hybrid_config(source_path)
        if kind == "minority_opus_supply"
        else load_v4_config(source_path)
    )
    if kind == "minority_opus_supply":
        settings = parent["hybrid_experiment"]
        if (
            float(settings["hybrid_claude_call_fraction"]) != 0.25
            or settings["source_balancing_enabled"] is not False
            or settings["maximum_development_per_arm"] != 8
        ):
            raise ConfigError(
                "hybrid parent is not the qualified 25%-Opus source-blind control"
            )
    value = deepcopy(raw)
    value["_config_path"] = str(source)
    value["_repository_root"] = str(root)
    value["_source_config_path"] = str(source_path)
    value["_output_directory"] = str(output)
    value["_parent"] = parent
    value["_source_balancing_enabled"] = False
    return value


def _protected_state(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    parent = config["_parent"]
    paths = parent["paths"]
    result: dict[str, dict[str, Any]] = {
        "source_config": {
            "path": config["_source_config_path"],
            "sha256": _sha(config["_source_config_path"]),
        }
    }
    for name in ("committed_task_registry", "reservoir"):
        path = Path(paths[name]).resolve()
        result[name] = {"path": str(path), "sha256": _sha(path)}
    return result


def _assert_protected_unchanged(expected: Mapping[str, Mapping[str, Any]]) -> None:
    changed = [
        name
        for name, item in expected.items()
        if not Path(str(item["path"])).is_file()
        or _sha(str(item["path"])) != item["sha256"]
    ]
    if changed:
        raise E1SelectionCohortError(
            "protected production input changed during cohort run: "
            + ", ".join(changed)
        )


def _bundle_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _file_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    unique = sorted({path.resolve() for path in paths if path.is_file()}, key=str)
    return [{"path": str(path), "sha256": _sha(path)} for path in unique]


def _dependency_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the exact non-state inputs behind supply and selection."""

    root = Path(config["_repository_root"]).resolve()
    package = Path(__file__).resolve().parent
    parent = config["_parent"]
    implementation = _file_rows(
        [package / filename for filename in IMPLEMENTATION_DEPENDENCIES]
    )

    prompt_dir = Path(parent.get("paths", {}).get("prompts", root / "prompts"))
    if config["producer_kind"] == "minority_opus_supply":
        prompt_names = (
            "v4_idea_open.md",
            "v4_idea_concept.md",
            "v4_idea_composition.md",
            "v4_4_global_quality_selector.md",
            "v4_5_claude_code_system.md",
        )
    else:
        prompt_names = (
            "v4_idea_open.md",
            "v4_idea_concept.md",
            "v4_idea_composition.md",
            "v4_shard_selector.md",
            "v4_global_selector.md",
        )
    prompts = _file_rows([prompt_dir / name for name in prompt_names])

    calibration_paths: list[Path] = []
    frozen_source = parent.get("paths", {}).get("frozen_generation_source_run")
    if frozen_source:
        calibration_paths.extend(
            sorted(
                Path(frozen_source).glob(
                    "selection_graph/v4_reservoir_supply/*/model_calls/"
                    "v4_idea_search/*/prompt.md"
                )
            )
        )
    else:
        annotations = parent.get("paths", {}).get("annotations")
        if annotations and Path(annotations).is_dir():
            calibration_paths.extend(
                sorted(path for path in Path(annotations).rglob("*") if path.is_file())
            )
    calibrations = _file_rows(calibration_paths)

    reference_paths: list[Path] = []
    for key in ("annotations", "tasks"):
        location = parent.get("paths", {}).get(key)
        if location and Path(location).is_dir():
            reference_paths.extend(
                sorted(path for path in Path(location).rglob("*") if path.is_file())
            )
    reference_bank = _file_rows(reference_paths)

    baseline_paths: list[Path] = [Path(config["_source_config_path"])]
    for key, item in parent.items():
        if not key.startswith("_") or not isinstance(item, str):
            continue
        candidate = Path(item)
        if candidate.is_file() and any(
            token in key for token in ("manifest", "config", "plan")
        ):
            baseline_paths.append(candidate)
    for key in (
        "fixed_anchor_manifest",
        "parent_manifest",
        "generated_reference_policy",
        "v4_1_feature_manifest",
        "v4_1_baseline_manifest",
        "v4_2_baseline_manifest",
    ):
        candidate = parent.get("paths", {}).get(key)
        if candidate and Path(candidate).is_file():
            baseline_paths.append(Path(candidate))
    baselines = _file_rows(baseline_paths)
    embedding = deepcopy(parent.get("embedding", {}))
    manifest = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "implementation_files": implementation,
        "implementation_bundle_sha256": _bundle_hash(implementation),
        "prompt_files": prompts,
        "prompt_bundle_sha256": _bundle_hash(prompts),
        "calibration_files": calibrations,
        "calibration_bundle_sha256": _bundle_hash(calibrations),
        "reference_bank_files": reference_bank,
        "reference_bank_bundle_sha256": _bundle_hash(reference_bank),
        "baseline_files": baselines,
        "baseline_bundle_sha256": _bundle_hash(baselines),
        "embedding_configuration": embedding,
        "embedding_configuration_sha256": _bundle_hash([embedding]),
    }
    manifest["dependency_bundle_sha256"] = _bundle_hash(
        [
            {
                key: value
                for key, value in manifest.items()
                if key != "dependency_bundle_sha256"
            }
        ]
    )
    return manifest


def _assert_dependency_manifest_unchanged(value: Mapping[str, Any]) -> None:
    for field in (
        "implementation_files",
        "prompt_files",
        "calibration_files",
        "reference_bank_files",
        "baseline_files",
    ):
        rows = value.get(field)
        if not isinstance(rows, list):
            raise E1SelectionCohortError(f"dependency manifest lacks {field}")
        changed = [
            str(item.get("path"))
            for item in rows
            if not isinstance(item, Mapping)
            or not Path(str(item.get("path", ""))).is_file()
            or _sha(str(item["path"])) != item.get("sha256")
        ]
        if changed:
            raise E1SelectionCohortError(
                f"frozen {field} changed during cohort run: " + ", ".join(changed)
            )
        hash_field = field.replace("_files", "_bundle_sha256")
        if value.get(hash_field) != _bundle_hash(rows):
            raise E1SelectionCohortError(f"dependency {field} bundle hash is invalid")
    if value.get("embedding_configuration_sha256") != _bundle_hash(
        [value.get("embedding_configuration", {})]
    ):
        raise E1SelectionCohortError("embedding configuration hash is invalid")
    observed_all = _bundle_hash(
        [
            {
                key: item
                for key, item in value.items()
                if key != "dependency_bundle_sha256"
            }
        ]
    )
    if value.get("dependency_bundle_sha256") != observed_all:
        raise E1SelectionCohortError("dependency manifest aggregate hash is invalid")


def _context_only_reservoir(source: Mapping[str, Any]) -> dict[str, Any]:
    """Keep selector memory/reservations while excluding selectable proposals."""

    active_ids = {item["proposal_id"] for item in source["active_reservations"]}
    value = deepcopy(dict(source))
    value["proposals"] = [
        deepcopy(item)
        for item in source["proposals"]
        if item["state"] == "accepted" or item["proposal_id"] in active_ids
    ]
    value["frozen_windows"] = []
    value["supply_history"] = []
    value["lifecycle_events"] = []
    value.pop("updated_at_epoch_seconds", None)
    return validate_reservoir(value, path="selection_context_snapshot")


def _copy_context(source: Path, destination: Path) -> None:
    if destination.exists():
        load_reservoir(destination)
        return
    atomic_write_json(destination, _context_only_reservoir(load_reservoir(source)))


def _merge_ledgers(*rows: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix, ledger in rows:
        for key, item in ledger.items():
            target = f"{prefix}/{key}"
            if target in result:
                raise E1SelectionCohortError(f"duplicate call ledger key {target}")
            result[target] = deepcopy(item)
    return result


def _cohort_scale_assignments(
    slates: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    fraction: float,
    maximum_per_slate: int,
    seed: int,
) -> dict[int, list[str]]:
    """Assign invitations over the full frozen cohort, never per tiny slate."""

    result = {int(item["slate_index"]): [] for item in slates}
    if not enabled:
        return result
    candidates = [
        (int(slate["slate_index"]), row)
        for slate in slates
        for row in slate["source_rows"]
    ]
    if not candidates:
        return result
    target = math.floor(len(candidates) * fraction + 0.5)
    # V4.5's frozen runner contract requires at least one invitation.  This is
    # only material for an unusually undersized cohort; it is still a ceiling,
    # not a quality quota, because planners may decline the invitation.
    target = max(1, target)
    target = min(target, len(slates) * maximum_per_slate)
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item[1]["selector_global_priority"]),
            sha256(
                f"{seed}:{item[0]}:{item[1]['candidate_id']}".encode()
            ).hexdigest(),
        ),
    )
    for slate_index, row in ordered:
        if len(result[slate_index]) >= maximum_per_slate:
            continue
        result[slate_index].append(str(row["candidate_id"]))
        if sum(len(ids) for ids in result.values()) == target:
            break
    if sum(len(ids) for ids in result.values()) != target:
        raise E1SelectionCohortError("could not assign the cohort-level scale target")
    return result


def _freeze_selected_context(
    *,
    working_context: Path,
    proposal_rows: Mapping[str, Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    block_id: str,
    slate_index: int,
    seed: int,
    bank_snapshot_id: str,
    accepted_hashes: Sequence[str],
    lease_seconds: int,
) -> dict[str, Any]:
    ids = [str(item["candidate_id"]) for item in selected]
    current = load_reservoir(working_context)
    active_by_id = {
        item["proposal_id"]: item for item in current["active_reservations"]
    }
    if ids and set(ids) <= set(active_by_id):
        expected_owner = block_id
        expected_slate = f"slate:{block_id}:{slate_index:02d}"
        if any(
            active_by_id[item]["owner_run_id"] != expected_owner
            or active_by_id[item]["owner_slate_id"] != expected_slate
            for item in ids
        ):
            raise E1SelectionCohortError(
                "resumed cross-slate reservations have a different owner"
            )
        window_id = f"window:{block_id}:selected-{slate_index:02d}"
        frozen = next(
            (
                item
                for item in current["frozen_windows"]
                if item["window_id"] == window_id
            ),
            None,
        )
        if frozen is None or set(frozen["proposal_ids"]) != set(ids):
            raise E1SelectionCohortError(
                "resumed cross-slate reservation lacks its exact frozen window"
            )
        return {
            "window": deepcopy(frozen),
            "reservations": {
                "reservations": [deepcopy(active_by_id[item]) for item in ids],
                "reserved_atomically": True,
                "resumed": True,
            },
        }
    if set(ids) & set(active_by_id):
        raise E1SelectionCohortError(
            "only part of a selected slate survived in the working reservation context"
        )
    proposals = [deepcopy(dict(proposal_rows[item])) for item in ids]
    if any(item["state"] != "proposed" for item in proposals):
        raise E1SelectionCohortError("pre-selection proposal snapshot is not proposed")
    append_supply(
        working_context,
        proposals,
        hard_capacity=max(
            144, len(load_reservoir(working_context)["proposals"]) + len(ids) + 8
        ),
        supply_run_id=f"{block_id}:reservation-import",
        supply_cycle_id=f"{block_id}:selected-slate-{slate_index:02d}",
    )
    hashes: dict[str, str] = {}
    calibrations: set[str] = set()
    for item in proposals:
        mode = str(item["search_mode"])
        previous = hashes.setdefault(mode, str(item["prompt_hash"]))
        if previous != item["prompt_hash"]:
            raise E1SelectionCohortError(
                "selected proposals disagree on a mode prompt hash"
            )
        calibrations.add(str(item["calibration_version"]))
    if len(calibrations) != 1:
        raise E1SelectionCohortError(
            "selected proposals disagree on calibration version"
        )
    window_id = f"window:{block_id}:selected-{slate_index:02d}"
    frozen = freeze_eligible_window(
        working_context,
        window_id=window_id,
        window_size=len(ids),
        shard_size=len(ids),
        seed=seed,
        current_bank_snapshot_id=bank_snapshot_id,
        current_prompt_hashes=hashes,
        calibration_version=next(iter(calibrations)),
        accepted_concept_hashes=accepted_hashes,
    )
    if set(frozen["proposal_ids"]) != set(ids):
        raise E1SelectionCohortError(
            "cross-slate reservation window changed selected IDs"
        )
    reservations = reserve_slate(
        working_context,
        window_id=window_id,
        owner_run_id=block_id,
        slate_id=f"slate:{block_id}:{slate_index:02d}",
        selections=[
            {
                "proposal_id": item["candidate_id"],
                "candidate_id": item["candidate_id"],
                "mechanism_summary": item["kernel"]["central_relation"],
            }
            for item in selected
        ],
        lease_seconds=lease_seconds,
    )
    return {"window": frozen, "reservations": reservations}


def _source_rows_from_v4(
    selector: V4ControlPipeline, outer_slate: int
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate_id,
            "lineage_id": candidate_id,
            "private_kernel_author_provider": "codex",
            "slate_index": outer_slate,
            "supply_cycle_index": record.get("supply_cycle_id"),
            "search_mode": record["search_mode"],
            "kernel": {
                key: deepcopy(record[key])
                for key in (
                    "candidate_id",
                    "search_mode",
                    "central_relation",
                    "why_coherent",
                    "novel_concept_potential",
                    "functional_composition_potential",
                    "why_multiple_training_examples_may_be_useful",
                    "possible_test_generalization",
                    "main_authoring_risk",
                )
                if key in record
            },
            "editorial_brief": deepcopy(record["editorial_brief"]),
            "selector_global_priority": int(record["development_priority"]),
            "selector_assessment": deepcopy(record.get("selector_assessment", {})),
        }
        for candidate_id, record in sorted(
            selector.state["records"].items(),
            key=lambda item: (int(item[1]["development_priority"]), item[0]),
        )
    ]


def _select_composite_slate(
    config: Mapping[str, Any],
    slate_dir: Path,
    slate_index: int,
    seed: int,
    working: Path,
) -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]
]:
    parent = deepcopy(config["_parent"])
    parent["seed"] = seed
    parent["run"] = {
        **deepcopy(parent["run"]),
        "id": f"{config['run']['id']}-selection-{slate_index:02d}",
        "mock": bool(config["run"]["mock"]),
        "automatically_commit_generated_tasks": False,
    }
    parent["_selection_reference_reservoir"] = str(working)
    execution = slate_dir / "execution"
    supply_runtime = _v46_runtime(
        parent,
        runs_root=execution,
        run_id="supply",
        reservoir=slate_dir / "supply_runtime_reservoir.json",
    )
    supply = generate_paired_supply(supply_runtime, parent)
    runtime = _v46_runtime(
        parent,
        runs_root=execution,
        run_id="hybrid",
        reservoir=slate_dir / "hybrid_reservoir.json",
    )
    supply_copy = runtime.run_dir / "paired_supply.private.json"
    if supply_copy.exists() and read_json(supply_copy) != supply:
        raise E1SelectionCohortError("paired supply changed during selector resume")
    if not supply_copy.exists():
        atomic_write_json(supply_copy, supply)
    window = build_arm_reservoir(
        config=parent,
        runtime=runtime,
        arm="hybrid",
        kernels=supply["arms"]["hybrid"],
        reservoir_path=slate_dir / "hybrid_reservoir.json",
    )
    before = {
        item["proposal_id"]: deepcopy(item)
        for item in load_reservoir(slate_dir / "hybrid_reservoir.json")["proposals"]
    }
    _write_immutable(
        slate_dir / "window_proposals.preselection.json",
        {"proposals": before},
        label="pre-selection proposal snapshot",
    )
    selected, selection = select_arm(
        runtime,
        arm="hybrid",
        kernels=supply["arms"]["hybrid"],
        window=window,
        reservoir_path=slate_dir / "hybrid_reservoir.json",
    )
    rows = []
    for row in selected:
        item = deepcopy(dict(row))
        item["supply_cycle_index"] = item["slate_index"]
        item["slate_index"] = slate_index
        rows.append(item)
    ledger = _merge_ledgers(
        ("supply", supply_runtime.state["call_ledger"]),
        ("selector", runtime.state["call_ledger"]),
    )
    attribution = {
        "actual_supply_calls": int(supply["actual_model_calls"]),
        "production_equivalent_supply_calls": 12,
        "paired_control_overhead_calls": int(supply["actual_model_calls"]) - 12,
        "paired_control_overhead_policy": "15 actual = 12 selectable + 3 paired Sol controls",
        "paired_control_overhead_slots": [
            {"cycle": int(item["cycle"]), "mode": str(item["mode"])}
            for item in supply["hybrid_schedule"]
            if item["provider"] == "claude"
        ],
        "opus_supply_calls": int(supply["claude_calls"]),
        "selector_calls": sum(
            item.get("role") == "batch_selector"
            for item in runtime.state["call_ledger"].values()
        ),
        "only_hybrid_window_visible_to_selector": True,
        "auxiliary_opus_routes_called": False,
    }
    return (
        rows,
        {"window": window, "selection": selection, "supply": supply},
        before,
        {
            "ledger": ledger,
            "attribution": attribution,
        },
    )


def _select_all_sol_slate(
    config: Mapping[str, Any],
    slate_dir: Path,
    slate_index: int,
    seed: int,
    working: Path,
) -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]
]:
    parent = deepcopy(config["_parent"])
    local = slate_dir / "all_sol_reservoir.json"
    if not local.exists():
        shutil.copy2(working, local)
    parent["seed"] = seed
    parent["paths"] = deepcopy(parent["paths"])
    parent["paths"]["runs"] = str(slate_dir / "execution")
    parent["paths"]["reservoir"] = str(local)
    parent["run"] = {
        **deepcopy(parent["run"]),
        "id": f"{config['run']['id']}-selector-{slate_index:02d}",
        "mock": bool(config["run"]["mock"]),
        "automatically_commit_generated_tasks": False,
    }
    parent["selection"] = deepcopy(parent["selection"])
    parent["selection"]["global_selector_max_selected"] = 8
    parent["selection"]["max_plans"] = 8
    parent["selection"]["max_realizations"] = 8
    active = sum(
        item["state"] in {"proposed", "in_flight"}
        for item in load_reservoir(local)["proposals"]
    )
    parent["reservoir"] = deepcopy(parent["reservoir"])
    parent["reservoir"]["hard_capacity"] = max(
        int(parent["reservoir"]["hard_capacity"]), active + 64
    )
    parent["reservoir"]["high_watermark"] = min(
        int(parent["reservoir"]["high_watermark"]),
        int(parent["reservoir"]["hard_capacity"]),
    )
    producer = V4ReservoirProducer(
        parent, supply_run_id=f"{config['run']['id']}-supply-{slate_index:02d}"
    )
    supply = producer.run(cycles=4, force_between_watermarks=True)
    selector = V4ControlPipeline(parent)
    selector.select_slate()
    rows = _source_rows_from_v4(selector, slate_index)
    before = {
        proposal_id: deepcopy(item)
        for proposal_id, item in selector.state["window_proposals"].items()
    }
    ledger = _merge_ledgers(
        ("supply", producer.state["call_ledger"]),
        ("selector", selector.state["call_ledger"]),
    )
    attribution = {
        "actual_supply_calls": sum(
            item.get("role") == "idea_search"
            for item in producer.state["call_ledger"].values()
        ),
        "production_equivalent_supply_calls": 12,
        "paired_control_overhead_calls": 0,
        "paired_control_overhead_policy": "12 actual = 12 selectable",
        "paired_control_overhead_slots": [],
        "opus_supply_calls": 0,
        "selector_calls": sum(
            item.get("role") == "batch_selector"
            for item in selector.state["call_ledger"].values()
        ),
        "only_frozen_48_kernel_window_visible_to_selector": True,
        "auxiliary_opus_routes_called": False,
    }
    return (
        rows,
        {
            "window": deepcopy(selector.state["frozen_window"]),
            "selection": deepcopy(selector.state["global_selector"]),
            "supply": supply,
            "selector_state_path": str(selector.state_path),
        },
        before,
        {"ledger": ledger, "attribution": attribution},
    )


def _freeze_supply_attestation(
    *,
    config: Mapping[str, Any],
    slate_dir: Path,
    selection_artifacts: Mapping[str, Any],
) -> tuple[Path | None, dict[str, Any] | None]:
    """Substantiate source composition without constraining selected sources."""

    if config["producer_kind"] == "all_sol":
        return None, None
    supply = selection_artifacts.get("supply")
    if not isinstance(supply, Mapping):
        raise E1SelectionCohortError("hybrid selection lacks its supply manifest")
    schedule = supply.get("hybrid_schedule")
    if not isinstance(schedule, list):
        raise E1SelectionCohortError("hybrid supply lacks its 12-call schedule")
    slots = {
        (int(item["cycle"]), str(item["mode"])): str(item["provider"])
        for item in schedule
        if isinstance(item, Mapping)
    }
    expected_slots = {
        (cycle, mode)
        for cycle in range(1, 5)
        for mode in ("open", "concept_led", "composition_led")
    }
    opus_modes = Counter(
        mode for (_, mode), provider in slots.items() if provider == "claude"
    )
    if (
        set(slots) != expected_slots
        or set(slots.values()) - {"codex", "claude"}
        or sum(provider == "claude" for provider in slots.values()) != 3
        or opus_modes != Counter(
            {"open": 1, "concept_led": 1, "composition_led": 1}
        )
        or int(supply.get("actual_model_calls", -1)) != 15
        or int(supply.get("sol_calls", -1)) != 12
        or int(supply.get("claude_calls", -1)) != 3
        or len(supply.get("arms", {}).get("hybrid", [])) != 48
    ):
        raise E1SelectionCohortError(
            "hybrid supply does not substantiate 12 selectable calls with 25% Opus "
            "and the explicit three-call paired-Sol overhead"
        )
    if config.get("_source_balancing_enabled", False) is not False:
        raise E1SelectionCohortError("source balancing unexpectedly entered selection")
    attestation = {
        "policy": "minority_opus_idea_supply",
        "selectable_idea_call_count": 12,
        "opus_selectable_idea_call_count": 3,
        "opus_fraction": 0.25,
        "source_balancing_enabled": False,
        "source_labels_hidden_from_selectors": True,
    }
    path = slate_dir / "minority_opus_supply_attestation.private.json"
    _write_immutable(
        path,
        {
            "schema_version": 1,
            "supply_attestation": attestation,
            "evidence": {
                "actual_paired_supply_calls": 15,
                "selectable_hybrid_supply_calls": 12,
                "paired_sol_control_overhead_calls": 3,
                "hybrid_kernel_count": 48,
                "full_paired_supply_sha256": _bundle_hash([deepcopy(dict(supply))]),
            },
        },
        label="minority Opus supply attestation",
    )
    return path, attestation


def _normalize_slate(
    *,
    config: Mapping[str, Any],
    slate_dir: Path,
    slate_index: int,
    source_rows: Sequence[Mapping[str, Any]],
    selection_artifacts: Mapping[str, Any],
    context_reservation: Mapping[str, Any],
    accounting_bundle: Mapping[str, Any],
    protected: Mapping[str, Any],
    dependency_manifest_path: Path,
    cohort_scale_assignment_path: Path,
    scale_ids: Sequence[str],
) -> dict[str, Any]:
    source_path = slate_dir / "selected_kernel_manifest.private.json"
    _write_immutable(
        source_path,
        {
            "schema_version": 1,
            "selection_completed": True,
            "selection_stopped_before_planning": True,
            "selected": list(source_rows),
            "selection_artifacts": deepcopy(dict(selection_artifacts)),
        },
        label="selector source manifest",
    )
    ledger_path = slate_dir / "pre_authoring_call_ledger.private.json"
    _write_immutable(
        ledger_path,
        {"call_ledger": deepcopy(accounting_bundle["ledger"])},
        label="pre-authoring call ledger",
    )
    accounting = v4_call_accounting(accounting_bundle["ledger"])
    accounting_path = slate_dir / "pre_authoring_accounting.json"
    _write_immutable(
        accounting_path,
        {
            **accounting,
            "attribution": deepcopy(accounting_bundle["attribution"]),
            "planning_calls": 0,
            "realization_calls": 0,
            "verification_calls": 0,
            "editorial_calls": 0,
        },
        label="pre-authoring accounting",
    )
    scale_ids = [str(item) for item in scale_ids]
    scale_path = slate_dir / "functional_scale_assignment.private.json"
    _write_immutable(
        scale_path,
        {
            "schema_version": 1,
            "selection_completed_before_assignment": True,
            "target_selected_fraction": float(
                config["functional_scale"]["target_selected_fraction"]
            ),
            "maximum_per_slate": 1,
            "assigned_candidate_ids": scale_ids,
            "assignment_did_not_affect_selection_rank": True,
            "assignment_scope": "entire_frozen_cohort",
            "planner_may_decline": True,
        },
        label="functional-scale assignment",
    )
    runner_kind = str(config["runner_kind"])
    supply_path, supply_attestation = _freeze_supply_attestation(
        config=config,
        slate_dir=slate_dir,
        selection_artifacts=selection_artifacts,
    )
    temporary = slate_dir / ".normalized-selected-slate.tmp.json"
    normalized = normalize_selected_slate(
        source_path,
        temporary,
        runner_kind=runner_kind,
        accounting=accounting_path,
        functional_scale_candidate_ids=scale_ids,
        supply_manifest=supply_path,
        supply_attestation=supply_attestation,
        cohort_scale_assignment=cohort_scale_assignment_path,
    )
    temporary.unlink(missing_ok=True)
    reservation_path = slate_dir / "cross_slate_reservation.json"
    _write_immutable(
        reservation_path,
        deepcopy(dict(context_reservation)),
        label="cross-slate reservation",
    )
    context_before = slate_dir / "selection_context.before.json"
    input_snapshot = slate_dir / "selection_input_snapshot.json"
    _write_immutable(
        input_snapshot,
        {
            "schema_version": 1,
            "accepted_bank_snapshot_id": context_reservation["window"][
                "accepted_bank_snapshot_id"
            ],
            "committed_registry": deepcopy(protected["committed_task_registry"]),
            "production_reservoir": deepcopy(protected["reservoir"]),
            "selection_context_before": {
                "path": str(context_before.resolve()),
                "sha256": _sha(context_before),
            },
            "reservation_snapshot_id": context_reservation["window"][
                "reservation_snapshot_id"
            ],
            "rejection_memory_snapshot_id": context_reservation["window"][
                "rejection_memory_snapshot_id"
            ],
        },
        label="selection input snapshot",
    )
    extra = [
        ("pre-authoring-call-ledger", ledger_path),
        ("functional-scale-assignment", scale_path),
        ("cross-slate-reservation", reservation_path),
        ("selection-context-before", context_before),
        ("selection-input-snapshot", input_snapshot),
        ("cohort-source-config", Path(config["_source_config_path"])),
        ("cohort-dependency-manifest", dependency_manifest_path),
        ("cohort-functional-scale-assignment", cohort_scale_assignment_path),
    ]
    existing_ids = {item["source_id"] for item in normalized["source_artifacts"]}
    for source_id, path in extra:
        if source_id not in existing_ids:
            normalized["source_artifacts"].append(
                {
                    "source_id": source_id,
                    "path": str(path.resolve()),
                    "sha256": _sha(path),
                }
            )
    normalized["source_hashes"] = {
        item["source_id"]: item["sha256"] for item in normalized["source_artifacts"]
    }
    normalized["cohort_provenance"] = {
        "pipeline": PIPELINE,
        "version": VERSION,
        "cohort_run_id": config["run"]["id"],
        "slate_index": slate_index,
        "producer_kind": config["producer_kind"],
        "runner_kind_frozen_cohort_wide": runner_kind,
        "selection_only": True,
        "stopped_before_planning": True,
        "bank_registry_sha256": protected["committed_task_registry"]["sha256"],
        "production_reservoir_sha256": protected["reservoir"]["sha256"],
        "functional_scale_assigned_after_selection": True,
        "functional_scale_assignment_scope": "entire_frozen_cohort",
        "functional_scale_planner_may_decline": True,
        "selected_provider_quota_applied": False,
        "actual_pre_authoring_attribution": deepcopy(accounting_bundle["attribution"]),
    }
    destination = slate_dir / "selected_slate.frozen.json"
    _write_immutable(destination, normalized, label="E1 selected slate")
    _validate_selected_slate(destination, kind=runner_kind)
    return {
        "slate_index": slate_index,
        "runner_kind": runner_kind,
        "selected_count": len(source_rows),
        "provider_counts": dict(
            sorted(
                Counter(
                    row["private_kernel_author_provider"] for row in source_rows
                ).items()
            )
        ),
        "functional_scale_candidate_ids": scale_ids,
        "selected_candidate_ids": [str(item["candidate_id"]) for item in source_rows],
        "selected_slate_manifest": {
            "path": str(destination),
            "sha256": _sha(destination),
        },
        "selector_source_manifest": {
            "path": str(source_path),
            "sha256": _sha(source_path),
        },
        "pre_authoring_accounting": deepcopy(accounting),
        "attribution": deepcopy(accounting_bundle["attribution"]),
        "cross_slate_reservation": {
            "path": str(reservation_path),
            "sha256": _sha(reservation_path),
        },
        "eligible_window_proposal_ids": list(
            selection_artifacts["window"]["proposal_ids"]
        ),
    }


def _selection_checkpoint(
    *,
    slate_index: int,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Any],
    accounting: Mapping[str, Any],
    reservation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "slate_index": slate_index,
        "seed": seed,
        "source_rows": deepcopy(list(rows)),
        "selection_artifacts": deepcopy(dict(artifacts)),
        "accounting_bundle": deepcopy(dict(accounting)),
        "context_reservation": deepcopy(dict(reservation)),
        "selection_complete": True,
        "planning_started": False,
    }


def _heartbeat_run_local(working: Path, run_id: str) -> dict[str, Any]:
    value = heartbeat_leases(working, owner_run_id=run_id)
    owned = [
        item
        for item in value["active_reservations"]
        if item["owner_run_id"] == run_id
    ]
    return {
        "owned_active_leases": len(owned),
        "heartbeat_epoch_seconds": max(
            (float(item["heartbeat_at_epoch_seconds"]) for item in owned),
            default=None,
        ),
    }


def _resolve_run_local_leases(
    *, working: Path, run_id: str, output: Path
) -> dict[str, Any]:
    """Close internal leases without semantically spending selected concepts.

    The immutable selected-slate manifests are the handoff to E1 live.  The
    run-local reservoir has served its sole purpose once every cohort window
    is frozen, so its leases are resolved as recoverable work rather than as a
    semantic rejection or an accepted-bank mutation.
    """

    path = output / "lease_terminal_resolution.json"
    before = load_reservoir(working)
    if path.exists():
        existing = read_json(path)
        if any(
            item["owner_run_id"] == run_id
            for item in before["active_reservations"]
        ):
            raise E1SelectionCohortError(
                "terminal lease ledger exists but owned leases remain active"
            )
        return {"path": str(path.resolve()), "sha256": _sha(path), **existing}
    active_ids = sorted(
        item["proposal_id"]
        for item in before["active_reservations"]
        if item["owner_run_id"] == run_id
    )
    previously_resolved_ids = {
        item["proposal_id"]
        for item in before["reservation_history"]
        if item["owner_run_id"] == run_id
        and item.get("resolution_note", "").startswith(
            "selection-only cohort handoff complete"
        )
    }
    owned_ids = sorted(set(active_ids) | previously_resolved_ids)
    for proposal_id in active_ids:
        resolve_proposal(
            working,
            proposal_id=proposal_id,
            outcome="rejected",
            reason=(
                "selection-only cohort handoff complete; no semantic disposition; "
                "recoverable by the downstream E1 run"
            ),
            source_run_id=run_id,
            memory_tier="recoverable_work",
        )
    after = load_reservoir(working)
    remaining = [
        item["proposal_id"]
        for item in after["active_reservations"]
        if item["owner_run_id"] == run_id
    ]
    if remaining:
        raise E1SelectionCohortError("run-local reservations failed terminal resolution")
    value = {
        "schema_version": 1,
        "owner_run_id": run_id,
        "resolution": "recoverable_downstream_handoff",
        "semantic_rejection": False,
        "production_state_mutated": False,
        "resolved_proposal_ids": owned_ids,
        "remaining_owned_active_leases": [],
        "reservation_history_count": sum(
            item["proposal_id"] in set(owned_ids)
            for item in after["reservation_history"]
        ),
    }
    _write_immutable(path, value, label="run-local lease terminal resolution")
    return {"path": str(path.resolve()), "sha256": _sha(path), **value}


def _validate_completed_report(report: Mapping[str, Any]) -> None:
    if report.get("pipeline") != PIPELINE or report.get("completed") is not True:
        raise E1SelectionCohortError("existing cohort manifest is incomplete")
    protected = report.get("protected_production_inputs")
    if not isinstance(protected, Mapping):
        raise E1SelectionCohortError("completed report lacks protected inputs")
    _assert_protected_unchanged(protected)
    dependency = report.get("dependency_manifest")
    if (
        not isinstance(dependency, Mapping)
        or not Path(str(dependency.get("path", ""))).is_file()
        or _sha(str(dependency["path"])) != dependency.get("sha256")
    ):
        raise E1SelectionCohortError("completed dependency manifest changed")
    _assert_dependency_manifest_unchanged(read_json(Path(str(dependency["path"]))))
    kind = str(report.get("runner_kind"))
    for row in report.get("slates", []):
        artifact = row.get("selected_slate_manifest", {})
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or _sha(path) != artifact.get("sha256"):
            raise E1SelectionCohortError("completed selected-slate artifact changed")
        _validate_selected_slate(path, kind=kind)
    working = report.get("working_selection_context", {})
    if (
        not isinstance(working, Mapping)
        or not Path(str(working.get("path", ""))).is_file()
        or _sha(str(working["path"])) != working.get("sha256")
    ):
        raise E1SelectionCohortError("completed working selection context changed")
    terminal = report.get("lease_terminal_resolution", {})
    if (
        not isinstance(terminal, Mapping)
        or not Path(str(terminal.get("path", ""))).is_file()
        or _sha(str(terminal["path"])) != terminal.get("sha256")
    ):
        raise E1SelectionCohortError("lease terminal-resolution artifact changed")


def run_selection_cohort(
    config: Mapping[str, Any], *, authorize_model_calls: bool
) -> dict[str, Any]:
    if not authorize_model_calls and not config["run"]["mock"]:
        raise E1SelectionCohortError(
            "real selection cohort requires --authorize-model-calls"
        )
    output = Path(config["_output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "selected_cohort_manifest.json"
    state_path = output / "state.json"
    if report_path.exists():
        report = read_json(report_path)
        _validate_completed_report(report)
        if not state_path.is_file():
            raise E1SelectionCohortError("completed report has no state ledger")
        state = read_json(state_path)
        report_hash = _sha(report_path)
        if state.get("completed") is True:
            if state.get("selected_cohort_manifest_sha256") != report_hash:
                raise E1SelectionCohortError("completed state/report hashes disagree")
        else:
            # A crash between the atomic report write and the state commit is
            # recoverable only after revalidating every referenced artifact.
            state.update(
                {
                    "completed": True,
                    "stage": "complete_before_planning",
                    "completed_at_epoch_seconds": round(time.time(), 3),
                    "selected_cohort_manifest_sha256": report_hash,
                    "report_state_reconciled_after_crash": True,
                }
            )
            atomic_write_json(state_path, state)
        return report

    snapshot = {
        key: deepcopy(value) for key, value in config.items() if not key.startswith("_")
    }
    _write_immutable(
        output / "config.snapshot.json", snapshot, label="cohort config snapshot"
    )
    protected = _protected_state(config)
    dependency_path = output / "frozen_inputs" / "dependency_manifest.json"
    dependency_path.parent.mkdir(parents=True, exist_ok=True)
    if dependency_path.exists():
        dependencies = read_json(dependency_path)
    else:
        dependencies = _dependency_manifest(config)
        _write_immutable(
            dependency_path, dependencies, label="cohort dependency manifest"
        )
    _assert_dependency_manifest_unchanged(dependencies)
    state = (
        read_json(state_path)
        if state_path.exists()
        else {
            "schema_version": 2,
            "pipeline": PIPELINE,
            "version": VERSION,
            "run_id": config["run"]["id"],
            "runner_kind": config["runner_kind"],
            "completed": False,
            "started_at_epoch_seconds": round(time.time(), 3),
            "protected_production_inputs": protected,
            "dependency_manifest_sha256": _sha(dependency_path),
            "selected_slates": [],
            "normalized_slates": [],
            "slates": {},
            "stage": "initialized",
        }
    )
    if state.get("schema_version") != 2:
        raise E1SelectionCohortError("selection cohort state schema is incompatible")
    if state.get("runner_kind") != config["runner_kind"]:
        raise E1SelectionCohortError("cohort runner_kind changed during resume")
    if state.get("completed") is True:
        raise E1SelectionCohortError("state is complete but final report is missing")
    if state.get("protected_production_inputs") != protected:
        raise E1SelectionCohortError("production input snapshot changed during resume")
    if state.get("dependency_manifest_sha256") != _sha(dependency_path):
        raise E1SelectionCohortError("dependency manifest changed during resume")
    _assert_protected_unchanged(protected)
    atomic_write_json(state_path, state)

    production_reservoir = Path(protected["reservoir"]["path"])
    pristine = output / "frozen_inputs" / "production_reservoir.start.json"
    if not pristine.exists():
        pristine.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(production_reservoir, pristine)
    if _sha(pristine) != protected["reservoir"]["sha256"]:
        raise E1SelectionCohortError("frozen production reservoir snapshot changed")
    working = output / "state" / "selection_context.working.json"
    working.parent.mkdir(parents=True, exist_ok=True)
    _copy_context(pristine, working)
    before_recovery = load_reservoir(working)
    before_active = len(before_recovery["active_reservations"])
    recovered = recover_expired_leases(
        working,
        active_owner_run_ids={str(config["run"]["id"])},
    )
    # Expired foreign work is context, not fresh supply.  Recovery changes it
    # back to ``proposed`` in the reservoir API, so immediately reapply the
    # context-only filter before any eligible window can see it as selectable.
    recovered_context = _context_only_reservoir(recovered)
    atomic_write_json(working, recovered_context)
    state["lease_recovery"] = {
        "expired_foreign_leases_recovered": max(
            0, before_active - len(recovered_context["active_reservations"])
        ),
        "active_owner_run_ids": [str(config["run"]["id"])],
        "production_reservoir_unchanged": True,
    }
    state["last_lease_heartbeat"] = _heartbeat_run_local(
        working, str(config["run"]["id"])
    )
    atomic_write_json(state_path, state)

    # One DataContext freezes the current accepted bank identity used for every
    # window.  Hash guards prevent later slates from silently observing an
    # appended registry.
    context = _DataContext(config["_parent"])
    selected_slates: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    seen_window_proposals: set[str] = set()
    count = int(config["cohort"]["slate_count"])
    for slate_index in range(1, count + 1):
        _assert_protected_unchanged(protected)
        _assert_dependency_manifest_unchanged(dependencies)
        state["last_lease_heartbeat"] = _heartbeat_run_local(
            working, str(config["run"]["id"])
        )
        slate_dir = output / "selection_slates" / f"slate-{slate_index:02d}"
        slate_dir.mkdir(parents=True, exist_ok=True)
        seed = int(config["seed"]) + slate_index * int(config["cohort"]["seed_stride"])
        checkpoint_path = slate_dir / "selection_checkpoint.private.json"
        if checkpoint_path.exists():
            checkpoint = read_json(checkpoint_path)
            expected = state.get("slates", {}).get(str(slate_index), {}).get(
                "selection_checkpoint_sha256"
            )
            if expected is not None and expected != _sha(checkpoint_path):
                raise E1SelectionCohortError(
                    f"selection checkpoint {slate_index} changed during resume"
                )
            if expected is None:
                state.setdefault("slates", {}).setdefault(str(slate_index), {})[
                    "selection_checkpoint_sha256"
                ] = _sha(checkpoint_path)
                state["selected_slates"] = sorted(
                    set(state.get("selected_slates", [])) | {slate_index}
                )
                state["selection_checkpoint_reconciled_after_crash"] = True
                atomic_write_json(state_path, state)
        else:
            before_path = slate_dir / "selection_context.before.json"
            if not before_path.exists():
                shutil.copy2(working, before_path)
            if config["producer_kind"] == "minority_opus_supply":
                rows, artifacts, proposals, accounting = _select_composite_slate(
                    config, slate_dir, slate_index, seed, working
                )
            else:
                rows, artifacts, proposals, accounting = _select_all_sol_slate(
                    config, slate_dir, slate_index, seed, working
                )
            if not rows:
                raise E1SelectionCohortError(
                    f"selector returned an empty development slate {slate_index}"
                )
            reservation_artifact = slate_dir / "cross_slate_reservation.json"
            reservation = (
                read_json(reservation_artifact)
                if reservation_artifact.exists()
                else _freeze_selected_context(
                    working_context=working,
                    proposal_rows=proposals,
                    selected=rows,
                    block_id=str(config["run"]["id"]),
                    slate_index=slate_index,
                    seed=seed,
                    bank_snapshot_id=context.bank_snapshot_id,
                    accepted_hashes=tuple(context.accepted_concept_hashes()),
                    lease_seconds=int(config["_parent"]["reservoir"]["lease_seconds"]),
                )
            )
            checkpoint = _selection_checkpoint(
                slate_index=slate_index,
                seed=seed,
                rows=rows,
                artifacts=artifacts,
                accounting=accounting,
                reservation=reservation,
            )
            _write_immutable(
                checkpoint_path, checkpoint, label="selection phase checkpoint"
            )
            state.setdefault("slates", {})[str(slate_index)] = {
                "selection_checkpoint_sha256": _sha(checkpoint_path)
            }
            state["selected_slates"] = list(range(1, slate_index + 1))
            state["stage"] = f"selected_slate_{slate_index:02d}_before_planning"
            state["last_lease_heartbeat"] = _heartbeat_run_local(
                working, str(config["run"]["id"])
            )
            atomic_write_json(state_path, state)
        if checkpoint.get("selection_complete") is not True:
            raise E1SelectionCohortError("selection checkpoint is incomplete")
        rows = checkpoint["source_rows"]
        artifacts = checkpoint["selection_artifacts"]
        window_ids = set(artifacts["window"]["proposal_ids"])
        overlap = window_ids & seen_window_proposals
        if overlap:
            raise E1SelectionCohortError(
                f"eligible windows are not disjoint: {sorted(overlap)[:3]}"
            )
        seen_window_proposals |= window_ids
        candidate_ids = {str(item["candidate_id"]) for item in rows}
        if candidate_ids & seen_candidates:
            raise E1SelectionCohortError(
                "selected candidate identity repeats across slates"
            )
        seen_candidates |= candidate_ids
        selected_slates.append(checkpoint)

    scale_assignments = _cohort_scale_assignments(
        selected_slates,
        enabled=bool(config["functional_scale"]["enabled"]),
        fraction=float(config["functional_scale"]["target_selected_fraction"]),
        maximum_per_slate=int(config["functional_scale"]["maximum_per_slate"]),
        seed=int(config["seed"]),
    )
    cohort_scale_path = output / "functional_scale_assignment.cohort.private.json"
    _write_immutable(
        cohort_scale_path,
        {
            "schema_version": 1,
            "runner_kind": config["runner_kind"],
            "enabled": bool(config["functional_scale"]["enabled"]),
            "assignment_scope": "entire_frozen_cohort",
            "selected_candidate_count": len(seen_candidates),
            "target_selected_fraction": float(
                config["functional_scale"]["target_selected_fraction"]
            ),
            "maximum_per_slate": 1,
            "assigned_by_slate": {
                str(key): value for key, value in sorted(scale_assignments.items())
            },
            "assigned_count": sum(len(value) for value in scale_assignments.values()),
            "assignment_did_not_affect_selection_rank": True,
            "planner_may_decline": True,
        },
        label="cohort functional-scale assignment",
    )

    slates: list[dict[str, Any]] = []
    for checkpoint in selected_slates:
        slate_index = int(checkpoint["slate_index"])
        slate_dir = output / "selection_slates" / f"slate-{slate_index:02d}"
        summary_path = slate_dir / "slate_summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            expected = state.get("slates", {}).get(str(slate_index), {}).get(
                "summary_sha256"
            )
            if expected is not None and expected != _sha(summary_path):
                raise E1SelectionCohortError(
                    f"normalized slate {slate_index} changed during resume"
                )
            _validate_selected_slate(
                Path(summary["selected_slate_manifest"]["path"]),
                kind=str(config["runner_kind"]),
            )
            if expected is None:
                state.setdefault("slates", {}).setdefault(str(slate_index), {})[
                    "summary_sha256"
                ] = _sha(summary_path)
                state["normalized_slates"] = sorted(
                    set(state.get("normalized_slates", [])) | {slate_index}
                )
                state["summary_reconciled_after_crash"] = True
                atomic_write_json(state_path, state)
        else:
            summary = _normalize_slate(
                config=config,
                slate_dir=slate_dir,
                slate_index=slate_index,
                source_rows=checkpoint["source_rows"],
                selection_artifacts=checkpoint["selection_artifacts"],
                context_reservation=checkpoint["context_reservation"],
                accounting_bundle=checkpoint["accounting_bundle"],
                protected=protected,
                dependency_manifest_path=dependency_path,
                cohort_scale_assignment_path=cohort_scale_path,
                scale_ids=scale_assignments[slate_index],
            )
            _write_immutable(
                summary_path, summary, label="selection slate resume summary"
            )
            state.setdefault("slates", {}).setdefault(str(slate_index), {})[
                "summary_sha256"
            ] = _sha(summary_path)
            state["normalized_slates"] = list(range(1, slate_index + 1))
            state["stage"] = f"normalized_slate_{slate_index:02d}_before_planning"
            state["last_lease_heartbeat"] = _heartbeat_run_local(
                working, str(config["run"]["id"])
            )
            atomic_write_json(state_path, state)
        slates.append(summary)

    _assert_protected_unchanged(protected)
    _assert_dependency_manifest_unchanged(dependencies)
    terminal_resolution = _resolve_run_local_leases(
        working=working, run_id=str(config["run"]["id"]), output=output
    )
    aggregate_calls = sum(
        row["pre_authoring_accounting"]["substantive_logical_calls"] for row in slates
    )
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "completed": True,
        "selection_only": True,
        "stopped_before_planning": True,
        "producer_kind": config["producer_kind"],
        "runner_kind": config["runner_kind"],
        "protected_production_inputs": protected,
        "dependency_manifest": {
            "path": str(dependency_path.resolve()),
            "sha256": _sha(dependency_path),
            "bundle_sha256": dependencies["dependency_bundle_sha256"],
        },
        "accepted_bank_snapshot_id": context.bank_snapshot_id,
        "working_selection_context": {"path": str(working), "sha256": _sha(working)},
        "lease_terminal_resolution": terminal_resolution,
        "funnel": {
            "slates": len(slates),
            "eligible_proposals": len(seen_window_proposals),
            "selected_candidates": len(seen_candidates),
            "functional_scale_invitations": sum(
                len(row["functional_scale_candidate_ids"]) for row in slates
            ),
            "planning_calls": 0,
            "realization_calls": 0,
            "verification_calls": 0,
            "editorial_calls": 0,
        },
        "accounting": {
            "actual_pre_authoring_substantive_calls": aggregate_calls,
            "paired_control_overhead_calls": sum(
                row["attribution"]["paired_control_overhead_calls"] for row in slates
            ),
            "production_equivalent_supply_calls": sum(
                row["attribution"]["production_equivalent_supply_calls"]
                for row in slates
            ),
            "opus_supply_calls": sum(
                row["attribution"]["opus_supply_calls"] for row in slates
            ),
            "selector_calls": sum(
                row["attribution"]["selector_calls"] for row in slates
            ),
        },
        "disjoint_window_check": {
            "passed": True,
            "unique_proposal_ids": len(seen_window_proposals),
        },
        "cross_slate_reservations_visible": True,
        "run_local_reservations_heartbeat_and_recovery_enabled": True,
        "run_local_reservations_resolved_after_immutable_handoff": True,
        "source_labels_hidden_from_selectors": True,
        "selected_source_quota_applied": False,
        "functional_scale_assigned_only_after_selection": True,
        "functional_scale_assignment_scope": "entire_frozen_cohort",
        "functional_scale_planner_may_decline": True,
        "auxiliary_opus_routes_enabled": False,
        "automatic_task_commit": False,
        "production_state_mutated": False,
        "slates": slates,
    }
    _write_immutable(report_path, report, label="selected cohort manifest")
    _validate_completed_report(report)
    state.update(
        {
            "completed": True,
            "stage": "complete_before_planning",
            "completed_at_epoch_seconds": round(time.time(), 3),
            "selected_cohort_manifest_sha256": _sha(report_path),
        }
    )
    atomic_write_json(state_path, state)
    return report


def validate_selection_cohort(config: Mapping[str, Any]) -> dict[str, Any]:
    _assert_protected_unchanged(_protected_state(config))
    kind = str(config["producer_kind"])
    return {
        "pipeline": PIPELINE,
        "version": VERSION,
        "producer_kind": kind,
        "runner_kind": str(config["runner_kind"]),
        "slate_count": int(config["cohort"]["slate_count"]),
        "maximum_slate_count": MAXIMUM_COHORT_SLATES,
        "eligible_window_size": 48,
        "maximum_selected_per_slate": 8,
        "source_labels_hidden_from_selectors": True,
        "functional_scale_assignment_stage": "after_selection",
        "functional_scale_assignment_scope": "entire_frozen_cohort",
        "functional_scale_planner_may_decline": True,
        "functional_scale_target_fraction": float(
            config["functional_scale"]["target_selected_fraction"]
        ),
        "selection_only": True,
        "stops_before_planning": True,
        "all_sol_fallback_makes_opus_calls": False,
        "hybrid_auxiliary_opus_routes_enabled": False,
        "hybrid_selected_source_quota": None,
        "hybrid_selectable_supply_calls": 12 if kind == "minority_opus_supply" else 0,
        "hybrid_paired_sol_overhead_calls": 3 if kind == "minority_opus_supply" else 0,
        "production_mutation_allowed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E1 selection-only cohort producer")
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--authorize-model-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_cohort_config(args.config)
        value = (
            validate_selection_cohort(config)
            if args.command == "validate"
            else run_selection_cohort(
                config, authorize_model_calls=args.authorize_model_calls
            )
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except (ConfigError, E1SelectionCohortError) as exc:
        print(f"E1 selection cohort error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAXIMUM_COHORT_SLATES",
    "PIPELINE",
    "PRODUCER_KINDS",
    "VERSION",
    "E1SelectionCohortError",
    "load_cohort_config",
    "run_selection_cohort",
    "validate_selection_cohort",
]
