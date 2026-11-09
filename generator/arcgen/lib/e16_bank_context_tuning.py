"""E16 accepted-bank context-size, sampling, prompt, and effort experiment.

The module is intentionally additive.  It freezes the current committed bank,
constructs reproducible bank subsets, generates kernels with the three frozen
V4 search modes, and exposes them to source-blind unchanged V4 selectors.  It
has no commit, promotion, or production-reservoir mutation API.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from random import Random
from typing import Any

from jsonschema import Draft202012Validator

from .committed_tasks import load_committed_task_registry
from .embeddings import LocalHashingTfidfProvider
from .model import ModelInvoker
from .schemas import SchemaError, parse_json_object
from .util import (
    PLACEHOLDER_RE,
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    read_json,
    render_prompt,
)
from .v4_config import load_v4_config
from .v4_reservoir import load_reservoir, make_proposal, validate_reservoir

PIPELINE = "arcgen-e16-bank-context-tuning"
VERSION = "1.0"
MODEL = "gpt-5.6-sol"
MODES = ("open", "concept_led", "composition_led")
POLICIES = (
    "task_uniform",
    "known_lineage_uniform",
    "stratified",
    "full_raw",
    "full_known_lineage",
)


class E16Error(RuntimeError):
    """Raised when the experiment loses its frozen design or isolation."""


TEXT = {"type": "string", "minLength": 1}
IDEA = {
    "type": "object",
    "properties": {
        "candidate_id": TEXT,
        "search_mode": {"type": "string", "enum": list(MODES)},
        "central_relation": TEXT,
        "why_coherent": TEXT,
        "novel_concept_potential": TEXT,
        "functional_composition_potential": TEXT,
        "why_multiple_training_examples_may_be_useful": TEXT,
        "possible_test_generalization": TEXT,
        "main_authoring_risk": TEXT,
    },
    "required": [
        "candidate_id",
        "search_mode",
        "central_relation",
        "why_coherent",
        "novel_concept_potential",
        "functional_composition_potential",
        "why_multiple_training_examples_may_be_useful",
        "possible_test_generalization",
        "main_authoring_risk",
    ],
    "additionalProperties": False,
}
BRIEF = {
    "type": "object",
    "properties": {"preserve": TEXT, "develop": TEXT, "avoid": TEXT},
    "required": ["preserve", "develop", "avoid"],
    "additionalProperties": False,
}
IDEA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "search_call_id": TEXT,
        "search_mode": {"type": "string", "enum": list(MODES)},
        "ideas": {"type": "array", "minItems": 4, "maxItems": 4, "items": IDEA},
        "recommended_candidate_id": TEXT,
        "recommended_development_brief": BRIEF,
    },
    "required": [
        "search_call_id",
        "search_mode",
        "ideas",
        "recommended_candidate_id",
        "recommended_development_brief",
    ],
    "additionalProperties": False,
}


def parse_e16_stage(
    stage: str, value: str | bytes | bytearray | Mapping[str, Any]
) -> dict[str, Any]:
    if stage != "e16_idea_search":
        raise SchemaError("$.stage", f"unsupported E16 stage {stage!r}")
    parsed = (
        parse_json_object(value)
        if isinstance(value, (str, bytes, bytearray))
        else dict(value)
    )
    errors = sorted(
        Draft202012Validator(IDEA_RESPONSE_SCHEMA).iter_errors(parsed),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        raise SchemaError(path, error.message)
    ids = [str(item["candidate_id"]) for item in parsed["ideas"]]
    if len(ids) != len(set(ids)):
        raise SchemaError("$.ideas", "candidate IDs must be unique")
    if any(item["search_mode"] != parsed["search_mode"] for item in parsed["ideas"]):
        raise SchemaError("$.ideas", "idea search modes must match the response")
    if parsed["recommended_candidate_id"] not in ids:
        raise SchemaError("$.recommended_candidate_id", "must name a returned idea")
    return parsed


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E16Error(f"expected mapping: {path}")
    return value


def _resolve(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise E16Error(f"{field} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _verified_ref(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, Mapping):
        raise E16Error(f"{field} must freeze path and sha256")
    path = _resolve(root, value.get("path"), f"{field}.path")
    if not path.is_file() or value.get("sha256") != _sha(path):
        raise E16Error(f"{field} changed or is missing: {path}")
    return path


def _immutable(path: Path, value: Any, label: str) -> None:
    if path.is_file():
        if read_json(path) != value:
            raise E16Error(f"immutable {label} changed: {path}")
        return
    atomic_write_json(path, value)


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read_mapping(source)
    if raw.get("pipeline") != PIPELINE or str(raw.get("version")) != VERSION:
        raise E16Error(f"config must target {PIPELINE} {VERSION}")
    root = source.parent.parent if source.parent.name == "config" else source.parent
    raw["_pre_e15_workflow_path"] = str(
        _verified_ref(root, raw.get("pre_e15_workflow"), "pre_e15_workflow")
    )
    raw["_base_v4_control_path"] = str(
        _verified_ref(root, raw.get("base_v4_control"), "base_v4_control")
    )
    registry_spec = raw.get("committed_registry")
    if not isinstance(registry_spec, Mapping):
        raise E16Error("committed_registry must be a mapping")
    registry = _resolve(root, registry_spec.get("path"), "committed_registry.path")
    verified = load_committed_task_registry(registry)
    expected = registry_spec.get("expected_count")
    if type(expected) is not int or len(verified["entries"]) != expected:
        raise E16Error(
            f"committed registry contains {len(verified['entries'])}; expected {expected}"
        )
    output = _resolve(root, raw.get("output_directory"), "output_directory")
    calibration = _resolve(
        root, raw.get("calibration_source_run"), "calibration_source_run"
    )
    if not calibration.is_dir():
        raise E16Error("calibration source run does not exist")
    screen = raw.get("screen")
    run = raw.get("run")
    if not isinstance(screen, Mapping) or not isinstance(run, Mapping):
        raise E16Error("screen and run sections are required")
    if screen.get("modes") != list(MODES):
        raise E16Error("the three frozen V4 search modes are required")
    if screen.get("model") != MODEL or screen.get("effort") != "high":
        raise E16Error("Phase A must use Sol/high")
    if screen.get("expected_calls") != 48 or screen.get("expected_kernels") != 192:
        raise E16Error("Phase A must remain 48 calls / 192 kernels")
    if output.name != run.get("id"):
        raise E16Error("output directory basename must match run.id")
    for field in (
        "automatically_commit_generated_tasks",
        "automatic_task_commit",
        "production_state_mutation_allowed",
        "production_reservoir_mutation_allowed",
        "automatic_promotion",
    ):
        if run.get(field) is not False:
            raise E16Error(f"run.{field} must be false")
    if run.get("mock") not in {True, False}:
        raise E16Error("run.mock must be boolean")
    raw["_repository_root"] = str(root)
    raw["_config_path"] = str(source)
    raw["_registry_path"] = str(registry)
    raw["_registry_sha256"] = _sha(registry)
    raw["_registry_count"] = len(verified["entries"])
    raw["_output_directory"] = str(output)
    raw["_calibration_source_run"] = str(calibration)
    return raw


def _normalise_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _dossier_review(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    dossier_manifest = root / "committed_tasks" / str(entry["dossier_path"])
    dossier_dir = dossier_manifest.parent
    path = dossier_dir / "review.json"
    if not path.is_file():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _grid_profile(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    task = read_json(root / "committed_tasks" / str(entry["task_path"]))
    input_dims: list[tuple[int, int]] = []
    output_dims: list[tuple[int, int]] = []
    for pair in task.get("train", []) + task.get("test", []):
        grid = pair["input"]
        input_dims.append((len(grid), len(grid[0])))
        output = pair.get("output")
        if output:
            output_dims.append((len(output), len(output[0])))
    maximum = max((max(*dims) for dims in input_dims + output_dims), default=0)
    bucket = "compact" if maximum <= 10 else "medium" if maximum <= 20 else "large"
    input_area = max((h * w for h, w in input_dims), default=0)
    output_area = max((h * w for h, w in output_dims), default=0)
    relation = (
        "input_small_output_large"
        if input_area <= 100 and output_area > 225
        else "input_large_output_small"
        if input_area > 225 and output_area <= 100
        else "same_scale"
    )
    return {
        "maximum_dimension": maximum,
        "size_bucket": bucket,
        "io_relation": relation,
    }


def _known_lineage_key(entry: Mapping[str, Any], review: Mapping[str, Any]) -> str:
    persisted = review.get("lineage_id") or review.get("source_lineage_id")
    if isinstance(persisted, str) and persisted:
        return "persisted:" + persisted
    summary = _normalise_text(str(entry["concept_summary"]))
    rule = _normalise_text(str(entry["intended_rule"]))
    # Exact normalized summaries are the strongest old-registry sibling signal;
    # exact normalized rules catch title/summary rewrites of the same episode.
    if entry.get("user_authorized_same_procedure_override") is True:
        return "known-sibling:" + content_hash({"summary": summary})
    return "exact-procedure:" + content_hash({"summary": summary, "rule": rule})


def freeze_bank(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(config["_output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "bank_snapshot" / "manifest.json"
    if manifest_path.is_file():
        value = read_json(manifest_path)
        if value.get("registry_sha256") != config["_registry_sha256"]:
            raise E16Error("frozen bank manifest no longer matches the config registry")
        rows_path = Path(value["rows_path"])
        if not rows_path.is_file() or _sha(rows_path) != value["rows_sha256"]:
            raise E16Error("frozen bank rows changed")
        return value
    registry = load_committed_task_registry(config["_registry_path"])
    root = Path(config["_repository_root"])
    rows = []
    for entry in registry["entries"]:
        review = _dossier_review(root, entry)
        profile = _grid_profile(root, entry)
        opaque = (
            "a"
            + content_hash(
                {"e16": "accepted", "registry_task_id": entry["registry_task_id"]}
            )[:13]
        )
        rows.append(
            {
                "accepted_concept_id": opaque,
                "registry_task_id": entry["registry_task_id"],
                "title": entry["title"],
                "concept_summary": entry["concept_summary"],
                "known_lineage_key": _known_lineage_key(entry, review),
                "search_mode": entry["search_mode"],
                "intrinsic_status": entry["intrinsic_status"],
                "source_pipeline": entry["source_pipeline"],
                "commit_set_id": entry["commit_set_id"],
                **profile,
            }
        )
    rows = sorted(rows, key=lambda item: item["accepted_concept_id"])
    bank_dir = output / "bank_snapshot"
    bank_dir.mkdir(parents=True, exist_ok=True)
    rows_path = bank_dir / "rows.private.json"
    atomic_write_json(rows_path, rows)
    public_rows_path = bank_dir / "generator_visible_rows.json"
    atomic_write_json(
        public_rows_path,
        [
            {
                "accepted_concept_id": item["accepted_concept_id"],
                "title": item["title"],
                "concept_summary": item["concept_summary"],
            }
            for item in rows
        ],
    )
    lineages = Counter(item["known_lineage_key"] for item in rows)
    value = {
        "schema_version": 1,
        "registry_path": str(Path(config["_registry_path"]).resolve()),
        "registry_sha256": config["_registry_sha256"],
        "registry_count": len(rows),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": _sha(rows_path),
        "generator_visible_rows_path": str(public_rows_path.resolve()),
        "generator_visible_rows_sha256": _sha(public_rows_path),
        "known_lineage_count": len(lineages),
        "known_multi_member_lineages": sum(size > 1 for size in lineages.values()),
        "known_lineage_method": (
            "persisted lineage IDs where available; otherwise exact normalized "
            "summary/rule identity and explicit known-sibling summaries"
        ),
        "generator_visible_fields": [
            "accepted_concept_id",
            "title",
            "concept_summary",
        ],
        "generator_hidden_fields": [
            "registry_task_id",
            "known_lineage_key",
            "search_mode",
            "intrinsic_status",
            "source_pipeline",
            "commit_set_id",
            "grid diagnostics",
            "intended_rule",
            "grids",
            "provenance",
            "judgments",
        ],
    }
    atomic_write_json(manifest_path, value)
    return value


def _public(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "accepted_concept_id": str(row["accepted_concept_id"]),
        "title": str(row["title"]),
        "concept_summary": str(row["concept_summary"]),
    }


def _lineage_order(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> list[Mapping[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["known_lineage_key"])].append(row)
    rng = Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    result = []
    for key in keys:
        members = sorted(groups[key], key=lambda item: str(item["accepted_concept_id"]))
        result.append(members[rng.randrange(len(members))])
    return result


def _stratified_order(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> list[Mapping[str, Any]]:
    strata: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source_era = str(row["commit_set_id"])
        # Coarse eras avoid one-entry strata while retaining historical spread.
        if source_era.startswith(("initial", "all-good", "v1")):
            source_era = "early"
        elif source_era.startswith(("e11", "e13", "e15", "post-v48")):
            source_era = "recent"
        else:
            source_era = "middle"
        key = (
            str(row["search_mode"]),
            str(row["intrinsic_status"]),
            str(row["size_bucket"]),
            source_era,
        )
        strata[key].append(row)
    rng = Random(seed)
    queues: dict[tuple[str, str, str, str], deque[Mapping[str, Any]]] = {}
    for key, values in strata.items():
        ordered = sorted(values, key=lambda item: str(item["accepted_concept_id"]))
        rng.shuffle(ordered)
        queues[key] = deque(ordered)
    keys = sorted(queues)
    rng.shuffle(keys)
    result: list[Mapping[str, Any]] = []
    while keys:
        next_keys = []
        for key in keys:
            queue = queues[key]
            if queue:
                result.append(queue.popleft())
            if queue:
                next_keys.append(key)
        rng.shuffle(next_keys)
        keys = next_keys
    return result


def _phase_a_conditions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    screen = config["screen"]
    conditions: list[dict[str, Any]] = []
    for policy, field in (
        ("task_uniform", "task_uniform_sizes"),
        ("known_lineage_uniform", "known_lineage_uniform_sizes"),
        ("stratified", "stratified_sizes"),
    ):
        for size in screen[field]:
            conditions.append({"sampling_policy": policy, "context_size": int(size)})
    if screen["include_full_raw"]:
        conditions.append({"sampling_policy": "full_raw", "context_size": "full"})
    if screen["include_full_known_lineage"]:
        conditions.append(
            {"sampling_policy": "full_known_lineage", "context_size": "full"}
        )
    if len(conditions) != 16:
        raise E16Error(
            f"Phase A requires 16 context conditions, found {len(conditions)}"
        )
    return conditions


def _write_implementation_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(config["_output_directory"])
    freeze = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "config": {
            "path": config["_config_path"],
            "sha256": _sha(config["_config_path"]),
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha(__file__),
        },
        "preregistration": {
            "path": str(
                (
                    Path(config["_repository_root"])
                    / "experiments/E16_BANK_CONTEXT_SCALE_AND_INTEGRATION_PLAN.md"
                ).resolve()
            ),
            "sha256": _sha(
                Path(config["_repository_root"])
                / "experiments/E16_BANK_CONTEXT_SCALE_AND_INTEGRATION_PLAN.md"
            ),
        },
        "prompts": {
            name: {
                "path": str((Path(config["_repository_root"]) / path_value).resolve()),
                "sha256": _sha(Path(config["_repository_root"]) / path_value),
            }
            for name, path_value in {
                "open": "prompts/v4_idea_open.md",
                "concept_led": "prompts/v4_idea_concept.md",
                "composition_led": "prompts/v4_idea_composition.md",
                "nomination": "prompts/e16_idea_nomination_addendum.md",
                "successful": "prompts/e15_bank_success_addendum.md",
                "diversity_preserving": "prompts/e16_bank_diversity_addendum.md",
            }.items()
        },
        "schema_hash": content_hash(IDEA_RESPONSE_SCHEMA),
        "protected_state": _protected_hashes(config),
    }
    _immutable(
        output / "implementation_manifest.json", freeze, "implementation manifest"
    )
    return freeze


def freeze_phase_a(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(config["_output_directory"])
    path = output / "phase_a" / "frozen_design.private.json"
    if path.is_file():
        _write_implementation_manifest(config)
        return read_json(path)
    bank = freeze_bank(config)
    rows = read_json(Path(bank["rows_path"]))
    samples_dir = output / "phase_a" / "contexts"
    samples_dir.mkdir(parents=True, exist_ok=True)
    calls = []
    for condition_index, condition in enumerate(_phase_a_conditions(config), 1):
        policy = condition["sampling_policy"]
        for mode_index, mode in enumerate(MODES, 1):
            seed = int(config["seed"]) + condition_index * 16001 + mode_index * 163
            if policy == "task_uniform":
                ordered = list(rows)
                Random(seed).shuffle(ordered)
            elif policy == "known_lineage_uniform":
                ordered = _lineage_order(rows, seed=seed)
            elif policy == "stratified":
                ordered = _stratified_order(rows, seed=seed)
            elif policy == "full_raw":
                ordered = list(rows)
                Random(seed).shuffle(ordered)
            elif policy == "full_known_lineage":
                ordered = _lineage_order(rows, seed=seed)
            else:
                raise E16Error(f"unknown sampling policy {policy}")
            requested = condition["context_size"]
            size = len(ordered) if requested == "full" else int(requested)
            if size > len(ordered):
                raise E16Error(
                    f"{policy} requested {size}, but only {len(ordered)} entries are available"
                )
            chosen = ordered[:size]
            condition_id = f"{policy}-{'full' if requested == 'full' else size}"
            call_id = (
                "e16a-"
                + content_hash(
                    {"seed": config["seed"], "condition": condition_id, "mode": mode}
                )[:14]
            )
            candidate_ids = [
                "k" + content_hash({"call": call_id, "sibling": index})[:13]
                for index in range(1, 5)
            ]
            context_path = samples_dir / f"{condition_id}__{mode}.json"
            context = [_public(item) for item in chosen]
            atomic_write_json(context_path, context)
            calls.append(
                {
                    "phase": "a",
                    "condition_id": condition_id,
                    "condition_index": condition_index,
                    "sampling_policy": policy,
                    "context_size_requested": requested,
                    "context_size_actual": len(context),
                    "context_fraction": round(len(context) / len(rows), 6),
                    "search_mode": mode,
                    "model": MODEL,
                    "effort": "high",
                    "prompt_arm": "successful",
                    "replication_seed": 0,
                    "call_id": call_id,
                    "candidate_ids": candidate_ids,
                    "context_path": str(context_path.resolve()),
                    "context_sha256": _sha(context_path),
                    "context_entry_ids": [
                        item["accepted_concept_id"] for item in context
                    ],
                }
            )
    if len(calls) != 48 or len({item["call_id"] for item in calls}) != 48:
        raise E16Error("Phase A frozen call design is not 48 unique calls")
    # Prove nestedness for every random policy/mode where the design requests it.
    nested_checks = []
    for policy in ("task_uniform", "known_lineage_uniform"):
        for mode in MODES:
            group = sorted(
                [
                    item
                    for item in calls
                    if item["sampling_policy"] == policy and item["search_mode"] == mode
                ],
                key=lambda item: item["context_size_actual"],
            )
            nested = all(
                smaller["context_entry_ids"]
                == larger["context_entry_ids"][: smaller["context_size_actual"]]
                for smaller, larger in pairwise(group)
            )
            # Per-call ordering seeds differ by condition. Freeze a canonical
            # nested ordering separately and replace contexts to satisfy the design.
            if not nested:
                base_seed = (
                    int(config["seed"])
                    + (1 if policy == "task_uniform" else 2) * 17011
                    + MODES.index(mode)
                )
                ordered = list(rows)
                if policy == "task_uniform":
                    Random(base_seed).shuffle(ordered)
                else:
                    ordered = _lineage_order(rows, seed=base_seed)
                for item in group:
                    size = item["context_size_actual"]
                    context = [_public(row) for row in ordered[:size]]
                    context_path = Path(item["context_path"])
                    atomic_write_json(context_path, context)
                    item["context_sha256"] = _sha(context_path)
                    item["context_entry_ids"] = [
                        row["accepted_concept_id"] for row in context
                    ]
                nested = True
            nested_checks.append({"policy": policy, "mode": mode, "nested": nested})
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "phase": "a",
        "registry_sha256": config["_registry_sha256"],
        "registry_count": config["_registry_count"],
        "bank_manifest_path": str(
            (output / "bank_snapshot" / "manifest.json").resolve()
        ),
        "bank_manifest_sha256": _sha(output / "bank_snapshot" / "manifest.json"),
        "calls": calls,
        "call_count": len(calls),
        "kernel_count": len(calls) * 4,
        "nested_sample_checks": nested_checks,
        "all_three_search_modes": True,
        "source_labels_hidden_downstream": True,
        "automatic_commit": False,
        "automatic_promotion": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)
    _write_implementation_manifest(config)
    return value


def _protected_hashes(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(config["_repository_root"])
    paths = {
        "committed_registry": Path(config["_registry_path"]),
        "production_reservoir": root / "reservoirs/v4_control.json",
        "pre_e15_workflow": Path(config["_pre_e15_workflow_path"]),
        "base_v4_control": Path(config["_base_v4_control_path"]),
    }
    return {
        name: {"path": str(path.resolve()), "sha256": _sha(path)}
        for name, path in paths.items()
    }


def _assert_protected(config: Mapping[str, Any]) -> None:
    frozen = read_json(
        Path(config["_output_directory"]) / "implementation_manifest.json"
    )
    if frozen["protected_state"] != _protected_hashes(config):
        raise E16Error("a protected production input changed during E16")


def _extract_calibrations(config: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    decoder = json.JSONDecoder()
    paths = sorted(
        Path(config["_calibration_source_run"]).glob(
            "selection_graph/v4_reservoir_supply/*/model_calls/v4_idea_search/*/prompt.md"
        )
    )
    for mode in MODES:
        marker = f"Search mode is `{mode}`"
        for path in paths:
            text = path.read_text(encoding="utf-8")
            if marker not in text or "Calibration descriptions:\n" not in text:
                continue
            tail = text.split("Calibration descriptions:\n", 1)[1].lstrip()
            try:
                descriptions, _ = decoder.raw_decode(tail)
            except json.JSONDecodeError:
                continue
            if isinstance(descriptions, list) and len(descriptions) == 4:
                result[mode] = [str(item) for item in descriptions]
                break
    if set(result) != set(MODES):
        raise E16Error("could not recover one frozen calibration set per search mode")
    return result


def _prompt_paths(root: Path) -> dict[str, Path]:
    return {
        "open": root / "prompts/v4_idea_open.md",
        "concept_led": root / "prompts/v4_idea_concept.md",
        "composition_led": root / "prompts/v4_idea_composition.md",
        "nomination": root / "prompts/e16_idea_nomination_addendum.md",
        "successful": root / "prompts/e15_bank_success_addendum.md",
        "diversity_preserving": root / "prompts/e16_bank_diversity_addendum.md",
    }


def _render_prompt(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    calibrations: Mapping[str, list[str]],
) -> tuple[str, str, str]:
    root = Path(config["_repository_root"])
    paths = _prompt_paths(root)
    names = [str(spec["search_mode"]), "nomination", str(spec["prompt_arm"])]
    sources = [load_prompt(paths[name]) for name in names]
    template = "\n\n".join(item[0] for item in sources)
    context_path = Path(str(spec["context_path"]))
    if not context_path.is_file() or _sha(context_path) != spec["context_sha256"]:
        raise E16Error(f"context artifact changed for {spec['call_id']}")
    values = {
        "CALIBRATION_DESCRIPTIONS": calibrations[str(spec["search_mode"])],
        "SEARCH_CALL_ID": spec["call_id"],
        "CANDIDATE_IDS": spec["candidate_ids"],
        "ACCEPTED_CONCEPT_BANK": read_json(context_path),
    }
    required = set(PLACEHOLDER_RE.findall(template))
    if required != set(values):
        raise E16Error(
            f"prompt binding mismatch for {spec['call_id']}: "
            f"missing={sorted(required - set(values))} extra={sorted(set(values) - required)}"
        )
    rendered = render_prompt(template, values)
    version = "+".join(item[1] for item in sources)
    template_hash = content_hash(
        {"templates": [item[2] for item in sources], "template": template}
    )
    return rendered, version, template_hash


class GenerationRuntime:
    def __init__(self, config: Mapping[str, Any], phase: str) -> None:
        self.config = config
        self.root = Path(config["_output_directory"]) / f"phase_{phase}" / "generation"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        base = load_v4_config(config["_base_v4_control_path"])
        base["paths"] = deepcopy(base["paths"])
        base["paths"]["runs"] = str(self.root)
        base["run"] = {
            **deepcopy(base["run"]),
            "id": f"e16-phase-{phase}",
            "mock": bool(config["run"]["mock"]),
        }
        base["model"] = deepcopy(base["model"])
        base["model"]["concurrency"] = int(config["screen"]["concurrency"])
        self.invoker = ModelInvoker(
            base,
            self.root,
            output_schemas={"e16_idea_search": IDEA_RESPONSE_SCHEMA},
            output_parser=parse_e16_stage,
        )
        self.lock = threading.RLock()
        self.state = (
            read_json(self.state_path)
            if self.state_path.is_file()
            else {
                "schema_version": 1,
                "pipeline": PIPELINE,
                "phase": phase,
                "completed": False,
                "call_ledger": {},
                "started_at_epoch_seconds": round(time.time(), 6),
            }
        )
        self.save()

    def save(self) -> None:
        with self.lock:
            atomic_write_json(self.state_path, self.state)

    def invoke(
        self, spec: Mapping[str, Any], prompt: str, version: str, template_hash: str
    ) -> dict[str, Any]:
        call_id = str(spec["call_id"])
        with self.lock:
            prior = self.state["call_ledger"].get(call_id)
            if isinstance(prior, Mapping) and prior.get("status") == "complete":
                return deepcopy(prior["response"])
            if isinstance(prior, Mapping) and prior.get("status") == "failed":
                raise E16Error(
                    f"terminal failed generation call requires explicit recovery: {call_id}"
                )
            self.state["call_ledger"][call_id] = {
                "status": "reserved",
                "role": "idea_search",
                "stage": "e16_idea_search",
                "model": spec["model"],
                "reasoning_effort": spec["effort"],
                "condition_id": spec["condition_id"],
                "search_mode": spec["search_mode"],
                "context_size": spec["context_size_actual"],
                "sampling_policy": spec["sampling_policy"],
                "prompt_arm": spec["prompt_arm"],
            }
            self.save()

        def validate(value: dict[str, Any]) -> None:
            if value.get("search_call_id") != call_id:
                raise E16Error("generator changed search call ID")
            if value.get("search_mode") != spec["search_mode"]:
                raise E16Error("generator changed search mode")
            if [item["candidate_id"] for item in value["ideas"]] != list(
                spec["candidate_ids"]
            ):
                raise E16Error("generator changed candidate IDs or order")

        try:
            result = self.invoker.invoke(
                stage="e16_idea_search",
                call_id=call_id,
                prompt=prompt,
                prompt_template_version=version,
                prompt_template_hash=template_hash,
                parent_ids=[],
                response_validator=validate,
                model_name=str(spec["model"]),
                reasoning_effort=str(spec["effort"]),
            )
            result_record = read_json(result.log_dir / "result.json")
            with self.lock:
                self.state["call_ledger"][call_id].update(
                    {
                        "status": "complete",
                        "attempts": result.attempts,
                        "retry_calls": max(0, result.attempts - 1),
                        "cache_source": result.cache_source,
                        "cache_key": result.cache_key,
                        "log_dir": str(result.log_dir.resolve()),
                        "prompt_hash": content_hash(prompt),
                        "prompt_bytes": len(prompt.encode("utf-8")),
                        "prompt_template_hash": template_hash,
                        "response": deepcopy(result.value),
                        "timing": result_record.get("timing"),
                        "usage": result_record.get("usage", {}),
                    }
                )
                self.save()
            return deepcopy(result.value)
        except Exception as exc:
            with self.lock:
                self.state["call_ledger"][call_id].update(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "finished_at_epoch_seconds": round(time.time(), 6),
                    }
                )
                self.save()
            raise


def run_generation_phase(
    config: Mapping[str, Any], *, phase: str, authorize_model_calls: bool
) -> dict[str, Any]:
    if phase != "a":
        raise E16Error("only Phase A is frozen before its shortlist exists")
    if not authorize_model_calls and not config["run"]["mock"]:
        raise E16Error("live E16 generation requires --authorize-model-calls")
    _assert_protected(config)
    design = freeze_phase_a(config)
    output = Path(config["_output_directory"]) / "phase_a"
    manifest_path = output / "generation_manifest.private.json"
    if manifest_path.is_file():
        value = read_json(manifest_path)
        if value.get("call_count") != 48 or value.get("kernel_count") != 192:
            raise E16Error("saved Phase-A generation manifest is incomplete")
        return value
    runtime = GenerationRuntime(config, "a")
    calibrations = _extract_calibrations(config)
    specs = deepcopy(design["calls"])
    Random(int(config["seed"]) + 16661).shuffle(specs)

    def one(spec: Mapping[str, Any]) -> dict[str, Any]:
        prompt, version, template_hash = _render_prompt(config, spec, calibrations)
        response = runtime.invoke(spec, prompt, version, template_hash)
        kernels = []
        for sibling, kernel in enumerate(response["ideas"], 1):
            kernels.append(
                {
                    "candidate_id": kernel["candidate_id"],
                    "kernel": deepcopy(kernel),
                    "private_phase": "a",
                    "private_condition_id": spec["condition_id"],
                    "private_sampling_policy": spec["sampling_policy"],
                    "private_context_size": spec["context_size_actual"],
                    "private_context_fraction": spec["context_fraction"],
                    "private_search_mode": spec["search_mode"],
                    "private_effort": spec["effort"],
                    "private_prompt_arm": spec["prompt_arm"],
                    "private_replication_seed": spec["replication_seed"],
                    "private_source_call_id": spec["call_id"],
                    "private_sibling_index": sibling,
                    "nominated_by_generator": kernel["candidate_id"]
                    == response["recommended_candidate_id"],
                    "self_development_brief": (
                        deepcopy(response["recommended_development_brief"])
                        if kernel["candidate_id"]
                        == response["recommended_candidate_id"]
                        else None
                    ),
                }
            )
        return {"spec": deepcopy(dict(spec)), "response": response, "kernels": kernels}

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=int(config["screen"]["concurrency"])) as pool:
        futures = {pool.submit(one, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - preserve every provider failure in the phase ledger
                failures.append({"call_id": str(spec["call_id"]), "error": str(exc)})
    if failures:
        atomic_write_json(output / "generation_failures.json", failures)
        raise E16Error(f"Phase A has {len(failures)} terminal generation failures")
    rows.sort(key=lambda item: str(item["spec"]["call_id"]))
    kernels = [kernel for row in rows for kernel in row["kernels"]]
    if (
        len(rows) != 48
        or len(kernels) != 192
        or len({item["candidate_id"] for item in kernels}) != 192
    ):
        raise E16Error("Phase-A generation cardinality changed")
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "phase": "a",
        "call_count": len(rows),
        "kernel_count": len(kernels),
        "calls": rows,
        "kernels": kernels,
        "call_ledger": deepcopy(runtime.state["call_ledger"]),
        "source_labels_hidden_downstream": True,
        "automatic_commit": False,
    }
    atomic_write_json(manifest_path, value)
    atomic_write_json(
        output / "generation_manifest.blinded.json",
        {
            "schema_version": 1,
            "source_labels_removed": True,
            "kernels": [
                {"candidate_id": item["candidate_id"], "kernel": item["kernel"]}
                for item in kernels
            ],
        },
    )
    runtime.state["completed"] = True
    runtime.state["completed_at_epoch_seconds"] = round(time.time(), 6)
    runtime.save()
    return value


def freeze_selector_windows(
    config: Mapping[str, Any], generation: Mapping[str, Any]
) -> dict[str, Any]:
    output = Path(config["_output_directory"]) / "phase_a"
    path = output / "selector_assignment.private.json"
    if path.is_file():
        return read_json(path)
    by_id = {str(item["candidate_id"]): item for item in generation["kernels"]}
    windows = {index: [] for index in range(1, 5)}
    assignments = []
    for call in generation["calls"]:
        for kernel in call["kernels"]:
            window = int(kernel["private_sibling_index"])
            candidate_id = str(kernel["candidate_id"])
            windows[window].append(candidate_id)
            assignments.append(
                {
                    "candidate_id": candidate_id,
                    "window": window,
                    "condition_id": kernel["private_condition_id"],
                    "sampling_policy": kernel["private_sampling_policy"],
                    "context_size": kernel["private_context_size"],
                    "search_mode": kernel["private_search_mode"],
                    "source_call_id": kernel["private_source_call_id"],
                    "sibling_index": kernel["private_sibling_index"],
                }
            )
    refs = []
    directory = output / "selector_windows"
    directory.mkdir(parents=True, exist_ok=True)
    for window, ids in windows.items():
        if len(ids) != 48 or len(ids) != len(set(ids)):
            raise E16Error(f"selector window {window} is not 48 unique kernels")
        # Each window must have one kernel from every condition-mode call.
        cells = {
            (by_id[item]["private_condition_id"], by_id[item]["private_search_mode"])
            for item in ids
        }
        if len(cells) != 48:
            raise E16Error(f"selector window {window} is not cell-balanced")
        ordered = list(ids)
        Random(int(config["seed"]) + 16901 * window).shuffle(ordered)
        value = {
            "schema_version": 1,
            "window": window,
            "source_labels_hidden": True,
            "candidate_ids": ordered,
            "kernels": [deepcopy(by_id[item]["kernel"]) for item in ordered],
        }
        destination = directory / f"window-{window:02d}.frozen.blinded.json"
        atomic_write_json(destination, value)
        refs.append(
            {
                "window": window,
                "path": str(destination.resolve()),
                "sha256": _sha(destination),
            }
        )
    value = {
        "schema_version": 1,
        "siblings_spread_one_per_window": True,
        "each_window_contains_every_condition_mode_once": True,
        "source_labels_hidden": True,
        "assignments": sorted(assignments, key=lambda item: item["candidate_id"]),
        "windows": refs,
    }
    atomic_write_json(path, value)
    return value


def _selector_config(
    config: Mapping[str, Any], *, window: int, reservoir: Path
) -> dict[str, Any]:
    base = load_v4_config(config["_base_v4_control_path"])
    base["seed"] = int(config["seed"]) + 17011 * window
    base["paths"] = deepcopy(base["paths"])
    base["paths"]["runs"] = str(
        Path(config["_output_directory"]) / "phase_a" / "selector_runs"
    )
    base["paths"]["reservoir"] = str(reservoir.resolve())
    base["paths"]["committed_task_registry"] = str(
        Path(config["_registry_path"]).resolve()
    )
    base["run"] = {
        **deepcopy(base["run"]),
        "id": f"e16-phase-a-selector-{window:02d}",
        "mock": bool(config["run"]["mock"]),
        "automatically_commit_generated_tasks": False,
    }
    base["model"] = deepcopy(base["model"])
    base["model"]["concurrency"] = 4
    base["reservoir"] = deepcopy(base["reservoir"])
    base["reservoir"].update(
        {"low_watermark": 48, "high_watermark": 192, "hard_capacity": 256}
    )
    base["selection"] = deepcopy(base["selection"])
    base["selection"].update(
        {
            "eligible_window_size": 48,
            "shard_size": 12,
            "global_selector_max_selected": 10,
            "max_plans": 10,
            "max_realizations": 10,
        }
    )
    return base


def _selector_reservoir(
    config: Mapping[str, Any],
    *,
    window: int,
    kernels: Sequence[Mapping[str, Any]],
    destination: Path,
) -> None:
    if destination.is_file():
        value = load_reservoir(destination)
        present = {
            str(item["proposal_id"])
            for item in value["proposals"]
            if item.get("supply_run_id") == config["run"]["id"]
        }
        expected = {str(item["candidate_id"]) for item in kernels}
        if present != expected:
            raise E16Error("selector reservoir changed during resume")
        return
    from .v4_pipeline import _DataContext

    selector_config = _selector_config(config, window=window, reservoir=destination)
    context = _DataContext(selector_config)
    production = load_reservoir(
        Path(config["_repository_root"]) / "reservoirs/v4_control.json"
    )
    active_ids = {item["proposal_id"] for item in production["active_reservations"]}
    context_proposals = [
        deepcopy(item)
        for item in production["proposals"]
        if item["proposal_id"] in active_ids
    ]
    proposals = []
    now = time.time()
    for index, kernel in enumerate(kernels):
        proposals.append(
            make_proposal(
                kernel,
                supply_run_id=str(config["run"]["id"]),
                supply_cycle_id=f"e16-phase-a-window-{window:02d}",
                calibration_task_ids=[],
                calibration_descriptions=[],
                calibration_version="e16-frozen-v47-calibrations-v1",
                prompt_version="e16-bank-context-screen-v1",
                prompt_hash=content_hash("e16-bank-context-screen-v1"),
                model=MODEL,
                reasoning_effort="high",
                source_bank_snapshot_id=context.bank_snapshot_id,
                embedding={
                    "provider": "deferred-selector-retrieval",
                    "model": "not-stored",
                    "dimension": 0,
                    "vector_hash": content_hash("not-stored"),
                },
                created_at_epoch_seconds=now + index / 1000,
            )
        )
    value = {
        "schema_version": 1,
        "reservoir_id": "arcgen-v4-persistent-idea-reservoir",
        "proposals": context_proposals + proposals,
        "active_reservations": deepcopy(production["active_reservations"]),
        "reservation_history": deepcopy(production["reservation_history"]),
        "rejection_memory": deepcopy(production["rejection_memory"]),
        "frozen_windows": [],
        "supply_history": [],
        "lifecycle_events": [],
    }
    validate_reservoir(value)
    atomic_write_json(destination, value)


def run_phase_a_selection(
    config: Mapping[str, Any], *, authorize_model_calls: bool
) -> dict[str, Any]:
    if not authorize_model_calls and not config["run"]["mock"]:
        raise E16Error("live E16 selection requires --authorize-model-calls")
    _assert_protected(config)
    output = Path(config["_output_directory"]) / "phase_a"
    destination = output / "selection_results.private.json"
    if destination.is_file():
        value = read_json(destination)
        if value.get("selector_call_count") != 20:
            raise E16Error("saved Phase-A selection is incomplete")
        return value
    generation = read_json(output / "generation_manifest.private.json")
    assignment = freeze_selector_windows(config, generation)
    private = {str(item["candidate_id"]): item for item in generation["kernels"]}

    def one(ref: Mapping[str, Any]) -> dict[str, Any]:
        from .v4_pipeline import V4ControlPipeline

        window = int(ref["window"])
        frozen = read_json(Path(ref["path"]))
        reservoir = output / "selector_windows" / f"window-{window:02d}.reservoir.json"
        _selector_reservoir(
            config, window=window, kernels=frozen["kernels"], destination=reservoir
        )
        selector = V4ControlPipeline(
            _selector_config(config, window=window, reservoir=reservoir)
        )
        briefs = selector.select_slate()
        records = selector.state["records"]
        assessments: dict[str, Any] = {}
        for shard in selector.state.get("shard_selectors", []):
            response = shard.get("response") or shard
            for item in response.get("candidate_assessments", []):
                assessments[str(item["candidate_id"])] = deepcopy(item)
        global_response = selector.state.get("global_selector") or {}
        response = (
            global_response.get("response")
            if isinstance(global_response, Mapping)
            else None
        )
        if isinstance(response, Mapping):
            for item in response.get("candidate_assessments", []):
                assessments[str(item["candidate_id"])] = deepcopy(item)
        selected = []
        for priority, brief in enumerate(briefs, 1):
            candidate_id = str(brief["candidate_id"])
            selected.append(
                {
                    "candidate_id": candidate_id,
                    "selector_priority": priority,
                    "editorial_brief": deepcopy(
                        records[candidate_id]["editorial_brief"]
                    ),
                    "selector_assessment": deepcopy(
                        records[candidate_id].get("selector_assessment")
                        or assessments.get(candidate_id)
                    ),
                    "private_condition_id": private[candidate_id][
                        "private_condition_id"
                    ],
                    "private_sampling_policy": private[candidate_id][
                        "private_sampling_policy"
                    ],
                    "private_context_size": private[candidate_id][
                        "private_context_size"
                    ],
                    "private_search_mode": private[candidate_id]["private_search_mode"],
                }
            )
        ledger = deepcopy(selector.state["call_ledger"])
        if sum(item.get("role") == "batch_selector" for item in ledger.values()) != 5:
            raise E16Error(f"window {window} did not use five unchanged selector calls")
        return {
            "window": window,
            "selected": selected,
            "assessments": assessments,
            "ledger": ledger,
            "run_dir": str(selector.run_dir.resolve()),
            "frozen_window": deepcopy(selector.state.get("frozen_window")),
            "shard_selectors": deepcopy(selector.state.get("shard_selectors")),
            "global_selector": deepcopy(selector.state.get("global_selector")),
        }

    refs = assignment["windows"]
    results: list[dict[str, Any]] = []
    # Two windows at a time, four model slots each, preserve the global cap of 8.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(one, ref): ref for ref in refs}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["window"])
    selected = [item for row in results for item in row["selected"]]
    ledgers = {
        f"window-{row['window']:02d}/{key}": value
        for row in results
        for key, value in row["ledger"].items()
    }
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "phase": "a",
        "windows": results,
        "selected": selected,
        "selected_count": len(selected),
        "selector_call_ledger": ledgers,
        "selector_call_count": sum(
            item.get("role") == "batch_selector" for item in ledgers.values()
        ),
        "source_labels_visible_to_selectors": False,
        "selector_thresholds_unchanged": True,
    }
    if value["selector_call_count"] != 20:
        raise E16Error("Phase A did not use twenty selector calls")
    atomic_write_json(destination, value)
    return value


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def analyze_phase_a(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(config["_output_directory"]) / "phase_a"
    destination = output / "report.json"
    if destination.is_file():
        return read_json(destination)
    generation = read_json(output / "generation_manifest.private.json")
    selection = read_json(output / "selection_results.private.json")
    selected_ids = {str(item["candidate_id"]) for item in selection["selected"]}
    assessments: dict[str, Any] = {}
    for window in selection["windows"]:
        assessments.update(window["assessments"])
    bank = read_json(
        Path(config["_output_directory"]) / "bank_snapshot" / "rows.private.json"
    )
    embedder = LocalHashingTfidfProvider(
        {"dimensions": 2048, "word_ngrams": [1, 2], "char_ngrams": [3, 5]}
    )
    bank_text = [f"{item['title']}\n{item['concept_summary']}" for item in bank]
    kernel_text = [
        str(item["kernel"]["central_relation"]) for item in generation["kernels"]
    ]
    bank_vectors = embedder.embed(bank_text)
    kernel_vectors = embedder.embed(kernel_text)
    nearest = []
    for kernel, vector in zip(generation["kernels"], kernel_vectors, strict=True):
        scores = [_cosine(vector, other) for other in bank_vectors]
        index = max(range(len(scores)), key=scores.__getitem__)
        nearest.append(
            {
                "candidate_id": kernel["candidate_id"],
                "nearest_accepted_concept_id": bank[index]["accepted_concept_id"],
                "similarity": round(scores[index], 6),
            }
        )
    nearest_by_id = {item["candidate_id"]: item for item in nearest}
    call_timing = {}
    for call_id, item in generation["call_ledger"].items():
        timing = item.get("timing") or {}
        call_timing[call_id] = float(timing.get("model_execution_seconds", 0.0))
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for kernel in generation["kernels"]:
        groups[str(kernel["private_condition_id"])].append(kernel)
    condition_rows = []
    for condition_id, kernels in sorted(groups.items()):
        ids = {str(item["candidate_id"]) for item in kernels}
        chosen = ids & selected_ids
        relevant_assessments = [
            assessments[item] for item in ids if item in assessments
        ]
        standout = sum(
            assessment.get("novel_concept_potential") == "standout"
            or assessment.get("functional_composition_potential") == "standout"
            for assessment in relevant_assessments
        )
        solid_both = sum(
            assessment.get("novel_concept_potential") in {"solid", "standout"}
            and assessment.get("functional_composition_potential")
            in {"solid", "standout"}
            for assessment in relevant_assessments
        )
        duplicates = sum(
            assessment.get("similarity_scope") == "same_minimal_procedure"
            for assessment in relevant_assessments
        )
        first = kernels[0]
        model_seconds = (
            sum(
                call_timing.get(str(item["private_source_call_id"]), 0.0)
                for item in kernels
            )
            / 4.0
        )
        similarities = [
            nearest_by_id[str(item["candidate_id"])]["similarity"] for item in kernels
        ]
        score = (
            len(chosen) * 8
            + standout * 2
            + solid_both
            - duplicates * 4
            - sum(score >= 0.9 for score in similarities) * 2
        )
        condition_rows.append(
            {
                "condition_id": condition_id,
                "sampling_policy": first["private_sampling_policy"],
                "context_size": first["private_context_size"],
                "kernels": len(kernels),
                "selector_promotions": len(chosen),
                "standout_assessments": standout,
                "solid_both_assessments": solid_both,
                "same_procedure_flags": duplicates,
                "mean_nearest_bank_similarity": round(
                    sum(similarities) / len(similarities), 6
                ),
                "near_bank_090": sum(item >= 0.9 for item in similarities),
                "active_model_seconds": round(model_seconds, 6),
                "active_model_minutes": round(model_seconds / 60.0, 6),
                "screen_score": round(score, 6),
            }
        )
    ranking = sorted(
        condition_rows,
        key=lambda item: (
            -item["screen_score"],
            -item["selector_promotions"],
            item["mean_nearest_bank_similarity"],
            item["active_model_seconds"],
            item["condition_id"],
        ),
    )
    finalists = []
    for row in ranking:
        if (
            not finalists
            or len(finalists) < int(config["screen"]["finalist_count"])
            and (
                any(
                    row["sampling_policy"] != prior["sampling_policy"]
                    or abs(int(row["context_size"]) - int(prior["context_size"])) >= 64
                    for prior in finalists
                    if isinstance(row["context_size"], int)
                    and isinstance(prior["context_size"], int)
                )
                or row["sampling_policy"]
                not in {item["sampling_policy"] for item in finalists}
            )
        ):
            finalists.append(row)
        if len(finalists) == int(config["screen"]["finalist_count"]):
            break
    if len(finalists) < int(config["screen"]["finalist_count"]):
        for row in ranking:
            if row not in finalists:
                finalists.append(row)
            if len(finalists) == int(config["screen"]["finalist_count"]):
                break
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "phase": "a",
        "completed": True,
        "registry_count": config["_registry_count"],
        "generation_calls": generation["call_count"],
        "kernels": generation["kernel_count"],
        "selector_calls": selection["selector_call_count"],
        "selected": selection["selected_count"],
        "condition_results": condition_rows,
        "ranking": [item["condition_id"] for item in ranking],
        "finalists": finalists,
        "nearest_bank_rows": nearest,
        "selection_labels_hidden": True,
        "screen_is_not_downstream_quality_proof": True,
        "automatic_promotion": False,
        "automatic_commit": False,
    }
    report["report_content_sha256"] = content_hash(report)
    atomic_write_json(destination, report)
    atomic_write_json(
        output / "phase_a_shortlist.frozen.json",
        {"finalists": finalists, "source_report_sha256": _sha(destination)},
    )
    lines = [
        "# E16 Phase A — bank context size and sampling screen",
        "",
        f"Generated {generation['kernel_count']} kernels in {generation['call_count']} calls; the unchanged selectors promoted {selection['selected_count']}.",
        "",
        "## Frozen finalists",
        "",
    ]
    for item in finalists:
        lines.append(
            f"- `{item['condition_id']}`: {item['selector_promotions']}/12 promoted; "
            f"score {item['screen_score']}; mean nearest-bank similarity {item['mean_nearest_bank_similarity']}."
        )
    lines.extend(
        ["", "These are screening finalists, not adopted production settings.", ""]
    )
    atomic_write_text(output / "report.md", "\n".join(lines))
    _assert_protected(config)
    return report


def validate(config: Mapping[str, Any]) -> dict[str, Any]:
    design = freeze_phase_a(config)
    bank = read_json(
        Path(config["_output_directory"]) / "bank_snapshot" / "manifest.json"
    )
    return {
        "valid": True,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "registry_count": bank["registry_count"],
        "known_lineage_count": bank["known_lineage_count"],
        "phase_a_calls": design["call_count"],
        "phase_a_kernels": design["kernel_count"],
        "nested_samples_valid": all(
            item["nested"] for item in design["nested_sample_checks"]
        ),
        "all_three_modes": True,
        "automatic_commit": False,
        "production_state_mutation": False,
        "automatic_promotion": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "validate",
            "phase-a-generate",
            "phase-a-select",
            "phase-a-report",
            "phase-a-all",
        ],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--authorize-model-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate":
        print(json.dumps(validate(config), indent=2))
        return 0
    if args.command in {"phase-a-generate", "phase-a-all"}:
        run_generation_phase(
            config, phase="a", authorize_model_calls=args.authorize_model_calls
        )
    if args.command in {"phase-a-select", "phase-a-all"}:
        run_phase_a_selection(config, authorize_model_calls=args.authorize_model_calls)
    if args.command in {"phase-a-report", "phase-a-all"}:
        print(json.dumps(analyze_phase_a(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
