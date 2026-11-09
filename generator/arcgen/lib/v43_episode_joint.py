"""Matched, arm-blind ARCGEN V4.3 episode-joint inference experiment.

The intervention occurs at concept-led idea search.  Control and treatment use
the same calibration descriptions, model, effort, schema and downstream V4.2
authoring/review path.  Treatment source is private after idea search.  The
post-hoc mechanism and diversity audits are descriptive and cannot affect task
quality or portfolio placement.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence
import json
import time

from jsonschema import Draft202012Validator

from .embeddings import build_embedding_provider
from .lean_v3_rendering import build_contact_sheet, render_episode
from .retrieval import retrieve_union
from .schemas import SchemaError, parse_json_object
from .util import (
    PLACEHOLDER_RE,
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    parallel_map,
    read_json,
)
from .v4_pipeline import V4ControlPipeline, _DataContext, _calibration_assignments
from .v4_reservoir import load_reservoir, memory_digest, reservation_digest
from .v41_audits import matched_noninferiority, mechanism_diversity_summary
from .v41_experiments import V41ExperimentError, V41Runtime, _task_metrics
from .v41_integration import _assert_hashes_unchanged, _protected_hashes
from .v41_schemas import STRICT_OUTPUT_SCHEMAS as V41_STRICT_OUTPUT_SCHEMAS
from .v41_structural_transfer import _task_set_metrics
from .v41_supported_departure import (
    SlateEscalationBudget,
    _realize_records,
    _record_stage,
    _run_blind,
    _run_technical,
)
from .v41_test_challenge import _board as _single_instance_boards
from .v43_config import V43_DESIGN, V43_PIPELINE, verify_v42_baseline


class EpisodeJointError(V41ExperimentError):
    pass


TEXT = {"type": "string", "minLength": 1}
MAYBE_TEXT = {"type": "string"}
BOOLEAN = {"type": "boolean"}
CONCEPT = {"type": "string", "enum": ["weak", "solid", "standout"]}
COMPOSITION = {"type": "string", "enum": ["thin", "solid", "standout"]}


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": deepcopy(dict(properties)),
        "required": list(properties),
        "additionalProperties": False,
    }


EDITORIAL_BRIEF = _object(
    {
        "route": {
            "type": "string",
            "enum": ["concept_led", "composition_led", "flagship", "wildcard"],
        },
        "preserve": TEXT,
        "develop": TEXT,
        "avoid": TEXT,
        "closest_similarity_concern": MAYBE_TEXT,
        "selection_reason": TEXT,
        "optional_development_suggestion": {"type": "null"},
    }
)

KERNEL_ASSESSMENT = _object(
    {
        "candidate_id": TEXT,
        "recommendation": {
            "type": "string",
            "enum": ["develop", "hold", "reject"],
        },
        "novel_concept_potential": CONCEPT,
        "functional_composition_potential": COMPOSITION,
        "fair_episode_potential": {
            "type": "string",
            "enum": ["plausible", "uncertain", "implausible"],
        },
        "derivative_of": {
            "type": "string",
            "enum": ["none", "real_task", "generated_task", "same_set"],
        },
        "same_minimal_procedure_with": {
            "type": "array",
            "items": TEXT,
            "minItems": 0,
            "maxItems": 15,
        },
        "reason": TEXT,
        "development_brief": {
            "anyOf": [deepcopy(EDITORIAL_BRIEF), {"type": "null"}]
        },
    }
)

BLINDED_SELECTOR_SCHEMA = _object(
    {
        "selector_id": TEXT,
        "assessments": {
            "type": "array",
            "items": deepcopy(KERNEL_ASSESSMENT),
            "minItems": 1,
            "maxItems": 16,
        },
        "quality_order": {
            "type": "array",
            "items": TEXT,
            "minItems": 1,
            "maxItems": 16,
        },
        "portfolio_observation": TEXT,
    }
)

JOINT_AUDIT_ENTRY = _object(
    {
        "task_id": TEXT,
        "status": {
            "type": "string",
            "enum": ["genuine", "weak", "absent", "invalid"],
        },
        "jointly_inferred_fact": MAYBE_TEXT,
        "single_pair_insufficiency": MAYBE_TEXT,
        "joint_support": MAYBE_TEXT,
        "order_independence": MAYBE_TEXT,
        "test_determination": MAYBE_TEXT,
        "reason": TEXT,
        "open_ended_signature": MAYBE_TEXT,
    }
)

JOINT_AUDIT_SCHEMA = _object(
    {
        "audit_id": TEXT,
        "judgments": {
            "type": "array",
            "items": deepcopy(JOINT_AUDIT_ENTRY),
            "minItems": 1,
            "maxItems": 16,
        },
        "non_gating_confirmation": BOOLEAN,
    }
)

STRICT_EXTENSION_SCHEMAS = {
    "v43_blinded_kernel_selector": BLINDED_SELECTOR_SCHEMA,
    "v43_episode_joint_plan": deepcopy(
        V41_STRICT_OUTPUT_SCHEMAS["v41_episode_joint_plan"]
    ),
    "v43_episode_joint_audit": JOINT_AUDIT_SCHEMA,
    "v43_episode_joint_diversity": deepcopy(
        V41_STRICT_OUTPUT_SCHEMAS["v41_diversity_audit"]
    ),
}


def _transport(value: Any) -> Any:
    unsupported = {
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
    if isinstance(value, dict):
        return {
            key: _transport(item)
            for key, item in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [_transport(item) for item in value]
    return value


def parse_v43_extension(
    stage: str, data: str | bytes | bytearray | Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in STRICT_EXTENSION_SCHEMAS:
        raise SchemaError("$.stage", f"unknown V4.3 extension stage {stage!r}")
    value = (
        parse_json_object(data)
        if isinstance(data, (str, bytes, bytearray))
        else dict(data)
    )
    errors = sorted(
        Draft202012Validator(STRICT_EXTENSION_SCHEMAS[stage]).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{item}]" if isinstance(item, int) else f".{item}"
            for item in error.absolute_path
        )
        raise SchemaError(path, error.message)
    if stage == "v43_blinded_kernel_selector":
        assessed = [item["candidate_id"] for item in value["assessments"]]
        ordered = value["quality_order"]
        if len(assessed) != len(set(assessed)) or len(ordered) != len(set(ordered)):
            raise SchemaError("$", "selector candidate IDs must be unique")
        if set(assessed) != set(ordered):
            raise SchemaError("$.quality_order", "quality order must cover every assessment")
        assessed_set = set(assessed)
        for index, item in enumerate(value["assessments"]):
            siblings = set(item["same_minimal_procedure_with"])
            if not siblings <= assessed_set - {item["candidate_id"]}:
                raise SchemaError(
                    f"$.assessments[{index}].same_minimal_procedure_with",
                    "siblings must identify other candidates in this comparison set",
                )
            develop = item["recommendation"] == "develop"
            if (item["development_brief"] is not None) != develop:
                raise SchemaError(
                    f"$.assessments[{index}].development_brief",
                    "a brief is required exactly for develop recommendations",
                )
            if develop and (
                item["novel_concept_potential"] == "weak"
                or item["functional_composition_potential"] == "thin"
                or item["fair_episode_potential"] != "plausible"
            ):
                raise SchemaError(
                    f"$.assessments[{index}]",
                    "develop recommendations must meet both quality floors and be plausible",
                )
    elif stage == "v43_episode_joint_plan":
        retained = value["property_status"] == "retained"
        abort = value["property_status"] == "abort"
        if (value["property_contract"] is not None) != retained:
            raise SchemaError(
                "$.property_contract", "contract is required exactly when retained"
            )
        if (value["concrete_plan"] is None) != abort:
            raise SchemaError("$.concrete_plan", "only abort may omit its plan")
    elif stage == "v43_episode_joint_audit":
        if value["non_gating_confirmation"] is not True:
            raise SchemaError("$.non_gating_confirmation", "audit must remain non-gating")
    elif value["property"] != "episode_joint_inference":
        raise SchemaError("$.property", "V4.3 diversity review has the wrong property")
    return value


def _install_extensions(runtime: V41Runtime) -> None:
    root = Path(runtime.config["_repository_root"])
    specifications = {
        "v43_episode_joint_idea": (
            "v4_3_episode_joint_idea_addendum.md",
            "v4_idea_search",
        ),
        "v43_episode_joint_plan": (
            "v4_3_episode_joint_planning_addendum.md",
            "v43_episode_joint_plan",
        ),
        "v43_blinded_selector": (
            "v4_3_blinded_kernel_selector.md",
            "v43_blinded_kernel_selector",
        ),
        "v43_episode_joint_auditor": (
            "v4_3_episode_joint_auditor.md",
            "v43_episode_joint_audit",
        ),
        "v43_episode_joint_diversity": (
            "v4_3_episode_joint_diversity.md",
            "v43_episode_joint_diversity",
        ),
    }
    destination = runtime.run_dir / "prompts" / "v43_extensions"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": V43_PIPELINE,
        "prompts": {},
        "implementation": {},
    }
    for name, (filename, stage) in specifications.items():
        text, version, digest = load_prompt(root / "prompts" / filename)
        runtime.prompts.entries[name] = {
            "filename": filename,
            "text": text,
            "version": version,
            "hash": digest,
            "stage": stage,
        }
        target = destination / filename
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise EpisodeJointError(f"V4.3 prompt changed during resume: {filename}")
        if not target.exists():
            atomic_write_text(target, text)
        manifest["prompts"][name] = {
            "filename": filename,
            "version": version,
            "sha256": digest,
            "stage": stage,
            "placeholder_names": sorted(PLACEHOLDER_RE.findall(text)),
        }
    for relative in (
        "arcgen.lib/v43_config.py",
        "arcgen.lib/v43_episode_joint.py",
        "config/v4_3_exp_episode_joint.yaml",
    ):
        path = root / relative
        manifest["implementation"][relative] = sha256(path.read_bytes()).hexdigest()
    path = runtime.run_dir / "v43_extension_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise EpisodeJointError("V4.3 implementation or prompt changed during resume")
    if not path.exists():
        atomic_write_json(path, manifest)

    original = runtime.invoker.output_parser

    def parse(stage: str, data: Any) -> Any:
        if stage in STRICT_EXTENSION_SCHEMAS:
            return parse_v43_extension(stage, data)
        return original(stage, data)

    runtime.invoker.output_parser = parse
    runtime.invoker.output_schemas.update(
        {stage: _transport(schema) for stage, schema in STRICT_EXTENSION_SCHEMAS.items()}
    )
    runtime.invoker._write_schemas()


def _opaque_kernel_id(seed: int, slate: int, pair: int, arm: str, index: int) -> str:
    digest = sha256(f"v43:{seed}:{slate}:{pair}:{arm}:{index}".encode()).hexdigest()
    return "k" + digest[:13]


def _freeze_design(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    path = run_dir / "frozen_experiment_design.private.json"
    if path.exists():
        return read_json(path)
    context = _DataContext(config)
    embedder = build_embedding_provider(config["embedding"], config["paths"]["cache"])
    settings = config["episode_joint"]
    slates = int(settings["independent_slates"])
    calls_per_slate = int(settings["paired_generation_calls_per_slate"])
    assignments = _calibration_assignments(
        context.generator_pool,
        count=int(config["reservoir"]["calibration_examples_per_call"]),
        calls=slates * calls_per_slate,
        embedder=embedder,
        seed=int(config["seed"]) + 43001,
    )
    paired_calls = []
    arm_map: dict[str, str] = {}
    offset = 0
    for slate in range(1, slates + 1):
        for pair in range(1, calls_per_slate + 1):
            calibration = assignments[offset]
            offset += 1
            calls = []
            for arm in ("control", "treatment"):
                ids = [
                    _opaque_kernel_id(int(config["seed"]), slate, pair, arm, index)
                    for index in range(1, int(settings["ideas_per_call"]) + 1)
                ]
                arm_map.update({candidate_id: arm for candidate_id in ids})
                calls.append(
                    {
                        "arm": arm,
                        "call_id": f"slate-{slate:02d}-pair-{pair:02d}-{arm}",
                        "candidate_ids": ids,
                    }
                )
            paired_calls.append(
                {
                    "slate_index": slate,
                    "pair_index": pair,
                    "calibration_task_ids": [item.task_id for item in calibration],
                    "calibration_descriptions": [
                        item.rule_description for item in calibration
                    ],
                    "calls": calls,
                }
            )
    reservoir = load_reservoir(config["paths"]["reservoir"])
    protected = _protected_hashes(config)
    value = {
        "schema_version": 1,
        "pipeline": V43_PIPELINE,
        "design": V43_DESIGN,
        "v42_parent": deepcopy(config["_v4_2_baseline_verification"]),
        "source_bank_snapshot_id": context.bank_snapshot_id,
        "accepted_bank_digest": context.accepted_digest(),
        "reservation_digest": reservation_digest(reservoir),
        "rejected_memory_digest": memory_digest(reservoir),
        "fixed_anchor_panel_id": context.anchor_manifest["panel_id"],
        "fixed_anchor_ids_excluded_from_calibration": sorted(context.anchor_ids),
        "model_and_effort_settings": deepcopy(config["model"]["roles"]),
        "assignment_seed": int(config["seed"]),
        "paired_calls": paired_calls,
        "private_candidate_arm_mapping": arm_map,
        "arm_labels_hidden_from_selection_and_downstream": True,
        "normal_concept_prompt_unchanged": True,
        "only_idea_search_differs_between_paired_generation_calls": True,
        "protected_production_state": protected,
        "automatic_task_commit": False,
        "automatic_promotion": False,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        run_dir / "frozen_comparison_manifest.public.json",
        {
            key: deepcopy(item)
            for key, item in value.items()
            if key not in {"private_candidate_arm_mapping"}
        },
    )
    return value


def _generate_kernels(runtime: V41Runtime, design: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = runtime.run_dir / "kernel_supply_manifest.private.json"
    if path.exists():
        return read_json(path)["kernels"]
    specs = []
    for pair in design["paired_calls"]:
        for call in pair["calls"]:
            specs.append((pair, call))

    def one(spec: tuple[Mapping[str, Any], Mapping[str, Any]]) -> list[dict[str, Any]]:
        pair, call = spec
        arm = str(call["arm"])
        ids = list(call["candidate_ids"])

        def validate(value: dict[str, Any]) -> None:
            if [item["candidate_id"] for item in value["ideas"]] != ids:
                raise EpisodeJointError("idea call returned unassigned candidate IDs")

        kwargs: dict[str, Any] = {}
        if arm == "treatment":
            kwargs["composed_with"] = ("v43_episode_joint_idea",)
        response = runtime.call(
            prompt_name="idea_concept",
            call_id=str(call["call_id"]),
            values={
                "CALIBRATION_DESCRIPTIONS": list(pair["calibration_descriptions"]),
                "SEARCH_CALL_ID": str(call["call_id"]),
                "CANDIDATE_IDS": ids,
            },
            parent_ids=[],
            role="idea_search",
            arm_id=f"private-{arm}-idea-supply",
            expected={
                "search_call_id": str(call["call_id"]),
                "search_mode": "concept_led",
            },
            validator=validate,
            **kwargs,
        )
        return [
            {
                "candidate_id": item["candidate_id"],
                "slate_index": int(pair["slate_index"]),
                "pair_index": int(pair["pair_index"]),
                "private_arm": arm,
                "kernel": deepcopy(item),
                "calibration_task_ids": list(pair["calibration_task_ids"]),
                "generation_call_id": str(call["call_id"]),
            }
            for item in response["ideas"]
        ]

    rows = parallel_map(
        one,
        specs,
        min(len(specs), int(runtime.config["model"]["concurrency"])),
    )
    kernels = [item for group in rows for item in group]
    expected = int(runtime.config["episode_joint"]["independent_slates"]) * 2 * int(
        runtime.config["episode_joint"]["paired_generation_calls_per_slate"]
    ) * int(runtime.config["episode_joint"]["ideas_per_call"])
    if len(kernels) != expected or len({item["candidate_id"] for item in kernels}) != expected:
        raise EpisodeJointError("paired generation did not produce the complete unique supply")
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "arm_mapping_is_private": True,
            "kernel_count": len(kernels),
            "kernels": kernels,
        },
    )
    atomic_write_json(
        runtime.run_dir / "kernel_supply_manifest.blinded.json",
        {
            "schema_version": 1,
            "arm_labels_removed": True,
            "kernels": [
                {
                    "candidate_id": item["candidate_id"],
                    "slate_index": item["slate_index"],
                    "kernel": item["kernel"],
                }
                for item in kernels
            ],
        },
    )
    return kernels


def _reference_packages(
    config: Mapping[str, Any], kernels: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    context = _DataContext(config)
    embedder = build_embedding_provider(config["embedding"], config["paths"]["cache"])
    packages = []
    for row in kernels:
        kernel = row["kernel"]
        hits = retrieve_union(
            kernel["central_relation"],
            context.references,
            embedder,
            top_k=int(config["selection"]["references_per_candidate"]),
            text_getter=lambda item: item["central_relation"],
            id_getter=lambda item: item["reference_id"],
        )
        packages.append(
            {
                "kernel": deepcopy(kernel),
                "nearest_references": [
                    {
                        "reference_id": hit.reference_id,
                        "source": hit.reference["source"],
                        "archive_tier": hit.reference["archive_tier"],
                        "summary": hit.reference["summary"][:700],
                        "similarity": round(hit.score, 6),
                    }
                    for hit in hits
                ],
            }
        )
    reservoir = load_reservoir(config["paths"]["reservoir"])
    return (
        packages,
        context.accepted_digest(),
        reservation_digest(reservoir),
        memory_digest(reservoir),
    )


def _select_kernels(
    runtime: V41Runtime, kernels: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = runtime.run_dir / "selected_kernel_manifest.private.json"
    if path.exists():
        value = read_json(path)
        return value["development_sample"], value["slates"]
    selected: list[dict[str, Any]] = []
    slate_ledgers: list[dict[str, Any]] = []
    settings = runtime.config["episode_joint"]
    maximum = int(settings["maximum_development_per_arm_per_slate"])
    for slate_index in range(1, int(settings["independent_slates"]) + 1):
        slate_rows = [item for item in kernels if item["slate_index"] == slate_index]
        rng = Random(int(runtime.config["seed"]) + slate_index * 43019)
        rng.shuffle(slate_rows)
        packages, accepted, reservations, rejected = _reference_packages(
            runtime.config, slate_rows
        )
        selector_id = f"v43-blinded-selector-{slate_index:02d}"
        expected_ids = {item["candidate_id"] for item in slate_rows}

        def validate(value: dict[str, Any]) -> None:
            returned = {item["candidate_id"] for item in value["assessments"]}
            if returned != expected_ids or set(value["quality_order"]) != expected_ids:
                raise EpisodeJointError("blinded selector did not cover the frozen set")

        response = runtime.call(
            prompt_name="v43_blinded_selector",
            call_id=selector_id,
            values={
                "CANDIDATES_WITH_REFERENCES": packages,
                "ACCEPTED_BANK_DIGEST": accepted,
                "IN_FLIGHT_RESERVATIONS": reservations,
                "REJECTED_MEMORY_DIGEST": rejected,
                "SELECTOR_ID": selector_id,
            },
            parent_ids=sorted(expected_ids),
            role="batch_selector",
            arm_id="blinded-development-selection",
            expected={"selector_id": selector_id},
            validator=validate,
            forbidden_prompt_fragments=(
                "episode-joint treatment",
                "control arm",
                "treatment arm",
                "experimental source",
            ),
        )
        by_id = {item["candidate_id"]: item for item in slate_rows}
        assessment = {item["candidate_id"]: item for item in response["assessments"]}
        chosen: list[str] = []
        counts = Counter()
        for candidate_id in response["quality_order"]:
            item = assessment[candidate_id]
            arm = by_id[candidate_id]["private_arm"]
            if item["recommendation"] != "develop" or counts[arm] >= maximum:
                continue
            if any(
                other in chosen for other in item["same_minimal_procedure_with"]
            ):
                continue
            if any(
                candidate_id
                in assessment[other]["same_minimal_procedure_with"]
                for other in chosen
            ):
                continue
            chosen.append(candidate_id)
            counts[arm] += 1
        for global_priority, candidate_id in enumerate(chosen, 1):
            source = by_id[candidate_id]
            arm = source["private_arm"]
            arm_priority = 1 + sum(
                item["private_arm"] == arm for item in selected if item["slate_index"] == slate_index
            )
            selected.append(
                {
                    **deepcopy(source),
                    "selector_global_priority": global_priority,
                    "source_development_priority": arm_priority,
                    "editorial_brief": deepcopy(
                        assessment[candidate_id]["development_brief"]
                    ),
                    "selector_assessment": deepcopy(assessment[candidate_id]),
                }
            )
        slate_ledgers.append(
            {
                "slate_index": slate_index,
                "selector_id": selector_id,
                "candidate_count": len(slate_rows),
                "quality_order": response["quality_order"],
                "development_counts_by_private_arm": dict(sorted(counts.items())),
                "development_candidate_ids": chosen,
                "supply_shortfall": {
                    arm: max(0, maximum - counts[arm])
                    for arm in ("control", "treatment")
                },
                "source_labels_visible_to_selector": False,
                "response": response,
            }
        )
    if not selected:
        raise EpisodeJointError("arm-blind quality selection found no developable kernels")
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "arm_mapping_is_private": True,
            "development_sample": selected,
            "slates": slate_ledgers,
        },
    )
    atomic_write_json(
        runtime.run_dir / "selected_kernel_manifest.blinded.json",
        {
            "schema_version": 1,
            "source_labels_removed": True,
            "development_sample": [
                {
                    key: deepcopy(value)
                    for key, value in item.items()
                    if key not in {"private_arm"}
                }
                for item in selected
            ],
        },
    )
    return selected, slate_ledgers


def _initialize_records(runtime: V41Runtime, selected: Sequence[Mapping[str, Any]]) -> None:
    with runtime.lock:
        if "semantic_records" in runtime.state:
            return
        records = {}
        for source in selected:
            task_id = str(source["candidate_id"])
            records[task_id] = {
                "task_id": task_id,
                "source_candidate_id": task_id,
                "arm": str(source["private_arm"]),
                "slate_index": int(source["slate_index"]),
                "source_development_priority": int(
                    source["source_development_priority"]
                ),
                "selector_global_priority": int(source["selector_global_priority"]),
                "search_mode": "concept_led",
                "kernel": deepcopy(source["kernel"]),
                "editorial_brief": deepcopy(source["editorial_brief"]),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
        runtime.state["semantic_records"] = records
        runtime.state["v43_pipeline"] = V43_PIPELINE
        runtime.state["v43_experimental_property"] = "episode_joint_inference"
        runtime.save()


PRIVATE_LABELS = (
    "control arm",
    "treatment arm",
    "experimental arm",
    "experimental source",
    "private arm",
)


def _plan_records(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]

    def one(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        record = deepcopy(records[task_id])
        brief = deepcopy(record["editorial_brief"])
        brief["optional_development_suggestion"] = None
        values = {
            "CANDIDATE": {**deepcopy(record["kernel"]), "candidate_id": task_id},
            "EDITORIAL_BRIEF": {**brief, "candidate_id": task_id},
            "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(brief["route"]),
            "CANDIDATE_ID": task_id,
        }
        try:
            if record["arm"] == "control":
                plan = runtime.call(
                    prompt_name="concrete_planner",
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="private-control-authoring",
                    expected={"candidate_id": task_id},
                )
                response = {
                    "candidate_id": task_id,
                    "property_status": "not_applicable",
                    "decision_reason": "ordinary frozen V4.2 concrete planning",
                    "property_contract": None,
                    "concrete_plan": plan,
                }
            else:
                response = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("v43_episode_joint_plan",),
                    call_id=f"{task_id}-episode-joint-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="private-treatment-authoring",
                    expected={"candidate_id": task_id},
                )
                nested = response.get("concrete_plan")
                if nested is not None and nested.get("candidate_id") != task_id:
                    raise EpisodeJointError("planner changed the nested candidate ID")
                if (
                    nested is not None
                    and nested.get("status") == "abort"
                    and response["property_status"] != "abort"
                ):
                    raise EpisodeJointError("an aborted nested plan must use abort status")
            plan = response["concrete_plan"]
            leaked = [
                label
                for label in PRIVATE_LABELS
                if label in json.dumps(plan, ensure_ascii=False).casefold()
            ] if plan is not None else []
            if leaked:
                raise EpisodeJointError(f"concrete plan leaked private labels: {leaked}")
            return task_id, response, None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [
        task_id for task_id, record in records.items() if "planning_result" not in record
    ]
    results = (
        parallel_map(
            one,
            pending,
            min(len(pending), int(runtime.config["model"]["concurrency"])),
        )
        if pending
        else []
    )
    for task_id, result, error in results:
        if error or result is None:
            _record_stage(
                runtime,
                task_id,
                "planning",
                status="operational_failure",
                planning_error=error,
            )
            continue
        plan = result["concrete_plan"]
        status = (
            "plan_aborted"
            if plan is None or plan.get("status") == "abort"
            else "plan_ready"
        )
        _record_stage(
            runtime,
            task_id,
            "planning",
            status=status,
            planning_result=result,
            concrete_plan=plan,
        )
        destination = runtime.run_dir / "plans" / records[task_id]["arm"]
        destination.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination / f"{task_id}.json", result)


def _budgets(runtime: V41Runtime) -> dict[tuple[int, str], SlateEscalationBudget]:
    settings = runtime.config["episode_joint"]
    limit = int(settings["escalation_calls_per_arm_per_slate"])
    with runtime.lock:
        usage = runtime.state.setdefault("v43_escalation_usage", {})
        for slate in range(1, int(settings["independent_slates"]) + 1):
            for arm in ("control", "treatment"):
                usage.setdefault(f"{slate}:{arm}", 0)
        runtime.save()
    result = {}
    for slate in range(1, int(settings["independent_slates"]) + 1):
        for arm in ("control", "treatment"):
            key = f"{slate}:{arm}"

            def persist(used: int, *, usage_key: str = key) -> None:
                with runtime.lock:
                    runtime.state["v43_escalation_usage"][usage_key] = used
                    runtime.save()

            result[(slate, arm)] = SlateEscalationBudget(
                limit,
                used=int(runtime.state["v43_escalation_usage"][key]),
                on_change=persist,
            )
    return result


def _mechanism_audit(runtime: V41Runtime) -> list[dict[str, Any]]:
    path = runtime.run_dir / "mechanism_audit_report.json"
    if path.exists():
        return read_json(path)["judgments"]
    records = runtime.state["semantic_records"]
    completed = [record for record in records.values() if record.get("complete_task")]
    rng = Random(int(runtime.config["seed"]) + 438123)
    rng.shuffle(completed)
    judgments: list[dict[str, Any]] = []
    calls = []
    batch = int(runtime.config["episode_joint"]["audit_batch_size"])
    for offset in range(0, len(completed), batch):
        chunk = completed[offset : offset + batch]
        audit_id = f"episode-joint-audit-{offset // batch + 1:02d}"
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

        def validate(value: dict[str, Any]) -> None:
            ids = [item["task_id"] for item in value["judgments"]]
            expected = {item["task_id"] for item in packages}
            if len(ids) != len(set(ids)) or set(ids) != expected:
                raise EpisodeJointError("mechanism audit omitted or duplicated a task")

        response = runtime.call(
            prompt_name="v43_episode_joint_auditor",
            call_id=audit_id,
            values={"TASK_PACKAGES": packages, "AUDIT_ID": audit_id},
            parent_ids=[item["task_id"] for item in packages],
            role="editorial_board",
            arm_id="blinded-episode-joint-audit",
            expected={"audit_id": audit_id},
            validator=validate,
            forbidden_prompt_fragments=PRIVATE_LABELS,
        )
        judgments.extend(response["judgments"])
        calls.append(response)
    atomic_write_json(
        path,
        {
            "arm_and_source_hidden": True,
            "non_gating": True,
            "calls": calls,
            "judgments": judgments,
        },
    )
    return judgments


def _diversity_audit(
    runtime: V41Runtime, judgments: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    path = runtime.run_dir / "diversity_cluster_report.json"
    if path.exists():
        return read_json(path)
    treatment_ids = {
        task_id
        for task_id, record in runtime.state["semantic_records"].items()
        if record["arm"] == "treatment"
    }
    positives = [
        item
        for item in judgments
        if item["task_id"] in treatment_ids and item["status"] == "genuine"
    ]
    if not positives:
        value = {
            "audit": None,
            "summary": {
                "raw_positive_count": 0,
                "unique_positive_mechanism_instances": 0,
                "material_cluster_count": 0,
                "largest_cluster_fraction": None,
                "same_procedure_tasks_counted_as_separate_successes": False,
            },
            "semantic_embedding_analysis": {
                "status": "not_applicable",
                "reason": "no genuine treatment instances",
            },
        }
        atomic_write_json(path, value)
        return value

    signatures = {item["task_id"]: item["open_ended_signature"] for item in positives}
    embedding: dict[str, Any]
    nearest: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in signatures}
    try:
        ordered = sorted(signatures)
        embedder = build_embedding_provider(
            runtime.config["embedding"], runtime.config["paths"]["cache"]
        )
        vectors = embedder.embed([signatures[task_id] for task_id in ordered])
        pairs = []
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                score = sum(a * b for a, b in zip(vectors[left], vectors[right]))
                row = {
                    "task_id_a": ordered[left],
                    "task_id_b": ordered[right],
                    "cosine_similarity": round(score, 6),
                }
                pairs.append(row)
                nearest[ordered[left]].append(
                    {"task_id": ordered[right], "cosine_similarity": row["cosine_similarity"]}
                )
                nearest[ordered[right]].append(
                    {"task_id": ordered[left], "cosine_similarity": row["cosine_similarity"]}
                )
        for task_id in nearest:
            nearest[task_id] = sorted(
                nearest[task_id],
                key=lambda item: (-item["cosine_similarity"], item["task_id"]),
            )[:3]
        embedding = {
            "status": "complete",
            "provider": runtime.config["embedding"]["provider"],
            "model": runtime.config["embedding"]["model"],
            "pairwise_similarities": sorted(
                pairs, key=lambda item: -item["cosine_similarity"]
            ),
        }
    except Exception as exc:
        embedding = {"status": "operational_failure", "error": str(exc)}
    instances = [
        {
            "task_id": item["task_id"],
            "signature": item["open_ended_signature"],
            "concrete_evidence": {
                key: value
                for key, value in item.items()
                if key not in {"task_id", "open_ended_signature"}
            },
            "nearest_signature_neighbors": nearest[item["task_id"]],
        }
        for item in positives
    ]
    audit_id = "episode-joint-diversity"

    def validate(value: dict[str, Any]) -> None:
        members = [
            task_id
            for cluster in value["clusters"]
            for task_id in cluster["member_task_ids"]
        ]
        expected = {item["task_id"] for item in positives}
        if len(members) != len(set(members)) or set(members) != expected:
            raise EpisodeJointError("diversity audit must cluster every positive exactly once")

    response = runtime.call(
        prompt_name="v43_episode_joint_diversity",
        call_id=audit_id,
        values={"INSTANCES": instances, "AUDIT_ID": audit_id},
        parent_ids=[item["task_id"] for item in positives],
        role="editorial_board",
        arm_id="blinded-episode-joint-diversity",
        expected={
            "audit_id": audit_id,
            "property": "episode_joint_inference",
        },
        validator=validate,
        forbidden_prompt_fragments=PRIVATE_LABELS,
    )
    summary = mechanism_diversity_summary(
        positive_task_ids=[item["task_id"] for item in positives],
        diversity_audit=response,
    )
    value = {"audit": response, "summary": summary, "semantic_embedding_analysis": embedding}
    atomic_write_json(path, value)
    return value


def _arm_quality(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = _task_set_metrics(records)
    count = len(records)
    hidden_matches = sum(
        bool(item.get("technical_results"))
        and bool(item["technical_results"][-1].get("hidden_output_match"))
        for item in records
    )
    below = sum(
        any(
            judgment.get("overall_quality") == "below_real_bar"
            for judgment in item.get("editorial_judgments", [])
        )
        for item in records
    )
    return {
        **metrics,
        "completed_task_count": count,
        "deterministic_validity_rate": metrics["deterministic_validity_rate"],
        "technical_pass_rate": metrics["technical_pass_rate"],
        "hidden_output_exact_match_rate": hidden_matches / count if count else 0.0,
        "serious_ambiguity_rate": metrics["serious_ambiguity_rate"],
        "core_quality_yield": metrics["core_yield"],
        "below_real_bar_count": below,
    }


def _human_pack(runtime: V41Runtime, judgments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    destination = runtime.run_dir / "human_review"
    tasks_dir = destination / "tasks"
    images_dir = destination / "images"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    records = [
        item
        for item in runtime.state["semantic_records"].values()
        if item.get("complete_task")
    ]
    rng = Random(int(runtime.config["seed"]) + 439931)
    rng.shuffle(records)
    audit_by_id = {item["task_id"]: item for item in judgments}
    public = []
    answer_key = []
    renders = []
    form = []
    for index, record in enumerate(records, 1):
        anonymous = "review-" + content_hash(
            {"seed": runtime.config["seed"], "task": record["task_id"]}
        )[:10]
        atomic_write_json(tasks_dir / f"{anonymous}.json", record["complete_task"])
        image = render_episode(
            record["complete_task"],
            title=f"Task {index}",
            subtitle="Source and author rule withheld",
            destination=images_dir / f"{anonymous}.png",
        )
        renders.append((f"Task {index}", image))
        public.append(
            {
                "review_index": index,
                "anonymous_id": anonymous,
                "task_json": f"tasks/{anonymous}.json",
                "image": f"images/{anonymous}.png",
            }
        )
        answer_key.append(
            {
                "review_index": index,
                "anonymous_id": anonymous,
                "task_id": record["task_id"],
                "private_arm": record["arm"],
                "slate_index": record["slate_index"],
                "mechanism_audit": audit_by_id.get(record["task_id"]),
                "intrinsic_quality": record.get("intrinsic_quality"),
                "portfolio_placement": record.get("portfolio_placement"),
                "intended_rule": record["realization"]["authoring_metadata"][
                    "intended_rule"
                ],
            }
        )
        form.append(
            {
                "review_index": index,
                "overall_preference_quality": None,
                "clear_conceptual_click": None,
                "fair_and_one_answer": None,
                "genuine_episode_joint_inference": None,
                "overcomplicated_or_overformalized": None,
                "feels_like_a_repeated_template": None,
                "notes": "",
            }
        )
    contact = build_contact_sheet(renders, destination / "review_sheet.png")
    atomic_write_json(
        destination / "manifest.json",
        {
            "single_instances_not_paired_variants": True,
            "source_and_arm_hidden": True,
            "rule_withheld": True,
            "task_count": len(public),
            "contact_sheet": "review_sheet.png",
            "tasks": public,
        },
    )
    atomic_write_json(destination / "review_form.json", form)
    atomic_write_json(destination / "DO_NOT_OPEN_FIRST.answer_key.json", answer_key)
    appendix = [
        "# Answer and rule appendix",
        "",
        "Open only after reviewing the blinded task sheet.",
        "",
    ]
    for item in answer_key:
        audit = item.get("mechanism_audit") or {}
        appendix.extend(
            [
                f"## Task {item['review_index']}",
                "",
                f"- Task ID: `{item['task_id']}`",
                f"- Source arm: `{item['private_arm']}`",
                f"- Mechanism audit: `{audit.get('status', 'not_audited')}` — {audit.get('reason', '')}",
                f"- Intrinsic / portfolio: `{item['intrinsic_quality']}` / `{item['portfolio_placement']}`",
                f"- Rule: {item['intended_rule']}",
                "",
            ]
        )
    atomic_write_text(destination / "answer_rule_appendix.md", "\n".join(appendix))
    return {
        "directory": str(destination),
        "task_count": len(public),
        "review_sheet": str(contact) if contact else None,
        "review_form": str(destination / "review_form.json"),
        "answer_appendix": str(destination / "answer_rule_appendix.md"),
        "human_results_available": False,
    }


def _progress(
    runtime: V41Runtime,
    *,
    stage: str,
    kernels: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    value = {
        "pipeline": V43_PIPELINE,
        "run_id": runtime.run_id,
        "completed": False,
        "current_safe_stage": stage,
        "generated_kernel_count": len(kernels),
        "control_kernel_count": sum(item["private_arm"] == "control" for item in kernels),
        "treatment_kernel_count": sum(item["private_arm"] == "treatment" for item in kernels),
        "selected_development_count": len(selected),
        "selected_control_count": sum(item["private_arm"] == "control" for item in selected),
        "selected_treatment_count": sum(item["private_arm"] == "treatment" for item in selected),
        "call_accounting": runtime.accounting(),
        "resume_command": (
            "python3 -m arcgen.lib v4-3-experiment --config "
            "config/v4_3_exp_episode_joint.yaml --through-phase complete"
        ),
        "production_promotion_occurred": False,
        "good_task_bank_mutated": False,
    }
    atomic_write_json(runtime.run_dir / "progress.json", value)
    return value


def _render_report(report: Mapping[str, Any]) -> str:
    funnel = report["funnel"]
    efficacy = report["efficacy"]
    quality = report["quality_noninferiority"]
    lines = [
        "# ARCGEN V4.3 — episode-joint inference experiment",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Recommendation: **{report['promotion_recommendation']}**",
        "",
        "This matched idea-level study branches from the checksummed V4.2 milestone. Control and treatment used the same real calibration descriptions, Sol/max settings, concept-led schema, committed-bank snapshot, downstream concrete planning, realization, technical verification, blind solving, and fixed-anchor boards. Only the treatment idea-search call received the abstract episode-joint requirement; source labels were hidden from selection and every downstream judge.",
        "",
        "## Funnel",
        "",
        f"- Generated kernels, control / treatment: {funnel['control_kernels']} / {funnel['treatment_kernels']}",
        f"- Selected for authoring, control / treatment: {funnel['selected_control']} / {funnel['selected_treatment']}",
        f"- Plans ready / aborted or failed: {funnel['plans_ready']} / {funnel['plans_not_ready']}",
        f"- Realized / technical pass / blind clear: {funnel['realized']} / {funnel['technical_pass']} / {funnel['blind_clear']}",
        f"- Core / reserve / ambiguous / other: {funnel['core']} / {funnel['reserve']} / {funnel['ambiguous']} / {funnel['other']}",
        "",
        "## Episode-joint efficacy",
        "",
        f"- Arm-blind audit counts by source: `{efficacy['audit_status_by_private_arm']}`",
        f"- Genuine treatment completions: {efficacy['genuine_treatment_count']}",
        f"- Genuine control completions: {efficacy['genuine_control_count']}",
        f"- Treatment planners retaining / dropping / aborting the hypothesis: `{efficacy['treatment_planner_status_counts']}`",
        "",
        "## Quality preservation",
        "",
        f"- Control: `{quality['control']}`",
        f"- Treatment: `{quality['treatment']}`",
        f"- Provisional model gates all pass: `{quality['all_provisional_model_gates_passed']}`",
        "- Human review is still required and no property label influenced core status.",
        "",
        "## Diversity",
        "",
        f"- Diversity summary: `{report['diversity']['summary']}`",
        "- Same-procedure siblings are not counted as separate positive mechanisms.",
        "",
        "## Calls and timing",
        "",
        f"- Experimental logical calls: {report['call_accounting']['experimental_calls']}",
        f"- Active model-minutes: {report['call_accounting']['active_model_minutes']}",
        f"- Wall-clock minutes: {report['timing']['wall_clock_minutes']}",
        "",
        "## Human review",
        "",
        f"- Blinded single-task sheet: `{report['human_review']['review_sheet']}`",
        f"- Review form: `{report['human_review']['review_form']}`",
        f"- Rule/arm appendix: `{report['human_review']['answer_appendix']}`",
        "",
        "## Task outcomes",
        "",
        "| Task | Slate | Audit | Technical | Blind | Intrinsic | Portfolio |",
        "|---|---:|---|---|---|---|---|",
    ]
    for task in report["tasks"]:
        lines.append(
            f"| `{task['task_id']}` | {task['slate_index']} | `{task['mechanism_audit']['status'] if task.get('mechanism_audit') else 'not_audited'}` | "
            f"`{task['technical_pass']}` | `{task['blind_clear']}` | `{task.get('intrinsic_quality') or task['terminal_status']}` | "
            f"`{task.get('portfolio_placement') or 'not_applicable'}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["promotion_reason"],
            "",
            "No V4.2 source/config/prompt, production reservoir, feature status, or good-task-bank entry was changed by this experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def run_episode_joint_experiment(
    config: dict[str, Any], *, through_phase: str = "complete"
) -> dict[str, Any]:
    if through_phase not in {"kernels", "selection", "complete"}:
        raise EpisodeJointError("through_phase must be kernels, selection, or complete")
    if config.get("v4_3_pipeline") != V43_PIPELINE or config.get("v4_3_design") != V43_DESIGN:
        raise EpisodeJointError("the isolated V4.3 episode-joint config is required")
    if config["run"].get("automatically_commit_generated_tasks") is not False:
        raise EpisodeJointError("experimental tasks cannot be committed automatically")

    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    start_path = run_dir / "experiment_start.json"
    if not start_path.exists():
        atomic_write_json(start_path, {"started_at_epoch_seconds": round(time.time(), 3)})
    protected_before = _protected_hashes(config)
    design = _freeze_design(config, run_dir)
    runtime = V41Runtime(config, design)
    _install_extensions(runtime)
    if runtime.state.get("completed") and (run_dir / "report.json").is_file():
        return read_json(run_dir / "report.json")

    kernels = _generate_kernels(runtime, design)
    if through_phase == "kernels":
        return _progress(runtime, stage="kernels_frozen", kernels=kernels)
    selected, slates = _select_kernels(runtime, kernels)
    _initialize_records(runtime, selected)
    if through_phase == "selection":
        return _progress(
            runtime, stage="development_sample_frozen", kernels=kernels, selected=selected
        )

    _plan_records(runtime)
    _realize_records(runtime)
    budgets = _budgets(runtime)
    _run_technical(runtime, budgets)
    _run_blind(runtime, budgets)
    _single_instance_boards(runtime)
    audit = _mechanism_audit(runtime)
    diversity = _diversity_audit(runtime, audit)
    human = _human_pack(runtime, audit)

    records = list(runtime.state["semantic_records"].values())
    records.sort(
        key=lambda item: (
            int(item["slate_index"]),
            int(item["selector_global_priority"]),
        )
    )
    by_arm = {
        arm: [item for item in records if item["arm"] == arm]
        for arm in ("control", "treatment")
    }
    arm_quality = {arm: _arm_quality(items) for arm, items in by_arm.items()}
    noninferiority = matched_noninferiority(
        control=arm_quality["control"], treatment=arm_quality["treatment"]
    )
    audit_by_id = {item["task_id"]: item for item in audit}
    audit_status = {
        arm: dict(
            sorted(
                Counter(
                    audit_by_id[item["task_id"]]["status"]
                    for item in items
                    if item["task_id"] in audit_by_id
                ).items()
            )
        )
        for arm, items in by_arm.items()
    }
    treatment_plan_status = Counter(
        item.get("planning_result", {}).get("property_status", "failed")
        for item in by_arm["treatment"]
    )
    status_counts = Counter(item.get("status", "unknown") for item in records)
    core = sum(item.get("intrinsic_quality") == "core_quality" for item in records)
    reserve = sum(item.get("intrinsic_quality") == "reserve_quality" for item in records)
    ambiguous = status_counts["ambiguous"]
    realized = sum(bool(item.get("complete_task")) for item in records)
    technical_pass = sum(
        bool(item.get("technical_results"))
        and item["technical_results"][-1].get("passed") is True
        for item in records
    )
    blind_clear = sum(item.get("status") in {"blind_pass", "editorial_complete"} for item in records)
    plans_ready = sum(item.get("concrete_plan", {}).get("status") == "ready" for item in records)
    accounting = runtime.accounting()
    protected_after = _protected_hashes(config)
    _assert_hashes_unchanged(protected_before, protected_after)
    baseline_after = verify_v42_baseline(
        config["_repository_root"], config["paths"]["v4_2_baseline_manifest"]
    )
    if baseline_after != config["_v4_2_baseline_verification"]:
        raise EpisodeJointError("frozen V4.2 verification changed during V4.3")
    started = read_json(start_path)["started_at_epoch_seconds"]
    completed = round(time.time(), 3)
    genuine_treatment = sum(
        item["status"] == "genuine"
        and runtime.state["semantic_records"][item["task_id"]]["arm"] == "treatment"
        for item in audit
    )
    genuine_control = sum(
        item["status"] == "genuine"
        and runtime.state["semantic_records"][item["task_id"]]["arm"] == "control"
        for item in audit
    )
    minimum = int(config["episode_joint"]["minimum_genuine_completed"])
    recommendation = "inconclusive"
    if genuine_treatment < minimum:
        promotion_reason = (
            f"Only {genuine_treatment} genuine treatment completions were independently "
            f"confirmed; at least {minimum} are required before promotion, and blinded "
            "human review is pending. Retain V4.2 and continue testing or reject based on the failure pattern."
        )
    else:
        promotion_reason = (
            "The model evidence reaches the minimum efficacy sample, but promotion remains "
            "inconclusive until the blinded human review is completed and the user explicitly authorizes it."
        )
    report = {
        "schema_version": 1,
        "pipeline": V43_PIPELINE,
        "run_id": runtime.run_id,
        "feature": "episode_joint_inference",
        "experimental_design": V43_DESIGN,
        "completed": True,
        "frozen_v42_baseline": deepcopy(config["_v4_2_baseline_verification"]),
        "funnel": {
            "independent_slates": int(config["episode_joint"]["independent_slates"]),
            "control_kernels": sum(item["private_arm"] == "control" for item in kernels),
            "treatment_kernels": sum(item["private_arm"] == "treatment" for item in kernels),
            "selected_control": len(by_arm["control"]),
            "selected_treatment": len(by_arm["treatment"]),
            "selection_supply_shortfalls": [item["supply_shortfall"] for item in slates],
            "plans_ready": plans_ready,
            "plans_not_ready": len(records) - plans_ready,
            "realized": realized,
            "technical_pass": technical_pass,
            "blind_clear": blind_clear,
            "core": core,
            "reserve": reserve,
            "ambiguous": ambiguous,
            "other": len(records) - core - reserve - ambiguous,
        },
        "efficacy": {
            "audit_status_by_private_arm": audit_status,
            "genuine_treatment_count": genuine_treatment,
            "genuine_control_count": genuine_control,
            "treatment_planner_status_counts": dict(sorted(treatment_plan_status.items())),
            "property_audit_non_gating": True,
            "ordinary_task_status_unaffected_by_property": True,
        },
        "quality_noninferiority": noninferiority,
        "coverage_retention": {
            "ordinary_control_task_count": len(by_arm["control"]),
            "ordinary_generator_prompt_unchanged": True,
            "palette_scheduler_inherited": True,
            "logical_leap_or_structural_hints_stacked": False,
            "acceptance_threshold_changed": False,
        },
        "diversity": diversity,
        "human_review": human,
        "call_accounting": accounting,
        "timing": {
            "started_at_epoch_seconds": started,
            "completed_at_epoch_seconds": completed,
            "wall_clock_minutes": round((completed - started) / 60.0, 3),
        },
        "tasks": [
            {
                "task_id": item["task_id"],
                "private_arm": item["arm"],
                "arm_hidden_downstream": True,
                "slate_index": item["slate_index"],
                "search_mode": item["search_mode"],
                "planner_property_status": item.get("planning_result", {}).get(
                    "property_status"
                ),
                "mechanism_audit": audit_by_id.get(item["task_id"]),
                "terminal_status": item.get("status"),
                "technical_pass": bool(item.get("technical_results"))
                and item["technical_results"][-1].get("passed") is True,
                "blind_clear": item.get("status") in {"blind_pass", "editorial_complete"},
                "intrinsic_quality": item.get("intrinsic_quality"),
                "portfolio_placement": item.get("portfolio_placement"),
                "editorial_judgments": item.get("editorial_judgments"),
                "grid_metrics": _task_metrics(item["complete_task"])
                if item.get("complete_task")
                else None,
                "task_json": str(
                    runtime.run_dir / "tasks" / item["arm"] / f"{item['task_id']}.arc.json"
                )
                if item.get("complete_task")
                else None,
                "plan_json": str(
                    runtime.run_dir / "plans" / item["arm"] / f"{item['task_id']}.json"
                ),
            }
            for item in records
        ],
        "promotion_recommendation": recommendation,
        "promotion_reason": promotion_reason,
        "production_promotion_occurred": False,
        "feature_manifest_mutated": False,
        "good_task_bank_mutated_by_experiment": False,
        "production_reservoir_mutated": False,
        "protected_production_state_unchanged": protected_before == protected_after,
        "unresolved_uncertainties": [
            "Blinded human review is pending.",
            "Three slates remain a minimum qualification sample rather than a production guarantee.",
            "Mechanism authenticity and diversity depend on completed grids, not kernel prose.",
        ],
    }
    atomic_write_json(
        runtime.run_dir / "promotion_recommendation.json",
        {
            "feature": "episode_joint_inference",
            "recommendation": recommendation,
            "production_promotion_occurred": False,
        },
    )
    atomic_write_json(
        runtime.run_dir / "quality_noninferiority_report.json", noninferiority
    )
    atomic_write_json(
        runtime.run_dir / "coverage_retention_report.json", report["coverage_retention"]
    )
    atomic_write_json(runtime.run_dir / "call_and_timing_ledger.json", accounting)
    return runtime.finish(report, _render_report(report))


def rebuild_v43_report(run: str | Path) -> dict[str, Any]:
    path = Path(run).resolve() / "report.json"
    if not path.is_file():
        raise EpisodeJointError(f"V4.3 report not found: {path}")
    report = read_json(path)
    if report.get("pipeline") != V43_PIPELINE:
        raise EpisodeJointError("run is not a V4.3 episode-joint experiment")
    atomic_write_text(path.parent / "report.md", _render_report(report))
    return report


def v43_human_review_status(run: str | Path) -> dict[str, Any]:
    directory = Path(run).resolve() / "human_review"
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise EpisodeJointError(f"V4.3 human review pack not found: {manifest}")
    value = read_json(manifest)
    return {
        "run": str(Path(run).resolve()),
        "task_count": value["task_count"],
        "review_sheet": str(directory / value["contact_sheet"]),
        "review_form": str(directory / "review_form.json"),
        "answer_appendix": str(directory / "answer_rule_appendix.md"),
        "source_and_arm_hidden": value["source_and_arm_hidden"],
    }


__all__ = [
    "EpisodeJointError",
    "STRICT_EXTENSION_SCHEMAS",
    "parse_v43_extension",
    "rebuild_v43_report",
    "run_episode_joint_experiment",
    "v43_human_review_status",
]

