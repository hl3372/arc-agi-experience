"""ARCGEN V4.2 selector-routed logical-leap pilot.

V4.2 branches from the exact frozen V4.1 palette + structural-transfer
milestone.  Selection remains quality-first.  At most three already-ranked
candidates enter one shared challenge budget: two receive an optional
logical-leap planning instruction and one exercises the frozen structural-
transfer behavior as an internal control.  At least five of eight candidates
receive ordinary planning.  Every candidate has one task instance, downstream
judges are route-blind, and the property audit is descriptive only.
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

from .lean_v3_rendering import build_contact_sheet, render_episode
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
from .v4_pipeline import V4ControlPipeline, v4_call_accounting
from .v4_schemas import STRICT_OUTPUT_SCHEMAS as V4_STRICT_OUTPUT_SCHEMAS
from .v41_experiments import V41ExperimentError, V41Runtime, _anonymous_id, _task_metrics
from .v41_integration import _assert_hashes_unchanged, _protected_hashes
from .v41_structural_transfer import (
    _ensure_structural_selection_window,
    _install_runtime_extensions as _install_structural_extensions,
    _task_set_metrics,
    validate_selector_nominations,
)
from .v41_supported_departure import (
    SlateEscalationBudget,
    _realize_records,
    _record_stage,
    _run_blind,
    _run_technical,
    _selection_sandbox_config,
)
from .v41_test_challenge import _board as _single_instance_boards
from .v42_config import V42_DESIGN, V42_PIPELINE, verify_v41_baseline


class LogicalLeapError(V41ExperimentError):
    pass


TEXT = {"type": "string", "minLength": 1}
MAYBE_TEXT = {"type": "string"}
BOOLEAN = {"type": "boolean"}


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": deepcopy(dict(properties)),
        "required": list(properties),
        "additionalProperties": False,
    }


LOGICAL_LEAP_CONTRACT = _object(
    {
        "training_established_principles": {
            "type": "array",
            "items": TEXT,
            "minItems": 1,
            # A sound episode can need several concise premises.  The first
            # real pilot produced five and was needlessly sent through parse
            # recovery even though the response was otherwise valid.
            "maxItems": 8,
        },
        "new_test_consequence": TEXT,
        "deduction_chain": {
            "type": "array",
            "items": TEXT,
            "minItems": 2,
            "maxItems": 6,
        },
        "why_not_directly_instantiated": TEXT,
        "why_direct_replay_is_insufficient": TEXT,
        "training_support": TEXT,
        "same_rule_no_new_semantics_audit": TEXT,
        "one_answer_justification": TEXT,
    }
)

LOGICAL_PLAN_SCHEMA = _object(
    {
        "candidate_id": TEXT,
        "decision": {"type": "string", "enum": ["use", "decline", "abort"]},
        "decision_reason": TEXT,
        "logical_leap_contract": {
            "anyOf": [deepcopy(LOGICAL_LEAP_CONTRACT), {"type": "null"}]
        },
        "concrete_plan": {
            "anyOf": [
                deepcopy(V4_STRICT_OUTPUT_SCHEMAS["v4_concrete_plan"]),
                {"type": "null"},
            ]
        },
    }
)

LOGICAL_AUDIT_ENTRY = _object(
    {
        "task_id": TEXT,
        "status": {
            "type": "string",
            "enum": [
                "genuine_logical_leap",
                "structural_transfer_only",
                "parametric_or_executive_extension",
                "absent",
                "unsupported_new_semantics",
            ],
        },
        "necessary_test_deduction": MAYBE_TEXT,
        "training_premises": {
            "type": "array",
            "items": TEXT,
            "minItems": 0,
            "maxItems": 8,
        },
        "why_not_direct_replay": MAYBE_TEXT,
        "derivation_to_output": MAYBE_TEXT,
        "same_rule": BOOLEAN,
        "one_answer_supported": BOOLEAN,
        "reason": TEXT,
        "open_ended_signature": MAYBE_TEXT,
    }
)

LOGICAL_AUDIT_SCHEMA = _object(
    {
        "audit_id": TEXT,
        "judgments": {
            "type": "array",
            "items": deepcopy(LOGICAL_AUDIT_ENTRY),
            "minItems": 1,
            "maxItems": 16,
        },
        "non_gating_confirmation": BOOLEAN,
    }
)

STRICT_EXTENSION_SCHEMAS = {
    "v42_logical_leap_plan": LOGICAL_PLAN_SCHEMA,
    "v42_logical_leap_audit": LOGICAL_AUDIT_SCHEMA,
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


def parse_logical_extension(
    stage: str, data: str | bytes | bytearray | Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in STRICT_EXTENSION_SCHEMAS:
        raise SchemaError("$.stage", f"unknown V4.2 extension stage {stage!r}")
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
    if stage == "v42_logical_leap_plan":
        decision = value["decision"]
        plan = value["concrete_plan"]
        contract = value["logical_leap_contract"]
        if decision == "abort":
            if plan is not None or contract is not None:
                raise SchemaError("$", "abort must return neither plan nor contract")
        elif plan is None:
            raise SchemaError("$.concrete_plan", "use or decline requires a plan")
        elif (contract is not None) != (decision == "use"):
            raise SchemaError(
                "$.logical_leap_contract",
                "contract is required exactly when the opportunity is used",
            )
    elif value["non_gating_confirmation"] is not True:
        raise SchemaError("$.non_gating_confirmation", "audit must confirm non-gating status")
    return value


def _install_v42_extensions(runtime: V41Runtime) -> None:
    root = Path(runtime.config["_repository_root"])
    specifications = {
        "logical_leap_planner_addendum": (
            "v4_2_logical_leap_planner_addendum.md",
            "v42_logical_leap_plan",
        ),
        "logical_leap_auditor": (
            "v4_2_logical_leap_auditor.md",
            "v42_logical_leap_audit",
        ),
    }
    destination = runtime.run_dir / "prompts" / "v42_extensions"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": V42_PIPELINE,
        "prompts": {},
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
            raise LogicalLeapError(f"V4.2 prompt changed during resume: {filename}")
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
        "arcgen.lib/v42_config.py",
        "arcgen.lib/v42_logical_leap.py",
        "config/v4_2_exp_logical_leap.yaml",
        "prompts/v4_2_challenge_selector.md",
    ):
        path = root / relative
        manifest.setdefault("implementation", {})[relative] = sha256(path.read_bytes()).hexdigest()
    path = runtime.run_dir / "v42_extension_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise LogicalLeapError("V4.2 extension changed during resume")
    if not path.exists():
        atomic_write_json(path, manifest)

    original = runtime.invoker.output_parser

    def parse(stage: str, data: Any) -> Any:
        if stage in STRICT_EXTENSION_SCHEMAS:
            return parse_logical_extension(stage, data)
        return original(stage, data)

    runtime.invoker.output_parser = parse
    runtime.invoker.output_schemas.update(
        {stage: _transport(schema) for stage, schema in STRICT_EXTENSION_SCHEMAS.items()}
    )
    runtime.invoker._write_schemas()


def _install_selector_prompt(
    selector: V4ControlPipeline, *, maximum_nominations: int
) -> dict[str, Any]:
    if maximum_nominations != 3:
        raise LogicalLeapError("the V4.2 selector prompt is qualified for three nominations")
    root = Path(selector.config["_repository_root"])
    filename = "v4_2_challenge_selector.md"
    text, version, digest = load_prompt(root / "prompts" / filename)
    expected = {
        "MAX_SELECTED",
        "CANDIDATES_WITH_REFERENCES",
        "ACCEPTED_BANK_DIGEST",
        "IN_FLIGHT_RESERVATIONS",
        "REJECTED_MEMORY_DIGEST",
        "SELECTOR_ID",
    }
    if set(PLACEHOLDER_RE.findall(text)) != expected:
        raise LogicalLeapError("V4.2 selector bindings differ from V4 global selector")
    selector.prompts.entries["global_selector"] = {
        "filename": filename,
        "text": text,
        "version": version,
        "hash": digest,
    }
    selector.config["selection"]["maximum_optional_development_suggestions"] = maximum_nominations
    extension = {
        "schema_version": 1,
        "filename": filename,
        "version": version,
        "sha256": digest,
        "quality_ranking_precedes_nomination": True,
        "maximum_nominations": maximum_nominations,
        "selector_stage_call_count_delta": 0,
    }
    snapshot = selector.run_dir / "prompts" / "extensions" / filename
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != text:
        raise LogicalLeapError("V4.2 selector prompt changed during resume")
    if not snapshot.exists():
        atomic_write_text(snapshot, text)
    atomic_write_json(selector.run_dir / "v42_selector_extension.json", extension)
    return extension


def assign_challenge_routes(
    candidate_ids: Sequence[str],
    nominated_ids: Sequence[str],
    *,
    seed: int,
    logical_limit: int = 2,
    structural_limit: int = 1,
) -> dict[str, str]:
    """Assign one private route without changing quality order.

    The selector only nominates candidates after ranking.  This deterministic
    assignment keeps intervention type out of selection, never stacks
    treatments, and leaves every unassigned candidate ordinary.
    """

    ordered = [str(item) for item in candidate_ids]
    if len(ordered) != len(set(ordered)):
        raise LogicalLeapError("candidate IDs must be unique")
    nominations = [str(item) for item in nominated_ids]
    if len(nominations) != len(set(nominations)) or not set(nominations) <= set(ordered):
        raise LogicalLeapError("nominations must be a unique subset of developed candidates")
    if logical_limit < 0 or structural_limit < 0:
        raise LogicalLeapError("route limits cannot be negative")
    shuffled = list(nominations)
    Random(int(seed) + 52042).shuffle(shuffled)
    logical_ids = set(shuffled[:logical_limit])
    structural_ids = set(shuffled[logical_limit : logical_limit + structural_limit])
    routes = {
        candidate_id: (
            "logical_leap"
            if candidate_id in logical_ids
            else "structural_transfer"
            if candidate_id in structural_ids
            else "ordinary"
        )
        for candidate_id in ordered
    }
    if logical_ids & structural_ids:
        raise LogicalLeapError("a candidate cannot receive both challenge treatments")
    return routes


def filter_replication_proposals(
    reservoir: Mapping[str, Any], excluded_proposal_ids: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    """Remove prior-slate IDs from an experimental reservoir copy only."""

    excluded = {str(item) for item in excluded_proposal_ids}
    value = deepcopy(dict(reservoir))
    proposals = value.get("proposals")
    if not isinstance(proposals, list):
        raise LogicalLeapError("replication reservoir has no proposal list")
    removed = sorted(
        str(item["proposal_id"])
        for item in proposals
        if isinstance(item, Mapping) and str(item.get("proposal_id")) in excluded
    )
    value["proposals"] = [
        item
        for item in proposals
        if not isinstance(item, Mapping)
        or str(item.get("proposal_id")) not in excluded
    ]
    return value, removed


def _apply_replication_exclusion(
    config: Mapping[str, Any], v4: Mapping[str, Any], run_dir: Path
) -> dict[str, Any] | None:
    """Make replication windows disjoint without creating novelty memory."""

    source_value = config["logical_leap"].get("replication_exclusion_manifest")
    if source_value is None:
        return None
    source = Path(str(source_value)).resolve()
    source_hash = sha256(source.read_bytes()).hexdigest()
    source_manifest = read_json(source)
    excluded = [str(item) for item in source_manifest["proposal_ids"]]
    ledger_path = run_dir / "replication_exclusion.json"
    private_path = Path(str(v4["paths"]["reservoir"]))

    if ledger_path.exists():
        ledger = read_json(ledger_path)
        if ledger.get("source_sha256") != source_hash:
            raise LogicalLeapError("replication exclusion changed during resume")
        remaining = {
            str(item.get("proposal_id"))
            for item in read_json(private_path).get("proposals", [])
        }
        if remaining & set(excluded):
            raise LogicalLeapError("excluded proposals reappeared during resume")
        return ledger

    reservoir = read_json(private_path)
    filtered, removed = filter_replication_proposals(reservoir, excluded)
    atomic_write_json(private_path, filtered)
    ledger = {
        "schema_version": 1,
        "purpose": "disjoint experimental replication window; not novelty memory",
        "production_reservoir_mutated": False,
        "source_path": str(source),
        "source_sha256": source_hash,
        "requested_exclusion_count": len(excluded),
        "removed_from_private_sandbox_count": len(removed),
        "removed_proposal_ids": removed,
        "missing_from_private_sandbox_count": len(set(excluded) - set(removed)),
    }
    atomic_write_json(ledger_path, ledger)
    return ledger


def _freeze_selected_sample(
    config: Mapping[str, Any], run_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = run_dir / "selected_kernel_manifest.json"
    if path.exists():
        value = read_json(path)
        return value["development_sample"], value["slate"]

    settings = config["logical_leap"]
    v4 = _selection_sandbox_config(config, run_dir, slate_index=1)
    replication_exclusion = _apply_replication_exclusion(config, v4, run_dir)
    selector = V4ControlPipeline(v4)
    supply = _ensure_structural_selection_window(
        config,
        v4,
        run_dir,
        accepted_concept_hashes=selector.context.accepted_concept_hashes(),
    )
    extension = _install_selector_prompt(
        selector,
        maximum_nominations=int(settings["maximum_challenge_nominations_per_slate"]),
    )
    briefs = selector.select_slate()
    if replication_exclusion is not None:
        frozen_ids = set(selector.state["frozen_window"]["proposal_ids"])
        excluded_ids = set(replication_exclusion["removed_proposal_ids"])
        if frozen_ids & excluded_ids:
            raise LogicalLeapError("replication window reused an excluded proposal")
    ordered = sorted(briefs, key=lambda item: int(item["development_priority"]))
    maximum = int(settings["maximum_development_candidates"])
    if not ordered:
        raise LogicalLeapError("quality-first selector returned no developable candidates")
    nominations = validate_selector_nominations(
        ordered,
        maximum_nominations=int(settings["maximum_challenge_nominations_per_slate"]),
        maximum_development_priority=maximum,
    )
    developed_ids = [str(item["candidate_id"]) for item in ordered[:maximum]]
    routes = assign_challenge_routes(
        developed_ids,
        [item for item in nominations if item in developed_ids],
        seed=int(config["seed"]),
        logical_limit=int(settings["logical_leap_invitations_per_slate"]),
        structural_limit=int(settings["structural_transfer_controls_per_slate"]),
    )
    if sum(route == "ordinary" for route in routes.values()) < int(
        settings["ordinary_planning_minimum"]
    ):
        raise LogicalLeapError("V4.2 route assignment did not preserve the ordinary majority")

    sample = []
    for brief in ordered[:maximum]:
        candidate_id = str(brief["candidate_id"])
        record = selector.state["records"][candidate_id]
        sample.append(
            {
                "source_candidate_id": candidate_id,
                "slate_index": 1,
                "source_development_priority": int(record["development_priority"]),
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
                },
                "editorial_brief": deepcopy(record["editorial_brief"]),
                "private_route": routes[candidate_id],
            }
        )
    slate = {
        "slate_index": 1,
        "selection_run_id": v4["run"]["id"],
        "frozen_window": deepcopy(selector.state["frozen_window"]),
        "full_selected_slate": [item["candidate_id"] for item in ordered],
        "development_sample": developed_ids,
        "nominated_candidate_ids": nominations,
        "route_counts": dict(sorted(Counter(routes.values()).items())),
        "private_routes": routes,
        "supply": supply,
        "state_path": str(selector.state_path),
        "selector_extension": extension,
        "quality_selection_and_ranking_preceded_nomination": True,
        "route_assignment_changed_quality_order": False,
        "separate_routing_model_call": False,
        "replication_exclusion": deepcopy(replication_exclusion),
    }
    value = {
        "schema_version": 1,
        "pipeline": V42_PIPELINE,
        "experimental_design": V42_DESIGN,
        "single_instance_only": True,
        "paired_task_variants": False,
        "development_sample_count": len(sample),
        "development_sample": sample,
        "slate": slate,
    }
    atomic_write_json(path, value)
    return sample, slate


def _aliases(selected: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    rng = Random(seed + 52421)
    used: set[str] = set()
    source_ids = [str(item["source_candidate_id"]) for item in selected]
    rng.shuffle(source_ids)
    return {source_id: _anonymous_id(rng, used) for source_id in source_ids}


def _source_manifest(
    config: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    slate: Mapping[str, Any],
    aliases: Mapping[str, str],
    protected: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": V42_PIPELINE,
        "experimental_design": V42_DESIGN,
        "branch_source": deepcopy(config["_v4_1_baseline_verification"]),
        "frozen_v4_parent": deepcopy(config["_v4_parent_verification"]),
        "feature_manifest_snapshot": deepcopy(config["_feature_manifest_snapshot"]),
        "effective_accepted_baseline": deepcopy(config["_effective_features"]),
        "protected_production_state": deepcopy(dict(protected)),
        "model_and_effort_settings": deepcopy(config["model"]["roles"]),
        "selected_kernels": deepcopy(list(selected)),
        "slate": deepcopy(dict(slate)),
        "opaque_task_mapping": [
            {
                "source_candidate_id": source_id,
                "opaque_task_id": alias,
                "downstream_prompts_receive_source_or_route_label": False,
            }
            for source_id, alias in sorted(aliases.items())
        ],
        "single_instance_only": True,
        "paired_task_variants": False,
        "quality_ranking_precedes_nomination": True,
        "routing_call_count": 0,
        "planner_call_count_delta": 0,
        "property_audit_is_non_gating": True,
    }


def _initialize_records(
    runtime: V41Runtime,
    selected: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str],
) -> None:
    with runtime.lock:
        if "semantic_records" in runtime.state:
            return
        records = {}
        for source in selected:
            source_id = str(source["source_candidate_id"])
            task_id = aliases[source_id]
            brief = deepcopy(source["editorial_brief"])
            route = str(source["private_route"])
            records[task_id] = {
                "task_id": task_id,
                "source_candidate_id": source_id,
                "arm": "single",
                "private_route": route,
                "slate_index": 1,
                "source_development_priority": int(source["source_development_priority"]),
                "search_mode": source["kernel"]["search_mode"],
                "kernel": deepcopy(source["kernel"]),
                "editorial_brief": brief,
                "challenge_nominated": route != "ordinary",
                "selector_suggestion": brief.get("optional_development_suggestion"),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
        runtime.state["semantic_records"] = records
        runtime.state["v42_pipeline"] = V42_PIPELINE
        runtime.save()


PRIVATE_LABELS = (
    "logical leap",
    "logical-leap",
    "structural transfer",
    "structural-transfer",
    "experimental wrapper",
    "experimental nomination",
    "treatment arm",
    "private route",
)


def _private_label_leaks(value: Any) -> list[str]:
    text = json.dumps(value, ensure_ascii=False).casefold()
    return [label for label in PRIVATE_LABELS if label in text]


def validate_v42_plan_response(
    *, task_id: str, route: str, value: Mapping[str, Any]
) -> None:
    if route not in {"ordinary", "structural_transfer", "logical_leap"}:
        raise LogicalLeapError(f"unknown private route {route!r}")
    if value.get("candidate_id") != task_id:
        raise LogicalLeapError("planner returned a mismatched candidate ID")
    decision = value.get("decision")
    if route == "ordinary" and decision != "not_invited":
        raise LogicalLeapError("ordinary planning acquired a challenge decision")
    plan = value.get("concrete_plan")
    if plan is not None and plan.get("candidate_id") != task_id:
        raise LogicalLeapError("nested ordinary plan returned a mismatched candidate ID")
    if plan is not None:
        leaks = _private_label_leaks(plan)
        if leaks:
            raise LogicalLeapError(f"ordinary downstream plan contains private labels: {leaks}")
    if route == "logical_leap":
        contract = value.get("logical_leap_contract")
        if (contract is not None) != (decision == "use"):
            raise LogicalLeapError("logical contract is required exactly when used")
    if route == "structural_transfer":
        contract = value.get("structural_transfer_contract")
        if (contract is not None) != (decision == "use"):
            raise LogicalLeapError("structural contract is required exactly when used")


def _plan_records(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]

    def one(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        record = deepcopy(records[task_id])
        brief = deepcopy(record["editorial_brief"])
        if record["private_route"] == "ordinary":
            brief["optional_development_suggestion"] = None
        values = {
            "CANDIDATE": {**deepcopy(record["kernel"]), "candidate_id": task_id},
            "EDITORIAL_BRIEF": {**brief, "candidate_id": task_id},
            "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(brief["route"]),
            "CANDIDATE_ID": task_id,
        }
        try:
            route = record["private_route"]
            if route == "ordinary":
                plan = runtime.call(
                    prompt_name="concrete_planner",
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="single-authoring",
                    expected={"candidate_id": task_id},
                )
                result = {
                    "candidate_id": task_id,
                    "decision": "not_invited",
                    "decision_reason": "ordinary frozen V4.1 planning",
                    "logical_leap_contract": None,
                    "structural_transfer_contract": None,
                    "concrete_plan": plan,
                }
            elif route == "logical_leap":
                result = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("logical_leap_planner_addendum",),
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="single-authoring",
                    expected={"candidate_id": task_id},
                )
                result["structural_transfer_contract"] = None
            else:
                result = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("structural_planner_addendum",),
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="single-authoring",
                    expected={"candidate_id": task_id},
                )
                result["logical_leap_contract"] = None
            validate_v42_plan_response(task_id=task_id, route=route, value=result)
            nested = result.get("concrete_plan")
            if nested is not None and nested.get("status") == "abort" and result["decision"] != "abort":
                raise LogicalLeapError("an aborted nested plan must use the abort decision")
            return task_id, result, None
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
        status = "plan_aborted" if plan is None or plan.get("status") == "abort" else "plan_ready"
        _record_stage(
            runtime,
            task_id,
            "planning",
            status=status,
            planning_result=result,
            concrete_plan=plan,
        )
        destination = runtime.run_dir / "plans" / "single"
        destination.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination / f"{task_id}.json", result)


def _audit(runtime: V41Runtime) -> list[dict[str, Any]]:
    records = runtime.state["semantic_records"]
    completed = [record for record in records.values() if record.get("complete_task")]
    rng = Random(int(runtime.config["seed"]) + 52883)
    rng.shuffle(completed)
    batch_size = int(runtime.config["logical_leap"]["audit_batch_size"])
    judgments: list[dict[str, Any]] = []
    calls = []
    for offset in range(0, len(completed), batch_size):
        chunk = completed[offset : offset + batch_size]
        audit_id = f"logical-leap-audit-{offset // batch_size + 1:02d}"
        packages = [
            {
                "task_id": record["task_id"],
                "episode": record["complete_task"],
                "concise_rule": record["realization"]["authoring_metadata"]["intended_rule"],
                "concrete_plan": record["concrete_plan"],
            }
            for record in chunk
        ]
        if any(_private_label_leaks(package) for package in packages):
            raise LogicalLeapError("private routing labels entered the arm-blind audit package")

        def validate(value: dict[str, Any]) -> None:
            returned = [item["task_id"] for item in value["judgments"]]
            expected = {item["task_id"] for item in packages}
            if len(returned) != len(set(returned)) or set(returned) != expected:
                raise LogicalLeapError("auditor did not cover every task exactly once")

        response = runtime.call(
            prompt_name="logical_leap_auditor",
            call_id=audit_id,
            values={"AUDIT_ID": audit_id, "TASK_PACKAGES": packages},
            parent_ids=[item["task_id"] for item in packages],
            role="editorial_board",
            arm_id="blinded-logical-leap-audit",
            expected={"audit_id": audit_id},
            validator=validate,
        )
        calls.append(response)
        judgments.extend(response["judgments"])
    atomic_write_json(
        runtime.run_dir / "logical_leap_audit.json",
        {
            "route_hidden": True,
            "source_hidden": True,
            "non_gating": True,
            "calls": calls,
            "judgments": judgments,
        },
    )
    return judgments


def _human_pack(
    runtime: V41Runtime, judgments: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    destination = runtime.run_dir / "human_review"
    task_dir = destination / "tasks"
    image_dir = destination / "images"
    explained_dir = runtime.run_dir / "inspection" / "renders_with_rules"
    task_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    explained_dir.mkdir(parents=True, exist_ok=True)
    by_task = {item["task_id"]: item for item in judgments}
    records = [
        record
        for record in runtime.state["semantic_records"].values()
        if record.get("complete_task")
    ]
    rng = Random(int(runtime.config["seed"]) + 52991)
    rng.shuffle(records)
    used: set[str] = set()
    public = []
    answer_key = []
    review_form = []
    blind_renders = []
    explained_renders = []
    for index, record in enumerate(records, 1):
        anonymous = _anonymous_id(rng, used)
        atomic_write_json(task_dir / f"{anonymous}.json", record["complete_task"])
        blind = render_episode(
            record["complete_task"],
            title=f"Task {index}",
            subtitle="Source, route, audit label, and intended rule withheld",
            destination=image_dir / f"{anonymous}.png",
        )
        rule = record["realization"]["authoring_metadata"]["intended_rule"]
        explained = render_episode(
            record["complete_task"],
            title=f"{record['task_id']} — {record.get('intrinsic_quality') or record.get('status')}",
            subtitle=rule,
            destination=explained_dir / f"{record['task_id']}.png",
        )
        blind_renders.append((f"Task {index}", blind))
        explained_renders.append((record["task_id"], explained))
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
                "source_candidate_id": record["source_candidate_id"],
                "search_mode": record["search_mode"],
                "private_route": record["private_route"],
                "planner_decision": record.get("planning_result", {}).get("decision"),
                "audit": deepcopy(by_task.get(record["task_id"])),
                "intended_rule": rule,
                "intrinsic_quality": record.get("intrinsic_quality"),
                "portfolio_placement": record.get("portfolio_placement"),
            }
        )
        review_form.append(
            {
                "review_index": index,
                "overall_quality": None,
                "clear_conceptual_click": None,
                "fair_and_one_answer": None,
                "test_requires_a_genuine_new_deduction": None,
                "test_is_structural_transfer_only": None,
                "test_is_mainly_parametric_or_execution": None,
                "test_uses_unsupported_new_semantics": None,
                "overcomplicated": None,
                "notes": "",
            }
        )
    contact = build_contact_sheet(blind_renders, destination / "review_sheet.png")
    explained_contact = build_contact_sheet(
        explained_renders, runtime.run_dir / "inspection" / "contact_sheet_with_rules.png"
    )
    atomic_write_json(
        destination / "manifest.json",
        {
            "single_instance_only": True,
            "paired_task_variants": False,
            "source_and_route_hidden": True,
            "rule_withheld": True,
            "task_count": len(public),
            "contact_sheet": "review_sheet.png",
            "tasks": public,
        },
    )
    atomic_write_json(destination / "review_form.json", review_form)
    atomic_write_json(destination / "DO_NOT_OPEN_FIRST.answer_key.json", answer_key)
    appendix = [
        "# Answer and rule appendix",
        "",
        "Open only after reviewing the single-instance task sheet.",
        "",
    ]
    for item in answer_key:
        audit = item.get("audit") or {}
        appendix.extend(
            [
                f"## Task {item['review_index']}",
                "",
                f"- Task ID: `{item['task_id']}`",
                f"- Private route: `{item['private_route']}`",
                f"- Planner decision: `{item['planner_decision']}`",
                f"- Audit: `{audit.get('status', 'not_audited')}` — {audit.get('reason', '')}",
                f"- Intrinsic / portfolio: `{item['intrinsic_quality']}` / `{item['portfolio_placement']}`",
                f"- Rule: {item['intended_rule']}",
                "",
            ]
        )
    atomic_write_text(destination / "answer_rule_appendix.md", "\n".join(appendix))
    return {
        "directory": str(destination),
        "task_count": len(public),
        "single_instance_only": True,
        "contact_sheet": str(contact) if contact else None,
        "contact_sheet_with_rules": str(explained_contact) if explained_contact else None,
        "human_results_available": False,
    }


def _open_ended_diversity(
    runtime: V41Runtime, judgments: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    positive = [
        item for item in judgments if item["status"] == "genuine_logical_leap"
    ]
    result: dict[str, Any] = {
        "genuine_instance_count": len(positive),
        "minimum_before_promotion_diversity_audit": 8,
        "minimum_independent_slates_before_promotion": 3,
        "model_cluster_call_run": False,
        "signatures": {
            item["task_id"]: item["open_ended_signature"] for item in positive
        },
    }
    if len(positive) < 2:
        result["semantic_embedding_analysis"] = {
            "status": "not_applicable",
            "reason": "fewer than two genuine instances",
        }
    else:
        try:
            from .embeddings import build_embedding_provider

            ordered = sorted(positive, key=lambda item: item["task_id"])
            embedder = build_embedding_provider(
                runtime.config["embedding"], runtime.config["paths"]["cache"]
            )
            vectors = embedder.embed(
                [item["open_ended_signature"] for item in ordered]
            )
            pairs = []
            for left in range(len(ordered)):
                for right in range(left + 1, len(ordered)):
                    similarity = sum(
                        a * b for a, b in zip(vectors[left], vectors[right])
                    )
                    pairs.append(
                        {
                            "task_id_a": ordered[left]["task_id"],
                            "task_id_b": ordered[right]["task_id"],
                            "cosine_similarity": round(similarity, 6),
                        }
                    )
            result["semantic_embedding_analysis"] = {
                "status": "complete",
                "provider": runtime.config["embedding"]["provider"],
                "model": runtime.config["embedding"]["model"],
                "pairwise_similarities": sorted(
                    pairs, key=lambda item: -item["cosine_similarity"]
                ),
            }
        except Exception as exc:
            result["semantic_embedding_analysis"] = {
                "status": "operational_failure",
                "error": str(exc),
            }
    result["conclusion"] = (
        "inconclusive until at least eight genuine completed instances across at least three slates are available"
    )
    atomic_write_json(runtime.run_dir / "diversity_cluster_report.json", result)
    return result


def _call_accounting(
    runtime: V41Runtime, slate: Mapping[str, Any]
) -> dict[str, Any]:
    runtime_accounting = runtime.accounting()
    ledger = runtime.state["call_ledger"]
    audit_calls = sum(
        row.get("arm_id") == "blinded-logical-leap-audit" for row in ledger.values()
    )
    selector_state = read_json(Path(slate["state_path"]))
    selector_accounting = v4_call_accounting(
        selector_state["call_ledger"], run_dir=Path(slate["state_path"]).parent
    )
    supply_rows = []
    for raw_path in slate.get("supply", {}).get("call_ledger_paths", []):
        path = Path(raw_path)
        if not path.is_file():
            continue
        state = read_json(path)
        supply_rows.append(
            v4_call_accounting(state.get("call_ledger", {}), run_dir=path.parent)
        )
    supply_calls = sum(item["substantive_logical_calls"] for item in supply_rows)
    supply_minutes = sum(item["active_model_minutes"] for item in supply_rows)
    supply_retries = sum(item["transport_retry_calls"] for item in supply_rows)
    selector_calls = selector_accounting["substantive_logical_calls"]
    downstream_calls = runtime_accounting["experimental_calls"]
    return {
        "selector_and_window_calls": selector_calls,
        "reservoir_supply_calls": supply_calls,
        "downstream_authoring_and_review_calls": downstream_calls - audit_calls,
        "non_gating_experimental_audit_calls": audit_calls,
        "separate_routing_calls": 0,
        "extra_planner_calls": 0,
        "normal_path_call_delta_for_routing": 0,
        "total_experimental_logical_calls_including_audit": selector_calls + downstream_calls,
        "total_attributed_calls_including_off_path_supply": selector_calls + supply_calls + downstream_calls,
        "active_model_minutes_excluding_supply": round(
            selector_accounting["active_model_minutes"]
            + runtime_accounting["active_model_minutes"],
            6,
        ),
        "active_model_minutes_including_supply": round(
            selector_accounting["active_model_minutes"]
            + supply_minutes
            + runtime_accounting["active_model_minutes"],
            6,
        ),
        "transport_retry_calls": (
            selector_accounting["transport_retry_calls"]
            + supply_retries
            + runtime_accounting["transport_retry_calls"]
        ),
        "runtime": runtime_accounting,
        "selector": selector_accounting,
        "reservoir_supply_cycles": supply_rows,
    }


def _render_report(report: Mapping[str, Any]) -> str:
    funnel = report["funnel"]
    efficacy = report["efficacy"]
    calls = report["call_accounting"]
    route_quality = report["quality"]["by_private_route"]
    lines = [
        "# ARCGEN V4.2 — optional logical-leap pilot",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Recommendation: **{report['promotion_recommendation']}**",
        (
            "Replication window: prior-slate proposal IDs were removed from the private experimental sandbox only; production memory was unchanged."
            if report.get("replication_exclusion")
            else "Replication window: no prior-slate exclusion was configured."
        ),
        "",
        "This single-slate experiment branches from the exact frozen V4.1 milestone: accepted global palette scheduling plus quality-first optional structural transfer. The ordinary open, concept-led, and composition-led idea producers were unchanged. The selector first selected and ranked the strongest slate, then could nominate up to three of the top eight for a generic test-challenge opportunity.",
        "",
        "After ranking was frozen, deterministic private routing assigned at most two nominated candidates to the new optional logical-leap planner instruction and at most one to the frozen V4.1 structural-transfer instruction. All remaining candidates used ordinary planning. No candidate received both, no paired variant was authored, and downstream technical, blind, and editorial prompts received neither route nor treatment labels.",
        "",
        "## Funnel",
        "",
        f"- Frozen eligible proposals: {funnel['eligible_window']}",
        f"- Globally selected / developed: {funnel['selected']} / {funnel['developed']}",
        f"- Route allocation (private during review): `{funnel['route_counts']}`",
        f"- Planner uses / declines / ordinary / aborts: {funnel['planner_use']} / {funnel['planner_decline']} / {funnel['ordinary']} / {funnel['plan_abort']}",
        f"- Realized / technical pass / blind clear: {funnel['realized']} / {funnel['technical_pass']} / {funnel['blind_pass']}",
        f"- Core / reserve / ambiguous / other: {funnel['core']} / {funnel['reserve']} / {funnel['ambiguous']} / {funnel['other']}",
        "",
        "## Efficacy",
        "",
        f"- Arm-blind audit counts: `{efficacy['audit_status_counts']}`",
        f"- Logical invitations used: {efficacy['logical_planner_use_count']}/{efficacy['logical_invitation_count']}",
        f"- Logical invitations producing a genuine completed leap: {efficacy['logical_invited_and_genuine']}/{efficacy['logical_invitation_count']}",
        f"- Genuine leaps arising outside the logical route: {efficacy['nonlogical_genuine_count']}",
        f"- Unsupported-new-semantics findings: {efficacy['unsupported_count']}",
        "",
        "The property audit is deliberately non-gating. It distinguishes a new deduction from structural transfer, parameter change, and execution burden. Ordinary correctness and quality decisions came only from the unchanged V4.1 technical, blind, and two-board path.",
        "",
        "## Quality by route",
        "",
        f"- Logical-leap invitation: `{route_quality['logical_leap']}`",
        f"- Structural-transfer control: `{route_quality['structural_transfer']}`",
        f"- Ordinary planning: `{route_quality['ordinary']}`",
        f"- Whole pilot: `{report['quality']['all_tasks']}`",
        f"- Frozen V4 qualification reference: `{report['quality']['v4_qualification_reference']}`",
        "",
        "These are descriptive same-slate subsets, not matched counterfactual task pairs. The sample is too small for a non-inferiority or mechanism-diversity claim.",
        "",
        "## Calls and timing",
        "",
        f"- Separate routing calls: **{calls['separate_routing_calls']}**",
        f"- Extra planner calls: **{calls['extra_planner_calls']}**",
        f"- Normal-path call delta for the treatment: **{calls['normal_path_call_delta_for_routing']}**",
        f"- Non-gating experimental audit calls: {calls['non_gating_experimental_audit_calls']}",
        f"- Selector/window + downstream calls, including audit: {calls['total_experimental_logical_calls_including_audit']}",
        f"- Off-critical-path reservoir-supply calls: {calls['reservoir_supply_calls']}",
        f"- Active model-minutes excluding / including supply: {calls['active_model_minutes_excluding_supply']} / {calls['active_model_minutes_including_supply']}",
        f"- Wall-clock minutes: {report['timing']['wall_clock_minutes']}",
        "",
        "The production-shaped treatment itself adds no standing call: nomination uses the existing selector response and use/decline uses the existing planner call. This experiment adds one batched auditor call only for measurement.",
        "",
        "## Distribution safeguards",
        "",
        f"- Ordinary planning share in this pilot: {report['coverage_retention']['ordinary_planning_share']:.1%}",
        "- Future integration, if ever authorized, begins at no more than one logical-leap invitation per eight planned tasks.",
        "- Rolling desired core coverage is 10–20%, with a 25% soft ceiling; future invitation supply changes rather than acceptance decisions.",
        "- Open, concept-led, and composition-led generation remains unchanged and dominant.",
        "",
        "## Diversity and human review",
        "",
        f"- Diversity conclusion: {report['diversity']['conclusion']}.",
        f"- Blind review sheet: `{report['human_review']['contact_sheet']}`",
        f"- Inspection sheet with rules: `{report['human_review']['contact_sheet_with_rules']}`",
        "- Human review is pending.",
        "",
        "## Task outcomes",
        "",
        "| Task | Mode | Private route | Planner | Audit | Intrinsic | Portfolio |",
        "|---|---|---|---|---|---|---|",
    ]
    for task in report["tasks"]:
        lines.append(
            f"| `{task['task_id']}` | `{task['search_mode']}` | `{task['private_route']}` | "
            f"`{task['planner_decision']}` | `{(task.get('audit') or {}).get('status', 'not_audited')}` | "
            f"`{task.get('intrinsic_quality') or task['terminal_status']}` | "
            f"`{task.get('portfolio_placement') or 'not_applicable'}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Recommendation: **{report['promotion_recommendation']}**.",
            "",
            report["promotion_reason"],
            "",
            "No production promotion, V4.1 baseline mutation, feature-manifest change, reservoir mutation, or good-task-bank commit occurred. No continuing campaign or later semantic experiment was started.",
            "",
        ]
    )
    return "\n".join(lines)


def run_logical_leap_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("v4_2_pipeline") != V42_PIPELINE:
        raise LogicalLeapError("the V4.2 logical-leap configuration is required")
    if config.get("experimental", {}).get("design") != V42_DESIGN:
        raise LogicalLeapError("the V4.2 experimental design differs from the qualified pilot")
    if config["experimental"].get("feature") != "supported_departure":
        raise LogicalLeapError("V4.2 must remain in the isolated, unpromoted semantic slot")
    if not config.get("_effective_features", {}).get("palette_scheduler"):
        raise LogicalLeapError("the accepted V4.1 palette baseline is missing")
    settings = config.get("logical_leap")
    if not isinstance(settings, Mapping):
        raise LogicalLeapError("logical_leap configuration is missing")
    if settings.get("paired_task_variants") is not False:
        raise LogicalLeapError("V4.2 must create one task instance per kernel")

    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    started_path = run_dir / "experiment_start.json"
    if not started_path.exists():
        atomic_write_json(started_path, {"started_at_epoch_seconds": round(time.time(), 3)})

    protected_before = _protected_hashes(config)
    selected, slate = _freeze_selected_sample(config, run_dir)
    aliases = _aliases(selected, int(config["seed"]))
    frozen = _source_manifest(config, selected, slate, aliases, protected_before)
    runtime = V41Runtime(config, frozen)
    _install_structural_extensions(runtime)
    _install_v42_extensions(runtime)
    if runtime.state.get("completed") and (runtime.run_dir / "report.json").is_file():
        return read_json(runtime.run_dir / "report.json")
    _initialize_records(runtime, selected, aliases)
    _plan_records(runtime)
    _realize_records(runtime)

    with runtime.lock:
        runtime.state.setdefault("escalation_usage_by_slate", {"1": 0})
        runtime.save()

    def persist(used: int) -> None:
        with runtime.lock:
            runtime.state["escalation_usage_by_slate"]["1"] = used
            runtime.save()

    budget = SlateEscalationBudget(
        int(settings["second_opinion_budget_per_slate"]),
        used=int(runtime.state["escalation_usage_by_slate"].get("1", 0)),
        on_change=persist,
    )
    budgets = {(1, "single"): budget}
    _run_technical(runtime, budgets)
    _run_blind(runtime, budgets)
    _single_instance_boards(runtime)
    audit = _audit(runtime)
    diversity = _open_ended_diversity(runtime, audit)
    human = _human_pack(runtime, audit)

    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda item: int(item["source_development_priority"]),
    )
    audit_by_id = {item["task_id"]: item for item in audit}
    route_groups = {
        route: [item for item in records if item["private_route"] == route]
        for route in ("logical_leap", "structural_transfer", "ordinary")
    }
    decisions = Counter(
        item.get("planning_result", {}).get("decision", "failed") for item in records
    )
    statuses = Counter(item.get("status", "unknown") for item in records)
    audit_counts = Counter(item["status"] for item in audit)
    genuine_ids = {
        item["task_id"] for item in audit if item["status"] == "genuine_logical_leap"
    }
    logical_ids = {item["task_id"] for item in route_groups["logical_leap"]}
    logical_use_ids = {
        item["task_id"]
        for item in route_groups["logical_leap"]
        if item.get("planning_result", {}).get("decision") == "use"
    }
    all_metrics = _task_set_metrics(records)
    qualification = read_json(Path(config["paths"]["frozen_v4_qualification"]) / "report.json")
    qualification_reference = {
        "selected": qualification["funnel"]["selected_and_reserved"],
        "realized": qualification["funnel"]["realized"],
        "technical_pass": qualification["funnel"]["technical_pass"],
        "blind_cleared": qualification["funnel"]["blind_cleared"],
        "core": qualification["funnel"]["intrinsic_quality"].get("core_quality", 0),
        "reserve": qualification["funnel"]["intrinsic_quality"].get("reserve_quality", 0),
        "note": "descriptive frozen V4 qualification; not a newly authored matched control",
    }
    protected_after = _protected_hashes(config)
    _assert_hashes_unchanged(protected_before, protected_after)
    baseline_after = verify_v41_baseline(
        config["_repository_root"], config["paths"]["v4_1_baseline_manifest"]
    )
    if baseline_after != config["_v4_1_baseline_verification"]:
        raise LogicalLeapError("frozen V4.1 baseline verification changed during the pilot")
    accounting = _call_accounting(runtime, slate)
    started = read_json(started_path)["started_at_epoch_seconds"]
    completed = round(time.time(), 3)
    developed = len(records)
    core = all_metrics["core_count"]
    reserve = all_metrics["reserve_count"]
    ambiguous = statuses["ambiguous"]
    report = {
        "schema_version": 1,
        "pipeline": V42_PIPELINE,
        "run_id": runtime.run_id,
        "feature": "logical_leap_planning",
        "experimental_design": V42_DESIGN,
        "completed": True,
        "single_instance_only": True,
        "paired_task_variants": False,
        "frozen_v41_baseline": deepcopy(config["_v4_1_baseline_verification"]),
        "replication_exclusion": deepcopy(slate.get("replication_exclusion")),
        "funnel": {
            "eligible_window": len(slate["frozen_window"]["proposal_ids"]),
            "selected": len(slate["full_selected_slate"]),
            "developed": developed,
            "route_counts": {route: len(rows) for route, rows in route_groups.items()},
            "planner_use": decisions["use"],
            "planner_decline": decisions["decline"],
            "ordinary": decisions["not_invited"],
            "plan_abort": decisions["abort"] + statuses["plan_aborted"],
            "realized": all_metrics["realized_count"],
            "technical_pass": all_metrics["technical_pass_count"],
            "blind_pass": sum(
                item.get("status") in {"blind_pass", "editorial_complete"}
                for item in records
            ),
            "core": core,
            "reserve": reserve,
            "ambiguous": ambiguous,
            "other": developed - core - reserve - ambiguous,
        },
        "efficacy": {
            "logical_invitation_count": len(logical_ids),
            "logical_planner_use_count": len(logical_use_ids),
            "audit_status_counts": dict(sorted(audit_counts.items())),
            "logical_invited_and_genuine": len(logical_ids & genuine_ids),
            "logical_used_and_genuine": len(logical_use_ids & genuine_ids),
            "nonlogical_genuine_count": len(genuine_ids - logical_ids),
            "unsupported_count": audit_counts["unsupported_new_semantics"],
            "property_is_non_gating": True,
        },
        "quality": {
            "all_tasks": all_metrics,
            "by_private_route": {
                route: _task_set_metrics(rows) for route, rows in route_groups.items()
            },
            "v4_qualification_reference": qualification_reference,
            "matched_control_claimed": False,
            "interpretation": "single-pilot descriptive comparison; ordinary and structural candidates are not matched counterfactuals",
        },
        "coverage_retention": {
            "ordinary_planning_share": len(route_groups["ordinary"]) / developed if developed else 0.0,
            "search_mode_counts": dict(sorted(Counter(item["search_mode"] for item in records).items())),
            "logical_route_share": len(logical_ids) / developed if developed else 0.0,
            "structural_control_share": len(route_groups["structural_transfer"]) / developed if developed else 0.0,
            "future_production_initial_maximum": "1 logical-leap invitation per 8 planned tasks",
            "rolling_core_target": {"lower": 0.10, "upper": 0.20, "soft_ceiling": 0.25},
            "acceptance_never_adjusted_for_coverage": True,
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
                "source_candidate_id": item["source_candidate_id"],
                "search_mode": item["search_mode"],
                "development_priority": item["source_development_priority"],
                "private_route": item["private_route"],
                "selector_suggestion": item["selector_suggestion"],
                "planner_decision": item.get("planning_result", {}).get("decision"),
                "planner_decision_reason": item.get("planning_result", {}).get("decision_reason"),
                "logical_leap_contract": item.get("planning_result", {}).get("logical_leap_contract"),
                "structural_transfer_contract": item.get("planning_result", {}).get("structural_transfer_contract"),
                "audit": audit_by_id.get(item["task_id"]),
                "terminal_status": item.get("status"),
                "deterministic_valid": bool(item.get("deterministic_diagnostics", {}).get("hard_valid")),
                "technical_results": item.get("technical_results"),
                "blind_results": item.get("blind_results"),
                "editorial_judgments": item.get("editorial_judgments"),
                "intrinsic_quality": item.get("intrinsic_quality"),
                "portfolio_placement": item.get("portfolio_placement"),
                "palette_treatment": item.get("palette_treatment"),
                "grid_metrics": _task_metrics(item["complete_task"]) if item.get("complete_task") else None,
                "task_json": str(runtime.run_dir / "tasks" / "single" / f"{item['task_id']}.arc.json") if item.get("complete_task") else None,
                "plan_json": str(runtime.run_dir / "plans" / "single" / f"{item['task_id']}.json"),
            }
            for item in records
        ],
        "promotion_recommendation": "continue_testing",
        "promotion_reason": (
            "One pilot can show whether the instruction produces any genuine logical leaps, but cannot establish the required distributional authenticity, diversity, or quality non-inferiority. Human review is pending and no automatic promotion is permitted."
        ),
        "production_promotion_occurred": False,
        "v41_baseline_mutated": False,
        "feature_manifest_mutated": False,
        "good_task_bank_mutated_by_experiment": False,
        "protected_production_state_unchanged": protected_before == protected_after,
        "next_experiment_started": False,
        "unresolved_uncertainties": [
            "Human review is pending.",
            "A single slate cannot establish mechanism diversity or quality non-inferiority.",
            "The private structural-control and ordinary subsets are descriptive, not matched counterfactuals.",
        ],
    }
    atomic_write_json(
        runtime.run_dir / "promotion_recommendation.json",
        {
            "feature": "logical_leap_planning",
            "recommendation": "continue_testing",
            "production_promotion_occurred": False,
        },
    )
    atomic_write_json(runtime.run_dir / "quality_noninferiority_report.json", report["quality"])
    atomic_write_json(runtime.run_dir / "coverage_retention_report.json", report["coverage_retention"])
    atomic_write_json(runtime.run_dir / "call_and_timing_ledger.json", accounting)
    atomic_write_json(
        runtime.run_dir / "route_assignment_ledger.private.json",
        {
            "route_hidden_downstream": True,
            "quality_order_unchanged": True,
            "assignments": [
                {
                    "task_id": item["task_id"],
                    "source_candidate_id": item["source_candidate_id"],
                    "development_priority": item["source_development_priority"],
                    "private_route": item["private_route"],
                }
                for item in records
            ],
        },
    )
    return runtime.finish(report, _render_report(report))


def rebuild_v42_report(run: str | Path) -> dict[str, Any]:
    path = Path(run).resolve() / "report.json"
    if not path.is_file():
        raise LogicalLeapError(f"V4.2 report not found: {path}")
    report = read_json(path)
    if report.get("pipeline") != V42_PIPELINE:
        raise LogicalLeapError("run is not an ARCGEN V4.2 logical-leap experiment")
    atomic_write_text(path.parent / "report.md", _render_report(report))
    return report


def v42_human_review_status(run: str | Path) -> dict[str, Any]:
    directory = Path(run).resolve() / "human_review"
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise LogicalLeapError(f"V4.2 human review pack not found: {manifest}")
    value = read_json(manifest)
    return {
        "run": str(Path(run).resolve()),
        "task_count": value["task_count"],
        "review_sheet": str(directory / value["contact_sheet"]),
        "review_form": str(directory / "review_form.json"),
        "answer_appendix": str(directory / "answer_rule_appendix.md"),
        "source_and_route_hidden": value["source_and_route_hidden"],
    }


__all__ = [
    "LogicalLeapError",
    "STRICT_EXTENSION_SCHEMAS",
    "assign_challenge_routes",
    "parse_logical_extension",
    "rebuild_v42_report",
    "run_logical_leap_experiment",
    "v42_human_review_status",
    "validate_v42_plan_response",
]
