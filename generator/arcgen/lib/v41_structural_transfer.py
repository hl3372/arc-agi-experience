"""Selector-routed, single-instance structural-transfer experiment.

The experiment keeps ordinary V4 selection quality-first.  The same global
selector response may nominate a small number of *already selected and
ranked* kernels.  A nominated kernel's ordinary planner call may use or decline
one conspicuous, rule-preserving test-configuration change.  No router or
second authoring call is added, and all downstream quality checks receive only
the ordinary concrete plan and completed task.
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
from .v4_pipeline import V4ControlPipeline, V4ReservoirProducer, v4_call_accounting
from .v4_reservoir import load_reservoir, reservoir_status
from .v4_schemas import STRICT_OUTPUT_SCHEMAS as V4_STRICT_OUTPUT_SCHEMAS
from .v41_experiments import V41ExperimentError, V41Runtime, _anonymous_id, _task_metrics
from .v41_integration import _assert_hashes_unchanged, _protected_hashes
from .v41_supported_departure import (
    SlateEscalationBudget,
    _realize_records,
    _record_stage,
    _run_blind,
    _run_technical,
    _selection_sandbox_config,
)
from .v41_test_challenge import _board as _single_instance_boards


class StructuralTransferError(V41ExperimentError):
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


STRUCTURAL_TRANSFER_CONTRACT = _object(
    {
        "apparent_training_regularity": TEXT,
        "test_configuration_change": TEXT,
        "invariant_relation": TEXT,
        "training_support": TEXT,
        "shallow_rule_defeated": TEXT,
        "no_new_semantics_check": TEXT,
        "one_answer_justification": TEXT,
    }
)

STRUCTURAL_PLAN_SCHEMA = _object(
    {
        "candidate_id": TEXT,
        "decision": {"type": "string", "enum": ["use", "decline", "abort"]},
        "decision_reason": TEXT,
        "structural_transfer_contract": {
            "anyOf": [deepcopy(STRUCTURAL_TRANSFER_CONTRACT), {"type": "null"}]
        },
        "concrete_plan": {
            "anyOf": [
                deepcopy(V4_STRICT_OUTPUT_SCHEMAS["v4_concrete_plan"]),
                {"type": "null"},
            ]
        },
    }
)

STRUCTURAL_AUDIT_ENTRY = _object(
    {
        "task_id": TEXT,
        "status": {
            "type": "string",
            "enum": [
                "substantial_structural_transfer",
                "soft_or_parametric_variation",
                "cosmetic",
                "absent",
                "unsupported",
            ],
        },
        "departure_location": MAYBE_TEXT,
        "apparent_training_regularity": MAYBE_TEXT,
        "configuration_change": MAYBE_TEXT,
        "invariant_relation": MAYBE_TEXT,
        "training_support": TEXT,
        "shallow_rule_defeated": MAYBE_TEXT,
        "same_roles_and_operations": BOOLEAN,
        "one_answer_supported": BOOLEAN,
        "reason": TEXT,
        "open_ended_signature": MAYBE_TEXT,
    }
)

STRUCTURAL_AUDIT_SCHEMA = _object(
    {
        "audit_id": TEXT,
        "judgments": {
            "type": "array",
            "items": deepcopy(STRUCTURAL_AUDIT_ENTRY),
            "minItems": 1,
            "maxItems": 16,
        },
        "non_gating_confirmation": BOOLEAN,
    }
)

STRICT_EXTENSION_SCHEMAS = {
    "v41_structural_transfer_plan": STRUCTURAL_PLAN_SCHEMA,
    "v41_structural_transfer_audit": STRUCTURAL_AUDIT_SCHEMA,
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


def parse_structural_extension(
    stage: str, data: str | bytes | bytearray | Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in STRICT_EXTENSION_SCHEMAS:
        raise SchemaError("$.stage", f"unknown structural-transfer stage {stage!r}")
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
    if stage == "v41_structural_transfer_plan":
        decision = value["decision"]
        plan = value["concrete_plan"]
        contract = value["structural_transfer_contract"]
        if decision == "abort":
            if plan is not None or contract is not None:
                raise SchemaError("$", "abort must return neither plan nor contract")
        else:
            if plan is None:
                raise SchemaError("$.concrete_plan", "use or decline requires a plan")
            if (contract is not None) != (decision == "use"):
                raise SchemaError(
                    "$.structural_transfer_contract",
                    "contract is required exactly when the opportunity is used",
                )
    elif value["non_gating_confirmation"] is not True:
        raise SchemaError("$.non_gating_confirmation", "audit must confirm non-gating status")
    return value


def _install_runtime_extensions(runtime: V41Runtime) -> None:
    root = Path(runtime.config["_repository_root"])
    specifications = {
        "structural_planner_addendum": (
            "v4_1_structural_transfer_planner_addendum.md",
            "v41_structural_transfer_plan",
        ),
        "structural_transfer_auditor": (
            "v4_1_structural_transfer_auditor.md",
            "v41_structural_transfer_audit",
        ),
    }
    destination = runtime.run_dir / "prompts" / "structural_transfer_extensions"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema_version": 1, "prompts": {}}
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
            raise StructuralTransferError(f"extension prompt changed during resume: {filename}")
        if not target.exists():
            atomic_write_text(target, text)
        manifest["prompts"][name] = {
            "filename": filename,
            "version": version,
            "sha256": digest,
            "stage": stage,
            "placeholder_names": sorted(PLACEHOLDER_RE.findall(text)),
        }
    manifest["implementation"] = {
        "file": "arcgen.lib/v41_structural_transfer.py",
        "sha256": sha256(
            (root / "arcgen.lib" / "v41_structural_transfer.py").read_bytes()
        ).hexdigest(),
    }
    path = runtime.run_dir / "structural_transfer_extension_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise StructuralTransferError("structural-transfer extension changed during resume")
    if not path.exists():
        atomic_write_json(path, manifest)

    original = runtime.invoker.output_parser

    def parse(stage: str, data: Any) -> Any:
        if stage in STRICT_EXTENSION_SCHEMAS:
            return parse_structural_extension(stage, data)
        return original(stage, data)

    runtime.invoker.output_parser = parse
    runtime.invoker.output_schemas.update(
        {
            stage: _transport(schema)
            for stage, schema in STRICT_EXTENSION_SCHEMAS.items()
        }
    )
    runtime.invoker._write_schemas()


def _install_selector_prompt(
    selector: V4ControlPipeline, *, maximum_nominations: int
) -> dict[str, Any]:
    if maximum_nominations != 3:
        raise StructuralTransferError(
            "the experimental selector prompt is qualified for exactly three maximum nominations"
        )
    root = Path(selector.config["_repository_root"])
    filename = "v4_1_structural_transfer_selector.md"
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
        raise StructuralTransferError("structural selector bindings differ from V4 global selector")
    selector.prompts.entries["global_selector"] = {
        "filename": filename,
        "text": text,
        "version": version,
        "hash": digest,
    }
    selector.config["selection"]["maximum_optional_development_suggestions"] = maximum_nominations
    extension = {
        "schema_version": 1,
        "selector_stage_call_count_delta": 0,
        "quality_ranking_precedes_nomination": True,
        "maximum_nominations": maximum_nominations,
        "filename": filename,
        "version": version,
        "sha256": digest,
    }
    path = selector.run_dir / "structural_transfer_selector_extension.json"
    if path.exists() and read_json(path) != extension:
        raise StructuralTransferError("selector extension changed during resume")
    if not path.exists():
        atomic_write_json(path, extension)
    snapshot = selector.run_dir / "prompts" / "extensions" / filename
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != text:
        raise StructuralTransferError("selector prompt snapshot changed during resume")
    if not snapshot.exists():
        atomic_write_text(snapshot, text)
    return extension


def validate_selector_nominations(
    briefs: Sequence[Mapping[str, Any]],
    *,
    maximum_nominations: int,
    maximum_development_priority: int,
) -> list[str]:
    nominations = [
        str(item["candidate_id"])
        for item in briefs
        if item.get("optional_development_suggestion") is not None
    ]
    if len(nominations) > maximum_nominations:
        raise StructuralTransferError("selector exceeded the nomination ceiling")
    for item in briefs:
        if (
            item.get("optional_development_suggestion") is not None
            and int(item["development_priority"]) > maximum_development_priority
        ):
            raise StructuralTransferError("selector nominated a candidate outside the realized slate")
    return nominations


def _ensure_structural_selection_window(
    config: Mapping[str, Any],
    v4: dict[str, Any],
    run_dir: Path,
    *,
    accepted_concept_hashes: set[str],
) -> dict[str, Any]:
    """Top up until the *retained* proposal count reaches the window size.

    A model call always returns four kernels, but content-addressed reservoir
    deduplication may retain fewer than twelve from a three-call cycle.  The
    baseline helper estimates one cycle from the raw deficit; this experiment
    loops on the actual retained count so a deduplicated supply cycle cannot
    leave an undersized frozen window.
    """

    required = int(v4["selection"]["eligible_window_size"])
    supply_root = run_dir / "selection_graph" / "v4_reservoir_supply"
    generated_results = []
    attempt = 0
    while True:
        reservoir = load_reservoir(v4["paths"]["reservoir"])
        reserved_hashes = {
            item["concept_hash"] for item in reservoir.get("active_reservations", [])
        }
        eligible = sum(
            item["state"] == "proposed"
            and item["concept_hash"] not in accepted_concept_hashes
            and item["concept_hash"] not in reserved_hashes
            for item in reservoir["proposals"]
        )
        if eligible >= required:
            break
        attempt += 1
        if attempt > 4:
            raise StructuralTransferError(
                f"reservoir retained only {eligible} eligible proposals after four top-up cycles; {required} required"
            )
        supply_run_id = (
            f"{config['run']['id']}-slate-01-structural-topup-{attempt:02d}"
        )
        producer = V4ReservoirProducer(v4, supply_run_id=supply_run_id)
        generated_results.append(
            producer.run(cycles=1, force_between_watermarks=True)
        )

    state_paths = sorted(str(path) for path in supply_root.glob("*/state.json"))
    return {
        "generated_topup_cycles_this_invocation": len(generated_results),
        "topup_results": generated_results,
        "call_ledger_paths": state_paths,
        "reservoir": reservoir_status(
            load_reservoir(v4["paths"]["reservoir"]), v4["reservoir"]
        ),
        "retained_proposed_count": sum(
            item["state"] == "proposed"
            for item in load_reservoir(v4["paths"]["reservoir"])["proposals"]
        ),
        "retained_eligible_count": eligible,
    }


def _freeze_selected_sample(
    config: Mapping[str, Any], run_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = run_dir / "selected_kernel_manifest.json"
    if manifest_path.exists():
        value = read_json(manifest_path)
        return value["development_sample"], value["slate"]

    settings = config["structural_transfer"]
    if int(settings["independent_slates"]) != 1:
        raise StructuralTransferError("the current single-instance pilot requires one slate")
    v4 = _selection_sandbox_config(config, run_dir, slate_index=1)
    selector = V4ControlPipeline(v4)
    supply = _ensure_structural_selection_window(
        config,
        v4,
        run_dir,
        accepted_concept_hashes=selector.context.accepted_concept_hashes(),
    )
    extension = _install_selector_prompt(
        selector,
        maximum_nominations=int(settings["maximum_nominations_per_slate"]),
    )
    briefs = selector.select_slate()
    ordered = sorted(briefs, key=lambda item: int(item["development_priority"]))
    maximum = int(settings["maximum_development_candidates"])
    if not ordered:
        raise StructuralTransferError("quality-first selector returned no developable candidates")
    nominations = validate_selector_nominations(
        ordered,
        maximum_nominations=int(settings["maximum_nominations_per_slate"]),
        maximum_development_priority=maximum,
    )
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
                "structural_transfer_nominated": candidate_id in nominations,
            }
        )
    slate = {
        "slate_index": 1,
        "selection_run_id": v4["run"]["id"],
        "frozen_window": deepcopy(selector.state["frozen_window"]),
        "full_selected_slate": [item["candidate_id"] for item in ordered],
        "development_sample": [item["source_candidate_id"] for item in sample],
        "nominated_candidate_ids": nominations,
        "nomination_count": len(nominations),
        "supply": supply,
        "state_path": str(selector.state_path),
        "selector_extension": extension,
        "quality_selection_and_ranking_preceded_nomination": True,
        "separate_routing_model_call": False,
    }
    value = {
        "schema_version": 1,
        "experimental_design": "selector_routed_structural_transfer_v3",
        "single_instance_only": True,
        "paired_task_variants": False,
        "development_sample_count": len(sample),
        "development_sample": sample,
        "slate": slate,
    }
    atomic_write_json(manifest_path, value)
    return sample, slate


def _aliases(
    selected: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, str]:
    rng = Random(seed + 45113)
    used: set[str] = set()
    keys = [str(item["source_candidate_id"]) for item in selected]
    rng.shuffle(keys)
    return {key: _anonymous_id(rng, used) for key in keys}


def _source_manifest(
    config: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    slate: Mapping[str, Any],
    aliases: Mapping[str, str],
    protected: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "feature": "supported_departure",
        "experimental_design": "selector_routed_structural_transfer_v3",
        "assignment_seed": config["seed"],
        "frozen_v4_parent": deepcopy(config["_v4_parent_verification"]),
        "feature_manifest_snapshot": deepcopy(config["_feature_manifest_snapshot"]),
        "effective_accepted_baseline": deepcopy(config["_effective_features"]),
        "protected_production_state": deepcopy(dict(protected)),
        "model_and_effort_settings": deepcopy(config["model"]["roles"]),
        "selected_kernels": deepcopy(list(selected)),
        "slate": deepcopy(dict(slate)),
        "opaque_task_mapping": [
            {
                "source_candidate_id": source,
                "opaque_task_id": alias,
                "downstream_prompts_receive_source_or_arm_label": False,
            }
            for source, alias in sorted(aliases.items())
        ],
        "single_instance_only": True,
        "paired_task_variants": False,
        "quality_ranking_precedes_nomination": True,
        "routing_call_count": 0,
        "planner_call_count_delta": 0,
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
            suggestion = brief.get("optional_development_suggestion")
            records[task_id] = {
                "task_id": task_id,
                "source_candidate_id": source_id,
                "arm": "single",
                "slate_index": 1,
                "source_development_priority": int(
                    source["source_development_priority"]
                ),
                "search_mode": source["kernel"]["search_mode"],
                "kernel": deepcopy(source["kernel"]),
                "editorial_brief": brief,
                "structural_transfer_nominated": suggestion is not None,
                "structural_transfer_suggestion": suggestion,
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
        runtime.state["semantic_records"] = records
        runtime.save()


DOWNSTREAM_PRIVATE_LABELS = (
    "structural transfer",
    "structural-transfer",
    "experimental wrapper",
    "experimental nomination",
    "treatment arm",
)


def _contains_private_label(value: Any) -> list[str]:
    text = json.dumps(value, ensure_ascii=False).casefold()
    return [item for item in DOWNSTREAM_PRIVATE_LABELS if item in text]


def validate_structural_plan_response(
    *, task_id: str, nominated: bool, value: Mapping[str, Any]
) -> None:
    if value.get("candidate_id") != task_id:
        raise StructuralTransferError("planner returned a mismatched candidate ID")
    decision = value["decision"]
    if not nominated and decision != "not_nominated":
        raise StructuralTransferError("an ordinary candidate acquired an intervention decision")
    plan = value.get("concrete_plan")
    if plan is not None and plan.get("candidate_id") != task_id:
        raise StructuralTransferError("nested ordinary plan returned a mismatched candidate ID")
    if plan is not None:
        leaked = _contains_private_label(plan)
        if leaked:
            raise StructuralTransferError(
                f"ordinary downstream plan contains private experiment labels: {leaked}"
            )


def _plan_records(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]

    def one(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        record = deepcopy(records[task_id])
        values = {
            "CANDIDATE": {**deepcopy(record["kernel"]), "candidate_id": task_id},
            "EDITORIAL_BRIEF": {
                **deepcopy(record["editorial_brief"]),
                "candidate_id": task_id,
            },
            "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(
                record["editorial_brief"]["route"]
            ),
            "CANDIDATE_ID": task_id,
        }
        try:
            if not record["structural_transfer_nominated"]:
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
                    "decision": "not_nominated",
                    "decision_reason": "ordinary V4 planning; no optional opportunity was routed",
                    "structural_transfer_contract": None,
                    "concrete_plan": plan,
                }
                validate_structural_plan_response(
                    task_id=task_id, nominated=False, value=result
                )
                return task_id, result, None

            def validate(value: dict[str, Any]) -> None:
                validate_structural_plan_response(
                    task_id=task_id, nominated=True, value=value
                )
                nested = value.get("concrete_plan")
                if (
                    nested is not None
                    and nested.get("status") == "abort"
                    and value["decision"] != "abort"
                ):
                    raise StructuralTransferError(
                        "an aborted nested plan must use the abort decision"
                    )

            response = runtime.call(
                prompt_name="concrete_planner",
                composed_with=("structural_planner_addendum",),
                call_id=f"{task_id}-concrete-plan",
                values=values,
                parent_ids=[task_id],
                role="concrete_planner",
                arm_id="single-authoring",
                expected={"candidate_id": task_id},
                validator=validate,
            )
            return task_id, response, None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [
        task_id
        for task_id, record in records.items()
        if "planning_result" not in record
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
        destination = runtime.run_dir / "plans" / "single"
        destination.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination / f"{task_id}.json", result)


def _audit(runtime: V41Runtime) -> list[dict[str, Any]]:
    records = runtime.state["semantic_records"]
    completed = [record for record in records.values() if record.get("complete_task")]
    rng = Random(int(runtime.config["seed"]) + 46271)
    rng.shuffle(completed)
    batch_size = int(runtime.config["structural_transfer"]["audit_batch_size"])
    judgments: list[dict[str, Any]] = []
    calls = []
    for offset in range(0, len(completed), batch_size):
        chunk = completed[offset : offset + batch_size]
        audit_id = f"structural-transfer-audit-{offset // batch_size + 1:02d}"
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
            returned = [item["task_id"] for item in value["judgments"]]
            expected = {item["task_id"] for item in packages}
            if len(returned) != len(set(returned)) or set(returned) != expected:
                raise StructuralTransferError("auditor did not cover every task exactly once")

        response = runtime.call(
            prompt_name="structural_transfer_auditor",
            call_id=audit_id,
            values={"AUDIT_ID": audit_id, "TASK_PACKAGES": packages},
            parent_ids=[item["task_id"] for item in packages],
            role="editorial_board",
            arm_id="blinded-structural-audit",
            expected={"audit_id": audit_id},
            validator=validate,
            forbidden_prompt_fragments=(
                "single-authoring",
                "optional suggestion",
                "experimental arm",
                "nominated candidate",
            ),
        )
        calls.append(response)
        judgments.extend(response["judgments"])
    atomic_write_json(
        runtime.run_dir / "structural_transfer_audit.json",
        {
            "arm_hidden": True,
            "nomination_hidden": True,
            "non_gating": True,
            "calls": calls,
            "judgments": judgments,
        },
    )
    return judgments


def _human_pack(runtime: V41Runtime) -> dict[str, Any]:
    destination = runtime.run_dir / "human_review"
    task_dir = destination / "tasks"
    image_dir = destination / "images"
    rule_image_dir = runtime.run_dir / "inspection" / "renders_with_rules"
    task_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    rule_image_dir.mkdir(parents=True, exist_ok=True)
    records = [
        record
        for record in runtime.state["semantic_records"].values()
        if record.get("complete_task")
    ]
    rng = Random(int(runtime.config["seed"]) + 47777)
    rng.shuffle(records)
    used: set[str] = set()
    public = []
    answer_key = []
    review_form = []
    blind_renders = []
    rule_renders = []
    for index, record in enumerate(records, 1):
        anonymous = _anonymous_id(rng, used)
        atomic_write_json(task_dir / f"{anonymous}.json", record["complete_task"])
        blind = render_episode(
            record["complete_task"],
            title=f"Task {index}",
            subtitle="Source, optional routing, and intended rule withheld",
            destination=image_dir / f"{anonymous}.png",
        )
        rule = record["realization"]["authoring_metadata"]["intended_rule"]
        explained = render_episode(
            record["complete_task"],
            title=f"{record['task_id']} — {record['intrinsic_quality'] if record.get('intrinsic_quality') else record.get('status')}",
            subtitle=rule,
            destination=rule_image_dir / f"{record['task_id']}.png",
        )
        blind_renders.append((f"Task {index}", blind))
        rule_renders.append((record["task_id"], explained))
        public.append(
            {
                "review_index": index,
                "anonymous_id": anonymous,
                "task_json": f"tasks/{anonymous}.json",
                "image": f"images/{anonymous}.png",
            }
        )
        planning = record.get("planning_result", {})
        answer_key.append(
            {
                "review_index": index,
                "anonymous_id": anonymous,
                "task_id": record["task_id"],
                "source_candidate_id": record["source_candidate_id"],
                "search_mode": record["search_mode"],
                "selector_nomination": record["structural_transfer_nominated"],
                "planner_decision": planning.get("decision"),
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
                "substantial_rule_preserving_configuration_transfer": None,
                "departure_is_too_soft": None,
                "departure_is_forced_or_unsupported": None,
                "overcomplicated": None,
                "notes": "",
            }
        )
    contact = build_contact_sheet(blind_renders, destination / "review_sheet.png")
    explained_contact = build_contact_sheet(
        rule_renders, runtime.run_dir / "inspection" / "contact_sheet_with_rules.png"
    )
    atomic_write_json(
        destination / "manifest.json",
        {
            "single_instance_only": True,
            "paired_task_variants": False,
            "source_hidden": True,
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
        appendix.extend(
            [
                f"## Task {item['review_index']}",
                "",
                f"- Task ID: `{item['task_id']}`",
                f"- Selector nomination: `{item['selector_nomination']}`",
                f"- Planner decision: `{item['planner_decision']}`",
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


def _task_set_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    realized = [item for item in records if item.get("complete_task")]
    technical_pass = [
        item
        for item in records
        if item.get("technical_results")
        and item["technical_results"][-1].get("passed") is True
    ]
    blind = [item for item in records if item.get("blind_results")]
    core = sum(item.get("intrinsic_quality") == "core_quality" for item in records)
    reserve = sum(item.get("intrinsic_quality") == "reserve_quality" for item in records)
    ambiguous = sum(item.get("status") == "ambiguous" for item in records)
    return {
        "candidate_count": count,
        "realized_count": len(realized),
        "deterministic_validity_rate": len(realized) / count if count else 0.0,
        "technical_pass_count": len(technical_pass),
        "technical_pass_rate": len(technical_pass) / count if count else 0.0,
        "blind_completed_count": len(blind),
        "serious_ambiguity_count": ambiguous,
        "serious_ambiguity_rate": ambiguous / len(blind) if blind else 0.0,
        "core_count": core,
        "reserve_count": reserve,
        "core_yield": core / count if count else 0.0,
        "core_or_reserve_yield": (core + reserve) / count if count else 0.0,
        "repair_count": sum(int(item.get("repair_count", 0)) > 0 for item in records),
        "plan_abort_count": sum(item.get("status") == "plan_aborted" for item in records),
        "realization_failure_count": sum(
            item.get("status") == "realization_failure" for item in records
        ),
    }


def _open_ended_diversity(
    runtime: V41Runtime, judgments: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    positive = [
        item
        for item in judgments
        if item["status"] == "substantial_structural_transfer"
    ]
    result: dict[str, Any] = {
        "substantial_instance_count": len(positive),
        "minimum_before_promotion_diversity_audit": 8,
        "model_cluster_call_run": False,
        "signatures": {
            item["task_id"]: item["open_ended_signature"] for item in positive
        },
    }
    if len(positive) < 2:
        result["semantic_embedding_analysis"] = {
            "status": "not_applicable",
            "reason": "fewer than two substantial instances",
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
        "inconclusive until at least eight substantial completed instances are available"
    )
    atomic_write_json(runtime.run_dir / "diversity_cluster_report.json", result)
    return result


def _call_accounting(
    runtime: V41Runtime, slate: Mapping[str, Any]
) -> dict[str, Any]:
    runtime_accounting = runtime.accounting()
    downstream_ledger = runtime.state["call_ledger"]
    audit_calls = sum(
        item.get("arm_id") == "blinded-structural-audit"
        for item in downstream_ledger.values()
    )
    downstream_calls = runtime_accounting["experimental_calls"]
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
    return {
        "selector_and_window_calls": selector_calls,
        "reservoir_supply_calls": supply_calls,
        "downstream_authoring_and_review_calls": downstream_calls - audit_calls,
        "non_gating_experimental_audit_calls": audit_calls,
        "separate_routing_calls": 0,
        "extra_planner_calls": 0,
        "normal_path_call_delta_for_routing": 0,
        "total_experimental_logical_calls_including_audit": (
            selector_calls + downstream_calls
        ),
        "total_attributed_calls_including_off_path_supply": (
            selector_calls + supply_calls + downstream_calls
        ),
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
    quality = report["quality"]
    calls = report["call_accounting"]
    lines = [
        "# ARCGEN V4.1 — selector-routed structural-transfer pilot",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Recommendation: **{report['promotion_recommendation']}**",
        "",
        "This is a single-instance, non-gating follow-up to the earlier supported-departure studies. The ordinary V4 global selector first selected and ranked the slate on quality. In that same response it could nominate up to three already-selected candidates. The normal planner call then used or declined the candidate-specific opportunity. No paired task variants, routing call, or second authoring call were created.",
        "",
        "## Funnel",
        "",
        f"- Frozen eligible proposals: {funnel['eligible_window']}",
        f"- Globally selected / developed: {funnel['selected']} / {funnel['developed']}",
        f"- Selector nominations: {funnel['nominated']}",
        f"- Planner uses / declines / ordinary / aborts: {funnel['planner_use']} / {funnel['planner_decline']} / {funnel['ordinary']} / {funnel['plan_abort']}",
        f"- Realized / technical pass / blind clear: {funnel['realized']} / {funnel['technical_pass']} / {funnel['blind_pass']}",
        f"- Core / reserve / ambiguous / other: {funnel['core']} / {funnel['reserve']} / {funnel['ambiguous']} / {funnel['other']}",
        "",
        "## Did the stronger instruction work?",
        "",
        f"- Arm-blind audit counts: `{efficacy['audit_status_counts']}`",
        f"- Used opportunities audited as substantial structural transfer: {efficacy['used_and_substantial']}/{efficacy['planner_use_count']}.",
        f"- Nominated opportunities resulting in substantial transfer: {efficacy['nominated_and_substantial']}/{efficacy['nomination_count']}.",
        f"- Unsupported configuration changes: {efficacy['unsupported_count']}.",
        f"- Substantial transfers arising naturally without nomination: {efficacy['ordinary_substantial_count']}.",
        "",
        "The audit is descriptive only. A task can remain excellent when its test variation is soft or absent; the property never contributes editorial credit.",
        "",
        "## Quality",
        "",
        f"- Whole pilot: `{quality['all_tasks']}`",
        f"- Nominated subset: `{quality['nominated']}`",
        f"- Ordinary subset: `{quality['ordinary']}`",
        f"- Frozen V4 qualification reference: `{quality['v4_qualification_reference']}`",
        "",
        "## Calls and cost",
        "",
        f"- Separate routing calls: **{calls['separate_routing_calls']}**",
        f"- Extra planner calls: **{calls['extra_planner_calls']}**",
        f"- Normal-path call delta for routing: **{calls['normal_path_call_delta_for_routing']}**",
        f"- Non-gating experimental audit calls: {calls['non_gating_experimental_audit_calls']}",
        f"- Selector/window + downstream calls (including audit): {calls['total_experimental_logical_calls_including_audit']}",
        f"- Off-critical-path reservoir supply calls: {calls['reservoir_supply_calls']}",
        f"- Active model-minutes excluding / including supply: {calls['active_model_minutes_excluding_supply']} / {calls['active_model_minutes_including_supply']}",
        f"- Wall-clock minutes: {report['timing']['wall_clock_minutes']}",
        "",
        "So the production-shaped mechanism itself costs no additional standing model call: nomination piggybacks the existing global selector, and use/decline piggybacks each existing planner. This pilot adds one non-gating batched audit call solely to measure efficacy.",
        "",
        "## Diversity and review",
        "",
        f"- Diversity conclusion: {report['diversity']['conclusion']}.",
        f"- Blind review sheet: `{report['human_review']['contact_sheet']}`",
        f"- Inspection sheet with rules: `{report['human_review']['contact_sheet_with_rules']}`",
        "- Human review is pending.",
        "",
        "## Task outcomes",
        "",
        "| Task | Mode | Nominated | Planner | Audit | Intrinsic | Portfolio |",
        "|---|---|---:|---|---|---|---|",
    ]
    for task in report["tasks"]:
        lines.append(
            f"| `{task['task_id']}` | `{task['search_mode']}` | "
            f"`{task['nominated']}` | `{task['planner_decision']}` | "
            f"`{(task.get('audit') or {}).get('status', 'not_audited')}` | "
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
            "No production feature promotion or good-task-bank mutation occurred. The later V4.1 semantic experiments were not started by this run.",
            "",
        ]
    )
    return "\n".join(lines)


def run_structural_transfer_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("experimental", {}).get("design") != "selector_routed_structural_transfer_v3":
        raise StructuralTransferError("selector-routed structural-transfer config is required")
    if config["experimental"]["feature"] != "supported_departure":
        raise StructuralTransferError("the experiment must remain isolated under supported_departure")
    feature_state = config["_feature_manifest_snapshot"]["features"][
        "supported_departure"
    ]["status"]
    if feature_state not in {"proposed", "testing"}:
        raise StructuralTransferError("supported departure is not awaiting evaluation")
    if not config["_effective_features"].get("palette_scheduler"):
        raise StructuralTransferError("the explicitly accepted palette baseline is missing")
    settings = config.get("structural_transfer")
    if not isinstance(settings, Mapping):
        raise StructuralTransferError("structural_transfer configuration is missing")
    if settings.get("paired_task_variants") is not False:
        raise StructuralTransferError("this experiment must create one task instance per kernel")

    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    started_path = run_dir / "experiment_start.json"
    if not started_path.exists():
        atomic_write_json(
            started_path, {"started_at_epoch_seconds": round(time.time(), 3)}
        )
    protected_before = _protected_hashes(config)
    selected, slate = _freeze_selected_sample(config, run_dir)
    aliases = _aliases(selected, int(config["seed"]))
    frozen = _source_manifest(
        config, selected, slate, aliases, protected_before
    )
    runtime = V41Runtime(config, frozen)
    _install_runtime_extensions(runtime)
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
    human = _human_pack(runtime)

    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda item: int(item["source_development_priority"]),
    )
    audit_by_id = {item["task_id"]: item for item in audit}
    decisions = Counter(
        item.get("planning_result", {}).get("decision", "failed") for item in records
    )
    audit_counts = Counter(item["status"] for item in audit)
    nominated = [item for item in records if item["structural_transfer_nominated"]]
    ordinary = [item for item in records if not item["structural_transfer_nominated"]]
    used_ids = {
        item["task_id"]
        for item in records
        if item.get("planning_result", {}).get("decision") == "use"
    }
    substantial_ids = {
        item["task_id"]
        for item in audit
        if item["status"] == "substantial_structural_transfer"
    }
    nomination_ids = {item["task_id"] for item in nominated}
    statuses = Counter(item.get("status", "unknown") for item in records)
    all_metrics = _task_set_metrics(records)
    qualification = read_json(
        Path(config["paths"]["frozen_v4_qualification"]) / "report.json"
    )
    qualification_reference = {
        "selected": qualification["funnel"]["selected_and_reserved"],
        "realized": qualification["funnel"]["realized"],
        "technical_pass": qualification["funnel"]["technical_pass"],
        "blind_cleared": qualification["funnel"]["blind_cleared"],
        "core": qualification["funnel"]["intrinsic_quality"].get(
            "core_quality", 0
        ),
        "reserve": qualification["funnel"]["intrinsic_quality"].get(
            "reserve_quality", 0
        ),
        "note": "descriptive frozen V4 qualification, not a newly authored paired control",
    }
    protected_after = _protected_hashes(config)
    _assert_hashes_unchanged(protected_before, protected_after)
    accounting = _call_accounting(runtime, slate)
    started = read_json(started_path)["started_at_epoch_seconds"]
    completed = round(time.time(), 3)
    developed = len(records)
    core = all_metrics["core_count"]
    reserve = all_metrics["reserve_count"]
    ambiguous = statuses["ambiguous"]
    report = {
        "schema_version": 1,
        "pipeline": "arcgen-v4.1-exp-selector-routed-structural-transfer",
        "run_id": runtime.run_id,
        "feature": "supported_departure",
        "experimental_design": "selector_routed_structural_transfer_v3",
        "completed": True,
        "single_instance_only": True,
        "paired_task_variants": False,
        "funnel": {
            "eligible_window": len(slate["frozen_window"]["proposal_ids"]),
            "selected": len(slate["full_selected_slate"]),
            "developed": developed,
            "nominated": len(nominated),
            "planner_use": decisions["use"],
            "planner_decline": decisions["decline"],
            "ordinary": decisions["not_nominated"],
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
            "nomination_count": len(nominated),
            "planner_use_count": decisions["use"],
            "planner_decline_count": decisions["decline"],
            "audit_status_counts": dict(sorted(audit_counts.items())),
            "used_and_substantial": len(used_ids & substantial_ids),
            "nominated_and_substantial": len(nomination_ids & substantial_ids),
            "ordinary_substantial_count": len(substantial_ids - nomination_ids),
            "unsupported_count": audit_counts["unsupported"],
            "property_is_non_gating": True,
        },
        "quality": {
            "all_tasks": all_metrics,
            "nominated": _task_set_metrics(nominated),
            "ordinary": _task_set_metrics(ordinary),
            "v4_qualification_reference": qualification_reference,
            "matched_control_claimed": False,
            "interpretation": "single-pilot descriptive comparison; ordinary candidates in the same slate are not matched counterfactuals",
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
                "nominated": item["structural_transfer_nominated"],
                "selector_suggestion": item["structural_transfer_suggestion"],
                "planner_decision": item.get("planning_result", {}).get("decision"),
                "planner_decision_reason": item.get("planning_result", {}).get(
                    "decision_reason"
                ),
                "structural_transfer_contract": item.get(
                    "planning_result", {}
                ).get("structural_transfer_contract"),
                "audit": audit_by_id.get(item["task_id"]),
                "terminal_status": item.get("status"),
                "deterministic_valid": bool(
                    item.get("deterministic_diagnostics", {}).get("hard_valid")
                ),
                "technical_results": item.get("technical_results"),
                "blind_results": item.get("blind_results"),
                "editorial_judgments": item.get("editorial_judgments"),
                "intrinsic_quality": item.get("intrinsic_quality"),
                "portfolio_placement": item.get("portfolio_placement"),
                "grid_metrics": _task_metrics(item["complete_task"])
                if item.get("complete_task")
                else None,
                "task_json": str(
                    runtime.run_dir / "tasks" / "single" / f"{item['task_id']}.arc.json"
                )
                if item.get("complete_task")
                else None,
                "plan_json": str(
                    runtime.run_dir / "plans" / "single" / f"{item['task_id']}.json"
                ),
            }
            for item in records
        ],
        "promotion_recommendation": "continue_testing",
        "promotion_reason": (
            "This is one routing pilot, not the three-slate/eight-genuine qualification. "
            "Human review is pending and no automatic promotion is permitted."
        ),
        "production_promotion_occurred": False,
        "good_task_bank_mutated_by_experiment": False,
        "protected_production_state_unchanged": protected_before == protected_after,
        "next_experiment_started": False,
        "unresolved_uncertainties": [
            "Human review is pending.",
            "A single slate cannot establish mechanism diversity or non-inferiority.",
            "The selector's quality-first ordering is enforced by prompt and routing logic, not by a paired counterfactual selector call.",
        ],
    }
    atomic_write_json(
        runtime.run_dir / "promotion_recommendation.json",
        {
            "feature": "supported_departure",
            "recommendation": "continue_testing",
            "production_promotion_occurred": False,
        },
    )
    atomic_write_json(runtime.run_dir / "quality_report.json", report["quality"])
    atomic_write_json(runtime.run_dir / "call_and_timing_ledger.json", accounting)
    return runtime.finish(report, _render_report(report))


__all__ = [
    "STRICT_EXTENSION_SCHEMAS",
    "StructuralTransferError",
    "parse_structural_extension",
    "run_structural_transfer_experiment",
    "validate_selector_nominations",
    "validate_structural_plan_response",
]
