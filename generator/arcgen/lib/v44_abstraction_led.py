"""Matched, arm-blind ARCGEN V4.4 abstraction-led idea experiment.

The ordinary control kernels and calibration contexts are checksummed V4.3
artifacts. Only treatment idea search receives the abstraction-led addendum.
Both arms share selection, concrete embodiment-quality guidance, realization,
verification, and editorial policy. Property audits are descriptive only.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence
import json
import re
import time

from .embeddings import build_embedding_provider
from .lean_v3_rendering import build_contact_sheet, render_episode
from .util import (
    PLACEHOLDER_RE,
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    parallel_map,
    read_json,
)
from .v4_pipeline import V4ControlPipeline
from .v41_audits import matched_noninferiority, mechanism_diversity_summary
from .v41_experiments import V41ExperimentError, V41Runtime, _task_metrics
from .v41_integration import _assert_hashes_unchanged, _protected_hashes
from .v41_structural_transfer import _task_set_metrics
from .v41_supported_departure import (
    SlateEscalationBudget,
    _realize_records,
    _record_stage,
    _run_blind,
    _run_technical,
)
from .v41_test_challenge import _board as _single_instance_boards
from .v43_config import verify_v42_baseline
from .v43_episode_joint import (
    PRIVATE_LABELS,
    _arm_quality,
    _reference_packages,
)
from .v431_episode_joint_quality import (
    GLOBAL_SELECTOR_SCHEMA,
    parse_v431_global_selector,
)
from .v44_config import (
    V44_DESIGN,
    V44_PIPELINE,
    verify_v43_control_source,
)


class AbstractionLedError(V41ExperimentError):
    pass


_ARC_COLOR_ALIASES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "fuchsia",
    7: "orange",
    8: "azure",
    9: "maroon",
}


def _normalize_palette_claims(value: Any) -> Any:
    """Canonicalize unambiguous planner color prose before V4.1 remapping.

    The frozen palette rewriter intentionally avoids bare numbers because they
    may be coordinates or counts.  Planners sometimes emit redundant aliases
    (``black 0``) or construction phrases (``grid of 0``) that are equally
    unambiguous but fall outside that frozen grammar.  Normalize only those
    narrow forms so V4.1's accepted global bijection can update them safely.
    """

    if isinstance(value, str):
        text = value
        for color, name in _ARC_COLOR_ALIASES.items():
            text = re.sub(
                rf"\b{re.escape(name)}\s+{color}\b",
                name,
                text,
                flags=re.IGNORECASE,
            )
        text = re.sub(
            r"\b(Start with an?\s+\d+x\d+\s+grid)\s+of\s+([0-9])\b",
            r"\1 filled with color \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b((?:Every|All)\s+(?:other\s+|remaining\s+|unspecified\s+|unlisted\s+)?(?:input\s+|output\s+)?cells?\s+(?:are|is|remain|remains|stay|stays))\s+([0-9])\b",
            r"\1 color \2",
            text,
            flags=re.IGNORECASE,
        )
        return text
    if isinstance(value, list):
        return [_normalize_palette_claims(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _normalize_palette_claims(item) for key, item in value.items()}
    return deepcopy(value)


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


def _install_extensions(runtime: V41Runtime) -> None:
    root = Path(runtime.config["_repository_root"])
    specs = {
        "v44_shard_selector": (
            "v4_4_blinded_kernel_selector.md",
            "v44_kernel_selector",
        ),
        "v44_global_selector": (
            "v4_4_global_quality_selector.md",
            "v44_kernel_selector",
        ),
        "v44_quality_plan": (
            "v4_4_concrete_quality_planning_addendum.md",
            "v4_concrete_plan",
        ),
    }
    destination = runtime.run_dir / "prompts" / "v44_extensions"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": V44_PIPELINE,
        "prompts": {},
        "implementation": {},
    }
    for name, (filename, stage) in specs.items():
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
            raise AbstractionLedError(f"V4.4 prompt changed during resume: {filename}")
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
        "arcgen.lib/v44_config.py",
        "arcgen.lib/v44_abstraction_led.py",
        "config/v4_4_exp_abstraction_led.yaml",
        "V4_4_ABSTRACTION_LED_EXPERIMENT.md",
    ):
        path = root / relative
        manifest["implementation"][relative] = sha256(path.read_bytes()).hexdigest()
    path = runtime.run_dir / "v44_extension_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise AbstractionLedError("V4.4 implementation or prompt changed during resume")
    if not path.exists():
        atomic_write_json(path, manifest)

    original = runtime.invoker.output_parser

    def parse(stage: str, data: Any) -> Any:
        if stage == "v44_kernel_selector":
            return parse_v431_global_selector(data)
        return original(stage, data)

    runtime.invoker.output_parser = parse
    runtime.invoker.output_schemas["v44_kernel_selector"] = _transport(
        GLOBAL_SELECTOR_SCHEMA
    )
    runtime.invoker._write_schemas()


def _opaque_treatment_id(seed: int, slate: int, pair: int, index: int) -> str:
    digest = sha256(f"v44:{seed}:{slate}:{pair}:treatment:{index}".encode()).hexdigest()
    return "k" + digest[:13]


def _freeze_design(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    path = run_dir / "frozen_experiment_design.private.json"
    if path.exists():
        return read_json(path)
    verification = verify_v43_control_source(config)
    source = Path(verification["source_run"])
    source_design = read_json(source / "frozen_experiment_design.private.json")
    source_supply = read_json(source / "kernel_supply_manifest.private.json")
    controls = [
        deepcopy(item)
        for item in source_supply["kernels"]
        if item["private_arm"] == "control"
    ]
    settings = config["abstraction_led"]
    expected = int(settings["frozen_control_kernel_count"])
    if len(controls) != expected or len({item["candidate_id"] for item in controls}) != expected:
        raise AbstractionLedError("frozen V4.3 control supply is incomplete or duplicated")
    controls_by_call: dict[str, list[dict[str, Any]]] = {}
    for item in controls:
        controls_by_call.setdefault(item["generation_call_id"], []).append(item)

    paired_calls = []
    arm_map: dict[str, str] = {item["candidate_id"]: "control" for item in controls}
    for source_pair in source_design["paired_calls"]:
        control_call = next(
            call for call in source_pair["calls"] if call["arm"] == "control"
        )
        frozen_rows = sorted(
            controls_by_call[str(control_call["call_id"])],
            key=lambda item: list(control_call["candidate_ids"]).index(item["candidate_id"]),
        )
        if [item["candidate_id"] for item in frozen_rows] != list(
            control_call["candidate_ids"]
        ):
            raise AbstractionLedError("control candidate order differs from its frozen call")
        slate = int(source_pair["slate_index"])
        pair = int(source_pair["pair_index"])
        treatment_ids = [
            _opaque_treatment_id(int(config["seed"]), slate, pair, index)
            for index in range(1, int(settings["ideas_per_call"]) + 1)
        ]
        arm_map.update({candidate_id: "treatment" for candidate_id in treatment_ids})
        paired_calls.append(
            {
                "slate_index": slate,
                "pair_index": pair,
                "calibration_task_ids": list(source_pair["calibration_task_ids"]),
                "calibration_descriptions": list(
                    source_pair["calibration_descriptions"]
                ),
                "frozen_control_call_id": str(control_call["call_id"]),
                "frozen_control_candidate_ids": list(control_call["candidate_ids"]),
                "treatment_call_id": f"v44-slate-{slate:02d}-pair-{pair:02d}-treatment",
                "treatment_candidate_ids": treatment_ids,
            }
        )
    value = {
        "schema_version": 1,
        "pipeline": V44_PIPELINE,
        "design": V44_DESIGN,
        "v42_parent": deepcopy(config["_v4_2_baseline_verification"]),
        "frozen_v43_control_source": verification,
        "paired_calls": paired_calls,
        "frozen_control_kernels": controls,
        "private_candidate_arm_mapping": arm_map,
        "same_calibration_descriptions": True,
        "same_model_effort_and_schema": True,
        "only_treatment_idea_search_receives_abstraction_instruction": True,
        "source_labels_hidden_from_selection_and_downstream": True,
        "shared_quality_gate_for_both_arms": True,
        "automatic_promotion": False,
        "automatic_task_commit": False,
        "protected_production_state": _protected_hashes(config),
    }
    atomic_write_json(path, value)
    atomic_write_json(
        run_dir / "frozen_experiment_design.public.json",
        {
            key: deepcopy(item)
            for key, item in value.items()
            if key not in {"private_candidate_arm_mapping", "frozen_control_kernels"}
        },
    )
    return value


def _generate_kernels(runtime: V41Runtime, design: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = runtime.run_dir / "kernel_supply_manifest.private.json"
    if path.exists():
        return read_json(path)["kernels"]
    controls = [deepcopy(item) for item in design["frozen_control_kernels"]]

    def one(pair: Mapping[str, Any]) -> list[dict[str, Any]]:
        ids = list(pair["treatment_candidate_ids"])
        call_id = str(pair["treatment_call_id"])

        def validate(value: dict[str, Any]) -> None:
            if [item["candidate_id"] for item in value["ideas"]] != ids:
                raise AbstractionLedError("treatment idea call changed assigned IDs")

        response = runtime.call(
            prompt_name="idea_concept",
            composed_with=("abstraction_idea_addendum",),
            call_id=call_id,
            values={
                "CALIBRATION_DESCRIPTIONS": list(pair["calibration_descriptions"]),
                "SEARCH_CALL_ID": call_id,
                "CANDIDATE_IDS": ids,
            },
            parent_ids=[],
            role="idea_search",
            arm_id="private-treatment-idea-supply",
            expected={"search_call_id": call_id, "search_mode": "concept_led"},
            validator=validate,
        )
        return [
            {
                "candidate_id": item["candidate_id"],
                "slate_index": int(pair["slate_index"]),
                "pair_index": int(pair["pair_index"]),
                "private_arm": "treatment",
                "kernel": deepcopy(item),
                "calibration_task_ids": list(pair["calibration_task_ids"]),
                "generation_call_id": call_id,
            }
            for item in response["ideas"]
        ]

    rows = parallel_map(
        one,
        list(design["paired_calls"]),
        min(len(design["paired_calls"]), int(runtime.config["model"]["concurrency"])),
    )
    treatments = [item for group in rows for item in group]
    expected = int(runtime.config["abstraction_led"]["frozen_control_kernel_count"])
    if len(treatments) != expected:
        raise AbstractionLedError("treatment generation did not match control supply")
    kernels = controls + treatments
    if len({item["candidate_id"] for item in kernels}) != len(kernels):
        raise AbstractionLedError("control and treatment candidate IDs collide")
    value = {
        "schema_version": 1,
        "control_supply_reused_from_frozen_v43": True,
        "new_control_idea_calls": 0,
        "new_treatment_idea_calls": len(design["paired_calls"]),
        "arm_mapping_is_private": True,
        "kernel_count": len(kernels),
        "kernels": kernels,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        runtime.run_dir / "kernel_supply_manifest.blinded.json",
        {
            "schema_version": 1,
            "source_labels_removed": True,
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


def _selector_call(
    runtime: V41Runtime,
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_name: str,
    selector_id: str,
    arm_id: str,
) -> dict[str, Any]:
    packages, accepted, reservations, rejected = _reference_packages(runtime.config, rows)
    experimental_memory = runtime.config.get("_v44_experimental_soft_memory", {})
    if experimental_memory.get("enabled"):
        rejected = deepcopy(rejected)
        rejected.setdefault("soft_mechanism", []).extend(
            deepcopy(experimental_memory["entries"])
        )
    expected_ids = {str(item["candidate_id"]) for item in rows}

    def validate(value: dict[str, Any]) -> None:
        returned = {item["candidate_id"] for item in value["assessments"]}
        if returned != expected_ids or set(value["quality_order"]) != expected_ids:
            raise AbstractionLedError("selector did not cover its frozen candidate set")

    return runtime.call(
        prompt_name=prompt_name,
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
        arm_id=arm_id,
        expected={"selector_id": selector_id},
        validator=validate,
        forbidden_prompt_fragments=PRIVATE_LABELS,
    )


def _choose(
    response: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_per_arm: int,
    maximum_total: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(item["candidate_id"]): item for item in rows}
    assessments = {
        str(item["candidate_id"]): item for item in response["assessments"]
    }
    chosen_ids: list[str] = []
    counts = Counter()
    exclusions: list[dict[str, Any]] = []
    for candidate_id in response["quality_order"]:
        assessment = assessments[candidate_id]
        arm = str(by_id[candidate_id]["private_arm"])
        reason: str | None = None
        siblings = [
            other
            for other in chosen_ids
            if other in assessment["same_minimal_procedure_with"]
            or candidate_id in assessments[other]["same_minimal_procedure_with"]
        ]
        if assessment["recommendation"] != "develop":
            reason = assessment["recommendation"]
        elif siblings:
            reason = "same_minimal_procedure_sibling"
        elif counts[arm] >= maximum_per_arm:
            reason = "arm_sampling_ceiling"
        elif len(chosen_ids) >= maximum_total:
            reason = "global_development_ceiling"
        if reason:
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "reason": reason,
                    "same_procedure_with_chosen": siblings,
                }
            )
            continue
        chosen_ids.append(candidate_id)
        counts[arm] += 1

    chosen = []
    for priority, candidate_id in enumerate(chosen_ids, 1):
        source = deepcopy(dict(by_id[candidate_id]))
        source["selector_global_priority"] = priority
        source["source_development_priority"] = 1 + sum(
            item["private_arm"] == source["private_arm"] for item in chosen
        )
        source["editorial_brief"] = deepcopy(
            assessments[candidate_id]["development_brief"]
        )
        source["selector_assessment"] = deepcopy(assessments[candidate_id])
        chosen.append(source)
    return chosen, exclusions


def _select_kernels(
    runtime: V41Runtime, kernels: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = runtime.run_dir / "selected_kernel_manifest.private.json"
    if path.exists():
        value = read_json(path)
        return value["development_sample"], value
    settings = runtime.config["abstraction_led"]
    shard_candidates: list[dict[str, Any]] = []
    shard_ledgers = []
    for slate in range(1, int(settings["independent_slates"]) + 1):
        rows = [deepcopy(item) for item in kernels if int(item["slate_index"]) == slate]
        Random(int(runtime.config["seed"]) + 44019 * slate).shuffle(rows)
        response = _selector_call(
            runtime,
            rows,
            prompt_name="v44_shard_selector",
            selector_id=f"v44-shard-selector-{slate:02d}",
            arm_id="blinded-shard-selection",
        )
        selected, excluded = _choose(
            response,
            rows,
            maximum_per_arm=int(settings["shard_maximum_per_arm"]),
            maximum_total=2 * int(settings["shard_maximum_per_arm"]),
        )
        shard_candidates.extend(selected)
        counts = Counter(item["private_arm"] for item in selected)
        shard_ledgers.append(
            {
                "slate_index": slate,
                "candidate_count": len(rows),
                "selected_count": len(selected),
                "selected_by_private_arm": dict(sorted(counts.items())),
                "excluded": excluded,
                "response": response,
                "source_labels_visible_to_selector": False,
            }
        )
    if not shard_candidates:
        raise AbstractionLedError("shard selection produced no developable candidates")
    global_rows = [deepcopy(item) for item in shard_candidates]
    Random(int(runtime.config["seed"]) + 44991).shuffle(global_rows)
    response = _selector_call(
        runtime,
        global_rows,
        prompt_name="v44_global_selector",
        selector_id="v44-global-quality-selector",
        arm_id="blinded-global-selection",
    )
    selected, global_excluded = _choose(
        response,
        global_rows,
        maximum_per_arm=int(settings["global_maximum_per_arm"]),
        maximum_total=int(settings["maximum_global_development"]),
    )
    if not selected:
        raise AbstractionLedError("global selection produced no developable candidates")
    counts = Counter(item["private_arm"] for item in selected)
    value = {
        "schema_version": 1,
        "arm_mapping_is_private": True,
        "source_labels_visible_to_selectors": False,
        "shards": shard_ledgers,
        "global_response": response,
        "global_excluded": global_excluded,
        "development_counts_by_private_arm": dict(sorted(counts.items())),
        "supply_shortfall": {
            arm: max(0, int(settings["global_maximum_per_arm"]) - counts[arm])
            for arm in ("control", "treatment")
        },
        "development_sample": selected,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        runtime.run_dir / "selected_kernel_manifest.blinded.json",
        {
            "schema_version": 1,
            "source_labels_removed": True,
            "shard_count": len(shard_ledgers),
            "development_sample": [
                {
                    key: deepcopy(item)
                    for key, item in row.items()
                    if key not in {"private_arm", "generation_call_id", "calibration_task_ids"}
                }
                for row in selected
            ],
        },
    )
    return selected, value


def _initialize_records(runtime: V41Runtime, selected: Sequence[Mapping[str, Any]]) -> None:
    with runtime.lock:
        if "semantic_records" in runtime.state:
            return
        records = {}
        arm_counts = Counter()
        for source in selected:
            task_id = str(source["candidate_id"])
            arm = str(source["private_arm"])
            arm_counts[arm] += 1
            records[task_id] = {
                "task_id": task_id,
                "source_candidate_id": task_id,
                "arm": arm,
                "slate_index": int(source["slate_index"]),
                "source_development_priority": arm_counts[arm],
                "selector_global_priority": int(source["selector_global_priority"]),
                "search_mode": "concept_led",
                "kernel": deepcopy(source["kernel"]),
                "editorial_brief": deepcopy(source["editorial_brief"]),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
        runtime.state["semantic_records"] = records
        runtime.state["v44_pipeline"] = V44_PIPELINE
        runtime.state["v44_experimental_property"] = "abstraction_led"
        runtime.save()


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
                    composed_with=("v44_quality_plan",),
                    call_id=f"{task_id}-v44-control-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="private-control-authoring",
                    expected={"candidate_id": task_id},
                    forbidden_prompt_fragments=PRIVATE_LABELS,
                )
                response = {
                    "candidate_id": task_id,
                    "property_status": "not_applicable",
                    "decision_reason": "ordinary planning with the shared quality gate",
                    "property_contract": None,
                    "concrete_plan": plan,
                }
            else:
                response = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("v44_quality_plan", "abstraction_plan_addendum"),
                    call_id=f"{task_id}-v44-abstraction-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="private-treatment-authoring",
                    expected={"candidate_id": task_id},
                )
                nested = response.get("concrete_plan")
                if nested is not None and nested.get("candidate_id") != task_id:
                    raise AbstractionLedError("planner changed the nested candidate ID")
                if nested is not None and nested.get("status") == "abort" and response[
                    "property_status"
                ] != "abort":
                    raise AbstractionLedError("aborted nested plan must use abort status")
            plan = response["concrete_plan"]
            if plan is not None:
                plan = _normalize_palette_claims(plan)
                response["concrete_plan"] = plan
            leaked = [
                label
                for label in PRIVATE_LABELS
                if label in json.dumps(plan, ensure_ascii=False).casefold()
            ] if plan is not None else []
            if leaked:
                raise AbstractionLedError(f"concrete plan leaked private labels: {leaked}")
            return task_id, response, None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [
        task_id for task_id, record in records.items() if "planning_result" not in record
    ]
    results = parallel_map(
        one,
        pending,
        min(len(pending), int(runtime.config["model"]["concurrency"])),
    ) if pending else []
    for task_id, response, error in results:
        if error or response is None:
            _record_stage(
                runtime,
                task_id,
                "planning",
                status="planning_operational_failure",
                planning_error=error,
            )
            continue
        plan = response["concrete_plan"]
        status = "plan_aborted" if plan is None or plan.get("status") == "abort" else "plan_ready"
        _record_stage(
            runtime,
            task_id,
            "planning",
            status=status,
            planning_result=response,
            concrete_plan=plan,
        )
        destination = runtime.run_dir / "plans" / records[task_id]["arm"]
        destination.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination / f"{task_id}.json", response)


def _budgets(runtime: V41Runtime) -> dict[tuple[int, str], SlateEscalationBudget]:
    settings = runtime.config["abstraction_led"]
    limit = int(settings["escalation_calls_per_arm_per_slate"])
    with runtime.lock:
        usage = runtime.state.setdefault("v44_escalation_usage", {})
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
                    runtime.state["v44_escalation_usage"][usage_key] = used
                    runtime.save()

            result[(slate, arm)] = SlateEscalationBudget(
                limit,
                used=int(runtime.state["v44_escalation_usage"][key]),
                on_change=persist,
            )
    return result


def _mechanism_audit(runtime: V41Runtime) -> list[dict[str, Any]]:
    path = runtime.run_dir / "mechanism_audit_report.json"
    if path.exists():
        return read_json(path)["judgments"]
    records = [
        record
        for record in runtime.state["semantic_records"].values()
        if record.get("complete_task")
    ]
    Random(int(runtime.config["seed"]) + 448123).shuffle(records)
    batch = int(runtime.config["abstraction_led"]["audit_batch_size"])
    judgments: list[dict[str, Any]] = []
    calls = []
    for offset in range(0, len(records), batch):
        chunk = records[offset : offset + batch]
        audit_id = f"abstraction-audit-{offset // batch + 1:02d}"
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
                raise AbstractionLedError("mechanism audit omitted or duplicated a task")

        response = runtime.call(
            prompt_name="mechanism_auditor",
            call_id=audit_id,
            values={
                "REQUESTED_PROPERTIES": ["abstraction_led"],
                "TASK_PACKAGES": packages,
                "AUDIT_ID": audit_id,
            },
            parent_ids=[item["task_id"] for item in packages],
            role="editorial_board",
            arm_id="blinded-abstraction-audit",
            expected={
                "audit_id": audit_id,
                "requested_properties": ["abstraction_led"],
                "non_gating_confirmation": True,
            },
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


def _audit_status(judgment: Mapping[str, Any]) -> str:
    assessment = judgment.get("abstraction_led")
    return str(assessment["status"]) if assessment else "not_audited"


def _audit_signature(judgment: Mapping[str, Any]) -> str:
    signatures = judgment.get("open_ended_mechanism_signatures") or {}
    return str(signatures.get("abstraction_led") or "")


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
        if item["task_id"] in treatment_ids and _audit_status(item) == "genuine"
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

    signatures = {item["task_id"]: _audit_signature(item) for item in positives}
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
            "signature": _audit_signature(item),
            "concrete_evidence": item["abstraction_led"],
            "nearest_signature_neighbors": nearest[item["task_id"]],
        }
        for item in positives
    ]
    audit_id = "abstraction-diversity"

    def validate(value: dict[str, Any]) -> None:
        members = [
            task_id
            for cluster in value["clusters"]
            for task_id in cluster["member_task_ids"]
        ]
        expected = {item["task_id"] for item in positives}
        if len(members) != len(set(members)) or set(members) != expected:
            raise AbstractionLedError("diversity audit must cluster every positive once")

    response = runtime.call(
        prompt_name="diversity_auditor",
        call_id=audit_id,
        values={
            "PROPERTY": "abstraction_led",
            "INSTANCES": instances,
            "AUDIT_ID": audit_id,
        },
        parent_ids=[item["task_id"] for item in positives],
        role="editorial_board",
        arm_id="blinded-abstraction-diversity",
        expected={"audit_id": audit_id, "property": "abstraction_led"},
        validator=validate,
        forbidden_prompt_fragments=PRIVATE_LABELS,
    )
    summary = mechanism_diversity_summary(
        positive_task_ids=[item["task_id"] for item in positives],
        diversity_audit=response,
    )
    value = {
        "audit": response,
        "summary": summary,
        "semantic_embedding_analysis": embedding,
    }
    atomic_write_json(path, value)
    return value


def _human_pack(
    runtime: V41Runtime, judgments: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
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
    Random(int(runtime.config["seed"]) + 449931).shuffle(records)
    audit_by_id = {item["task_id"]: item for item in judgments}
    public = []
    answer_key = []
    renders = []
    form = []
    for index, record in enumerate(records, 1):
        anonymous = "review-" + content_hash(
            {"seed": runtime.config["seed"], "task": record["task_id"], "v": 44}
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
                "genuine_operational_abstraction": None,
                "overcomplicated_or_merely_formal": None,
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
        assessment = audit.get("abstraction_led") or {}
        appendix.extend(
            [
                f"## Task {item['review_index']}",
                "",
                f"- Task ID: `{item['task_id']}`",
                f"- Source arm: `{item['private_arm']}`",
                f"- Abstraction audit: `{assessment.get('status', 'not_audited')}` — {assessment.get('reason', '')}",
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
        "pipeline": V44_PIPELINE,
        "run_id": runtime.run_id,
        "completed": False,
        "current_safe_stage": stage,
        "control_kernel_count": sum(item["private_arm"] == "control" for item in kernels),
        "treatment_kernel_count": sum(item["private_arm"] == "treatment" for item in kernels),
        "selected_control_count": sum(item["private_arm"] == "control" for item in selected),
        "selected_treatment_count": sum(item["private_arm"] == "treatment" for item in selected),
        "call_accounting": runtime.accounting(),
        "resume_command": (
            "python3 -m arcgen.lib v4-4-experiment --config "
            "config/v4_4_exp_abstraction_led.yaml --through-phase complete"
        ),
        "production_promotion_occurred": False,
        "good_task_bank_mutated": False,
    }
    atomic_write_json(runtime.run_dir / "progress.json", value)
    return value


def _render_report(report: Mapping[str, Any]) -> str:
    funnel = report["funnel"]
    quality = report["quality_noninferiority"]
    lines = [
        "# ARCGEN V4.4 — abstraction-led idea experiment",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Recommendation: **{report['promotion_recommendation']}**",
        "",
        "The experiment reused the 24 checksummed ordinary concept-led V4.3 control kernels and generated 24 matched treatment kernels from the same six calibration contexts. Only treatment idea search received the abstraction-led instruction. Source labels were hidden from shard selection, global selection, realization, verification, boards, and human review.",
        "",
        "## Funnel",
        "",
        f"- Kernels, control / treatment: {funnel['control_kernels']} / {funnel['treatment_kernels']}",
        f"- Globally selected, control / treatment: {funnel['selected_control']} / {funnel['selected_treatment']}",
        f"- Plans ready / not ready: {funnel['plans_ready']} / {funnel['plans_not_ready']}",
        f"- Realized / technical pass / blind clear: {funnel['realized']} / {funnel['technical_pass']} / {funnel['blind_clear']}",
        f"- Core / reserve / ambiguous / other: {funnel['core']} / {funnel['reserve']} / {funnel['ambiguous']} / {funnel['other']}",
        "",
        "## Efficacy and quality",
        "",
        f"- Arm-blind abstraction audit: `{report['efficacy']['audit_status_by_private_arm']}`",
        f"- Treatment plan status: `{report['efficacy']['treatment_planner_status_counts']}`",
        f"- Control quality: `{quality['control']}`",
        f"- Treatment quality: `{quality['treatment']}`",
        f"- Provisional model non-inferiority gates all pass: `{quality['all_provisional_model_gates_passed']}`",
        "",
        "## Diversity",
        "",
        f"- `{report['diversity']['summary']}`",
        "- Same-minimal-procedure siblings are not counted as separate positive instances.",
        "",
        "## Cost and human review",
        "",
        f"- New control / treatment idea calls: 0 / {report['new_idea_calls']['treatment']}",
        f"- Experimental logical calls: {report['call_accounting']['experimental_calls']}",
        f"- Active model-minutes: {report['call_accounting']['active_model_minutes']}",
        f"- Wall-clock minutes: {report['timing']['wall_clock_minutes']}",
        f"- Blinded review sheet: `{report['human_review']['review_sheet']}`",
        "",
        "## Tasks",
        "",
        "| Task | Audit | Technical | Blind | Intrinsic | Portfolio | Max dim |",
        "|---|---|---|---|---|---|---:|",
    ]
    for task in report["tasks"]:
        lines.append(
            f"| `{task['task_id']}` | `{task['mechanism_status']}` | `{task['technical_pass']}` | "
            f"`{task['blind_clear']}` | `{task.get('intrinsic_quality') or task['terminal_status']}` | "
            f"`{task.get('portfolio_placement') or 'not_applicable'}` | "
            f"{task['grid_metrics']['maximum_dimension'] if task.get('grid_metrics') else ''} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["promotion_reason"],
            "",
            "No production feature was promoted and no task was committed automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def run_abstraction_led_experiment(
    config: dict[str, Any], *, through_phase: str = "complete"
) -> dict[str, Any]:
    if through_phase not in {"kernels", "selection", "complete"}:
        raise AbstractionLedError("through_phase must be kernels, selection, or complete")
    if config.get("v4_4_pipeline") != V44_PIPELINE or config.get("v4_4_design") != V44_DESIGN:
        raise AbstractionLedError("the isolated V4.4 abstraction config is required")
    if config["run"].get("automatically_commit_generated_tasks") is not False:
        raise AbstractionLedError("V4.4 cannot commit experimental tasks automatically")

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
    selected, selection = _select_kernels(runtime, kernels)
    _initialize_records(runtime, selected)
    if through_phase == "selection":
        return _progress(
            runtime,
            stage="development_sample_frozen",
            kernels=kernels,
            selected=selected,
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

    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda item: int(item["selector_global_priority"]),
    )
    by_arm = {
        arm: [item for item in records if item["arm"] == arm]
        for arm in ("control", "treatment")
    }
    quality_by_arm = {arm: _arm_quality(items) for arm, items in by_arm.items()}
    noninferiority = matched_noninferiority(
        control=quality_by_arm["control"], treatment=quality_by_arm["treatment"]
    )
    audit_by_id = {item["task_id"]: item for item in audit}
    audit_status = {
        arm: dict(
            sorted(
                Counter(
                    _audit_status(audit_by_id[item["task_id"]])
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
    realized = sum(bool(item.get("complete_task")) for item in records)
    technical_pass = sum(
        bool(item.get("technical_results"))
        and item["technical_results"][-1].get("passed") is True
        for item in records
    )
    blind_clear = sum(
        item.get("status") in {"blind_pass", "editorial_complete"} for item in records
    )
    plans_ready = sum(
        item.get("concrete_plan", {}).get("status") == "ready" for item in records
    )
    genuine_treatment = sum(
        _audit_status(item) == "genuine"
        and runtime.state["semantic_records"][item["task_id"]]["arm"] == "treatment"
        for item in audit
    )
    genuine_control = sum(
        _audit_status(item) == "genuine"
        and runtime.state["semantic_records"][item["task_id"]]["arm"] == "control"
        for item in audit
    )
    summary = diversity["summary"]
    minimum = int(config["abstraction_led"]["minimum_genuine_completed"])
    minimum_clusters = int(config["abstraction_led"]["minimum_mechanism_clusters"])
    largest_limit = float(
        config["abstraction_led"]["maximum_largest_cluster_fraction"]
    )
    recommendation = "continue_testing"
    if genuine_treatment < minimum:
        recommendation = "inconclusive"
        reason = (
            f"Only {genuine_treatment} genuine treatment completions were independently "
            f"confirmed; at least {minimum} are required. Retain V4.2."
        )
    elif int(summary.get("material_cluster_count") or 0) < minimum_clusters:
        reason = (
            "The efficacy sample is large enough, but its open-ended mechanisms are too "
            "concentrated for promotion. Continue testing with fresh supply."
        )
    elif float(summary.get("largest_cluster_fraction") or 0.0) > largest_limit:
        reason = (
            "One abstraction mechanism accounts for more than the permitted diversity "
            "share. Continue testing rather than promoting a template."
        )
    elif not noninferiority["all_provisional_model_gates_passed"]:
        reason = (
            "The property is authentic and diverse enough for analysis, but matched quality "
            "non-inferiority is not established. Retain V4.2 and inspect failures."
        )
    else:
        reason = (
            "Model-side efficacy, diversity, and provisional quality gates pass, but blinded "
            "human review and explicit user authorization remain required before promotion."
        )

    protected_after = _protected_hashes(config)
    _assert_hashes_unchanged(protected_before, protected_after)
    if verify_v43_control_source(config) != config["_v4_3_control_source_verification"]:
        raise AbstractionLedError("frozen V4.3 control source changed during V4.4")
    baseline_after = verify_v42_baseline(
        config["_repository_root"], config["paths"]["v4_2_baseline_manifest"]
    )
    if baseline_after != config["_v4_2_baseline_verification"]:
        raise AbstractionLedError("frozen V4.2 baseline changed during V4.4")
    started = read_json(start_path)["started_at_epoch_seconds"]
    completed = round(time.time(), 3)
    accounting = runtime.accounting()
    report = {
        "schema_version": 1,
        "pipeline": V44_PIPELINE,
        "run_id": runtime.run_id,
        "feature": "abstraction_led",
        "experimental_design": V44_DESIGN,
        "completed": True,
        "frozen_v42_baseline": deepcopy(config["_v4_2_baseline_verification"]),
        "frozen_v43_control_source": deepcopy(
            config["_v4_3_control_source_verification"]
        ),
        "funnel": {
            "control_kernels": sum(item["private_arm"] == "control" for item in kernels),
            "treatment_kernels": sum(item["private_arm"] == "treatment" for item in kernels),
            "shard_qualified": sum(
                shard["selected_count"] for shard in selection["shards"]
            ),
            "selected_control": len(by_arm["control"]),
            "selected_treatment": len(by_arm["treatment"]),
            "selection_supply_shortfall": selection["supply_shortfall"],
            "plans_ready": plans_ready,
            "plans_not_ready": len(records) - plans_ready,
            "realized": realized,
            "technical_pass": technical_pass,
            "blind_clear": blind_clear,
            "core": core,
            "reserve": reserve,
            "ambiguous": status_counts["ambiguous"],
            "other": len(records) - core - reserve - status_counts["ambiguous"],
        },
        "efficacy": {
            "audit_status_by_private_arm": audit_status,
            "genuine_treatment_count": genuine_treatment,
            "genuine_control_count": genuine_control,
            "treatment_planner_status_counts": dict(
                sorted(treatment_plan_status.items())
            ),
            "property_audit_non_gating": True,
            "ordinary_task_status_unaffected_by_property": True,
        },
        "quality_noninferiority": noninferiority,
        "coverage_retention": {
            "ordinary_control_kernel_supply_reused": True,
            "ordinary_concept_prompt_unchanged": True,
            "shared_quality_gate_applied_to_both_arms": True,
            "other_semantic_treatments_intentionally_stacked": False,
            "acceptance_threshold_changed": False,
        },
        "diversity": diversity,
        "human_review": human,
        "new_idea_calls": {"control": 0, "treatment": len(design["paired_calls"])},
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
                "mechanism_status": _audit_status(audit_by_id[item["task_id"]])
                if item["task_id"] in audit_by_id
                else "not_audited",
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
                    run_dir / "tasks" / item["arm"] / f"{item['task_id']}.arc.json"
                )
                if item.get("complete_task")
                else None,
                "plan_json": str(
                    run_dir / "plans" / item["arm"] / f"{item['task_id']}.json"
                ),
            }
            for item in records
        ],
        "promotion_recommendation": recommendation,
        "promotion_reason": reason,
        "production_promotion_occurred": False,
        "feature_manifest_mutated": False,
        "good_task_bank_mutated_by_experiment": False,
        "production_reservoir_mutated": False,
        "protected_production_state_unchanged": protected_before == protected_after,
        "unresolved_uncertainties": [
            "Blinded human review is pending.",
            "A single matched study is insufficient for production promotion.",
            "The frozen control supply avoids regeneration cost but was sampled earlier in time.",
        ],
    }
    atomic_write_json(
        run_dir / "promotion_recommendation.json",
        {
            "feature": "abstraction_led",
            "recommendation": recommendation,
            "production_promotion_occurred": False,
        },
    )
    atomic_write_json(run_dir / "quality_noninferiority_report.json", noninferiority)
    atomic_write_json(
        run_dir / "coverage_retention_report.json", report["coverage_retention"]
    )
    atomic_write_json(run_dir / "call_and_timing_ledger.json", accounting)
    return runtime.finish(report, _render_report(report))


def rebuild_v44_report(run: str | Path) -> dict[str, Any]:
    path = Path(run).resolve() / "report.json"
    if not path.is_file():
        raise AbstractionLedError(f"V4.4 report not found: {path}")
    report = read_json(path)
    if report.get("pipeline") != V44_PIPELINE:
        raise AbstractionLedError("run is not a V4.4 abstraction-led experiment")
    atomic_write_text(path.parent / "report.md", _render_report(report))
    return report


def v44_human_review_status(run: str | Path) -> dict[str, Any]:
    directory = Path(run).resolve() / "human_review"
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise AbstractionLedError(f"V4.4 review pack not found: {manifest}")
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
    "AbstractionLedError",
    "rebuild_v44_report",
    "run_abstraction_led_experiment",
    "v44_human_review_status",
]
