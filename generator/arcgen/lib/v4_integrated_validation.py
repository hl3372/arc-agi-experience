"""Integrated qualification of every accepted V4-to-V4.4 authoring change.

The harness deliberately validates coexistence at portfolio level.  A task may
receive at most one deliberate semantic/scale intervention; palette treatment
is the only orthogonal transformation.  All production state is copied into a
run-local sandbox and protected by pre/post hashes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence
import fcntl
import json
import math
import time

from .committed_tasks import load_committed_task_registry
from .embeddings import build_embedding_provider
from .lean_v3_rendering import build_contact_sheet, render_episode
from .util import (
    atomic_write_json,
    atomic_write_text,
    content_hash,
    parallel_map,
    read_json,
)
from .v4_diagnostics import complete_task, diagnose_public_arc_task_v4
from .v4_pipeline import (
    V4ControlPipeline,
    V4ReservoirProducer,
    _DataContext,
    _calibration_assignments,
    v4_call_accounting,
)
from .v4_reservoir import (
    append_supply,
    load_reservoir,
    make_proposal,
    memory_digest,
    reservation_digest,
    reservoir_status,
    validate_reservoir,
)
from .v41_experiments import V41ExperimentError, V41Runtime, _anonymous_id, _task_metrics
from .v41_audits import mechanism_diversity_summary
from .v41_integration import (
    _assert_hashes_unchanged,
    _committed_distribution,
    _palette_target,
    _protected_hashes,
    transform_realization_palette,
)
from .v41_palette import (
    background_metrics,
    infer_episode_background,
    measure_background_distribution,
    schedule_backgrounds,
)
from .v41_structural_transfer import (
    _audit as _structural_audit,
    _install_runtime_extensions as _install_structural_extensions,
    validate_structural_plan_response,
)
from .v41_supported_departure import (
    SPECIAL_LABEL_FRAGMENTS,
    SlateEscalationBudget,
    _record_stage,
    _run_blind,
    _run_technical,
    _selection_sandbox_config,
    _validate_realization,
)
from .v41_test_challenge import _board as _editorial_boards
from .v42_logical_leap import (
    _audit as _logical_audit,
    _install_selector_prompt,
    _install_v42_extensions,
    validate_v42_plan_response,
)
from .v43_episode_joint import (
    _install_extensions as _install_v43_extensions,
)
from .v44_abstraction_led import (
    _install_extensions as _install_v44_extensions,
    _normalize_palette_claims,
)
from .v44_functional_scale import (
    _arm_summary,
    _genuinely_expanded,
    _install_scale_extension,
    task_scale_metrics,
)
from .v45_config import verify_v44_baseline
from .v4_integrated_config import INTEGRATED_PIPELINE, ROUTES


class IntegratedQualificationError(V41ExperimentError):
    pass


PRIVATE_LABELS = (
    "ordinary_control",
    "ordinary_current",
    "logical_leap",
    "structural_transfer",
    "functional_scale",
    "episode_joint_inference",
    "abstraction_led",
    "experimental source",
    "private route",
    "control arm",
    "treatment arm",
)

# The arm-blind mechanism auditor is the one downstream role that must be told
# the names of the properties it is being asked to assess.  It still must not
# see the private source/route assignment or control/treatment terminology.
MECHANISM_AUDITOR_FORBIDDEN_LABELS = tuple(
    label
    for label in PRIVATE_LABELS
    if label not in {"episode_joint_inference", "abstraction_led"}
)

GENUINE_STATUSES = {
    "logical_leap": {"genuine_logical_leap"},
    "structural_transfer": {"substantial_structural_transfer"},
    "episode_joint_inference": {"genuine"},
    "abstraction_led": {"genuine"},
}


def _kernel(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
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
    return {key: deepcopy(record[key]) for key in fields}


def _special_id(seed: int, slate_index: int, source: str, index: int) -> str:
    digest = sha256(
        f"integrated:{seed}:{slate_index}:{source}:{index}".encode("utf-8")
    ).hexdigest()
    return "k" + digest[:13]


def _eligible(
    reservoir: Mapping[str, Any], accepted_hashes: set[str]
) -> list[dict[str, Any]]:
    reserved = {item["concept_hash"] for item in reservoir.get("active_reservations", [])}
    return [
        item
        for item in reservoir["proposals"]
        if item["state"] == "proposed"
        and item["concept_hash"] not in accepted_hashes
        and item["concept_hash"] not in reserved
    ]


def _install_extensions(runtime: V41Runtime) -> None:
    _install_v44_extensions(runtime)
    _install_v43_extensions(runtime)
    _install_v42_extensions(runtime)
    _install_structural_extensions(runtime)
    _install_scale_extension(runtime)


def _special_supply(
    runtime: V41Runtime,
    *,
    sandbox: Mapping[str, Any],
    slate_index: int,
    source: str,
    calibration: Sequence[Any],
    context: _DataContext,
    embedder: Any,
) -> list[str]:
    if source not in {"episode_joint_inference", "abstraction_led"}:
        raise IntegratedQualificationError(f"unknown special source {source}")
    ids = [_special_id(int(runtime.config["seed"]), slate_index, source, index) for index in range(1, 5)]
    existing = {item["proposal_id"] for item in load_reservoir(sandbox["paths"]["reservoir"])["proposals"]}
    if set(ids) <= existing:
        return ids
    if set(ids) & existing:
        raise IntegratedQualificationError("special supply was only partially persisted")
    call_id = f"integrated-slate-{slate_index:02d}-special-supply"
    addendum = (
        "v43_episode_joint_idea"
        if source == "episode_joint_inference"
        else "abstraction_idea_addendum"
    )

    def validate(value: dict[str, Any]) -> None:
        if [item["candidate_id"] for item in value["ideas"]] != ids:
            raise IntegratedQualificationError("special source changed assigned IDs")

    response = runtime.call(
        prompt_name="idea_concept",
        composed_with=(addendum,),
        call_id=call_id,
        values={
            "CALIBRATION_DESCRIPTIONS": [item.rule_description for item in calibration],
            "SEARCH_CALL_ID": call_id,
            "CANDIDATE_IDS": ids,
        },
        parent_ids=[],
        role="idea_search",
        arm_id="private-special-supply",
        expected={"search_call_id": call_id, "search_mode": "concept_led"},
        validator=validate,
    )
    vectors = embedder.embed([item["central_relation"] for item in response["ideas"]])
    base_entry = runtime.prompts.entries["idea_concept"]
    extra_entry = runtime.prompts.entries[addendum]
    combined_hash = content_hash(
        {"templates": [base_entry["hash"], extra_entry["hash"]], "source": source}
    )
    proposals = []
    for kernel, vector in zip(response["ideas"], vectors, strict=True):
        proposals.append(
            make_proposal(
                kernel,
                supply_run_id=runtime.run_id,
                supply_cycle_id=f"{runtime.run_id}:special:{slate_index:02d}",
                calibration_task_ids=[item.task_id for item in calibration],
                calibration_descriptions=[item.rule_description for item in calibration],
                calibration_version=runtime.config["reservoir"]["calibration_version"],
                prompt_version=f"{base_entry['version']}+{extra_entry['version']}",
                prompt_hash=combined_hash,
                model=runtime.config["model"]["roles"]["idea_search"]["model"],
                reasoning_effort=runtime.config["model"]["roles"]["idea_search"]["reasoning_effort"],
                source_bank_snapshot_id=context.bank_snapshot_id,
                embedding={
                    "provider": runtime.config["embedding"]["provider"],
                    "model": runtime.config["embedding"]["model"],
                    "dimension": len(vector),
                    "vector_hash": content_hash(vector),
                },
            )
        )
    append_supply(
        sandbox["paths"]["reservoir"],
        proposals,
        hard_capacity=int(sandbox["reservoir"]["hard_capacity"]),
        supply_run_id=runtime.run_id,
        supply_cycle_id=f"{runtime.run_id}:special:{slate_index:02d}",
    )
    return ids


def _top_up_normal_supply(
    config: Mapping[str, Any],
    sandbox: dict[str, Any],
    run_dir: Path,
    *,
    slate_index: int,
    accepted_hashes: set[str],
    excluded_proposal_ids: set[str] | None = None,
) -> dict[str, Any]:
    required = int(config["integrated_qualification"]["eligible_window_size"])
    excluded_proposal_ids = excluded_proposal_ids or set()
    results: list[dict[str, Any]] = []
    state_paths: list[str] = []
    for attempt in range(1, 8):
        rows = [
            item
            for item in _eligible(load_reservoir(sandbox["paths"]["reservoir"]), accepted_hashes)
            if item["proposal_id"] not in excluded_proposal_ids
        ]
        if len(rows) >= required:
            break
        supply_id = f"{config['run']['id']}-integrated-slate-{slate_index:02d}-topup-{attempt:02d}"
        producer = V4ReservoirProducer(sandbox, supply_run_id=supply_id)
        results.append(producer.run(cycles=1, force_between_watermarks=True))
        state_path = run_dir / "selection_graph" / "v4_reservoir_supply" / supply_id / "state.json"
        if state_path.is_file():
            state_paths.append(str(state_path))
    else:
        rows = [
            item
            for item in _eligible(load_reservoir(sandbox["paths"]["reservoir"]), accepted_hashes)
            if item["proposal_id"] not in excluded_proposal_ids
        ]
    if len(rows) < required:
        raise IntegratedQualificationError(
            f"slate {slate_index} retained {len(rows)} ordinary proposals; {required} required"
        )
    return {
        "generated_topup_cycles": len(results),
        "topup_results": results,
        "call_ledger_paths": state_paths,
        "retained_eligible_count": len(rows),
        "reservoir": reservoir_status(
            load_reservoir(sandbox["paths"]["reservoir"]), sandbox["reservoir"]
        ),
    }


def _freeze_integrated_window(
    selector: V4ControlPipeline,
    *,
    special_ids: Sequence[str],
    all_special_ids: Sequence[str],
    slate_index: int,
) -> dict[str, Any]:
    """Freeze exactly 44 ordinary plus four source-blind special proposals."""

    path = Path(selector.config["paths"]["reservoir"])
    window_id = f"window:{selector.run_id}:integrated:{slate_index:02d}"
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        reservoir = load_reservoir(path)
        existing = next(
            (item for item in reservoir["frozen_windows"] if item["window_id"] == window_id),
            None,
        )
        if existing is not None:
            frozen = deepcopy(existing)
        else:
            accepted = selector.context.accepted_concept_hashes()
            eligible = _eligible(reservoir, accepted)
            by_id = {item["proposal_id"]: item for item in eligible}
            if not set(special_ids) <= set(by_id):
                raise IntegratedQualificationError("special proposals are not eligible for their frozen window")
            known_special = set(all_special_ids)
            ordinary = [item for item in eligible if item["proposal_id"] not in known_special]
            if len(ordinary) < 44:
                raise IntegratedQualificationError("fewer than 44 ordinary proposals remain eligible")
            rng = Random(int(selector.config["seed"]) + slate_index * 7907)
            rng.shuffle(ordinary)
            chosen = ordinary[:44] + [by_id[item] for item in special_ids]
            rng.shuffle(chosen)
            ids = [item["proposal_id"] for item in chosen]
            frozen = {
                "window_id": window_id,
                "created_at_epoch_seconds": round(time.time(), 3),
                "seed": int(selector.config["seed"]),
                "window_size": 48,
                "shard_size": 12,
                "proposal_ids": ids,
                "shards": [ids[index : index + 12] for index in range(0, 48, 12)],
                "accepted_bank_snapshot_id": selector.context.bank_snapshot_id,
                "reservation_snapshot_id": content_hash(reservoir["active_reservations"]),
                "rejection_memory_snapshot_id": content_hash(reservoir["rejection_memory"]),
                "reservoir_snapshot_id": content_hash(
                    {
                        "proposals": [
                            (item["proposal_id"], item["state"], item["concept_hash"])
                            for item in reservoir["proposals"]
                        ],
                        "reservations": reservoir["active_reservations"],
                        "memory": reservoir["rejection_memory"],
                    }
                ),
                "integrated_source_blind_window": True,
                "ordinary_proposal_count": 44,
                "special_proposal_count": 4,
            }
            reservoir["frozen_windows"].append(deepcopy(frozen))
            now = round(time.time(), 3)
            for item in chosen:
                item["staleness"] = {
                    "status": "revalidated",
                    "reasons": [],
                    "checked_at_epoch_seconds": now,
                    "disposition": "eligible after latest exact-bank and reservation recheck",
                }
                item["novelty_status"] = {
                    "status": "pending_frozen_window_selector_review",
                    "checked_at_epoch_seconds": now,
                    "bank_snapshot_id": selector.context.bank_snapshot_id,
                    "reservation_snapshot_id": frozen["reservation_snapshot_id"],
                    "reason": "semantic retrieval and source-blind selector review required",
                }
                item["selection_attempts"].append(
                    {
                        "window_id": window_id,
                        "stage": "eligible_window",
                        "decision": "included",
                        "reason": "preregistered 44+4 integrated frozen window",
                        "at_epoch_seconds": now,
                    }
                )
            validate_reservoir(reservoir)
            atomic_write_json(path, reservoir)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    reservoir = load_reservoir(path)
    proposals = {item["proposal_id"]: item for item in reservoir["proposals"]}
    snapshot = {proposal_id: deepcopy(proposals[proposal_id]) for proposal_id in frozen["proposal_ids"]}
    with selector.lock:
        selector.state["frozen_window"] = deepcopy(frozen)
        selector.state["window_proposals"] = snapshot
        selector.state["reservoir_snapshot_at_selection"] = {
            "status": reservoir_status(reservoir, selector.config["reservoir"]),
            "reservoir_snapshot_id": frozen["reservoir_snapshot_id"],
            "memory": memory_digest(reservoir),
            "reservations": reservation_digest(reservoir),
        }
        selector.state["stage_progress"]["frozen_window"] = True
        selector._save()
    data = selector.run_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    atomic_write_json(data / "frozen_eligible_window.json", frozen)
    atomic_write_json(data / "frozen_proposals.json", snapshot)
    return frozen


def _select_slates(
    config: Mapping[str, Any], runtime: V41Runtime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    manifest_path = runtime.run_dir / "selected_kernel_manifest.private.json"
    source_path = runtime.run_dir / "special_source_mapping.private.json"
    if manifest_path.exists() and source_path.exists():
        value = read_json(manifest_path)
        return value["development_sample"], value["slates"], read_json(source_path)["mapping"]

    context = _DataContext(config)
    embedder = build_embedding_provider(config["embedding"], config["paths"]["cache"])
    slate_count = int(config["integrated_qualification"]["independent_slates"])
    calibrations = _calibration_assignments(
        context.generator_pool,
        count=int(config["reservoir"]["calibration_examples_per_call"]),
        calls=slate_count,
        embedder=embedder,
        seed=int(config["seed"]) + 91003,
    )
    source_cycle = list(config["integrated_qualification"]["special_source_cycle"])
    selected: list[dict[str, Any]] = []
    slates: list[dict[str, Any]] = []
    source_mapping: dict[str, str] = {}
    for slate_index in range(1, slate_count + 1):
        sandbox = _selection_sandbox_config(config, runtime.run_dir, slate_index=slate_index)
        sandbox["selection"]["eligible_window_size"] = 48
        sandbox["selection"]["maximum_optional_development_suggestions"] = 3
        sandbox["reservoir"]["hard_capacity"] = 512
        sandbox["reservoir"]["high_watermark"] = 480
        selector = V4ControlPipeline(sandbox)
        supply = _top_up_normal_supply(
            config,
            sandbox,
            runtime.run_dir,
            slate_index=slate_index,
            accepted_hashes=selector.context.accepted_concept_hashes(),
            excluded_proposal_ids=set(source_mapping),
        )
        source = source_cycle[(slate_index - 1) % len(source_cycle)]
        special_ids = _special_supply(
            runtime,
            sandbox=sandbox,
            slate_index=slate_index,
            source=source,
            calibration=calibrations[slate_index - 1],
            context=context,
            embedder=embedder,
        )
        source_mapping.update({candidate_id: source for candidate_id in special_ids})
        frozen = _freeze_integrated_window(
            selector,
            special_ids=special_ids,
            all_special_ids=list(source_mapping),
            slate_index=slate_index,
        )
        selector_extension = _install_selector_prompt(selector, maximum_nominations=3)
        briefs = selector.select_slate()
        ordered = sorted(briefs, key=lambda item: int(item["development_priority"]))
        maximum = int(config["integrated_qualification"]["maximum_development_per_slate"])
        sample: list[dict[str, Any]] = []
        retained_special = 0
        for brief in ordered:
            candidate_id = str(brief["candidate_id"])
            is_special = candidate_id in source_mapping
            if is_special and retained_special >= int(
                config["integrated_qualification"]["maximum_special_source_per_slate"]
            ):
                continue
            record = selector.state["records"][candidate_id]
            sample.append(
                {
                    "source_candidate_id": candidate_id,
                    "slate_index": slate_index,
                    "source_development_priority": int(record["development_priority"]),
                    "kernel": _kernel(record),
                    "editorial_brief": deepcopy(record["editorial_brief"]),
                    "private_source": source_mapping.get(candidate_id, "ordinary"),
                }
            )
            retained_special += int(is_special)
            if len(sample) == maximum:
                break
        selected.extend(sample)
        slates.append(
            {
                "slate_index": slate_index,
                "selection_run_id": sandbox["run"]["id"],
                "frozen_window": deepcopy(frozen),
                "full_selected_slate": [item["candidate_id"] for item in ordered],
                "development_sample": [item["source_candidate_id"] for item in sample],
                "special_source_offered": source,
                "special_source_selected_count": retained_special,
                "supply": supply,
                "state_path": str(selector.state_path),
                "selector_extension": selector_extension,
            }
        )
    value = {
        "schema_version": 1,
        "pipeline": INTEGRATED_PIPELINE,
        "selection_precedes_private_routing": True,
        "source_labels_hidden_from_selectors": True,
        "development_sample": selected,
        "slates": slates,
    }
    atomic_write_json(manifest_path, value)
    atomic_write_json(
        source_path,
        {
            "schema_version": 1,
            "private": True,
            "mapping": source_mapping,
        },
    )
    atomic_write_json(
        runtime.run_dir / "selected_kernel_manifest.public.json",
        {
            "schema_version": 1,
            "source_labels_removed": True,
            "development_count": len(selected),
            "slates": [
                {
                    "slate_index": item["slate_index"],
                    "frozen_window_proposal_ids": item["frozen_window"]["proposal_ids"],
                    "development_count": len(item["development_sample"]),
                }
                for item in slates
            ],
        },
    )
    return selected, slates, source_mapping


def _control_selection(
    rows: Sequence[Mapping[str, Any]], *, count: int, seed: int, slate_index: int
) -> set[str]:
    ordinary = [item for item in rows if item["private_source"] == "ordinary"]
    if len(ordinary) < count:
        raise IntegratedQualificationError("not enough ordinary-source tasks for embedded controls")
    ids = [str(item["source_candidate_id"]) for item in ordinary]
    rng = Random(seed + slate_index * 13007)
    tie = {item: rng.random() for item in ids}
    mode_total = Counter(item["kernel"]["search_mode"] for item in ordinary)
    choices = []
    for picked in combinations(ids, count):
        chosen = set(picked)
        by_mode = Counter(
            item["kernel"]["search_mode"]
            for item in ordinary
            if item["source_candidate_id"] in chosen
        )
        imbalance = sum(
            abs(len(ordinary) * by_mode[mode] - count * total)
            for mode, total in mode_total.items()
        )
        priority_spread = sum(
            int(item["source_development_priority"])
            for item in ordinary
            if item["source_candidate_id"] in chosen
        )
        choices.append((imbalance, abs(priority_spread - count * 4.5), sum(tie[item] for item in chosen), tuple(sorted(chosen)), chosen))
    return min(choices, key=lambda item: item[:-1])[-1]


def assign_integrated_routes(
    selected: Sequence[Mapping[str, Any]], *, seed: int, route_cycle: Sequence[Sequence[str]]
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    by_slate: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_slate[int(row["slate_index"])].append(row)
    for slate_index, rows in sorted(by_slate.items()):
        available_ordinary = sum(item["private_source"] == "ordinary" for item in rows)
        # Two concurrent controls are the planned ceiling.  A quality selector
        # may legitimately return fewer than two ordinary kernels (or none),
        # and rejected kernels must never be revived merely to fill the control
        # cohort.  The shortfall is retained in the route and funnel ledgers.
        control_count = min(2, available_ordinary)
        controls = _control_selection(
            rows, count=control_count, seed=seed, slate_index=slate_index
        )
        for row in rows:
            candidate_id = str(row["source_candidate_id"])
            if candidate_id in controls:
                assignments[candidate_id] = "ordinary_control"
            elif row["private_source"] in {
                "episode_joint_inference",
                "abstraction_led",
            }:
                assignments[candidate_id] = str(row["private_source"])

        desired = list(route_cycle[(slate_index - 1) % len(route_cycle)])
        unassigned = [
            row
            for row in sorted(rows, key=lambda item: int(item["source_development_priority"]))
            if str(row["source_candidate_id"]) not in assignments
            and row["private_source"] == "ordinary"
        ]
        for route in desired:
            candidates = unassigned
            if route in {"logical_leap", "structural_transfer"}:
                candidates = [
                    item
                    for item in unassigned
                    if item["editorial_brief"].get("optional_development_suggestion") is not None
                ]
            if not candidates:
                continue
            if route == "functional_scale":
                chosen = min(
                    candidates,
                    key=lambda item: content_hash(
                        {
                            "seed": seed,
                            "slate": slate_index,
                            "scale": item["source_candidate_id"],
                        }
                    ),
                )
            else:
                chosen = candidates[0]
            candidate_id = str(chosen["source_candidate_id"])
            assignments[candidate_id] = route
            unassigned = [item for item in unassigned if item["source_candidate_id"] != candidate_id]
        for row in rows:
            assignments.setdefault(str(row["source_candidate_id"]), "ordinary_current")
    if set(assignments.values()) - ROUTES:
        raise IntegratedQualificationError("route assignment produced an unknown route")
    if set(assignments) != {str(item["source_candidate_id"]) for item in selected}:
        raise IntegratedQualificationError("route assignment omitted selected candidates")
    return assignments


def assign_fresh_scale_integrated_routes(
    selected: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    scale_target_slates: Sequence[int],
    controls_after_scale_reservation: int,
    semantic_route_cycle: Sequence[str],
) -> dict[str, str]:
    """Route a fresh integrated scale qualification after quality selection.

    The scale opportunity is reserved before embedded controls only on the
    preregistered target slates.  This prevents a small but otherwise strong
    selected slate from losing its sole eligible scale candidate merely
    because controls were assigned first.  No rejected kernel is revived and
    no candidate receives more than one deliberate route.
    """

    targets = {int(value) for value in scale_target_slates}
    if not semantic_route_cycle or set(semantic_route_cycle) - {
        "logical_leap",
        "structural_transfer",
    }:
        raise IntegratedQualificationError(
            "fresh scale semantic route cycle must contain only accepted planning routes"
        )
    assignments: dict[str, str] = {}
    by_slate: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_slate[int(row["slate_index"])].append(row)

    for slate_index, rows in sorted(by_slate.items()):
        for row in rows:
            candidate_id = str(row["source_candidate_id"])
            if row["private_source"] in {
                "episode_joint_inference",
                "abstraction_led",
            }:
                assignments[candidate_id] = str(row["private_source"])

        ordinary = [
            row
            for row in sorted(
                rows,
                key=lambda item: (
                    int(item["source_development_priority"]),
                    content_hash(
                        {
                            "seed": seed,
                            "slate": slate_index,
                            "candidate": item["source_candidate_id"],
                        }
                    ),
                ),
            )
            if row["private_source"] == "ordinary"
            and str(row["source_candidate_id"]) not in assignments
        ]

        # Every candidate in this list has already survived the independent
        # portfolio selector.  Choosing the highest-priority one changes only
        # private authoring routing, never development rank or eligibility.
        if slate_index in targets and ordinary:
            scale = ordinary.pop(0)
            assignments[str(scale["source_candidate_id"])] = "functional_scale"

        control_count = min(int(controls_after_scale_reservation), len(ordinary))
        controls = _control_selection(
            ordinary,
            count=control_count,
            seed=seed + 44041,
            slate_index=slate_index,
        )
        for row in ordinary:
            candidate_id = str(row["source_candidate_id"])
            if candidate_id in controls:
                assignments[candidate_id] = "ordinary_control"

        remaining = [
            row
            for row in ordinary
            if str(row["source_candidate_id"]) not in assignments
        ]
        semantic_route = semantic_route_cycle[
            (slate_index - 1) % len(semantic_route_cycle)
        ]
        semantic_candidates = [
            row
            for row in remaining
            if row["editorial_brief"].get("optional_development_suggestion") is not None
        ]
        if semantic_candidates:
            chosen = semantic_candidates[0]
            assignments[str(chosen["source_candidate_id"])] = semantic_route

        for row in rows:
            assignments.setdefault(
                str(row["source_candidate_id"]),
                "ordinary_current",
            )

    if set(assignments.values()) - ROUTES:
        raise IntegratedQualificationError(
            "fresh integrated scale route assignment produced an unknown route"
        )
    if set(assignments) != {
        str(item["source_candidate_id"]) for item in selected
    }:
        raise IntegratedQualificationError(
            "fresh integrated scale route assignment omitted selected candidates"
        )
    return assignments


def _aliases(selected: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    rng = Random(seed + 71881)
    used: set[str] = set()
    ids = [str(item["source_candidate_id"]) for item in selected]
    rng.shuffle(ids)
    return {candidate_id: _anonymous_id(rng, used) for candidate_id in ids}


def _freeze_routes(
    config: Mapping[str, Any],
    runtime: V41Runtime,
    selected: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str],
) -> dict[str, Any]:
    path = runtime.run_dir / "integrated_route_assignment.private.json"
    if path.exists():
        return read_json(path)
    fresh_scale = config.get("fresh_scale_integration")
    if isinstance(fresh_scale, Mapping):
        routes = assign_fresh_scale_integrated_routes(
            selected,
            seed=int(config["seed"]),
            scale_target_slates=fresh_scale["scale_target_slates"],
            controls_after_scale_reservation=int(
                fresh_scale["embedded_controls_after_scale_reservation"]
            ),
            semantic_route_cycle=fresh_scale["semantic_route_cycle"],
        )
        assignment_strategy = "fresh_scale_reserved_before_controls_v1"
    else:
        routes = assign_integrated_routes(
            selected,
            seed=int(config["seed"]),
            route_cycle=config["integrated_qualification"]["planning_route_cycle"],
        )
        assignment_strategy = "frozen_integrated_control_v1"
    rows = [
        {
            "source_candidate_id": item["source_candidate_id"],
            "opaque_task_id": aliases[str(item["source_candidate_id"])],
            "slate_index": int(item["slate_index"]),
            "development_priority": int(item["source_development_priority"]),
            "search_mode": item["kernel"]["search_mode"],
            "private_source": item["private_source"],
            "private_route": routes[str(item["source_candidate_id"])],
        }
        for item in selected
    ]
    value = {
        "schema_version": 1,
        "pipeline": INTEGRATED_PIPELINE,
        "assignment_seed": int(config["seed"]),
        "selection_completed_before_assignment": True,
        "one_deliberate_intervention_per_candidate": True,
        "source_and_route_hidden_downstream": True,
        "functional_scale_remains_unpromoted": True,
        "assignment_strategy": assignment_strategy,
        "assignments": rows,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        runtime.run_dir / "integrated_route_assignment.public.json",
        {
            "schema_version": 1,
            "selection_completed_before_assignment": True,
            "counts": dict(sorted(Counter(item["private_route"] for item in rows).items())),
            "source_and_route_labels_hidden_downstream": True,
            "assignment_strategy": assignment_strategy,
        },
    )
    return value


def _initialize_records(
    runtime: V41Runtime,
    selected: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Any],
) -> None:
    with runtime.lock:
        if "semantic_records" in runtime.state:
            return
        assigned = {item["source_candidate_id"]: item for item in assignment["assignments"]}
        records = {}
        for row in selected:
            source_id = str(row["source_candidate_id"])
            private = assigned[source_id]
            task_id = str(private["opaque_task_id"])
            records[task_id] = {
                "task_id": task_id,
                "source_candidate_id": source_id,
                "arm": "single",
                "private_source": private["private_source"],
                "private_route": private["private_route"],
                "slate_index": int(row["slate_index"]),
                "source_development_priority": int(row["source_development_priority"]),
                "search_mode": row["kernel"]["search_mode"],
                "kernel": deepcopy(row["kernel"]),
                "editorial_brief": deepcopy(row["editorial_brief"]),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
        runtime.state["semantic_records"] = records
        runtime.state["integrated_pipeline"] = INTEGRATED_PIPELINE
        runtime.save()


def _plan_records(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]

    def one(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        record = deepcopy(records[task_id])
        route = str(record["private_route"])
        brief = deepcopy(record["editorial_brief"])
        if route not in {"logical_leap", "structural_transfer"}:
            brief["optional_development_suggestion"] = None
        values = {
            "CANDIDATE": {**deepcopy(record["kernel"]), "candidate_id": task_id},
            "EDITORIAL_BRIEF": {**brief, "candidate_id": task_id},
            "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(brief["route"]),
            "CANDIDATE_ID": task_id,
        }
        try:
            if route == "ordinary_control":
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
                    "decision": "ordinary_control",
                    "decision_reason": "embedded pre-palette V4 authoring control",
                    "concrete_plan": plan,
                }
            elif route == "ordinary_current":
                plan = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("v44_quality_plan",),
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="single-authoring",
                    expected={"candidate_id": task_id},
                )
                result = {
                    "candidate_id": task_id,
                    "decision": "ordinary_current",
                    "decision_reason": "current V4.4 quality planning without an optional intervention",
                    "concrete_plan": plan,
                }
            elif route == "logical_leap":
                result = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("v44_quality_plan", "logical_leap_planner_addendum"),
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="single-authoring",
                    expected={"candidate_id": task_id},
                )
                result["structural_transfer_contract"] = None
                validate_v42_plan_response(task_id=task_id, route=route, value=result)
            elif route == "structural_transfer":
                result = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("v44_quality_plan", "structural_planner_addendum"),
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="single-authoring",
                    expected={"candidate_id": task_id},
                )
                result["logical_leap_contract"] = None
                validate_structural_plan_response(task_id=task_id, nominated=True, value=result)
            elif route == "functional_scale":
                result = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("v44_quality_plan", "functional_scale_addendum"),
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="single-authoring",
                    expected={"candidate_id": task_id},
                )
            elif route == "episode_joint_inference":
                result = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("v44_quality_plan", "v43_episode_joint_plan"),
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="single-authoring",
                    expected={"candidate_id": task_id},
                )
            elif route == "abstraction_led":
                result = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("v44_quality_plan", "abstraction_plan_addendum"),
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="single-authoring",
                    expected={"candidate_id": task_id},
                )
            else:
                raise IntegratedQualificationError(f"unknown private route {route}")
            plan = result.get("concrete_plan")
            if plan is not None and plan.get("candidate_id") != task_id:
                raise IntegratedQualificationError("planner changed nested candidate ID")
            if plan is not None and route != "ordinary_control":
                plan = _normalize_palette_claims(plan)
                result["concrete_plan"] = plan
            if plan is not None and plan.get("status") == "abort":
                decision = result.get("decision") or result.get("property_status") or result.get("scale_decision")
                if decision != "abort":
                    raise IntegratedQualificationError("aborted nested plan did not use abort decision")
            return task_id, result, None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [task_id for task_id, record in records.items() if "planning_result" not in record]
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
                status="planning_operational_failure",
                planning_error=error,
            )
            continue
        plan = result.get("concrete_plan")
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


def _palette_assignments(runtime: V41Runtime) -> dict[str, int | None]:
    path = runtime.run_dir / "palette_assignments.json"
    if path.exists():
        return read_json(path)["assignments"]
    records = runtime.state["semantic_records"]
    current_ids = sorted(
        task_id
        for task_id, record in records.items()
        if record["private_route"] != "ordinary_control"
    )
    target, reference = _palette_target(runtime.config)
    committed = _committed_distribution(runtime.config)
    current = schedule_backgrounds(
        current_ids,
        seed=int(runtime.config["seed"]),
        current_counts=committed["counts"],
        target_distribution=target,
        eligible={task_id: True for task_id in current_ids},
    )
    assignments = {
        task_id: (None if record["private_route"] == "ordinary_control" else current[task_id])
        for task_id, record in records.items()
    }
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "control_palette_disabled": True,
            "current_palette_portfolio_scheduled_together": True,
            "assignments": assignments,
            "target_distribution": {str(key): value for key, value in target.items()},
            "real_reference": reference,
            "committed_distribution": committed,
        },
    )
    return assignments


def _realize_records(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]
    assignments = _palette_assignments(runtime)

    def one(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        record = deepcopy(records[task_id])
        plan = record.get("concrete_plan")
        if not plan or plan.get("status") != "ready":
            return task_id, None, "plan is not ready"
        try:
            realization = runtime.call(
                prompt_name="task_realizer",
                call_id=f"{task_id}-realization",
                values={"CONCRETE_PLAN": plan, "CANDIDATE_ID": task_id},
                parent_ids=[task_id],
                role="task_realizer",
                arm_id="single-authoring",
                expected={"candidate_id": task_id},
                forbidden_prompt_fragments=PRIVATE_LABELS,
            )
            if record["private_route"] == "ordinary_control":
                final_plan = plan
                final_realization = realization
                inferred = (
                    infer_episode_background(complete_task(realization))
                    if realization.get("status") == "realized"
                    else None
                )
                palette = {
                    "status": "embedded_control_not_scheduled",
                    "safe": None,
                    "applied": False,
                    "source_background_color": inferred,
                    "requested_background_color": None,
                    "realized_background_color": inferred,
                    "mapping": {str(color): color for color in range(10)},
                }
            else:
                final_plan, final_realization, palette = transform_realization_palette(
                    candidate_id=task_id,
                    plan=plan,
                    realization=realization,
                    target_background=assignments[task_id],
                    palette_config=runtime.config["palette_experiment"],
                )
            return task_id, {
                "concrete_plan": final_plan,
                "realization": final_realization,
                "palette_treatment": palette,
            }, None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [
        task_id
        for task_id, record in records.items()
        if record.get("status") in {"plan_ready", "realization_operational_failure"}
        and "realization" not in record
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
                "realization",
                status="realization_operational_failure",
                realization_error=error,
            )
            continue
        okay, reason, diagnostics, fidelity = _validate_realization(
            task_id, result["concrete_plan"], result["realization"]
        )
        if not okay:
            _record_stage(
                runtime,
                task_id,
                "realization",
                status="realization_failure",
                realization=result["realization"],
                concrete_plan=result["concrete_plan"],
                palette_treatment=result["palette_treatment"],
                plan_fidelity=fidelity,
                realization_error=reason,
            )
            continue
        complete = complete_task(result["realization"])
        public_diagnostics = diagnose_public_arc_task_v4(result["realization"]["task"])
        _record_stage(
            runtime,
            task_id,
            "realization",
            status="realized",
            realization=result["realization"],
            concrete_plan=result["concrete_plan"],
            complete_task=complete,
            palette_treatment=result["palette_treatment"],
            plan_fidelity=fidelity,
            deterministic_diagnostics=diagnostics,
            technical_diagnostics=public_diagnostics,
        )
        directory = runtime.run_dir / "tasks" / "single"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / f"{task_id}.arc.json", complete)
        atomic_write_json(directory / f"{task_id}.authoring.json", result)


def _budgets(runtime: V41Runtime) -> dict[tuple[int, str], SlateEscalationBudget]:
    limit = int(runtime.config["integrated_qualification"]["escalation_calls_per_slate"])
    slates = sorted({int(item["slate_index"]) for item in runtime.state["semantic_records"].values()})
    with runtime.lock:
        usage = runtime.state.setdefault("integrated_escalation_usage", {})
        for slate in slates:
            usage.setdefault(str(slate), 0)
        runtime.save()
    result: dict[tuple[int, str], SlateEscalationBudget] = {}
    for slate in slates:
        key = str(slate)

        def persist(used: int, *, usage_key: str = key) -> None:
            with runtime.lock:
                runtime.state["integrated_escalation_usage"][usage_key] = used
                runtime.save()

        result[(slate, "single")] = SlateEscalationBudget(
            limit,
            used=int(runtime.state["integrated_escalation_usage"][key]),
            on_change=persist,
        )
    return result


def _joint_abstraction_audit(runtime: V41Runtime) -> list[dict[str, Any]]:
    path = runtime.run_dir / "joint_abstraction_audit.json"
    if path.exists():
        return read_json(path)["judgments"]
    records = [
        item for item in runtime.state["semantic_records"].values() if item.get("complete_task")
    ]
    Random(int(runtime.config["seed"]) + 99181).shuffle(records)
    batch = int(runtime.config["integrated_qualification"]["audit_batch_size"])
    judgments: list[dict[str, Any]] = []
    calls = []
    for offset in range(0, len(records), batch):
        chunk = records[offset : offset + batch]
        audit_id = f"integrated-joint-abstraction-{offset // batch + 1:02d}"
        packages = [
            {
                "task_id": record["task_id"],
                "episode": record["complete_task"],
                "concise_rule": record["realization"]["authoring_metadata"]["intended_rule"],
                "concrete_plan": record["concrete_plan"],
            }
            for record in chunk
        ]

        def validate(value: dict[str, Any]) -> None:
            returned = [item["task_id"] for item in value["judgments"]]
            expected = {item["task_id"] for item in packages}
            if len(returned) != len(set(returned)) or set(returned) != expected:
                raise IntegratedQualificationError("mechanism auditor omitted or duplicated tasks")

        response = runtime.call(
            prompt_name="mechanism_auditor",
            call_id=audit_id,
            values={
                "REQUESTED_PROPERTIES": ["episode_joint_inference", "abstraction_led"],
                "TASK_PACKAGES": packages,
                "AUDIT_ID": audit_id,
            },
            parent_ids=[item["task_id"] for item in packages],
            role="editorial_board",
            arm_id="blinded-integrated-mechanism-audit",
            expected={
                "audit_id": audit_id,
                "requested_properties": ["episode_joint_inference", "abstraction_led"],
                "non_gating_confirmation": True,
            },
            validator=validate,
            forbidden_prompt_fragments=MECHANISM_AUDITOR_FORBIDDEN_LABELS,
        )
        calls.append(response)
        judgments.extend(response["judgments"])
    atomic_write_json(
        path,
        {
            "source_and_route_hidden": True,
            "non_gating": True,
            "calls": calls,
            "judgments": judgments,
        },
    )
    return judgments


def _audit_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["task_id"]): deepcopy(dict(item)) for item in rows}


def _feature_audits(runtime: V41Runtime) -> dict[str, Any]:
    """Run the accepted non-gating audits once over every completed episode."""

    path = runtime.run_dir / "integrated_mechanism_audits.json"
    if path.exists():
        return read_json(path)
    logical = _logical_audit(runtime)
    structural = _structural_audit(runtime)
    joint_abstraction = _joint_abstraction_audit(runtime)
    value = {
        "schema_version": 1,
        "non_gating": True,
        "route_and_source_hidden": True,
        "logical_leap": logical,
        "structural_transfer": structural,
        "joint_and_abstraction": joint_abstraction,
    }
    atomic_write_json(path, value)
    return value


def _feature_statuses(
    record: Mapping[str, Any], audits: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    task_id = str(record["task_id"])
    logical = _audit_lookup(audits["logical_leap"]).get(task_id, {})
    structural = _audit_lookup(audits["structural_transfer"]).get(task_id, {})
    combined = _audit_lookup(audits["joint_and_abstraction"]).get(task_id, {})
    joint = combined.get("episode_joint_inference") or {}
    abstraction = combined.get("abstraction_led") or {}
    signatures = combined.get("open_ended_mechanism_signatures") or {}
    expanded = _genuinely_expanded(record)
    palette = record.get("palette_treatment") or {}
    return {
        "logical_leap": {
            "status": logical.get("status", "not_audited"),
            "genuine": logical.get("status") == "genuine_logical_leap",
            "reason": logical.get("reason"),
            "signature": logical.get("open_ended_signature", ""),
        },
        "structural_transfer": {
            "status": structural.get("status", "not_audited"),
            "genuine": structural.get("status") == "substantial_structural_transfer",
            "reason": structural.get("reason"),
            "signature": structural.get("open_ended_signature", ""),
        },
        "episode_joint_inference": {
            "status": joint.get("status", "not_audited"),
            "genuine": joint.get("status") == "genuine",
            "reason": joint.get("reason"),
            "signature": signatures.get("episode_joint_inference", ""),
        },
        "abstraction_led": {
            "status": abstraction.get("status", "not_audited"),
            "genuine": abstraction.get("status") == "genuine",
            "reason": abstraction.get("reason"),
            "signature": signatures.get("abstraction_led", ""),
        },
        "functional_scale": {
            "status": "functionally_expanded" if expanded else "not_confirmed",
            "genuine": expanded,
            "reason": record.get("planning_result", {}).get("scale_reason"),
            "signature": (
                record.get("planning_result", {}).get("scale_reason", "") if expanded else ""
            ),
        },
        "palette_diversity": {
            "status": palette.get("status", "not_recorded"),
            "genuine": bool(palette.get("applied")),
            "reason": palette.get("reason"),
            "signature": "",
        },
    }


def _embedding_diversity(
    runtime: V41Runtime,
    records: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Descriptive open-ended similarity analysis; never an acceptance gate."""

    path = runtime.run_dir / "diversity_cluster_report.json"
    if path.exists():
        return read_json(path)
    result: dict[str, Any] = {
        "schema_version": 1,
        "non_gating": True,
        "method": "open-ended signatures plus semantic-embedding nearest neighbors",
        "closed_taxonomy_used": False,
        "features": {},
    }
    embedder = build_embedding_provider(
        runtime.config["embedding"], runtime.config["paths"]["cache"]
    )
    for feature in (
        "logical_leap",
        "structural_transfer",
        "episode_joint_inference",
        "abstraction_led",
    ):
        positive = [
            str(item["task_id"])
            for item in records
            if statuses[str(item["task_id"])][feature]["genuine"]
        ]
        signatures = {
            task_id: str(statuses[task_id][feature].get("signature") or "")
            for task_id in positive
        }
        feature_result: dict[str, Any] = {
            "genuine_instance_count": len(positive),
            "task_ids": positive,
            "signatures": signatures,
            "same_procedure_tasks_counted_as_separate_successes": False,
            "promotion_reassessment_not_performed": True,
        }
        if len(positive) < 2:
            feature_result["semantic_embedding_analysis"] = {
                "status": "not_applicable",
                "reason": "fewer than two genuine instances in this qualification",
            }
        else:
            try:
                vectors = embedder.embed([signatures[item] for item in positive])
                pairs = []
                for left in range(len(positive)):
                    for right in range(left + 1, len(positive)):
                        score = sum(a * b for a, b in zip(vectors[left], vectors[right]))
                        pairs.append(
                            {
                                "task_id_a": positive[left],
                                "task_id_b": positive[right],
                                "cosine_similarity": round(score, 6),
                            }
                        )
                feature_result["semantic_embedding_analysis"] = {
                    "status": "complete",
                    "provider": runtime.config["embedding"]["provider"],
                    "model": runtime.config["embedding"]["model"],
                    "pairwise_similarities": sorted(
                        pairs, key=lambda item: -item["cosine_similarity"]
                    ),
                    "nearest_pairs_for_human_cluster_review": sorted(
                        pairs, key=lambda item: -item["cosine_similarity"]
                    )[: min(12, len(pairs))],
                }
            except Exception as exc:
                feature_result["semantic_embedding_analysis"] = {
                    "status": "operational_failure",
                    "error": str(exc),
                }
        result["features"][feature] = feature_result
    atomic_write_json(path, result)
    return result


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    return round(float(numerator) / float(denominator), 6) if denominator else None


def _quality_and_coverage(
    runtime: V41Runtime,
    records: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = [item for item in records if item["private_route"] == "ordinary_control"]
    current = [item for item in records if item["private_route"] != "ordinary_control"]
    quality_by_cohort = {
        "embedded_pre_palette_v4_control": _arm_summary(controls),
        "integrated_current_routes": _arm_summary(current),
        "all_tasks": _arm_summary(records),
        "by_assigned_route": {
            route: _arm_summary([item for item in records if item["private_route"] == route])
            for route in sorted(ROUTES)
        },
    }
    control = quality_by_cohort["embedded_pre_palette_v4_control"]
    treatment = quality_by_cohort["integrated_current_routes"]
    core_ratio = (
        _safe_ratio(treatment["core_yield"], control["core_yield"])
        if control["core_yield"] not in {None, 0}
        else None
    )
    useful_ratio = (
        _safe_ratio(treatment["core_or_reserve_yield"], control["core_or_reserve_yield"])
        if control["core_or_reserve_yield"] not in {None, 0}
        else None
    )
    core_rows = [item for item in records if item.get("intrinsic_quality") == "core_quality"]
    reserve_rows = [item for item in records if item.get("intrinsic_quality") == "reserve_quality"]
    core_serious = [
        item["task_id"]
        for item in core_rows
        if item.get("status") == "ambiguous"
        or (
            item.get("status") not in {"blind_pass", "editorial_complete"}
            and any(
                bool(result.get("serious_different_output_alternative"))
                for result in item.get("blind_results", [])
            )
        )
    ]
    hidden_mismatches = [
        item["task_id"]
        for item in records
        if item.get("technical_results")
        and item["technical_results"][-1].get("hidden_output_match") is False
    ]
    artificially_inflated = [
        item["task_id"]
        for item in records
        if (
            ((item.get("realization") or {}).get("authoring_metadata") or {}).get(
                "canvas_use_classification"
            )
        )
        == "artificially_inflated"
    ]
    gates = runtime.config["integrated_qualification"]["quality_gates"]
    quality = {
        "schema_version": 1,
        "cohorts": quality_by_cohort,
        "noninferiority": {
            "core_yield_ratio": core_ratio,
            "required_floor": gates["core_yield_ratio_floor"],
            "core_yield_gate": core_ratio is not None
            and core_ratio >= float(gates["core_yield_ratio_floor"]),
            "core_or_reserve_yield_ratio": useful_ratio,
            "required_useful_floor": gates["core_or_reserve_yield_ratio_floor"],
            "core_or_reserve_gate": useful_ratio is not None
            and useful_ratio >= float(gates["core_or_reserve_yield_ratio_floor"]),
            "embedded_control_is_fresh_and_concurrent": True,
            "comparison_is_portfolio_level_not_task_paired": True,
        },
        "one_answer_and_mechanical_safety": {
            "core_serious_ambiguity_task_ids": core_serious,
            "core_serious_ambiguity_gate": not core_serious,
            "hidden_output_mismatch_task_ids": hidden_mismatches,
            "hidden_output_exact_match_gate": not hidden_mismatches,
            "artificially_inflated_task_ids": artificially_inflated,
            "artificial_inflation_gate": not artificially_inflated,
        },
    }

    semantic = (
        "logical_leap",
        "structural_transfer",
        "episode_joint_inference",
        "abstraction_led",
    )
    assigned_counts = Counter(str(item["private_route"]) for item in records)
    core_feature_ids = {
        feature: [
            str(item["task_id"])
            for item in core_rows
            if statuses[str(item["task_id"])][feature]["genuine"]
        ]
        for feature in semantic
    }
    all_feature_ids = {
        feature: [
            str(item["task_id"])
            for item in records
            if statuses[str(item["task_id"])][feature]["genuine"]
        ]
        for feature in (*semantic, "functional_scale")
    }
    ordinary_core = [
        str(item["task_id"])
        for item in core_rows
        if item["private_route"] in {"ordinary_control", "ordinary_current"}
    ]
    overlap = {}
    for left, right in combinations(semantic, 2):
        members = sorted(set(core_feature_ids[left]) & set(core_feature_ids[right]))
        overlap[f"{left}__{right}"] = {"count": len(members), "task_ids": members}
    bands = runtime.config["integrated_qualification"]["rolling_core_bands"]
    band_results = {}
    for feature in semantic:
        share = _safe_ratio(len(core_feature_ids[feature]), len(core_rows))
        specification = bands[feature]
        if len(core_rows) < 20:
            disposition = "descriptive_small_window"
        elif share is not None and share > specification["soft_ceiling"]:
            disposition = "above_soft_ceiling"
        elif share is not None and share < specification["lower"]:
            disposition = "below_operating_band"
        else:
            disposition = "within_operating_band"
        band_results[feature] = {
            "genuine_core_count": len(core_feature_ids[feature]),
            "genuine_core_task_ids": core_feature_ids[feature],
            "core_share": share,
            "operating_band": deepcopy(specification),
            "disposition": disposition,
        }
    ordinary_share = _safe_ratio(len(ordinary_core), len(core_rows))
    coverage = {
        "schema_version": 1,
        "developed_count": len(records),
        "realized_count": sum(bool(item.get("complete_task")) for item in records),
        "core_count": len(core_rows),
        "reserve_count": len(reserve_rows),
        "assigned_route_counts": dict(sorted(assigned_counts.items())),
        "assigned_route_shares": {
            key: _safe_ratio(value, len(records)) for key, value in sorted(assigned_counts.items())
        },
        "genuine_feature_counts_all_completed": {
            key: len(value) for key, value in all_feature_ids.items()
        },
        "genuine_feature_task_ids_all_completed": all_feature_ids,
        "core_feature_bands": band_results,
        "ordinary_unforced_core": {
            "count": len(ordinary_core),
            "task_ids": ordinary_core,
            "share": ordinary_share,
            "minimum": bands["ordinary_unforced_minimum"],
            "gate": ordinary_share is not None
            and ordinary_share >= float(bands["ordinary_unforced_minimum"]),
        },
        "core_feature_overlap_matrix": overlap,
        "quality_not_relaxed_for_feature_supply": True,
        "special_source_ceiling_not_quota": True,
        "one_deliberate_intervention_per_candidate": all(
            item["private_route"] in ROUTES for item in records
        ),
    }
    atomic_write_json(runtime.run_dir / "quality_noninferiority_report.json", quality)
    atomic_write_json(runtime.run_dir / "coverage_retention_report.json", coverage)
    return quality, coverage


def _palette_and_scale_report(
    runtime: V41Runtime, records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    palette_config = runtime.config["palette_experiment"]
    completed = [(item["task_id"], item["complete_task"]) for item in records if item.get("complete_task")]
    all_distribution = measure_background_distribution(
        completed,
        modal_fraction=float(palette_config["conservative_modal_fraction"]),
        require_consistent_input_mode=bool(palette_config["require_consistent_input_mode"]),
    )
    control_distribution = measure_background_distribution(
        [
            (item["task_id"], item["complete_task"])
            for item in records
            if item.get("complete_task") and item["private_route"] == "ordinary_control"
        ],
        modal_fraction=float(palette_config["conservative_modal_fraction"]),
        require_consistent_input_mode=bool(palette_config["require_consistent_input_mode"]),
    )
    current_distribution = measure_background_distribution(
        [
            (item["task_id"], item["complete_task"])
            for item in records
            if item.get("complete_task") and item["private_route"] != "ordinary_control"
        ],
        modal_fraction=float(palette_config["conservative_modal_fraction"]),
        require_consistent_input_mode=bool(palette_config["require_consistent_input_mode"]),
    )
    transformations = [
        deepcopy(item.get("palette_treatment"))
        for item in records
        if item.get("palette_treatment")
    ]
    palette = {
        "schema_version": 1,
        "all_completed": all_distribution,
        "embedded_control": control_distribution,
        "integrated_current": current_distribution,
        "committed_bank_before_run": _committed_distribution(runtime.config),
        "real_bank_reference": _palette_target(runtime.config)[1],
        "applied_count": sum(bool(item.get("applied")) for item in transformations),
        "safe_opt_out_count": sum(
            (not item.get("applied"))
            and (
                "not safe" in str(item.get("reason", ""))
                or "stale authoring-claim" in str(item.get("reason", ""))
            )
            for item in transformations
        ),
        "control_unscheduled_count": sum(
            item.get("status") == "embedded_control_not_scheduled" for item in transformations
        ),
        "global_bijection_failures": [
            item for item in transformations if item.get("applied") and not item.get("inverse_round_trip_exact", False)
        ],
        "semantic_authoring_unchanged_by_palette": True,
    }
    scale_rows = [
        {
            "task_id": item["task_id"],
            "route": item["private_route"],
            "planning_decision": item.get("planning_result", {}).get("scale_decision"),
            "functionally_expanded": _genuinely_expanded(item),
            "canvas_use_classification": (
                (item.get("realization") or {}).get("authoring_metadata") or {}
            ).get("canvas_use_classification"),
            "metrics": task_scale_metrics(item["complete_task"]),
        }
        for item in records
        if item.get("complete_task")
    ]
    scale = {
        "schema_version": 1,
        "functional_scale_remains_unpromoted": True,
        "all_completed": _arm_summary(records)["scale"],
        "assigned_functional_scale": _arm_summary(
            [item for item in records if item["private_route"] == "functional_scale"]
        )["scale"],
        "functionally_expanded_count": sum(item["functionally_expanded"] for item in scale_rows),
        "artificially_inflated_task_ids": [
            item["task_id"]
            for item in scale_rows
            if item["canvas_use_classification"] == "artificially_inflated"
        ],
        "tasks": scale_rows,
    }
    atomic_write_json(runtime.run_dir / "palette_distribution_report.json", palette)
    atomic_write_json(runtime.run_dir / "grid_scale_report.json", scale)
    return palette, scale


def _committed_bank_census(runtime: V41Runtime) -> dict[str, Any]:
    path = runtime.run_dir / "committed_bank_reference.json"
    if path.exists():
        return read_json(path)
    registry_path = Path(runtime.config["paths"]["committed_task_registry"])
    registry = load_committed_task_registry(registry_path)
    tasks = []
    scale = []
    for entry in registry["entries"]:
        task_path = registry_path.parent / entry["task_path"]
        if not task_path.is_file():
            continue
        task = read_json(task_path)
        tasks.append((entry["registry_task_id"], task))
        scale.append(task_scale_metrics(task))
    source_counts = Counter(item["source_pipeline"] for item in registry["entries"])
    search_counts = Counter(item["search_mode"] for item in registry["entries"])
    dimensions = [item["maximum_dimension"] for item in scale]
    value = {
        "schema_version": 1,
        "registry_id": registry.get("registry_id"),
        "registry_task_count": len(registry["entries"]),
        "readable_task_count": len(tasks),
        "source_pipeline_counts": dict(sorted(source_counts.items())),
        "search_mode_counts": dict(sorted(search_counts.items())),
        "background_distribution": measure_background_distribution(
            tasks,
            modal_fraction=float(runtime.config["palette_experiment"]["conservative_modal_fraction"]),
            require_consistent_input_mode=bool(
                runtime.config["palette_experiment"]["require_consistent_input_mode"]
            ),
        ),
        "grid_scale": {
            "maximum_dimension_mean": round(sum(dimensions) / len(dimensions), 6)
            if dimensions
            else None,
            "above_dimension_15_share": _safe_ratio(sum(value > 15 for value in dimensions), len(dimensions)),
            "above_dimension_20_share": _safe_ratio(sum(value > 20 for value in dimensions), len(dimensions)),
            "above_dimension_25_share": _safe_ratio(sum(value > 25 for value in dimensions), len(dimensions)),
        },
        "v4_4_manifest_qualification": deepcopy(
            runtime.config["_v4_4_baseline_verification"].get("qualification", {})
        ),
        "limitation": (
            "The historical registry preserves provenance and dossiers but not one unified "
            "post-hoc property audit over all 126 tasks. Source-pipeline counts are therefore "
            "context, not claims that every task from a source genuinely instantiates its feature."
        ),
    }
    atomic_write_json(path, value)
    return value


def _call_accounting(
    runtime: V41Runtime, slates: Sequence[Mapping[str, Any]], core_count: int
) -> dict[str, Any]:
    runtime_accounting = runtime.accounting()
    selector_rows = []
    supply_rows = []
    seen_supply: set[str] = set()
    for slate in slates:
        state_path = Path(slate["state_path"])
        state = read_json(state_path)
        selector_rows.append(
            v4_call_accounting(state.get("call_ledger", {}), run_dir=state_path.parent)
        )
        for raw in slate.get("supply", {}).get("call_ledger_paths", []):
            if raw in seen_supply:
                continue
            seen_supply.add(raw)
            supply_path = Path(raw)
            if supply_path.is_file():
                supply_state = read_json(supply_path)
                supply_rows.append(
                    v4_call_accounting(
                        supply_state.get("call_ledger", {}), run_dir=supply_path.parent
                    )
                )
    selector_calls = sum(item["substantive_logical_calls"] for item in selector_rows)
    supply_calls = sum(item["substantive_logical_calls"] for item in supply_rows)
    runtime_calls = int(runtime_accounting["experimental_calls"])
    by_stage = Counter(
        item["stage"]
        for item in runtime.state["call_ledger"].values()
        if item.get("status") == "complete"
    )
    by_role = Counter(
        item["role"]
        for item in runtime.state["call_ledger"].values()
        if item.get("status") == "complete"
    )
    active_selector = sum(item["active_model_minutes"] for item in selector_rows)
    active_supply = sum(item["active_model_minutes"] for item in supply_rows)
    active_runtime = float(runtime_accounting["active_model_minutes"])
    total_calls = selector_calls + supply_calls + runtime_calls
    total_minutes = active_selector + active_supply + active_runtime
    value = {
        "schema_version": 1,
        "selector_calls": selector_calls,
        "off_critical_path_reservoir_supply_calls": supply_calls,
        "integrated_runtime_calls": runtime_calls,
        "total_substantive_logical_calls": total_calls,
        "runtime_calls_by_stage": dict(sorted(by_stage.items())),
        "runtime_calls_by_role": dict(sorted(by_role.items())),
        "active_model_minutes": {
            "selectors": round(active_selector, 6),
            "reservoir_supply": round(active_supply, 6),
            "integrated_runtime": round(active_runtime, 6),
            "total": round(total_minutes, 6),
        },
        "calls_per_core_task": _safe_ratio(total_calls, core_count),
        "model_minutes_per_core_task": _safe_ratio(total_minutes, core_count),
        "transport_retry_calls": sum(
            item["transport_retry_calls"] for item in selector_rows + supply_rows
        )
        + int(runtime_accounting["transport_retry_calls"]),
        "measured_runtime_peak_concurrent_calls": runtime_accounting[
            "measured_peak_concurrent_calls"
        ],
        "measured_runtime_effective_concurrency": runtime_accounting[
            "measured_effective_concurrency"
        ],
        "selector_ledgers": selector_rows,
        "supply_ledgers": supply_rows,
        "runtime_ledger": runtime_accounting,
    }
    atomic_write_json(runtime.run_dir / "call_and_timing_ledger.json", value)
    return value


def _human_review_pack(
    runtime: V41Runtime,
    records: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    destination = runtime.run_dir / "human_review"
    tasks_dir = destination / "tasks"
    images_dir = destination / "images"
    dossiers_dir = destination / "dossiers_private"
    for directory in (tasks_dir, images_dir, dossiers_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rng = Random(int(runtime.config["seed"]) + 424301)
    shuffled = [item for item in records if item.get("complete_task")]
    rng.shuffle(shuffled)
    used: set[str] = set()
    public = []
    answer_key = []
    review_form = []
    by_slate: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    by_slate_rules: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for index, record in enumerate(shuffled, 1):
        anonymous = _anonymous_id(rng, used)
        task_path = tasks_dir / f"{anonymous}.json"
        atomic_write_json(task_path, record["complete_task"])
        image_path = render_episode(
            record["complete_task"],
            title=f"Review task {index}",
            subtitle="Rule, source, route, and status withheld",
            destination=images_dir / f"{anonymous}.png",
        )
        rule = record["realization"]["authoring_metadata"]["intended_rule"]
        explained = render_episode(
            record["complete_task"],
            title=f"Review task {index}",
            subtitle=rule,
            destination=images_dir / f"{anonymous}_with_rule.png",
        )
        slate = int(record["slate_index"])
        by_slate[slate].append((f"Task {index}", image_path))
        by_slate_rules[slate].append((f"Task {index}", explained))
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
                "private_route": record["private_route"],
                "private_source": record["private_source"],
                "search_mode": record["search_mode"],
                "intended_rule": rule,
                "intrinsic_quality": record.get("intrinsic_quality"),
                "portfolio_placement": record.get("portfolio_placement"),
                "feature_assessments": deepcopy(statuses[str(record["task_id"])]),
            }
        )
        review_form.append(
            {
                "review_index": index,
                "conceptual_interest_1_to_5": None,
                "functional_composition_1_to_5": None,
                "visual_legibility_1_to_5": None,
                "fair_and_one_answer": None,
                "palette_is_natural": None,
                "canvas_is_functionally_used": None,
                "mechanisms_feel_authentic_not_forced": None,
                "overall_quality_1_to_5": None,
                "notes": "",
            }
        )
        atomic_write_json(
            dossiers_dir / f"{record['task_id']}.json",
            {
                "task_id": record["task_id"],
                "kernel": record["kernel"],
                "editorial_brief": record["editorial_brief"],
                "concrete_plan": record.get("concrete_plan"),
                "realization": record.get("realization"),
                "technical_results": record.get("technical_results"),
                "blind_results": record.get("blind_results"),
                "editorial_judgments": record.get("editorial_judgments"),
                "feature_assessments": deepcopy(statuses[str(record["task_id"])]),
            },
        )
    contact_sheets = {}
    for slate in sorted(by_slate):
        blind = build_contact_sheet(
            by_slate[slate], destination / f"slate_{slate:02d}_blind.png", columns=2
        )
        explained = build_contact_sheet(
            by_slate_rules[slate],
            destination / f"slate_{slate:02d}_with_rules.png",
            columns=2,
        )
        contact_sheets[str(slate)] = {
            "blind": str(blind) if blind else None,
            "with_rules": str(explained) if explained else None,
        }
    atomic_write_json(
        destination / "review_manifest.json",
        {
            "provenance_hidden": True,
            "rules_withheld_from_first_pass": True,
            "task_count": len(public),
            "tasks": public,
            "contact_sheets": contact_sheets,
        },
    )
    atomic_write_json(destination / "review_form.json", {"tasks": review_form})
    atomic_write_json(destination / "DO_NOT_OPEN_FIRST.answer_key.json", {"tasks": answer_key})
    atomic_write_text(
        destination / "README.md",
        "# Integrated qualification human review\n\n"
        "Review each `slate_*_blind.png` before opening the rule sheets or answer key. "
        "Judge ordinary quality first; no task should receive credit merely for exhibiting "
        "a named feature. Fill `review_form.json` and inspect whether feature mechanisms "
        "are authentic, varied, and proportionate rather than forced.\n",
    )
    return {
        "directory": str(destination),
        "task_count": len(public),
        "human_results_available": False,
        "contact_sheets": contact_sheets,
        "review_manifest": str(destination / "review_manifest.json"),
        "review_form": str(destination / "review_form.json"),
        "answer_key_private": str(destination / "DO_NOT_OPEN_FIRST.answer_key.json"),
    }


def _export_outcomes(runtime: V41Runtime, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    root = runtime.run_dir / "exports"
    categories = {
        "core_quality": root / "core",
        "reserve_quality": root / "reserve",
        "ambiguous": root / "ambiguous",
        "rejected_quality": root / "rejected",
        "unreviewed_or_failed": root / "other",
    }
    for directory in categories.values():
        directory.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    paths = {}
    for record in records:
        if not record.get("complete_task"):
            continue
        quality = record.get("intrinsic_quality")
        if quality not in categories:
            quality = "ambiguous" if record.get("status") == "ambiguous" else "unreviewed_or_failed"
        destination = categories[str(quality)] / f"{record['task_id']}.json"
        atomic_write_json(destination, record["complete_task"])
        counts[str(quality)] += 1
        paths[str(record["task_id"])] = str(destination)
    value = {
        "automatic_commit_occurred": False,
        "experimental_exports_only": True,
        "counts": dict(sorted(counts.items())),
        "paths": paths,
    }
    atomic_write_json(root / "manifest.json", value)
    return value


def _task_report_rows(
    runtime: V41Runtime,
    records: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        task_id = str(record["task_id"])
        task = record.get("complete_task")
        boards = record.get("editorial_judgments", [])
        rows.append(
            {
                "task_id": task_id,
                "source_candidate_id": record["source_candidate_id"],
                "slate_index": record["slate_index"],
                "development_priority": record["source_development_priority"],
                "search_mode": record["search_mode"],
                "private_source": record["private_source"],
                "private_route": record["private_route"],
                "terminal_status": record.get("status"),
                "realized": bool(task),
                "technical_pass": bool(record.get("technical_results"))
                and record["technical_results"][-1].get("passed") is True,
                "hidden_output_exact_match": (
                    record["technical_results"][-1].get("hidden_output_match")
                    if record.get("technical_results")
                    else None
                ),
                "blind_clear": record.get("status") in {"blind_pass", "editorial_complete"},
                "serious_blind_alternative": any(
                    bool(item.get("serious_different_output_alternative"))
                    for item in record.get("blind_results", [])
                ),
                "intrinsic_quality": record.get("intrinsic_quality"),
                "portfolio_placement": record.get("portfolio_placement"),
                "editorial_reason": record.get("editorial_reason"),
                "board_novel_concept": [item.get("novel_concept") for item in boards],
                "board_functional_composition": [
                    item.get("functional_composition") for item in boards
                ],
                "board_overall_quality": [item.get("overall_quality") for item in boards],
                "intended_rule": (
                    ((record.get("realization") or {}).get("authoring_metadata") or {})
                    .get("intended_rule")
                ),
                "feature_assessments": deepcopy(statuses.get(task_id, {})),
                "palette_treatment": deepcopy(record.get("palette_treatment")),
                "scale_metrics": task_scale_metrics(task) if task else None,
                "task_json": (
                    str(runtime.run_dir / "tasks" / "single" / f"{task_id}.arc.json")
                    if task
                    else None
                ),
                "plan_json": str(runtime.run_dir / "plans" / "single" / f"{task_id}.json"),
                "dossier_json": str(
                    runtime.run_dir / "human_review" / "dossiers_private" / f"{task_id}.json"
                ),
                "stage_history": deepcopy(record.get("stage_history", [])),
                "technical_results": deepcopy(record.get("technical_results")),
                "blind_results": deepcopy(record.get("blind_results")),
                "editorial_judgments": deepcopy(boards),
            }
        )
    return rows


def _render_report(report: Mapping[str, Any]) -> str:
    funnel = report["funnel"]
    coverage = report["coverage"]
    quality = report["quality"]
    palette = report["palette"]
    scale = report["scale"]
    lines = [
        "# ARCGEN V4-to-V4.4 integrated qualification",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Model-only disposition: **{report['recommendation']}**",
        "",
        "This is a non-mutating portfolio validation rooted in the frozen V4.4 baseline. "
        "It tested the ordinary V4 authoring route together with deterministic palette scheduling, "
        "selector-routed logical leaps, structural transfer, episode-joint inference, abstraction-led "
        "supply, and the still-unpromoted functional-scale overlay. Every candidate received at most "
        "one deliberate semantic or scale intervention; palette was the only orthogonal treatment.",
        "",
        "## Experimental design",
        "",
        f"- Frozen independent slates: {funnel['slates']}",
        f"- Eligible kernels examined: {funnel['eligible_proposals_total']}",
        f"- Developed kernels: {funnel['developed']}",
        f"- Embedded fresh pre-palette V4 controls: {coverage['assigned_route_counts'].get('ordinary_control', 0)}",
        "- Selection happened before private routing; source/route labels were hidden downstream.",
        "- The exact V4 technical verifier, fully blind solver, and two fixed-anchor boards remained gating.",
        "- This run could not promote features or commit tasks.",
        "",
        "## Funnel",
        "",
        f"- Plan ready / aborted / operational failures: {funnel['plans_ready']} / {funnel['plan_aborted']} / {funnel['planning_operational_failure']}",
        f"- Realized / realization failures: {funnel['realized']} / {funnel['realization_failures']}",
        f"- Technical pass / hidden-output mismatches: {funnel['technical_pass']} / {funnel['hidden_output_mismatches']}",
        f"- Blind clear / ambiguous: {funnel['blind_clear']} / {funnel['ambiguous']}",
        f"- Core / reserve / rejected-or-other: {funnel['core']} / {funnel['reserve']} / {funnel['other']}",
        "",
        "## Route supply and genuine retained coverage",
        "",
        f"Assigned routes: `{coverage['assigned_route_counts']}`",
        "",
        "| Feature | Genuine completed | Genuine core | Core share | Operating band | Result |",
        "|---|---:|---:|---:|---|---|",
    ]
    for feature, item in coverage["core_feature_bands"].items():
        lines.append(
            f"| `{feature}` | {coverage['genuine_feature_counts_all_completed'].get(feature, 0)} | "
            f"{item['genuine_core_count']} | {item['core_share']} | "
            f"{item['operating_band']['lower']}–{item['operating_band']['upper']} "
            f"(soft {item['operating_band']['soft_ceiling']}) | `{item['disposition']}` |"
        )
    lines.extend(
        [
            "",
            f"Ordinary/unforced core share: `{coverage['ordinary_unforced_core']['share']}` "
            f"(minimum `{coverage['ordinary_unforced_core']['minimum']}`).",
            "",
            "The counts above come from arm-blind audits of completed grids, not from source labels. "
            "Natural occurrences count; specialized-source misses do not. Functional scale remains "
            "experimental and is reported separately rather than treated as an accepted semantic band.",
            "",
            "## Quality preservation",
            "",
            f"- Embedded control: `{quality['cohorts']['embedded_pre_palette_v4_control']}`",
            f"- Integrated current routes: `{quality['cohorts']['integrated_current_routes']}`",
            f"- Core-yield ratio: `{quality['noninferiority']['core_yield_ratio']}` "
            f"(floor `{quality['noninferiority']['required_floor']}`)",
            f"- Core-or-reserve ratio: `{quality['noninferiority']['core_or_reserve_yield_ratio']}` "
            f"(floor `{quality['noninferiority']['required_useful_floor']}`)",
            f"- Core serious ambiguities: `{quality['one_answer_and_mechanical_safety']['core_serious_ambiguity_task_ids']}`",
            f"- Hidden-output mismatches: `{quality['one_answer_and_mechanical_safety']['hidden_output_mismatch_task_ids']}`",
            f"- Artificially inflated tasks: `{quality['one_answer_and_mechanical_safety']['artificially_inflated_task_ids']}`",
            "",
            "## Palette and canvas behavior",
            "",
            f"- Integrated-current backgrounds: `{palette['integrated_current']['counts']}`",
            f"- Embedded-control backgrounds: `{palette['embedded_control']['counts']}`",
            f"- Applied global permutations / safe opt-outs: {palette['applied_count']} / {palette['safe_opt_out_count']}",
            f"- Functionally expanded tasks: {scale['functionally_expanded_count']}",
            f"- All-task scale summary: `{scale['all_completed']}`",
            "",
            "## Calls and wall time",
            "",
            f"- Selector / reservoir-supply / integrated-runtime calls: "
            f"{report['call_accounting']['selector_calls']} / "
            f"{report['call_accounting']['off_critical_path_reservoir_supply_calls']} / "
            f"{report['call_accounting']['integrated_runtime_calls']}",
            f"- Total substantive calls: {report['call_accounting']['total_substantive_logical_calls']}",
            f"- Active model-minutes: {report['call_accounting']['active_model_minutes']['total']}",
            f"- Wall-clock minutes: {report['timing']['wall_clock_minutes']}",
            f"- Calls / model-minutes per core: {report['call_accounting']['calls_per_core_task']} / "
            f"{report['call_accounting']['model_minutes_per_core_task']}",
            "",
            "## Task outcomes",
            "",
            "| Task | Slate | Mode | Private route | Tech | Blind | Intrinsic | Portfolio | Genuine properties | Max dim |",
            "|---|---:|---|---|---|---|---|---|---|---:|",
        ]
    )
    for task in report["tasks"]:
        genuine = [
            key
            for key, value in task["feature_assessments"].items()
            if value.get("genuine")
        ]
        maximum = task.get("scale_metrics", {}).get("maximum_dimension") if task.get("scale_metrics") else "n/a"
        lines.append(
            f"| `{task['task_id']}` | {task['slate_index']} | `{task['search_mode']}` | "
            f"`{task['private_route']}` | `{task['technical_pass']}` | `{task['blind_clear']}` | "
            f"`{task.get('intrinsic_quality') or task['terminal_status']}` | "
            f"`{task.get('portfolio_placement')}` | {', '.join(genuine) or 'none audited'} | {maximum} |"
        )
    lines.extend(
        [
            "",
            "## Human inspection and limitations",
            "",
            f"- Blinded review pack: `{report['human_review']['directory']}`",
            f"- Per-slate sheets: `{report['human_review']['contact_sheets']}`",
            "- Human review is pending, so this report is a model-qualified integration result, not a production promotion.",
            "- The embedded control is concurrent and fresh but not task-paired; yield ratios are descriptive for this finite portfolio.",
            "- Previously accepted features were not re-promoted. Functional scale remains explicitly unpromoted.",
            "- The committed bank stores complete dossiers but lacks one retrospective unified mechanism audit across every historical task; provenance counts are not substituted for genuine-property labels.",
            "",
            f"Recommendation: **{report['recommendation']}**. {report['recommendation_reason']}",
            "",
            "No production reservoir, committed-task registry, feature manifest, or frozen V4.4 artifact changed. "
            "No task was committed and no feature was promoted automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_integrated_qualification(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("integrated_pipeline") != INTEGRATED_PIPELINE:
        raise IntegratedQualificationError("integrated qualification config is required")
    verify_v44_baseline(
        config["_repository_root"], config["_v4_4_baseline_verification"]["manifest_path"]
    )
    return {
        "valid": True,
        "pipeline": INTEGRATED_PIPELINE,
        "run_id": config["run"]["id"],
        "slates": config["integrated_qualification"]["independent_slates"],
        "eligible_proposals": config["integrated_qualification"]["independent_slates"]
        * config["integrated_qualification"]["eligible_window_size"],
        "maximum_developed": config["integrated_qualification"]["independent_slates"]
        * config["integrated_qualification"]["maximum_development_per_slate"],
        "accepted_features": [
            key
            for key, value in config["_v4_4_feature_manifest_snapshot"]["features"].items()
            if str(value.get("status", "")).startswith("accepted")
        ],
        "functional_scale_status": "unpromoted_continue_testing",
        "answer_aware_verifier_enabled": False,
        "automatic_commit": False,
        "automatic_promotion": False,
    }


def run_integrated_qualification(config: dict[str, Any]) -> dict[str, Any]:
    validate_integrated_qualification(config)
    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    start_path = run_dir / "experiment_start.json"
    if not start_path.exists():
        atomic_write_json(start_path, {"started_at_epoch_seconds": round(time.time(), 3)})
    protected_before = _protected_hashes(config)
    source_manifest = {
        "schema_version": 1,
        "pipeline": INTEGRATED_PIPELINE,
        "frozen_v4_4_baseline": deepcopy(config["_v4_4_baseline_verification"]),
        "accepted_v4_4_feature_manifest": deepcopy(config["_v4_4_feature_manifest_snapshot"]),
        "functional_scale_evidence": deepcopy(config["_functional_scale_evidence"]),
        "protected_production_state_before": protected_before,
        "model_and_effort_settings": deepcopy(config["model"]["roles"]),
        "registered_schedule": deepcopy(config["integrated_qualification"]),
        "selector_quality_precedes_private_routing": True,
        "route_and_source_hidden_downstream": True,
        "one_deliberate_intervention_per_candidate": True,
        "automatic_promotion": False,
        "automatic_task_commit": False,
    }
    runtime = V41Runtime(config, source_manifest)
    _install_extensions(runtime)
    if runtime.state.get("completed") and (run_dir / "report.json").is_file():
        return read_json(run_dir / "report.json")

    selected, slates, _source_mapping = _select_slates(config, runtime)
    aliases = _aliases(selected, int(config["seed"]))
    assignments = _freeze_routes(config, runtime, selected, aliases)
    _initialize_records(runtime, selected, assignments)
    _plan_records(runtime)
    _realize_records(runtime)
    budgets = _budgets(runtime)
    _run_technical(runtime, budgets)
    _run_blind(runtime, budgets)
    _editorial_boards(runtime)

    audits = _feature_audits(runtime)
    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda item: (int(item["slate_index"]), int(item["source_development_priority"])),
    )
    statuses = {str(item["task_id"]): _feature_statuses(item, audits) for item in records}
    diversity = _embedding_diversity(runtime, records, statuses)
    quality, coverage = _quality_and_coverage(runtime, records, statuses)
    palette, scale = _palette_and_scale_report(runtime, records)
    committed_reference = _committed_bank_census(runtime)
    human = _human_review_pack(runtime, records, statuses)
    exports = _export_outcomes(runtime, records)
    core_count = sum(item.get("intrinsic_quality") == "core_quality" for item in records)
    accounting = _call_accounting(runtime, slates, core_count)

    protected_after = _protected_hashes(config)
    _assert_hashes_unchanged(protected_before, protected_after)
    baseline_after = verify_v44_baseline(
        config["_repository_root"], config["_v4_4_baseline_verification"]["manifest_path"]
    )
    if baseline_after != config["_v4_4_baseline_verification"]:
        raise IntegratedQualificationError("frozen V4.4 baseline changed during qualification")

    statuses_counter = Counter(item.get("status", "unknown") for item in records)
    plans_ready = sum(item.get("concrete_plan", {}).get("status") == "ready" for item in records)
    realized = sum(bool(item.get("complete_task")) for item in records)
    technical_pass = sum(
        bool(item.get("technical_results"))
        and item["technical_results"][-1].get("passed") is True
        for item in records
    )
    blind_clear = sum(item.get("status") in {"blind_pass", "editorial_complete"} for item in records)
    reserve_count = sum(item.get("intrinsic_quality") == "reserve_quality" for item in records)
    ambiguous_count = sum(
        item.get("status") == "ambiguous" or item.get("intrinsic_quality") == "ambiguous"
        for item in records
    )
    other = len(records) - core_count - reserve_count - ambiguous_count
    model_gates = [
        quality["noninferiority"]["core_yield_gate"],
        quality["noninferiority"]["core_or_reserve_gate"],
        quality["one_answer_and_mechanical_safety"]["core_serious_ambiguity_gate"],
        quality["one_answer_and_mechanical_safety"]["hidden_output_exact_match_gate"],
        quality["one_answer_and_mechanical_safety"]["artificial_inflation_gate"],
        coverage["ordinary_unforced_core"]["gate"],
    ]
    if all(model_gates):
        recommendation = "continue_to_blinded_human_review"
        recommendation_reason = (
            "All preregistered model-only quality and safety gates passed; proportions and mechanism "
            "authenticity should now be checked in the supplied blinded human pack before any baseline decision."
        )
    else:
        recommendation = "do_not_integrate_without_recalibration"
        failed = [
            name
            for name, passed in zip(
                (
                    "core_yield",
                    "core_or_reserve_yield",
                    "core_ambiguity",
                    "hidden_output_match",
                    "scale_inflation",
                    "ordinary_majority",
                ),
                model_gates,
            )
            if not passed
        ]
        recommendation_reason = "Model-only gates failed: " + ", ".join(failed)

    completed = round(time.time(), 3)
    started = float(read_json(start_path)["started_at_epoch_seconds"])
    tasks = _task_report_rows(runtime, records, statuses)
    funnel = {
        "slates": len(slates),
        "eligible_proposals_total": sum(len(item["frozen_window"]["proposal_ids"]) for item in slates),
        "developed": len(records),
        "plans_ready": plans_ready,
        "plan_aborted": statuses_counter["plan_aborted"],
        "planning_operational_failure": statuses_counter["planning_operational_failure"],
        "realized": realized,
        "realization_failures": statuses_counter["realization_failure"]
        + statuses_counter["realization_operational_failure"],
        "technical_pass": technical_pass,
        "hidden_output_mismatches": len(
            quality["one_answer_and_mechanical_safety"]["hidden_output_mismatch_task_ids"]
        ),
        "blind_clear": blind_clear,
        "ambiguous": ambiguous_count,
        "core": core_count,
        "reserve": reserve_count,
        "other": other,
    }
    report = {
        "schema_version": 1,
        "pipeline": INTEGRATED_PIPELINE,
        "run_id": runtime.run_id,
        "completed": True,
        "frozen_v4_4_baseline": deepcopy(config["_v4_4_baseline_verification"]),
        "experimental_design": {
            "independent_slates": len(slates),
            "frozen_window_size": 48,
            "maximum_developed_per_slate": 8,
            "embedded_controls_per_slate": 2,
            "private_route_assignment": read_json(
                run_dir / "integrated_route_assignment.public.json"
            ),
            "one_deliberate_intervention_per_candidate": True,
            "palette_is_only_orthogonal_treatment": True,
            "answer_aware_verifier_excluded": True,
        },
        "funnel": funnel,
        "quality": quality,
        "coverage": coverage,
        "palette": palette,
        "scale": scale,
        "diversity": diversity,
        "committed_bank_reference": committed_reference,
        "human_review": human,
        "experimental_exports": exports,
        "call_accounting": accounting,
        "timing": {
            "started_at_epoch_seconds": started,
            "completed_at_epoch_seconds": completed,
            "wall_clock_minutes": round((completed - started) / 60, 3),
        },
        "tasks": tasks,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "automatic_promotion_occurred": False,
        "automatic_task_commit_occurred": False,
        "functional_scale_promoted": False,
        "production_state_unchanged": protected_before == protected_after,
        "human_review_complete": False,
        "unresolved_uncertainties": [
            "The blinded human review pack has not yet been scored.",
            "Feature bands are evaluated on the finite set of core tasks and may remain sample-sensitive.",
            "Functional scale has integration evidence here but remains an explicitly unpromoted treatment.",
            "Historical portfolio property prevalence is not retroactively audited under one unified rubric.",
        ],
    }
    atomic_write_json(
        run_dir / "non_mutation_attestation.json",
        {
            "before": protected_before,
            "after": protected_after,
            "unchanged": protected_before == protected_after,
            "frozen_v4_4_verified_after": True,
            "automatic_task_commit": False,
            "automatic_promotion": False,
        },
    )
    atomic_write_json(
        run_dir / "promotion_recommendation.json",
        {
            "recommendation": recommendation,
            "user_review_required": True,
            "automatic_promotion_occurred": False,
            "functional_scale_remains_unpromoted": True,
        },
    )
    atomic_write_text(
        run_dir / "reproduction_commands.md",
        "# Reproduction\n\n"
        "```bash\n"
        "python -m arcgen.lib v4-integrated-validate --config config/v4_integrated_qualification.yaml\n"
        "python -m arcgen.lib v4-integrated-run --config config/v4_integrated_qualification.yaml\n"
        f"python -m arcgen.lib v4-integrated-report --run {run_dir}\n"
        "```\n",
    )
    return runtime.finish(report, _render_report(report))


def rebuild_integrated_report(run: str | Path) -> dict[str, Any]:
    directory = Path(run).resolve()
    report = read_json(directory / "report.json")
    if report.get("pipeline") != INTEGRATED_PIPELINE:
        raise IntegratedQualificationError("run is not an integrated qualification")
    atomic_write_text(directory / "report.md", _render_report(report))
    return report


def integrated_review_status(run: str | Path) -> dict[str, Any]:
    report = read_json(Path(run).resolve() / "report.json")
    if report.get("pipeline") != INTEGRATED_PIPELINE:
        raise IntegratedQualificationError("run is not an integrated qualification")
    return deepcopy(report["human_review"])


__all__ = [
    "IntegratedQualificationError",
    "assign_fresh_scale_integrated_routes",
    "assign_integrated_routes",
    "integrated_review_status",
    "rebuild_integrated_report",
    "run_integrated_qualification",
    "validate_integrated_qualification",
]
