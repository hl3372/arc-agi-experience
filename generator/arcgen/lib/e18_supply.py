"""Frozen E18 supply schedule, bank context, reservoir, and selection.

This module deliberately owns no development or review calls.  It creates
source-stratified 48-kernel windows, records proposal lifecycle transitions in
one run-local bounded reservoir, and delegates qualitative selection to the
unchanged V4 shard/global selector with all source labels removed.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
from math import floor
from pathlib import Path
from random import Random
from typing import Any

from . import e16_bank_context_tuning as tuning
from .committed_tasks import load_committed_task_registry
from .e3_live_overlap import SharedCallPool, _LimiterInvoker
from .e18_calls import E18AttemptError, E18AttemptInvoker
from .model import ModelInvoker
from .util import (
    PLACEHOLDER_RE,
    atomic_write_json,
    content_hash,
    load_prompt,
    read_json,
    render_prompt,
)
from .v4_config import load_v4_config
from .v4_pipeline import V4ControlPipeline, _DataContext
from .v4_reservoir import (
    load_reservoir,
    make_proposal,
    resolve_proposal,
    validate_reservoir,
)

PIPELINE = "arcgen-e18-production-candidate"
SOL = "gpt-5.6-sol"
MODES = ("open", "concept_led", "composition_led")
SOURCES = (
    "ordinary_unconditioned",
    "ordinary_bank_conditioned",
    "logical_leap",
    "episode_joint_inference",
)
SPECIAL_SOURCES = frozenset({"logical_leap", "episode_joint_inference"})
ACTIVE_PROPOSAL_STATES = frozenset({"proposed", "in_flight"})
TERMINAL_PROPOSAL_STATES = frozenset(
    {"completed_uncommitted", "rejected", "recoverable"}
)
# The accepted bank crossed the original 700k safety ceiling after the
# user-authorized 2026-08-31 backlog commit.  E20 live studies already use this
# 2M ceiling for the same full-lineage snapshot.  Raising the bound preserves
# the existing no-truncation semantics; it does not change which rows are
# exposed or how representatives are selected.
MAXIMUM_BANK_CONTEXT_CHARACTERS = 2_000_000


class E18SupplyError(RuntimeError):
    """Raised when a supply or selector artifact loses its frozen identity."""


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _immutable(path: Path, value: Any, label: str) -> None:
    if path.is_file():
        if read_json(path) != value:
            raise E18SupplyError(f"immutable {label} changed: {path}")
        return
    atomic_write_json(path, value)


def _output(config: Mapping[str, Any]) -> Path:
    return Path(str(config["_output_directory"]))


def _root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["_repository_root"]))


def _base_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = load_v4_config(config["_references"]["base_v4_control"])
    value = deepcopy(value)
    value["paths"] = deepcopy(value["paths"])
    value["paths"]["committed_task_registry"] = config["_paths"]["active_registry"]
    value["paths"]["reservoir"] = config["_paths"]["production_reservoir"]
    value["paths"]["cache"] = config["_paths"]["cache"]
    value["model"] = deepcopy(value["model"])
    value["model"]["concurrency"] = 8
    return value


def freeze_bank_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze all explicitly committed active lineages as positive-only context."""

    directory = _output(config) / "bank_snapshot"
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        value = read_json(manifest_path)
        rows_path = Path(str(value.get("generator_visible_rows_path") or ""))
        selector_memory_path = Path(
            str(value.get("selector_memory_snapshot_path") or "")
        )
        if (
            value.get("active_registry_sha256")
            != _sha(config["_paths"]["active_registry"])
            or value.get("production_reservoir_sha256")
            != _sha(config["_paths"]["production_reservoir"])
            or value.get("quarantine_registry_sha256")
            != _sha(config["_paths"]["quarantine_registry"])
            or not rows_path.is_file()
            or value.get("generator_visible_rows_sha256") != _sha(rows_path)
            or not selector_memory_path.is_file()
            or value.get("selector_memory_snapshot_sha256")
            != _sha(selector_memory_path)
        ):
            raise E18SupplyError("frozen E18 bank context no longer matches inputs")
        return value

    registry = load_committed_task_registry(config["_paths"]["active_registry"])
    root = _root(config)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    private_rows: list[dict[str, Any]] = []
    for entry in registry["entries"]:
        if entry["intrinsic_status"] not in {"core_quality", "reserve_quality", "boardless_verified"}:
            raise E18SupplyError("active positive bank contains a non-useful status")
        review = tuning._dossier_review(root, entry)
        lineage = tuning._known_lineage_key(entry, review)
        row = {
            "accepted_concept_id": "a"
            + content_hash(
                {"e18": "active", "registry_task_id": entry["registry_task_id"]}
            )[:13],
            "registry_task_id": entry["registry_task_id"],
            "known_lineage_key": lineage,
            "title": entry["title"],
            "concept_summary": entry["concept_summary"],
            "rule_summary": entry["intended_rule"],
            "intrinsic_status": entry["intrinsic_status"],
        }
        groups[lineage].append(row)
        private_rows.append(row)

    rng = Random(int(config["seed"]) + 18_001)
    lineages = sorted(groups)
    rng.shuffle(lineages)
    visible: list[dict[str, str]] = []
    representatives: list[dict[str, Any]] = []
    for lineage in lineages:
        members = sorted(groups[lineage], key=lambda item: item["accepted_concept_id"])
        chosen = members[rng.randrange(len(members))]
        representatives.append(chosen)
        visible.append(
            {
                "accepted_concept_id": str(chosen["accepted_concept_id"]),
                "title": str(chosen["title"]),
                "concept_summary": str(chosen["concept_summary"]),
                "rule_summary": str(chosen["rule_summary"]),
            }
        )

    serialized = json.dumps(visible, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > MAXIMUM_BANK_CONTEXT_CHARACTERS:
        raise E18SupplyError(
            "full known-lineage bank exceeds the frozen E18 context ceiling; "
            "silent truncation is forbidden"
        )
    directory.mkdir(parents=True, exist_ok=True)
    private_path = directory / "active_rows.private.json"
    visible_path = directory / "generator_visible_rows.json"
    selector_memory_path = directory / "selector_memory.private.json"
    atomic_write_json(
        private_path, sorted(private_rows, key=lambda item: item["accepted_concept_id"])
    )
    atomic_write_json(visible_path, visible)
    production_reservoir = load_reservoir(config["_paths"]["production_reservoir"])
    production_by_id = {
        str(item["proposal_id"]): item for item in production_reservoir["proposals"]
    }
    reservations = deepcopy(production_reservoir["active_reservations"])
    reserved_proposals = []
    for reservation in reservations:
        proposal_id = str(reservation["proposal_id"])
        proposal = production_by_id.get(proposal_id)
        if not isinstance(proposal, Mapping) or proposal.get("state") != "in_flight":
            raise E18SupplyError(
                "production reservation lacks its frozen in-flight proposal"
            )
        reserved_proposals.append(deepcopy(dict(proposal)))
    quarantine = read_json(Path(config["_paths"]["quarantine_registry"]))
    quarantine_memory = [
        {
            "memory_id": f"e18-quarantine-warning:{item['registry_task_id']}",
            "tier": "failure_mode",
            "proposal_id": f"quarantine:{item['registry_task_id']}",
            "concept_hash": content_hash(
                {"quarantined_registry_task_id": item["registry_task_id"]}
            ),
            "mechanism_summary": str(item.get("title") or "")[:700],
            "reason": str(item.get("evidence_issue_summary") or "")[:1000],
            "source_run_id": "e18-frozen-quarantine-warning",
            "created_at_epoch_seconds": 0.0,
            "active": True,
        }
        for item in quarantine.get("entries", [])
    ]
    selector_memory = {
        "schema_version": 1,
        "production_reservoir_sha256": _sha(config["_paths"]["production_reservoir"]),
        "reservation_proposals": sorted(
            reserved_proposals, key=lambda item: str(item["proposal_id"])
        ),
        "active_reservations": sorted(
            reservations, key=lambda item: str(item["reservation_id"])
        ),
        "tiered_rejection_memory": deepcopy(production_reservoir["rejection_memory"]),
        "quarantine_failure_mode_warnings": quarantine_memory,
        "generators_receive_this_memory": False,
        "selectors_receive_this_memory": True,
        "production_state_mutated": False,
    }
    atomic_write_json(selector_memory_path, selector_memory)
    value = {
        "schema_version": 1,
        "active_registry_path": str(
            Path(config["_paths"]["active_registry"]).resolve()
        ),
        "active_registry_sha256": _sha(config["_paths"]["active_registry"]),
        "quarantine_registry_sha256": _sha(config["_paths"]["quarantine_registry"]),
        "production_reservoir_sha256": _sha(config["_paths"]["production_reservoir"]),
        "active_task_count": len(private_rows),
        "known_lineage_count": len(visible),
        "known_multi_member_lineages": sum(len(rows) > 1 for rows in groups.values()),
        "private_rows_path": str(private_path.resolve()),
        "private_rows_sha256": _sha(private_path),
        "generator_visible_rows_path": str(visible_path.resolve()),
        "generator_visible_rows_sha256": _sha(visible_path),
        "selector_memory_snapshot_path": str(selector_memory_path.resolve()),
        "selector_memory_snapshot_sha256": _sha(selector_memory_path),
        "production_in_flight_reservation_count": len(reservations),
        "production_rejection_memory_count": len(
            production_reservoir["rejection_memory"]
        ),
        "quarantine_failure_mode_warning_count": len(quarantine_memory),
        "generator_visible_fields": [
            "accepted_concept_id",
            "title",
            "concept_summary",
            "rule_summary",
        ],
        "generator_hidden_fields": [
            "registry_task_id",
            "intrinsic_status",
            "portfolio_status",
            "grids",
            "dossiers",
            "provenance",
            "review",
            "quarantine",
        ],
        "serialized_context_characters": len(serialized),
        "estimated_context_tokens_at_four_characters_per_token": (len(serialized) + 3)
        // 4,
        "maximum_context_characters": MAXIMUM_BANK_CONTEXT_CHARACTERS,
        "silent_truncation": False,
        "basic_valid_included": False,
    }
    atomic_write_json(manifest_path, value)
    return value


def validate_selector_memory_snapshot(
    bank_manifest: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate frozen production memory without exposing it to generators."""

    path = Path(str(bank_manifest.get("selector_memory_snapshot_path") or ""))
    if not path.is_file() or _sha(path) != bank_manifest.get(
        "selector_memory_snapshot_sha256"
    ):
        raise E18SupplyError("selector-memory snapshot is missing or changed")
    memory = read_json(path)
    value = {
        "schema_version": 1,
        "reservoir_id": "arcgen-v4-persistent-idea-reservoir",
        "proposals": deepcopy(memory["reservation_proposals"]),
        "active_reservations": deepcopy(memory["active_reservations"]),
        "reservation_history": [],
        "rejection_memory": [
            *deepcopy(memory["tiered_rejection_memory"]),
            *deepcopy(memory["quarantine_failure_mode_warnings"]),
        ],
        "frozen_windows": [],
        "supply_history": [],
        "lifecycle_events": [],
    }
    validate_reservoir(value, path="e18.selector_memory")
    production_ids = {str(item["proposal_id"]) for item in value["proposals"]}
    run_ids = {
        str(candidate_id)
        for spec in design.get("specs") or []
        for candidate_id in spec.get("candidate_ids") or []
    }
    if production_ids & run_ids:
        raise E18SupplyError("E18 candidate IDs collide with production reservations")
    if memory.get("generators_receive_this_memory") is not False:
        raise E18SupplyError("production memory would leak to idea generators")
    if memory.get("selectors_receive_this_memory") is not True:
        raise E18SupplyError("production memory is not available to selectors")
    return {
        "valid": True,
        "production_reservations": len(value["active_reservations"]),
        "tiered_rejection_memory": len(memory["tiered_rejection_memory"]),
        "quarantine_failure_mode_warnings": len(
            memory["quarantine_failure_mode_warnings"]
        ),
        "candidate_ids_disjoint": True,
        "generator_boundary_preserved": True,
    }


def _candidate_ids(config: Mapping[str, Any], slate: int, call: int) -> list[str]:
    return [
        "k"
        + content_hash(
            {
                "e18": config["run"]["id"],
                "seed": config["seed"],
                "slate": slate,
                "call": call,
                "sibling": sibling,
            }
        )[:13]
        for sibling in range(1, 5)
    ]


def _source_specs_for_slate(
    config: Mapping[str, Any], slate: int
) -> list[dict[str, Any]]:
    # Six conditioned ordinary calls: two of every ordinary search mode.
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for copy_index in range(1, 3):
            rows.append(
                {
                    "source": "ordinary_bank_conditioned",
                    "search_mode": mode,
                    "conditioned": True,
                    "label": f"bank-{mode}-{copy_index}",
                    "effort": "xhigh",
                }
            )
    # Four unconditioned ordinary calls: one of each plus a rotating extra.
    extra = MODES[(slate - 1) % len(MODES)]
    plain_counts: Counter[str] = Counter()
    for mode in (*MODES, extra):
        plain_counts[mode] += 1
        copy_index = plain_counts[mode]
        rows.append(
            {
                "source": "ordinary_unconditioned",
                "search_mode": mode,
                "conditioned": False,
                "label": f"plain-{mode}-{copy_index}",
                "effort": "max",
            }
        )

    # Counterbalance which specialized source sees the bank within each pair.
    pair = (slate - 1) // 2
    first_is_a = (
        int(content_hash({"seed": config["seed"], "pair": pair})[:2], 16) % 2 == 0
    )
    pattern_a = first_is_a if slate % 2 == 1 else not first_is_a
    rows.extend(
        [
            {
                "source": "logical_leap",
                "search_mode": "concept_led",
                "conditioned": pattern_a,
                "label": "logical-leap",
                "effort": "xhigh",
            },
            {
                "source": "episode_joint_inference",
                "search_mode": "concept_led",
                "conditioned": not pattern_a,
                "label": "episode-joint",
                "effort": "xhigh",
            },
        ]
    )
    if len(rows) != 12:
        raise E18SupplyError("E18 source schedule must contain twelve calls")
    return rows


def _calibration_groups(config: Mapping[str, Any], slate: int) -> list[list[Any]]:
    """Choose 48 distinct real calibration examples within each slate.

    The selection is deterministic and source-neutral.  Unlike the historical
    global allocation bug, a 90-example pool is never asked to supply 576
    globally disjoint examples.
    """

    base = _base_config(config)
    context = _DataContext(base)
    pool = sorted(context.generator_pool, key=lambda item: item.task_id)
    if len(pool) < 48:
        raise E18SupplyError("fewer than 48 generator-visible calibration tasks")
    rng = Random(int(config["seed"]) + slate * 180_043)
    rng.shuffle(pool)
    selected = pool[:48]
    return [selected[index * 4 : (index + 1) * 4] for index in range(12)]


def freeze_supply_design(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _output(config) / "frozen_supply_design.private.json"
    if path.is_file():
        value = read_json(path)
        if value.get("active_bank_sha256") != _sha(config["_paths"]["active_registry"]):
            raise E18SupplyError("frozen supply design outlived its active bank")
        return value
    bank = freeze_bank_snapshot(config)
    specs: list[dict[str, Any]] = []
    for slate in range(1, int(config["design"]["slates"]) + 1):
        rows = _source_specs_for_slate(config, slate)
        calibrations = _calibration_groups(config, slate)
        if len({item.task_id for group in calibrations for item in group}) != 48:
            raise E18SupplyError("calibration examples repeat within a slate")
        for call_index, (row, group) in enumerate(
            zip(rows, calibrations, strict=True), 1
        ):
            call_id = f"e18-s{slate:02d}-c{call_index:02d}-{row['label']}"
            specs.append(
                {
                    "slate_index": slate,
                    "call_index": call_index,
                    "call_id": call_id,
                    "candidate_ids": _candidate_ids(config, slate, call_index),
                    "search_mode": row["search_mode"],
                    "private_source": row["source"],
                    "private_conditioned": row["conditioned"],
                    "model": SOL,
                    "effort": row["effort"],
                    "calibration_task_ids": [item.task_id for item in group],
                    "calibration_descriptions": [
                        item.rule_description for item in group
                    ],
                    "bank_context": (
                        {
                            "path": bank["generator_visible_rows_path"],
                            "sha256": bank["generator_visible_rows_sha256"],
                            "known_lineage_count": bank["known_lineage_count"],
                        }
                        if row["conditioned"]
                        else None
                    ),
                }
            )
    expected = int(config["design"]["slates"]) * 12
    if len(specs) != expected or len({item["call_id"] for item in specs}) != expected:
        raise E18SupplyError("frozen supply design call identities are not unique")
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "run_id": config["run"]["id"],
        "seed": config["seed"],
        "slates": config["design"]["slates"],
        "calls_per_slate": 12,
        "kernels_per_call": 4,
        "kernels_per_slate": 48,
        "active_bank_sha256": _sha(config["_paths"]["active_registry"]),
        "bank_snapshot": bank,
        "specs": specs,
        "source_labels_private": True,
        "selector_thresholds_unchanged": True,
        "automatic_commit": True,
    }
    _immutable(path, value, "E18 supply design")
    return value


def render_idea_prompt(
    config: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[str, str, str]:
    root = _root(config)
    base = {
        "open": root / "prompts/v4_idea_open.md",
        "concept_led": root / "prompts/v4_idea_concept.md",
        "composition_led": root / "prompts/v4_idea_composition.md",
    }[str(spec["search_mode"])]
    paths = [base]
    source = str(spec["private_source"])
    if source == "logical_leap":
        paths.append(root / "prompts/e17_logical_leap_idea_addendum.md")
    elif source == "episode_joint_inference":
        paths.append(root / "prompts/v4_3_episode_joint_idea_addendum.md")
    paths.append(root / "prompts/e16_idea_nomination_addendum.md")
    if spec["private_conditioned"]:
        paths.append(root / "prompts/e16_bank_diversity_addendum.md")
    loaded = [load_prompt(path) for path in paths]
    template = "\n\n".join(item[0] for item in loaded)
    values: dict[str, Any] = {
        "CALIBRATION_DESCRIPTIONS": list(spec["calibration_descriptions"]),
        "SEARCH_CALL_ID": spec["call_id"],
        "CANDIDATE_IDS": list(spec["candidate_ids"]),
    }
    if spec["private_conditioned"]:
        reference = spec.get("bank_context") or {}
        path = Path(str(reference.get("path") or ""))
        if not path.is_file() or _sha(path) != reference.get("sha256"):
            raise E18SupplyError("conditioned idea call lost its frozen bank context")
        values["ACCEPTED_CONCEPT_BANK"] = read_json(path)
    if set(PLACEHOLDER_RE.findall(template)) != set(values):
        raise E18SupplyError("E18 idea prompt placeholder boundary changed")
    rendered = render_prompt(template, values)
    version = "+".join(item[1] for item in loaded)
    digest = content_hash(
        {"template_hashes": [item[2] for item in loaded], "template": template}
    )
    return rendered, version, digest


class IdeaRuntime:
    """Attempt-bounded, resumable E18 idea-call runtime."""

    def __init__(self, config: Mapping[str, Any], pool: SharedCallPool) -> None:
        self.config = config
        self.root = _output(config) / "idea_generation"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        base = _base_config(config)
        base["paths"]["runs"] = str(self.root)
        base["run"] = {
            **deepcopy(base["run"]),
            "id": "e18-idea-supply",
            "mock": bool(config["run"]["mock"]),
            "automatically_commit_generated_tasks": False,
        }
        inner = ModelInvoker(
            base,
            self.root,
            output_schemas={"e16_idea_search": tuning.IDEA_RESPONSE_SCHEMA},
            output_parser=tuning.parse_e16_stage,
        )
        self.invoker = E18AttemptInvoker(
            _LimiterInvoker(inner, pool, "selection_codex"),
            self.root / "attempt_journal.json",
        )
        self.lock = threading.RLock()
        self.state = (
            read_json(self.state_path)
            if self.state_path.is_file()
            else {"schema_version": 1, "call_ledger": {}}
        )
        self.save()

    def save(self) -> None:
        with self.lock:
            atomic_write_json(self.state_path, self.state)

    def call(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        call_id = str(spec["call_id"])
        with self.lock:
            prior = self.state["call_ledger"].get(call_id)
            if isinstance(prior, Mapping) and prior.get("status") == "complete":
                return deepcopy(prior["response"])
            if isinstance(prior, Mapping) and prior.get("status") == "failed":
                raise E18AttemptError(f"terminal idea call failure: {call_id}")
            prompt, version, digest = render_idea_prompt(self.config, spec)
            self.state["call_ledger"][call_id] = {
                "status": "reserved",
                "role": "idea_search",
                "stage": "e16_idea_search",
                "model": SOL,
                "reasoning_effort": spec["effort"],
                "private_source": spec["private_source"],
                "private_conditioned": spec["private_conditioned"],
                "prompt_hash": content_hash(prompt),
                "prompt_template_hash": digest,
            }
            self.save()

        def check(value: dict[str, Any]) -> None:
            if value.get("search_call_id") != call_id:
                raise E18SupplyError("idea response changed search call ID")
            if value.get("search_mode") != spec["search_mode"]:
                raise E18SupplyError("idea response changed search mode")
            if [item["candidate_id"] for item in value["ideas"]] != list(
                spec["candidate_ids"]
            ):
                raise E18SupplyError("idea response changed assigned candidate IDs")

        try:
            result = self.invoker.invoke(
                stage="e16_idea_search",
                call_id=call_id,
                prompt=prompt,
                prompt_template_version=version,
                prompt_template_hash=digest,
                parent_ids=[],
                response_validator=check,
                model_name=SOL,
                reasoning_effort=str(spec["effort"]),
            )
            stored = read_json(result.log_dir / "result.json")
            with self.lock:
                self.state["call_ledger"][call_id].update(
                    {
                        "status": "complete",
                        "attempts": result.attempts,
                        "retry_calls": max(0, result.attempts - 1),
                        "cache_source": result.cache_source,
                        "cache_key": result.cache_key,
                        "log_dir": str(result.log_dir.resolve()),
                        "response": deepcopy(result.value),
                        "timing": stored.get("timing"),
                    }
                )
                self.save()
            return deepcopy(result.value)
        except Exception as exc:
            with self.lock:
                self.state["call_ledger"][call_id].update(
                    {"status": "failed", "error": str(exc)}
                )
                self.save()
            raise


class ReservoirLedger:
    """Run-local E18 proposal lifecycle with hard capacity and leases."""

    def __init__(self, path: Path, config: Mapping[str, Any]) -> None:
        self.path = path
        self.config = config
        self.lock = threading.RLock()
        if path.is_file():
            self.value = read_json(path)
        else:
            self.value = {
                "schema_version": 1,
                "pipeline": PIPELINE,
                "run_id": config["run"]["id"],
                "hard_capacity": config["reservoir"]["hard_capacity"],
                "proposals": {},
                "reservations": {},
                "events": [],
                "generated_slates": [],
                "selected_slates": [],
                "finalized_slates": [],
            }
            self.save()
        self.validate()

    def save(self) -> None:
        atomic_write_json(self.path, self.value)

    def validate(self) -> None:
        if self.value.get("schema_version") != 1:
            raise E18SupplyError("unsupported E18 reservoir ledger")
        if self.value.get("run_id") != self.config["run"]["id"]:
            raise E18SupplyError("E18 reservoir belongs to another run")
        proposals = self.value.get("proposals")
        reservations = self.value.get("reservations")
        if not isinstance(proposals, Mapping) or not isinstance(reservations, Mapping):
            raise E18SupplyError("malformed E18 reservoir ledger")
        active = sum(
            isinstance(row, Mapping) and row.get("state") in ACTIVE_PROPOSAL_STATES
            for row in proposals.values()
        )
        if active > int(self.config["reservoir"]["hard_capacity"]):
            raise E18SupplyError("E18 reservoir exceeded hard capacity")
        for proposal_id, lease in reservations.items():
            if (
                proposal_id not in proposals
                or proposals[proposal_id].get("state") != "in_flight"
            ):
                raise E18SupplyError("active lease lacks an in-flight proposal")
            if not isinstance(lease, Mapping) or lease.get("terminal") is True:
                raise E18SupplyError("malformed active E18 lease")

    def active_count(self) -> int:
        return sum(
            row.get("state") in ACTIVE_PROPOSAL_STATES
            for row in self.value["proposals"].values()
        )

    def append_slate(self, slate: int, kernels: Sequence[Mapping[str, Any]]) -> None:
        with self.lock:
            if slate in self.value["generated_slates"]:
                existing = {
                    proposal_id
                    for proposal_id, row in self.value["proposals"].items()
                    if row.get("slate_index") == slate
                }
                wanted = {str(row["candidate_id"]) for row in kernels}
                if existing != wanted:
                    raise E18SupplyError("resumed reservoir slate changed")
                return
            if len(kernels) != 48:
                raise E18SupplyError(
                    "reservoir additions must be complete 48-kernel slates"
                )
            if self.active_count() + 48 > int(
                self.config["reservoir"]["hard_capacity"]
            ):
                raise E18SupplyError("reservoir hard capacity blocks full-slate append")
            now = round(time.time(), 3)
            for row in kernels:
                proposal_id = str(row["candidate_id"])
                if proposal_id in self.value["proposals"]:
                    raise E18SupplyError("duplicate E18 proposal ID")
                self.value["proposals"][proposal_id] = {
                    "proposal_id": proposal_id,
                    "slate_index": slate,
                    "state": "proposed",
                    "concept_hash": content_hash(row["kernel"]),
                    "mechanism_summary": str(row["kernel"]["central_relation"]),
                    "private_source": row["private_source"],
                    "private_conditioned": row["private_conditioned"],
                    "source_call_id": row["private_source_call_id"],
                    "created_at_epoch_seconds": now,
                    "history": [
                        {
                            "state": "proposed",
                            "at_epoch_seconds": now,
                            "reason": "supply complete",
                        }
                    ],
                }
            self.value["generated_slates"].append(slate)
            self.value["events"].append(
                {
                    "event": "slate_appended",
                    "slate_index": slate,
                    "count": 48,
                    "at_epoch_seconds": now,
                }
            )
            self.save()
            self.validate()

    def reserve_selected(
        self,
        *,
        slate: int,
        selected_ids: Sequence[str],
        rejected_ids: Sequence[str],
    ) -> None:
        with self.lock:
            if slate in self.value["selected_slates"]:
                actual = {
                    proposal_id
                    for proposal_id, row in self.value["proposals"].items()
                    if row.get("slate_index") == slate
                    and row.get("state") == "in_flight"
                }
                if actual != set(selected_ids):
                    raise E18SupplyError("resumed E18 slate reservation changed")
                return
            now = round(time.time(), 3)
            for proposal_id in selected_ids:
                row = self.value["proposals"].get(proposal_id)
                if not isinstance(row, dict) or row.get("state") != "proposed":
                    raise E18SupplyError("cannot reserve a non-proposed E18 idea")
                row["state"] = "in_flight"
                row["history"].append(
                    {
                        "state": "in_flight",
                        "at_epoch_seconds": now,
                        "reason": "source-blind selector retained",
                    }
                )
                self.value["reservations"][proposal_id] = {
                    "reservation_id": f"e18:{self.config['run']['id']}:{slate}:{proposal_id}",
                    "owner_run_id": self.config["run"]["id"],
                    "owner_slate_id": slate,
                    "owner_candidate_id": proposal_id,
                    "created_at_epoch_seconds": now,
                    "heartbeat_at_epoch_seconds": now,
                    "lease_seconds": self.config["reservoir"]["lease_seconds"],
                    "terminal": False,
                }
            for proposal_id in rejected_ids:
                row = self.value["proposals"].get(proposal_id)
                if not isinstance(row, dict) or row.get("state") != "proposed":
                    raise E18SupplyError("cannot reject a non-proposed E18 idea")
                row["state"] = "rejected"
                row["rejection_tier"] = "soft_mechanism"
                row["history"].append(
                    {
                        "state": "rejected",
                        "at_epoch_seconds": now,
                        "reason": "not retained for development",
                    }
                )
            self.value["selected_slates"].append(slate)
            self.value["events"].append(
                {
                    "event": "whole_slate_reserved",
                    "slate_index": slate,
                    "selected_proposal_ids": sorted(selected_ids),
                    "at_epoch_seconds": now,
                }
            )
            if not selected_ids:
                # A source-blind selector is allowed to retain nothing, and a
                # small slate may also become empty after the preregistered
                # specialist ceiling is applied.  That is a legitimate
                # terminal slate outcome, not an orchestration failure.
                self.value["finalized_slates"].append(slate)
                self.value["events"].append(
                    {
                        "event": "empty_selected_slate_terminal",
                        "slate_index": slate,
                        "at_epoch_seconds": now,
                    }
                )
            self.save()
            self.validate()

    def heartbeat(self) -> None:
        with self.lock:
            now = round(time.time(), 3)
            for lease in self.value["reservations"].values():
                lease["heartbeat_at_epoch_seconds"] = now
            self.save()

    def recover_expired_leases(
        self, *, now_epoch_seconds: float | None = None, reason: str
    ) -> list[str]:
        """Convert genuinely expired run-local leases into recoverable work.

        Normal restart never calls this method: the unchanged coordinator
        reconstructs and resumes its candidate graphs even after a long host
        outage.  This explicit recovery operation is for an abandoned or
        terminal run, so elapsed wall time can never silently steal live work.
        """

        with self.lock:
            now = float(
                now_epoch_seconds if now_epoch_seconds is not None else time.time()
            )
            recovered: list[str] = []
            for proposal_id, lease in sorted(self.value["reservations"].items()):
                heartbeat = float(lease.get("heartbeat_at_epoch_seconds") or 0)
                duration = float(lease.get("lease_seconds") or 0)
                if heartbeat + duration > now:
                    continue
                row = self.value["proposals"].get(proposal_id)
                if not isinstance(row, dict) or row.get("state") != "in_flight":
                    raise E18SupplyError("expired lease lacks its in-flight proposal")
                row["state"] = "recoverable"
                row["terminal_reason"] = str(reason)
                row["history"].append(
                    {
                        "state": "recoverable",
                        "at_epoch_seconds": round(now, 3),
                        "reason": str(reason),
                    }
                )
                self.value["reservations"].pop(proposal_id)
                self.value["events"].append(
                    {
                        "event": "expired_reservation_recovered",
                        "proposal_id": proposal_id,
                        "reason": str(reason),
                        "at_epoch_seconds": round(now, 3),
                    }
                )
                recovered.append(proposal_id)
            self.save()
            self.validate()
            return recovered

    def finalize(self, proposal_id: str, *, outcome: str, reason: str) -> None:
        if outcome not in TERMINAL_PROPOSAL_STATES:
            raise E18SupplyError(f"invalid terminal E18 proposal outcome {outcome}")
        with self.lock:
            row = self.value["proposals"].get(proposal_id)
            if not isinstance(row, dict):
                raise E18SupplyError("cannot finalize an unknown proposal")
            if row.get("state") in TERMINAL_PROPOSAL_STATES:
                if row.get("state") != outcome:
                    raise E18SupplyError("terminal E18 proposal outcome changed")
                return
            if row.get("state") != "in_flight":
                raise E18SupplyError("only in-flight E18 proposals may finalize")
            now = round(time.time(), 3)
            row["state"] = outcome
            row["terminal_reason"] = str(reason)
            row["history"].append(
                {"state": outcome, "at_epoch_seconds": now, "reason": str(reason)}
            )
            lease = self.value["reservations"].pop(proposal_id, None)
            if not isinstance(lease, Mapping):
                raise E18SupplyError("final proposal lacks an active lease")
            self.value["events"].append(
                {
                    "event": "reservation_resolved",
                    "proposal_id": proposal_id,
                    "outcome": outcome,
                    "at_epoch_seconds": now,
                }
            )
            self.save()
            self.validate()

    def fail_selection_slate(self, slate: int, *, reason: str) -> None:
        """Terminalize a complete unselected window as recoverable work."""

        with self.lock:
            now = round(time.time(), 3)
            rows = [
                row
                for row in self.value["proposals"].values()
                if int(row.get("slate_index") or 0) == slate
            ]
            if rows and all(row.get("state") == "recoverable" for row in rows):
                return
            if len(rows) != 48 or any(row.get("state") != "proposed" for row in rows):
                raise E18SupplyError(
                    "selection failure can only terminalize an untouched 48-kernel window"
                )
            for row in rows:
                row["state"] = "recoverable"
                row["terminal_reason"] = reason
                row["history"].append(
                    {
                        "state": "recoverable",
                        "at_epoch_seconds": now,
                        "reason": reason,
                    }
                )
            self.value["events"].append(
                {
                    "event": "selection_slate_recoverable",
                    "slate_index": slate,
                    "reason": reason,
                    "at_epoch_seconds": now,
                }
            )
            self.save()
            self.validate()


def generate_slate(
    config: Mapping[str, Any],
    runtime: IdeaRuntime,
    design: Mapping[str, Any],
    reservoir: ReservoirLedger,
    slate: int,
) -> dict[str, Any]:
    path = _output(config) / "supply_slates" / f"slate-{slate:02d}.private.json"
    if path.is_file():
        value = read_json(path)
        if value.get("selection_eligible") is True:
            reservoir.append_slate(slate, value["kernels"])
        return value
    specs = [item for item in design["specs"] if item["slate_index"] == slate]
    if len(specs) != 12:
        raise E18SupplyError("supply slate does not contain twelve calls")

    def one(spec: Mapping[str, Any]) -> dict[str, Any]:
        response = runtime.call(spec)
        return {
            "spec": deepcopy(dict(spec)),
            "kernels": [
                {
                    "candidate_id": item["candidate_id"],
                    "kernel": deepcopy(item),
                    "private_source": spec["private_source"],
                    "private_conditioned": spec["private_conditioned"],
                    "private_source_call_id": spec["call_id"],
                    "private_effort": spec["effort"],
                    "nominated_by_generator": response.get("recommended_candidate_id")
                    == item["candidate_id"],
                    "self_development_brief": (
                        deepcopy(response.get("recommended_development_brief"))
                        if response.get("recommended_candidate_id")
                        == item["candidate_id"]
                        else None
                    ),
                }
                for item in response["ideas"]
            ],
        }

    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(one, spec): spec for spec in specs}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except E18AttemptError as exc:
                spec = futures[future]
                failures.append(
                    {
                        "call_id": spec["call_id"],
                        "call_index": spec["call_index"],
                        "error": str(exc),
                        "terminal_after_bounded_attempts": True,
                    }
                )
    rows.sort(key=lambda item: str(item["spec"]["call_id"]))
    kernels = [item for row in rows for item in row["kernels"]]
    complete = (
        not failures
        and len(kernels) == 48
        and len({item["candidate_id"] for item in kernels}) == 48
    )
    if not failures and not complete:
        raise E18SupplyError("supply slate did not produce 48 unique kernels")
    value = {
        "schema_version": 1,
        "slate_index": slate,
        "call_count": 12,
        "completed_call_count": len(rows),
        "failed_call_count": len(failures),
        "nominal_kernel_count": 48,
        "kernel_count": len(kernels),
        "selection_eligible": complete,
        "terminal_status": "eligible_window_frozen"
        if complete
        else "insufficient_supply",
        "supply_failures": sorted(failures, key=lambda item: int(item["call_index"])),
        "calls": rows,
        "kernels": kernels,
        "source_counts": dict(
            sorted(Counter(item["private_source"] for item in kernels).items())
        ),
        "conditioned_kernel_count": sum(
            item["private_conditioned"] for item in kernels
        ),
        "source_labels_private": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _immutable(path, value, f"E18 supply slate {slate}")
    if complete:
        reservoir.append_slate(slate, kernels)
    return value


def _selector_config(
    config: Mapping[str, Any], *, slate: int, reservoir_path: Path
) -> dict[str, Any]:
    value = _base_config(config)
    value["seed"] = int(config["seed"]) + slate * 180_169
    value["paths"]["runs"] = str(_output(config) / "selector_runs")
    value["paths"]["reservoir"] = str(reservoir_path.resolve())
    value["run"] = {
        **deepcopy(value["run"]),
        "id": f"e18-selector-{slate:02d}",
        "mock": bool(config["run"]["mock"]),
        "automatically_commit_generated_tasks": False,
    }
    value["model"]["concurrency"] = int(config["scheduling"]["codex_cap"])
    value["reservoir"] = deepcopy(value["reservoir"])
    value["reservoir"].update(
        {"low_watermark": 48, "high_watermark": 192, "hard_capacity": 640}
    )
    value["selection"] = deepcopy(value["selection"])
    value["selection"].update(
        {
            "eligible_window_size": 48,
            "shard_size": 12,
            "global_selector_max_selected": 10,
            "max_plans": 10,
            "max_realizations": 10,
            "maximum_optional_development_suggestions": 2,
        }
    )
    return value


def _selector_reservoir(
    config: Mapping[str, Any],
    ledger: ReservoirLedger,
    *,
    slate: int,
    supply: Mapping[str, Any],
    destination: Path,
) -> None:
    if destination.is_file():
        return
    selector_config = _selector_config(config, slate=slate, reservoir_path=destination)
    context = _DataContext(selector_config)
    now = time.time()
    source_by_id = {str(item["candidate_id"]): item for item in supply["kernels"]}
    bank_manifest = read_json(_output(config) / "bank_snapshot/manifest.json")
    selector_memory = read_json(Path(bank_manifest["selector_memory_snapshot_path"]))
    proposals = deepcopy(selector_memory["reservation_proposals"])
    active_reservations = deepcopy(selector_memory["active_reservations"])
    rejection_memory = [
        *deepcopy(selector_memory["tiered_rejection_memory"]),
        *deepcopy(selector_memory["quarantine_failure_mode_warnings"]),
    ]
    production_proposal_ids = {
        str(item["proposal_id"]) for item in proposals if isinstance(item, Mapping)
    }
    for index, (proposal_id, state_row) in enumerate(
        sorted(ledger.value["proposals"].items())
    ):
        if proposal_id in production_proposal_ids:
            raise E18SupplyError(
                "run-local proposal ID collides with frozen production memory"
            )
        if state_row["state"] not in ACTIVE_PROPOSAL_STATES:
            tier = {
                "rejected": "soft_mechanism",
                "completed_uncommitted": "hard_duplicate",
                "recoverable": "recoverable_work",
            }.get(str(state_row["state"]))
            if tier is not None:
                rejection_memory.append(
                    {
                        "memory_id": f"e18-{tier}:{proposal_id}",
                        "tier": tier,
                        "proposal_id": proposal_id,
                        "concept_hash": state_row["concept_hash"],
                        "mechanism_summary": state_row["mechanism_summary"][:700],
                        "reason": state_row.get("terminal_reason", "not retained"),
                        "source_run_id": config["run"]["id"],
                        "created_at_epoch_seconds": state_row[
                            "created_at_epoch_seconds"
                        ],
                        "active": True,
                    }
                )
            continue
        # Prefetch may already have placed the next frozen window in the
        # run-local ledger. It is not eligible for this slate and must not
        # enter the selector as a placeholder proposal. Earlier in-flight
        # mechanisms remain visible solely as reservation memory.
        if state_row["slate_index"] != slate and state_row["state"] == "proposed":
            continue
        if state_row["slate_index"] == slate:
            row = source_by_id[proposal_id]
            kernel = row["kernel"]
            calibration_ids = next(
                call["spec"]["calibration_task_ids"]
                for call in supply["calls"]
                if call["spec"]["call_id"] == row["private_source_call_id"]
            )
            calibration_descriptions = next(
                call["spec"]["calibration_descriptions"]
                for call in supply["calls"]
                if call["spec"]["call_id"] == row["private_source_call_id"]
            )
        else:
            # Earlier in-flight work exists only as a reservation/novelty block;
            # it must not enter the current eligible window.
            kernel = {
                "candidate_id": proposal_id,
                "search_mode": "concept_led",
                "central_relation": state_row["mechanism_summary"],
                "why_coherent": "run-local reserved mechanism",
                "novel_concept_potential": "reserved",
                "functional_composition_potential": "reserved",
                "why_multiple_training_examples_may_be_useful": "reserved",
                "possible_test_generalization": "reserved",
                "main_authoring_risk": "duplicate in-flight development",
            }
            calibration_ids = []
            calibration_descriptions = []
        proposal = make_proposal(
            kernel,
            supply_run_id=config["run"]["id"],
            supply_cycle_id=f"e18-slate-{state_row['slate_index']:02d}",
            calibration_task_ids=calibration_ids,
            calibration_descriptions=calibration_descriptions,
            calibration_version="e18-per-slate-disjoint-real-v1",
            prompt_version="e18-integrated-supply-v1",
            prompt_hash=content_hash({"call": state_row["source_call_id"]}),
            model=SOL,
            reasoning_effort=(
                source_by_id[proposal_id]["private_effort"]
                if proposal_id in source_by_id
                else "xhigh"
            ),
            source_bank_snapshot_id=context.bank_snapshot_id,
            embedding={
                "provider": "deferred-selector-retrieval",
                "model": "not-stored",
                "dimension": 0,
                "vector_hash": content_hash("not-stored"),
            },
            created_at_epoch_seconds=now + index / 1000,
        )
        if state_row["state"] == "in_flight":
            proposal["state"] = "in_flight"
            lease = ledger.value["reservations"][proposal_id]
            active_reservations.append(
                {
                    "reservation_id": lease["reservation_id"],
                    "proposal_id": proposal_id,
                    "concept_hash": proposal["concept_hash"],
                    "mechanism_summary": state_row["mechanism_summary"],
                    "owner_run_id": lease["owner_run_id"],
                    "owner_slate_id": str(lease["owner_slate_id"]),
                    "owner_candidate_id": lease["owner_candidate_id"],
                    "created_at_epoch_seconds": lease["created_at_epoch_seconds"],
                    "heartbeat_at_epoch_seconds": lease["heartbeat_at_epoch_seconds"],
                    "lease_seconds": lease["lease_seconds"],
                }
            )
        proposals.append(proposal)
    value = {
        "schema_version": 1,
        "reservoir_id": "arcgen-v4-persistent-idea-reservoir",
        "proposals": proposals,
        "active_reservations": active_reservations,
        "reservation_history": [],
        "rejection_memory": rejection_memory,
        "frozen_windows": [],
        "supply_history": [],
        "lifecycle_events": [],
    }
    validate_reservoir(value)
    atomic_write_json(destination, value)


def select_slate(
    config: Mapping[str, Any],
    pool: SharedCallPool,
    ledger: ReservoirLedger,
    supply: Mapping[str, Any],
    slate: int,
) -> dict[str, Any]:
    directory = _output(config) / "selection_slates" / f"slate-{slate:02d}"
    checkpoint = directory / "selection_checkpoint.private.json"
    if checkpoint.is_file():
        return read_json(checkpoint)
    directory.mkdir(parents=True, exist_ok=True)
    reservoir_path = directory / "selector_reservoir.json"
    _selector_reservoir(
        config,
        ledger,
        slate=slate,
        supply=supply,
        destination=reservoir_path,
    )
    selector = V4ControlPipeline(
        _selector_config(config, slate=slate, reservoir_path=reservoir_path)
    )
    selector.runtime.invoker = E18AttemptInvoker(
        _LimiterInvoker(selector.runtime.invoker, pool, "selection_codex"),
        directory / "selector_attempt_journal.json",
    )
    briefs = selector.select_slate()
    source = {str(item["candidate_id"]): item for item in supply["kernels"]}
    rows: list[dict[str, Any]] = []
    special_counts: Counter[str] = Counter()
    dropped_selected: list[str] = []
    for brief in sorted(briefs, key=lambda item: int(item["development_priority"])):
        candidate_id = str(brief["candidate_id"])
        private = source[candidate_id]
        special = str(private["private_source"])
        if special in SPECIAL_SOURCES:
            if special_counts[special] >= 1 or sum(special_counts.values()) >= 2:
                dropped_selected.append(candidate_id)
                continue
            special_counts[special] += 1
        record = selector.state["records"][candidate_id]
        assessment = deepcopy(record.get("selector_assessment") or {})
        rows.append(
            {
                "source_candidate_id": candidate_id,
                "lineage_id": content_hash({"e18_source": candidate_id}),
                "slate_index": slate,
                "source_development_priority": int(record["development_priority"]),
                "search_mode": private["kernel"]["search_mode"],
                "kernel": deepcopy(private["kernel"]),
                "editorial_brief": deepcopy(record["editorial_brief"]),
                "private_source": private["private_source"],
                "private_conditioned": private["private_conditioned"],
                "private_source_call_id": private["private_source_call_id"],
                "private_effort": private["private_effort"],
                "selector_assessment": assessment,
                "novelty": {
                    "selector_archive_checked": True,
                    "confirmed_same_procedure_duplicate": False,
                    "within_window_sibling_ids": list(
                        assessment.get("same_minimal_procedure_with") or []
                    ),
                    "selected_as_sibling_representative": bool(
                        assessment.get("same_minimal_procedure_with")
                    ),
                },
            }
        )
    # Specialized source limits are ceilings, never selection bonuses.  The
    # global selector ranks without source labels; only after that judgment do
    # we enforce the frozen >=70% invitation-free floor.  If a small slate
    # cannot carry every selected specialized source, drop the lowest-ranked
    # specialist rather than weakening ordinary quality selection.
    maximum_invited = floor(
        len(rows) * (1.0 - float(config["design"]["minimum_invitation_free_fraction"]))
        + 1e-9
    )
    specialists = [item for item in rows if item["private_source"] in SPECIAL_SOURCES]
    while len(specialists) > maximum_invited:
        dropped = max(
            specialists,
            key=lambda item: (
                int(item["source_development_priority"]),
                str(item["source_candidate_id"]),
            ),
        )
        rows.remove(dropped)
        dropped_selected.append(str(dropped["source_candidate_id"]))
        special_counts[str(dropped["private_source"])] -= 1
        specialists.remove(dropped)
    # The selector already atomically reserved every qualitative selection.
    # Release source-ceiling drops in its isolated reservoir before planning.
    for candidate_id in dropped_selected:
        resolve_proposal(
            reservoir_path,
            proposal_id=candidate_id,
            outcome="rejected",
            reason="specialized-source ceiling; qualitative rank preserved",
            source_run_id=config["run"]["id"],
            memory_tier="soft_mechanism",
        )
    selected_ids = [str(row["source_candidate_id"]) for row in rows]
    all_ids = [str(item["candidate_id"]) for item in supply["kernels"]]
    ledger.reserve_selected(
        slate=slate,
        selected_ids=selected_ids,
        rejected_ids=[item for item in all_ids if item not in selected_ids],
    )
    value = {
        "schema_version": 1,
        "slate_index": slate,
        "selection_completed": True,
        "terminal_status": (
            "selected_slate_frozen" if rows else "no_development_selected"
        ),
        "planning_started": False,
        "source_labels_hidden_from_selector": True,
        "whole_development_slate_reserved_before_planning": True,
        "full_selector_selected_count": len(briefs),
        "source_ceiling_drop_ids": sorted(dropped_selected),
        "development_count": len(rows),
        "specialized_selected_counts": dict(sorted(special_counts.items())),
        "source_rows": rows,
        "selector_state_path": str(selector.state_path.resolve()),
        "selector_call_ledger": deepcopy(selector.state["call_ledger"]),
        "frozen_window": deepcopy(selector.state["frozen_window"]),
        "slate_reservation": deepcopy(selector.state["slate_reservation"]),
    }
    _immutable(checkpoint, value, f"E18 selection slate {slate}")
    atomic_write_json(
        directory / "selected_slate.blinded.json",
        {
            "schema_version": 1,
            "slate_index": slate,
            "private_labels_removed": True,
            "candidates": [
                {
                    "candidate_id": row["source_candidate_id"],
                    "kernel": row["kernel"],
                    "editorial_brief": row["editorial_brief"],
                }
                for row in rows
            ],
        },
    )
    return value


__all__ = [
    "ACTIVE_PROPOSAL_STATES",
    "SPECIAL_SOURCES",
    "E18SupplyError",
    "IdeaRuntime",
    "ReservoirLedger",
    "freeze_bank_snapshot",
    "freeze_supply_design",
    "generate_slate",
    "render_idea_prompt",
    "select_slate",
    "validate_selector_memory_snapshot",
]
