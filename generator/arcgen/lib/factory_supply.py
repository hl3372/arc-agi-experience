"""Explicit current-factory supply and source-blind selection.

The historical E18 prompts, source schedule, calibration allocation, and
reservoir lifecycle remain useful primitives.  This module binds them to an
explicit runtime, freezes model-visible inputs, and implements the adopted
twelve-plus-four capacity policy without constructing a legacy experiment.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from math import floor
from pathlib import Path
from random import Random
from typing import Any
import threading
import time

import yaml
from jsonschema import Draft202012Validator

from . import e18_supply as legacy
from .embeddings import build_embedding_provider
from .factory_config import resolve_role
from .retrieval import retrieve_union
from .schemas import SchemaError, parse_json_object
from .util import atomic_write_json, atomic_write_text, content_hash, load_prompt, read_json, render_prompt
from .v4_pipeline import _DataContext
from .v4_reservoir import concept_hash
from .v4_schemas import STRICT_OUTPUT_SCHEMAS


FactorySupplyError = legacy.E18SupplyError
SPECIAL_SOURCES = legacy.SPECIAL_SOURCES
IDEA_PURPOSES = {
    "ordinary_bank_conditioned": "idea_conditioned",
    "ordinary_unconditioned": "idea_unconditioned",
    "logical_leap": "idea_logical_leap",
    "episode_joint_inference": "idea_episode_joint",
}
I1_SHA256 = "d57888cb31f0c5853db8f51ad9bbc63430901c836974fcae820aa60fdae27039"
DEFERRED = "qualified_but_deferred"
SELECTOR_SCHEMAS = {stage: deepcopy(STRICT_OUTPUT_SCHEMAS[stage]) for stage in ("v4_shard_selector", "v4_global_selector")}
# The runtime checks exact membership of the actual frozen shard. Historical
# schemas keep their twelve-item requirement; working shards may contain 8–12.
SELECTOR_SCHEMAS["v4_shard_selector"]["properties"]["candidate_assessments"]["minItems"] = 8
SELECTOR_SCHEMAS["v4_global_selector"]["properties"]["selected"]["maxItems"] = 16
SELECTOR_SCHEMAS["v4_global_selector"]["properties"]["selected"]["items"]["properties"]["development_priority"]["maximum"] = 16


def parse_factory_selector(stage: str, data: str | bytes | bytearray | Mapping[str, Any]) -> dict[str, Any]:
    """The V4 selector contract, with only the adopted global depth extended."""
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else dict(data)
    if stage not in SELECTOR_SCHEMAS:
        raise SchemaError("$.stage", f"unknown factory selector stage: {stage}")
    errors = sorted(Draft202012Validator(SELECTOR_SCHEMAS[stage]).iter_errors(value), key=lambda error: list(error.absolute_path))
    if errors:
        raise SchemaError("$", errors[0].message)
    assessed = [row["candidate_id"] for row in value["candidate_assessments"]]
    selected = [row["candidate_id"] for row in value["selected"]]
    decisions = {row["candidate_id"] for row in value["candidate_assessments"] if row["decision"] == "select"}
    if len(assessed) != len(set(assessed)) or len(selected) != len(set(selected)) or set(selected) != decisions:
        raise SchemaError("$", "selector IDs or select decisions are inconsistent")
    if sorted(row["development_priority"] for row in value["selected"]) != list(range(1, len(selected) + 1)):
        raise SchemaError("$.selected", "development priorities must be contiguous")
    return value


def _sha(path: str | Path) -> str:
    return content_hash(Path(path).read_bytes())


def _identity(config: Mapping[str, Any]) -> str:
    return content_hash({
        "run": {key: config["run"][key] for key in ("id", "account", "replicate")}, "seed": config["seed"],
        "slates": config["run"]["slates"], "policy": config["policy"],
        "supply": config["supply"], "selection": config["selection"],
        "embedding": config["embedding"],
        "execution": config.get("execution", {"minimum_idea_kernels": 48}),
        "roles": {purpose: resolve_role(config, purpose) for purpose in (
            *IDEA_PURPOSES.values(), "shard_selection", "global_selection")},
    })


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _verify_artifact(row: Mapping[str, Any], root: Path) -> Path:
    path = Path(str(row["path"])).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or _sha(path) != row["sha256"]:
        raise FactorySupplyError(f"frozen factory supply artifact missing or changed: {path}")
    return path


def _base_context(config: Mapping[str, Any]) -> _DataContext:
    """Read the saved calibration recipe, without a Sol-only config validator."""
    base = yaml.safe_load(Path(config["_references"]["base_v4_control"]).read_text(encoding="utf-8"))
    root = Path(config["_repository_root"])
    base["paths"] = {key: str((root / str(value)).resolve()) for key, value in base["paths"].items()}
    base["paths"]["committed_task_registry"] = str(config["_paths"]["active_registry"])
    # The original base seed chooses the 90-example pool; the campaign seed
    # independently chooses each slate's 48 distinct examples from that pool.
    return _DataContext(base)


def render_idea_prompt(config: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[str, str, str]:
    """Preserve the exact tested I1 append/hash seam, on its accepted source."""
    prompt, version, digest = legacy.render_idea_prompt(config, spec)
    if spec["private_source"] != "ordinary_unconditioned" or not config["policy"]["ordinary_unconditioned_i1"]:
        return prompt, version, digest
    path = Path(config["_references"]["i1_prompt"])
    if _sha(path) != I1_SHA256:
        raise FactorySupplyError("accepted ordinary-unconditioned I1 addendum changed")
    addendum, addendum_version, addendum_hash = load_prompt(path)
    combined = prompt + "\n\n" + addendum
    return combined, version + "+" + addendum_version, content_hash({
        "base_prompt_hash": digest, "addendum_hash": addendum_hash, "combined_prompt": combined,
    })


def freeze_supply_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze bank, real calibration, prompt text and routing once per run.

    Resume validates those saved artifacts, never today's mutable task bank.
    No embedding or language-model request is made by this function.
    """
    root = Path(config["_output_directory"])
    path = root / "frozen_supply_design.private.json"
    identity = _identity(config)
    if path.is_file():
        design = read_json(path)
        if design.get("factory_identity") != identity or design.get("design_hash") != content_hash({key: value for key, value in design.items() if key != "design_hash"}):
            raise FactorySupplyError("factory supply profile or recipe changed during resume")
        for artifact in design["artifacts"]:
            _verify_artifact(artifact, root)
        return design

    bank = legacy.freeze_bank_snapshot(config)
    context = _base_context(config)
    pool = sorted(context.generator_pool, key=lambda item: item.task_id)
    if len(pool) < 48:
        raise FactorySupplyError("fewer than 48 generator-visible calibration tasks")
    artifacts = [_artifact(root / "bank_snapshot/manifest.json")]
    for name in ("private_rows", "generator_visible_rows", "selector_memory_snapshot"):
        artifacts.append(_artifact(Path(bank[f"{name}_path"])))
    selector_memory = read_json(Path(bank["selector_memory_snapshot_path"]))
    selector_context = {
        # A1's full-rule bank context belongs to generators. Selectors retain
        # the actual saved digest plus concise retrieved references.
        "accepted_bank_digest": context.accepted_digest(),
        "accepted_concept_hashes": sorted(context.accepted_concept_hashes()),
        "references": deepcopy(context.references),
        "selector_memory": selector_memory,
        "anchor_ids": sorted(context.anchor_ids),
        "embedding": deepcopy(config["embedding"]),
        "calibration_pool": [{"task_id": item.task_id, "rule_description": item.rule_description} for item in pool],
        "base_calibration_seed": context.seed,
    }
    context_path = root / "bank_snapshot/selector_context.private.json"
    atomic_write_json(context_path, selector_context)
    artifacts.append(_artifact(context_path))
    selector_prompts = {}
    for purpose, filename in (("shard_selection", "v4_shard_selector.md"), ("global_selection", "v4_global_selector.md")):
        text, version, digest = load_prompt(Path(config["_repository_root"]) / "prompts" / filename)
        destination = root / "supply_prompts" / filename
        atomic_write_text(destination, text)
        row = {**_artifact(destination), "version": version, "template_hash": digest}
        selector_prompts[purpose] = row
        artifacts.append(_artifact(destination))

    specs = []
    for slate in range(1, int(config["run"]["slates"]) + 1):
        calibration = list(pool)
        Random(int(config["seed"]) + slate * 180_043).shuffle(calibration)
        for index, source in enumerate(legacy._source_specs_for_slate(config, slate), 1):
            role = resolve_role(config, IDEA_PURPOSES[source["source"]])
            group = calibration[(index - 1) * 4:index * 4]
            spec = {
                "slate_index": slate, "call_index": index,
                "call_id": f"factory-s{slate:02d}-c{index:02d}-{source['label']}",
                "candidate_ids": legacy._candidate_ids(config, slate, index),
                "search_mode": source["search_mode"], "private_source": source["source"],
                "private_conditioned": source["conditioned"],
                "model": role["model"], "effort": role["reasoning_effort"],
                "calibration_task_ids": [item.task_id for item in group],
                "calibration_descriptions": [item.rule_description for item in group],
                "bank_context": ({
                    "path": bank["generator_visible_rows_path"],
                    "sha256": bank["generator_visible_rows_sha256"],
                    "known_lineage_count": bank["known_lineage_count"],
                } if source["conditioned"] else None),
            }
            prompt, version, digest = render_idea_prompt(config, spec)
            destination = root / "supply_prompts" / f"{spec['call_id']}.md"
            atomic_write_text(destination, prompt)
            spec["rendered_prompt"] = {**_artifact(destination), "version": version, "template_hash": digest}
            artifacts.append(_artifact(destination))
            specs.append(spec)
    design = {
        "schema_version": 1, "pipeline": config["pipeline"], "factory_identity": identity,
        "run_id": config["run"]["id"], "seed": config["seed"],
        "slates": config["run"]["slates"], "calls_per_slate": 12,
        "kernels_per_call": 4, "kernels_per_slate": 48,
        "bank_snapshot": bank, "selector_context": _artifact(context_path),
        "selector_prompts": selector_prompts, "specs": specs, "artifacts": artifacts,
        "source_labels_private": True, "silent_truncation": False,
        "automatic_commit": False,
    }
    design["design_hash"] = content_hash(design)
    legacy.validate_selector_memory_snapshot(bank, design)
    legacy._immutable(path, design, "factory supply design")
    return design


class ReservoirLedger(legacy.ReservoirLedger):
    """The existing lifecycle with atomic bounded capacity deferral."""

    def __init__(self, path: Path, config: Mapping[str, Any]) -> None:
        if not path.resolve().is_relative_to(Path(config["_output_directory"]).resolve()):
            raise FactorySupplyError("factory reservoir must be inside its own run directory")
        super().__init__(path, config)
        self.value["pipeline"] = config["pipeline"]
        self.save()

    def append_slate(self, slate: int, kernels: Sequence[Mapping[str, Any]]) -> None:
        with self.lock:
            minimum = self.config.get("execution", {}).get("minimum_idea_kernels", 48)
            if not minimum <= len(kernels) <= 48 or len({str(row["candidate_id"]) for row in kernels}) != len(kernels):
                raise FactorySupplyError("factory supply requires a valid unique frozen window before reservoir mutation")
            for item in kernels:
                prior = self.value["proposals"].get(str(item["candidate_id"]))
                if prior is not None and prior["concept_hash"] != content_hash(item["kernel"]):
                    raise FactorySupplyError("resumed factory kernel changed")
            if len(kernels) == 48:
                super().append_slate(slate, kernels)
            elif slate in self.value["generated_slates"]:
                existing = {key for key, row in self.value["proposals"].items() if row["slate_index"] == slate}
                if existing != {str(row["candidate_id"]) for row in kernels}:
                    raise FactorySupplyError("resumed reservoir slate changed")
            else:
                if self.active_count() + len(kernels) > self.config["reservoir"]["hard_capacity"]:
                    raise FactorySupplyError("reservoir hard capacity blocks window append")
                if any(str(row["candidate_id"]) in self.value["proposals"] for row in kernels):
                    raise FactorySupplyError("duplicate factory proposal ID")
                now = round(time.time(), 3)
                for row in kernels:
                    key = str(row["candidate_id"])
                    self.value["proposals"][key] = {
                        "proposal_id": key, "slate_index": slate, "state": "proposed",
                        "concept_hash": content_hash(row["kernel"]),
                        "mechanism_summary": str(row["kernel"]["central_relation"]),
                        "private_source": row["private_source"], "private_conditioned": row["private_conditioned"],
                        "source_call_id": row["private_source_call_id"], "created_at_epoch_seconds": now,
                        "history": [{"state": "proposed", "at_epoch_seconds": now,
                                     "reason": "explicit bounded best-effort window"}],
                    }
                self.value["generated_slates"].append(slate)
                self.value["events"].append({"event": "slate_appended", "slate_index": slate,
                                             "count": len(kernels), "at_epoch_seconds": now})
            for item in kernels:
                self.value["proposals"][str(item["candidate_id"])]["kernel"] = deepcopy(item["kernel"])
            self.save()
            self.validate()

    def reserve_selected(self, *, slate: int, selected_ids: Sequence[str], rejected_ids: Sequence[str], deferred_ids: Sequence[str] = ()) -> None:
        decision = {"selected": list(selected_ids), "rejected": list(rejected_ids), "deferred": list(deferred_ids)}
        with self.lock:
            decisions = self.value.get("selection_dispositions", {})
            if str(slate) in decisions:
                if decisions[str(slate)] != decision:
                    raise FactorySupplyError("resumed factory slate disposition changed")
                self.validate()
                return
            groups = [*selected_ids, *rejected_ids, *deferred_ids]
            slate_ids = {key for key, row in self.value["proposals"].items() if row["slate_index"] == slate}
            minimum = self.config.get("execution", {}).get("minimum_idea_kernels", 48)
            if len(set(groups)) != len(groups) or set(groups) != slate_ids or not minimum <= len(slate_ids) <= 48:
                raise FactorySupplyError("factory reservation must partition the entire frozen slate")
            if len(selected_ids) > 12 or len(deferred_ids) > 4:
                raise FactorySupplyError("factory reservation exceeded adopted routine/deferred capacity")
            if any(self.value["proposals"][key]["state"] != "proposed" for key in groups):
                raise FactorySupplyError("factory reservation requires untouched proposals")
            mechanisms = [concept_hash(self.value["proposals"][key]["kernel"]["central_relation"])
                          for key in selected_ids]
            if len(mechanisms) != len(set(mechanisms)):
                raise FactorySupplyError("selected slate contains exact mechanism siblings")
            updated = deepcopy(self.value)
            now = round(time.time(), 3)
            for key in groups:
                row = updated["proposals"][key]
                state = "in_flight" if key in selected_ids else DEFERRED if key in deferred_ids else "rejected"
                row["state"] = state
                reason = "source-blind selector retained" if state == "in_flight" else "qualified capacity overflow" if state == DEFERRED else "not retained for development"
                row["history"].append({"state": state, "at_epoch_seconds": now, "reason": reason})
                if state == "rejected":
                    row["rejection_tier"] = "soft_mechanism"
                if state == "in_flight":
                    updated["reservations"][key] = {
                        "reservation_id": f"factory:{self.config['run']['id']}:{slate}:{key}",
                        "owner_run_id": self.config["run"]["id"], "owner_slate_id": slate,
                        "owner_candidate_id": key, "created_at_epoch_seconds": now,
                        "heartbeat_at_epoch_seconds": now,
                        "lease_seconds": self.config["reservoir"]["lease_seconds"], "terminal": False,
                    }
            updated.setdefault("selection_dispositions", {})[str(slate)] = decision
            updated["selected_slates"].append(slate)
            updated["events"].append({"event": "whole_slate_reserved", "slate_index": slate, "selected_proposal_ids": list(selected_ids), "deferred_proposal_ids": list(deferred_ids), "at_epoch_seconds": now})
            if not selected_ids:
                updated["finalized_slates"].append(slate)
                updated["events"].append({"event": "empty_selected_slate_terminal", "slate_index": slate, "at_epoch_seconds": now})
            previous = self.value
            self.value = updated
            try:
                self.validate()
                self.save()
            except BaseException:
                self.value = previous
                raise

    def validate(self) -> None:
        super().validate()
        deferred = Counter()
        for key, row in self.value["proposals"].items():
            if row["state"] not in {*legacy.ACTIVE_PROPOSAL_STATES, *legacy.TERMINAL_PROPOSAL_STATES, DEFERRED}:
                raise FactorySupplyError("unknown factory proposal state")
            if row["state"] == DEFERRED:
                deferred[row["slate_index"]] += 1
                if key in self.value["reservations"] or "rejection_tier" in row:
                    raise FactorySupplyError("qualified deferred work acquired a reservation or rejection tier")
        if any(count > 4 for count in deferred.values()):
            raise FactorySupplyError("qualified deferred slate exceeded four candidates")
        for slate, decision in self.value.get("selection_dispositions", {}).items():
            for disposition, ids in decision.items():
                for key in ids:
                    row = self.value["proposals"].get(key)
                    allowed = ({"in_flight", *legacy.TERMINAL_PROPOSAL_STATES} if disposition == "selected" else {DEFERRED} if disposition == "deferred" else {"recoverable"} if disposition == "recoverable" else {"rejected"})
                    if row is None or str(row["slate_index"]) != slate or row["state"] not in allowed:
                        raise FactorySupplyError("factory proposal disagrees with its frozen slate disposition")

    def terminalize_ineligible_slate(self, slate: int, *, reason: str) -> None:
        """Preserve an incomplete eligible window as operationally recoverable."""
        with self.lock:
            rows = [row for row in self.value["proposals"].values() if row["slate_index"] == slate]
            if len(rows) == 48:
                self.fail_selection_slate(slate, reason=reason)
            else:
                if any(row["state"] not in {"proposed", "recoverable"} for row in rows):
                    raise FactorySupplyError("cannot abort an already selected window")
                now = round(time.time(), 3)
                for row in rows:
                    if row["state"] == "proposed":
                        row.update(state="recoverable", terminal_reason=reason)
                        row["history"].append({"state": "recoverable", "at_epoch_seconds": now, "reason": reason})
            ids = [key for key, row in self.value["proposals"].items() if row["slate_index"] == slate]
            self.value.setdefault("selection_dispositions", {})[str(slate)] = {"recoverable": ids}
            for name in ("selected_slates", "finalized_slates"):
                if slate not in self.value[name]:
                    self.value[name].append(slate)
            self.validate()
            self.save()


def apply_source_ceilings(rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply the existing ceilings after qualitative judgment; never add rows."""
    kept, dropped = [], []
    counts: Counter[str] = Counter()
    for original in sorted(rows, key=lambda item: int(item["source_development_priority"])):
        row = deepcopy(dict(original))
        source = str(row["private_source"])
        if source in SPECIAL_SOURCES:
            if counts[source] >= 1 or sum(counts.values()) >= 2:
                dropped.append(str(row["source_candidate_id"]))
                continue
            counts[source] += 1
        kept.append(row)
    # Retain the saved floor computation: its conservative result on short
    # slates is part of the current policy, not a new specialist quota.
    maximum = floor(len(kept) * (1.0 - float(selection["invitation_free_minimum_share"])) + 1e-9)
    specialists = [row for row in kept if row["private_source"] in SPECIAL_SOURCES]
    while len(specialists) > maximum:
        row = max(specialists, key=lambda item: (int(item["source_development_priority"]), str(item["source_candidate_id"])))
        specialists.remove(row)
        kept.remove(row)
        dropped.append(str(row["source_candidate_id"]))
    return kept, dropped


class FactorySupply:
    """Small supply/selection service; the caller owns workers and runtime."""

    def __init__(self, config: Mapping[str, Any], runtime: Any, ledger: ReservoirLedger | None = None, *, embedder: Any = None, context_guard: Callable[[str, str], None] | None = None) -> None:
        self.config, self.runtime = config, runtime
        self.root = Path(config["_output_directory"])
        profile = getattr(runtime, "role_profile", None)
        if profile is not None and any(profile.get(purpose) != resolve_role(config, purpose) for purpose in (*IDEA_PURPOSES.values(), "shard_selection", "global_selection")):
            raise FactorySupplyError("supply profile disagrees with the actual invocation runtime")
        self.ledger = ledger if ledger is not None else ReservoirLedger(self.root / "reservoir.json", config)
        self.embedder, self.context_guard = embedder, context_guard
        self.design: dict[str, Any] | None = None
        self.lock = threading.RLock()

    def freeze_supply_design(self) -> dict[str, Any]:
        with self.lock:
            if self.design is None:
                design = freeze_supply_inputs(self.config)
                if self.context_guard is not None:
                    for spec in design["specs"]:
                        path = _verify_artifact(spec["rendered_prompt"], self.root)
                        self.context_guard(path.read_text(encoding="utf-8"), IDEA_PURPOSES[spec["private_source"]])
                self.design = design
            return deepcopy(self.design)

    freeze_supply_inputs = freeze_supply_design

    def _invoke(self, *, purpose: str, stage: str, call_id: str, prompt: str, version: str, digest: str, parent_ids: Sequence[str], validator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        if self.context_guard is not None:
            self.context_guard(prompt, purpose)
        result = self.runtime.invoke_rendered(purpose=purpose, stage=stage, call_id=call_id, prompt=prompt, prompt_template_version=version, prompt_template_hash=digest, parent_ids=list(parent_ids), response_validator=validator)
        return deepcopy(result.value)

    def call(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """Idea-call protocol consumed by the pure legacy slate assembler."""
        metadata = spec["rendered_prompt"]
        path = _verify_artifact(metadata, self.root)
        def validate(value: dict[str, Any]) -> None:
            if value.get("search_call_id") != spec["call_id"] or value.get("search_mode") != spec["search_mode"] or [item["candidate_id"] for item in value["ideas"]] != spec["candidate_ids"]:
                raise FactorySupplyError("idea response changed assigned call, mode, or candidate IDs")
        return self._invoke(purpose=IDEA_PURPOSES[str(spec["private_source"])], stage="e16_idea_search", call_id=str(spec["call_id"]), prompt=path.read_text(encoding="utf-8"), version=str(metadata["version"]), digest=str(metadata["template_hash"]), parent_ids=[], validator=validate)

    def generate_slate(self, slate: int) -> dict[str, Any]:
        design = self.freeze_supply_design()
        result = legacy.generate_slate(self.config, self, design, self.ledger, slate)
        expected = {row["call_id"]: row for row in design["specs"] if row["slate_index"] == slate}
        for call in result["calls"]:
            if call["spec"] != expected.get(call["spec"]["call_id"]):
                raise FactorySupplyError("resumed factory supply call changed frozen design")
        minimum = self.config.get("execution", {}).get("minimum_idea_kernels", 48)
        if minimum < 48 and not result["selection_eligible"] and len(result["kernels"]) >= minimum:
            # Keep the original supply/failure artifact immutable. Selection
            # uses this separately frozen operational view, never a retry.
            result = deepcopy(result)
            result.update(selection_eligible=True, terminal_status="bounded_partial_window_frozen",
                          complete_supply=False, minimum_idea_kernels=minimum)
            destination = self.root / "supply_slates" / f"slate-{slate:02d}.eligible.private.json"
            legacy._immutable(destination, result, "best-effort eligible window")
            self.ledger.append_slate(slate, result["kernels"])
        return result

    def _selector_inputs(self, supply: Mapping[str, Any], slate: int, design: Mapping[str, Any]) -> dict[str, Any]:
        directory = self.root / "selection_slates" / f"slate-{slate:02d}"
        path = directory / "selector_inputs.blinded.json"
        identity = content_hash({"factory": design["factory_identity"], "supply": supply})
        if path.is_file():
            value = read_json(path)
            if value.get("input_identity") != identity or value.get("payload_hash") != content_hash(value["payload"]) or value.get("snapshot_hash") != content_hash({key: row for key, row in value.items() if key != "snapshot_hash"}):
                raise FactorySupplyError("frozen source-blind selector inputs changed")
            return value
        minimum = self.config.get("execution", {}).get("minimum_idea_kernels", 48)
        if supply.get("selection_eligible") is not True or not minimum <= len(supply["kernels"]) <= 48:
            raise FactorySupplyError("selection requires a valid frozen idea window")
        context = read_json(_verify_artifact(design["selector_context"], self.root))
        bank = design["bank_snapshot"]
        _verify_artifact({"path": bank["selector_memory_snapshot_path"], "sha256": bank["selector_memory_snapshot_sha256"]}, self.root)
        memory = context["selector_memory"]
        exact_hashes = set(context["accepted_concept_hashes"])
        exact_hashes.update(row.get("concept_hash") or concept_hash(row["mechanism_summary"])
                            for row in memory["active_reservations"])
        reservations = [{"reservation_id": row["reservation_id"], "mechanism_summary": row["mechanism_summary"]} for row in memory["active_reservations"]]
        rejected = {tier: [] for tier in ("hard_duplicate", "soft_mechanism", "failure_mode", "recoverable_work")}
        for row in [*memory["tiered_rejection_memory"], *memory["quarantine_failure_mode_warnings"]]:
            if row.get("active", True):
                rejected[row["tier"]].append({"mechanism_summary": row["mechanism_summary"], "reason": row["reason"]})
                if row["tier"] == "hard_duplicate":
                    # Full native V4 hashes remain available even when the
                    # model-visible summary is short or outside its last 30.
                    exact_hashes.add(row.get("concept_hash") or concept_hash(row["mechanism_summary"]))
        with self.ledger.lock:
            for key, row in sorted(self.ledger.value["proposals"].items()):
                if row["slate_index"] == slate:
                    continue
                if row["state"] == "in_flight":
                    reservations.append({"reservation_id": self.ledger.value["reservations"][key]["reservation_id"], "mechanism_summary": row["mechanism_summary"]})
                    exact_hashes.add(concept_hash(row["mechanism_summary"]))
                tier = {"completed_uncommitted": "hard_duplicate", "rejected": "soft_mechanism", "recoverable": "recoverable_work"}.get(row["state"])
                if tier:
                    rejected[tier].append({"mechanism_summary": row["mechanism_summary"][:700], "reason": row.get("terminal_reason", "not retained")})
                    if tier == "hard_duplicate":
                        # E18's stored "concept_hash" hashes the WHOLE kernel;
                        # use its full relation text for normalized-mechanism
                        # matching, not that differently defined field.
                        exact_hashes.add(concept_hash(row["mechanism_summary"]))
                # Future proposed slates and qualified capacity overflow are
                # neither reservations nor semantic rejection memory.
        reservations.sort(key=lambda row: row["reservation_id"])
        rejected = {tier: rows[-30:] for tier, rows in rejected.items()}
        ids = sorted(str(row["candidate_id"]) for row in supply["kernels"])
        if len(set(ids)) != len(supply["kernels"]):
            raise FactorySupplyError("selector window repeats candidate IDs")
        Random(int(self.config["seed"]) + slate * 180_169).shuffle(ids)
        source = {str(row["candidate_id"]): row for row in supply["kernels"]}
        conflicts = {key: "exact normalized mechanism is already accepted, reserved, or completed in this campaign" for key in ids if concept_hash(source[key]["kernel"]["central_relation"]) in exact_hashes}
        # Strict controls still abort on any exact conflict. The explicitly
        # smaller working window excludes only those conflicts, not siblings
        # selected earlier/later by model-call completion order.
        if minimum < 48:
            ids = [key for key in ids if key not in conflicts]
        selection_eligible = len(ids) >= minimum and (minimum < 48 or not conflicts)
        if selection_eligible and self.embedder is None:
            self.embedder = build_embedding_provider(context["embedding"], str(self.config["_paths"]["cache"]))
        packages = {}
        for key in ids:
            kernel = deepcopy(source[key]["kernel"])
            hits = [] if not selection_eligible else retrieve_union(kernel["central_relation"], context["references"], self.embedder, top_k=3, text_getter=lambda item: item["central_relation"], id_getter=lambda item: item["reference_id"])
            packages[key] = {"kernel": kernel, "nearest_references": [{"reference_id": hit.reference_id, "source": hit.reference["source"], "archive_tier": hit.reference["archive_tier"], "summary": hit.reference["summary"][:700], "similarity": round(hit.score, 6)} for hit in hits], "staleness": {"status": "current", "reasons": []}}
        payload = {
            "accepted_bank_digest": context["accepted_bank_digest"],
            "in_flight_reservations": reservations, "rejected_memory_digest": rejected,
            "packages": packages, "proposal_ids": ids,
            "shards": [ids[index * len(ids) // 4:(index + 1) * len(ids) // 4] for index in range(4)],
        }
        value = {"schema_version": 1, "input_identity": identity, "payload": payload, "payload_hash": content_hash(payload), "source_labels_hidden_from_selector": True, "exact_conflicts": conflicts,
                 "selection_eligible": selection_eligible, "minimum_idea_kernels": minimum}
        value["snapshot_hash"] = content_hash(value)
        legacy._immutable(path, value, "source-blind selector inputs")
        return value

    def _select(self, inputs: Mapping[str, Any], *, purpose: str, call_id: str, selector_id: str, packages: Sequence[Mapping[str, Any]], maximum: int, design: Mapping[str, Any]) -> dict[str, Any]:
        spec = design["selector_prompts"][purpose]
        template = _verify_artifact(spec, self.root).read_text(encoding="utf-8")
        payload = inputs["payload"]
        prompt = render_prompt(template, {
            "CANDIDATES_WITH_REFERENCES": list(packages), "ACCEPTED_BANK_DIGEST": payload["accepted_bank_digest"],
            "IN_FLIGHT_RESERVATIONS": payload["in_flight_reservations"], "REJECTED_MEMORY_DIGEST": payload["rejected_memory_digest"],
            "MAX_SELECTED": maximum, "SELECTOR_ID": selector_id,
        })
        ids = {str(item["kernel"]["candidate_id"]) for item in packages}
        stage = "v4_shard_selector" if purpose == "shard_selection" else "v4_global_selector"
        def validate(value: dict[str, Any]) -> None:
            parse_factory_selector(stage, value)
            assessed = {item["candidate_id"] for item in value["candidate_assessments"]}
            if value.get("selector_id") != selector_id or assessed != ids or len(value["selected"]) > maximum:
                raise FactorySupplyError("selector changed identity, candidate coverage, or capacity")
            if sum(row.get("optional_development_suggestion") is not None for row in value["selected"]) > 2:
                raise FactorySupplyError("selector exceeded the optional development suggestion cap")
        return self._invoke(purpose=purpose, stage=stage, call_id=call_id, prompt=prompt, version=str(spec["version"]), digest=str(spec["template_hash"]), parent_ids=sorted(ids), validator=validate)

    def select_slate(self, supply: Mapping[str, Any], slate: int) -> dict[str, Any]:
        design = self.freeze_supply_design()
        directory = self.root / "selection_slates" / f"slate-{slate:02d}"
        checkpoint = directory / "selection_checkpoint.private.json"
        identity = content_hash({"factory": design["factory_identity"], "supply": supply})
        if checkpoint.is_file():
            value = read_json(checkpoint)
            if value.get("input_identity") != identity or value.get("checkpoint_hash") != content_hash({key: row for key, row in value.items() if key != "checkpoint_hash"}):
                raise FactorySupplyError("resumed selected slate changed its frozen supply")
            self._reserve_checkpoint(value, supply, slate)
            self._write_selection_views(value, directory)
            return value
        frozen = self._selector_inputs(supply, slate, design)
        packages = frozen["payload"]["packages"]
        def shard(spec: tuple[int, list[str]]) -> dict[str, Any]:
            index, ids = spec
            return self._select(frozen, purpose="shard_selection", call_id=f"factory-s{slate:02d}-shard-{index + 1:02d}", selector_id=f"shard-selector-{index + 1:02d}", packages=[packages[key] for key in ids], maximum=5, design=design)
        if not frozen.get("selection_eligible", not frozen["exact_conflicts"]):
            shards = []
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                shards = list(executor.map(shard, enumerate(frozen["payload"]["shards"])))
        finalists = []
        for ids, selection in zip(frozen["payload"]["shards"] if shards else [], shards, strict=True):
            assessments = {row["candidate_id"]: row for row in selection["candidate_assessments"]}
            briefs = {row["candidate_id"]: row for row in selection["selected"]}
            finalists.extend({"kernel": packages[key]["kernel"], "nearest_references": packages[key]["nearest_references"], "shard_assessment": assessments[key], "shard_brief": briefs[key]} for key in ids if key in briefs)
        global_result = self._select(frozen, purpose="global_selection", call_id=f"factory-s{slate:02d}-global-01", selector_id="global-selector-01", packages=finalists, maximum=16, design=design) if finalists else {
            "selector_id": "global-selector-01", "candidate_assessments": [],
            "within_batch_duplicate_groups": [], "selected": [],
            "portfolio_observations": "No shard finalist qualified for global comparison.",
        }
        private = {str(row["candidate_id"]): row for row in supply["kernels"]}
        assessments = {row["candidate_id"]: row for row in global_result["candidate_assessments"]}
        frontier = []
        for brief in sorted(global_result["selected"], key=lambda row: int(row["development_priority"])):
            key = str(brief["candidate_id"])
            row, assessment = private[key], assessments[key]
            frontier.append({
                "source_candidate_id": key, "lineage_id": content_hash({"e18_source": key}), "slate_index": slate,
                "source_development_priority": int(brief["development_priority"]), "search_mode": row["kernel"]["search_mode"],
                "kernel": deepcopy(row["kernel"]), "editorial_brief": deepcopy(brief),
                "private_source": row["private_source"], "private_conditioned": row["private_conditioned"],
                "private_source_call_id": row["private_source_call_id"], "private_effort": row["private_effort"],
                "selector_assessment": deepcopy(assessment),
                "novelty": {"selector_archive_checked": True, "confirmed_same_procedure_duplicate": False, "within_window_sibling_ids": list(assessment.get("same_minimal_procedure_with") or []), "selected_as_sibling_representative": bool(assessment.get("same_minimal_procedure_with"))},
            })
        eligible, dropped = apply_source_ceilings(frontier, self.config["selection"])
        routine, deferred = eligible[:12], eligible[12:16]
        # The invitation-free floor also applies to the routine development
        # set when the global frontier includes ordinary deferred candidates.
        routine, routine_drops = apply_source_ceilings(routine, self.config["selection"])
        dropped.extend(routine_drops)
        value = {
            "schema_version": 1, "slate_index": slate, "input_identity": identity,
            "selection_completed": frozen.get("selection_eligible", not frozen["exact_conflicts"]), "terminal_status": "insufficient_eligible_window" if not frozen.get("selection_eligible", not frozen["exact_conflicts"]) else "selected_slate_frozen" if routine else "no_development_selected",
            "planning_started": False, "source_labels_hidden_from_selector": True,
            "empty_selection_reason": ("exact_conflicts_leave_fewer_than_required_eligible_candidates" if not frozen.get("selection_eligible", not frozen["exact_conflicts"]) else "no_shard_finalists" if not finalists else "no_eligible_qualified_candidates") if not routine else None,
            "exact_conflicts": deepcopy(frozen["exact_conflicts"]),
            "whole_development_slate_reserved_before_planning": True,
            "full_selector_selected_count": len(frontier), "source_ceiling_drop_ids": sorted(dropped),
            "development_count": len(routine), "deferred_count": len(deferred), "deferred_status": DEFERRED,
            "source_rows": routine, "deferred_rows": deferred,
            "specialized_selected_counts": dict(Counter(row["private_source"] for row in routine if row["private_source"] in SPECIAL_SOURCES)),
            "frozen_window": {"proposal_ids": frozen["payload"]["proposal_ids"], "shards": frozen["payload"]["shards"], "input_snapshot_path": str((directory / "selector_inputs.blinded.json").resolve()), "input_snapshot_sha256": _sha(directory / "selector_inputs.blinded.json")},
            "shard_selectors": shards, "global_selector": global_result,
        }
        self._reserve_checkpoint(value, supply, slate)
        value["checkpoint_hash"] = content_hash(value)
        legacy._immutable(checkpoint, value, "factory selected slate")
        self._write_selection_views(value, directory)
        return value

    @staticmethod
    def _write_selection_views(value: Mapping[str, Any], directory: Path) -> None:
        legacy._immutable(directory / "selected_slate.blinded.json", {"schema_version": 1, "slate_index": value["slate_index"], "private_labels_removed": True, "candidates": [{"candidate_id": row["source_candidate_id"], "kernel": row["kernel"], "editorial_brief": row["editorial_brief"]} for row in value["source_rows"]]}, "blinded selected slate")
        legacy._immutable(directory / "qualified_deferred.private.json", {"schema_version": 1, "status": DEFERRED, "candidates": value["deferred_rows"], "reserved": False, "planning_started": False}, "qualified deferred slate")

    def _reserve_checkpoint(self, value: Mapping[str, Any], supply: Mapping[str, Any], slate: int) -> None:
        snapshot = value["frozen_window"]
        _verify_artifact({"path": snapshot["input_snapshot_path"], "sha256": snapshot["input_snapshot_sha256"]}, self.root)
        if value["terminal_status"] == "insufficient_eligible_window":
            self.ledger.terminalize_ineligible_slate(slate, reason=str(value["empty_selection_reason"]))
            return
        selected = [str(row["source_candidate_id"]) for row in value["source_rows"]]
        deferred = [str(row["source_candidate_id"]) for row in value["deferred_rows"]]
        self.ledger.reserve_selected(slate=slate, selected_ids=selected, deferred_ids=deferred, rejected_ids=[str(row["candidate_id"]) for row in supply["kernels"] if row["candidate_id"] not in {*selected, *deferred}])
        if value.get("exact_conflicts"):
            with self.ledger.lock:
                for key, reason in value["exact_conflicts"].items():
                    row = self.ledger.value["proposals"][key]
                    if row["state"] != "rejected":
                        raise FactorySupplyError("exact conflict escaped rejection")
                    row.update(rejection_tier="hard_duplicate", terminal_reason=reason)
                self.ledger.save()


__all__ = ["FactorySupply", "FactorySupplyError", "ReservoirLedger", "DEFERRED", "IDEA_PURPOSES", "SELECTOR_SCHEMAS", "apply_source_ceilings", "freeze_supply_inputs", "parse_factory_selector", "render_idea_prompt"]
