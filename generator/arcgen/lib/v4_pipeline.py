"""ARCGEN V4 control: persistent supply and candidate-level authoring graph.

The module is deliberately separate from every historical pipeline.  It uses
the saved V3.1.2 authoring policy as its parent, but V4 state, prompts, schemas,
reservoir entries, calls, and reports all live in a disjoint namespace.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any, Callable, Mapping, Sequence
import json
import threading
import time

from .committed_tasks import (
    assert_generated_references_are_committed,
    committed_generated_novelty_references,
    load_committed_task_registry,
)
from .data import TaskRecord, load_task_bank
from .embeddings import build_embedding_provider
from .lean_pipeline import (
    assert_no_fixed_anchor_leakage,
    serious_blind_alternative,
    select_diverse,
)
from .lean_v3_pipeline import classify_v3_editorial_results
from .model import CallResult, ModelInvoker
from .retrieval import retrieve_union
from .util import (
    PLACEHOLDER_RE,
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    parallel_map,
    read_json,
    render_prompt,
)
from .validation import validate_arc_task
from .v4_config import (
    CONTROL_PIPELINE_ID,
    NORMAL_STAGE_LIMITS,
    SEARCH_MODES,
    V4_PIPELINE_VERSION,
    snapshot_v4_config,
    verify_v4_parent,
)
from .v4_diagnostics import (
    check_plan_fidelity,
    complete_task,
    diagnose_arc_task_v4,
    diagnose_public_arc_task_v4,
    hidden_outputs_match,
)
from .v4_reservoir import (
    V4ReservoirError,
    append_supply,
    concept_hash,
    freeze_eligible_window,
    heartbeat_leases,
    load_reservoir,
    make_proposal,
    mark_selection_decisions,
    memory_digest,
    recover_interrupted_slate,
    replenishment_decision,
    reservation_digest,
    reserve_slate,
    reservoir_status,
    resolve_proposal,
)
from .v4_schemas import (
    OUTPUT_SCHEMAS,
    parse_stage_output,
)


PROMPT_FILES = {
    "idea_open": "v4_idea_open.md",
    "idea_concept": "v4_idea_concept.md",
    "idea_composition": "v4_idea_composition.md",
    "shard_selector": "v4_shard_selector.md",
    "global_selector": "v4_global_selector.md",
    "concrete_planner": "v4_concrete_planner.md",
    "task_realizer": "v4_task_realizer.md",
    "technical_verifier": "v4_technical_verifier.md",
    "blind_solver": "v4_blind_solver.md",
    "ambiguity_adjudicator": "v4_ambiguity_adjudicator.md",
    "targeted_repair": "v4_targeted_repair.md",
    "editorial_board": "v4_editorial_board.md",
    "plan_critic": "v4_plan_critic_shadow.md",
    "plan_revision": "v4_plan_revision.md",
    "answer_aware": "v4_answer_aware.md",
    "preliminary_curator": "v4_preliminary_curator.md",
}
PROMPT_TO_STAGE = {
    "idea_open": "v4_idea_search",
    "idea_concept": "v4_idea_search",
    "idea_composition": "v4_idea_search",
    "shard_selector": "v4_shard_selector",
    "global_selector": "v4_global_selector",
    "concrete_planner": "v4_concrete_plan",
    "task_realizer": "v4_task_realizer",
    "technical_verifier": "v4_technical_verifier",
    "blind_solver": "v4_blind_solver",
    "ambiguity_adjudicator": "v4_ambiguity_adjudicator",
    "targeted_repair": "v4_targeted_repair",
    "editorial_board": "v4_editorial_board",
    "plan_critic": "v4_plan_critic",
    "plan_revision": "v4_plan_revision",
    "answer_aware": "v4_answer_aware",
    "preliminary_curator": "v4_preliminary_curation",
}
ROLE_FOR_PROMPT = {
    "shard_selector": "batch_selector",
    "global_selector": "batch_selector",
    "concrete_planner": "concrete_planner",
    "task_realizer": "task_realizer",
    "technical_verifier": "technical_verifier",
    "blind_solver": "blind_solver",
    "ambiguity_adjudicator": "ambiguity_adjudicator",
    "targeted_repair": "targeted_repair",
    "editorial_board": "editorial_board",
}


class V4PipelineError(ValueError):
    pass


def v4_call_accounting(
    ledger: Mapping[str, Any], *, run_dir: str | Path | None = None
) -> dict[str, Any]:
    """Account for every logical call and every persisted model attempt.

    Failed substantive calls still consume budget and model time.  Attempt
    logs are authoritative because a successful retry's ``result.json`` only
    describes its final attempt.
    """

    logical = list(ledger.values())
    completed = [item for item in logical if item.get("status") == "complete"]
    failed = [item for item in logical if item.get("status") == "failed"]
    normal = [item for item in logical if item.get("budget_kind") == "normal"]
    escalation = [item for item in logical if item.get("budget_kind") == "escalation"]
    attempt_timings: list[dict[str, Any]] = []
    attempt_counts: dict[str, int] = {}
    root = Path(run_dir) if run_dir is not None else None
    for item in logical:
        rows: list[dict[str, Any]] = []
        candidates: list[Path] = []
        if isinstance(item.get("log_dir"), str):
            candidates.append(Path(item["log_dir"]))
        if root is not None and item.get("stage") and item.get("call_id"):
            candidates.append(root / "model_calls" / str(item["stage"]) / str(item["call_id"]))
        seen: set[Path] = set()
        for directory in candidates:
            try:
                resolved = directory.resolve()
            except OSError:
                resolved = directory
            if resolved in seen or not directory.is_dir():
                continue
            seen.add(resolved)
            for path in sorted(directory.glob("*attempt-*.timing.json")):
                try:
                    timing = read_json(path)
                except Exception:
                    continue
                if isinstance(timing, dict):
                    rows.append({**timing, "timing_path": str(path)})
        if not rows and isinstance(item.get("timing"), dict):
            rows.append(dict(item["timing"]))
        attempt_timings.extend(rows)
        attempt_counts[str(item.get("call_id"))] = len(rows)
    model_seconds = sum(
        float(item.get("model_execution_seconds") or 0)
        for item in attempt_timings
    )
    queue_seconds = sum(
        float(item.get("queue_wait_seconds") or 0)
        for item in attempt_timings
    )
    retry_calls = sum(max(0, count - 1) for count in attempt_counts.values())
    intervals = []
    for timing in attempt_timings:
        start = timing.get("slot_acquired_at_epoch_seconds")
        end = timing.get("model_finished_at_epoch_seconds") or timing.get("completed_at_epoch_seconds")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            intervals.append((float(start), float(end)))
    events = sorted(
        [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals],
        key=lambda item: (item[0], item[1]),
    )
    active = 0
    peak = 0
    area = 0.0
    previous = events[0][0] if events else 0.0
    for moment, delta in events:
        area += active * (moment - previous)
        active += delta
        peak = max(peak, active)
        previous = moment
    span = (
        max(end for _, end in intervals) - min(start for start, _ in intervals)
        if intervals else 0.0
    )
    return {
        "substantive_logical_calls": len(logical),
        "completed_substantive_calls": len(completed),
        "failed_substantive_calls": len(failed),
        "incomplete_substantive_calls": len(logical) - len(completed) - len(failed),
        "normal_calls": len(normal),
        "completed_normal_calls": sum(item.get("status") == "complete" for item in normal),
        "escalation_calls": len(escalation),
        "completed_escalation_calls": sum(item.get("status") == "complete" for item in escalation),
        "transport_retry_calls": retry_calls,
        "recorded_model_attempts": len(attempt_timings),
        "failed_or_malformed_attempts": sum(
            item.get("outcome") in {"error", "timeout"}
            for item in attempt_timings
        ),
        "attempt_counts_by_call": dict(sorted(attempt_counts.items())),
        "cache_hits": sum(item.get("cache_source") is not None for item in completed),
        "calls_by_role": dict(sorted(Counter(item["role"] for item in logical).items())),
        "completed_calls_by_role": dict(
            sorted(Counter(item["role"] for item in completed).items())
        ),
        "active_model_seconds": round(model_seconds, 3),
        "active_model_minutes": round(model_seconds / 60, 3),
        "model_slot_queue_wait_seconds": round(queue_seconds, 3),
        "measured_peak_concurrent_calls": peak,
        "measured_effective_concurrency": round(area / span, 3) if span else 0.0,
        "timed_call_count": len(intervals),
    }


TERMINAL_REJECTION_MEMORY = {
    "aborted_plan": "failure_mode",
    "authored_reserve": "soft_mechanism",
    "recoverable": "recoverable_work",
    "rejected_realization": "failure_mode",
    "rejected_technical": "failure_mode",
    "ambiguous": "failure_mode",
    "rejected_quality": "soft_mechanism",
    "portfolio_duplicate": "hard_duplicate",
    "portfolio_hold": "soft_mechanism",
    "rejected_by_user": "soft_mechanism",
}
TERMINAL_CANDIDATE_STATUSES = set(TERMINAL_REJECTION_MEMORY) | {
    "awaiting_explicit_commit",
    "accepted",
    "core_quality",
    "reserve_quality",
}


class V4PromptStore:
    def __init__(self, prompt_dir: Path, snapshot_dir: Path, *, pipeline: str) -> None:
        self.entries: dict[str, dict[str, str]] = {}
        destination = snapshot_dir / "prompts" / "templates"
        destination.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {"pipeline": pipeline, "templates": {}}
        for name, filename in PROMPT_FILES.items():
            path = prompt_dir / filename
            if not path.is_file():
                raise V4PipelineError(f"missing V4 prompt: {path}")
            text, version, digest = load_prompt(path)
            self.entries[name] = {
                "text": text,
                "version": version,
                "hash": digest,
                "filename": filename,
            }
            target = destination / filename
            if target.exists() and target.read_text(encoding="utf-8") != text:
                raise V4PipelineError(f"prompt changed inside existing V4 run: {filename}")
            if not target.exists():
                atomic_write_text(target, text)
            manifest["templates"][name] = {
                "filename": filename,
                "version": version,
                "sha256": digest,
                "production_control": name not in {
                    "plan_critic", "plan_revision", "answer_aware", "preliminary_curator"
                },
            }
        path = snapshot_dir / "prompts" / "manifest.json"
        if path.exists() and read_json(path) != manifest:
            raise V4PipelineError("V4 prompt manifest changed inside an existing run")
        if not path.exists():
            atomic_write_json(path, manifest)

    def render(self, name: str, values: Mapping[str, Any]) -> tuple[str, str, str]:
        entry = self.entries[name]
        required = set(PLACEHOLDER_RE.findall(entry["text"]))
        if required != set(values):
            raise V4PipelineError(
                f"{name} prompt bindings differ: missing={sorted(required-set(values))}, "
                f"extra={sorted(set(values)-required)}"
            )
        return render_prompt(entry["text"], dict(values)), entry["version"], entry["hash"]

    def idea_hashes(self) -> dict[str, str]:
        return {
            "open": self.entries["idea_open"]["hash"],
            "concept_led": self.entries["idea_concept"]["hash"],
            "composition_led": self.entries["idea_composition"]["hash"],
        }


def _implementation_value(repository_root: Path) -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    names = {
        "v4_config.py", "v4_schemas.py", "v4_reservoir.py", "v4_diagnostics.py",
        "v4_pipeline.py", "v4_reporting.py", "v4_experiments.py", "model.py",
        "embeddings.py", "retrieval.py", "validation.py", "data.py", "util.py",
    }
    files = {
        name: content_hash((package / name).read_bytes())
        for name in sorted(names)
        if (package / name).is_file()
    }
    parent = verify_v4_parent(repository_root, repository_root / "config/v4_parent_manifest.json")
    value: dict[str, Any] = {
        "pipeline_version": V4_PIPELINE_VERSION,
        "branch_source": "saved arcgen-lean-authoring-v3.1.2",
        "frozen_parent": parent,
        "files": files,
    }
    value["implementation_hash"] = content_hash(value)
    return value


def _implementation_manifest(run_dir: Path, repository_root: Path) -> dict[str, Any]:
    value = _implementation_value(repository_root)
    path = run_dir / "implementation_manifest.json"
    if path.exists() and read_json(path) != value:
        raise V4PipelineError("V4 implementation changed inside an existing run")
    if not path.exists():
        atomic_write_json(path, value)
    return value


class _DataContext:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.seed = int(config["seed"])
        self.bank = load_task_bank(config["paths"]["annotations"], config["paths"]["tasks"])
        self.by_task_id = {item.task_id: item for item in self.bank}
        self.anchor_manifest = read_json(Path(config["paths"]["fixed_anchor_manifest"]))
        if self.anchor_manifest.get("panel_id") != "lean-authoring-v1-provisional-fixed-six":
            raise V4PipelineError("V4 must retain the fixed V1 anchor panel")
        ids = [item.get("task_id") for item in self.anchor_manifest.get("anchors", [])]
        if len(ids) != 6 or len(set(ids)) != 6 or not set(ids) <= set(self.by_task_id):
            raise V4PipelineError("fixed anchor manifest must contain six valid unique tasks")
        self.anchor_ids = set(ids)
        self.anchors = tuple(self.by_task_id[item] for item in ids)
        available = sorted(
            (item for item in self.bank if item.task_id not in self.anchor_ids),
            key=lambda item: item.task_id,
        )
        Random(self.seed).shuffle(available)
        self.generator_pool = tuple(
            available[: int(config["reservoir"]["generator_visible_pool_count"])]
        )
        for item in self.bank:
            result = validate_arc_task({"train": list(item.train), "test": list(item.test)})
            if not result.valid:
                raise V4PipelineError(f"source task {item.task_id} is invalid: {result.errors}")
        self.registry = load_committed_task_registry(config["paths"]["committed_task_registry"])
        committed = committed_generated_novelty_references(self.registry)
        assert_generated_references_are_committed(committed, self.registry)
        self.committed_references = committed
        self.bank_snapshot_id = content_hash(
            {
                "real_bank": [item.to_dict() for item in self.bank],
                "committed": [
                    {key: value for key, value in item.items() if key != "absolute_task_path"}
                    for item in self.registry["entries"]
                ],
            }
        )
        self.references = [
            {
                "reference_id": f"real:{item.task_id}",
                "source": "real_task",
                "archive_tier": "real_task",
                "central_relation": item.rule_description,
                "summary": item.rule_description,
            }
            for item in self.bank
            if item.task_id not in self.anchor_ids
        ] + [
            {
                "reference_id": item["reference_id"],
                "source": "accepted_generated_task",
                "archive_tier": "hard_duplicate_memory",
                "central_relation": item["combined_text"],
                "summary": item["rule_description"],
            }
            for item in committed
        ]

    def accepted_digest(self) -> dict[str, Any]:
        return {
            "bank_snapshot_id": self.bank_snapshot_id,
            "committed_generated_task_count": len(self.registry["entries"]),
            "concepts": [
                {
                    "registry_task_id": item["registry_task_id"],
                    "title": item["title"],
                    "concept_summary": item["concept_summary"],
                }
                for item in self.registry["entries"]
            ],
            "policy": "Only explicitly committed completed generated tasks are accepted-bank novelty memory.",
        }

    def accepted_concept_hashes(self) -> set[str]:
        result: set[str] = set()
        for item in self.registry["entries"]:
            result.add(concept_hash(item["concept_summary"]))
            result.add(concept_hash(item["intended_rule"]))
        return result


class _CallRuntime:
    def __init__(
        self,
        config: dict[str, Any],
        run_dir: Path,
        prompts: V4PromptStore,
        state: dict[str, Any],
        state_path: Path,
        state_lock: threading.RLock,
        *,
        budgeted: bool,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.prompts = prompts
        self.state = state
        self.state_path = state_path
        self.state_lock = state_lock
        self.budgeted = budgeted
        self.invoker = ModelInvoker(
            config,
            run_dir,
            output_schemas=OUTPUT_SCHEMAS,
            output_parser=parse_stage_output,
        )

    def save(self) -> None:
        with self.state_lock:
            atomic_write_json(self.state_path, self.state)

    def role_settings(self, role: str) -> dict[str, Any]:
        setting = self.config["model"]["roles"][role]
        return {
            "model_name": setting["model"],
            "reasoning_effort": setting["reasoning_effort"],
            "timeout_seconds": setting.get("timeout_seconds"),
            "retries": setting.get("retries"),
        }

    def _reserve_call(
        self,
        *,
        role: str,
        stage: str,
        call_id: str,
        budget_kind: str,
        arm_id: str | None,
        parent_ids: Sequence[str],
    ) -> str:
        key = f"{budget_kind}/{stage}/{call_id}"
        with self.state_lock:
            ledger = self.state.setdefault("call_ledger", {})
            if key in ledger:
                old = ledger[key]
                if old["role"] != role or old.get("arm_id") != arm_id:
                    raise V4PipelineError(f"call identity changed during resume: {key}")
                for parent_id in parent_ids:
                    record = self.state.get("records", {}).get(parent_id)
                    if record is None:
                        continue
                    active = record.setdefault("active_call_ids", [])
                    if key not in active:
                        active.append(key)
                self.save()
                return key
            if self.budgeted:
                normal = sum(item["budget_kind"] == "normal" for item in ledger.values())
                escalation = sum(item["budget_kind"] == "escalation" for item in ledger.values())
                if budget_kind == "normal":
                    role_count = sum(
                        item["budget_kind"] == "normal" and item["role"] == role
                        for item in ledger.values()
                    )
                    if role not in NORMAL_STAGE_LIMITS or role_count >= NORMAL_STAGE_LIMITS[role]:
                        raise V4PipelineError(f"normal V4 call limit for {role} would be exceeded")
                    if normal >= int(self.config["budget"]["normal_control_maximum"]):
                        raise V4PipelineError("V4 normal substantive-call budget exhausted")
                elif budget_kind == "escalation":
                    if escalation >= int(self.config["budget"]["escalation_call_limit"]):
                        raise V4PipelineError("V4 conditional-escalation budget exhausted")
                else:
                    raise V4PipelineError("control calls must be normal or escalation")
                if normal + escalation >= int(self.config["budget"]["hard_control_maximum"]):
                    raise V4PipelineError("V4 hard substantive-call budget exhausted")
            ledger[key] = {
                "role": role,
                "stage": stage,
                "call_id": call_id,
                "budget_kind": budget_kind,
                "arm_id": arm_id,
                "status": "reserved",
                "reserved_at_epoch_seconds": round(time.time(), 3),
                "attempts": 0,
                "retry_calls": 0,
                "cache_source": None,
            }
            for parent_id in parent_ids:
                record = self.state.get("records", {}).get(parent_id)
                if record is None:
                    continue
                active = record.setdefault("active_call_ids", [])
                if key not in active:
                    active.append(key)
            self.save()
        return key

    def _finish_parent_call_tracking(
        self, key: str, parent_ids: Sequence[str], *, succeeded: bool
    ) -> None:
        for parent_id in parent_ids:
            record = self.state.get("records", {}).get(parent_id)
            if record is None:
                continue
            record.setdefault("active_call_ids", [])
            record["active_call_ids"] = [item for item in record["active_call_ids"] if item != key]
            destination = "completed_call_ids" if succeeded else "failed_call_ids"
            values = record.setdefault(destination, [])
            if key not in values:
                values.append(key)

    def remaining_escalation(self) -> int:
        with self.state_lock:
            used = sum(
                item["budget_kind"] == "escalation"
                for item in self.state.get("call_ledger", {}).values()
            )
        return int(self.config["budget"]["escalation_call_limit"]) - used

    def call(
        self,
        *,
        prompt_name: str,
        role: str,
        call_id: str,
        values: Mapping[str, Any],
        parent_ids: Sequence[str],
        budget_kind: str,
        expected: Mapping[str, Any] | None = None,
        response_validator: Callable[[dict[str, Any]], None] | None = None,
        anchors: Sequence[TaskRecord] = (),
        pre_editorial_anchor_blind: bool = False,
        model_name: str | None = None,
        reasoning_effort: str | None = None,
        arm_id: str | None = None,
    ) -> dict[str, Any]:
        prompt, version, digest = self.prompts.render(prompt_name, values)
        if pre_editorial_anchor_blind:
            assert_no_fixed_anchor_leakage(prompt, anchors)

        def validate(value: dict[str, Any]) -> None:
            for key, wanted in (expected or {}).items():
                if value.get(key) != wanted:
                    raise V4PipelineError(
                        f"{call_id} returned {key}={value.get(key)!r}; expected {wanted!r}"
                    )
            if response_validator:
                response_validator(value)

        stage = PROMPT_TO_STAGE[prompt_name]
        key = self._reserve_call(
            role=role, stage=stage, call_id=call_id,
            budget_kind=budget_kind, arm_id=arm_id, parent_ids=parent_ids,
        )
        settings = self.role_settings(role)
        dispatch = time.time()
        with self.state_lock:
            self.state["call_ledger"][key].update(
                {
                    "status": "dispatched",
                    "dispatched_at_epoch_seconds": round(dispatch, 3),
                    "model_name": model_name or settings["model_name"],
                    "reasoning_effort": reasoning_effort or settings["reasoning_effort"],
                }
            )
            self.save()
        kwargs: dict[str, Any] = {
            "stage": stage,
            "call_id": call_id,
            "prompt": prompt,
            "prompt_template_version": version,
            "prompt_template_hash": digest,
            "parent_ids": list(parent_ids),
            "response_validator": validate,
            "model_name": model_name or settings["model_name"],
            "reasoning_effort": reasoning_effort or settings["reasoning_effort"],
        }
        if settings["timeout_seconds"] is not None:
            kwargs["timeout_seconds"] = settings["timeout_seconds"]
        if settings["retries"] is not None:
            kwargs["retries"] = settings["retries"]
        try:
            result: CallResult = self.invoker.invoke(**kwargs)
        except Exception as exc:
            with self.state_lock:
                self.state["call_ledger"][key].update(
                    {"status": "failed", "error": str(exc), "finished_at_epoch_seconds": round(time.time(), 3)}
                )
                self._finish_parent_call_tracking(key, parent_ids, succeeded=False)
                self.save()
            raise
        record = read_json(result.log_dir / "result.json")
        with self.state_lock:
            self.state["call_ledger"][key].update(
                {
                    "status": "complete",
                    "finished_at_epoch_seconds": round(time.time(), 3),
                    "attempts": result.attempts,
                    "retry_calls": max(0, result.attempts - 1),
                    "cache_source": result.cache_source,
                    "cache_key": result.cache_key,
                    "log_dir": str(result.log_dir),
                    "timing": record.get("timing"),
                }
            )
            self._finish_parent_call_tracking(key, parent_ids, succeeded=True)
            self.save()
        return result.value


def _calibration_assignments(
    pool: Sequence[TaskRecord],
    *,
    count: int,
    calls: int,
    embedder: Any,
    seed: int,
) -> list[list[TaskRecord]]:
    if count * calls > len(pool):
        raise V4PipelineError("not enough generator-visible examples for disjoint supply calibration")
    remaining = list(pool)
    groups: list[list[TaskRecord]] = []
    for index in range(calls):
        selected = select_diverse(
            remaining, count, embedder, seed + index,
            text_getter=lambda item: item.rule_description,
        )
        groups.append(selected)
        used = {item.task_id for item in selected}
        remaining = [item for item in remaining if item.task_id not in used]
    return groups


class V4ReservoirProducer:
    """Generate model-authored kernels independently of development slates."""

    def __init__(self, config: dict[str, Any], *, supply_run_id: str) -> None:
        if config["pipeline"] != CONTROL_PIPELINE_ID:
            raise V4PipelineError("production reservoir supply requires v4_control")
        self.config = config
        self.context = _DataContext(config)
        self.supply_run_id = supply_run_id
        self.run_dir = Path(config["paths"]["runs"]) / "v4_reservoir_supply" / supply_run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.lock = threading.RLock()
        _implementation_manifest(self.run_dir, Path(config["_repository_root"]))
        snapshot = snapshot_v4_config(config)
        snapshot_path = self.run_dir / "config.snapshot.json"
        if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
            raise V4PipelineError("reservoir-supply configuration changed during resume")
        if not snapshot_path.exists():
            atomic_write_json(snapshot_path, snapshot)
        self.prompts = V4PromptStore(Path(config["paths"]["prompts"]), self.run_dir, pipeline="arcgen-v4-reservoir-supply")
        if self.state_path.exists():
            self.state = read_json(self.state_path)
        else:
            self.state = {
                "pipeline_version": V4_PIPELINE_VERSION,
                "supply_run_id": supply_run_id,
                "started_at_epoch_seconds": round(time.time(), 3),
                "completed_cycles": [],
                "call_ledger": {},
                "critical_path": False,
            }
            atomic_write_json(self.state_path, self.state)
        self.embedder = build_embedding_provider(config["embedding"], config["paths"]["cache"])
        self.runtime = _CallRuntime(
            config, self.run_dir, self.prompts, self.state, self.state_path, self.lock,
            budgeted=False,
        )

    def _kernel_ids(self, cycle: int, mode: str) -> list[str]:
        prefix = "".join(character if character.isalnum() else "-" for character in self.supply_run_id).strip("-")
        return [f"{prefix}-c{cycle:02d}-{mode.replace('_', '-')}-{index}" for index in range(1, 5)]

    def run(self, *, cycles: int | None = None, force_between_watermarks: bool = False) -> dict[str, Any]:
        reservoir_path = self.config["paths"]["reservoir"]
        reservoir = load_reservoir(reservoir_path)
        decision = replenishment_decision(reservoir, self.config["reservoir"])
        if decision == "pause":
            return {"supply_run_id": self.supply_run_id, "generated": 0, "reason": "high watermark reached", "reservoir": reservoir_status(reservoir, self.config["reservoir"])}
        if decision == "idle" and not force_between_watermarks:
            return {"supply_run_id": self.supply_run_id, "generated": 0, "reason": "routine supply is idle between watermarks", "reservoir": reservoir_status(reservoir, self.config["reservoir"])}
        requested = cycles or int(self.config["reservoir"]["replenishment_cycles_per_command"])
        remaining_cycles = [index for index in range(1, requested + 1) if index not in self.state["completed_cycles"]]
        assignment_groups = _calibration_assignments(
            self.context.generator_pool,
            count=int(self.config["reservoir"]["calibration_examples_per_call"]),
            calls=requested * 3,
            embedder=self.embedder,
            seed=int(self.config["seed"]) + int(content_hash(self.supply_run_id)[:8], 16),
        )
        generated_model_kernels = 0
        added_to_reservoir = 0
        mode_plan = [
            ("open", "idea_open"),
            ("concept_led", "idea_concept"),
            ("composition_led", "idea_composition"),
        ]
        for cycle in remaining_cycles:
            latest = load_reservoir(reservoir_path)
            available = int(self.config["reservoir"]["hard_capacity"]) - sum(
                item["state"] in {"proposed", "in_flight"} for item in latest["proposals"]
            )
            if available < 12:
                break

            def generate(spec: tuple[int, str, str]) -> tuple[str, dict[str, Any], list[TaskRecord]]:
                mode_index, mode, prompt_name = spec
                assignment = assignment_groups[(cycle - 1) * 3 + mode_index]
                call_id = f"cycle-{cycle:02d}-{mode.replace('_', '-')}-supply"
                ids = self._kernel_ids(cycle, mode)

                def validate(value: dict[str, Any]) -> None:
                    if [item["candidate_id"] for item in value["ideas"]] != ids:
                        raise V4PipelineError(f"{call_id} returned unassigned kernel IDs")

                result = self.runtime.call(
                    prompt_name=prompt_name,
                    role="idea_search",
                    call_id=call_id,
                    values={
                        "CALIBRATION_DESCRIPTIONS": [item.rule_description for item in assignment],
                        "SEARCH_CALL_ID": call_id,
                        "CANDIDATE_IDS": ids,
                    },
                    parent_ids=[],
                    budget_kind="reservoir",
                    expected={"search_call_id": call_id, "search_mode": mode},
                    response_validator=validate,
                    anchors=self.context.anchors,
                    pre_editorial_anchor_blind=True,
                    arm_id="production-supply",
                )
                return mode, result, assignment

            results = parallel_map(
                generate,
                [(index, mode, prompt) for index, (mode, prompt) in enumerate(mode_plan)],
                min(3, int(self.config["model"]["concurrency"])),
            )
            proposals = []
            for mode, response, assignment in results:
                entry = self.prompts.entries["idea_" + ("concept" if mode == "concept_led" else "composition" if mode == "composition_led" else "open")]
                vectors = self.embedder.embed([item["central_relation"] for item in response["ideas"]])
                for kernel, vector in zip(response["ideas"], vectors, strict=True):
                    proposals.append(
                        make_proposal(
                            kernel,
                            supply_run_id=self.supply_run_id,
                            supply_cycle_id=f"{self.supply_run_id}:cycle-{cycle:02d}",
                            calibration_task_ids=[item.task_id for item in assignment],
                            calibration_descriptions=[item.rule_description for item in assignment],
                            calibration_version=self.config["reservoir"]["calibration_version"],
                            prompt_version=entry["version"],
                            prompt_hash=entry["hash"],
                            model=self.config["model"]["roles"]["idea_search"]["model"],
                            reasoning_effort=self.config["model"]["roles"]["idea_search"]["reasoning_effort"],
                            source_bank_snapshot_id=self.context.bank_snapshot_id,
                            embedding={
                                "provider": self.config["embedding"]["provider"],
                                "model": self.config["embedding"]["model"],
                                "dimension": len(vector),
                                "vector_hash": content_hash(vector),
                            },
                        )
                    )
            before_ids = {
                item["proposal_id"] for item in load_reservoir(reservoir_path)["proposals"]
            }
            updated_reservoir = append_supply(
                reservoir_path,
                proposals,
                hard_capacity=int(self.config["reservoir"]["hard_capacity"]),
                supply_run_id=self.supply_run_id,
                supply_cycle_id=f"{self.supply_run_id}:cycle-{cycle:02d}",
            )
            after_ids = {item["proposal_id"] for item in updated_reservoir["proposals"]}
            self.state["completed_cycles"].append(cycle)
            self.runtime.save()
            generated_model_kernels += len(proposals)
            added_to_reservoir += len(after_ids - before_ids)
        self.state["completed_at_epoch_seconds"] = round(time.time(), 3)
        self.runtime.save()
        final = load_reservoir(reservoir_path)
        return {
            "supply_run_id": self.supply_run_id,
            # `generated` remains the compatibility name for newly persisted
            # proposals.  Model-authored kernels are reported separately so a
            # crash-resume replay cannot pretend to have enlarged the reservoir.
            "generated": added_to_reservoir,
            "model_authored_kernels": generated_model_kernels,
            "newly_stored_proposals": added_to_reservoir,
            "completed_cycles": self.state["completed_cycles"],
            "reservoir": reservoir_status(final, self.config["reservoir"]),
            "call_ledger_path": str(self.state_path),
        }


class V4ControlPipeline:
    """One frozen-window V4 production-control qualification slate."""

    def __init__(self, config: dict[str, Any]) -> None:
        if config["pipeline"] != CONTROL_PIPELINE_ID:
            raise V4PipelineError("V4 control runner refuses experimental configurations")
        if config["experimental"] != {
            "enabled": False, "kind": "none", "gating": False,
            "auto_promote": False, "arm_id": "control",
        }:
            raise V4PipelineError("experimental treatments are not disabled in V4 control")
        self.config = config
        self.seed = int(config["seed"])
        self.run_id = str(config["run"]["id"])
        self.run_dir = Path(config["paths"]["runs"]) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.lock = threading.RLock()
        self.context = _DataContext(config)
        self.implementation = _implementation_manifest(
            self.run_dir, Path(config["_repository_root"])
        )
        snapshot = snapshot_v4_config(config)
        config_path = self.run_dir / "config.snapshot.json"
        if config_path.exists() and read_json(config_path) != snapshot:
            raise V4PipelineError("V4 control configuration changed during resume")
        if not config_path.exists():
            atomic_write_json(config_path, snapshot)
            atomic_write_text(
                config_path.with_suffix(".yaml"),
                json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            )
        self.prompts = V4PromptStore(
            Path(config["paths"]["prompts"]), self.run_dir,
            pipeline=CONTROL_PIPELINE_ID,
        )
        self.state = self._load_or_initialize_state()
        self.runtime = _CallRuntime(
            config, self.run_dir, self.prompts, self.state, self.state_path,
            self.lock, budgeted=True,
        )
        self._reconcile_terminal_reservoir_dispositions()
        self.embedder = build_embedding_provider(config["embedding"], config["paths"]["cache"])
        self._write_immutable_inputs()

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            value = read_json(self.state_path)
            if (
                value.get("pipeline_version") != V4_PIPELINE_VERSION
                or value.get("run_id") != self.run_id
                or value.get("implementation_hash") != self.implementation["implementation_hash"]
            ):
                raise V4PipelineError("existing V4 control state is stale or incompatible")
            return value
        value = {
            "pipeline_version": V4_PIPELINE_VERSION,
            "pipeline": CONTROL_PIPELINE_ID,
            "run_id": self.run_id,
            "seed": self.seed,
            "implementation_hash": self.implementation["implementation_hash"],
            "started_at_epoch_seconds": round(time.time(), 3),
            "completed": False,
            "experimental_treatments_active": False,
            "idea_generation_on_development_critical_path": False,
            "accepted_bank_snapshot_id": self.context.bank_snapshot_id,
            "fixed_anchor_panel_id": self.context.anchor_manifest["panel_id"],
            "stage_progress": {},
            "call_ledger": {},
            "frozen_window": None,
            "window_proposals": {},
            "shard_selectors": {},
            "global_selector": None,
            "records": {},
            "candidate_events": [],
            "editorial_queue": [],
            "editorial_groups": [],
            "reservoir_generation_cost": {"included_in_control_critical_path": False},
            "automatic_commits_enabled": False,
            "implementation_limitations": [
                "The fixed six-anchor panel remains provisional.",
                "Embedding retrieval proposes comparisons but never decides novelty.",
                "Core and reserve outcomes remain in flight until explicit good-task commitment or rejection.",
                "One qualification slate cannot establish a quality or throughput improvement.",
            ],
        }
        atomic_write_json(self.state_path, value)
        return value

    def _write_immutable_inputs(self) -> None:
        data = self.run_dir / "data"
        data.mkdir(parents=True, exist_ok=True)
        values = {
            "fixed_anchor_panel.json": self.context.anchor_manifest,
            "generator_visible_pool.json": {
                "seed": self.seed,
                "task_ids": [item.task_id for item in self.context.generator_pool],
                "fixed_anchor_ids_excluded": sorted(self.context.anchor_ids),
            },
            "accepted_bank.start.json": self.context.accepted_digest(),
            "bank_manifest.json": {
                "real_task_count": len(self.context.bank),
                "committed_generated_count": len(self.context.registry["entries"]),
                "snapshot_id": self.context.bank_snapshot_id,
            },
            "information_boundaries.json": {
                "idea_producer": ["calibration descriptions", "search-mode and ARC quality prompt"],
                "selector": ["frozen proposals", "concise references", "accepted digest", "reservations", "tiered rejected memory"],
                "planner": ["selected kernel", "editorial brief"],
                "realizer": ["frozen concrete storyboard"],
                "technical_verifier": ["rule", "storyboard", "public grids", "diagnostics"],
                "blind_solver": ["public task only"],
                "editorial_board": ["complete episodes", "concise rules", "fixed anchors", "portfolio digest"],
            },
        }
        for name, value in values.items():
            path = data / name
            if path.exists() and read_json(path) != value:
                raise V4PipelineError(f"immutable V4 input changed during resume: {name}")
            if not path.exists():
                atomic_write_json(path, value)

    def _save(self) -> None:
        self.runtime.save()

    def _reconcile_terminal_reservoir_dispositions(self) -> None:
        """Finish only the reservoir half of an interrupted terminal transition.

        Run state is written before the reservoir is changed.  An abrupt stop
        between those two atomic writes therefore leaves a terminal candidate
        with a live lease.  Resume uses the durable terminal metadata to finish
        that one missing operation; it never reruns semantic work.
        """

        records = self.state.get("records", {})
        if not records:
            return
        reservoir = load_reservoir(self.config["paths"]["reservoir"])
        by_id = {item["proposal_id"]: item for item in reservoir["proposals"]}
        reconciled: list[str] = []
        for candidate_id, record in records.items():
            status = record.get("status")
            memory_tier = record.get("terminal_memory_tier") or TERMINAL_REJECTION_MEMORY.get(status)
            if memory_tier is None:
                continue
            proposal = by_id.get(candidate_id)
            if proposal is None:
                raise V4PipelineError(f"terminal candidate {candidate_id} is absent from the reservoir")
            if proposal["state"] == "rejected":
                continue
            if proposal["state"] == "accepted":
                raise V4PipelineError(
                    f"terminally rejected candidate {candidate_id} is accepted in the reservoir"
                )
            reason = record.get("final_reason") or "terminal V4 disposition recovered during resume"
            resolve_proposal(
                self.config["paths"]["reservoir"],
                proposal_id=candidate_id,
                outcome="rejected",
                reason=reason,
                source_run_id=self.run_id,
                memory_tier=memory_tier,
            )
            reconciled.append(candidate_id)
        if reconciled:
            with self.lock:
                self.state.setdefault("resume_reconciliation", []).append(
                    {
                        "at_epoch_seconds": round(time.time(), 3),
                        "reservoir_dispositions_completed": sorted(reconciled),
                    }
                )
                self.runtime.save()

    def _event(self, candidate_id: str, stage: str, event: str, **extra: Any) -> None:
        with self.lock:
            row = {
                "candidate_id": candidate_id,
                "stage": stage,
                "event": event,
                "at_epoch_seconds": round(time.time(), 3),
                **extra,
            }
            self.state["candidate_events"].append(row)
            record = self.state["records"].get(candidate_id)
            if record is not None:
                record.setdefault("events", []).append(deepcopy(row))
                if event == "started":
                    record["current_stage"] = stage
                    record.setdefault("stage_timings", {}).setdefault(stage, {})[
                        "started_at_epoch_seconds"
                    ] = row["at_epoch_seconds"]
                elif event in {"completed", "aborted", "failed", "queued"}:
                    record.setdefault("stage_timings", {}).setdefault(stage, {})[
                        "completed_at_epoch_seconds"
                    ] = row["at_epoch_seconds"]
            self._save()

    def _nearest_references(self, text: str, count: int) -> list[dict[str, Any]]:
        hits = retrieve_union(
            text, self.context.references, self.embedder, top_k=count,
            text_getter=lambda item: item["central_relation"],
            id_getter=lambda item: item["reference_id"],
        )
        return [
            {
                "reference_id": hit.reference_id,
                "source": hit.reference["source"],
                "archive_tier": hit.reference["archive_tier"],
                "summary": hit.reference["summary"][:700],
                "similarity": round(hit.score, 6),
            }
            for hit in hits
        ]

    def freeze_window(self) -> dict[str, Any]:
        if self.state["frozen_window"] is not None:
            return self.state["frozen_window"]
        window_id = f"window:{self.run_id}:01"
        frozen = freeze_eligible_window(
            self.config["paths"]["reservoir"],
            window_id=window_id,
            window_size=int(self.config["selection"]["eligible_window_size"]),
            shard_size=int(self.config["selection"]["shard_size"]),
            seed=self.seed,
            current_bank_snapshot_id=self.context.bank_snapshot_id,
            current_prompt_hashes=self.prompts.idea_hashes(),
            calibration_version=self.config["reservoir"]["calibration_version"],
            accepted_concept_hashes=self.context.accepted_concept_hashes(),
        )
        reservoir = load_reservoir(self.config["paths"]["reservoir"])
        proposals = {item["proposal_id"]: item for item in reservoir["proposals"]}
        snapshot = {proposal_id: deepcopy(proposals[proposal_id]) for proposal_id in frozen["proposal_ids"]}
        with self.lock:
            self.state["frozen_window"] = frozen
            self.state["window_proposals"] = snapshot
            self.state["reservoir_snapshot_at_selection"] = {
                "status": reservoir_status(reservoir, self.config["reservoir"]),
                "reservoir_snapshot_id": frozen["reservoir_snapshot_id"],
                "memory": memory_digest(reservoir),
                "reservations": reservation_digest(reservoir),
            }
            self.state["stage_progress"]["frozen_window"] = True
            self._save()
        atomic_write_json(self.run_dir / "data/frozen_eligible_window.json", frozen)
        atomic_write_json(self.run_dir / "data/frozen_proposals.json", snapshot)
        return frozen

    @staticmethod
    def _validate_selector(value: dict[str, Any], candidate_ids: set[str], maximum: int, suggestion_maximum: int) -> None:
        assessments = {item["candidate_id"] for item in value["candidate_assessments"]}
        selected = {item["candidate_id"] for item in value["selected"]}
        decisions = {item["candidate_id"] for item in value["candidate_assessments"] if item["decision"] == "select"}
        if assessments != candidate_ids or selected != decisions or len(selected) > maximum:
            raise V4PipelineError("selector coverage, decisions, or maximum is invalid")
        if sum(item.get("optional_development_suggestion") is not None for item in value["selected"]) > suggestion_maximum:
            raise V4PipelineError("selector exceeded the conservative optional-suggestion cap")

    def _selector_context(self) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
        reservoir = load_reservoir(self.config["paths"]["reservoir"])
        return self.context.accepted_digest(), reservation_digest(reservoir), memory_digest(reservoir)

    def _package_proposals(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        result = []
        for proposal_id in ids:
            proposal = self.state["window_proposals"][proposal_id]
            result.append(
                {
                    "kernel": proposal["kernel"],
                    "nearest_references": self._nearest_references(
                        proposal["kernel"]["central_relation"],
                        int(self.config["selection"]["references_per_candidate"]),
                    ),
                    "staleness": proposal["staleness"],
                }
            )
        return result

    def select_slate(self) -> list[dict[str, Any]]:
        if self.state["stage_progress"].get("whole_slate_reserved"):
            return sorted(
                [record["editorial_brief"] for record in self.state["records"].values()],
                key=lambda item: item["development_priority"],
            )
        window = self.freeze_window()
        accepted, reservations, rejected = self._selector_context()

        def run_shard(spec: tuple[int, list[str]]) -> tuple[int, dict[str, Any]]:
            index, ids = spec
            selector_id = f"shard-selector-{index + 1:02d}"
            value = self.runtime.call(
                prompt_name="shard_selector",
                role="batch_selector",
                call_id=selector_id,
                values={
                    "CANDIDATES_WITH_REFERENCES": self._package_proposals(ids),
                    "ACCEPTED_BANK_DIGEST": accepted,
                    "IN_FLIGHT_RESERVATIONS": reservations,
                    "REJECTED_MEMORY_DIGEST": rejected,
                    "MAX_SELECTED": int(self.config["selection"]["shard_selector_max_selected"]),
                    "SELECTOR_ID": selector_id,
                },
                parent_ids=list(ids), budget_kind="normal",
                expected={"selector_id": selector_id},
                response_validator=lambda response: self._validate_selector(
                    response, set(ids),
                    int(self.config["selection"]["shard_selector_max_selected"]),
                    int(self.config["selection"]["maximum_optional_development_suggestions"]),
                ),
                anchors=self.context.anchors,
                pre_editorial_anchor_blind=True,
            )
            mark_selection_decisions(
                self.config["paths"]["reservoir"],
                window_id=window["window_id"], selector_id=selector_id,
                assessments=value["candidate_assessments"],
            )
            return index, value

        results = parallel_map(
            run_shard, list(enumerate(window["shards"])),
            min(len(window["shards"]), int(self.config["model"]["concurrency"])),
        )
        finalists: list[dict[str, Any]] = []
        with self.lock:
            for index, selector in results:
                self.state["shard_selectors"][str(index + 1)] = selector
                assessment = {item["candidate_id"]: item for item in selector["candidate_assessments"]}
                briefs = {item["candidate_id"]: item for item in selector["selected"]}
                for proposal_id in window["shards"][index]:
                    if proposal_id in briefs:
                        finalists.append(
                            {
                                "kernel": self.state["window_proposals"][proposal_id]["kernel"],
                                "shard_assessment": assessment[proposal_id],
                                "shard_brief": briefs[proposal_id],
                                "nearest_references": self._nearest_references(
                                    self.state["window_proposals"][proposal_id]["kernel"]["central_relation"],
                                    int(self.config["selection"]["references_per_candidate"]),
                                ),
                            }
                        )
            self.state["stage_progress"]["shard_selectors"] = True
            self._save()
        finalist_ids = {item["kernel"]["candidate_id"] for item in finalists}
        selector_id = "global-selector-01"
        global_value = self.runtime.call(
            prompt_name="global_selector", role="batch_selector", call_id=selector_id,
            values={
                "CANDIDATES_WITH_REFERENCES": finalists,
                "ACCEPTED_BANK_DIGEST": accepted,
                "IN_FLIGHT_RESERVATIONS": reservations,
                "REJECTED_MEMORY_DIGEST": rejected,
                "MAX_SELECTED": int(self.config["selection"]["global_selector_max_selected"]),
                "SELECTOR_ID": selector_id,
            },
            parent_ids=sorted(finalist_ids), budget_kind="normal",
            expected={"selector_id": selector_id},
            response_validator=lambda response: self._validate_selector(
                response, finalist_ids,
                int(self.config["selection"]["global_selector_max_selected"]),
                int(self.config["selection"]["maximum_optional_development_suggestions"]),
            ),
            anchors=self.context.anchors, pre_editorial_anchor_blind=True,
        )
        mark_selection_decisions(
            self.config["paths"]["reservoir"], window_id=window["window_id"],
            selector_id=selector_id, assessments=global_value["candidate_assessments"],
        )
        briefs = sorted(global_value["selected"], key=lambda item: item["development_priority"])
        selected = [
            {
                "proposal_id": brief["candidate_id"],
                "candidate_id": brief["candidate_id"],
                "mechanism_summary": self.state["window_proposals"][brief["candidate_id"]]["kernel"]["central_relation"],
            }
            for brief in briefs
        ]
        reservation_result = reserve_slate(
            self.config["paths"]["reservoir"],
            window_id=window["window_id"], owner_run_id=self.run_id,
            slate_id=f"slate:{self.run_id}:01", selections=selected,
            lease_seconds=int(self.config["reservoir"]["lease_seconds"]),
        )
        assessment_by_id = {item["candidate_id"]: item for item in global_value["candidate_assessments"]}
        reservation_by_id = {item["proposal_id"]: item for item in reservation_result["reservations"]}
        with self.lock:
            self.state["global_selector"] = global_value
            self.state["slate_reservation"] = reservation_result
            for brief in briefs:
                candidate_id = brief["candidate_id"]
                proposal = self.state["window_proposals"][candidate_id]
                self.state["records"][candidate_id] = {
                    **deepcopy(proposal["kernel"]),
                    "proposal_id": candidate_id,
                    "editorial_brief": deepcopy(brief),
                    "selector_assessment": deepcopy(assessment_by_id[candidate_id]),
                    "development_priority": brief["development_priority"],
                    "reservation": deepcopy(reservation_by_id[candidate_id]),
                    "status": "selected_reserved",
                    "current_stage": "reserved",
                    "events": [],
                    "stage_timings": {},
                    "active_call_ids": [],
                    "completed_call_ids": [],
                    "failed_call_ids": [],
                    "rejections": [],
                    "repair_count": 0,
                }
            self.state["stage_progress"]["global_selector"] = True
            self.state["stage_progress"]["whole_slate_reserved"] = True
            self._save()
        atomic_write_json(self.run_dir / "reservoir/slate_reservations.json", reservation_result)
        return briefs

    @staticmethod
    def _route_guidance(route: str) -> str:
        return {
            "concept_led": (
                "Preserve the unusual insight and build the smallest robust episode "
                "that leaves its functional composition at least solid."
            ),
            "composition_led": (
                "Preserve one coherent procedure in which several necessary pieces "
                "interact; remove apparatus that does not change output decisions."
            ),
            "flagship": (
                "Protect both conceptual freshness and functional composition while "
                "preferring insight and legibility over machinery."
            ),
            "wildcard": (
                "Test the high-upside idea honestly; abort if exact, fair, economical "
                "grids cannot preserve both editorial floors."
            ),
        }[route]

    def _record_copy(self, candidate_id: str) -> dict[str, Any]:
        with self.lock:
            return deepcopy(self.state["records"][candidate_id])

    def _set_record(self, candidate_id: str, **values: Any) -> None:
        with self.lock:
            self.state["records"][candidate_id].update(deepcopy(values))
            self._save()

    def _append_rejection(self, candidate_id: str, stage: str, category: str, reason: str) -> None:
        with self.lock:
            self.state["records"][candidate_id]["rejections"].append(
                {"stage": stage, "category": category, "reason": reason}
            )
            self._save()

    def _resolve_terminal_rejection(
        self,
        candidate_id: str,
        *,
        status: str,
        stage: str,
        category: str,
        reason: str,
        memory_tier: str,
    ) -> None:
        self._append_rejection(candidate_id, stage, category, reason)
        self._set_record(
            candidate_id,
            status=status,
            current_stage="terminal",
            final_reason=reason,
            terminal_disposition="rejected",
            terminal_memory_tier=memory_tier,
            terminal_category=category,
            terminal_at_epoch_seconds=round(time.time(), 3),
        )
        # Resolution is deliberately after the run state is durable.  If the
        # process dies between these operations, resume observes terminal state
        # and can safely perform only the missing lease resolution.
        latest = load_reservoir(self.config["paths"]["reservoir"])
        proposal = next(item for item in latest["proposals"] if item["proposal_id"] == candidate_id)
        if proposal["state"] == "in_flight":
            resolve_proposal(
                self.config["paths"]["reservoir"],
                proposal_id=candidate_id,
                outcome="rejected",
                reason=reason,
                source_run_id=self.run_id,
                memory_tier=memory_tier,
            )
        self._event(candidate_id, stage, "failed", reason=reason, memory_tier=memory_tier)

    def _plan(self, candidate_id: str) -> bool:
        record = self._record_copy(candidate_id)
        if record["status"] not in {"selected_reserved", "planning"}:
            return record["status"] == "plan_ready"
        self._event(candidate_id, "concrete_planning", "started")
        self._set_record(candidate_id, status="planning")
        try:
            plan = self.runtime.call(
                prompt_name="concrete_planner", role="concrete_planner",
                call_id=f"{candidate_id}-concrete-plan",
                values={
                    "CANDIDATE": {
                        key: record[key]
                        for key in (
                            "candidate_id", "search_mode", "central_relation", "why_coherent",
                            "novel_concept_potential", "functional_composition_potential",
                            "why_multiple_training_examples_may_be_useful",
                            "possible_test_generalization", "main_authoring_risk",
                        )
                    },
                    "EDITORIAL_BRIEF": record["editorial_brief"],
                    "ROUTE_GUIDANCE": self._route_guidance(record["editorial_brief"]["route"]),
                    "CANDIDATE_ID": candidate_id,
                },
                parent_ids=[candidate_id], budget_kind="normal",
                expected={"candidate_id": candidate_id},
                anchors=self.context.anchors, pre_editorial_anchor_blind=True,
            )
        except Exception as exc:
            self._resolve_terminal_rejection(
                candidate_id, status="recoverable", stage="concrete_planning",
                category="infrastructure_failure", reason=f"planning call failed operationally: {exc}",
                memory_tier="recoverable_work",
            )
            return False
        self._set_record(candidate_id, concrete_plan=plan)
        atomic_write_json(self.run_dir / "plans" / f"{candidate_id}.json", plan)
        if plan["status"] == "abort":
            self._resolve_terminal_rejection(
                candidate_id, status="aborted_plan", stage="concrete_planning",
                category="semantic_plan_abort", reason=plan["abort_reason"],
                memory_tier="failure_mode",
            )
            return False
        if int(record["development_priority"]) > int(self.config["selection"]["max_realizations"]):
            reason = "ready concrete plan held below the configured realization cap"
            self._resolve_terminal_rejection(
                candidate_id, status="authored_reserve", stage="concrete_planning",
                category="realization_cap", reason=reason, memory_tier="soft_mechanism",
            )
            return False
        self._set_record(candidate_id, status="plan_ready", current_stage="realization")
        self._event(candidate_id, "concrete_planning", "completed")
        return True

    def _store_realization(
        self,
        candidate_id: str,
        plan: Mapping[str, Any],
        realization: Mapping[str, Any],
        *,
        repaired: bool = False,
    ) -> tuple[bool, str]:
        if realization["status"] != "realized":
            return False, realization["refusal_reason"]
        fidelity = check_plan_fidelity(plan, realization)
        if not fidelity["passed"]:
            return False, "; ".join(fidelity["objective_failures"])
        public = deepcopy(realization["task"])
        hidden = realization["authoring_metadata"]["expected_test_outputs"]
        public_validation = validate_arc_task(public, require_test_outputs=False)
        if not public_validation.valid:
            return False, "public ARC validation failed: " + "; ".join(public_validation.errors)
        full = complete_task(realization)
        full_validation = validate_arc_task(full, require_test_outputs=True)
        if not full_validation.valid:
            return False, "complete ARC validation failed: " + "; ".join(full_validation.errors)
        diagnostics = diagnose_arc_task_v4(full)
        public_diagnostics = diagnose_public_arc_task_v4(public)
        if not diagnostics["hard_valid"]:
            return False, "deterministic hard validation failed: " + "; ".join(diagnostics.get("errors", []))
        record = self._record_copy(candidate_id)
        if repaired:
            history = record.get("repair_history", [])
            history.append(
                {
                    "prior_plan": record.get("concrete_plan"),
                    "prior_realization": record.get("realization"),
                    "prior_diagnostics": record.get("deterministic_diagnostics"),
                }
            )
        else:
            history = record.get("repair_history", [])
        self._set_record(
            candidate_id,
            concrete_plan=deepcopy(plan),
            realization=deepcopy(realization),
            complete_task=full,
            deterministic_diagnostics=diagnostics,
            technical_diagnostics=public_diagnostics,
            plan_fidelity=fidelity,
            repair_history=history,
            hidden_output_hash=content_hash(hidden),
        )
        task_dir = self.run_dir / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(task_dir / f"{candidate_id}.arc.json", full)
        atomic_write_json(
            task_dir / f"{candidate_id}.authoring.json",
            {
                **deepcopy(realization),
                "concrete_storyboard": deepcopy(plan),
                "selector_brief": record["editorial_brief"],
                "provenance": {
                    "pipeline": CONTROL_PIPELINE_ID,
                    "run_id": self.run_id,
                    "candidate_id": candidate_id,
                    "proposal_id": record["proposal_id"],
                    "search_mode": record["search_mode"],
                },
            },
        )
        return True, ""

    def _realize(self, candidate_id: str) -> bool:
        record = self._record_copy(candidate_id)
        if record["status"] not in {"plan_ready", "realizing"}:
            return record["status"] in {"realized", "technical_pass", "editorial_queued"}
        self._event(candidate_id, "realization", "started")
        self._set_record(candidate_id, status="realizing")
        try:
            realization = self.runtime.call(
                prompt_name="task_realizer", role="task_realizer",
                call_id=f"{candidate_id}-realization",
                values={"CONCRETE_PLAN": record["concrete_plan"], "CANDIDATE_ID": candidate_id},
                parent_ids=[candidate_id], budget_kind="normal",
                expected={"candidate_id": candidate_id},
                anchors=self.context.anchors, pre_editorial_anchor_blind=True,
            )
        except Exception as exc:
            self._resolve_terminal_rejection(
                candidate_id, status="recoverable", stage="realization",
                category="infrastructure_failure", reason=f"realization call failed operationally: {exc}",
                memory_tier="recoverable_work",
            )
            return False
        okay, reason = self._store_realization(candidate_id, record["concrete_plan"], realization)
        if not okay:
            # A refusal or objective fidelity defect is a concrete failed
            # episode, not a reason to close the broader concept forever.
            self._resolve_terminal_rejection(
                candidate_id, status="rejected_realization", stage="realization",
                category="plan_execution_or_fidelity", reason=reason,
                memory_tier="failure_mode",
            )
            return False
        self._set_record(candidate_id, status="realized", current_stage="technical_verification")
        self._event(candidate_id, "realization", "completed")
        return True

    def _technical_call(self, candidate_id: str, suffix: str, *, budget_kind: str) -> dict[str, Any]:
        record = self._record_copy(candidate_id)
        verifier_id = f"technical-{suffix}"
        # The technical diagnostics are intentionally computed from the public
        # task, so even aggregate test-output statistics cannot leak the hidden
        # author answer into this call.
        return self.runtime.call(
            prompt_name="technical_verifier", role="technical_verifier",
            call_id=f"{candidate_id}-{suffix}",
            values={
                "INTENDED_RULE": record["realization"]["authoring_metadata"]["intended_rule"],
                "CONCRETE_PLAN": record["concrete_plan"],
                "PUBLIC_TASK": record["realization"]["task"],
                "DIAGNOSTICS": record["technical_diagnostics"],
                "TASK_ID": candidate_id,
                "VERIFIER_ID": verifier_id,
            },
            parent_ids=[candidate_id], budget_kind=budget_kind,
            expected={"task_id": candidate_id, "verifier_id": verifier_id},
            anchors=self.context.anchors, pre_editorial_anchor_blind=True,
        )

    def _technical_evaluation(self, candidate_id: str, response: Mapping[str, Any]) -> dict[str, Any]:
        record = self._record_copy(candidate_id)
        hidden_match = hidden_outputs_match(response["derived_test_outputs"], record["realization"])
        required_codes = set(record["technical_diagnostics"].get("pair_economy_review_codes", []))
        assessed = {item["flag_code"]: item for item in response["diagnostic_flag_assessments"]}
        unresolved_flags = [
            code for code in sorted(required_codes)
            if code not in assessed or not assessed[code]["justified"]
        ]
        passed = (
            response["status"] == "pass"
            and response["mechanically_correct"]
            and response["test_supported"]
            and response["one_answer_supported"]
            and response["plan_fidelity"] != "fail"
            and response["evidence_sufficiency"] == "sufficient"
            and response["pair_economy"] != "fail"
            and response["composition_preserved"]
            and not response["arbitrary_exceptions"]
            and not response["unsupported_test_semantics"]
            and not response["unapproved_realizer_decisions"]
            and hidden_match
            and not unresolved_flags
        )
        return {
            "response": deepcopy(response),
            "hidden_output_match": hidden_match,
            "unresolved_pair_economy_flags": unresolved_flags,
            "passed": passed,
        }

    def _targeted_repair(
        self,
        candidate_id: str,
        *,
        repair_type: str,
        instruction: str,
        evidence: Any,
    ) -> bool:
        record = self._record_copy(candidate_id)
        if record.get("repair_count", 0) or self.runtime.remaining_escalation() < 1:
            return False
        try:
            response = self.runtime.call(
                prompt_name="targeted_repair", role="targeted_repair",
                call_id=f"{candidate_id}-{repair_type}-repair",
                values={
                    "CANDIDATE_ID": candidate_id,
                    "REPAIR_TYPE": repair_type,
                    "REPAIR_INSTRUCTION": instruction,
                    "AUTHOR_SPECIFICATION": {
                        "intended_rule": record["realization"]["authoring_metadata"]["intended_rule"],
                        "concrete_plan": record["concrete_plan"],
                        "realization": record["realization"],
                    },
                    "EVALUATOR_EVIDENCE": evidence,
                },
                parent_ids=[candidate_id], budget_kind="escalation",
                expected={"candidate_id": candidate_id, "repair_type": repair_type},
            )
        except Exception:
            return False
        self._set_record(candidate_id, repair_count=record.get("repair_count", 0) + 1, repair_response=response)
        if response["status"] != "repaired":
            return False
        original_task = record["realization"]["task"]
        original_hidden = record["realization"]["authoring_metadata"]["expected_test_outputs"]
        repaired = response["repaired_realization"]
        # Repair may adjust training evidence, but the test input/output and
        # purpose are frozen.  A change would be a new task, not a local repair.
        if (
            repaired["task"]["test"] != original_task["test"]
            or repaired["authoring_metadata"]["expected_test_outputs"] != original_hidden
        ):
            return False
        okay, _ = self._store_realization(
            candidate_id, response["repaired_plan"], repaired, repaired=True
        )
        return okay

    def _technical(self, candidate_id: str) -> bool:
        record = self._record_copy(candidate_id)
        if record["status"] not in {"realized", "technical_verifying"}:
            return record["status"] in {"technical_pass", "editorial_queued"}
        self._event(candidate_id, "technical_verification", "started")
        self._set_record(candidate_id, status="technical_verifying")
        try:
            primary = self._technical_call(candidate_id, "technical-primary", budget_kind="normal")
        except Exception as exc:
            self._resolve_terminal_rejection(
                candidate_id, status="recoverable", stage="technical_verification",
                category="infrastructure_failure", reason=f"technical call failed operationally: {exc}",
                memory_tier="recoverable_work",
            )
            return False
        evaluations = [self._technical_evaluation(candidate_id, primary)]
        self._set_record(candidate_id, technical_results=evaluations)
        evaluation = evaluations[-1]
        if evaluation["passed"] and primary["confidence"] != "low":
            self._set_record(candidate_id, status="technical_pass", current_stage="blind_verification")
            self._event(candidate_id, "technical_verification", "completed")
            return True

        if primary["status"] == "repairable" and primary["repair_instruction"].strip() and self.runtime.remaining_escalation() >= 2:
            if self._targeted_repair(
                candidate_id, repair_type="mechanical",
                instruction=primary["repair_instruction"], evidence=evaluation,
            ):
                try:
                    fresh = self._technical_call(candidate_id, "technical-after-repair", budget_kind="escalation")
                    evaluations.append(self._technical_evaluation(candidate_id, fresh))
                    self._set_record(candidate_id, technical_results=evaluations)
                    if evaluations[-1]["passed"] and fresh["confidence"] != "low":
                        self._set_record(candidate_id, status="technical_pass", current_stage="blind_verification")
                        self._event(candidate_id, "technical_verification", "completed", repaired=True)
                        return True
                except Exception:
                    pass
        elif (
            primary["status"] in {"uncertain", "pass"}
            or primary["confidence"] == "low"
            or not evaluation["hidden_output_match"]
        ) and self.runtime.remaining_escalation() >= 1:
            try:
                second = self._technical_call(candidate_id, "technical-second", budget_kind="escalation")
                evaluations.append(self._technical_evaluation(candidate_id, second))
                self._set_record(candidate_id, technical_results=evaluations)
                if evaluations[-1]["passed"] and second["confidence"] != "low":
                    self._set_record(candidate_id, status="technical_pass", current_stage="blind_verification")
                    self._event(candidate_id, "technical_verification", "completed", escalated=True)
                    return True
            except Exception:
                pass
        final = evaluations[-1]
        reasons = []
        if not final["hidden_output_match"]:
            reasons.append("independently derived test output mismatched the hidden author output")
        reasons.extend(final["unresolved_pair_economy_flags"])
        reasons.extend(final["response"]["arbitrary_exceptions"])
        reasons.extend(final["response"]["unsupported_test_semantics"])
        reasons.extend(final["response"]["unapproved_realizer_decisions"])
        if not reasons:
            reasons.append(final["response"].get("repair_instruction") or f"technical status {final['response']['status']} did not pass")
        self._resolve_terminal_rejection(
            candidate_id, status="rejected_technical", stage="technical_verification",
            category="invalidity_or_authorship", reason="; ".join(reasons),
            memory_tier="failure_mode",
        )
        return False

    def _blind_call(self, candidate_id: str, suffix: str, *, budget_kind: str) -> dict[str, Any]:
        record = self._record_copy(candidate_id)
        public = deepcopy(record["realization"]["task"])
        # Schema itself excludes test outputs; this assertion is deliberately
        # local so no author material can be accidentally bound to the prompt.
        if any("output" in pair for pair in public["test"]):
            raise V4PipelineError("hidden output entered the blind task")
        opaque_task_id = "blind-" + content_hash({"run": self.run_id, "candidate": candidate_id})[:12]
        solver_id = f"blind-{suffix}"
        return self.runtime.call(
            prompt_name="blind_solver", role="blind_solver",
            call_id=f"{candidate_id}-{suffix}",
            values={"PUBLIC_TASK": public, "TASK_ID": opaque_task_id, "SOLVER_ID": solver_id},
            parent_ids=[candidate_id], budget_kind=budget_kind,
            expected={"task_id": opaque_task_id, "solver_id": solver_id},
        )

    def _wrap_blind(self, candidate_id: str, response: Mapping[str, Any]) -> dict[str, Any]:
        record = self._record_copy(candidate_id)
        hidden = record["realization"]["authoring_metadata"]["expected_test_outputs"]
        return {
            "response": deepcopy(response),
            "primary_matches_hidden": response["predicted_test_outputs"] == hidden,
            "serious_different_output_alternative": serious_blind_alternative(dict(response), hidden),
        }

    def _ambiguity_adjudication(self, candidate_id: str) -> dict[str, Any]:
        record = self._record_copy(candidate_id)
        return self.runtime.call(
            prompt_name="ambiguity_adjudicator", role="ambiguity_adjudicator",
            call_id=f"{candidate_id}-ambiguity-adjudication",
            values={
                "TASK_ID": candidate_id,
                "AUTHOR_SPECIFICATION": {
                    "intended_rule": record["realization"]["authoring_metadata"]["intended_rule"],
                    "concrete_plan": record["concrete_plan"],
                },
                "COMPLETE_TASK": record["complete_task"],
                "BLIND_RESULTS": record["blind_results"],
            },
            parent_ids=[candidate_id], budget_kind="escalation",
            expected={"task_id": candidate_id},
        )

    def _ambiguity_repair_and_recheck(self, candidate_id: str, instruction: str) -> bool:
        record = self._record_copy(candidate_id)
        if record.get("repair_count", 0) or self.runtime.remaining_escalation() < 3:
            return False
        if not self._targeted_repair(
            candidate_id, repair_type="ambiguity", instruction=instruction,
            evidence=record.get("blind_results", []),
        ):
            return False
        try:
            technical = self._technical_call(candidate_id, "technical-after-ambiguity-repair", budget_kind="escalation")
            evaluation = self._technical_evaluation(candidate_id, technical)
            current = self._record_copy(candidate_id).get("technical_results", [])
            current.append(evaluation)
            self._set_record(candidate_id, technical_results=current)
            if not evaluation["passed"] or technical["confidence"] == "low":
                return False
            blind = self._blind_call(candidate_id, "blind-after-ambiguity-repair", budget_kind="escalation")
            wrapped = self._wrap_blind(candidate_id, blind)
            results = self._record_copy(candidate_id).get("blind_results", [])
            results.append(wrapped)
            self._set_record(candidate_id, blind_results=results)
            return not wrapped["serious_different_output_alternative"]
        except Exception:
            return False

    def _blind(self, candidate_id: str) -> bool:
        record = self._record_copy(candidate_id)
        if record["status"] not in {"technical_pass", "blind_verifying"}:
            return record["status"] in {"editorial_queued", "core_quality", "reserve_quality"}
        self._event(candidate_id, "blind_verification", "started")
        self._set_record(candidate_id, status="blind_verifying")
        try:
            primary = self._blind_call(candidate_id, "blind-primary", budget_kind="normal")
        except Exception as exc:
            self._resolve_terminal_rejection(
                candidate_id, status="recoverable", stage="blind_verification",
                category="infrastructure_failure", reason=f"blind call failed operationally: {exc}",
                memory_tier="recoverable_work",
            )
            return False
        results = [self._wrap_blind(candidate_id, primary)]
        self._set_record(candidate_id, blind_results=results)
        if not results[0]["serious_different_output_alternative"]:
            # A missed intended answer is evidence of difficulty, not by itself
            # evidence that the task is invalid.
            self._set_record(
                candidate_id,
                ambiguity_resolution={
                    "one_answer_supported": True,
                    "blind_primary_matches_hidden": results[0]["primary_matches_hidden"],
                    "reason": "no comparably simple, well-supported different-output rule was identified",
                },
            )
            self._event(candidate_id, "blind_verification", "completed")
            return True
        if self.runtime.remaining_escalation() < 1:
            self._resolve_terminal_rejection(
                candidate_id, status="ambiguous", stage="blind_verification",
                category="serious_different_output_alternative",
                reason="a serious blind alternative remained when the conditional budget was exhausted",
                memory_tier="failure_mode",
            )
            return False
        try:
            second_response = self._blind_call(candidate_id, "blind-second", budget_kind="escalation")
        except Exception:
            second_response = None
        if second_response is None:
            self._resolve_terminal_rejection(
                candidate_id, status="recoverable", stage="blind_verification",
                category="infrastructure_failure", reason="conditional blind call failed operationally",
                memory_tier="recoverable_work",
            )
            return False
        second = self._wrap_blind(candidate_id, second_response)
        results.append(second)
        self._set_record(candidate_id, blind_results=results)
        if second["serious_different_output_alternative"]:
            alternatives = [
                item["response"]["strongest_plausible_alternatives"] for item in results
            ]
            instruction = (
                "Adjust or replace one training pair to discriminate the intended rule "
                "from these concrete different-output alternatives without changing the "
                f"test or core task: {json.dumps(alternatives, ensure_ascii=False)}"
            )
            if self._ambiguity_repair_and_recheck(candidate_id, instruction):
                self._set_record(
                    candidate_id,
                    ambiguity_resolution={"one_answer_supported": True, "reason": "one targeted evidence repair removed the concrete alternative"},
                )
                self._event(candidate_id, "blind_verification", "completed", repaired=True)
                return True
            self._resolve_terminal_rejection(
                candidate_id, status="ambiguous", stage="blind_verification",
                category="serious_different_output_alternative",
                reason="two blind solvers identified unresolved serious different-output alternatives",
                memory_tier="failure_mode",
            )
            return False
        if self.runtime.remaining_escalation() < 1:
            self._resolve_terminal_rejection(
                candidate_id, status="ambiguous", stage="blind_verification",
                category="unresolved_blind_disagreement",
                reason="blind opinions materially disagreed and no adjudication budget remained",
                memory_tier="failure_mode",
            )
            return False
        adjudication = self._ambiguity_adjudication(candidate_id)
        self._set_record(candidate_id, ambiguity_adjudication=adjudication)
        if adjudication["decision"] == "clear":
            self._set_record(
                candidate_id,
                ambiguity_resolution={"one_answer_supported": True, "reason": adjudication["reason"]},
            )
            self._event(candidate_id, "blind_verification", "completed", adjudicated=True)
            return True
        if adjudication["decision"] == "repairable" and self._ambiguity_repair_and_recheck(
            candidate_id, adjudication["repair_instruction"]
        ):
            self._set_record(
                candidate_id,
                ambiguity_resolution={"one_answer_supported": True, "reason": "adjudicated local ambiguity was repaired"},
            )
            self._event(candidate_id, "blind_verification", "completed", repaired=True)
            return True
        self._resolve_terminal_rejection(
            candidate_id, status="ambiguous", stage="blind_verification",
            category="serious_different_output_alternative", reason=adjudication["reason"],
            memory_tier="failure_mode",
        )
        return False

    def _enqueue_editorial(self, candidate_id: str) -> None:
        with self.lock:
            record = self.state["records"][candidate_id]
            if record["status"] == "editorial_queued":
                return
            now = round(time.time(), 3)
            record["status"] = "editorial_queued"
            record["current_stage"] = "editorial_queue"
            record["editorial_enqueued_at_epoch_seconds"] = now
            if not any(item["candidate_id"] == candidate_id for item in self.state["editorial_queue"]):
                self.state["editorial_queue"].append(
                    {
                        "candidate_id": candidate_id,
                        "enqueued_at_epoch_seconds": now,
                        "status": "pending",
                        "group_id": None,
                    }
                )
            self._save()
        self._event(candidate_id, "editorial_queue", "queued")

    def _candidate_chain(self, candidate_id: str) -> dict[str, Any]:
        heartbeat_leases(self.config["paths"]["reservoir"], owner_run_id=self.run_id)
        try:
            status = self._record_copy(candidate_id)["status"]
            if status in TERMINAL_CANDIDATE_STATUSES:
                return {"candidate_id": candidate_id, "status": status}
            if status in {"selected_reserved", "planning"} and not self._plan(candidate_id):
                return {"candidate_id": candidate_id, "status": self._record_copy(candidate_id)["status"]}
            if self._record_copy(candidate_id)["status"] in {"plan_ready", "realizing"} and not self._realize(candidate_id):
                return {"candidate_id": candidate_id, "status": self._record_copy(candidate_id)["status"]}
            if self._record_copy(candidate_id)["status"] in {"realized", "technical_verifying"} and not self._technical(candidate_id):
                return {"candidate_id": candidate_id, "status": self._record_copy(candidate_id)["status"]}
            if self._record_copy(candidate_id)["status"] in {"technical_pass", "blind_verifying"} and not self._blind(candidate_id):
                return {"candidate_id": candidate_id, "status": self._record_copy(candidate_id)["status"]}
            if self._record_copy(candidate_id)["status"] not in {"editorial_queued", "core_quality", "reserve_quality"}:
                self._enqueue_editorial(candidate_id)
            return {"candidate_id": candidate_id, "status": self._record_copy(candidate_id)["status"]}
        except Exception as exc:
            # Candidate-local failures never cancel siblings.  A semantic stage
            # failure is captured more specifically above; this is operational.
            status = self._record_copy(candidate_id)["status"]
            if status not in {"recoverable", "rejected_realization", "rejected_technical", "ambiguous"}:
                self._resolve_terminal_rejection(
                    candidate_id, status="recoverable", stage="candidate_graph",
                    category="infrastructure_failure", reason=f"candidate graph failed: {exc}",
                    memory_tier="recoverable_work",
                )
            return {"candidate_id": candidate_id, "status": "recoverable", "error": str(exc)}

    def _board_package(
        self, group_id: str, board_index: int, candidate_ids: Sequence[str]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        items: list[tuple[str, str, dict[str, Any]]] = [
            (
                f"anchor:{anchor.task_id}", anchor.rule_description,
                {"train": list(anchor.train), "test": list(anchor.test)},
            )
            for anchor in self.context.anchors
        ]
        for candidate_id in candidate_ids:
            record = self._record_copy(candidate_id)
            items.append(
                (
                    f"generated:{candidate_id}",
                    record["realization"]["authoring_metadata"]["intended_rule"],
                    record["complete_task"],
                )
            )
        Random(self.seed + board_index + int(content_hash(group_id)[:8], 16)).shuffle(items)
        mapping: dict[str, str] = {}
        package: list[dict[str, Any]] = []
        for index, (source_id, rule, task) in enumerate(items, 1):
            opaque = f"{group_id}-b{board_index + 1}-{index:02d}"
            mapping[opaque] = source_id
            package.append({"task_id": opaque, "rule_summary": rule, "episode": task})
        return package, mapping

    def _review_group(self, group_id: str, candidate_ids: Sequence[str]) -> dict[str, Any]:
        with self.lock:
            existing_group = next(
                (item for item in self.state["editorial_groups"] if item["group_id"] == group_id),
                None,
            )
            if existing_group is None:
                raise V4PipelineError(f"unknown editorial group {group_id}")
            if existing_group["status"] == "complete":
                return deepcopy(existing_group)
        preparations = [self._board_package(group_id, index, candidate_ids) for index in range(2)]
        portfolio = {
            **self.context.accepted_digest(),
            "current_slate_mechanisms": [
                self._record_copy(candidate_id)["central_relation"] for candidate_id in candidate_ids
            ],
            "theme_commonality_is_descriptive_only": True,
        }

        def board(spec: tuple[int, list[dict[str, Any]], dict[str, str]]) -> dict[str, Any]:
            index, package, mapping = spec
            board_id = f"{group_id}-editorial-board-{index + 1}"
            expected_ids = {item["task_id"] for item in package}

            def validate(value: dict[str, Any]) -> None:
                ids = [item["task_id"] for item in value["judgments"]]
                if len(ids) != len(set(ids)) or set(ids) != expected_ids:
                    raise V4PipelineError("editorial board must judge every opaque task exactly once")

            response = self.runtime.call(
                prompt_name="editorial_board", role="editorial_board", call_id=board_id,
                values={"BLINDED_TASKS": package, "PORTFOLIO_DIGEST": portfolio, "BOARD_ID": board_id},
                parent_ids=list(candidate_ids), budget_kind="normal",
                expected={"board_id": board_id}, response_validator=validate,
            )
            return {"response": response, "opaque_mapping": mapping}

        boards = parallel_map(
            board,
            [(index, package, mapping) for index, (package, mapping) in enumerate(preparations)],
            min(2, int(self.config["model"]["concurrency"])),
        )
        for candidate_id in candidate_ids:
            existing = self._record_copy(candidate_id)
            if (
                existing.get("editorial_judgments")
                and existing.get("status") in TERMINAL_CANDIDATE_STATUSES
            ):
                # Resume after a board group was only partly committed.  The
                # already durable candidate decision must not be applied twice.
                continue
            judgments = []
            source = f"generated:{candidate_id}"
            for result in boards:
                opaque = next(key for key, value in result["opaque_mapping"].items() if value == source)
                judgment = next(item for item in result["response"]["judgments"] if item["task_id"] == opaque)
                judgments.append({**deepcopy(judgment), "task_id": candidate_id})
            intrinsic, placement, intrinsic_reason, placement_reason = classify_v3_editorial_results(judgments)
            final_reason = intrinsic_reason + "; " + placement_reason
            if intrinsic == "rejected_quality":
                self._set_record(
                    candidate_id, editorial_judgments=judgments,
                    intrinsic_quality=intrinsic, portfolio_placement=placement,
                )
                self._resolve_terminal_rejection(
                    candidate_id, status="rejected_quality", stage="editorial_board",
                    category="intrinsic_quality", reason=final_reason,
                    memory_tier="soft_mechanism",
                )
            elif placement == "same_procedure_duplicate":
                self._set_record(
                    candidate_id, editorial_judgments=judgments,
                    intrinsic_quality=intrinsic, portfolio_placement=placement,
                )
                self._resolve_terminal_rejection(
                    candidate_id, status="portfolio_duplicate", stage="editorial_board",
                    category="same_minimal_procedure", reason=final_reason,
                    memory_tier="hard_duplicate",
                )
            elif placement == "hold_crowded":
                self._set_record(
                    candidate_id, editorial_judgments=judgments,
                    intrinsic_quality=intrinsic, portfolio_placement=placement,
                )
                self._resolve_terminal_rejection(
                    candidate_id, status="portfolio_hold", stage="editorial_board",
                    category="portfolio_crowding", reason=final_reason,
                    memory_tier="soft_mechanism",
                )
            else:
                # No automatic bank mutation: retain the recoverable lease until
                # the user explicitly commits or rejects this good task.
                self._set_record(
                    candidate_id,
                    status="awaiting_explicit_commit",
                    current_stage="terminal_pending_user_disposition",
                    editorial_judgments=judgments,
                    intrinsic_quality=intrinsic,
                    portfolio_placement=placement,
                    final_reason=final_reason,
                    terminal_at_epoch_seconds=round(time.time(), 3),
                )
                self._event(candidate_id, "editorial_board", "completed", intrinsic=intrinsic, placement=placement)
        result = {"group_id": group_id, "candidate_ids": list(candidate_ids), "boards": boards}
        with self.lock:
            group = next(item for item in self.state["editorial_groups"] if item["group_id"] == group_id)
            group.update(
                {
                    "status": "complete",
                    "completed_at_epoch_seconds": round(time.time(), 3),
                    "boards": deepcopy(boards),
                }
            )
            for item in self.state["editorial_queue"]:
                if item["candidate_id"] in candidate_ids:
                    item["status"] = "complete"
            self._save()
        for candidate_id in candidate_ids:
            self._event(candidate_id, "editorial_review", "completed", group_id=group_id)
        return result

    def _dispatch_editorial_group(
        self, candidate_ids: Sequence[str], executor: ThreadPoolExecutor
    ) -> Future[dict[str, Any]]:
        minimum = int(self.config["editorial_queue"]["minimum_generated_tasks"])
        maximum = int(self.config["editorial_queue"]["maximum_generated_tasks"])
        if not candidate_ids or len(candidate_ids) > maximum:
            raise V4PipelineError("editorial group is empty or exceeds its configured maximum")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise V4PipelineError("editorial group contains duplicate candidates")
        with self.lock:
            group_id = f"editorial-group-{len(self.state['editorial_groups']) + 1:02d}"
            now = round(time.time(), 3)
            selected_queue_entries = [
                item for item in self.state["editorial_queue"]
                if item["candidate_id"] in candidate_ids
            ]
            oldest_wait = max(
                (now - float(item["enqueued_at_epoch_seconds"]) for item in selected_queue_entries),
                default=0.0,
            )
            if len(candidate_ids) >= minimum:
                flush_reason = "minimum comparative group reached"
            elif oldest_wait >= float(self.config["editorial_queue"]["maximum_wait_seconds"]):
                flush_reason = "maximum queue wait reached"
            else:
                flush_reason = "end-of-run tail flush"
            group = {
                "group_id": group_id,
                "candidate_ids": list(candidate_ids),
                "status": "running",
                "created_at_epoch_seconds": now,
                "flush_reason": flush_reason,
            }
            self.state["editorial_groups"].append(group)
            for item in self.state["editorial_queue"]:
                if item["candidate_id"] in candidate_ids:
                    item["status"] = "reviewing"
                    item["group_id"] = group_id
                    item["queue_wait_seconds"] = round(now - item["enqueued_at_epoch_seconds"], 3)
                    self.state["records"][item["candidate_id"]]["editorial_queue_wait_seconds"] = item[
                        "queue_wait_seconds"
                    ]
            self._save()
        for candidate_id in candidate_ids:
            self._event(candidate_id, "editorial_review", "started", group_id=group_id)
        return executor.submit(self._review_group, group_id, list(candidate_ids))

    def _resume_editorial_groups(
        self, executor: ThreadPoolExecutor
    ) -> list[Future[dict[str, Any]]]:
        with self.lock:
            groups = [
                (item["group_id"], list(item["candidate_ids"]))
                for item in self.state["editorial_groups"]
                if item["status"] == "running"
            ]
        return [executor.submit(self._review_group, group_id, candidate_ids) for group_id, candidate_ids in groups]

    def _pending_editorial_ids(self) -> list[str]:
        with self.lock:
            return [
                item["candidate_id"] for item in self.state["editorial_queue"]
                if item["status"] == "pending"
            ]

    def _next_editorial_group(self, *, force_tail: bool = False) -> list[str]:
        with self.lock:
            pending = sorted(
                (item for item in self.state["editorial_queue"] if item["status"] == "pending"),
                key=lambda item: (item["enqueued_at_epoch_seconds"], item["candidate_id"]),
            )
        if not pending:
            return []
        minimum = int(self.config["editorial_queue"]["minimum_generated_tasks"])
        maximum = int(self.config["editorial_queue"]["maximum_generated_tasks"])
        waited = time.time() - float(pending[0]["enqueued_at_epoch_seconds"])
        maximum_wait = float(self.config["editorial_queue"]["maximum_wait_seconds"])
        if len(pending) < minimum and not force_tail and waited < maximum_wait:
            return []
        return [item["candidate_id"] for item in pending[:maximum]]

    def run_candidate_graph(self) -> None:
        incomplete_groups = any(
            item.get("status") != "complete" for item in self.state.get("editorial_groups", [])
        )
        incomplete_queue = any(
            item.get("status") != "complete" for item in self.state.get("editorial_queue", [])
        )
        if self.state["stage_progress"].get("candidate_graph") and not incomplete_groups and not incomplete_queue:
            return
        selected = sorted(
            self.state["records"],
            key=lambda candidate_id: self.state["records"][candidate_id]["development_priority"],
        )
        # Candidate threads represent dependency chains, not model slots.  Use
        # one lightweight chain worker per selected candidate so a candidate
        # waiting on the invoker semaphore cannot prevent another completed
        # chain from advancing.  ModelInvoker remains the single global
        # concurrency authority.
        maximum_workers = max(1, len(selected))
        with self.lock:
            self.state["candidate_scheduler"] = {
                "candidate_chain_workers": maximum_workers,
                "global_model_call_limit": int(self.config["model"]["concurrency"]),
                "scheduling_policy": "one dependency-chain worker per selected candidate; shared model semaphore",
            }
            self._save()
        board_futures: list[Future[dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=maximum_workers) as candidate_executor, ThreadPoolExecutor(max_workers=2) as board_executor:
            board_futures.extend(self._resume_editorial_groups(board_executor))
            candidate_futures = {
                candidate_executor.submit(self._candidate_chain, candidate_id): candidate_id
                for candidate_id in selected
            }
            unfinished = set(candidate_futures)
            while unfinished:
                completed, unfinished = wait(
                    unfinished, timeout=1.0, return_when=FIRST_COMPLETED
                )
                for future in completed:
                    future.result()
                while True:
                    group = self._next_editorial_group(force_tail=False)
                    if not group:
                        break
                    board_futures.append(self._dispatch_editorial_group(group, board_executor))
            while True:
                group = self._next_editorial_group(force_tail=True)
                if not group:
                    break
                board_futures.append(self._dispatch_editorial_group(group, board_executor))
            for future in board_futures:
                future.result()
        with self.lock:
            self.state["stage_progress"]["candidate_graph"] = True
            self.state["stage_progress"]["editorial_queue_flushed"] = True
            self._save()

    def _call_accounting(self, ledger: Mapping[str, Any]) -> dict[str, Any]:
        return v4_call_accounting(ledger, run_dir=getattr(self, "run_dir", None))

    def _write_exports(self) -> None:
        exports = self.run_dir / "exports"
        groups = {
            "core_quality": [],
            "reserve_quality": [],
            "ambiguous": [],
            "rejected": [],
            "awaiting_explicit_commit": [],
        }
        for candidate_id, record in sorted(self.state["records"].items()):
            if "complete_task" not in record:
                continue
            intrinsic = record.get("intrinsic_quality")
            if intrinsic in {"core_quality", "reserve_quality"}:
                groups[intrinsic].append((candidate_id, record["complete_task"]))
            if record["status"] == "ambiguous":
                groups["ambiguous"].append((candidate_id, record["complete_task"]))
            if record["status"].startswith("rejected") or record["status"] in {"portfolio_duplicate", "portfolio_hold"}:
                groups["rejected"].append((candidate_id, record["complete_task"]))
            if record["status"] == "awaiting_explicit_commit":
                groups["awaiting_explicit_commit"].append((candidate_id, record["complete_task"]))
        manifests = {}
        for group, entries in groups.items():
            folder = exports / group
            files = []
            for candidate_id, task in entries:
                path = folder / f"{candidate_id}.json"
                atomic_write_json(path, task)
                files.append(str(path.relative_to(self.run_dir)))
            manifests[group] = files
        atomic_write_json(exports / "manifest.json", manifests)
        # These are proposals for the user's explicit bank decision, never an
        # automatic mutation of the frozen committed-task registry.
        commit_candidates = [
            {
                "candidate_id": candidate_id,
                "intrinsic_quality": record.get("intrinsic_quality"),
                "portfolio_placement": record.get("portfolio_placement"),
                "task_path": f"tasks/{candidate_id}.arc.json",
                "concept_summary": record["central_relation"],
                "intended_rule": record.get("realization", {}).get("authoring_metadata", {}).get("intended_rule"),
                "requires_explicit_user_authorization": True,
            }
            for candidate_id, record in sorted(self.state["records"].items())
            if record.get("status") == "awaiting_explicit_commit"
        ]
        atomic_write_json(self.run_dir / "commit_candidates.json", commit_candidates)

    def _reservoir_cost_attribution(self) -> dict[str, Any]:
        window = self.state.get("frozen_window") or {}
        proposals = self.state.get("window_proposals", {})
        supply_ids = sorted({item["supply_run_id"] for item in proposals.values()})
        ledgers = []
        total_calls = 0
        model_seconds = 0.0
        retries = 0
        for supply_id in supply_ids:
            path = Path(self.config["paths"]["runs"]) / "v4_reservoir_supply" / supply_id / "state.json"
            if not path.is_file():
                ledgers.append({"supply_run_id": supply_id, "state_path": str(path), "available": False})
                continue
            state = read_json(path)
            accounting = v4_call_accounting(
                state.get("call_ledger", {}), run_dir=path.parent
            )
            total_calls += accounting["substantive_logical_calls"]
            model_seconds += accounting["active_model_seconds"]
            retries += accounting["transport_retry_calls"]
            ledgers.append({"supply_run_id": supply_id, "state_path": str(path), "available": True, "accounting": accounting})
        return {
            "idea_generation_on_control_critical_path": False,
            "frozen_window_id": window.get("window_id"),
            "attributed_supply_run_ids": supply_ids,
            "attributed_supply_calls": total_calls,
            "attributed_supply_model_minutes": round(model_seconds / 60, 3),
            "attributed_supply_transport_retries": retries,
            "supply_ledgers": ledgers,
        }

    def finalize(self) -> dict[str, Any]:
        with self.lock:
            if self.state.get("completed"):
                return self.state
            self.state["completed"] = True
            self.state["completed_at_epoch_seconds"] = round(time.time(), 3)
            self.state["wall_time_seconds"] = round(
                self.state["completed_at_epoch_seconds"] - self.state["started_at_epoch_seconds"], 3
            )
            self.state["control_call_accounting"] = self._call_accounting(self.state["call_ledger"])
            self.state["reservoir_generation_cost"] = self._reservoir_cost_attribution()
            self.state["reservoir_at_completion"] = reservoir_status(
                load_reservoir(self.config["paths"]["reservoir"]), self.config["reservoir"]
            )
            statuses = Counter(record["status"] for record in self.state["records"].values())
            intrinsic = Counter(record.get("intrinsic_quality", "not_reviewed") for record in self.state["records"].values())
            placement = Counter(record.get("portfolio_placement", "not_reviewed") for record in self.state["records"].values())
            self.state["funnel_summary"] = {
                "eligible_window": len((self.state.get("frozen_window") or {}).get("proposal_ids", [])),
                "shard_finalists": sum(len(item["selected"]) for item in self.state["shard_selectors"].values()),
                "selected_and_reserved": len(self.state["records"]),
                "plans_ready": sum(record.get("concrete_plan", {}).get("status") == "ready" for record in self.state["records"].values()),
                "plans_aborted": sum(record.get("status") == "aborted_plan" for record in self.state["records"].values()),
                "realized": sum("realization" in record for record in self.state["records"].values()),
                "technical_pass": sum(any(item.get("passed") for item in record.get("technical_results", [])) for record in self.state["records"].values()),
                "blind_cleared": sum(
                    bool(record.get("ambiguity_resolution", {}).get("one_answer_supported"))
                    for record in self.state["records"].values()
                ),
                "serious_primary_blind_alternatives": sum(
                    bool(record.get("blind_results"))
                    and bool(record["blind_results"][0].get("serious_different_output_alternative"))
                    for record in self.state["records"].values()
                ),
                "tasks_repaired": sum(bool(record.get("repair_count")) for record in self.state["records"].values()),
                "editorially_reviewed": sum("editorial_judgments" in record for record in self.state["records"].values()),
                "terminal_statuses": dict(sorted(statuses.items())),
                "intrinsic_quality": dict(sorted(intrinsic.items())),
                "portfolio_placement": dict(sorted(placement.items())),
            }
            self._save()
        self._write_exports()
        atomic_write_json(
            self.run_dir / "reservoir/reservoir_state.at_completion.json",
            load_reservoir(self.config["paths"]["reservoir"]),
        )
        atomic_write_json(
            self.run_dir / "timing/candidate_events.json",
            self.state["candidate_events"],
        )
        return self.state

    def run(self) -> dict[str, Any]:
        if self.state.get("completed"):
            return self.state
        self.freeze_window()
        self.select_slate()
        self.run_candidate_graph()
        return self.finalize()


def recover_interrupted_v4_control_run(
    config: dict[str, Any],
    run_dir: str | Path,
    *,
    reason: str,
) -> dict[str, Any]:
    """Explicitly migrate and resume an interrupted control slate.

    This is deliberately not part of ordinary resume.  It is intended for a
    diagnosed interface/transport interruption where the exact frozen window
    and selected slate must be preserved.  Raw planner responses are salvaged
    only when the current strict parser validates the complete response after
    unambiguous referential-label normalization.
    """

    explanation = " ".join(reason.split())
    if not explanation:
        raise V4PipelineError("interrupted-run recovery requires a reason")
    directory = Path(run_dir).resolve()
    state_path = directory / "state.json"
    if not state_path.is_file():
        raise V4PipelineError(f"missing interrupted V4 state: {state_path}")
    state = read_json(state_path)
    if state.get("pipeline") != CONTROL_PIPELINE_ID or state.get("completed"):
        raise V4PipelineError("only an incomplete V4 control run may be recovered")
    if str(config["run"]["id"]) != state.get("run_id"):
        raise V4PipelineError("recovery config run ID differs from the interrupted run")
    if not state.get("frozen_window") or not state.get("records"):
        raise V4PipelineError("interrupted run has no frozen selected slate")

    migration_index = len(state.get("resume_migrations", [])) + 1
    migration_id = f"interrupted-recovery-{migration_index:02d}"
    migration_dir = directory / "resume_migrations" / migration_id
    migration_dir.mkdir(parents=True, exist_ok=False)
    old_manifest = read_json(directory / "implementation_manifest.json")
    old_reservoir = load_reservoir(config["paths"]["reservoir"])
    atomic_write_json(migration_dir / "implementation_manifest.before.json", old_manifest)
    atomic_write_json(migration_dir / "state.before.json", state)
    atomic_write_json(migration_dir / "reservoir.before.json", old_reservoir)

    salvaged: dict[str, Any] = {}
    rerun_required: dict[str, Any] = {}
    for candidate_id, record in sorted(
        state["records"].items(), key=lambda item: item[1]["development_priority"]
    ):
        call_id = f"{candidate_id}-concrete-plan"
        log_dir = directory / "model_calls" / "v4_concrete_plan" / call_id
        request_path = log_dir / "request.json"
        if not request_path.is_file():
            rerun_required[candidate_id] = {"reason": "planner request was never persisted"}
            continue
        request = read_json(request_path)
        valid_attempt = None
        failures = []
        for raw_path in sorted(log_dir.glob("attempt-*.raw_response.txt")):
            try:
                attempt = int(raw_path.name.split("-", 1)[1].split(".", 1)[0])
                raw = raw_path.read_text(encoding="utf-8")
                raw_value = json.loads(raw)
                before = [
                    item.get("pair") for item in raw_value.get("canvas_plan", [])
                ] if isinstance(raw_value, dict) else []
                parsed = parse_stage_output("v4_concrete_plan", raw)
                if parsed.get("candidate_id") != candidate_id:
                    raise V4PipelineError("planner response has the wrong candidate ID")
                after = [item.get("pair") for item in parsed.get("canvas_plan", [])]
                changes = [
                    {"from": source, "to": target}
                    for source, target in zip(before, after, strict=False)
                    if source != target
                ]
                valid_attempt = (attempt, raw_path, raw, parsed, changes)
                break
            except Exception as exc:
                failures.append({"path": str(raw_path), "error": str(exc)})
        if valid_attempt is None:
            # Preserve all failed/partial attempt evidence before the generic
            # invoker starts again at attempt 1.
            for path in sorted(log_dir.glob("attempt-*")):
                path.rename(log_dir / f"pre-{migration_id}-{path.name}")
            parse_failures = log_dir / "parse_failures.json"
            if parse_failures.exists():
                parse_failures.rename(log_dir / f"pre-{migration_id}-parse_failures.json")
            rerun_required[candidate_id] = {
                "reason": "no fully valid raw response was recoverable",
                "failed_raw_responses": failures,
            }
            continue
        attempt, raw_path, raw, parsed, changes = valid_attempt
        timing_path = log_dir / f"attempt-{attempt}.timing.json"
        timing = read_json(timing_path) if timing_path.is_file() else None
        atomic_write_text(log_dir / "raw_response.txt", raw)
        atomic_write_json(log_dir / "parsed_response.json", parsed)
        atomic_write_json(
            log_dir / "result.json",
            {
                "cache_key": request["cache_key"],
                "cache_source": "explicit-interrupted-run-recovery",
                "retry_count": attempt - 1,
                "elapsed_seconds": (timing or {}).get("total_seconds"),
                "timing": timing,
                "raw_response_hash": content_hash(raw),
                "parsed_response_hash": content_hash(parsed),
            },
        )
        recovery_record = {
            "migration_id": migration_id,
            "source_attempt": attempt,
            "source_raw_response": str(raw_path),
            "normalizations": changes,
            "strict_validation_passed": True,
            "semantic_content_changed": False,
        }
        atomic_write_json(log_dir / "explicit_recovery.json", recovery_record)
        salvaged[candidate_id] = recovery_record

    slate_id = next(iter(state["records"].values()))["reservation"]["owner_slate_id"]
    selections = [
        {
            "proposal_id": candidate_id,
            "candidate_id": candidate_id,
            "mechanism_summary": record["central_relation"],
        }
        for candidate_id, record in sorted(
            state["records"].items(), key=lambda item: item[1]["development_priority"]
        )
    ]
    reservation = recover_interrupted_slate(
        config["paths"]["reservoir"],
        window_id=state["frozen_window"]["window_id"],
        owner_run_id=state["run_id"],
        slate_id=slate_id,
        selections=selections,
        lease_seconds=int(config["reservoir"]["lease_seconds"]),
        reason=explanation,
    )
    reservations = {item["proposal_id"]: item for item in reservation["reservations"]}

    now = round(time.time(), 3)
    prior_records = {}
    for candidate_id, record in state["records"].items():
        prior_records[candidate_id] = {
            "status": record.get("status"),
            "current_stage": record.get("current_stage"),
            "rejections": deepcopy(record.get("rejections", [])),
            "failed_call_ids": deepcopy(record.get("failed_call_ids", [])),
            "active_call_ids": deepcopy(record.get("active_call_ids", [])),
            "stage_timing": deepcopy(record.get("stage_timings", {}).get("concrete_planning")),
        }
        record.setdefault("recovery_history", []).append(
            {"migration_id": migration_id, **prior_records[candidate_id]}
        )
        record["status"] = "selected_reserved"
        record["current_stage"] = "concrete_planning"
        record["reservation"] = reservations[candidate_id]
        record["active_call_ids"] = []
        record["completed_call_ids"] = []
        record["failed_call_ids"] = []
        record["rejections"] = []
        record.setdefault("stage_timings", {}).pop("concrete_planning", None)
        for key in (
            "final_reason", "failure_reason", "terminal_disposition",
            "terminal_memory_tier", "terminal_category", "terminal_at_epoch_seconds",
        ):
            record.pop(key, None)
        event = {
            "candidate_id": candidate_id,
            "stage": "interrupted_run_recovery",
            "event": "reopened",
            "migration_id": migration_id,
            "planner_response_salvaged": candidate_id in salvaged,
            "at_epoch_seconds": now,
        }
        record.setdefault("events", []).append(deepcopy(event))
        state.setdefault("candidate_events", []).append(event)

    for key, item in state.get("call_ledger", {}).items():
        if item.get("role") != "concrete_planner":
            continue
        item.setdefault("recovery_history", []).append(
            {
                "migration_id": migration_id,
                "status": item.get("status"),
                "error": item.get("error"),
                "finished_at_epoch_seconds": item.get("finished_at_epoch_seconds"),
            }
        )
        item.update(
            {
                "status": "reserved",
                "attempts": 0,
                "retry_calls": 0,
                "cache_source": None,
            }
        )
        item.pop("error", None)
        item.pop("finished_at_epoch_seconds", None)

    new_manifest = _implementation_value(Path(config["_repository_root"]))
    state["implementation_hash"] = new_manifest["implementation_hash"]
    state["slate_reservation"] = reservation
    state["completed"] = False
    state.setdefault("stage_progress", {}).pop("candidate_graph", None)
    state["stage_progress"].pop("editorial_queue_flushed", None)
    migration = {
        "migration_id": migration_id,
        "at_epoch_seconds": now,
        "reason": explanation,
        "old_implementation_hash": old_manifest["implementation_hash"],
        "new_implementation_hash": new_manifest["implementation_hash"],
        "same_frozen_window": state["frozen_window"]["window_id"],
        "same_selected_slate": list(state["records"]),
        "salvaged_planner_responses": sorted(salvaged),
        "rerun_planners": sorted(rerun_required),
        "semantic_content_changed": False,
        "prior_record_states": prior_records,
    }
    state.setdefault("resume_migrations", []).append(migration)
    atomic_write_json(migration_dir / "migration.json", migration)
    atomic_write_json(migration_dir / "salvaged_responses.json", salvaged)
    atomic_write_json(migration_dir / "rerun_required.json", rerun_required)
    atomic_write_json(directory / "implementation_manifest.json", new_manifest)
    atomic_write_json(state_path, state)
    return {
        "run_id": state["run_id"],
        "migration_id": migration_id,
        "frozen_window_id": state["frozen_window"]["window_id"],
        "selected_slate_count": len(state["records"]),
        "salvaged_planner_response_count": len(salvaged),
        "rerun_planner_count": len(rerun_required),
        "salvaged_candidate_ids": sorted(salvaged),
        "rerun_candidate_ids": sorted(rerun_required),
        "migration_directory": str(migration_dir),
    }


def run_v4_control(config: dict[str, Any]) -> dict[str, Any]:
    return V4ControlPipeline(config).run()


def sync_explicit_v4_commits(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Convert pending V4 leases only after a verified external bank commit."""

    directory = Path(run_dir).resolve()
    state_path = directory / "state.json"
    state = read_json(state_path)
    registry = load_committed_task_registry(config["paths"]["committed_task_registry"])
    changed = []
    for entry in registry["entries"]:
        if entry["source_run"] != state["run_id"]:
            continue
        candidate_id = entry["source_candidate_id"]
        record = state["records"].get(candidate_id)
        if not record or record.get("status") != "awaiting_explicit_commit":
            continue
        committed_task = read_json(Path(entry["absolute_task_path"]))
        if committed_task != record["complete_task"]:
            raise V4PipelineError(f"committed payload differs from V4 task {candidate_id}")
        resolve_proposal(
            config["paths"]["reservoir"], proposal_id=candidate_id,
            outcome="accepted", reason="explicit user-authorized good-task registry commit",
            source_run_id=state["run_id"],
            committed_registry_task_id=entry["registry_task_id"],
            explicit_commit_verified=True,
        )
        record["status"] = "accepted"
        record["committed_registry_task_id"] = entry["registry_task_id"]
        changed.append(candidate_id)
    if changed:
        atomic_write_json(state_path, state)
    return {"run_id": state["run_id"], "accepted_transitions": changed, "count": len(changed)}


def reject_pending_v4_candidate(
    config: dict[str, Any], run_dir: str | Path, *, candidate_id: str, reason: str
) -> dict[str, Any]:
    """Explicitly close a reviewed good-task lease without committing it."""

    if not reason.strip():
        raise V4PipelineError("an explicit rejection reason is required")
    directory = Path(run_dir).resolve()
    state_path = directory / "state.json"
    state = read_json(state_path)
    record = state.get("records", {}).get(candidate_id)
    if record is None or record.get("status") != "awaiting_explicit_commit":
        raise V4PipelineError(
            f"{candidate_id} is not awaiting an explicit V4 disposition"
        )
    resolve_proposal(
        config["paths"]["reservoir"], proposal_id=candidate_id,
        outcome="rejected", reason=reason.strip(), source_run_id=state["run_id"],
        memory_tier="soft_mechanism",
    )
    record.update(
        {
            "status": "rejected_by_user",
            "current_stage": "terminal",
            "final_reason": reason.strip(),
            "terminal_disposition": "rejected",
            "terminal_memory_tier": "soft_mechanism",
            "terminal_category": "explicit_user_disposition",
            "terminal_at_epoch_seconds": round(time.time(), 3),
        }
    )
    record.setdefault("rejections", []).append(
        {
            "stage": "explicit_user_disposition",
            "category": "explicit_user_disposition",
            "reason": reason.strip(),
        }
    )
    atomic_write_json(state_path, state)
    return {
        "run_id": state["run_id"],
        "candidate_id": candidate_id,
        "status": "rejected_by_user",
        "memory_tier": "soft_mechanism",
    }


__all__ = [
    "PROMPT_FILES",
    "PROMPT_TO_STAGE",
    "V4ControlPipeline",
    "V4PipelineError",
    "V4PromptStore",
    "V4ReservoirProducer",
    "run_v4_control",
    "reject_pending_v4_candidate",
    "sync_explicit_v4_commits",
]
