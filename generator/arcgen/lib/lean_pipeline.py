"""Lean, authoring-first ARC generation pipeline.

The module is a separate versioned execution path.  It reuses stable storage,
model, retrieval, and ARC-validation infrastructure without importing any
factor-mode decisions from the frozen MVP pipeline.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any, Callable, Iterable, Sequence
import inspect
import json
import threading
import time

from .concept_archive import (
    append_lean_concepts,
    build_legacy_archive,
    initialize_lean_archive,
    load_archive,
    portfolio_digest,
    reference_records,
)
from .data import TaskRecord, load_task_bank
from .embeddings import build_embedding_provider, cosine_similarity
from .lean_config import (
    NORMAL_CALL_LIMIT,
    NORMAL_STAGE_LIMITS,
    PIPELINE_ID,
    SEARCH_MODES,
    snapshot_lean_config,
)
from .lean_diagnostics import (
    PAIR_ECONOMY_FLAG_CODES,
    combine_public_and_hidden,
    diagnose_arc_task,
)
from .lean_schemas import OUTPUT_SCHEMAS, parse_stage_output
from .model import CallResult, ModelInvocationError, ModelInvoker
from .retrieval import retrieve_union, select_diverse
from .schemas import OUTPUT_SCHEMAS as LEGACY_SCHEMAS
from .schemas import parse_stage_output as parse_legacy_stage_output
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
from .validation import (
    assert_no_hidden_test_leakage,
    solver_facing_task,
    validate_arc_task,
)


LEAN_PIPELINE_VERSION = "arcgen-lean-authoring-v1.0.1"
PROMPT_FILES = {
    "idea_open": "lean_v1_idea_open.md",
    "idea_concept": "lean_v1_idea_concept.md",
    "idea_composition": "lean_v1_idea_composition.md",
    "batch_selector": "lean_v1_batch_selector.md",
    "episode_architect": "lean_v1_episode_architect.md",
    "task_realizer": "lean_v1_task_realizer.md",
    "technical_verifier": "lean_v1_technical_verifier.md",
    "blind_solver": "lean_v1_blind_solver.md",
    "ambiguity_adjudicator": "lean_v1_ambiguity_adjudicator.md",
    "targeted_repair": "lean_v1_targeted_repair.md",
    "editorial_board": "lean_v1_editorial_board.md",
}
ROLE_TO_STAGE = {
    "idea_search": "lean_idea_search",
    "batch_selector": "lean_batch_selector",
    "episode_architect": "lean_episode_architect",
    "task_realizer": "lean_task_realizer",
    "technical_verifier": "lean_technical_verifier",
    "blind_solver": "lean_blind_solver",
    "ambiguity_adjudicator": "lean_ambiguity_adjudicator",
    "targeted_repair": "lean_targeted_repair",
    "editorial_board": "lean_editorial_board",
}


class LeanPipelineError(RuntimeError):
    pass


class LeanPromptStore:
    def __init__(self, prompt_dir: Path, run_dir: Path) -> None:
        self.entries: dict[str, dict[str, str]] = {}
        destination = run_dir / "prompts" / "templates"
        destination.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "pipeline": PIPELINE_ID,
            "templates": {},
        }
        for name, filename in PROMPT_FILES.items():
            path = prompt_dir / filename
            if not path.is_file():
                raise LeanPipelineError(f"missing lean prompt: {path}")
            text, version, digest = load_prompt(path)
            self.entries[name] = {
                "text": text,
                "version": version,
                "hash": digest,
                "filename": filename,
            }
            snapshot = destination / filename
            if snapshot.exists() and snapshot.read_text(encoding="utf-8") != text:
                raise LeanPipelineError(
                    f"prompt changed within existing run: {filename}"
                )
            if not snapshot.exists():
                atomic_write_text(snapshot, text)
            manifest["templates"][name] = {
                "filename": filename,
                "version": version,
                "sha256": digest,
            }
        manifest_path = run_dir / "prompts" / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise LeanPipelineError("lean prompt manifest changed within existing run")
        if not manifest_path.exists():
            atomic_write_json(manifest_path, manifest)

    def render(self, name: str, values: dict[str, Any]) -> tuple[str, str, str]:
        entry = self.entries[name]
        required = set(PLACEHOLDER_RE.findall(entry["text"]))
        if required != set(values):
            raise LeanPipelineError(
                f"{name} prompt bindings differ: missing={sorted(required-set(values))}, "
                f"extra={sorted(set(values)-required)}"
            )
        return (
            render_prompt(entry["text"], values),
            entry["version"],
            entry["hash"],
        )


def assert_no_fixed_anchor_leakage(
    prompt: str, anchors: Iterable[TaskRecord]
) -> None:
    leaks: list[str] = []
    for anchor in anchors:
        if anchor.task_id in prompt:
            leaks.append(f"ID {anchor.task_id}")
        if anchor.rule_description in prompt:
            leaks.append(f"description {anchor.task_id}")
    if leaks:
        raise LeanPipelineError(
            "fixed editorial-anchor leakage into pre-editorial prompt: "
            + ", ".join(leaks)
        )


def technical_output_matches_hidden(
    response: dict[str, Any], hidden_outputs: Sequence[Any]
) -> bool:
    """Programmatic, exact hidden-output comparison used by the acceptance gate."""

    return response.get("derived_test_outputs") == list(hidden_outputs)


def serious_blind_alternative(
    response: dict[str, Any], hidden_outputs: Sequence[Any]
) -> bool:
    return any(
        alternative.get("fits_all_training_examples") is True
        and alternative.get("comparably_simple") is True
        and alternative.get("well_supported") is True
        and alternative.get("predicted_test_outputs") != list(hidden_outputs)
        for alternative in response.get("strongest_plausible_alternatives", [])
    )


def classify_editorial_results(
    judgments: Sequence[dict[str, Any]],
) -> tuple[str, str]:
    """Apply the one-answer, two-axis final policy to two board judgments."""

    if len(judgments) != 2:
        raise LeanPipelineError("exactly two editorial judgments are required")
    if any(item["novel_concept"] == "weak" for item in judgments):
        return "rejected", "an editorial board rated the concept weak"
    if any(item["functional_composition"] == "thin" for item in judgments):
        return "rejected", "an editorial board rated functional composition thin"
    if any(item["overall_quality"] == "below_real_bar" for item in judgments):
        return "rejected", "an editorial board rated quality below the real-task bar"
    if any(item["portfolio_relation"] == "same_family" for item in judgments):
        return "rejected", "an editorial board found a same-family portfolio duplicate"

    standout_each = all(
        item["novel_concept"] == "standout"
        or item["functional_composition"] == "standout"
        for item in judgments
    )
    quality_each = all(
        item["overall_quality"] in {"real_quality", "exceptional"}
        for item in judgments
    )
    distinct_each = all(
        item["portfolio_relation"] == "distinct" for item in judgments
    )
    no_board_reject = all(item["decision"] != "reject" for item in judgments)
    if standout_each and quality_each and distinct_each and no_board_reject:
        return (
            "core",
            "both boards cleared both axes, found a standout axis, and rated real quality",
        )
    return (
        "reserve",
        "valid and useful, but the boards did not unanimously clear the core editorial bar",
    )


class LeanAuthoringPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config["seed"])
        self.run_id = str(config["run"]["id"])
        self.run_dir = Path(config["paths"]["runs"]) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self._state_lock = threading.RLock()

        self.implementation_hash = self._snapshot_implementation()
        self._snapshot_config()
        self.bank = load_task_bank(
            config["paths"]["annotations"], config["paths"]["tasks"]
        )
        self.by_task_id = {record.task_id: record for record in self.bank}
        self.anchor_manifest = self._load_anchors()
        self.anchors = tuple(
            self.by_task_id[item["task_id"]]
            for item in self.anchor_manifest["anchors"]
        )
        self.anchor_ids = {item.task_id for item in self.anchors}
        self.generator_pool = self._generator_pool()
        self._validate_source_bank()

        legacy_path = Path(config["paths"]["legacy_concept_archive"])
        if not legacy_path.exists():
            build_legacy_archive(config["paths"]["runs"], legacy_path)
        self.legacy_archive = load_archive(legacy_path, expected_frozen=True)
        current_lean_archive = initialize_lean_archive(
            config["paths"]["lean_concept_archive"]
        )
        # A moving global archive may gain entries after this run completes.
        # Freeze the exact start-of-run view so resume and retrieval remain
        # reproducible rather than silently seeing future batches.
        archive_snapshot = (
            self.run_dir / "data" / "lean_concept_archive.start.json"
        )
        if archive_snapshot.exists():
            self.lean_archive = read_json(archive_snapshot)
        else:
            self.lean_archive = deepcopy(current_lean_archive)
            atomic_write_json(archive_snapshot, self.lean_archive)
        self.references = reference_records(
            self.bank,
            self.legacy_archive,
            self.lean_archive,
            excluded_real_ids=self.anchor_ids,
        )

        self.state = self._load_or_initialize_state()
        self.prompts = LeanPromptStore(
            Path(config["paths"]["prompts"]), self.run_dir
        )
        self.embedder = build_embedding_provider(
            config["embedding"], config["paths"]["cache"]
        )
        self.invoker = ModelInvoker(
            config,
            self.run_dir,
            output_schemas=OUTPUT_SCHEMAS,
            output_parser=parse_stage_output,
        )
        self._write_data_artifacts()

    # ---------- immutable run preparation ----------

    def _snapshot_implementation(self) -> str:
        package = Path(__file__).resolve().parent
        names = {
            "lean_pipeline.py",
            "lean_config.py",
            "lean_schemas.py",
            "lean_diagnostics.py",
            "lean_reporting.py",
            "concept_archive.py",
            "model.py",
            "embeddings.py",
            "retrieval.py",
            "validation.py",
            "data.py",
            "util.py",
        }
        files = {
            name: content_hash((package / name).read_bytes())
            for name in sorted(names)
            if (package / name).exists()
        }
        manifest = {
            "pipeline_version": LEAN_PIPELINE_VERSION,
            "files": files,
        }
        digest = content_hash(manifest)
        manifest["implementation_hash"] = digest
        path = self.run_dir / "implementation_manifest.json"
        if path.exists() and read_json(path) != manifest:
            raise LeanPipelineError(
                "lean implementation changed within an existing run"
            )
        if not path.exists():
            atomic_write_json(path, manifest)
        return digest

    def _snapshot_config(self) -> None:
        snapshot = snapshot_lean_config(self.config)
        path = self.run_dir / "config.snapshot.json"
        if path.exists() and read_json(path) != snapshot:
            raise LeanPipelineError(
                f"run {self.run_id!r} has a different configuration snapshot"
            )
        if not path.exists():
            atomic_write_json(path, snapshot)
            atomic_write_text(
                self.run_dir / "config.snapshot.yaml",
                json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            )

    def _load_anchors(self) -> dict[str, Any]:
        value = read_json(Path(self.config["paths"]["fixed_anchor_manifest"]))
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("anchors"), list)
            or len(value["anchors"]) != 6
        ):
            raise LeanPipelineError("fixed-anchor manifest must contain six anchors")
        ids = [item.get("task_id") for item in value["anchors"]]
        if len(set(ids)) != 6 or any(task_id not in self.by_task_id for task_id in ids):
            raise LeanPipelineError("fixed-anchor manifest contains invalid task IDs")
        roles = Counter(item.get("editorial_role") for item in value["anchors"])
        if roles != Counter({"concept_led": 2, "composition_led": 2, "both": 2}):
            raise LeanPipelineError(
                "fixed anchors must contain two concept-led, two composition-led, and two both"
            )
        return value

    def _generator_pool(self) -> tuple[TaskRecord, ...]:
        available = sorted(
            (record for record in self.bank if record.task_id not in self.anchor_ids),
            key=lambda item: item.task_id,
        )
        Random(self.seed).shuffle(available)
        count = int(self.config["experiment"]["generator_visible_pool_count"])
        return tuple(available[:count])

    def _validate_source_bank(self) -> None:
        failures: list[str] = []
        for record in self.bank:
            result = validate_arc_task(
                {"train": list(record.train), "test": list(record.test)}
            )
            if not result.valid:
                failures.append(f"{record.task_id}: {'; '.join(result.errors)}")
        if failures:
            raise LeanPipelineError(
                "source ARC bank failed deterministic validation: "
                + " | ".join(failures[:5])
            )

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = read_json(self.state_path)
            if (
                state.get("pipeline_version") != LEAN_PIPELINE_VERSION
                or state.get("seed") != self.seed
                or state.get("implementation_hash") != self.implementation_hash
            ):
                raise LeanPipelineError("existing lean run is stale or incompatible")
            return state
        state = {
            "pipeline_version": LEAN_PIPELINE_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "implementation_hash": self.implementation_hash,
            "completed": False,
            "started_at_epoch_seconds": time.time(),
            "stage_progress": {},
            "ideas": [],
            "records": {},
            "call_budget": {
                "normal_limit": int(self.config["budget"]["normal_call_limit"]),
                "escalation_limit": int(
                    self.config["budget"]["escalation_call_limit"]
                ),
                "hard_limit": int(
                    self.config["budget"]["hard_substantive_call_limit"]
                ),
                "normal_stage_limits": deepcopy(NORMAL_STAGE_LIMITS),
                "ledger": {},
            },
            "fixed_anchor_panel_id": self.anchor_manifest["panel_id"],
            "difficulty_factor_path_active": False,
            "implementation_limitations": [
                "The fixed six-anchor panel is provisional and awaits user ratification.",
                "Embedding similarity scouts references but never determines acceptance.",
                "One blind solver is the normal path; difficult solver failure alone does not reject a task.",
                "Only one targeted repair is permitted per realized task.",
            ],
        }
        atomic_write_json(self.state_path, state)
        return state

    def _save(self) -> None:
        with self._state_lock:
            atomic_write_json(self.state_path, self.state)

    def _write_data_artifacts(self) -> None:
        data = self.run_dir / "data"
        data.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "generator_visible_pool.json": {
                "seed": self.seed,
                "task_ids": [item.task_id for item in self.generator_pool],
                "fixed_anchor_ids_excluded": sorted(self.anchor_ids),
            },
            "fixed_anchor_panel.json": self.anchor_manifest,
            "reference_archive_manifest.json": {
                "real_reference_count": sum(
                    item["source"] == "real_task" for item in self.references
                ),
                "generated_reference_count": sum(
                    item["source"] == "previous_generated"
                    for item in self.references
                ),
                "legacy_archive_hash": content_hash(self.legacy_archive),
                "lean_archive_hash_at_start": content_hash(self.lean_archive),
                "fixed_anchor_ids_excluded": sorted(self.anchor_ids),
            },
            "bank_manifest.json": {
                "task_count": len(self.bank),
                "logical_hash": content_hash([item.to_dict() for item in self.bank]),
            },
        }
        for filename, value in artifacts.items():
            path = data / filename
            if path.exists() and read_json(path) != value:
                raise LeanPipelineError(
                    f"source/selection changed within run: {filename}"
                )
            if not path.exists():
                atomic_write_json(path, value)

    # ---------- call budget and invocation ----------

    def _role_settings(self, role: str) -> dict[str, Any]:
        setting = self.config["model"].get("roles", {}).get(role, {})
        return {
            "model_name": setting.get("model", self.config["model"]["name"]),
            "reasoning_effort": setting.get(
                "reasoning_effort", self.config["model"]["reasoning_effort"]
            ),
            "timeout_seconds": setting.get("timeout_seconds"),
            "retries": setting.get("retries"),
        }

    def _reserve_call(
        self, *, role: str, stage: str, call_id: str, budget_kind: str
    ) -> str:
        key = f"{stage}/{call_id}"
        with self._state_lock:
            ledger = self.state["call_budget"]["ledger"]
            if key in ledger:
                existing = ledger[key]
                if (
                    existing["role"] != role
                    or existing["budget_kind"] != budget_kind
                ):
                    raise LeanPipelineError(f"call identity reused inconsistently: {key}")
                return key
            normal_count = sum(
                entry["budget_kind"] == "normal" for entry in ledger.values()
            )
            escalation_count = sum(
                entry["budget_kind"] == "escalation" for entry in ledger.values()
            )
            if budget_kind == "normal":
                stage_count = sum(
                    entry["budget_kind"] == "normal" and entry["role"] == role
                    for entry in ledger.values()
                )
                limit = NORMAL_STAGE_LIMITS.get(role)
                if limit is None or stage_count >= limit:
                    raise LeanPipelineError(
                        f"normal call limit for {role} would be exceeded"
                    )
                if normal_count >= self.state["call_budget"]["normal_limit"]:
                    raise LeanPipelineError("normal 23-call budget would be exceeded")
            elif budget_kind == "escalation":
                if escalation_count >= self.state["call_budget"]["escalation_limit"]:
                    raise LeanPipelineError("five-call escalation budget is exhausted")
            else:
                raise LeanPipelineError(f"unknown budget kind {budget_kind}")
            if normal_count + escalation_count >= self.state["call_budget"]["hard_limit"]:
                raise LeanPipelineError("hard 28-call batch budget would be exceeded")
            ledger[key] = {
                "role": role,
                "stage": stage,
                "call_id": call_id,
                "budget_kind": budget_kind,
                "status": "reserved",
                "attempts": 0,
                "retry_calls": 0,
                "cache_source": None,
            }
            self._save()
        return key

    def _remaining_escalation(self) -> int:
        ledger = self.state["call_budget"]["ledger"]
        used = sum(
            item["budget_kind"] == "escalation" for item in ledger.values()
        )
        return int(self.config["budget"]["escalation_call_limit"]) - used

    def _call(
        self,
        *,
        prompt_name: str,
        role: str,
        call_id: str,
        values: dict[str, Any],
        parent_ids: list[str],
        budget_kind: str = "normal",
        expected: dict[str, Any] | None = None,
        response_validator: Callable[[dict[str, Any]], None] | None = None,
        pre_editorial_anchor_blind: bool = False,
    ) -> dict[str, Any]:
        prompt, version, digest = self.prompts.render(prompt_name, values)
        if pre_editorial_anchor_blind:
            assert_no_fixed_anchor_leakage(prompt, self.anchors)

        def validate(value: dict[str, Any]) -> None:
            for key, expected_value in (expected or {}).items():
                if value.get(key) != expected_value:
                    raise LeanPipelineError(
                        f"{role}/{call_id} returned {key}={value.get(key)!r}, "
                        f"expected {expected_value!r}"
                    )
            if response_validator is not None:
                response_validator(value)

        stage = ROLE_TO_STAGE[role]
        ledger_key = self._reserve_call(
            role=role, stage=stage, call_id=call_id, budget_kind=budget_kind
        )
        settings = self._role_settings(role)
        kwargs: dict[str, Any] = {
            "stage": stage,
            "call_id": call_id,
            "prompt": prompt,
            "prompt_template_version": version,
            "prompt_template_hash": digest,
            "parent_ids": parent_ids,
            "response_validator": validate,
            "model_name": settings["model_name"],
            "reasoning_effort": settings["reasoning_effort"],
        }
        if settings["timeout_seconds"] is not None:
            kwargs["timeout_seconds"] = settings["timeout_seconds"]
        if settings["retries"] is not None:
            kwargs["retries"] = settings["retries"]
        try:
            result: CallResult = self.invoker.invoke(**kwargs)
        except Exception as exc:
            with self._state_lock:
                self.state["call_budget"]["ledger"][ledger_key].update(
                    {"status": "failed", "error": str(exc)}
                )
                self._save()
            raise
        with self._state_lock:
            result_record = read_json(result.log_dir / "result.json")
            self.state["call_budget"]["ledger"][ledger_key].update(
                {
                    "status": "complete",
                    "attempts": result.attempts,
                    "retry_calls": max(0, result.attempts - 1),
                    "cache_source": result.cache_source,
                    "cache_key": result.cache_key,
                    "log_dir": str(result.log_dir),
                    "timing": result_record.get("timing"),
                }
            )
            self._save()
        return result.value

    # ---------- generation and selection ----------

    def _calibration_assignments(self) -> list[list[TaskRecord]]:
        existing = self.state.get("calibration_assignments")
        if existing:
            return [
                [self.by_task_id[task_id] for task_id in group]
                for group in existing
            ]
        remaining = list(self.generator_pool)
        assignments: list[list[TaskRecord]] = []
        count = int(self.config["experiment"]["calibration_examples_per_call"])
        for index in range(3):
            selected = select_diverse(
                remaining,
                count,
                self.embedder,
                self.seed + 1000 + index,
                text_getter=lambda record: record.rule_description,
            )
            assignments.append(selected)
            selected_ids = {item.task_id for item in selected}
            remaining = [
                item for item in remaining if item.task_id not in selected_ids
            ]
        self.state["calibration_assignments"] = [
            [item.task_id for item in group] for group in assignments
        ]
        self._save()
        return assignments

    @staticmethod
    def _candidate_ids(mode: str) -> list[str]:
        prefix = mode.replace("_", "-")
        return [f"{prefix}-{index}" for index in range(1, 5)]

    def generate_ideas(self) -> None:
        if self.state["stage_progress"].get("idea_search"):
            return
        assignments = self._calibration_assignments()
        plan = [
            ("open", "idea_open"),
            ("concept_led", "idea_concept"),
            ("composition_led", "idea_composition"),
        ]

        def generate(spec: tuple[int, str, str]) -> dict[str, Any]:
            index, mode, prompt_name = spec
            call_id = f"{mode.replace('_', '-')}-search-1"
            ids = self._candidate_ids(mode)
            descriptions = [
                item.rule_description for item in assignments[index]
            ]

            def validate(value: dict[str, Any]) -> None:
                actual = [item["candidate_id"] for item in value["ideas"]]
                if actual != ids:
                    raise LeanPipelineError(
                        f"{call_id} candidate IDs differ from assigned IDs"
                    )

            return self._call(
                prompt_name=prompt_name,
                role="idea_search",
                call_id=call_id,
                values={
                    "CALIBRATION_DESCRIPTIONS": descriptions,
                    "SEARCH_CALL_ID": call_id,
                    "CANDIDATE_IDS": ids,
                },
                parent_ids=[],
                expected={"search_call_id": call_id, "search_mode": mode},
                response_validator=validate,
                pre_editorial_anchor_blind=True,
            )

        results = parallel_map(
            generate,
            [
                (index, mode, prompt_name)
                for index, (mode, prompt_name) in enumerate(plan)
            ],
            min(3, int(self.config["model"]["concurrency"])),
        )
        ideas: list[dict[str, Any]] = []
        for index, result in enumerate(results):
            calibration_ids = [
                item.task_id for item in assignments[index]
            ]
            for idea in result["ideas"]:
                record = deepcopy(idea)
                record.update(
                    {
                        "search_call_id": result["search_call_id"],
                        "calibration_task_ids": calibration_ids,
                        "status": "generated",
                        "rejections": [],
                    }
                )
                ideas.append(record)
        if len(ideas) != 12 or len({item["candidate_id"] for item in ideas}) != 12:
            raise LeanPipelineError("idea search did not produce 12 unique kernels")
        self.state["ideas"] = ideas
        self.state["records"] = {
            item["candidate_id"]: deepcopy(item) for item in ideas
        }
        self.state["stage_progress"]["idea_search"] = True
        self._save()

    def _nearest_references(
        self, text: str, *, count: int = 3
    ) -> list[dict[str, Any]]:
        hits = retrieve_union(
            text,
            self.references,
            self.embedder,
            top_k=count,
            text_getter=lambda item: item["central_relation"],
            id_getter=lambda item: item["reference_id"],
        )
        return [
            {
                "reference_id": hit.reference_id,
                "source": hit.reference["source"],
                "summary": hit.reference["summary"][:700],
                "similarity": round(hit.score, 6),
            }
            for hit in hits
        ]

    def select_batch(self) -> None:
        if self.state["stage_progress"].get("batch_selector"):
            return
        packages = []
        selections: dict[str, list[dict[str, Any]]] = {}
        for idea in self.state["ideas"]:
            neighbors = self._nearest_references(
                idea["central_relation"],
                count=int(
                    self.config["experiment"][
                        "selector_references_per_candidate"
                    ]
                ),
            )
            selections[idea["candidate_id"]] = neighbors
            packages.append({"kernel": idea, "nearest_references": neighbors})
        digest = portfolio_digest(self.legacy_archive, self.lean_archive)
        candidate_ids = {item["candidate_id"] for item in self.state["ideas"]}

        def validate(value: dict[str, Any]) -> None:
            assessed = {
                item["candidate_id"] for item in value["candidate_assessments"]
            }
            if assessed != candidate_ids:
                raise LeanPipelineError(
                    "selector must assess exactly all 12 candidates"
                )
            selected = {item["candidate_id"] for item in value["selected"]}
            if not selected <= candidate_ids:
                raise LeanPipelineError("selector selected an unknown candidate")
            decisions = {
                item["candidate_id"]: item["decision"]
                for item in value["candidate_assessments"]
            }
            if selected != {
                candidate_id
                for candidate_id, decision in decisions.items()
                if decision == "select"
            }:
                raise LeanPipelineError(
                    "selector assessments and editorial briefs disagree"
                )

        selector = self._call(
            prompt_name="batch_selector",
            role="batch_selector",
            call_id="batch-selector-1",
            values={
                "CANDIDATES_WITH_REFERENCES": packages,
                "PORTFOLIO_DIGEST": digest,
                "SELECTOR_ID": "batch-selector-1",
            },
            parent_ids=sorted(candidate_ids),
            expected={"selector_id": "batch-selector-1"},
            response_validator=validate,
            pre_editorial_anchor_blind=True,
        )
        self.state["reference_selections"] = selections
        self.state["selector"] = selector
        briefs = {
            item["candidate_id"]: item for item in selector["selected"]
        }
        assessments = {
            item["candidate_id"]: item
            for item in selector["candidate_assessments"]
        }
        for candidate_id, record in self.state["records"].items():
            record["selector_assessment"] = assessments[candidate_id]
            if candidate_id in briefs:
                record["editorial_brief"] = briefs[candidate_id]
                record["status"] = "selected_for_architecture"
            else:
                assessment = assessments[candidate_id]
                record["status"] = "rejected"
                record["rejections"].append(
                    {
                        "stage": "batch_selector",
                        "category": (
                            "novelty_or_duplication"
                            if assessment["derivative_of"] != "none"
                            else "development_potential"
                        ),
                        "reason": assessment["reason"],
                    }
                )
        self.state["stage_progress"]["batch_selector"] = True
        self._save()

    # ---------- architecture and realization ----------

    @staticmethod
    def _route_guidance(route: str) -> str:
        return {
            "concept_led": (
                "Preserve the unusual central insight. Add only enough functional "
                "structure, variation, and evidence to make a substantial episode."
            ),
            "composition_led": (
                "Develop multiple interdependent meaningful roles around one "
                "coherent mechanism; do not produce adjacent independent subproblems."
            ),
            "flagship": (
                "Preserve both conceptual novelty and functional composition; "
                "do not simplify one axis away while strengthening the other."
            ),
            "wildcard": (
                "Protect the high-upside insight while testing honestly whether it "
                "can reach solid composition and unique support; abort if it cannot."
            ),
        }[route]

    def create_architectures(self) -> None:
        if self.state["stage_progress"].get("episode_architect"):
            return
        selected = sorted(
            (
                record
                for record in self.state["records"].values()
                if record["status"] == "selected_for_architecture"
            ),
            key=lambda item: item["editorial_brief"]["development_priority"],
        )

        def architect(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            candidate_id = record["candidate_id"]
            brief = record["editorial_brief"]
            result = self._call(
                prompt_name="episode_architect",
                role="episode_architect",
                call_id=f"{candidate_id}-architecture",
                values={
                    "CANDIDATE": {
                        key: record[key]
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
                    "EDITORIAL_BRIEF": brief,
                    "ROUTE_GUIDANCE": self._route_guidance(brief["route"]),
                    "CANDIDATE_ID": candidate_id,
                },
                parent_ids=[candidate_id],
                expected={"candidate_id": candidate_id},
                pre_editorial_anchor_blind=True,
            )
            return candidate_id, result

        results = parallel_map(
            architect,
            selected,
            min(len(selected) or 1, int(self.config["model"]["concurrency"])),
        )
        for candidate_id, architecture in results:
            record = self.state["records"][candidate_id]
            record["architecture"] = architecture
            if architecture["status"] == "abort":
                record["status"] = "rejected"
                record["rejections"].append(
                    {
                        "stage": "episode_architect",
                        "category": "architecture",
                        "reason": architecture["abort_reason"],
                    }
                )
            else:
                record["status"] = "architecture_ready"
        ready = sorted(
            (
                record
                for record in self.state["records"].values()
                if record["status"] == "architecture_ready"
            ),
            key=lambda item: item["editorial_brief"]["development_priority"],
        )
        for record in ready[int(self.config["experiment"]["max_realizations"]) :]:
            record["status"] = "architecture_reserve"
            record["final_reason"] = (
                "ready architecture held as the fifth authored reserve under the "
                "four-realization batch cap"
            )
        self.state["stage_progress"]["episode_architect"] = True
        self._save()

    @staticmethod
    def _validate_realizer_response(
        value: dict[str, Any],
        architecture: dict[str, Any],
        candidate_id: str,
        *,
        allow_one_added_training_pair: bool = False,
    ) -> None:
        # A realizer may restate the architecture's rule more concretely while
        # constructing exact grids. Semantic preservation is inspected by the
        # author-aware verifier; byte-for-byte equality here would reject
        # faithful paraphrases and trigger wasteful retries.
        public = value["task"]
        assert_no_hidden_test_leakage(public)
        planned_count = len(architecture["training_pair_plan"])
        allowed_counts = (
            {planned_count, planned_count + 1}
            if allow_one_added_training_pair
            else {planned_count}
        )
        if len(public["train"]) not in allowed_counts:
            raise LeanPipelineError(
                f"{candidate_id} realization changed the planned training-pair count"
            )
        if len(public["test"]) != 1:
            raise LeanPipelineError(
                f"{candidate_id} architecture contains one test plan and must realize one test pair"
            )
        labels = {
            item["pair"]
            for item in value["authoring_metadata"]["pair_construction_trace"]
        }
        expected_labels = {
            *(f"train_{index}" for index in range(1, len(public["train"]) + 1)),
            "test_1",
        }
        if labels != expected_labels:
            raise LeanPipelineError(
                f"{candidate_id} pair construction trace has incomplete coverage"
            )

    def _store_realization(
        self, record: dict[str, Any], realization: dict[str, Any], *, repaired: bool = False
    ) -> None:
        metadata = realization["authoring_metadata"]
        complete = combine_public_and_hidden(
            realization["task"], metadata["expected_test_outputs"]
        )
        diagnostics = diagnose_arc_task(complete)
        if repaired:
            record.setdefault("repair_history", []).append(
                {
                    "prior_realization": record.get("realization"),
                    "prior_diagnostics": record.get("deterministic_diagnostics"),
                }
            )
        record["realization"] = realization
        record["complete_task"] = complete
        record["deterministic_diagnostics"] = diagnostics
        task_dir = self.run_dir / "tasks"
        atomic_write_json(
            task_dir / f"{record['candidate_id']}.authoring.json",
            {
                **deepcopy(realization),
                "architecture_contract": record["architecture"],
                "editorial_brief": record["editorial_brief"],
                "provenance": {
                    "pipeline": PIPELINE_ID,
                    "run_id": self.run_id,
                    "candidate_id": record["candidate_id"],
                    "search_call_id": record["search_call_id"],
                    "search_mode": record["search_mode"],
                },
            },
        )
        atomic_write_json(
            task_dir / f"{record['candidate_id']}.arc.json", complete
        )

    def realize_tasks(self) -> None:
        if self.state["stage_progress"].get("task_realizer"):
            return
        ready = sorted(
            (
                record
                for record in self.state["records"].values()
                if record["status"] == "architecture_ready"
            ),
            key=lambda item: item["editorial_brief"]["development_priority"],
        )[: int(self.config["experiment"]["max_realizations"])]

        def realize(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            candidate_id = record["candidate_id"]

            def validate(value: dict[str, Any]) -> None:
                self._validate_realizer_response(
                    value, record["architecture"], candidate_id
                )

            result = self._call(
                prompt_name="task_realizer",
                role="task_realizer",
                call_id=f"{candidate_id}-realization",
                values={
                    "CANDIDATE": {
                        "candidate_id": candidate_id,
                        "central_relation": record["central_relation"],
                        "why_coherent": record["why_coherent"],
                    },
                    "EDITORIAL_BRIEF": record["editorial_brief"],
                    "ARCHITECTURE": record["architecture"],
                    "CANDIDATE_ID": candidate_id,
                },
                parent_ids=[candidate_id],
                expected={"candidate_id": candidate_id},
                response_validator=validate,
                pre_editorial_anchor_blind=True,
            )
            return candidate_id, result

        results = parallel_map(
            realize,
            ready,
            min(len(ready) or 1, int(self.config["model"]["concurrency"])),
        )
        for candidate_id, realization in results:
            record = self.state["records"][candidate_id]
            self._store_realization(record, realization)
            if record["deterministic_diagnostics"]["hard_valid"]:
                record["status"] = "deterministic_pass"
            else:
                record["status"] = "rejected"
                record["rejections"].append(
                    {
                        "stage": "deterministic_diagnostics",
                        "category": "invalidity",
                        "reason": "; ".join(
                            record["deterministic_diagnostics"]["hard_errors"]
                        ),
                    }
                )
        self.state["stage_progress"]["task_realizer"] = True
        self._save()

    # ---------- technical verification and repairs ----------

    def _technical_prompt_neighbors(
        self, record: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return self._nearest_references(
            record["architecture"]["intended_rule"], count=3
        )

    def _technical_call(
        self,
        record: dict[str, Any],
        suffix: str,
        *,
        budget_kind: str,
    ) -> dict[str, Any]:
        candidate_id = record["candidate_id"]
        verifier_id = f"technical-{suffix}"
        public = solver_facing_task(record["complete_task"])
        # This is author-aware but expected-output blind: public contains only
        # training outputs and test inputs.
        assert_no_hidden_test_leakage(public)
        return self._call(
            prompt_name="technical_verifier",
            role="technical_verifier",
            call_id=f"{candidate_id}-{suffix}",
            values={
                "TASK_ID": candidate_id,
                "INTENDED_RULE": record["architecture"]["intended_rule"],
                "ARCHITECTURE": record["architecture"],
                "PUBLIC_TASK": public,
                "DIAGNOSTICS": record["deterministic_diagnostics"],
                "NEAREST_SUMMARIES": self._technical_prompt_neighbors(record),
                "VERIFIER_ID": verifier_id,
            },
            parent_ids=[candidate_id],
            budget_kind=budget_kind,
            expected={"task_id": candidate_id, "verifier_id": verifier_id},
            pre_editorial_anchor_blind=True,
        )

    @staticmethod
    def _mandatory_flags(record: dict[str, Any]) -> set[str]:
        return {
            flag["code"]
            for flag in record["deterministic_diagnostics"]["flags"]
            if flag["code"] in PAIR_ECONOMY_FLAG_CODES
        }

    def _technical_evaluation(
        self, record: dict[str, Any], response: dict[str, Any]
    ) -> dict[str, Any]:
        hidden = record["realization"]["authoring_metadata"][
            "expected_test_outputs"
        ]
        output_match = technical_output_matches_hidden(response, hidden)
        by_code = {
            item["flag_code"]: item
            for item in response["diagnostic_flag_assessments"]
        }
        mandatory = self._mandatory_flags(record)
        unjustified = sorted(
            code
            for code in mandatory
            if code not in by_code
            or not by_code[code]["justified"]
            or not by_code[code]["evidentially_necessary"]
        )
        passed = bool(
            response["status"] == "pass"
            and response["mechanically_correct"]
            and response["test_supported"]
            and response["pair_economy"] != "fail"
            and response["composition_preserved"]
            and not response["arbitrary_exceptions"]
            and not response["unsupported_test_semantics"]
            and output_match
            and not unjustified
        )
        return {
            "passed": passed,
            "hidden_output_match": output_match,
            "unjustified_pair_economy_flags": unjustified,
            "response": response,
        }

    def _repair(
        self,
        record: dict[str, Any],
        *,
        repair_type: str,
        instruction: str,
        evidence: Any,
    ) -> bool:
        if record.get("repair_count", 0) >= 1:
            return False
        candidate_id = record["candidate_id"]
        prior_public = deepcopy(record["realization"]["task"])
        prior_expected = deepcopy(
            record["realization"]["authoring_metadata"][
                "expected_test_outputs"
            ]
        )
        result = self._call(
            prompt_name="targeted_repair",
            role="targeted_repair",
            call_id=f"{candidate_id}-{repair_type}-repair",
            values={
                "CANDIDATE_ID": candidate_id,
                "REPAIR_TYPE": repair_type,
                "REPAIR_INSTRUCTION": instruction,
                "AUTHOR_SPECIFICATION": {
                    "kernel": {
                        "central_relation": record["central_relation"],
                        "why_coherent": record["why_coherent"],
                    },
                    "editorial_brief": record["editorial_brief"],
                    "architecture": record["architecture"],
                },
                "AUTHORED_TASK": record["realization"],
                "EVALUATOR_EVIDENCE": evidence,
            },
            parent_ids=[candidate_id],
            budget_kind="escalation",
            expected={
                "candidate_id": candidate_id,
                "repair_type": repair_type,
            },
        )
        record["repair_count"] = record.get("repair_count", 0) + 1
        record.setdefault("repairs", []).append(result)
        if result["status"] != "repaired":
            return False
        repaired = result["repaired_task"]
        if repaired is None:
            raise LeanPipelineError("repaired status returned null task")
        self._validate_realizer_response(
            repaired,
            record["architecture"],
            candidate_id,
            allow_one_added_training_pair=(repair_type == "ambiguity"),
        )
        if repair_type == "ambiguity":
            if repaired["task"]["test"] != prior_public["test"]:
                raise LeanPipelineError("ambiguity repair changed the test input")
            if (
                repaired["authoring_metadata"]["expected_test_outputs"]
                != prior_expected
            ):
                raise LeanPipelineError(
                    "ambiguity repair changed the hidden expected test output"
                )
            old_train = prior_public["train"]
            new_train = repaired["task"]["train"]
            valid_delta = (
                len(new_train) == len(old_train) + 1
                and new_train[:-1] == old_train
            ) or (
                len(new_train) == len(old_train)
                and sum(
                    left != right
                    for left, right in zip(old_train, new_train, strict=True)
                )
                == 1
            )
            if not valid_delta:
                raise LeanPipelineError(
                    "ambiguity repair must add or replace exactly one training pair"
                )
        self._store_realization(record, repaired, repaired=True)
        return bool(record["deterministic_diagnostics"]["hard_valid"])

    def technical_verification(self) -> None:
        if self.state["stage_progress"].get("technical_verifier"):
            return
        candidates = sorted(
            (
                record
                for record in self.state["records"].values()
                if record["status"] == "deterministic_pass"
            ),
            key=lambda item: item["editorial_brief"]["development_priority"],
        )

        def verify(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            return (
                record["candidate_id"],
                self._technical_call(record, "technical-primary", budget_kind="normal"),
            )

        initial = parallel_map(
            verify,
            candidates,
            min(len(candidates) or 1, int(self.config["model"]["concurrency"])),
        )
        for candidate_id, response in initial:
            record = self.state["records"][candidate_id]
            record["technical_results"] = [
                self._technical_evaluation(record, response)
            ]

        # Conditional work is priority ordered so the global five-call cap is
        # spent on the strongest candidates first.
        for record in candidates:
            evaluation = record["technical_results"][-1]
            response = evaluation["response"]
            if evaluation["passed"] and response["confidence"] != "low":
                record["status"] = "technical_pass"
                continue

            needs_repair = response["status"] == "repairable"
            if needs_repair and self._remaining_escalation() >= 2:
                repaired = self._repair(
                    record,
                    repair_type="mechanical",
                    instruction=response["repair_instruction"],
                    evidence=evaluation,
                )
                if repaired:
                    fresh = self._technical_call(
                        record,
                        "technical-after-repair",
                        budget_kind="escalation",
                    )
                    fresh_eval = self._technical_evaluation(record, fresh)
                    record["technical_results"].append(fresh_eval)
                    if fresh_eval["passed"] and fresh["confidence"] != "low":
                        record["status"] = "technical_pass"
                        continue
            elif (
                response["status"] in {"uncertain", "pass"}
                or response["confidence"] == "low"
                or not evaluation["hidden_output_match"]
            ) and self._remaining_escalation() >= 1:
                second = self._technical_call(
                    record,
                    "technical-second",
                    budget_kind="escalation",
                )
                second_eval = self._technical_evaluation(record, second)
                record["technical_results"].append(second_eval)
                if second_eval["passed"] and second["confidence"] != "low":
                    record["status"] = "technical_pass"
                    continue

            record["status"] = "rejected"
            reasons: list[str] = []
            final = record["technical_results"][-1]
            if not final["hidden_output_match"]:
                reasons.append(
                    "technical verifier's independently derived test output "
                    "did not match the hidden author output"
                )
            reasons.extend(final["unjustified_pair_economy_flags"])
            reasons.extend(final["response"]["arbitrary_exceptions"])
            reasons.extend(final["response"]["unsupported_test_semantics"])
            if not reasons:
                reasons.append(
                    final["response"].get("repair_instruction")
                    or f"technical status {final['response']['status']} did not resolve"
                )
            record["rejections"].append(
                {
                    "stage": "technical_verifier",
                    "category": "invalidity_or_authorship",
                    "reason": "; ".join(reasons),
                }
            )
        self.state["stage_progress"]["technical_verifier"] = True
        self._save()

    # ---------- blind ambiguity path ----------

    def _blind_call(
        self,
        record: dict[str, Any],
        suffix: str,
        *,
        budget_kind: str,
    ) -> dict[str, Any]:
        public = solver_facing_task(record["complete_task"])
        assert_no_hidden_test_leakage(public)
        task_id = "blind-" + content_hash(
            {"seed": self.seed, "candidate_id": record["candidate_id"]}
        )[:12]
        solver_id = f"blind-{suffix}"
        return self._call(
            prompt_name="blind_solver",
            role="blind_solver",
            call_id=f"{record['candidate_id']}-{suffix}",
            values={
                "TASK_ID": task_id,
                "PUBLIC_TASK": public,
                "SOLVER_ID": solver_id,
            },
            parent_ids=[record["candidate_id"]],
            budget_kind=budget_kind,
            expected={"task_id": task_id, "solver_id": solver_id},
        )

    def _blind_wrapper(
        self, record: dict[str, Any], response: dict[str, Any]
    ) -> dict[str, Any]:
        expected = record["realization"]["authoring_metadata"][
            "expected_test_outputs"
        ]
        serious = serious_blind_alternative(response, expected)
        return {
            "response": response,
            "primary_matches_hidden": response["predicted_test_outputs"] == expected,
            "serious_different_output_alternative": serious,
        }

    def _ambiguity_adjudication(
        self, record: dict[str, Any]
    ) -> dict[str, Any]:
        return self._call(
            prompt_name="ambiguity_adjudicator",
            role="ambiguity_adjudicator",
            call_id=f"{record['candidate_id']}-ambiguity-adjudication",
            values={
                "TASK_ID": record["candidate_id"],
                "AUTHOR_SPECIFICATION": {
                    "intended_rule": record["architecture"]["intended_rule"],
                    "architecture": record["architecture"],
                },
                "COMPLETE_TASK": record["complete_task"],
                "BLIND_RESULTS": record["blind_results"],
            },
            parent_ids=[record["candidate_id"]],
            budget_kind="escalation",
            expected={"task_id": record["candidate_id"]},
        )

    def _ambiguity_repair_and_revalidate(
        self, record: dict[str, Any], instruction: str
    ) -> bool:
        if self._remaining_escalation() < 3 or record.get("repair_count", 0):
            return False
        if not self._repair(
            record,
            repair_type="ambiguity",
            instruction=instruction,
            evidence=record["blind_results"],
        ):
            return False
        technical = self._technical_call(
            record, "technical-after-ambiguity-repair", budget_kind="escalation"
        )
        technical_eval = self._technical_evaluation(record, technical)
        record.setdefault("technical_results", []).append(technical_eval)
        if not technical_eval["passed"] or technical["confidence"] == "low":
            return False
        blind = self._blind_call(
            record, "blind-after-ambiguity-repair", budget_kind="escalation"
        )
        wrapped = self._blind_wrapper(record, blind)
        record.setdefault("blind_results", []).append(wrapped)
        return not wrapped["serious_different_output_alternative"]

    def blind_verification(self) -> None:
        if self.state["stage_progress"].get("blind_solver"):
            return
        candidates = sorted(
            (
                record
                for record in self.state["records"].values()
                if record["status"] == "technical_pass"
            ),
            key=lambda item: item["editorial_brief"]["development_priority"],
        )

        def solve(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            return (
                record["candidate_id"],
                self._blind_call(record, "blind-primary", budget_kind="normal"),
            )

        initial = parallel_map(
            solve,
            candidates,
            min(len(candidates) or 1, int(self.config["model"]["concurrency"])),
        )
        for candidate_id, response in initial:
            record = self.state["records"][candidate_id]
            record["blind_results"] = [self._blind_wrapper(record, response)]

        for record in candidates:
            first = record["blind_results"][0]
            if not first["serious_different_output_alternative"]:
                record["status"] = "blind_pass"
                record["ambiguity_resolution"] = {
                    "one_answer_supported": True,
                    "reason": (
                        "the blind probe found no comparably simple, well-supported "
                        "different-output rule"
                    ),
                }
                continue
            if self._remaining_escalation() < 1:
                record["status"] = "ambiguous_reserve"
                record["final_reason"] = (
                    "serious blind alternative remained when the batch escalation "
                    "budget was exhausted"
                )
                continue
            second_response = self._blind_call(
                record, "blind-second", budget_kind="escalation"
            )
            second = self._blind_wrapper(record, second_response)
            record["blind_results"].append(second)
            if second["serious_different_output_alternative"]:
                alternatives = [
                    item["response"]["strongest_plausible_alternatives"]
                    for item in record["blind_results"]
                ]
                instruction = (
                    "Add or replace one training pair that directly discriminates "
                    "the intended rule from the independently identified serious "
                    f"different-output alternatives: {json.dumps(alternatives, ensure_ascii=False)}"
                )
                if self._ambiguity_repair_and_revalidate(record, instruction):
                    record["status"] = "blind_pass"
                    record["ambiguity_resolution"] = {
                        "one_answer_supported": True,
                        "reason": "one targeted evidence repair removed the serious alternative",
                    }
                else:
                    record["status"] = "ambiguous_reserve"
                    record["final_reason"] = (
                        "two blind solvers independently identified serious "
                        "different-output alternatives that were not resolved "
                        "within one repair and the batch escalation budget"
                    )
                continue

            # Material disagreement: invoke the one adjudicator allowed by the
            # conditional path.
            if self._remaining_escalation() < 1:
                record["status"] = "ambiguous_reserve"
                record["final_reason"] = (
                    "blind results materially disagreed and no escalation call remained"
                )
                continue
            adjudication = self._ambiguity_adjudication(record)
            record["ambiguity_adjudication"] = adjudication
            if adjudication["decision"] == "clear":
                record["status"] = "blind_pass"
                record["ambiguity_resolution"] = {
                    "one_answer_supported": True,
                    "reason": adjudication["reason"],
                }
            elif (
                adjudication["decision"] == "repairable"
                and self._ambiguity_repair_and_revalidate(
                    record, adjudication["repair_instruction"]
                )
            ):
                record["status"] = "blind_pass"
                record["ambiguity_resolution"] = {
                    "one_answer_supported": True,
                    "reason": "adjudicated ambiguity was removed by one evidence repair",
                }
            else:
                record["status"] = "ambiguous_reserve"
                record["final_reason"] = adjudication["reason"]
        self.state["stage_progress"]["blind_solver"] = True
        self._save()

    # ---------- editorial boards and finalization ----------

    def _board_package(
        self, board_index: int, generated: Sequence[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        items: list[tuple[str, str, dict[str, Any]]] = []
        for anchor in self.anchors:
            items.append(
                (
                    f"anchor:{anchor.task_id}",
                    anchor.rule_description,
                    {"train": list(anchor.train), "test": list(anchor.test)},
                )
            )
        for record in generated:
            items.append(
                (
                    f"generated:{record['candidate_id']}",
                    record["architecture"]["intended_rule"],
                    record["complete_task"],
                )
            )
        Random(self.seed + 7000 + board_index).shuffle(items)
        mapping: dict[str, str] = {}
        package: list[dict[str, Any]] = []
        for index, (source_id, rule, task) in enumerate(items, 1):
            opaque = f"panel-{board_index + 1}-{index:02d}"
            mapping[opaque] = source_id
            package.append(
                {
                    "task_id": opaque,
                    "rule_summary": rule,
                    "episode": task,
                }
            )
        return package, mapping

    def editorial_boards(self) -> None:
        if self.state["stage_progress"].get("editorial_board"):
            return
        generated = sorted(
            (
                record
                for record in self.state["records"].values()
                if record["status"] == "blind_pass"
            ),
            key=lambda item: item["editorial_brief"]["development_priority"],
        )
        if not generated:
            self.state["editorial_boards"] = []
            self.state["stage_progress"]["editorial_board"] = True
            self._save()
            return
        preparations = [
            self._board_package(index, generated) for index in range(2)
        ]

        def judge(spec: tuple[int, list[dict[str, Any]], dict[str, str]]) -> dict[str, Any]:
            index, package, mapping = spec
            board_id = f"editorial-board-{index + 1}"
            expected_ids = {item["task_id"] for item in package}

            def validate(value: dict[str, Any]) -> None:
                actual = {item["task_id"] for item in value["judgments"]}
                if actual != expected_ids or len(value["judgments"]) != len(expected_ids):
                    raise LeanPipelineError(
                        f"{board_id} must judge every opaque task exactly once"
                    )

            response = self._call(
                prompt_name="editorial_board",
                role="editorial_board",
                call_id=board_id,
                values={
                    "BLINDED_TASKS": package,
                    "PORTFOLIO_DIGEST": portfolio_digest(
                        self.legacy_archive, self.lean_archive
                    ),
                    "BOARD_ID": board_id,
                },
                parent_ids=[record["candidate_id"] for record in generated],
                expected={"board_id": board_id},
                response_validator=validate,
            )
            return {"response": response, "opaque_mapping": mapping}

        boards = parallel_map(
            judge,
            [
                (index, package, mapping)
                for index, (package, mapping) in enumerate(preparations)
            ],
            min(2, int(self.config["model"]["concurrency"])),
        )
        self.state["editorial_boards"] = boards
        for record in generated:
            judgments: list[dict[str, Any]] = []
            source_id = f"generated:{record['candidate_id']}"
            for board in boards:
                opaque = next(
                    opaque
                    for opaque, source in board["opaque_mapping"].items()
                    if source == source_id
                )
                judgment = next(
                    item
                    for item in board["response"]["judgments"]
                    if item["task_id"] == opaque
                )
                judgments.append({**deepcopy(judgment), "task_id": record["candidate_id"]})
            record["editorial_judgments"] = judgments
            status, reason = classify_editorial_results(judgments)
            record["status"] = status
            record["final_reason"] = reason
            if status == "rejected":
                record["rejections"].append(
                    {
                        "stage": "editorial_board",
                        "category": "quality_novelty_or_composition",
                        "reason": reason,
                    }
                )
        self.state["stage_progress"]["editorial_board"] = True
        self._save()

    def _write_exports(self) -> None:
        root = self.run_dir / "exports"
        for name in ("core", "reserve", "ambiguous_reserve"):
            (root / name).mkdir(parents=True, exist_ok=True)
        rejected = root / "rejected_summaries"
        rejected.mkdir(parents=True, exist_ok=True)
        for record in self.state["records"].values():
            status = record["status"]
            if status in {"core", "reserve", "ambiguous_reserve"} and record.get(
                "complete_task"
            ):
                atomic_write_json(
                    root / status / f"{record['candidate_id']}.json",
                    record["complete_task"],
                )
            if status == "rejected":
                atomic_write_json(
                    rejected / f"{record['candidate_id']}.json",
                    {
                        "candidate_id": record["candidate_id"],
                        "realized": "complete_task" in record,
                        "rejections": record.get("rejections", []),
                    },
                )

    def _update_concept_archive(self) -> None:
        entries: list[dict[str, Any]] = []
        for record in self.state["records"].values():
            status = record["status"]
            assessment = record.get("selector_assessment", {})
            strong_duplicate = (
                status == "rejected"
                and assessment.get("derivative_of")
                in {
                    "previous_generated",
                    "same_batch",
                    "portfolio_metaphor",
                }
                and assessment.get("novel_concept_potential") in {"solid", "standout"}
                and assessment.get("functional_composition_potential")
                in {"solid", "standout"}
            )
            if status not in {
                "core",
                "reserve",
                "ambiguous_reserve",
                "architecture_reserve",
            } and not strong_duplicate:
                continue
            entries.append(
                {
                    "reference_id": f"lean:{self.run_id}:{record['candidate_id']}",
                    "source_kind": (
                        "strong_duplicate"
                        if strong_duplicate
                        else (
                            "architecture_reserve"
                            if status == "architecture_reserve"
                            else status
                        )
                    ),
                    "source_run": str(self.run_dir.resolve()),
                    "candidate_id": record["candidate_id"],
                    "central_relation": record["central_relation"],
                    "intended_rule": record.get("architecture", {}).get(
                        "intended_rule", ""
                    ),
                    "status": status,
                    "portfolio_note": record.get("final_reason", "")
                    or assessment.get("reason", ""),
                }
            )
        archive = append_lean_concepts(
            self.config["paths"]["lean_concept_archive"], entries
        )
        self.state["concept_archive_update"] = {
            "path": self.config["paths"]["lean_concept_archive"],
            "entries_considered": len(entries),
            "archive_entry_count": len(archive["entries"]),
            "archive_hash": content_hash(archive),
        }

    # ---------- non-gating legacy audit ----------

    def _legacy_prompt(
        self, filename: str, values: dict[str, Any]
    ) -> tuple[str, str, str]:
        text, version, digest = load_prompt(
            Path(self.config["paths"]["prompts"]) / filename
        )
        return render_prompt(text, values), version, digest

    def legacy_audit(self) -> None:
        if self.state["stage_progress"].get("legacy_audit"):
            return
        if not self.config["run"].get("legacy_audit", True):
            self.state["legacy_audit"] = {
                "status": "disabled",
                "gating": False,
                "additional_call_count": 0,
            }
            self.state["stage_progress"]["legacy_audit"] = True
            self._save()
            return
        core = sorted(
            (
                record
                for record in self.state["records"].values()
                if record["status"] == "core"
            ),
            key=lambda item: item["candidate_id"],
        )
        if not core:
            self.state["legacy_audit"] = {
                "status": "skipped_no_core_task",
                "gating": False,
                "additional_call_count": 0,
                "material_issue": None,
            }
            self.state["stage_progress"]["legacy_audit"] = True
            self._save()
            return
        record = Random(self.seed + 9901).choice(core)
        audit_dir = self.run_dir / "legacy_audit"
        invoker = ModelInvoker(
            self.config,
            audit_dir,
            output_schemas=LEGACY_SCHEMAS,
            output_parser=parse_legacy_stage_output,
        )
        calls: list[dict[str, Any]] = []

        def invoke(
            stage: str,
            call_id: str,
            filename: str,
            values: dict[str, Any],
            expected: dict[str, Any],
        ) -> dict[str, Any]:
            prompt, version, digest = self._legacy_prompt(filename, values)

            def validate(response: dict[str, Any]) -> None:
                for key, expected_value in expected.items():
                    if response.get(key) != expected_value:
                        raise LeanPipelineError(
                            f"legacy audit response changed {key}"
                        )

            result = invoker.invoke(
                stage=stage,
                call_id=call_id,
                prompt=prompt,
                prompt_template_version=version,
                prompt_template_hash=digest,
                parent_ids=[record["candidate_id"]],
                response_validator=validate,
            )
            calls.append(
                {
                    "stage": stage,
                    "call_id": call_id,
                    "attempts": result.attempts,
                    "cache_source": result.cache_source,
                    "log_dir": str(result.log_dir),
                }
            )
            return result.value

        try:
            def validator(index: int) -> dict[str, Any]:
                validator_id = f"legacy-validator-{index + 1}"
                return invoke(
                    "rule_aware_validator",
                    f"{record['candidate_id']}-{validator_id}",
                    "rule_aware_validator.md",
                    {
                        "VALIDATOR_ID": validator_id,
                        "CANDIDATE_ID": record["candidate_id"],
                        "INTENDED_RULE_JSON": record["architecture"]["intended_rule"],
                        "BLUEPRINT_JSON": record["architecture"],
                        "COMPLETE_TASK_JSON": record["complete_task"],
                    },
                    {
                        "candidate_id": record["candidate_id"],
                        "validator_id": validator_id,
                    },
                )

            validators = parallel_map(
                validator,
                range(2),
                min(2, int(self.config["model"]["concurrency"])),
            )
            public = solver_facing_task(record["complete_task"])
            opaque = "legacy-blind-" + content_hash(record["candidate_id"])[:10]

            def blind(index: int) -> dict[str, Any]:
                solver_id = f"legacy-blind-{index + 1}"
                return invoke(
                    "blind_solver",
                    f"{record['candidate_id']}-{solver_id}",
                    "blind_solver.md",
                    {
                        "SOLVER_ID": solver_id,
                        "TASK_ID": opaque,
                        "SOLVER_TASK_JSON": public,
                    },
                    {"task_id": opaque, "solver_id": solver_id},
                )

            blinds = parallel_map(
                blind,
                range(2),
                min(2, int(self.config["model"]["concurrency"])),
            )
            expected_outputs = [
                pair["output"] for pair in record["complete_task"]["test"]
            ]
            adjudication = invoke(
                "ambiguity_adjudicator",
                f"{record['candidate_id']}-legacy-ambiguity",
                "ambiguity_adjudicator.md",
                {
                    "INTENDED_TASK_JSON": {
                        "candidate_id": record["candidate_id"],
                        "intended_rule": record["architecture"]["intended_rule"],
                        "pair_blueprint": record["architecture"],
                        "complete_task": record["complete_task"],
                    },
                    "BLIND_SOLVER_RESULTS_JSON": blinds,
                },
                {"candidate_id": record["candidate_id"]},
            )
            if adjudication["intended_test_outputs"] != expected_outputs:
                raise LeanPipelineError(
                    "legacy ambiguity adjudicator changed hidden outputs"
                )
            material = (
                not all(item["valid"] for item in validators)
                or adjudication["decision"] != "pass"
                or adjudication[
                    "serious_competing_alternative_predicts_different_output"
                ]
            )
            useful = []
            if any(not item["valid"] for item in validators):
                useful.append("the second standing mechanical panel found a defect")
            if adjudication[
                "serious_competing_alternative_predicts_different_output"
            ]:
                useful.append(
                    "the standing two-solver ambiguity panel found a serious alternative"
                )
            if not useful:
                useful.append(
                    "the removed standing panels added corroboration but no material new issue"
                )
            self.state["legacy_audit"] = {
                "status": "complete",
                "gating": False,
                "audited_candidate_id": record["candidate_id"],
                "random_selection_seed": self.seed + 9901,
                "validators": validators,
                "blind_solvers": blinds,
                "ambiguity_adjudication": adjudication,
                "material_issue": material,
                "useful_information": useful,
                "calls": calls,
                "additional_call_count": len(calls),
                "additional_retry_count": sum(
                    max(0, item["attempts"] - 1) for item in calls
                ),
            }
        except Exception as exc:
            self.state["legacy_audit"] = {
                "status": "infrastructure_failure",
                "gating": False,
                "audited_candidate_id": record["candidate_id"],
                "material_issue": None,
                "failure": str(exc),
                "calls": calls,
                "additional_call_count": len(calls),
                "note": "The lean decision is unchanged by legacy-audit infrastructure failure.",
            }
        self.state["stage_progress"]["legacy_audit"] = True
        self._save()

    # ---------- completion ----------

    def run(self) -> dict[str, Any]:
        if self.state.get("completed"):
            from .lean_reporting import build_lean_report

            return build_lean_report(self.run_dir)
        self.generate_ideas()
        self.select_batch()
        self.create_architectures()
        self.realize_tasks()
        self.technical_verification()
        self.blind_verification()
        self.editorial_boards()
        self._write_exports()
        self._update_concept_archive()
        self.legacy_audit()
        self.state["completed"] = True
        self.state["completed_at_epoch_seconds"] = time.time()
        self.state["run_wall_seconds"] = round(
            self.state["completed_at_epoch_seconds"]
            - self.state["started_at_epoch_seconds"],
            3,
        )
        self.state["stage_progress"]["complete"] = True
        self._save()
        from .lean_reporting import build_lean_report

        return build_lean_report(self.run_dir)


def lean_call_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline": PIPELINE_ID,
        "normal_stage_limits": deepcopy(NORMAL_STAGE_LIMITS),
        "normal_maximum": NORMAL_CALL_LIMIT,
        "conditional_escalation_maximum": int(
            config["budget"]["escalation_call_limit"]
        ),
        "hard_lean_maximum": int(
            config["budget"]["hard_substantive_call_limit"]
        ),
        "ideas": 12,
        "architectures": 5,
        "realizations": 4,
        "difficulty_factor_stages": [],
    }


def run_lean_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    return LeanAuthoringPipeline(config).run()


__all__ = [
    "LEAN_PIPELINE_VERSION",
    "LeanAuthoringPipeline",
    "LeanPipelineError",
    "assert_no_fixed_anchor_leakage",
    "classify_editorial_results",
    "lean_call_plan",
    "run_lean_benchmark",
    "serious_blind_alternative",
    "technical_output_matches_hidden",
]
