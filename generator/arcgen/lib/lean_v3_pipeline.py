"""Lean Authoring V3: the saved V1 pipeline with restrained policy overrides."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import json
import threading
import time

from .data import TaskRecord, load_task_bank
from .embeddings import build_embedding_provider
from .lean_pipeline import (
    LeanAuthoringPipeline,
    LeanPipelineError,
    assert_no_fixed_anchor_leakage,
)
from .lean_v3_archive import (
    append_v3_concepts,
    archive_tier_report,
    build_tiered_archive_view,
    reference_records,
    tiered_portfolio_digest,
)
from .lean_v3_config import (
    NORMAL_CALL_LIMIT,
    NORMAL_STAGE_LIMITS,
    PIPELINE_ID,
    SEARCH_MODES,
    snapshot_lean_v3_config,
    verify_frozen_baselines,
)
from .lean_v3_diagnostics import (
    PAIR_ECONOMY_FLAG_CODES,
    combine_public_and_hidden,
    diagnose_arc_task_v3,
)
from .lean_v3_schemas import OUTPUT_SCHEMAS, parse_stage_output
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


LEAN_V3_PIPELINE_VERSION = "arcgen-lean-authoring-v3.0.0"
PROMPT_FILES = {
    "idea_open": "lean_v3_idea_open.md",
    "idea_concept": "lean_v3_idea_concept.md",
    "idea_composition": "lean_v3_idea_composition.md",
    "batch_selector": "lean_v3_batch_selector.md",
    "episode_architect": "lean_v3_episode_architect.md",
    "task_realizer": "lean_v3_task_realizer.md",
    "technical_verifier": "lean_v3_technical_verifier.md",
    "blind_solver": "lean_v3_blind_solver.md",
    "ambiguity_adjudicator": "lean_v3_ambiguity_adjudicator.md",
    "targeted_repair": "lean_v3_targeted_repair.md",
    "editorial_board": "lean_v3_editorial_board.md",
}
ROLE_TO_STAGE = {
    "idea_search": "lean_v3_idea_search",
    "batch_selector": "lean_v3_batch_selector",
    "episode_architect": "lean_v3_episode_architect",
    "task_realizer": "lean_v3_task_realizer",
    "technical_verifier": "lean_v3_technical_verifier",
    "blind_solver": "lean_v3_blind_solver",
    "ambiguity_adjudicator": "lean_v3_ambiguity_adjudicator",
    "targeted_repair": "lean_v3_targeted_repair",
    "editorial_board": "lean_v3_editorial_board",
}


class LeanV3PipelineError(LeanPipelineError):
    pass


class LeanV3PromptStore:
    def __init__(self, prompt_dir: Path, run_dir: Path) -> None:
        self.entries: dict[str, dict[str, str]] = {}
        destination = run_dir / "prompts" / "templates"
        destination.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {"pipeline": PIPELINE_ID, "templates": {}}
        for name, filename in PROMPT_FILES.items():
            path = prompt_dir / filename
            if not path.is_file():
                raise LeanV3PipelineError(f"missing Lean V3 prompt: {path}")
            text, version, digest = load_prompt(path)
            self.entries[name] = {
                "text": text,
                "version": version,
                "hash": digest,
                "filename": filename,
            }
            snapshot = destination / filename
            if snapshot.exists() and snapshot.read_text(encoding="utf-8") != text:
                raise LeanV3PipelineError(
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
            raise LeanV3PipelineError("V3 prompt manifest changed within existing run")
        if not manifest_path.exists():
            atomic_write_json(manifest_path, manifest)

    def render(self, name: str, values: dict[str, Any]) -> tuple[str, str, str]:
        entry = self.entries[name]
        required = set(PLACEHOLDER_RE.findall(entry["text"]))
        if required != set(values):
            raise LeanV3PipelineError(
                f"{name} prompt bindings differ: missing="
                f"{sorted(required-set(values))}, extra={sorted(set(values)-required)}"
            )
        return (
            render_prompt(entry["text"], values),
            entry["version"],
            entry["hash"],
        )


def classify_v3_editorial_results(
    judgments: Sequence[dict[str, Any]],
) -> tuple[str, str, str, str]:
    """Return independent intrinsic-quality and portfolio-placement decisions."""

    if len(judgments) != 2:
        raise LeanV3PipelineError("exactly two editorial judgments are required")

    if any(item["novel_concept"] == "weak" for item in judgments):
        intrinsic = "rejected_quality"
        intrinsic_reason = "an editorial board rated the concept weak"
    elif any(item["functional_composition"] == "thin" for item in judgments):
        intrinsic = "rejected_quality"
        intrinsic_reason = "an editorial board rated functional composition thin"
    elif any(item["overall_quality"] == "below_real_bar" for item in judgments):
        intrinsic = "rejected_quality"
        intrinsic_reason = "an editorial board rated quality below the real-task bar"
    elif any(
        item["intrinsic_recommendation"] == "rejected_quality"
        for item in judgments
    ):
        intrinsic = "rejected_quality"
        intrinsic_reason = "an editorial board explicitly rejected intrinsic quality"
    else:
        standout_each = all(
            item["novel_concept"] == "standout"
            or item["functional_composition"] == "standout"
            for item in judgments
        )
        quality_each = all(
            item["overall_quality"] in {"real_quality", "exceptional"}
            for item in judgments
        )
        core_recommended = all(
            item["intrinsic_recommendation"] == "core_quality"
            for item in judgments
        )
        if standout_each and quality_each and core_recommended:
            intrinsic = "core_quality"
            intrinsic_reason = (
                "both boards cleared both axes, found a standout axis, rated "
                "real quality, and recommended intrinsic core quality"
            )
        else:
            intrinsic = "reserve_quality"
            intrinsic_reason = (
                "the task is intrinsically usable, but the boards did not "
                "unanimously clear the V3 core-quality rule"
            )

    same_procedure = any(
        item["same_minimal_procedure"]
        or item["portfolio_recommendation"] == "same_procedure_duplicate"
        for item in judgments
    )
    if same_procedure:
        placement = "same_procedure_duplicate"
        placement_reason = (
            "at least one board identified the same minimal solution procedure"
        )
    elif any(
        item["addition_is_excessive"]
        or item["portfolio_recommendation"] == "hold_crowded"
        for item in judgments
    ):
        placement = "hold_crowded"
        placement_reason = (
            "the task retains its intrinsic rating but is held as an excessive "
            "addition to the current portfolio"
        )
    elif intrinsic == "rejected_quality":
        placement = "not_applicable"
        placement_reason = "portfolio placement is not applicable to rejected work"
    else:
        placement = "include_now"
        placement_reason = "no same-procedure or excessive-addition concern was found"
    return intrinsic, placement, intrinsic_reason, placement_reason


class LeanV3AuthoringPipeline(LeanAuthoringPipeline):
    """A minimal subclass of the actual frozen V1 execution engine."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config["seed"])
        self.run_id = str(config["run"]["id"])
        self.run_dir = Path(config["paths"]["runs"]) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self._state_lock = threading.RLock()

        if config["palette"]["enabled"]:
            raise LeanV3PipelineError(
                "V3 palette intervention is intentionally deferred because the "
                "V1 realizer has no structured color-role contract"
            )
        self.baseline_verification = verify_frozen_baselines(
            config["_repository_root"],
            config["paths"]["baseline_hash_manifest"],
        )
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

        current_view = build_tiered_archive_view(
            legacy_path=config["paths"]["legacy_concept_archive"],
            v1_path=config["paths"]["v1_concept_archive"],
            v2_path=config["paths"]["v2_concept_archive"],
            v3_path=config["paths"]["v3_concept_archive"],
        )
        archive_snapshot = self.run_dir / "data" / "tiered_archive.start.json"
        if archive_snapshot.exists():
            self.archive_view = read_json(archive_snapshot)
        else:
            self.archive_view = current_view
            atomic_write_json(archive_snapshot, self.archive_view)
        self.references = reference_records(
            self.bank,
            self.archive_view,
            excluded_real_ids=self.anchor_ids,
        )

        self.state = self._load_or_initialize_state()
        self.prompts = LeanV3PromptStore(
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

    # ---------- V3 immutable preparation ----------

    def _snapshot_implementation(self) -> str:
        package = Path(__file__).resolve().parent
        names = {
            "lean_v3_pipeline.py",
            "lean_v3_config.py",
            "lean_v3_schemas.py",
            "lean_v3_diagnostics.py",
            "lean_v3_archive.py",
            "lean_v3_reporting.py",
            # The execution inheritance is explicit and fingerprinted.
            "lean_pipeline.py",
            "lean_diagnostics.py",
            "lean_schemas.py",
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
            "pipeline_version": LEAN_V3_PIPELINE_VERSION,
            "branch_source": "arcgen-lean-authoring-v1.0.1",
            "baseline_hash_manifest_sha256": self.baseline_verification[
                "manifest_sha256"
            ],
            "files": files,
        }
        digest = content_hash(manifest)
        manifest["implementation_hash"] = digest
        path = self.run_dir / "implementation_manifest.json"
        if path.exists() and read_json(path) != manifest:
            raise LeanV3PipelineError(
                "V3 implementation changed within an existing run"
            )
        if not path.exists():
            atomic_write_json(path, manifest)
        return digest

    def _snapshot_config(self) -> None:
        snapshot = snapshot_lean_v3_config(self.config)
        path = self.run_dir / "config.snapshot.json"
        if path.exists() and read_json(path) != snapshot:
            raise LeanV3PipelineError(
                f"run {self.run_id!r} has a different configuration snapshot"
            )
        if not path.exists():
            atomic_write_json(path, snapshot)
            atomic_write_text(
                self.run_dir / "config.snapshot.yaml",
                json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            )

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = read_json(self.state_path)
            if (
                state.get("pipeline_version") != LEAN_V3_PIPELINE_VERSION
                or state.get("seed") != self.seed
                or state.get("implementation_hash") != self.implementation_hash
            ):
                raise LeanV3PipelineError("existing V3 run is stale or incompatible")
            return state
        state = {
            "pipeline_version": LEAN_V3_PIPELINE_VERSION,
            "branch_source": "arcgen-lean-authoring-v1.0.1",
            "run_id": self.run_id,
            "seed": self.seed,
            "implementation_hash": self.implementation_hash,
            "baseline_verification": self.baseline_verification,
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
            "special_generation_lanes_active": False,
            "palette_policy": deepcopy(self.config["palette"]),
            "implementation_limitations": [
                "The fixed six-anchor V1 panel remains provisional.",
                "The pilot is one batch and cannot establish distributional recovery by itself.",
                "Embedding retrieval scouts candidates but never decides similarity.",
                "Palette intervention is deferred to preserve the exact V1 realizer contract.",
                "Optional toolbox use is not separately judged or counted.",
                "Only one targeted repair is permitted per realized task.",
            ],
        }
        atomic_write_json(self.state_path, state)
        return state

    def _write_data_artifacts(self) -> None:
        data = self.run_dir / "data"
        data.mkdir(parents=True, exist_ok=True)
        tier_report = archive_tier_report(self.archive_view)
        artifacts = {
            "generator_visible_pool.json": {
                "seed": self.seed,
                "task_ids": [item.task_id for item in self.generator_pool],
                "fixed_anchor_ids_excluded": sorted(self.anchor_ids),
            },
            "fixed_anchor_panel.json": self.anchor_manifest,
            "archive_tiers.start.json": tier_report,
            "reference_archive_manifest.json": {
                "real_reference_count": sum(
                    item["source"] == "real_task" for item in self.references
                ),
                "generated_reference_count": sum(
                    item["source"] == "previous_generated"
                    for item in self.references
                ),
                "generated_reference_counts_by_tier": dict(
                    Counter(
                        item["archive_tier"]
                        for item in self.references
                        if item["source"] == "previous_generated"
                    )
                ),
                "tiered_archive_hash_at_start": content_hash(self.archive_view),
                "fixed_anchor_ids_excluded": sorted(self.anchor_ids),
            },
            "bank_manifest.json": {
                "task_count": len(self.bank),
                "logical_hash": content_hash([item.to_dict() for item in self.bank]),
            },
            "baseline_verification.json": self.baseline_verification,
            "palette_policy.json": {
                **deepcopy(self.config["palette"]),
                "decision": (
                    "disabled: changing the V1 realizer schema merely to carry "
                    "color-role metadata would be wider than this quality branch"
                ),
            },
        }
        for filename, value in artifacts.items():
            path = data / filename
            if path.exists() and read_json(path) != value:
                raise LeanV3PipelineError(
                    f"source or selection changed within run: {filename}"
                )
            if not path.exists():
                atomic_write_json(path, value)

    # ---------- V3 invocation boundary ----------

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
                    raise LeanV3PipelineError(
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

    # ---------- Tier-aware selector ----------

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
                "archive_tier": hit.reference["archive_tier"],
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
                    self.config["experiment"]["selector_references_per_candidate"]
                ),
            )
            selections[idea["candidate_id"]] = neighbors
            packages.append({"kernel": idea, "nearest_references": neighbors})
        digest = tiered_portfolio_digest(self.archive_view)
        candidate_ids = {item["candidate_id"] for item in self.state["ideas"]}

        def validate(value: dict[str, Any]) -> None:
            assessed = {
                item["candidate_id"] for item in value["candidate_assessments"]
            }
            if assessed != candidate_ids:
                raise LeanV3PipelineError(
                    "selector must assess exactly all 12 candidates"
                )
            selected = {item["candidate_id"] for item in value["selected"]}
            if not selected <= candidate_ids:
                raise LeanV3PipelineError("selector selected an unknown candidate")
            decisions = {
                item["candidate_id"]: item["decision"]
                for item in value["candidate_assessments"]
            }
            if selected != {
                candidate_id
                for candidate_id, decision in decisions.items()
                if decision == "select"
            }:
                raise LeanV3PipelineError(
                    "selector assessments and editorial briefs disagree"
                )
            suggestion_count = sum(
                item["optional_development_suggestion"] is not None
                for item in value["selected"]
            )
            maximum = int(
                self.config["experiment"][
                    "maximum_optional_development_suggestions"
                ]
            )
            if suggestion_count > maximum:
                raise LeanV3PipelineError(
                    f"selector returned {suggestion_count} optional suggestions; "
                    f"the configured maximum is {maximum}"
                )
            for item in value["candidate_assessments"]:
                if (
                    item["similarity_scope"] == "same_minimal_procedure"
                    and item["derivative_of"] == "none"
                ):
                    raise LeanV3PipelineError(
                        "same-procedure assessment must identify a derivative source"
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
                            "same_procedure_duplicate"
                            if assessment["similarity_scope"]
                            == "same_minimal_procedure"
                            else (
                                "novelty_or_development_potential"
                                if assessment["derivative_of"] != "none"
                                else "development_potential"
                            )
                        ),
                        "reason": assessment["reason"],
                    }
                )
        self.state["stage_progress"]["batch_selector"] = True
        self._save()

    # ---------- V1 authorship and verification with refined diagnostics ----------

    def _store_realization(
        self,
        record: dict[str, Any],
        realization: dict[str, Any],
        *,
        repaired: bool = False,
    ) -> None:
        metadata = realization["authoring_metadata"]
        complete = combine_public_and_hidden(
            realization["task"], metadata["expected_test_outputs"]
        )
        diagnostics = diagnose_arc_task_v3(complete)
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
                    "pipeline_version": LEAN_V3_PIPELINE_VERSION,
                    "branch_source": "arcgen-lean-authoring-v1.0.1",
                    "run_id": self.run_id,
                    "candidate_id": record["candidate_id"],
                    "search_call_id": record["search_call_id"],
                    "search_mode": record["search_mode"],
                },
            },
        )
        atomic_write_json(task_dir / f"{record['candidate_id']}.arc.json", complete)

    @staticmethod
    def _mandatory_flags(record: dict[str, Any]) -> set[str]:
        return {
            flag["code"]
            for flag in record["deterministic_diagnostics"]["flags"]
            if flag["code"] in PAIR_ECONOMY_FLAG_CODES
        }

    # ---------- Split editorial result ----------

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
        if generated:
            preparations = [
                self._board_package(index, generated) for index in range(2)
            ]

            def judge(
                spec: tuple[int, list[dict[str, Any]], dict[str, str]]
            ) -> dict[str, Any]:
                index, package, mapping = spec
                board_id = f"editorial-board-{index + 1}"
                expected_ids = {item["task_id"] for item in package}

                def validate(value: dict[str, Any]) -> None:
                    actual = {item["task_id"] for item in value["judgments"]}
                    if actual != expected_ids or len(value["judgments"]) != len(
                        expected_ids
                    ):
                        raise LeanV3PipelineError(
                            f"{board_id} must judge every opaque task exactly once"
                        )

                response = self._call(
                    prompt_name="editorial_board",
                    role="editorial_board",
                    call_id=board_id,
                    values={
                        "BLINDED_TASKS": package,
                        "PORTFOLIO_DIGEST": tiered_portfolio_digest(
                            self.archive_view
                        ),
                        "BOARD_ID": board_id,
                    },
                    parent_ids=[
                        record["candidate_id"] for record in generated
                    ],
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
        else:
            boards = []
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
                judgments.append(
                    {**deepcopy(judgment), "task_id": record["candidate_id"]}
                )
            record["editorial_judgments"] = judgments
            (
                intrinsic,
                placement,
                intrinsic_reason,
                placement_reason,
            ) = classify_v3_editorial_results(judgments)
            record["intrinsic_quality_status"] = intrinsic
            record["portfolio_placement_status"] = placement
            record["intrinsic_quality_reason"] = intrinsic_reason
            record["portfolio_placement_reason"] = placement_reason
            record["status"] = intrinsic
            record["final_reason"] = (
                f"Intrinsic: {intrinsic_reason}. Portfolio: {placement_reason}."
            )
            if intrinsic == "rejected_quality":
                record["rejections"].append(
                    {
                        "stage": "editorial_board",
                        "category": "intrinsic_quality",
                        "reason": intrinsic_reason,
                    }
                )

        # Preserve the V1 ambiguity path while expressing its terminal result
        # in the separate V3 status vocabulary.
        for record in self.state["records"].values():
            if record["status"] == "ambiguous_reserve":
                record["intrinsic_quality_status"] = "ambiguous"
                record["portfolio_placement_status"] = "not_applicable"
                record["intrinsic_quality_reason"] = record.get(
                    "final_reason",
                    "a serious different-output interpretation remains unresolved",
                )
                record["portfolio_placement_reason"] = (
                    "ambiguous tasks do not enter the headline training portfolio"
                )
                record["status"] = "ambiguous"
            elif record["status"] == "rejected" and record.get("complete_task"):
                record["intrinsic_quality_status"] = "rejected_quality"
                record["portfolio_placement_status"] = "not_applicable"
                record["intrinsic_quality_reason"] = (
                    record.get("rejections", [{}])[-1].get(
                        "reason", "failed a pre-editorial quality or validity gate"
                    )
                )
                record["portfolio_placement_reason"] = (
                    "portfolio placement is not applicable to rejected work"
                )
                record["status"] = "rejected_quality"
        self.state["stage_progress"]["editorial_board"] = True
        self._save()

    # ---------- V3 exports and moving archive ----------

    def _write_exports(self) -> None:
        root = self.run_dir / "exports"
        names = (
            "core",
            "core_quality",
            "reserve_quality",
            "portfolio_hold",
            "ambiguous",
            "rejected_realized",
            "rejected_summaries",
        )
        for name in names:
            (root / name).mkdir(parents=True, exist_ok=True)
        for record in self.state["records"].values():
            candidate_id = record["candidate_id"]
            complete = record.get("complete_task")
            intrinsic = record.get("intrinsic_quality_status")
            placement = record.get("portfolio_placement_status")
            if complete is not None:
                if intrinsic == "core_quality":
                    atomic_write_json(
                        root / "core_quality" / f"{candidate_id}.json", complete
                    )
                    if placement == "include_now":
                        atomic_write_json(
                            root / "core" / f"{candidate_id}.json", complete
                        )
                    else:
                        atomic_write_json(
                            root / "portfolio_hold" / f"{candidate_id}.json",
                            complete,
                        )
                elif intrinsic == "reserve_quality":
                    atomic_write_json(
                        root / "reserve_quality" / f"{candidate_id}.json",
                        complete,
                    )
                elif intrinsic == "ambiguous":
                    atomic_write_json(
                        root / "ambiguous" / f"{candidate_id}.json", complete
                    )
                elif intrinsic == "rejected_quality":
                    atomic_write_json(
                        root / "rejected_realized" / f"{candidate_id}.json",
                        complete,
                    )
            if record["status"] in {"rejected", "rejected_quality"}:
                atomic_write_json(
                    root / "rejected_summaries" / f"{candidate_id}.json",
                    {
                        "candidate_id": candidate_id,
                        "realized": complete is not None,
                        "intrinsic_quality_status": intrinsic,
                        "portfolio_placement_status": placement,
                        "rejections": record.get("rejections", []),
                    },
                )

    def _update_concept_archive(self) -> None:
        entries: list[dict[str, Any]] = []
        for record in self.state["records"].values():
            assessment = record.get("selector_assessment", {})
            status = record["status"]
            intrinsic = record.get("intrinsic_quality_status")
            placement = record.get("portfolio_placement_status")
            tier: str | None = None
            source_kind = ""
            reason = ""
            if record.get("operational_failure"):
                tier = "recoverable_work"
                source_kind = "operational_failure"
                reason = "operational failure did not receive substantive semantic review"
            elif (
                assessment.get("similarity_scope") == "same_minimal_procedure"
                or placement == "same_procedure_duplicate"
            ):
                tier = "hard_duplicate_memory"
                source_kind = "confirmed_same_procedure"
                reason = "selector or completed-grid board confirmed the same procedure"
            elif intrinsic in {"core_quality", "reserve_quality"} and record.get(
                "complete_task"
            ):
                tier = "hard_duplicate_memory"
                source_kind = intrinsic
                reason = (
                    "completed V3 task; future hard exclusion applies only to "
                    "the same minimal procedure"
                )
            elif status == "architecture_reserve":
                tier = "soft_mechanism_memory"
                source_kind = "architecture_reserve"
                reason = "strong unrealized architecture is a soft warning"
            elif intrinsic == "ambiguous":
                tier = "soft_mechanism_memory"
                source_kind = "ambiguous"
                reason = "completed ambiguous work is mechanism context, not a hard boundary"
            elif (
                assessment.get("similarity_scope")
                in {"broad_theme", "close_sibling"}
                and assessment.get("novel_concept_potential")
                in {"solid", "standout"}
                and assessment.get("functional_composition_potential")
                in {"solid", "standout"}
            ):
                tier = "soft_mechanism_memory"
                source_kind = "soft_similarity_warning"
                reason = "broad theme or close sibling is retained only as a soft warning"
            if tier is None:
                continue
            entries.append(
                {
                    "tier": tier,
                    "reference_id": f"lean-v3:{self.run_id}:{record['candidate_id']}",
                    "source_kind": source_kind,
                    "source_run": str(self.run_dir.resolve()),
                    "candidate_id": record["candidate_id"],
                    "central_relation": record["central_relation"],
                    "intended_rule": record.get("architecture", {}).get(
                        "intended_rule", ""
                    ),
                    "status": intrinsic or status,
                    "portfolio_note": record.get("final_reason", "")
                    or assessment.get("reason", ""),
                    "classification_reason": reason,
                }
            )
        archive = append_v3_concepts(
            self.config["paths"]["v3_concept_archive"], entries
        )
        self.state["concept_archive_update"] = {
            "path": self.config["paths"]["v3_concept_archive"],
            "entries_considered": len(entries),
            "entry_counts_by_tier": {
                tier: len(archive["tiers"][tier])
                for tier in archive["tiers"]
            },
            "archive_hash": content_hash(archive),
        }

    # ---------- completion ----------

__all__ = [
    "LEAN_V3_PIPELINE_VERSION",
    "LeanV3AuthoringPipeline",
    "LeanV3PipelineError",
    "classify_v3_editorial_results",
]
