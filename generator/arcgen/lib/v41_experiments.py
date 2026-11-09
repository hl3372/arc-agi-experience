"""Isolated ARCGEN V4.1 coverage experiments.

Only palette qualification is authorized in the initial V4.1 turn.  The
semantic runners share the same freezing, blinding, downstream, audit, and
report interfaces, but configuration ordering prevents them from executing
before prior user decisions are terminal.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from random import Random
from string import ascii_lowercase, digits
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json
import threading
import time

from .committed_tasks import load_committed_task_registry
from .data import TaskRecord, load_task_bank
from .lean_pipeline import serious_blind_alternative
from .lean_v3_pipeline import classify_v3_editorial_results
from .lean_v3_rendering import build_contact_sheet, render_episode
from .model import CallResult, ModelInvoker
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
from .v4_diagnostics import diagnose_arc_task_v4, diagnose_public_arc_task_v4, grid_canvas_metrics
from .v4_pipeline import v4_call_accounting
from .v41_config import EXPERIMENT_PIPELINES, snapshot_v41_config
from .v41_palette import (
    V41PaletteError,
    apply_scheduled_palette,
    background_metrics,
    infer_episode_background,
    make_episode_envelope,
    measure_background_distribution,
    measure_real_bank_backgrounds,
    palette_equivalent_hash,
    schedule_backgrounds,
    simulate_future_scheduling,
    soft_target_distribution,
    used_colors,
)
from .v41_schemas import OUTPUT_SCHEMAS, parse_stage_output


class V41ExperimentError(ValueError):
    pass


V41_PROMPTS = {
    "idea_concept": ("v4_idea_concept.md", "v4_idea_search"),
    "concrete_planner": ("v4_concrete_planner.md", "v4_concrete_plan"),
    "task_realizer": ("v4_task_realizer.md", "v4_task_realizer"),
    "technical_verifier": ("v4_technical_verifier.md", "v4_technical_verifier"),
    "blind_solver": ("v4_blind_solver.md", "v4_blind_solver"),
    "ambiguity_adjudicator": ("v4_ambiguity_adjudicator.md", "v4_ambiguity_adjudicator"),
    "targeted_repair": ("v4_targeted_repair.md", "v4_targeted_repair"),
    "editorial_board": ("v4_editorial_board.md", "v4_editorial_board"),
    "supported_addendum": ("v4_1_supported_departure_addendum.md", "v41_supported_plan"),
    "joint_idea_addendum": ("v4_1_episode_joint_addendum.md", "v4_idea_search"),
    "joint_plan_addendum": ("v4_1_episode_joint_planning_addendum.md", "v41_episode_joint_plan"),
    "abstraction_idea_addendum": ("v4_1_abstraction_led_addendum.md", "v4_idea_search"),
    "abstraction_plan_addendum": ("v4_1_abstraction_planning_addendum.md", "v41_abstraction_plan"),
    "mechanism_auditor": ("v4_1_mechanism_auditor.md", "v41_mechanism_audit"),
    "diversity_auditor": ("v4_1_diversity_auditor.md", "v41_diversity_audit"),
    "palette_equivalence": ("v4_1_palette_equivalence_review.md", "v41_palette_equivalence"),
}


class V41PromptStore:
    def __init__(self, prompt_dir: Path, run_dir: Path, pipeline: str) -> None:
        self.entries: dict[str, dict[str, str]] = {}
        destination = run_dir / "prompts" / "templates"
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {"pipeline": pipeline, "templates": {}}
        for name, (filename, stage) in V41_PROMPTS.items():
            path = prompt_dir / filename
            text, version, digest = load_prompt(path)
            self.entries[name] = {
                "filename": filename,
                "text": text,
                "version": version,
                "hash": digest,
                "stage": stage,
            }
            target = destination / filename
            if target.exists() and target.read_text(encoding="utf-8") != text:
                raise V41ExperimentError(f"prompt changed during resume: {filename}")
            if not target.exists():
                atomic_write_text(target, text)
            manifest["templates"][name] = {
                "filename": filename,
                "stage": stage,
                "version": version,
                "sha256": digest,
            }
        path = run_dir / "prompts" / "manifest.json"
        if path.exists() and read_json(path) != manifest:
            raise V41ExperimentError("V4.1 prompt manifest changed during resume")
        if not path.exists():
            atomic_write_json(path, manifest)

    def render(self, name: str, values: Mapping[str, Any]) -> tuple[str, str, str, str]:
        entry = self.entries[name]
        required = set(PLACEHOLDER_RE.findall(entry["text"]))
        if required != set(values):
            raise V41ExperimentError(
                f"{name} bindings differ: missing={sorted(required-set(values))}, extra={sorted(set(values)-required)}"
            )
        return (
            render_prompt(entry["text"], dict(values)),
            entry["version"],
            entry["hash"],
            entry["stage"],
        )


def _implementation_manifest(run_dir: Path, root: Path) -> dict[str, Any]:
    names = (
        "v41_config.py", "v41_features.py", "v41_schemas.py",
        "v41_palette.py", "v41_audits.py", "v41_experiments.py",
        "v41_reporting.py", "v41_integration.py",
        "v41_supported_departure.py",
    )
    value = {
        "pipeline_version": "arcgen-v4.1-experiments-4.1.0",
        "branch_source": "frozen qualified arcgen-v4-control-4.0.0",
        "files": {
            name: content_hash((root / "arcgen.lib" / name).read_bytes())
            for name in names if (root / "arcgen.lib" / name).is_file()
        },
    }
    path = run_dir / "implementation_manifest.json"
    if path.exists() and read_json(path) != value:
        raise V41ExperimentError("V4.1 implementation changed during run resume")
    if not path.exists():
        atomic_write_json(path, value)
    return value


class V41Runtime:
    def __init__(self, config: dict[str, Any], source_manifest: Mapping[str, Any]) -> None:
        feature = config["experimental"]["feature"]
        if not config["experimental"]["enabled"] or config["pipeline"] != EXPERIMENT_PIPELINES[feature]:
            raise V41ExperimentError("an explicit V4.1 experiment configuration is required")
        if config["experimental"]["auto_promote"] or not config["experimental"]["non_gating"]:
            raise V41ExperimentError("V4.1 experiments must be non-gating and cannot auto-promote")
        self.config = config
        self.run_id = config["run"]["id"]
        self.run_dir = Path(config["paths"]["runs"]) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.state_path = self.run_dir / "state.json"
        root = Path(config["_repository_root"])
        _implementation_manifest(self.run_dir, root)
        snapshot = snapshot_v41_config(config)
        config_path = self.run_dir / "config.snapshot.json"
        if config_path.exists() and read_json(config_path) != snapshot:
            raise V41ExperimentError("V4.1 experiment config changed during resume")
        if not config_path.exists():
            atomic_write_json(config_path, snapshot)
        frozen = deepcopy(dict(source_manifest))
        source_path = self.run_dir / "frozen_control_manifest.json"
        if source_path.exists() and read_json(source_path) != frozen:
            raise V41ExperimentError("V4.1 frozen comparison changed during resume")
        if not source_path.exists():
            atomic_write_json(source_path, frozen)
        self.prompts = V41PromptStore(Path(config["paths"]["prompts"]), self.run_dir, config["pipeline"])
        if self.state_path.exists():
            self.state = read_json(self.state_path)
        else:
            self.state = {
                "pipeline": config["pipeline"],
                "run_id": self.run_id,
                "feature": feature,
                "completed": False,
                "non_gating": True,
                "auto_promote": False,
                "production_state_mutation_allowed": False,
                "good_task_bank_mutation_allowed": False,
                "feature_manifest_mutation_allowed": False,
                "started_at_epoch_seconds": round(time.time(), 3),
                "source_manifest_hash": content_hash(frozen),
                "call_ledger": {},
                "results": {},
            }
            atomic_write_json(self.state_path, self.state)
        self.invoker = ModelInvoker(
            config,
            self.run_dir,
            output_schemas=OUTPUT_SCHEMAS,
            output_parser=parse_stage_output,
        )

    def save(self) -> None:
        with self.lock:
            atomic_write_json(self.state_path, self.state)

    def call(
        self,
        *,
        prompt_name: str,
        composed_with: Sequence[str] = (),
        call_id: str,
        values: Mapping[str, Any],
        parent_ids: Sequence[str],
        role: str,
        arm_id: str,
        expected: Mapping[str, Any] | None = None,
        validator: Callable[[dict[str, Any]], None] | None = None,
        forbidden_prompt_fragments: Sequence[str] = (),
    ) -> dict[str, Any]:
        if composed_with:
            names = (prompt_name, *composed_with)
            entries = [self.prompts.entries[name] for name in names]
            combined = "\n\n".join(entry["text"] for entry in entries)
            required = set(PLACEHOLDER_RE.findall(combined))
            if required != set(values):
                raise V41ExperimentError(
                    f"{'+'.join(names)} bindings differ: "
                    f"missing={sorted(required-set(values))}, extra={sorted(set(values)-required)}"
                )
            prompt = render_prompt(combined, dict(values))
            version = "+".join(entry["version"] for entry in entries)
            digest = content_hash(
                {"templates": [entry["hash"] for entry in entries], "combined": combined}
            )
            stage = entries[-1]["stage"]
        else:
            prompt, version, digest, stage = self.prompts.render(prompt_name, values)
        lowered = prompt.casefold()
        leaks = [item for item in forbidden_prompt_fragments if item.casefold() in lowered]
        if leaks:
            raise V41ExperimentError(f"downstream prompt leaked experimental label(s): {leaks}")

        def validate(value: dict[str, Any]) -> None:
            for key, wanted in (expected or {}).items():
                if value.get(key) != wanted:
                    raise V41ExperimentError(f"{call_id} returned wrong {key}")
            if validator:
                validator(value)

        settings = self.config["model"]["roles"][role]
        key = f"{arm_id}/{stage}/{call_id}"
        with self.lock:
            if key not in self.state["call_ledger"]:
                self.state["call_ledger"][key] = {
                    "arm_id": arm_id,
                    "stage": stage,
                    "role": role,
                    "call_id": call_id,
                    "model": settings["model"],
                    "reasoning_effort": settings["reasoning_effort"],
                    "status": "reserved",
                    "substantive_experimental_call": True,
                }
                self.save()
        try:
            result: CallResult = self.invoker.invoke(
                stage=stage,
                call_id=call_id,
                prompt=prompt,
                prompt_template_version=version,
                prompt_template_hash=digest,
                parent_ids=list(parent_ids),
                response_validator=validate,
                model_name=settings["model"],
                reasoning_effort=settings["reasoning_effort"],
            )
        except Exception as exc:
            with self.lock:
                self.state["call_ledger"][key].update({"status": "failed", "error": str(exc)})
                self.save()
            raise
        timing = read_json(result.log_dir / "result.json").get("timing")
        with self.lock:
            self.state["call_ledger"][key].update(
                {
                    "status": "complete",
                    "attempts": result.attempts,
                    "retry_calls": max(0, result.attempts - 1),
                    "cache_source": result.cache_source,
                    "cache_key": result.cache_key,
                    "log_dir": str(result.log_dir),
                    "timing": timing,
                }
            )
            self.save()
        return result.value

    def accounting(self) -> dict[str, Any]:
        overall = v4_call_accounting(self.state["call_ledger"], run_dir=self.run_dir)
        by_arm = {}
        for arm in sorted({item["arm_id"] for item in self.state["call_ledger"].values()}):
            ledger = {
                key: item for key, item in self.state["call_ledger"].items()
                if item["arm_id"] == arm
            }
            by_arm[arm] = v4_call_accounting(ledger, run_dir=self.run_dir)
        return {
            "production_calls": 0,
            "experimental_calls": overall["substantive_logical_calls"],
            "active_model_minutes": overall["active_model_minutes"],
            "transport_retry_calls": overall["transport_retry_calls"],
            "recorded_model_attempts": overall["recorded_model_attempts"],
            "measured_peak_concurrent_calls": overall["measured_peak_concurrent_calls"],
            "measured_effective_concurrency": overall["measured_effective_concurrency"],
            "by_arm": by_arm,
        }

    def finish(self, report: Mapping[str, Any], markdown: str) -> dict[str, Any]:
        accounting = self.accounting()
        supplied_accounting = deepcopy(dict(report)).get("call_accounting")
        completed = {
            **deepcopy(dict(report)),
            "call_accounting": supplied_accounting or accounting,
            "runtime_call_accounting": accounting,
        }
        with self.lock:
            self.state["completed"] = True
            self.state["completed_at_epoch_seconds"] = round(time.time(), 3)
            self.state["results"] = completed
            self.state["call_accounting"] = accounting
            self.save()
        atomic_write_json(self.run_dir / "report.json", completed)
        atomic_write_text(self.run_dir / "report.md", markdown)
        return completed


def _anonymous_id(rng: Random, used: set[str], length: int = 12) -> str:
    alphabet = ascii_lowercase + digits
    while True:
        value = "".join(rng.choice(alphabet) for _ in range(length))
        if value not in used:
            used.add(value)
            return value


def _task_metrics(task: Mapping[str, Any]) -> dict[str, Any]:
    grids = list(
        pair[field]
        for split in ("train", "test")
        for pair in task[split]
        for field in ("input", "output")
        if field in pair
    )
    colors = {cell for grid in grids for row in grid for cell in row}
    canvas = [grid_canvas_metrics(grid) for grid in grids]
    return {
        "maximum_dimension": max(item["maximum_dimension"] for item in canvas),
        "maximum_area": max(item["area"] for item in canvas),
        "mean_non_background_occupancy": round(
            sum(item["non_background_occupancy"] for item in canvas) / len(canvas), 6
        ),
        "color_count": len(colors),
        "training_pair_count": len(task["train"]),
        "test_pair_count": len(task["test"]),
    }


def _v4_realized_sources(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    directory = Path(config["paths"]["frozen_v4_qualification"])
    state = read_json(directory / "state.json")
    sources = []
    for candidate_id, record in state["records"].items():
        if not record.get("complete_task"):
            continue
        sources.append(
            {
                "task_id": candidate_id,
                "source": "v4_qualification",
                "complete_task": deepcopy(record["complete_task"]),
                "intended_rule": record["realization"]["authoring_metadata"]["intended_rule"],
                "concise_rule": record["central_relation"],
                "concrete_plan": deepcopy(record["concrete_plan"]),
                "search_mode": record["search_mode"],
                "intrinsic_quality": record.get("intrinsic_quality"),
                "portfolio_placement": record.get("portfolio_placement"),
                "source_path": str((directory / "tasks" / f"{candidate_id}.arc.json").resolve()),
                "metrics": _task_metrics(record["complete_task"]),
                "prior_blind_completed": bool(record.get("blind_results")),
            }
        )
    return sorted(sources, key=lambda item: item["task_id"])


def _committed_sources(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry_path = Path(config["paths"]["committed_task_registry"])
    registry = load_committed_task_registry(registry_path)
    base = registry_path.parent
    sources = []
    for entry in registry["entries"]:
        path = base / entry["task_path"]
        task = read_json(path)
        sources.append(
            {
                "task_id": entry["registry_task_id"],
                "source": "committed_good_task_bank",
                "complete_task": task,
                "intended_rule": entry["intended_rule"],
                "concise_rule": entry["concept_summary"],
                "concrete_plan": None,
                "search_mode": entry["search_mode"],
                "intrinsic_quality": entry["intrinsic_status"],
                "portfolio_placement": entry["portfolio_status"],
                "source_path": str(path.resolve()),
                "metrics": _task_metrics(task),
                "prior_blind_completed": False,
            }
        )
    return sorted(sources, key=lambda item: item["task_id"])


def _representative_committed(
    sources: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
    modal_fraction: float,
    require_consistent_input_mode: bool,
) -> list[dict[str, Any]]:
    if count > len(sources):
        raise V41ExperimentError("not enough committed tasks for palette sample")
    pool = [deepcopy(dict(item)) for item in sources]
    for item in pool:
        item["inferred_background"] = infer_episode_background(
            item["complete_task"],
            modal_fraction=modal_fraction,
            require_consistent_input_mode=require_consistent_input_mode,
        )
    selected: list[dict[str, Any]] = []
    unresolved = [item for item in pool if item["inferred_background"] is None]
    if unresolved:
        selected.append(min(unresolved, key=lambda item: content_hash({"seed": seed, "id": item["task_id"]})))
    remaining = [item for item in pool if item["task_id"] not in {entry["task_id"] for entry in selected}]
    # Explicitly cover the cases the palette study needs: a saturated palette,
    # large and compact episodes with at least one unused target color, and a
    # sparse episode.  This avoids accidentally choosing only saturated tasks
    # while remaining deterministic and quality-neutral.
    strata = (
        (
            lambda item: item["inferred_background"] is not None
            and item["metrics"]["color_count"] == 10,
            lambda item: (item["task_id"]),
        ),
        (
            lambda item: item["inferred_background"] is not None
            and item["metrics"]["color_count"] < 10,
            lambda item: (-item["metrics"]["maximum_dimension"], item["task_id"]),
        ),
        (
            lambda item: item["inferred_background"] is not None
            and item["metrics"]["color_count"] < 10,
            lambda item: (item["metrics"]["maximum_dimension"], item["task_id"]),
        ),
        (
            lambda item: item["inferred_background"] is not None
            and item["metrics"]["color_count"] < 10,
            lambda item: (item["metrics"]["mean_non_background_occupancy"], item["task_id"]),
        ),
    )
    for predicate, criterion in strata:
        if len(selected) >= count:
            break
        candidates = [item for item in remaining if predicate(item)]
        if not candidates:
            continue
        candidate = min(candidates, key=criterion)
        selected.append(candidate)
        remaining = [item for item in remaining if item["task_id"] != candidate["task_id"]]
    while len(selected) < count:
        candidate = min(remaining, key=lambda item: content_hash({"seed": seed, "id": item["task_id"]}))
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def palette_replay_sources(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    palette = config["palette_experiment"]
    v4 = _v4_realized_sources(config)
    target = int(palette["replay_sample_size"])
    committed_count = target - len(v4)
    if committed_count < 1:
        return v4[:target]
    committed = _representative_committed(
        _committed_sources(config),
        count=committed_count,
        seed=int(config["seed"]),
        modal_fraction=float(palette["conservative_modal_fraction"]),
        require_consistent_input_mode=bool(palette["require_consistent_input_mode"]),
    )
    return [*v4, *committed]


def _source_manifest(config: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    root = Path(config["_repository_root"])
    anchors = read_json(Path(config["paths"]["fixed_anchor_manifest"]))
    feature_manifest = deepcopy(config["_feature_manifest_snapshot"])
    return {
        "schema_version": 1,
        "feature": config["experimental"]["feature"],
        "assignment_seed": config["seed"],
        "frozen_v4_parent": deepcopy(config["_v4_parent_verification"]),
        "frozen_v4_control_config": config["_v4_control_config"],
        "v4_control_config_sha256": hashlib.sha256(Path(config["_v4_control_config"]).read_bytes()).hexdigest(),
        "accepted_task_bank_snapshot": {
            "path": config["paths"]["committed_task_registry"],
            "sha256": hashlib.sha256(Path(config["paths"]["committed_task_registry"]).read_bytes()).hexdigest(),
        },
        "rejected_memory_snapshot": {
            "path": config["paths"]["reservoir"],
            "sha256": hashlib.sha256(Path(config["paths"]["reservoir"]).read_bytes()).hexdigest(),
        },
        "in_flight_reservation_snapshot": read_json(Path(config["paths"]["reservoir"])).get("active_reservations", []),
        "feature_manifest_snapshot": feature_manifest,
        "fixed_anchor_manifest": anchors,
        "model_and_effort_settings": deepcopy(config["model"]["roles"]),
        "calibration_configuration": {
            "version": config["reservoir"]["calibration_version"],
            "generator_visible_pool_count": config["reservoir"]["generator_visible_pool_count"],
        },
        "sample": [
            {
                "task_id": item["task_id"],
                "source": item["source"],
                "source_path": item["source_path"],
                "complete_task_sha256": content_hash(item["complete_task"]),
                "palette_equivalent_identity_hash": palette_equivalent_hash(item["complete_task"]),
                "search_mode": item["search_mode"],
                "metrics": item["metrics"],
            }
            for item in sources
        ],
        "repository_root": str(root),
    }


def _distribution_from_sources(
    sources: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    palette = config["palette_experiment"]
    return measure_background_distribution(
        ((item["task_id"], item["complete_task"]) for item in sources),
        modal_fraction=float(palette["conservative_modal_fraction"]),
        require_consistent_input_mode=bool(palette["require_consistent_input_mode"]),
    )


def _diverse_v4_subset(
    sources: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    pool = [deepcopy(dict(item)) for item in sources if item["source"] == "v4_qualification" and item["prior_blind_completed"]]
    if len(pool) < count:
        raise V41ExperimentError("not enough fully reviewed V4 tasks for paired model checks")
    # Seed with a deterministic maximum-size item, then maximize normalized
    # distance from the selected set.  This is descriptive sampling, not a
    # task-quality decision.
    selected = [
        min(
            pool,
            key=lambda item: (
                -item["metrics"]["maximum_dimension"],
                content_hash({"seed": seed, "id": item["task_id"]}),
            ),
        )
    ]
    pool = [item for item in pool if item["task_id"] != selected[0]["task_id"]]

    def vector(item: Mapping[str, Any]) -> tuple[float, ...]:
        metrics = item["metrics"]
        return (
            metrics["maximum_dimension"] / 30,
            metrics["color_count"] / 10,
            metrics["mean_non_background_occupancy"],
            metrics["training_pair_count"] / 10,
        )

    while len(selected) < count:
        def score(item: Mapping[str, Any]) -> tuple[float, str]:
            candidate = vector(item)
            nearest = min(
                sum((a - b) ** 2 for a, b in zip(candidate, vector(other)))
                for other in selected
            )
            return nearest, content_hash({"seed": seed, "id": item["task_id"]})

        choice = max(pool, key=score)
        selected.append(choice)
        pool = [item for item in pool if item["task_id"] != choice["task_id"]]
    return selected


def _build_palette_variants(
    runtime: V41Runtime,
    sources: Sequence[Mapping[str, Any]],
    bank: Sequence[TaskRecord],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = runtime.config
    palette = config["palette_experiment"]
    real_distribution = measure_real_bank_backgrounds(
        bank,
        modal_fraction=float(palette["conservative_modal_fraction"]),
        require_consistent_input_mode=bool(palette["require_consistent_input_mode"]),
    )
    target = soft_target_distribution(
        real_distribution,
        allowed_colors=palette["allowed_background_colors"],
        real_reference_weight=float(palette["real_reference_weight"]),
        diversity_weight=float(palette["diversity_weight"]),
    )
    committed = _committed_sources(config)
    committed_distribution = _distribution_from_sources(committed, config)
    envelopes = {
        item["task_id"]: make_episode_envelope(
            task_id=item["task_id"],
            complete_task=item["complete_task"],
            intended_rule=item["intended_rule"],
            concrete_plan=item["concrete_plan"],
            source=item["source"],
            modal_fraction=float(palette["conservative_modal_fraction"]),
            require_consistent_input_mode=bool(palette["require_consistent_input_mode"]),
        )
        for item in sources
    }
    ordered_ids = sorted(
        envelopes,
        key=lambda task_id: content_hash({"seed": config["seed"], "task_id": task_id}),
    )
    assignments = schedule_backgrounds(
        ordered_ids,
        seed=int(config["seed"]),
        current_counts=committed_distribution["counts"],
        target_distribution=target,
        eligible={task_id: bool(envelopes[task_id]["palette_permutation_safe"]) for task_id in envelopes},
        allowed_targets_by_task={
            task_id: sorted(
                {
                    int(envelopes[task_id]["background_color"]),
                    *(set(range(10)) - used_colors(envelopes[task_id]["complete_task"])),
                }
            )
            if envelopes[task_id]["palette_permutation_safe"]
            and envelopes[task_id]["background_color"] is not None
            else []
            for task_id in envelopes
        },
    )
    control = {
        task_id: apply_scheduled_palette(envelopes[task_id], envelopes[task_id]["background_color"])
        for task_id in envelopes
    }
    treatment = {
        task_id: apply_scheduled_palette(envelopes[task_id], assignments[task_id])
        for task_id in envelopes
    }
    original_dir = runtime.run_dir / "tasks" / "control"
    treatment_dir = runtime.run_dir / "tasks" / "treatment"
    envelope_dir = runtime.run_dir / "palette_envelopes"
    for directory in (original_dir, treatment_dir, envelope_dir):
        directory.mkdir(parents=True, exist_ok=True)
    deterministic = []
    for task_id in ordered_ids:
        original = control[task_id]
        mapped = treatment[task_id]
        original_path = original_dir / f"{task_id}.json"
        mapped_path = treatment_dir / f"{task_id}.json"
        atomic_write_json(original_path, original["complete_task"])
        atomic_write_json(mapped_path, mapped["complete_task"])
        atomic_write_json(envelope_dir / f"{task_id}.control.json", original)
        atomic_write_json(envelope_dir / f"{task_id}.treatment.json", mapped)
        serialized_exact = (
            read_json(original_path) == original["complete_task"]
            and read_json(mapped_path) == mapped["complete_task"]
        )
        mapping = mapped["palette_treatment"]["mapping"]
        values = {int(key): int(value) for key, value in mapping.items()}
        bijective = set(values) == set(range(10)) and set(values.values()) == set(range(10))
        same_identity = (
            original["palette_equivalent_identity_hash"]
            == mapped["palette_equivalent_identity_hash"]
        )
        metadata_consistent = (
            mapped["background_color"] == mapped["palette_treatment"]["realized_background_color"]
            and all(0 <= int(role["color"]) <= 9 for role in mapped["color_roles"])
        )
        control_diagnostics = diagnose_arc_task_v4(original["complete_task"])
        treatment_diagnostics = diagnose_arc_task_v4(mapped["complete_task"])
        deterministic.append(
            {
                "task_id": task_id,
                "source": envelopes[task_id]["source"],
                "safe": mapped["palette_permutation_safe"],
                "requested_background": assignments[task_id],
                "original_background": envelopes[task_id]["background_color"],
                "realized_background": mapped["background_color"],
                "mapping": mapping,
                "globally_bijective": bijective,
                "inverse_round_trip_exact": mapped["palette_treatment"]["inverse_round_trip_exact"],
                "serialization_exact": serialized_exact,
                "metadata_consistent": metadata_consistent,
                "palette_equivalent_identity_preserved": same_identity,
                "control_hard_valid": control_diagnostics["hard_valid"],
                "treatment_hard_valid": treatment_diagnostics["hard_valid"],
                "control_diagnostics": control_diagnostics,
                "treatment_diagnostics": treatment_diagnostics,
                "control_exact_hash": original["exact_task_hash"],
                "treatment_exact_hash": mapped["exact_task_hash"],
                "palette_equivalent_identity_hash": mapped["palette_equivalent_identity_hash"],
                "opt_out_reason": None if mapped["palette_permutation_safe"] else mapped["safety_assessment"]["reasons"],
            }
        )
    assignment_ledger = {
        "seed": config["seed"],
        "policy": "largest soft-target deficit with deterministic content-hash tie break",
        "task_order": ordered_ids,
        "assignments": assignments,
        "target_distribution": {str(key): round(value, 8) for key, value in target.items()},
        "starting_committed_distribution": committed_distribution,
        "arm_labels_exposed_downstream": False,
        "pairwise_recoloring_allowed": False,
        "black_remained_eligible": 0 in target,
    }
    atomic_write_json(runtime.run_dir / "assignment_ledger.json", assignment_ledger)
    atomic_write_json(runtime.run_dir / "deterministic_replay_report.json", {"tasks": deterministic})
    atomic_write_json(
        runtime.run_dir / "control_manifest.json",
        {
            "arm": "control",
            "task_ids": ordered_ids,
            "tasks": {
                task_id: {
                    "exact_hash": control[task_id]["exact_task_hash"],
                    "palette_equivalent_identity_hash": control[task_id]["palette_equivalent_identity_hash"],
                }
                for task_id in ordered_ids
            },
        },
    )
    atomic_write_json(
        runtime.run_dir / "treatment_manifest.json",
        {
            "arm": "treatment",
            "task_ids": ordered_ids,
            "tasks": {
                task_id: {
                    "exact_hash": treatment[task_id]["exact_task_hash"],
                    "palette_equivalent_identity_hash": treatment[task_id]["palette_equivalent_identity_hash"],
                    "mapping": treatment[task_id]["palette_treatment"]["mapping"],
                }
                for task_id in ordered_ids
            },
        },
    )
    reference = {
        "real_bank": real_distribution,
        "soft_target": assignment_ledger["target_distribution"],
        "committed_bank": committed_distribution,
    }
    atomic_write_json(runtime.run_dir / "palette_reference.json", reference)
    return control, treatment, {"assignment_ledger": assignment_ledger, "deterministic": deterministic, "reference": reference}


def _opaque_variant_ids(
    task_ids: Sequence[str], *, seed: int, namespace: str
) -> dict[tuple[str, str], str]:
    rng = Random(seed + int(content_hash(namespace)[:12], 16))
    used: set[str] = set()
    keys = [(task_id, arm) for task_id in task_ids for arm in ("control", "treatment")]
    rng.shuffle(keys)
    return {key: _anonymous_id(rng, used) for key in keys}


def _technical_and_blind_review(
    runtime: V41Runtime,
    sources: Sequence[Mapping[str, Any]],
    control: Mapping[str, Mapping[str, Any]],
    treatment: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    count = max(
        int(runtime.config["palette_experiment"]["technical_subset_size"]),
        int(runtime.config["palette_experiment"]["blind_subset_size"]),
    )
    reviewed_count = sum(
        item["source"] == "v4_qualification" and item["prior_blind_completed"]
        for item in sources
    )
    ordered = _diverse_v4_subset(
        sources, reviewed_count, int(runtime.config["seed"])
    )
    # A paired subset must actually exercise the treatment.  Preserve the
    # deterministic diversity order within each stratum, but include all
    # available non-identity V4 variants before identity/opt-out controls.
    changed = [
        item for item in ordered
        if control[item["task_id"]]["exact_task_hash"]
        != treatment[item["task_id"]]["exact_task_hash"]
    ]
    unchanged = [
        item for item in ordered
        if control[item["task_id"]]["exact_task_hash"]
        == treatment[item["task_id"]]["exact_task_hash"]
    ]
    selected = [*changed, *unchanged][:count]
    selected_ids = [item["task_id"] for item in selected]
    aliases = _opaque_variant_ids(selected_ids, seed=int(runtime.config["seed"]), namespace="technical-blind")
    source_by_id = {item["task_id"]: item for item in sources}
    forbidden = ("palette-treatment", "palette-control", "scheduled-palette", "experimental arm")

    def technical(spec: tuple[str, str]) -> dict[str, Any]:
        task_id, arm = spec
        opaque = aliases[(task_id, arm)]
        envelope = control[task_id] if arm == "control" else treatment[task_id]
        verifier_id = "verifier-" + content_hash({"opaque": opaque, "seed": runtime.config["seed"]})[:10]
        plan = envelope.get("effective_concrete_plan") or source_by_id[task_id]["concrete_plan"]
        rule = envelope.get("effective_intended_rule") or source_by_id[task_id]["intended_rule"]
        try:
            response = runtime.call(
                prompt_name="technical_verifier",
                call_id=f"tech-{opaque}",
                values={
                    "INTENDED_RULE": rule,
                    "CONCRETE_PLAN": plan,
                    "PUBLIC_TASK": envelope["public_task"],
                    "DIAGNOSTICS": diagnose_public_arc_task_v4(envelope["public_task"]),
                    "TASK_ID": opaque,
                    "VERIFIER_ID": verifier_id,
                },
                parent_ids=[opaque],
                role="technical_verifier",
                arm_id=f"palette-{arm}",
                expected={"task_id": opaque, "verifier_id": verifier_id},
                forbidden_prompt_fragments=forbidden,
            )
            hidden_match = response["derived_test_outputs"] == envelope["hidden_test_outputs"]
            passed = (
                response["status"] == "pass"
                and response["mechanically_correct"]
                and response["test_supported"]
                and response["one_answer_supported"]
                and not response["arbitrary_exceptions"]
                and not response["unsupported_test_semantics"]
                and hidden_match
            )
            return {"task_id": task_id, "opaque_id": opaque, "arm": arm, "status": "complete", "response": response, "hidden_output_exact_match": hidden_match, "passed": passed}
        except Exception as exc:
            return {"task_id": task_id, "opaque_id": opaque, "arm": arm, "status": "operational_failure", "error": str(exc), "hidden_output_exact_match": False, "passed": False}

    def blind(spec: tuple[str, str]) -> dict[str, Any]:
        task_id, arm = spec
        opaque = aliases[(task_id, arm)]
        envelope = control[task_id] if arm == "control" else treatment[task_id]
        solver_id = "solver-" + content_hash({"opaque": opaque, "seed": runtime.config["seed"]})[:10]
        try:
            response = runtime.call(
                prompt_name="blind_solver",
                call_id=f"blind-{opaque}",
                values={"PUBLIC_TASK": envelope["public_task"], "TASK_ID": opaque, "SOLVER_ID": solver_id},
                parent_ids=[opaque],
                role="blind_solver",
                arm_id=f"palette-{arm}",
                expected={"task_id": opaque, "solver_id": solver_id},
                forbidden_prompt_fragments=forbidden,
            )
            serious = serious_blind_alternative(response, envelope["hidden_test_outputs"])
            return {
                "task_id": task_id,
                "opaque_id": opaque,
                "arm": arm,
                "status": "complete",
                "response": response,
                "primary_matches_hidden": response["predicted_test_outputs"] == envelope["hidden_test_outputs"],
                "serious_different_output_alternative": serious,
            }
        except Exception as exc:
            return {"task_id": task_id, "opaque_id": opaque, "arm": arm, "status": "operational_failure", "error": str(exc), "primary_matches_hidden": False, "serious_different_output_alternative": None}

    technical_ids = selected_ids[: int(runtime.config["palette_experiment"]["technical_subset_size"])]
    blind_ids = selected_ids[: int(runtime.config["palette_experiment"]["blind_subset_size"])]
    technical_specs = [(task_id, arm) for task_id in technical_ids for arm in ("control", "treatment")]
    blind_specs = [(task_id, arm) for task_id in blind_ids for arm in ("control", "treatment")]
    with ThreadPoolExecutor(max_workers=int(runtime.config["model"]["concurrency"])) as executor:
        technical_futures = [executor.submit(technical, spec) for spec in technical_specs]
        blind_futures = [executor.submit(blind, spec) for spec in blind_specs]
        technical_results = [future.result() for future in technical_futures]
        blind_results = [future.result() for future in blind_futures]
    value = {
        "selection_method": "deterministic diversity sample from fully reviewed V4 qualification tasks",
        "selected_task_ids": selected_ids,
        "opaque_aliases": [
            {"task_id": task_id, "arm": arm, "opaque_id": alias}
            for (task_id, arm), alias in sorted(aliases.items())
        ],
        "arm_hidden_from_prompts": True,
        "technical": technical_results,
        "blind": blind_results,
    }
    atomic_write_json(runtime.run_dir / "verification" / "paired_technical_blind.json", value)
    return value


def _anchors(config: Mapping[str, Any]) -> list[TaskRecord]:
    bank = load_task_bank(config["paths"]["annotations"], config["paths"]["tasks"])
    by_id = {record.task_id: record for record in bank}
    manifest = read_json(Path(config["paths"]["fixed_anchor_manifest"]))
    ids = [item["task_id"] for item in manifest["anchors"]]
    if len(ids) != 6 or any(task_id not in by_id for task_id in ids):
        raise V41ExperimentError("frozen six-anchor panel is unavailable")
    return [by_id[task_id] for task_id in ids]


def _portfolio_digest(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = load_committed_task_registry(config["paths"]["committed_task_registry"])
    return {
        "explicitly_committed_task_count": len(registry["entries"]),
        "committed_concepts": [item["concept_summary"] for item in registry["entries"]],
        "theme_commonality_is_descriptive_only": True,
    }


def _editorial_review(
    runtime: V41Runtime,
    sources: Sequence[Mapping[str, Any]],
    control: Mapping[str, Mapping[str, Any]],
    treatment: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected = _diverse_v4_subset(
        sources,
        int(runtime.config["palette_experiment"]["editorial_subset_size"]),
        int(runtime.config["seed"]) + 100,
    )
    task_ids = [item["task_id"] for item in selected]
    source_by_id = {item["task_id"]: item for item in sources}
    anchors = _anchors(runtime.config)
    portfolio = _portfolio_digest(runtime.config)
    forbidden = ("palette-treatment", "palette-control", "scheduled-palette", "experimental arm")
    private_maps: dict[str, Any] = {}

    def board(spec: tuple[str, int]) -> dict[str, Any]:
        arm, board_index = spec
        seed = int(runtime.config["seed"]) + 2000 + board_index * 41 + (0 if arm == "control" else 17)
        rng = Random(seed)
        used: set[str] = set()
        rows: list[tuple[str, str, dict[str, Any]]] = []
        for anchor in anchors:
            rows.append(
                (
                    f"anchor:{anchor.task_id}",
                    anchor.rule_description,
                    {"train": list(anchor.train), "test": list(anchor.test)},
                )
            )
        variants = control if arm == "control" else treatment
        for task_id in task_ids:
            rows.append(
                (
                    f"sample:{task_id}",
                    variants[task_id].get("effective_intended_rule")
                    or source_by_id[task_id]["concise_rule"],
                    variants[task_id]["complete_task"],
                )
            )
        rng.shuffle(rows)
        package = []
        mapping = {}
        for source_id, rule, task in rows:
            opaque = _anonymous_id(rng, used)
            mapping[opaque] = source_id
            package.append({"task_id": opaque, "rule_summary": rule, "episode": task})
        board_id = "board-" + content_hash({"seed": seed, "items": sorted(mapping)})[:12]
        expected_ids = set(mapping)

        def validate(value: dict[str, Any]) -> None:
            returned = [item["task_id"] for item in value["judgments"]]
            if len(returned) != len(set(returned)) or set(returned) != expected_ids:
                raise V41ExperimentError("editorial board did not judge every opaque task")

        try:
            response = runtime.call(
                prompt_name="editorial_board",
                call_id=board_id,
                values={"BLINDED_TASKS": package, "PORTFOLIO_DIGEST": portfolio, "BOARD_ID": board_id},
                parent_ids=list(expected_ids),
                role="editorial_board",
                arm_id=f"palette-{arm}",
                expected={"board_id": board_id},
                validator=validate,
                forbidden_prompt_fragments=forbidden,
            )
            result = {"arm": arm, "board_index": board_index, "status": "complete", "board_id": board_id, "response": response, "opaque_mapping": mapping}
        except Exception as exc:
            result = {"arm": arm, "board_index": board_index, "status": "operational_failure", "board_id": board_id, "error": str(exc), "opaque_mapping": mapping}
        return result

    specs = [(arm, index) for arm in ("control", "treatment") for index in (1, 2)]
    boards = parallel_map(board, specs, min(4, int(runtime.config["model"]["concurrency"])))
    for item in boards:
        private_maps[item["board_id"]] = item["opaque_mapping"]
    comparisons = []
    for task_id in task_ids:
        arms = {}
        for arm in ("control", "treatment"):
            judgments = []
            for result in boards:
                if result["arm"] != arm or result["status"] != "complete":
                    continue
                opaque = next(key for key, value in result["opaque_mapping"].items() if value == f"sample:{task_id}")
                judgment = next(item for item in result["response"]["judgments"] if item["task_id"] == opaque)
                judgments.append({**deepcopy(judgment), "task_id": task_id})
            if len(judgments) == 2:
                intrinsic, placement, intrinsic_reason, placement_reason = classify_v3_editorial_results(judgments)
                arms[arm] = {
                    "status": "complete",
                    "judgments": judgments,
                    "intrinsic_quality": intrinsic,
                    "portfolio_placement": placement,
                    "reason": intrinsic_reason + "; " + placement_reason,
                }
            else:
                arms[arm] = {"status": "incomplete", "judgments": judgments}
        comparisons.append({"task_id": task_id, "arms": arms})
    value = {
        "selected_task_ids": task_ids,
        "arm_hidden_from_boards": True,
        "fixed_anchor_ids": [item.task_id for item in anchors],
        "boards": boards,
        "task_comparisons": comparisons,
    }
    atomic_write_json(runtime.run_dir / "editorial" / "paired_editorial_review.json", value)
    atomic_write_json(runtime.run_dir / "editorial" / "private_opaque_mapping.json", private_maps)
    return value


def _palette_equivalence_review(
    runtime: V41Runtime,
    task_ids: Sequence[str],
    control: Mapping[str, Mapping[str, Any]],
    treatment: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rng = Random(int(runtime.config["seed"]) + 4100)
    used: set[str] = set()
    pair_specs = []
    private = []
    for task_id in sorted(task_ids, key=lambda item: content_hash({"seed": runtime.config["seed"], "id": item})):
        pair_id = _anonymous_id(rng, used)
        first = _anonymous_id(rng, used)
        second = _anonymous_id(rng, used)
        treatment_first = bool(rng.randrange(2))
        task_a = treatment[task_id] if treatment_first else control[task_id]
        task_b = control[task_id] if treatment_first else treatment[task_id]
        pair_specs.append(
            {
                "pair_id": pair_id,
                "task_a_id": first,
                "task_b_id": second,
                "task_a": task_a["complete_task"],
                "task_b": task_b["complete_task"],
            }
        )
        private.append(
            {
                "pair_id": pair_id,
                "source_task_id": task_id,
                "task_a_variant": "treatment" if treatment_first else "control",
                "task_b_variant": "control" if treatment_first else "treatment",
            }
        )
    chunks = [pair_specs[index : index + 6] for index in range(0, len(pair_specs), 6)]

    def review(spec: tuple[int, list[dict[str, Any]]]) -> dict[str, Any]:
        index, rows = spec
        review_id = "equivalence-" + content_hash({"seed": runtime.config["seed"], "chunk": index})[:12]
        expected = {item["pair_id"] for item in rows}

        def validate(value: dict[str, Any]) -> None:
            returned = {item["pair_id"] for item in value["judgments"]}
            if returned != expected:
                raise V41ExperimentError("palette reviewer did not cover every opaque pair")

        try:
            response = runtime.call(
                prompt_name="palette_equivalence",
                call_id=review_id,
                values={"PAIRED_TASKS": rows, "REVIEW_ID": review_id},
                parent_ids=sorted(expected),
                role="editorial_board",
                arm_id="blinded-paired-review",
                expected={"review_id": review_id},
                validator=validate,
                forbidden_prompt_fragments=("palette-treatment", "palette-control", "scheduled-palette", "experimental arm"),
            )
            return {"status": "complete", "review_id": review_id, "response": response}
        except Exception as exc:
            return {"status": "operational_failure", "review_id": review_id, "error": str(exc)}

    results = parallel_map(review, list(enumerate(chunks, 1)), min(len(chunks), int(runtime.config["model"]["concurrency"])))
    judgments = [
        judgment
        for result in results if result["status"] == "complete"
        for judgment in result["response"]["judgments"]
    ]
    value = {
        "arm_hidden": True,
        "review_calls": results,
        "judgments": judgments,
        "private_assignment": private,
        "all_pairs_judged": len(judgments) == len(task_ids),
    }
    atomic_write_json(runtime.run_dir / "mechanism_audit_report.json", value)
    return value


def _human_palette_pack(
    runtime: V41Runtime,
    sources: Sequence[Mapping[str, Any]],
    control: Mapping[str, Mapping[str, Any]],
    treatment: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    destination = runtime.run_dir / "human_review"
    tasks_dir = destination / "tasks"
    images_dir = destination / "images"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = Random(int(runtime.config["seed"]) + 5100)
    used: set[str] = set()
    source_by_id = {item["task_id"]: item for item in sources}
    public = []
    key = []
    rendered = []
    questionnaire = []
    source_ids = sorted(
        source_by_id,
        key=lambda task_id: content_hash({"seed": runtime.config["seed"], "human": task_id}),
    )
    for task_id in source_ids:
        pair_id = _anonymous_id(rng, used, length=8)
        treatment_first = bool(rng.randrange(2))
        arm_order = ("treatment", "control") if treatment_first else ("control", "treatment")
        pair_public = {"pair_id": pair_id, "variants": []}
        for label, arm in zip(("a", "b"), arm_order):
            anonymous = _anonymous_id(rng, used)
            envelope = control[task_id] if arm == "control" else treatment[task_id]
            task = envelope["complete_task"]
            atomic_write_json(tasks_dir / f"{anonymous}.json", task)
            image = render_episode(
                task,
                title=f"Pair {pair_id.upper()} — variant {label.upper()}",
                subtitle="Source, palette assignment, and author rule withheld",
                destination=images_dir / f"{anonymous}.png",
            )
            rendered.append((f"{pair_id}-{label}", image))
            pair_public["variants"].append(
                {
                    "label": label,
                    "anonymous_id": anonymous,
                    "task_json": f"tasks/{anonymous}.json",
                    "image": f"images/{anonymous}.png",
                    "task_hash": content_hash(task),
                }
            )
            key.append(
                {
                    "pair_id": pair_id,
                    "label": label,
                    "anonymous_id": anonymous,
                    "source_task_id": task_id,
                    "variant": arm,
                    "intended_rule": envelope.get("effective_intended_rule")
                    or source_by_id[task_id]["intended_rule"],
                    "background_color": envelope["background_color"],
                    "mapping": envelope["palette_treatment"]["mapping"],
                }
            )
        public.append(pair_public)
        questionnaire.append(
            {
                "pair_id": pair_id,
                "same_rule_and_task_identity": None,
                "preferred_presentation": None,
                "inferability_preference": None,
                "visual_legibility_preference": None,
                "conceptual_quality_preference": None,
                "either_variant_overcomplicated": None,
                "notes": "",
            }
        )
    contact = build_contact_sheet(rendered, destination / "review_sheet.png")
    public.sort(key=lambda item: item["pair_id"])
    questionnaire.sort(key=lambda item: item["pair_id"])
    atomic_write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "feature": "palette_diversity",
            "provenance_hidden": True,
            "rules_withheld_from_first_page": True,
            "variant_identity_hidden": True,
            "task_count": 2 * len(public),
            "pair_count": len(public),
            "source_pair_count": len(sources),
            "contact_sheet": "review_sheet.png" if contact else None,
            "pairs": public,
        },
    )
    atomic_write_json(destination / "review_form.json", {"responses": questionnaire})
    atomic_write_json(
        destination / "DO_NOT_OPEN_FIRST.answer_key.json",
        {"warning": "Open only after recording blind judgments.", "answers": key},
    )
    lines = [
        "# Blinded V4.1 palette review",
        "",
        "Review the contact sheet and anonymous task JSON before opening the appendix.",
        "No source, author rule, or palette-arm label appears here.",
        "",
        "For each explicit A/B pair, judge whether the rule and inferability are",
        "unchanged, then record any preference in visual legibility, conceptual",
        "quality, or overall presentation. Use `a`, `b`, or `neither`.",
        "",
        "Record responses in `review_form.json`.",
        "",
    ]
    atomic_write_text(destination / "review_sheet.md", "\n".join(lines))
    appendix = ["# Rule and assignment appendix", "", "Open only after blind review.", ""]
    for item in sorted(key, key=lambda row: (row["pair_id"], row["label"])):
        appendix.extend(
            [
                f"## Pair {item['pair_id']} — {item['label'].upper()} ({item['anonymous_id']})",
                "",
                f"Source task: `{item['source_task_id']}`  ",
                f"Variant: `{item['variant']}`  ",
                f"Background: `{item['background_color']}`  ",
                f"Mapping: `{item['mapping']}`",
                "",
                f"Rule: {item['intended_rule']}",
                "",
            ]
        )
    atomic_write_text(destination / "answer_rule_appendix.md", "\n".join(appendix))
    result = {
        "directory": str(destination.resolve()),
        "contact_sheet": str(contact.resolve()) if contact else None,
        "task_count": 2 * len(public),
        "pair_count": len(public),
        "source_pair_count": len(sources),
        "human_results_available": False,
        "review_form": str((destination / "review_form.json").resolve()),
    }
    atomic_write_json(destination / "human_review_status.json", result)
    return result


def _arm_verification_stats(results: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    rows = [item for item in results if item["arm"] == arm]
    complete = [item for item in rows if item["status"] == "complete"]
    return {
        "attempted": len(rows),
        "completed": len(complete),
        "operational_failures": len(rows) - len(complete),
        "pass_count": sum(bool(item.get("passed")) for item in complete),
        "pass_rate": round(sum(bool(item.get("passed")) for item in complete) / len(complete), 6) if complete else None,
        "hidden_output_exact_match_count": sum(bool(item.get("hidden_output_exact_match")) for item in complete),
    }


def _arm_blind_stats(results: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    rows = [item for item in results if item["arm"] == arm]
    complete = [item for item in rows if item["status"] == "complete"]
    serious = sum(item.get("serious_different_output_alternative") is True for item in complete)
    return {
        "attempted": len(rows),
        "completed": len(complete),
        "operational_failures": len(rows) - len(complete),
        "primary_hidden_match_count": sum(bool(item.get("primary_matches_hidden")) for item in complete),
        "serious_different_output_alternative_count": serious,
        "serious_ambiguity_rate": round(serious / len(complete), 6) if complete else None,
    }


def _arm_editorial_stats(editorial: Mapping[str, Any], arm: str) -> dict[str, Any]:
    rows = [item["arms"][arm] for item in editorial["task_comparisons"]]
    complete = [item for item in rows if item["status"] == "complete"]
    intrinsic = Counter(item["intrinsic_quality"] for item in complete)
    quality = Counter(
        judgment["overall_quality"]
        for item in complete for judgment in item["judgments"]
    )
    concept = Counter(
        judgment["novel_concept"]
        for item in complete for judgment in item["judgments"]
    )
    composition = Counter(
        judgment["functional_composition"]
        for item in complete for judgment in item["judgments"]
    )
    core = intrinsic.get("core_quality", 0)
    reserve = intrinsic.get("reserve_quality", 0)
    return {
        "reviewed_tasks": len(complete),
        "incomplete_tasks": len(rows) - len(complete),
        "intrinsic_quality_counts": dict(sorted(intrinsic.items())),
        "core_quality_count": core,
        "core_or_reserve_count": core + reserve,
        "core_quality_yield": round(core / len(complete), 6) if complete else None,
        "core_or_reserve_yield": round((core + reserve) / len(complete), 6) if complete else None,
        "board_overall_quality_distribution": dict(sorted(quality.items())),
        "board_novel_concept_distribution": dict(sorted(concept.items())),
        "board_functional_composition_distribution": dict(sorted(composition.items())),
    }


def _quality_noninferiority(
    deterministic: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
    editorial: Mapping[str, Any],
    equivalence: Mapping[str, Any],
) -> dict[str, Any]:
    technical = {
        arm: _arm_verification_stats(verification["technical"], arm)
        for arm in ("control", "treatment")
    }
    blind = {
        arm: _arm_blind_stats(verification["blind"], arm)
        for arm in ("control", "treatment")
    }
    boards = {
        arm: _arm_editorial_stats(editorial, arm)
        for arm in ("control", "treatment")
    }
    applied = [item for item in deterministic if item["safe"]]
    deterministic_gate = all(
        item["globally_bijective"]
        and item["inverse_round_trip_exact"]
        and item["serialization_exact"]
        and item["metadata_consistent"]
        and item["palette_equivalent_identity_preserved"]
        and item["control_hard_valid"]
        and item["treatment_hard_valid"]
        for item in applied
    )
    paired = equivalence["judgments"]
    equivalence_gate = (
        equivalence["all_pairs_judged"]
        and all(
            item["global_bijection_exists"]
            and item["full_episode_consistent"]
            and item["rule_unchanged"]
            and item["inferability_unchanged"]
            and not item["substantive_difference"]
            for item in paired
        )
    )
    technical_gate = (
        technical["control"]["completed"] == technical["control"]["attempted"]
        and technical["treatment"]["completed"] == technical["treatment"]["attempted"]
        and technical["treatment"]["pass_count"] >= technical["control"]["pass_count"]
        and technical["treatment"]["hidden_output_exact_match_count"]
        >= technical["control"]["hidden_output_exact_match_count"]
    )
    blind_gate = (
        blind["control"]["completed"] == blind["control"]["attempted"]
        and blind["treatment"]["completed"] == blind["treatment"]["attempted"]
        and blind["treatment"]["serious_different_output_alternative_count"]
        <= blind["control"]["serious_different_output_alternative_count"]
    )
    control_core = boards["control"]["core_quality_yield"]
    treatment_core = boards["treatment"]["core_quality_yield"]
    control_retained = boards["control"]["core_or_reserve_yield"]
    treatment_retained = boards["treatment"]["core_or_reserve_yield"]
    editorial_gate = (
        control_core is not None
        and treatment_core is not None
        and control_retained is not None
        and treatment_retained is not None
        and treatment_core + 1e-9 >= 0.8 * control_core
        and treatment_retained + 1e-9 >= 0.9 * control_retained
        and boards["treatment"]["board_overall_quality_distribution"].get("below_real_bar", 0)
        <= boards["control"]["board_overall_quality_distribution"].get("below_real_bar", 0) + 1
    )
    return {
        "deterministic": {"gate_passed": deterministic_gate, "safe_task_count": len(applied)},
        "technical": technical,
        "blind": blind,
        "editorial": boards,
        "palette_equivalence_model_review": {
            "gate_passed": equivalence_gate,
            "judged_pairs": len(paired),
            "preference_counts": dict(sorted(Counter(item["preferred_presentation"] for item in paired).items())),
        },
        "provisional_model_noninferiority_gates": {
            "deterministic": deterministic_gate,
            "technical": technical_gate,
            "blind": blind_gate,
            "editorial": editorial_gate,
            "all_model_and_code_gates": all((deterministic_gate, equivalence_gate, technical_gate, blind_gate, editorial_gate)),
            "human_review": "pending",
        },
    }


def _palette_distribution_study(
    runtime: V41Runtime,
    sources: Sequence[Mapping[str, Any]],
    treatment: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    committed = artifacts["reference"]["committed_bank"]
    real = artifacts["reference"]["real_bank"]
    target = {int(key): float(value) for key, value in artifacts["assignment_ledger"]["target_distribution"].items()}
    starting = Counter({str(key): int(value) for key, value in committed["counts"].items() if key != "none"})
    after_sample = Counter(starting)
    for item in treatment.values():
        if item["background_color"] is not None:
            after_sample[str(item["background_color"])] += 1
    simulation = simulate_future_scheduling(
        seed=int(runtime.config["seed"]) + 1,
        starting_counts=starting,
        target_distribution=target,
        slate_count=int(runtime.config["palette_experiment"]["simulated_future_slate_count"]),
        tasks_per_slate=int(runtime.config["palette_experiment"]["simulated_tasks_per_slate"]),
    )
    final = Counter({str(key): int(value) for key, value in simulation["final_counts"].items()})
    sample_distribution = Counter(
        "none" if item["background_color"] is None else str(item["background_color"])
        for item in treatment.values()
    )
    original_distribution = _distribution_from_sources(sources, runtime.config)
    result = {
        "real_bank_reference": real,
        "original_replay_sample": original_distribution,
        "scheduled_replay_sample_counts": dict(sorted(sample_distribution.items())),
        "committed_bank_before": background_metrics(starting, real),
        "committed_plus_replay_sample": background_metrics(after_sample, real),
        "simulated_future_slates": simulation,
        "simulated_portfolio_after": background_metrics(final, real),
        "opt_out_count": sum(not item["palette_permutation_safe"] for item in treatment.values()),
        "opt_out_rate": round(sum(not item["palette_permutation_safe"] for item in treatment.values()) / len(treatment), 6),
        "black_remains_common": background_metrics(final, real)["black_background_share"] > 0.2,
        "one_nonblack_became_default": (
            background_metrics(final, real)["dominant_nonblack_share"]
            >= background_metrics(final, real)["black_background_share"]
        ),
    }
    atomic_write_json(runtime.run_dir / "palette_distribution_report.json", result)
    return result


def _coverage_retention(
    sources: Sequence[Mapping[str, Any]],
    deterministic: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    distribution: Mapping[str, Any],
) -> dict[str, Any]:
    modes = Counter(item["search_mode"] for item in sources)
    identity_preserved = sum(item["palette_equivalent_identity_preserved"] for item in deterministic)
    value = {
        "rolling_retention_matrix": {
            "ordinary_mode_share": {"control": 1.0, "treatment": 1.0, "difference": 0.0},
            "search_mode_counts": {"control": dict(sorted(modes.items())), "treatment": dict(sorted(modes.items()))},
            "black_background_share": {
                "control_sample": distribution["original_replay_sample"]["frequencies_among_confident"]["0"],
                "treatment_plus_committed": distribution["committed_plus_replay_sample"]["black_background_share"],
            },
            "supported_departure_share": {"control": None, "treatment": None, "not_studied": True},
            "episode_joint_share": {"control": None, "treatment": None, "not_studied": True},
            "abstraction_led_share": {"control": None, "treatment": None, "not_studied": True},
            "core_quality_yield": {
                "control": quality["editorial"]["control"]["core_quality_yield"],
                "treatment": quality["editorial"]["treatment"]["core_quality_yield"],
            },
            "core_or_reserve_yield": {
                "control": quality["editorial"]["control"]["core_or_reserve_yield"],
                "treatment": quality["editorial"]["treatment"]["core_or_reserve_yield"],
            },
            "serious_ambiguity_rate": {
                "control": quality["blind"]["control"]["serious_ambiguity_rate"],
                "treatment": quality["blind"]["treatment"]["serious_ambiguity_rate"],
            },
        },
        "palette_equivalent_identities_preserved": identity_preserved,
        "task_count": len(deterministic),
        "ordinary_v4_concepts_suppressed": 0,
        "same_procedure_duplicates_added": 0,
        "coverage_intervention_is_surface_only": True,
    }
    return value


def prepare_semantic_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Freeze an authorized semantic study without silently starting model work.

    The complete intervention contracts, sample minima, prompts, schemas, and
    downstream routing are configuration-validated.  A future explicit CLI
    invocation uses this preparation manifest before paired calls begin.
    """

    feature = config["experimental"]["feature"]
    if feature not in {"supported_departure", "episode_joint_inference", "abstraction_led"}:
        raise V41ExperimentError("semantic preparation requires C, J, or A")
    value = {
        "pipeline": config["pipeline"],
        "feature": feature,
        "authorized_to_start": True,
        "minimums": deepcopy(config["semantic_experiments"]),
        "effective_baseline_features": deepcopy(config["_effective_features"]),
        "non_gating_mechanism_auditor": True,
        "arm_labels_hidden_downstream": True,
        "automatic_promotion": False,
        "production_state_mutation_allowed": False,
        "next_action": "paired model execution requires this explicitly selected experiment config",
    }
    return value


__all__ = [
    "V41ExperimentError",
    "V41PromptStore",
    "V41Runtime",
    "palette_replay_sources",
    "prepare_semantic_experiment",
]
