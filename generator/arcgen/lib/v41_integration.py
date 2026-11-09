"""Accepted-feature integration runs for ARCGEN V4.1.

The integration runner deliberately wraps the frozen V4 control rather than
modifying it.  It executes against a private reservoir copy and applies the
accepted palette scheduler between realization and deterministic diagnostics,
so every downstream verifier sees the final episode while production state is
left byte-for-byte unchanged.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import hashlib
import math
import time

from .committed_tasks import load_committed_task_registry
from .data import load_task_bank
from .util import atomic_write_json, atomic_write_text, content_hash, read_json
from .v4_config import load_v4_config
from .v4_diagnostics import complete_task
from .v4_pipeline import V4ControlPipeline, V4PipelineError, V4ReservoirProducer
from .v4_reservoir import load_reservoir, reservoir_status
from .v41_palette import (
    apply_scheduled_palette,
    background_metrics,
    infer_episode_background,
    make_episode_envelope,
    measure_background_distribution,
    measure_real_bank_backgrounds,
    remap_authoring_contract,
    schedule_backgrounds,
    soft_target_distribution,
)


class V41IntegrationError(ValueError):
    pass


_PROTECTED_STATE_KEYS = (
    "reservoir",
    "committed_task_registry",
    "v4_1_feature_manifest",
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _protected_hashes(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result = {}
    for key in _PROTECTED_STATE_KEYS:
        path = Path(config["paths"][key])
        result[key] = {"path": str(path), "sha256": _sha256(path)}
    return result


def _assert_hashes_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    changed = [key for key in before if before[key]["sha256"] != after[key]["sha256"]]
    if changed:
        raise V41IntegrationError(
            "V4.1 integration mutated protected production state: " + ", ".join(changed)
        )


def _committed_distribution(config: Mapping[str, Any]) -> dict[str, Any]:
    registry_path = Path(config["paths"]["committed_task_registry"])
    registry = load_committed_task_registry(registry_path)
    tasks = []
    for entry in registry["entries"]:
        task_path = registry_path.parent / entry["task_path"]
        tasks.append((entry["registry_task_id"], read_json(task_path)))
    palette = config["palette_experiment"]
    return measure_background_distribution(
        tasks,
        modal_fraction=float(palette["conservative_modal_fraction"]),
        require_consistent_input_mode=bool(palette["require_consistent_input_mode"]),
    )


def _palette_target(config: Mapping[str, Any]) -> tuple[dict[int, float], dict[str, Any]]:
    palette = config["palette_experiment"]
    bank = load_task_bank(config["paths"]["annotations"], config["paths"]["tasks"])
    real = measure_real_bank_backgrounds(
        bank,
        modal_fraction=float(palette["conservative_modal_fraction"]),
        require_consistent_input_mode=bool(palette["require_consistent_input_mode"]),
    )
    target = soft_target_distribution(
        real,
        allowed_colors=palette["allowed_background_colors"],
        real_reference_weight=float(palette["real_reference_weight"]),
        diversity_weight=float(palette["diversity_weight"]),
    )
    return target, real


def transform_realization_palette(
    *,
    candidate_id: str,
    plan: Mapping[str, Any],
    realization: Mapping[str, Any],
    target_background: int | None,
    palette_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return a final-palette plan/realization and an inspectable treatment record."""

    if realization.get("status") != "realized":
        return deepcopy(dict(plan)), deepcopy(dict(realization)), {
            "task_id": candidate_id,
            "applied": False,
            "requested_background_color": target_background,
            "reason": "realizer refused before palette treatment",
        }
    metadata = realization["authoring_metadata"]
    envelope = make_episode_envelope(
        task_id=candidate_id,
        complete_task=complete_task(realization),
        intended_rule=metadata["intended_rule"],
        concrete_plan=plan,
        source="v4_1_palette_integration",
        modal_fraction=float(palette_config["conservative_modal_fraction"]),
        require_consistent_input_mode=bool(palette_config["require_consistent_input_mode"]),
    )
    mapped = apply_scheduled_palette(envelope, target_background)
    evidence = {
        **deepcopy(mapped["palette_treatment"]),
        "task_id": candidate_id,
        "palette_permutation_safe": mapped["palette_permutation_safe"],
        "safety_assessment": deepcopy(mapped["safety_assessment"]),
        "color_roles": deepcopy(mapped["color_roles"]),
        "palette_equivalent_identity_hash": mapped["palette_equivalent_identity_hash"],
        "original_exact_task_hash": envelope["exact_task_hash"],
        "final_exact_task_hash": mapped["exact_task_hash"],
    }
    if "effective_concrete_plan" not in mapped:
        return deepcopy(dict(plan)), deepcopy(dict(realization)), evidence

    mapping = {int(key): int(value) for key, value in mapped["palette_treatment"]["mapping"].items()}
    final_plan = deepcopy(mapped["effective_concrete_plan"])
    final_realization = remap_authoring_contract(realization, mapping)
    # Bare integers inside grid arrays are deliberately not interpreted as
    # palette claims by the prose rewriter.  Install the exact globally mapped
    # public and hidden arrays explicitly.
    final_realization["task"] = deepcopy(mapped["public_task"])
    final_realization["authoring_metadata"]["expected_test_outputs"] = deepcopy(
        mapped["hidden_test_outputs"]
    )
    final_realization["authoring_metadata"]["intended_rule"] = mapped[
        "effective_intended_rule"
    ]
    return final_plan, final_realization, evidence


class PaletteIntegratedV4Pipeline(V4ControlPipeline):
    """Frozen V4 control with one accepted post-realization palette hook."""

    def __init__(self, v4_config: dict[str, Any], v41_config: Mapping[str, Any]) -> None:
        self.v41_config = deepcopy(dict(v41_config))
        self.palette_target, self.real_background_reference = _palette_target(v41_config)
        self.committed_background_distribution = _committed_distribution(v41_config)
        super().__init__(v4_config)
        if "v4_1_palette_integration" not in self.state:
            self.state["v4_1_palette_integration"] = {
                "enabled": True,
                "accepted_run_id": v41_config["_feature_manifest_snapshot"]["features"]
                ["palette_diversity"]["accepted_run_id"],
                "target_distribution": {
                    str(key): value for key, value in self.palette_target.items()
                },
                "committed_background_distribution": deepcopy(
                    self.committed_background_distribution
                ),
                "real_background_reference": deepcopy(self.real_background_reference),
                "assignments": {},
                "treatments": {},
            }
            self._save()

    def select_slate(self) -> list[dict[str, Any]]:
        briefs = super().select_slate()
        integration = self.state["v4_1_palette_integration"]
        if not integration["assignments"]:
            ordered_ids = [item["candidate_id"] for item in briefs]
            assignments = schedule_backgrounds(
                ordered_ids,
                seed=int(self.v41_config["seed"]),
                current_counts=self.committed_background_distribution["counts"],
                target_distribution=self.palette_target,
                # Safety is checked from the completed episode after
                # realization.  Assigning all targets now removes any
                # completion-order effect from the streaming graph.
                eligible={task_id: True for task_id in ordered_ids},
            )
            integration["assignments"] = {
                task_id: assignments[task_id] for task_id in ordered_ids
            }
            integration["assignment_order"] = ordered_ids
            integration["assigned_before_candidate_execution"] = True
            self._save()
            directory = self.run_dir / "palette"
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write_json(directory / "slate_assignments.json", integration)
        return briefs

    def _store_realization(
        self,
        candidate_id: str,
        plan: Mapping[str, Any],
        realization: Mapping[str, Any],
        *,
        repaired: bool = False,
    ) -> tuple[bool, str]:
        integration = self.state["v4_1_palette_integration"]
        if repaired and candidate_id in integration["treatments"]:
            # A repair receives the already recolored rule and storyboard.  A
            # second palette permutation would be both unnecessary and unsafe.
            final_plan = deepcopy(dict(plan))
            final_realization = deepcopy(dict(realization))
            evidence = {
                **deepcopy(integration["treatments"][candidate_id]),
                "repair_reused_existing_palette": True,
            }
        else:
            target = integration["assignments"].get(candidate_id)
            final_plan, final_realization, evidence = transform_realization_palette(
                candidate_id=candidate_id,
                plan=plan,
                realization=realization,
                target_background=target,
                palette_config=self.v41_config["palette_experiment"],
            )
        okay, reason = super()._store_realization(
            candidate_id, final_plan, final_realization, repaired=repaired
        )
        if okay:
            with self.lock:
                integration = self.state["v4_1_palette_integration"]
                integration["treatments"][candidate_id] = deepcopy(evidence)
                self.state["records"][candidate_id]["palette_treatment"] = deepcopy(evidence)
                self._save()
            directory = self.run_dir / "palette"
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write_json(directory / f"{candidate_id}.json", evidence)
        return okay, reason


def _sandbox_v4_config(config: Mapping[str, Any], outer_run_dir: Path) -> dict[str, Any]:
    v4 = load_v4_config(config["_v4_control_config"])
    sandbox = outer_run_dir / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    source_reservoir = Path(config["paths"]["reservoir"])
    private_reservoir = sandbox / "reservoir.json"
    source_manifest = {
        "source_path": str(source_reservoir),
        "source_sha256": _sha256(source_reservoir),
        "copied_at_epoch_seconds": round(time.time(), 3),
    }
    manifest_path = sandbox / "reservoir_source.json"
    if private_reservoir.exists():
        previous = read_json(manifest_path)
        if previous["source_sha256"] != source_manifest["source_sha256"]:
            raise V41IntegrationError(
                "production reservoir changed since this integration sandbox was frozen"
            )
    else:
        atomic_write_json(private_reservoir, read_json(source_reservoir))
        atomic_write_json(manifest_path, source_manifest)
    v4["paths"]["reservoir"] = str(private_reservoir)
    v4["paths"]["runs"] = str((outer_run_dir / "v4_graph").resolve())
    v4["run"]["id"] = "palette-integration-slate"
    v4["run"]["automatically_commit_generated_tasks"] = False
    return v4


def _ensure_window_supply(
    config: Mapping[str, Any], v4_config: dict[str, Any], outer_run_dir: Path
) -> dict[str, Any]:
    reservoir = load_reservoir(v4_config["paths"]["reservoir"])
    proposed = sum(item["state"] == "proposed" for item in reservoir["proposals"])
    needed = max(0, int(v4_config["selection"]["eligible_window_size"]) - proposed)
    if needed == 0:
        return {
            "generated": 0,
            "reason": "sandbox already contains a complete eligible window",
            "reservoir": reservoir_status(reservoir, v4_config["reservoir"]),
        }
    cycles = max(1, math.ceil(needed / 12))
    producer = V4ReservoirProducer(
        v4_config,
        supply_run_id=f"{config['run']['id']}-sandbox-supply",
    )
    result = producer.run(cycles=cycles, force_between_watermarks=True)
    final = load_reservoir(v4_config["paths"]["reservoir"])
    final_proposed = sum(item["state"] == "proposed" for item in final["proposals"])
    if final_proposed < int(v4_config["selection"]["eligible_window_size"]):
        raise V41IntegrationError(
            f"sandbox supply left only {final_proposed} eligible proposals"
        )
    atomic_write_json(outer_run_dir / "sandbox/supply_result.json", result)
    return result


def _integration_report(
    config: Mapping[str, Any], state: Mapping[str, Any], *,
    before: Mapping[str, Any], after: Mapping[str, Any], supply: Mapping[str, Any],
) -> dict[str, Any]:
    records = state["records"]
    treatments = state["v4_1_palette_integration"]["treatments"]
    realized = [record for record in records.values() if record.get("complete_task")]
    requested = Counter(
        str(value) for value in state["v4_1_palette_integration"]["assignments"].values()
        if value is not None
    )
    final_counts = Counter()
    for evidence in treatments.values():
        color = evidence.get("realized_background_color")
        if color is not None:
            final_counts[str(color)] += 1
    reference = state["v4_1_palette_integration"]["real_background_reference"]
    statuses = Counter(record.get("status", "unknown") for record in records.values())
    intrinsic = Counter(record.get("intrinsic_quality", "not_reviewed") for record in records.values())
    placement = Counter(record.get("portfolio_placement", "not_reviewed") for record in records.values())
    return {
        "schema_version": 1,
        "pipeline": config["pipeline"],
        "run_id": config["run"]["id"],
        "integration_stage": "palette_diversity",
        "accepted_feature_run_id": config["_feature_manifest_snapshot"]["features"]
        ["palette_diversity"]["accepted_run_id"],
        "frozen_v4_control_implementation": True,
        "production_state_unchanged": before == after,
        "protected_state_before": deepcopy(dict(before)),
        "protected_state_after": deepcopy(dict(after)),
        "sandbox_supply": deepcopy(dict(supply)),
        "funnel": {
            "frozen_window": len(state["frozen_window"]["proposal_ids"]),
            "selected": len(records),
            "planned_ready": sum(record.get("concrete_plan", {}).get("status") == "ready" for record in records.values()),
            "realized": len(realized),
            # Final candidate records retain a list of verifier evaluations;
            # the V4 aggregate already resolves repairs and escalations.
            "technical_pass": int(state.get("funnel_summary", {}).get(
                "technical_pass",
                sum(
                    any(item.get("passed") is True for item in record.get("technical_results", []))
                    for record in records.values()
                ),
            )),
            "blind_completed": sum(bool(record.get("blind_results")) for record in records.values()),
            "editorially_reviewed": sum(bool(record.get("editorial_judgments")) for record in records.values()),
        },
        "terminal_statuses": dict(sorted(statuses.items())),
        "intrinsic_quality": dict(sorted(intrinsic.items())),
        "portfolio_placement": dict(sorted(placement.items())),
        "palette": {
            "requested_background_counts": dict(sorted(requested.items())),
            "realized_background_counts": dict(sorted(final_counts.items())),
            "treatment_count": len(treatments),
            "applied_non_identity_count": sum(item.get("applied") is True for item in treatments.values()),
            "safe_opt_out_count": sum(not item.get("palette_permutation_safe", False) for item in treatments.values()),
            # Eligibility is a policy property, not a claim that a small slate
            # happened to draw black from the scheduler.
            "black_remains_eligible": 0 in config["palette_experiment"]["allowed_background_colors"],
            "realized_distribution_metrics": background_metrics(final_counts, reference),
        },
        "call_accounting": deepcopy(state.get("control_call_accounting", {})),
        "tasks": [
            {
                "task_id": candidate_id,
                "search_mode": record["search_mode"],
                "status": record.get("status"),
                "intrinsic_quality": record.get("intrinsic_quality"),
                "portfolio_placement": record.get("portfolio_placement"),
                "palette_treatment": deepcopy(record.get("palette_treatment")),
                "task_path": str(
                    Path(config["paths"]["runs"])
                    / config["run"]["id"] / "v4_graph" / "palette-integration-slate"
                    / "tasks" / f"{candidate_id}.arc.json"
                ),
            }
            for candidate_id, record in sorted(records.items())
        ],
        "automatic_good_task_commit": False,
        "promotion_state_changed": False,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    funnel = report["funnel"]
    palette = report["palette"]
    lines = [
        "# ARCGEN V4.1 palette integration slate",
        "",
        "This is the clean integration run after explicit palette acceptance. It wraps the frozen V4 control, uses a private reservoir, and applies one global color bijection after realization. Every diagnostic, technical verifier, blind solver, and editorial board sees the final recolored grids.",
        "",
        "## State isolation",
        "",
        f"- Production state unchanged: **{report['production_state_unchanged']}**",
        "- Automatic good-task commit: **disabled**",
        "- Feature promotion mutation: **disabled**",
        "",
        "## Funnel",
        "",
        f"- Frozen proposals: {funnel['frozen_window']}",
        f"- Selected: {funnel['selected']}",
        f"- Ready plans: {funnel['planned_ready']}",
        f"- Realized: {funnel['realized']}",
        f"- Technical pass: {funnel['technical_pass']}",
        f"- Blind completed: {funnel['blind_completed']}",
        f"- Editorially reviewed: {funnel['editorially_reviewed']}",
        "",
        "## Palette behavior",
        "",
        f"- Completed palette treatments: {palette['treatment_count']}",
        f"- Non-identity recolorings: {palette['applied_non_identity_count']}",
        f"- Safe opt-outs: {palette['safe_opt_out_count']}",
        f"- Requested backgrounds: `{palette['requested_background_counts']}`",
        f"- Realized backgrounds: `{palette['realized_background_counts']}`",
        f"- Black remained eligible: **{palette['black_remains_eligible']}**",
        "",
        "## Quality dispositions",
        "",
        f"- Intrinsic quality: `{report['intrinsic_quality']}`",
        f"- Portfolio placement: `{report['portfolio_placement']}`",
        f"- Terminal statuses: `{report['terminal_statuses']}`",
        "",
        "No task was committed automatically. This integration validates the accepted surface treatment in the live authoring path; it does not reopen or automatically promote any semantic experiment.",
    ]
    return "\n".join(lines) + "\n"


def run_palette_integration(config: dict[str, Any]) -> dict[str, Any]:
    if config["integration"] != {"enabled": True, "stage": "palette_diversity"}:
        raise V41IntegrationError("only the explicitly accepted palette integration is runnable")
    if not config["_effective_features"].get("palette_scheduler"):
        raise V41IntegrationError("palette scheduler is not in the accepted baseline")
    outer = Path(config["paths"]["runs"]) / config["run"]["id"]
    outer.mkdir(parents=True, exist_ok=True)
    before = _protected_hashes(config)
    state_manifest = outer / "protected_state.before.json"
    if state_manifest.exists() and read_json(state_manifest) != before:
        raise V41IntegrationError("protected production snapshot changed during integration resume")
    if not state_manifest.exists():
        atomic_write_json(state_manifest, before)
    v4_config = _sandbox_v4_config(config, outer)
    completed_state_path = (
        Path(v4_config["paths"]["runs"]) / v4_config["run"]["id"] / "state.json"
    )
    if completed_state_path.is_file() and read_json(completed_state_path).get("completed"):
        # A report-only resume must not try to replenish a private reservoir
        # whose completed slate still owns uncommitted reservations.
        supply_path = outer / "sandbox/supply_result.json"
        supply = (
            read_json(supply_path)
            if supply_path.is_file()
            else {
                "generated": 0,
                "reason": "completed integration resumed without reservoir mutation",
            }
        )
    else:
        supply = _ensure_window_supply(config, v4_config, outer)
    pipeline = PaletteIntegratedV4Pipeline(v4_config, config)
    state = pipeline.run()
    after = _protected_hashes(config)
    _assert_hashes_unchanged(before, after)
    report = _integration_report(config, state, before=before, after=after, supply=supply)
    atomic_write_json(outer / "report.json", report)
    atomic_write_text(outer / "report.md", _markdown(report))
    atomic_write_json(outer / "protected_state.after.json", after)
    atomic_write_json(
        outer / "feature_manifest.snapshot.json", config["_feature_manifest_snapshot"]
    )
    return report


__all__ = [
    "PaletteIntegratedV4Pipeline",
    "V41IntegrationError",
    "run_palette_integration",
    "transform_realization_palette",
]
