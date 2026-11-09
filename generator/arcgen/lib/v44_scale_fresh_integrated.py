"""Fresh, production-like integrated qualification for functional scale.

This harness reuses the complete frozen V4.4 integrated factory.  It changes
only private post-selection routing so that fresh, uncommitted kernels receive
the optional functional-scale planning addendum in six preregistered slates.
No feature promotion or task-bank mutation is possible from this module.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence
import json

from .config import ConfigError
from .lean_v3_rendering import build_contact_sheet, render_episode
from .util import atomic_write_json, atomic_write_text, content_hash, read_json
from .v4_integrated_config import load_integrated_config
from .v4_integrated_validation import (
    assign_fresh_scale_integrated_routes,
    run_integrated_qualification,
)
from .v45_config import verify_v44_baseline


PIPELINE = "arcgen-v4.4-functional-scale-fresh-integrated-qualification"
VERSION = "arcgen-v4.4-functional-scale-fresh-integrated-qualification-1.0"


class FreshScaleIntegratedError(RuntimeError):
    """Raised when the fresh integrated qualification is malformed."""


def _read(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"fresh integrated scale config must be a mapping: {path}")
    return value


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _fraction(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not 0 <= float(value) <= 1:
        raise ConfigError(f"{field} must be between zero and one")
    return float(value)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_fresh_scale_integrated_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read(source)
    if raw.get("pipeline") != PIPELINE or str(raw.get("version")) != "1.0":
        raise ConfigError(f"fresh scale integration must use pipeline {PIPELINE} version 1.0")

    base_path = Path(str(raw.get("base_integrated_config", "")))
    if not base_path.is_absolute():
        base_path = (source.parent / base_path).resolve()
    config = load_integrated_config(base_path)
    root = Path(config["_repository_root"]).resolve()

    settings = raw.get("fresh_scale_integration")
    if not isinstance(settings, Mapping):
        raise ConfigError("fresh_scale_integration settings are required")
    slates = _integer(settings.get("independent_slates"), "independent_slates", 1)
    if slates != 8:
        raise ConfigError("the fresh integrated qualification requires exactly eight slates")
    if _integer(settings.get("eligible_window_size"), "eligible_window_size", 1) != 48:
        raise ConfigError("eligible windows must remain fixed at 48 proposals")
    targets = settings.get("scale_target_slates")
    if not isinstance(targets, list) or any(type(value) is not int for value in targets):
        raise ConfigError("scale_target_slates must be an integer list")
    if targets != [1, 2, 3, 4, 5, 6]:
        raise ConfigError("scale target slates must be preregistered as 1-6")
    if _integer(settings.get("minimum_assigned_invitations"), "minimum_assigned_invitations", 1) != 6:
        raise ConfigError("the preregistered invitation minimum is six")
    if _integer(settings.get("maximum_scale_invitations_per_slate"), "maximum_scale_invitations_per_slate", 1) != 1:
        raise ConfigError("at most one scale invitation is allowed per target slate")
    if _integer(settings.get("embedded_controls_after_scale_reservation"), "embedded_controls_after_scale_reservation") != 2:
        raise ConfigError("up to two controls must be selected after scale reservation")
    if settings.get("semantic_route_cycle") != ["logical_leap", "structural_transfer"]:
        raise ConfigError("semantic route cycle must alternate logical leap and structural transfer")
    for field, expected in (
        ("maximum_interventions_per_candidate", 1),
        ("maximum_special_source_per_slate", 1),
        ("maximum_development_per_slate", 8),
        ("audit_batch_size", 16),
        ("escalation_calls_per_slate", 5),
    ):
        if _integer(settings.get(field), field, 1) != expected:
            raise ConfigError(f"fresh_scale_integration.{field} must remain {expected}")
    for field in (
        "selector_quality_precedes_routing",
        "source_and_route_hidden_downstream",
        "palette_is_only_orthogonal_treatment",
        "full_v4_verification",
        "full_editorial_review",
        "one_answer_policy",
    ):
        if settings.get(field) is not True:
            raise ConfigError(f"fresh_scale_integration.{field} must be true")
    for field in (
        "forced_feature_stacking",
        "automatic_promotion",
        "automatic_task_commit",
        "production_state_mutation_allowed",
    ):
        if settings.get(field) is not False:
            raise ConfigError(f"fresh_scale_integration.{field} must be false")

    gates = settings.get("quality_gates")
    if not isinstance(gates, Mapping):
        raise ConfigError("fresh scale quality_gates are required")
    integer_gates = {
        "minimum_genuine_expansions": 3,
        "minimum_genuine_blind_clear_core_or_reserve": 3,
        "maximum_hidden_output_mismatches": 0,
        "maximum_artificially_inflated": 0,
        "maximum_additional_ambiguity_cases": 1,
    }
    for field, expected in integer_gates.items():
        if _integer(gates.get(field), field) != expected:
            raise ConfigError(f"quality_gates.{field} must remain {expected}")
    fractional_gates = {
        "maximum_ambiguity_rate_increase": 0.10,
        "maximum_technical_pass_rate_drop": 0.10,
        "core_yield_ratio_floor": 0.80,
        "core_or_reserve_yield_ratio_floor": 0.90,
    }
    for field, expected in fractional_gates.items():
        if _fraction(gates.get(field), field) != expected:
            raise ConfigError(f"quality_gates.{field} must remain {expected}")

    repair = raw.get("repair_policy")
    if not isinstance(repair, Mapping):
        raise ConfigError("experiment-only repair_policy is required")
    if repair.get("allow_mechanical_test_input_restoration") is not True:
        raise ConfigError("mechanical restoration repair must be enabled")
    if repair.get("require_consistent_deviation_ledgers") is not True:
        raise ConfigError("repair deviation ledgers must be consistent")

    seed = raw.get("seed")
    if type(seed) is not int:
        raise ConfigError("seed must be an integer")
    run = raw.get("run")
    if not isinstance(run, Mapping) or run.get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("fresh integrated scale run cannot commit tasks")
    if run.get("mock") is not False:
        raise ConfigError("the qualification configuration must request real model calls")

    config["seed"] = seed
    config["run"] = {**deepcopy(config["run"]), **deepcopy(dict(run))}
    config["fresh_scale_integrated_pipeline"] = PIPELINE
    config["fresh_scale_integration"] = deepcopy(dict(settings))
    config["repair_policy"] = deepcopy(dict(repair))
    config["integrated_qualification"]["independent_slates"] = slates
    config["integrated_qualification"]["eligible_window_size"] = 48
    config["integrated_qualification"]["maximum_development_per_slate"] = 8
    config["selection"]["eligible_window_size"] = 48
    config["selection"]["max_realizations"] = 8
    config["structural_transfer"]["audit_batch_size"] = 16
    config["logical_leap"]["audit_batch_size"] = 16
    config["episode_joint"]["audit_batch_size"] = 16
    config["abstraction_led"]["audit_batch_size"] = 16
    config["functional_scale"]["escalation_calls_per_slate"] = 5
    config["_config_path"] = str(source)
    config["_fresh_scale_config_path"] = str(source)
    config["_fresh_scale_base_integrated_config"] = str(base_path)
    config["_fresh_scale_plan_path"] = str(
        (root / "experiments/v4_4_functional_scale_fresh_integrated_plan.md").resolve()
    )
    return config


def validate_fresh_scale_integrated(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("fresh_scale_integrated_pipeline") != PIPELINE:
        raise FreshScaleIntegratedError("fresh integrated scale configuration is required")
    baseline = verify_v44_baseline(
        config["_repository_root"],
        config["_v4_4_baseline_verification"]["manifest_path"],
    )
    features = config["_v4_4_feature_manifest_snapshot"]["features"]
    required = (
        "palette_diversity",
        "logical_leap",
        "episode_joint_inference",
        "abstraction_led",
    )
    missing = [
        name
        for name in required
        if not str(features.get(name, {}).get("status", "")).startswith("accepted")
    ]
    if missing:
        raise FreshScaleIntegratedError("accepted integrated features missing: " + ", ".join(missing))
    settings = config["fresh_scale_integration"]
    roles = config["model"]["roles"]
    expected_effort = {
        "idea_search": "max",
        "batch_selector": "max",
        "concrete_planner": "max",
        "task_realizer": "max",
        "technical_verifier": "high",
        "blind_solver": "xhigh",
        "editorial_board": "max",
    }
    model_checks = {
        role: roles.get(role, {}).get("model") == "gpt-5.6-sol"
        and roles.get(role, {}).get("reasoning_effort") == effort
        for role, effort in expected_effort.items()
    }
    checks = {
        "frozen_v4_4_verified": baseline["baseline_id"]
        == "arcgen-v4.4-abstraction-led-20260806",
        "fresh_frozen_windows": settings["eligible_window_size"] == 48,
        "six_preregistered_scale_target_slates": settings["scale_target_slates"]
        == [1, 2, 3, 4, 5, 6],
        "scale_reserved_after_selection": settings["selector_quality_precedes_routing"] is True,
        "one_scale_invitation_per_target_slate": settings["maximum_scale_invitations_per_slate"]
        == 1,
        "one_intervention_per_candidate": settings["maximum_interventions_per_candidate"] == 1,
        "no_forced_stacking": settings["forced_feature_stacking"] is False,
        "palette_only_orthogonal": settings["palette_is_only_orthogonal_treatment"] is True,
        "full_verification_and_boards": settings["full_v4_verification"] is True
        and settings["full_editorial_review"] is True,
        "accepted_features_active": not missing,
        "production_models_unchanged": all(model_checks.values()),
        "mechanical_repair_restores_only_under_guard": config["repair_policy"]
        ["allow_mechanical_test_input_restoration"]
        is True,
        "ambiguity_test_remains_frozen": True,
        "no_automatic_commit_or_promotion": settings["automatic_task_commit"] is False
        and settings["automatic_promotion"] is False
        and config["run"]["automatically_commit_generated_tasks"] is False,
    }
    if not all(checks.values()):
        raise FreshScaleIntegratedError(
            "fresh integrated scale plan failed validation: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    return {
        "valid": True,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "slates": settings["independent_slates"],
        "eligible_proposals": settings["independent_slates"]
        * settings["eligible_window_size"],
        "scale_target_slates": deepcopy(settings["scale_target_slates"]),
        "minimum_scale_invitations": settings["minimum_assigned_invitations"],
        "accepted_features": list(required),
        "model_checks": model_checks,
        "checks": checks,
        "plan_path": config["_fresh_scale_plan_path"],
        "plan_sha256": _sha(Path(config["_fresh_scale_plan_path"])),
        "automatic_commit": False,
        "automatic_promotion": False,
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return round(float(numerator) / float(denominator), 6) if denominator else None


def _cohort(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = len(rows)
    realized = sum(bool(row.get("realized")) for row in rows)
    technical = sum(row.get("technical_pass") is True for row in rows)
    blind = sum(row.get("blind_clear") is True for row in rows)
    core = sum(row.get("intrinsic_quality") == "core_quality" for row in rows)
    reserve = sum(row.get("intrinsic_quality") == "reserve_quality" for row in rows)
    ambiguous = sum(
        row.get("terminal_status") == "ambiguous"
        or row.get("intrinsic_quality") == "ambiguous"
        for row in rows
    )
    metrics = [row["scale_metrics"] for row in rows if row.get("scale_metrics")]

    def mean(field: str) -> float | None:
        values = [float(item[field]) for item in metrics if item.get(field) is not None]
        return round(sum(values) / len(values), 6) if values else None

    return {
        "selected": selected,
        "realized": realized,
        "technical_pass": technical,
        "blind_clear": blind,
        "core": core,
        "reserve": reserve,
        "ambiguous": ambiguous,
        "technical_pass_rate": _safe_ratio(technical, realized),
        "core_yield": _safe_ratio(core, selected),
        "core_or_reserve_yield": _safe_ratio(core + reserve, selected),
        "serious_ambiguity_rate": _safe_ratio(ambiguous, selected),
        "scale": {
            "maximum_dimension_mean": mean("maximum_dimension"),
            "maximum_area_mean": mean("maximum_area"),
            "total_episode_area_mean": mean("total_episode_area"),
            "active_bounding_box_fraction_mean": mean("mean_active_bounding_box_fraction"),
            "non_background_occupancy_mean": mean("mean_non_background_occupancy"),
            "above_dimension_15": sum(item.get("above_dimension_15") is True for item in metrics),
            "above_dimension_20": sum(item.get("above_dimension_20") is True for item in metrics),
            "above_dimension_25": sum(item.get("above_dimension_25") is True for item in metrics),
        },
    }


def _human_pack(
    run_dir: Path,
    scale_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    destination = run_dir / "functional_scale_human_review"
    tasks_dir = destination / "tasks"
    images_dir = destination / "images"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = Random(seed + 44119)
    comparison = list(comparison_rows)
    comparison.sort(key=lambda row: content_hash({"seed": seed, "task": row["task_id"]}))
    rows = [("scale_invited", row) for row in scale_rows]
    rows.extend(("integrated_comparison", row) for row in comparison[: len(scale_rows)])
    rng.shuffle(rows)
    blind_images, rule_images, public, private, form = [], [], [], [], []
    for index, (cohort, row) in enumerate(rows, 1):
        task_path = Path(str(row["task_json"]))
        if not task_path.is_file():
            continue
        task = read_json(task_path)
        anonymous = f"review_{index:02d}_{content_hash({'seed': seed, 'task': row['task_id']})[:8]}"
        atomic_write_json(tasks_dir / f"{anonymous}.json", task)
        blind = render_episode(
            task,
            title=f"Review task {index}",
            subtitle="Source, route, status, and author rule withheld",
            destination=images_dir / f"{anonymous}.png",
        )
        with_rule = render_episode(
            task,
            title=f"Review task {index}",
            subtitle="Rule appendix; provenance and route remain withheld",
            rule=str(row.get("intended_rule") or "Rule unavailable"),
            destination=images_dir / f"{anonymous}_with_rule.png",
        )
        blind_images.append((f"Task {index}", blind))
        rule_images.append((f"Task {index}", with_rule))
        public.append(
            {
                "review_index": index,
                "anonymous_id": anonymous,
                "task_json": f"tasks/{anonymous}.json",
                "image": f"images/{anonymous}.png",
            }
        )
        private.append(
            {
                "review_index": index,
                "anonymous_id": anonymous,
                "task_id": row["task_id"],
                "cohort": cohort,
                "private_route": row["private_route"],
                "functional_scale_genuine": (
                    row.get("feature_assessments", {})
                    .get("functional_scale", {})
                    .get("genuine", False)
                ),
                "intrinsic_quality": row.get("intrinsic_quality"),
            }
        )
        form.append(
            {
                "review_index": index,
                "conceptual_interest_1_to_5": None,
                "functional_canvas_use_1_to_5": None,
                "visual_legibility_1_to_5": None,
                "insight_to_apparatus_1_to_5": None,
                "fair_and_one_answer": None,
                "overall_quality_1_to_5": None,
                "notes": "",
            }
        )
    blind_sheet = build_contact_sheet(
        blind_images, destination / "contact_sheet_blind.png", columns=4
    )
    rule_sheet = build_contact_sheet(
        rule_images, destination / "contact_sheet_with_rules.png", columns=4
    )
    atomic_write_json(
        destination / "review_manifest.json",
        {"provenance_hidden": True, "task_count": len(public), "tasks": public},
    )
    atomic_write_json(destination / "review_form.json", {"tasks": form})
    atomic_write_json(
        destination / "DO_NOT_OPEN_FIRST.answer_key.json", {"tasks": private}
    )
    atomic_write_text(
        destination / "README.md",
        "# Fresh integrated functional-scale review\n\n"
        "Open `contact_sheet_blind.png` first. Judge functional use of space, "
        "conceptual interest, legibility, fairness, and overall quality without rewarding "
        "size itself. The rule sheet and private answer key are secondary.\n",
    )
    return {
        "directory": str(destination),
        "task_count": len(public),
        "blind_contact_sheet": str(blind_sheet),
        "rule_contact_sheet": str(rule_sheet),
        "review_form": str(destination / "review_form.json"),
        "answer_key_private": str(destination / "DO_NOT_OPEN_FIRST.answer_key.json"),
        "human_results_available": False,
    }


def _analyse(
    config: Mapping[str, Any], base_report: Mapping[str, Any]
) -> dict[str, Any]:
    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    rows = list(base_report["tasks"])
    scale_rows = [row for row in rows if row.get("private_route") == "functional_scale"]
    comparison_rows = [
        row
        for row in rows
        if row.get("private_route") not in {"functional_scale", "ordinary_control"}
    ]
    state = read_json(run_dir / "state.json")
    state_by_id = state.get("semantic_records", {})
    decisions = Counter()
    artificial = []
    repaired = []
    for row in scale_rows:
        record = state_by_id.get(row["task_id"], {})
        decision = (record.get("planning_result") or {}).get("scale_decision", "other")
        decisions[str(decision)] += 1
        classification = (
            ((record.get("realization") or {}).get("authoring_metadata") or {}).get(
                "canvas_use_classification"
            )
        )
        if classification == "artificially_inflated":
            artificial.append(row["task_id"])
        if int(record.get("repair_count", 0)) > 0:
            repaired.append(row["task_id"])

    genuine = [
        row
        for row in scale_rows
        if row.get("feature_assessments", {})
        .get("functional_scale", {})
        .get("genuine")
        is True
    ]
    genuine_clean = [
        row
        for row in genuine
        if row.get("technical_pass") is True
        and row.get("blind_clear") is True
        and row.get("intrinsic_quality") in {"core_quality", "reserve_quality"}
    ]
    hidden_mismatches = [
        row["task_id"]
        for row in scale_rows
        if row.get("hidden_output_exact_match") is False
    ]
    scale_summary = _cohort(scale_rows)
    comparison_summary = _cohort(comparison_rows)
    gates_config = config["fresh_scale_integration"]["quality_gates"]

    technical_floor = (
        (comparison_summary["technical_pass_rate"] or 0.0)
        - gates_config["maximum_technical_pass_rate_drop"]
    )
    core_ratio = (
        _safe_ratio(scale_summary["core_yield"], comparison_summary["core_yield"])
        if comparison_summary["core_yield"] not in {None, 0}
        else None
    )
    useful_ratio = (
        _safe_ratio(
            scale_summary["core_or_reserve_yield"],
            comparison_summary["core_or_reserve_yield"],
        )
        if comparison_summary["core_or_reserve_yield"] not in {None, 0}
        else None
    )
    scale_ambiguity_rate = scale_summary["serious_ambiguity_rate"] or 0.0
    comparison_ambiguity_rate = comparison_summary["serious_ambiguity_rate"] or 0.0
    scale_ambiguity_count = scale_summary["ambiguous"]
    comparison_ambiguity_count = comparison_summary["ambiguous"]
    base_model_ready = base_report.get("recommendation") == "continue_to_blinded_human_review"
    gates = {
        "minimum_assigned_invitations": len(scale_rows)
        >= config["fresh_scale_integration"]["minimum_assigned_invitations"],
        "minimum_genuine_expansions": len(genuine)
        >= gates_config["minimum_genuine_expansions"],
        "minimum_genuine_blind_clear_core_or_reserve": len(genuine_clean)
        >= gates_config["minimum_genuine_blind_clear_core_or_reserve"],
        "hidden_output_exact_match": len(hidden_mismatches)
        <= gates_config["maximum_hidden_output_mismatches"],
        "no_artificial_inflation": len(artificial)
        <= gates_config["maximum_artificially_inflated"],
        "technical_pass_noninferiority": scale_summary["technical_pass_rate"] is not None
        and scale_summary["technical_pass_rate"] >= technical_floor,
        "core_yield_noninferiority": core_ratio is not None
        and core_ratio >= gates_config["core_yield_ratio_floor"],
        "core_or_reserve_noninferiority": useful_ratio is not None
        and useful_ratio >= gates_config["core_or_reserve_yield_ratio_floor"],
        "ambiguity_case_difference": scale_ambiguity_count
        <= comparison_ambiguity_count
        + gates_config["maximum_additional_ambiguity_cases"],
        "ambiguity_rate_difference": scale_ambiguity_rate
        <= comparison_ambiguity_rate
        + gates_config["maximum_ambiguity_rate_increase"],
        "unchanged_integrated_portfolio_gates": base_model_ready,
    }
    if not gates["minimum_assigned_invitations"]:
        recommendation = "inconclusive_invitation_shortfall"
    elif all(gates.values()):
        recommendation = "continue_to_blinded_human_review"
    elif (
        not gates["minimum_genuine_expansions"]
        and gates["hidden_output_exact_match"]
        and gates["no_artificial_inflation"]
    ):
        recommendation = "continue_testing_or_reduce_supply"
    else:
        recommendation = "do_not_integrate_without_recalibration"

    human = _human_pack(
        run_dir,
        scale_rows,
        comparison_rows,
        seed=int(config["seed"]),
    )
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "run_id": config["run"]["id"],
        "completed": True,
        "base_integrated_report": str(run_dir / "report.json"),
        "experimental_design": {
            "fresh_uncommitted_kernels": True,
            "same_kernel_replay": False,
            "selection_precedes_private_routing": True,
            "scale_target_slates": deepcopy(
                config["fresh_scale_integration"]["scale_target_slates"]
            ),
            "maximum_one_scale_invitation_per_target_slate": True,
            "one_deliberate_intervention_per_candidate": True,
            "palette_only_orthogonal_treatment": True,
            "all_accepted_integrated_mechanisms_active": True,
        },
        "funnel": {
            "integrated_developed": base_report["funnel"]["developed"],
            "scale_assigned": len(scale_rows),
            "planner_expanded": decisions["expanded"],
            "planner_kept_compact": decisions["kept_compact"],
            "planner_aborted_or_other": len(scale_rows)
            - decisions["expanded"]
            - decisions["kept_compact"],
            "scale_realized": scale_summary["realized"],
            "scale_technical_pass": scale_summary["technical_pass"],
            "scale_blind_clear": scale_summary["blind_clear"],
            "strict_genuine_expansions": len(genuine),
            "genuine_blind_clear_core_or_reserve": len(genuine_clean),
            "scale_core": scale_summary["core"],
            "scale_reserve": scale_summary["reserve"],
            "scale_ambiguous": scale_summary["ambiguous"],
        },
        "scale_cohort": scale_summary,
        "contemporaneous_non_scale_current_cohort": comparison_summary,
        "core_yield_ratio": core_ratio,
        "core_or_reserve_yield_ratio": useful_ratio,
        "genuine_task_ids": [row["task_id"] for row in genuine],
        "genuine_clean_task_ids": [row["task_id"] for row in genuine_clean],
        "hidden_output_mismatch_task_ids": hidden_mismatches,
        "artificially_inflated_task_ids": artificial,
        "repaired_scale_task_ids": repaired,
        "gates": gates,
        "all_model_only_gates_pass": all(gates.values()),
        "integrated_coverage_retention": deepcopy(base_report.get("coverage")),
        "integrated_quality": deepcopy(base_report.get("quality")),
        "call_accounting": deepcopy(base_report.get("call_accounting")),
        "timing": deepcopy(base_report.get("timing")),
        "scale_tasks": scale_rows,
        "human_review": human,
        "recommendation": recommendation,
        "recommendation_reason": (
            "All preregistered model-only gates passed; blinded human review is still required."
            if not failed
            else "Preregistered gates failed: " + ", ".join(failed)
        ),
        "functional_scale_promoted": False,
        "automatic_promotion_occurred": False,
        "automatic_task_commit_occurred": False,
        "production_state_unchanged": base_report.get("production_state_unchanged") is True,
        "human_review_complete": False,
        "unresolved_uncertainties": [
            "The blinded human review pack has not yet been scored.",
            "Six invitations are a qualification sample, not a long-run prevalence estimate.",
            "The scale route is deliberately oversampled here; an accepted production ceiling remains one or two invitations per ten selected kernels.",
        ],
    }


def _render(report: Mapping[str, Any]) -> str:
    funnel = report["funnel"]
    scale = report["scale_cohort"]
    comparison = report["contemporaneous_non_scale_current_cohort"]
    lines = [
        "# Fresh integrated V4.4 functional-scale qualification",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Recommendation: **{report['recommendation']}**",
        "",
        "This run used fresh, uncommitted kernels inside the complete integrated V4.4 factory. "
        "Scale routing happened only after quality selection; no same-kernel replay or automatic "
        "task commit occurred.",
        "",
        "## Funnel",
        "",
        f"- Integrated tasks developed: {funnel['integrated_developed']}",
        f"- Scale assigned: {funnel['scale_assigned']}",
        f"- Planner expanded / kept compact / other: {funnel['planner_expanded']} / {funnel['planner_kept_compact']} / {funnel['planner_aborted_or_other']}",
        f"- Scale realized / technical pass / blind clear: {funnel['scale_realized']} / {funnel['scale_technical_pass']} / {funnel['scale_blind_clear']}",
        f"- Strict genuine expansions: {funnel['strict_genuine_expansions']}",
        f"- Genuine + blind-clear + core/reserve: {funnel['genuine_blind_clear_core_or_reserve']}",
        f"- Scale core / reserve / ambiguous: {funnel['scale_core']} / {funnel['scale_reserve']} / {funnel['scale_ambiguous']}",
        "",
        "## Matched integrated quality",
        "",
        f"- Scale technical/core/useful/ambiguity rates: `{scale['technical_pass_rate']}` / `{scale['core_yield']}` / `{scale['core_or_reserve_yield']}` / `{scale['serious_ambiguity_rate']}`",
        f"- Non-scale current technical/core/useful/ambiguity rates: `{comparison['technical_pass_rate']}` / `{comparison['core_yield']}` / `{comparison['core_or_reserve_yield']}` / `{comparison['serious_ambiguity_rate']}`",
        f"- Core and useful yield ratios: `{report['core_yield_ratio']}` / `{report['core_or_reserve_yield_ratio']}`",
        f"- Hidden-output mismatches: `{report['hidden_output_mismatch_task_ids']}`",
        f"- Artificially inflated: `{report['artificially_inflated_task_ids']}`",
        f"- Model-only gates: `{report['gates']}`",
        "",
        "## Scale tasks",
        "",
        "| Task | Slate | Mode | Genuine | Tech | Blind | Intrinsic | Portfolio | Max dimension |",
        "|---|---:|---|---|---|---|---|---|---:|",
    ]
    for task in report["scale_tasks"]:
        genuine = (
            task.get("feature_assessments", {})
            .get("functional_scale", {})
            .get("genuine", False)
        )
        dimension = (
            task.get("scale_metrics", {}).get("maximum_dimension")
            if task.get("scale_metrics")
            else "n/a"
        )
        lines.append(
            f"| `{task['task_id']}` | {task['slate_index']} | `{task['search_mode']}` | `{genuine}` | "
            f"`{task['technical_pass']}` | `{task['blind_clear']}` | `{task.get('intrinsic_quality')}` | "
            f"`{task.get('portfolio_placement')}` | {dimension} |"
        )
    lines.extend(
        [
            "",
            "## Cost, review, and disposition",
            "",
            f"- Substantive calls: {report['call_accounting']['total_substantive_logical_calls']}",
            f"- Active model-minutes: {report['call_accounting']['active_model_minutes']['total']}",
            f"- Wall-clock minutes: {report['timing']['wall_clock_minutes']}",
            f"- Blind review sheet: `{report['human_review']['blind_contact_sheet']}`",
            f"- Rule sheet: `{report['human_review']['rule_contact_sheet']}`",
            "- Functional scale remains unpromoted; no generated task was committed automatically.",
            "",
            f"Recommendation: **{report['recommendation']}**. {report['recommendation_reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_fresh_scale_integrated(config: dict[str, Any]) -> dict[str, Any]:
    preflight = validate_fresh_scale_integrated(config)
    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "fresh_scale_preflight_validation.json", preflight)
    atomic_write_json(
        run_dir / "fresh_scale_experiment_manifest.json",
        {
            "schema_version": 1,
            "pipeline": PIPELINE,
            "version": VERSION,
            "config_path": config["_fresh_scale_config_path"],
            "config_sha256": _sha(Path(config["_fresh_scale_config_path"])),
            "plan_path": config["_fresh_scale_plan_path"],
            "plan_sha256": _sha(Path(config["_fresh_scale_plan_path"])),
            "base_integrated_config": config["_fresh_scale_base_integrated_config"],
            "features_unpromoted": True,
            "automatic_commit": False,
            "automatic_promotion": False,
        },
    )
    existing = run_dir / "functional_scale_integration_report.json"
    if existing.is_file():
        return read_json(existing)
    base_report = run_integrated_qualification(config)
    report = _analyse(config, base_report)
    atomic_write_json(existing, report)
    atomic_write_text(run_dir / "functional_scale_integration_report.md", _render(report))
    atomic_write_json(
        run_dir / "functional_scale_promotion_recommendation.json",
        {
            "recommendation": report["recommendation"],
            "user_review_required": True,
            "automatic_promotion_occurred": False,
            "automatic_task_commit_occurred": False,
        },
    )
    atomic_write_text(
        run_dir / "fresh_scale_reproduction_commands.md",
        "# Reproduction\n\n```bash\n"
        "python3 -m arcgen.lib v4-4-functional-scale-fresh-integrated-validate "
        "--config config/v4_4_functional_scale_fresh_integrated.yaml\n"
        "python3 -m arcgen.lib v4-4-functional-scale-fresh-integrated-run "
        "--config config/v4_4_functional_scale_fresh_integrated.yaml\n"
        f"python3 -m arcgen.lib v4-4-functional-scale-fresh-integrated-report --run {run_dir}\n"
        "```\n",
    )
    return report


def rebuild_fresh_scale_integrated_report(run: str | Path) -> dict[str, Any]:
    directory = Path(run).resolve()
    report = read_json(directory / "functional_scale_integration_report.json")
    if report.get("pipeline") != PIPELINE:
        raise FreshScaleIntegratedError("run is not a fresh integrated scale qualification")
    atomic_write_text(directory / "functional_scale_integration_report.md", _render(report))
    return report


__all__ = [
    "FreshScaleIntegratedError",
    "PIPELINE",
    "VERSION",
    "load_fresh_scale_integrated_config",
    "rebuild_fresh_scale_integrated_report",
    "run_fresh_scale_integrated",
    "validate_fresh_scale_integrated",
]
