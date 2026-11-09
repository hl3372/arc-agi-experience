"""Reports and exports for the lean authoring pipeline."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence
import json

from .lean_config import PIPELINE_ID
from .lean_pipeline import LEAN_PIPELINE_VERSION
from .util import atomic_write_json, atomic_write_text, content_hash, read_json


class LeanReportError(ValueError):
    pass


def _grid_text(grid: Sequence[Sequence[int]]) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def _episode_markdown(task: dict[str, Any]) -> str:
    lines: list[str] = []
    for split in ("train", "test"):
        for index, pair in enumerate(task[split], 1):
            lines.extend(
                [
                    f"{split.title()} {index} input ({len(pair['input'])}×{len(pair['input'][0])}):",
                    "",
                    "```text",
                    _grid_text(pair["input"]),
                    "```",
                    "",
                    f"{split.title()} {index} output ({len(pair['output'])}×{len(pair['output'][0])}):",
                    "",
                    "```text",
                    _grid_text(pair["output"]),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def _call_budget(state: dict[str, Any]) -> dict[str, Any]:
    budget = state["call_budget"]
    ledger = list(budget["ledger"].values())
    normal = [item for item in ledger if item["budget_kind"] == "normal"]
    escalation = [item for item in ledger if item["budget_kind"] == "escalation"]
    by_stage: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "logical_calls": 0,
            "executed_uncached_calls": 0,
            "cache_hits": 0,
            "retry_calls": 0,
            "failed_calls": 0,
            "successful_execution_seconds": 0.0,
            "model_execution_seconds": 0.0,
        }
    )
    for item in ledger:
        row = by_stage[item["role"]]
        row["logical_calls"] += 1
        row["retry_calls"] += int(item.get("retry_calls", 0))
        if item.get("status") == "failed":
            row["failed_calls"] += 1
        elif item.get("cache_source"):
            row["cache_hits"] += 1
        else:
            row["executed_uncached_calls"] += 1
            timing = item.get("timing") or {}
            row["successful_execution_seconds"] += float(
                timing.get("total_seconds") or 0.0
            )
            row["model_execution_seconds"] += float(
                timing.get("model_execution_seconds") or 0.0
            )
    return {
        "normal_limit": budget["normal_limit"],
        "escalation_limit": budget["escalation_limit"],
        "hard_limit": budget["hard_limit"],
        "normal_logical_calls": len(normal),
        "escalation_logical_calls": len(escalation),
        "total_lean_logical_calls": len(ledger),
        "executed_uncached_calls": sum(
            not item.get("cache_source") and item.get("status") == "complete"
            for item in ledger
        ),
        "cache_hits": sum(bool(item.get("cache_source")) for item in ledger),
        "retry_calls": sum(int(item.get("retry_calls", 0)) for item in ledger),
        "failed_logical_calls": sum(
            item.get("status") == "failed" for item in ledger
        ),
        "successful_execution_seconds": round(
            sum(
                float((item.get("timing") or {}).get("total_seconds") or 0.0)
                for item in ledger
                if not item.get("cache_source")
            ),
            3,
        ),
        "model_execution_seconds": round(
            sum(
                float(
                    (item.get("timing") or {}).get(
                        "model_execution_seconds"
                    )
                    or 0.0
                )
                for item in ledger
                if not item.get("cache_source")
            ),
            3,
        ),
        "by_stage": dict(sorted(by_stage.items())),
        "ledger": budget["ledger"],
    }


def _task_card(record: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    candidate_id = record["candidate_id"]
    complete = record.get("complete_task")
    return {
        "candidate_id": candidate_id,
        "search_mode": record["search_mode"],
        "editorial_route": record.get("editorial_brief", {}).get("route"),
        "development_priority": record.get("editorial_brief", {}).get(
            "development_priority"
        ),
        "central_relation": record["central_relation"],
        "intended_rule": record.get("architecture", {}).get("intended_rule"),
        "architecture_status": record.get("architecture", {}).get("status"),
        "architecture_summary": {
            "functional_elements": record.get("architecture", {}).get(
                "functional_elements", []
            ),
            "composition_audit": record.get("architecture", {}).get(
                "composition_audit"
            ),
            "axis_self_assessment": record.get("architecture", {}).get(
                "axis_self_assessment"
            ),
        },
        "realized": complete is not None,
        "complete_task": complete,
        "deterministic_diagnostics": record.get("deterministic_diagnostics"),
        "technical_results": record.get("technical_results", []),
        "blind_results": record.get("blind_results", []),
        "ambiguity_adjudication": record.get("ambiguity_adjudication"),
        "editorial_judgments": record.get("editorial_judgments", []),
        "final_status": record["status"],
        "final_reason": record.get("final_reason"),
        "rejections": record.get("rejections", []),
        "repair_count": record.get("repair_count", 0),
        "full_task_json_path": (
            str((run_dir / "tasks" / f"{candidate_id}.arc.json").resolve())
            if complete is not None
            else None
        ),
        "authoring_json_path": (
            str(
                (
                    run_dir
                    / "tasks"
                    / f"{candidate_id}.authoring.json"
                ).resolve()
            )
            if complete is not None
            else None
        ),
    }


def _metrics(
    state: dict[str, Any], cards: Sequence[dict[str, Any]], calls: dict[str, Any]
) -> dict[str, Any]:
    records = list(state["records"].values())
    realized = [item for item in records if item.get("complete_task")]
    board_judged = [
        item for item in records if len(item.get("editorial_judgments", [])) == 2
    ]
    blind_assessed = [item for item in realized if item.get("blind_results")]
    serious_primary = sum(
        bool(item["blind_results"][0]["serious_different_output_alternative"])
        for item in blind_assessed
    )
    pair_warning = sum(
        any(
            flag["code"]
            in {
                "over_90_percent_identical_inputs",
                "dihedral_equivalent_inputs",
                "repeated_component_or_subgrid",
            }
            for flag in item.get("deterministic_diagnostics", {}).get("flags", [])
        )
        for item in realized
    )
    final_counts = Counter(
        item["status"]
        for item in realized
        if item["status"] in {"core", "reserve", "ambiguous_reserve", "rejected"}
    )
    core_count = final_counts["core"]

    unanimous_concept = sum(
        all(
            judgment["novel_concept"] == "standout"
            for judgment in item["editorial_judgments"]
        )
        for item in board_judged
    )
    unanimous_composition = sum(
        all(
            judgment["functional_composition"] == "standout"
            for judgment in item["editorial_judgments"]
        )
        for item in board_judged
    )
    unanimous_both = sum(
        all(
            judgment["novel_concept"] == "standout"
            and judgment["functional_composition"] == "standout"
            for judgment in item["editorial_judgments"]
        )
        for item in board_judged
    )
    selected = [
        item for item in records if "editorial_brief" in item
    ]
    ready = [
        item
        for item in selected
        if item.get("architecture", {}).get("status") == "ready"
    ]
    deterministic_pass = [
        item
        for item in realized
        if item.get("deterministic_diagnostics", {}).get("hard_valid")
    ]
    technical_pass = [
        item
        for item in realized
        if any(
            result.get("passed") for result in item.get("technical_results", [])
        )
    ]
    blind_pass = [
        item
        for item in realized
        if item["status"] in {"blind_pass", "core", "reserve"}
    ]
    logical = calls["total_lean_logical_calls"]
    return {
        "funnel": {
            "initial_kernels": len(records),
            "selector_selected": len(selected),
            "architectures_ready": len(ready),
            "architectures_aborted": sum(
                item.get("architecture", {}).get("status") == "abort"
                for item in selected
            ),
            "architecture_reserves": sum(
                item["status"] == "architecture_reserve" for item in records
            ),
            "tasks_realized": len(realized),
            "deterministic_pass": len(deterministic_pass),
            "technical_pass": len(technical_pass),
            "blind_one_answer_pass": len(blind_pass),
            "editorially_judged": len(board_judged),
        },
        "core_accepted_count": final_counts["core"],
        "reserve_count": final_counts["reserve"],
        "ambiguous_reserve_count": final_counts["ambiguous_reserve"],
        "rejected_realized_count": final_counts["rejected"],
        "rejected_kernel_or_architecture_count": sum(
            item["status"] == "rejected" and not item.get("complete_task")
            for item in records
        ),
        "standout_concept_count": unanimous_concept,
        "standout_composition_count": unanimous_composition,
        "standout_on_both_count": unanimous_both,
        "standout_count_definition": "both independent boards supplied the standout rating",
        "serious_blind_alternative_rate": (
            serious_primary / len(blind_assessed) if blind_assessed else None
        ),
        "serious_blind_alternative_numerator": serious_primary,
        "serious_blind_alternative_denominator": len(blind_assessed),
        "pair_economy_warning_rate": (
            pair_warning / len(realized) if realized else None
        ),
        "pair_economy_warning_numerator": pair_warning,
        "pair_economy_warning_denominator": len(realized),
        "calls_per_realized_task": logical / len(realized) if realized else None,
        "calls_per_core_accepted_task": logical / core_count if core_count else None,
    }


def _prompt_schema_versions(run_dir: Path) -> dict[str, Any]:
    prompt_manifest = read_json(run_dir / "prompts" / "manifest.json")
    schemas: dict[str, str] = {}
    for path in sorted((run_dir / "schemas").glob("lean_*.schema.json")):
        schemas[path.stem] = content_hash(path.read_bytes())
    return {
        "prompt_manifest": prompt_manifest,
        "schema_hashes": schemas,
    }


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    calls = report["call_budget"]
    funnel = metrics["funnel"]
    lines = [
        f"# Lean ARC authoring benchmark: {report['run_id']}",
        "",
        f"Pipeline: `{report['pipeline_version']}`",
        "",
        (
            f"Completed: **{report['completed']}**. The batch produced "
            f"**{metrics['core_accepted_count']} core**, "
            f"**{metrics['reserve_count']} reserve**, "
            f"**{metrics['ambiguous_reserve_count']} ambiguous reserve**, and "
            f"**{metrics['rejected_realized_count']} rejected realized** tasks."
        ),
        "",
        "## Funnel",
        "",
        "| Stage | Count |",
        "|---|---:|",
        f"| Initial kernels | {funnel['initial_kernels']} |",
        f"| Selected for architecture | {funnel['selector_selected']} |",
        f"| Ready architectures | {funnel['architectures_ready']} |",
        f"| Aborted architectures | {funnel['architectures_aborted']} |",
        f"| Authored architecture reserves | {funnel['architecture_reserves']} |",
        f"| Realized tasks | {funnel['tasks_realized']} |",
        f"| Deterministic pass | {funnel['deterministic_pass']} |",
        f"| Technical pass | {funnel['technical_pass']} |",
        f"| One-answer blind/ambiguity pass | {funnel['blind_one_answer_pass']} |",
        f"| Editorially judged | {funnel['editorially_judged']} |",
        "",
        "## Editorial and ambiguity results",
        "",
        (
            f"- Standout concept (both boards): "
            f"{metrics['standout_concept_count']}"
        ),
        (
            f"- Standout composition (both boards): "
            f"{metrics['standout_composition_count']}"
        ),
        f"- Standout on both axes (both boards): {metrics['standout_on_both_count']}",
        (
            "- Serious blind-alternative rate: "
            + (
                "n/a"
                if metrics["serious_blind_alternative_rate"] is None
                else f"{metrics['serious_blind_alternative_rate']:.1%} "
                f"({metrics['serious_blind_alternative_numerator']}/"
                f"{metrics['serious_blind_alternative_denominator']})"
            )
        ),
        (
            "- Pair-economy warning rate: "
            + (
                "n/a"
                if metrics["pair_economy_warning_rate"] is None
                else f"{metrics['pair_economy_warning_rate']:.1%} "
                f"({metrics['pair_economy_warning_numerator']}/"
                f"{metrics['pair_economy_warning_denominator']})"
            )
        ),
        "",
        "## Model-call budget",
        "",
        "| Quantity | Calls |",
        "|---|---:|",
        f"| Normal logical calls | {calls['normal_logical_calls']} / {calls['normal_limit']} |",
        f"| Conditional escalation | {calls['escalation_logical_calls']} / {calls['escalation_limit']} |",
        f"| Lean substantive logical total | {calls['total_lean_logical_calls']} / {calls['hard_limit']} |",
        f"| Uncached successful executions | {calls['executed_uncached_calls']} |",
        f"| Cache hits | {calls['cache_hits']} |",
        f"| Transport/parse retries (excluded from substantive budget) | {calls['retry_calls']} |",
        f"| Run wall time | {report.get('run_wall_seconds') or 0:.1f} seconds |",
        f"| Summed uncached call time | {calls['successful_execution_seconds']:.1f} seconds |",
        "",
        (
            f"The frozen historical pipeline budget was "
            f"{report['historical_call_range']['minimum']}–"
            f"{report['historical_call_range']['maximum']} calls per batch. "
            f"This lean run scheduled {calls['total_lean_logical_calls']} lean "
            "substantive calls; the mandatory first-run legacy audit is reported "
            "separately and did not gate decisions."
        ),
        "",
        "Calls by stage:",
        "",
        "| Stage | Logical | Uncached | Cache hits | Retries |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, row in calls["by_stage"].items():
        lines.append(
            f"| {stage} | {row['logical_calls']} | "
            f"{row['executed_uncached_calls']} | {row['cache_hits']} | "
            f"{row['retry_calls']} |"
        )
    lines.extend(
        [
            "",
            "## Portfolio observations",
            "",
            report["portfolio_theme_clusters"]["selector_observations"]
            or "No selector portfolio observation was returned.",
            "",
        ]
    )
    for index, observation in enumerate(
        report["portfolio_theme_clusters"]["editorial_board_observations"], 1
    ):
        lines.extend([f"- Board {index}: {observation}", ""])

    lines.extend(["## Task cards", ""])
    for card in report["tasks"]:
        lines.extend(
            [
                f"### {card['candidate_id']} — {card['final_status']}",
                "",
                f"Search mode: `{card['search_mode']}`  ",
                f"Editorial route: `{card['editorial_route'] or 'not selected'}`",
                "",
                f"Central relation: {card['central_relation']}",
                "",
            ]
        )
        if card["intended_rule"]:
            lines.extend([f"Intended rule: {card['intended_rule']}", ""])
        if card["architecture_summary"]["functional_elements"]:
            lines.extend(["Functional elements:", ""])
            for element in card["architecture_summary"]["functional_elements"]:
                lines.append(
                    f"- {element['element']}: {element['role']} "
                    f"(removal: {element['what_breaks_or_simplifies_if_removed']})"
                )
            lines.append("")
        if card["complete_task"]:
            lines.extend(
                [
                    _episode_markdown(card["complete_task"]),
                    f"Full task JSON: `{card['full_task_json_path']}`",
                    "",
                    "Deterministic flags:",
                    "",
                ]
            )
            flags = card["deterministic_diagnostics"]["flags"]
            if flags:
                lines.extend(
                    f"- `{item['code']}`: {item['detail']}" for item in flags
                )
            else:
                lines.append("- None")
            lines.append("")
            if card["technical_results"]:
                final_technical = card["technical_results"][-1]
                lines.extend(
                    [
                        (
                            "Technical result: "
                            f"{'pass' if final_technical['passed'] else 'fail'}; "
                            f"hidden-output match="
                            f"{final_technical['hidden_output_match']}; "
                            f"model status="
                            f"{final_technical['response']['status']}."
                        ),
                        "",
                    ]
                )
            if card["blind_results"]:
                lines.extend(
                    [
                        (
                            "Blind result: primary output match="
                            f"{card['blind_results'][0]['primary_matches_hidden']}; "
                            "serious different-output alternative="
                            f"{card['blind_results'][0]['serious_different_output_alternative']}."
                        ),
                        "",
                    ]
                )
            if card["editorial_judgments"]:
                lines.extend(["Editorial boards:", ""])
                for index, item in enumerate(card["editorial_judgments"], 1):
                    lines.append(
                        f"- Board {index}: concept `{item['novel_concept']}`, "
                        f"composition `{item['functional_composition']}`, "
                        f"quality `{item['overall_quality']}`, decision "
                        f"`{item['decision']}` — {item['reason']}"
                    )
                lines.append("")
        reason = card["final_reason"]
        if not reason and card["rejections"]:
            reason = card["rejections"][-1]["reason"]
        lines.extend([f"Final reason: {reason or 'stage outcome recorded above.'}", ""])

    lines.extend(
        [
            "## Rejected-task summary",
            "",
            "| Candidate | Realized | Stage | Category | Reason |",
            "|---|---:|---|---|---|",
        ]
    )
    for card in report["tasks"]:
        if card["final_status"] != "rejected":
            continue
        rejection = card["rejections"][-1] if card["rejections"] else {}
        reason = str(rejection.get("reason", card["final_reason"] or "")).replace(
            "|", "\\|"
        )
        lines.append(
            f"| {card['candidate_id']} | {'yes' if card['realized'] else 'no'} | "
            f"{rejection.get('stage', 'unknown')} | "
            f"{rejection.get('category', 'unknown')} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Fixed anchor panel",
            "",
            f"Manifest: `{report['fixed_anchor_manifest_path']}`",
            "",
        ]
    )
    for anchor in report["fixed_anchor_panel"]["anchors"]:
        lines.append(
            f"- `{anchor['task_id']}` — `{anchor['editorial_role']}`: "
            f"{anchor['selection_reason']}"
        )
    lines.extend(
        [
            "",
            "## Non-gating legacy audit",
            "",
            "```json",
            json.dumps(report["legacy_audit"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Reproduction",
            "",
            "```bash",
            f"python3 -m arcgen.lib lean-run --config {report['config_path']}",
            f"python3 -m arcgen.lib lean-report --run {report['run_dir']}",
            f"python3 -m arcgen.lib lean-export --run {report['run_dir']} --status all",
            "```",
            "",
            "## Known limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def build_lean_report(run: str | Path) -> dict[str, Any]:
    run_dir = Path(run).resolve()
    state = read_json(run_dir / "state.json")
    if state.get("pipeline_version") != LEAN_PIPELINE_VERSION:
        raise LeanReportError(f"not a {PIPELINE_ID} run: {run_dir}")
    config = read_json(run_dir / "config.snapshot.json")
    records = list(state["records"].values())
    records.sort(
        key=lambda item: (
            item.get("editorial_brief", {}).get("development_priority", 99),
            item["candidate_id"],
        )
    )
    cards = [_task_card(record, run_dir) for record in records]
    calls = _call_budget(state)
    metrics = _metrics(state, cards, calls)
    boards = state.get("editorial_boards", [])
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE_ID,
        "pipeline_version": LEAN_PIPELINE_VERSION,
        "run_id": state["run_id"],
        "run_dir": str(run_dir),
        "config_path": config.get(
            "_config_path", "config/lean_authoring_v1.yaml"
        ),
        "completed": bool(state.get("completed")),
        "started_at_epoch_seconds": state.get("started_at_epoch_seconds"),
        "completed_at_epoch_seconds": state.get(
            "completed_at_epoch_seconds"
        ),
        "run_wall_seconds": state.get("run_wall_seconds"),
        "call_budget": calls,
        "historical_call_range": config["budget"]["historical_call_range"],
        "metrics": metrics,
        "tasks": cards,
        "selector": state.get("selector"),
        "portfolio_theme_clusters": {
            "within_batch_duplicate_groups": state.get("selector", {}).get(
                "within_batch_duplicate_groups", []
            ),
            "selector_observations": state.get("selector", {}).get(
                "portfolio_observations", ""
            ),
            "editorial_board_observations": [
                board["response"]["batch_observations"] for board in boards
            ],
            "portfolio_relations_by_task": {
                item["candidate_id"]: [
                    judgment["portfolio_relation"]
                    for judgment in item["editorial_judgments"]
                ]
                for item in cards
                if item["editorial_judgments"]
            },
        },
        "fixed_anchor_panel": read_json(
            run_dir / "data" / "fixed_anchor_panel.json"
        ),
        "fixed_anchor_manifest_path": str(
            (run_dir / "data" / "fixed_anchor_panel.json").resolve()
        ),
        "legacy_audit": state.get("legacy_audit", {}),
        "concept_archive_update": state.get("concept_archive_update"),
        "prompt_and_schema_versions": _prompt_schema_versions(run_dir),
        "limitations": state.get("implementation_limitations", []),
        "exports": {
            status: sorted(
                str(path.resolve())
                for path in (run_dir / "exports" / status).glob("*.json")
            )
            for status in ("core", "reserve", "ambiguous_reserve")
        },
    }
    # The private loader path is intentionally absent from config snapshots.
    report["config_path"] = str(
        Path("config/lean_authoring_v1.yaml")
    )
    atomic_write_json(run_dir / "report.json", report)
    atomic_write_text(run_dir / "report.md", _markdown(report))
    return report


def export_lean_tasks(
    run: str | Path,
    *,
    status: str = "all",
    output: str | Path | None = None,
) -> list[Path]:
    allowed = {"core", "reserve", "ambiguous_reserve"}
    if status != "all" and status not in allowed:
        raise LeanReportError(
            "status must be core, reserve, ambiguous_reserve, or all"
        )
    run_dir = Path(run).resolve()
    source_statuses = sorted(allowed) if status == "all" else [status]
    destination = (
        Path(output).resolve()
        if output
        else run_dir / "exported_tasks" / status
    )
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for source_status in source_statuses:
        for source in sorted((run_dir / "exports" / source_status).glob("*.json")):
            target = destination / f"{source_status}__{source.name}"
            atomic_write_json(target, read_json(source))
            paths.append(target)
    return paths


__all__ = [
    "LeanReportError",
    "build_lean_report",
    "export_lean_tasks",
]
