"""Call-free descriptive measurements; never a selection or acceptance gate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
from typing import Any

from .e18_artifacts import strict_technical_clear
from .e18_evidence_gate import evidence_is_clear
from .v4_diagnostics import diagnose_arc_task_v4


def task_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    episode = record.get("complete_task")
    diagnostic = diagnose_arc_task_v4(episode) if episode else {}
    palette = record.get("palette_treatment") or {}
    safety = palette.get("safety_assessment") or {}
    background = next((value for value in (
        palette.get("target_background") if palette.get("eligible") else palette.get("source_background"),
        palette.get("realized_background_color"), palette.get("source_background_color"),
        safety.get("declared_background_color"),
    ) if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 9), None)
    grids, pairs = [], []
    if diagnostic.get("hard_valid"):
        for split in ("train", "test"):
            for index, pair in enumerate(episode[split], 1):
                areas = {}
                for side in ("input", "output"):
                    grid = pair[side]
                    height, width = len(grid), len(grid[0])
                    colors = Counter(cell for row in grid for cell in row)
                    area = height * width
                    areas[side] = area
                    grids.append({
                        "split": split, "pair": index, "side": side,
                        "height": height, "width": width, "area": area,
                        "color_count": len(colors), "color_counts": {str(color): count for color, count in sorted(colors.items())},
                        "declared_background_color": background,
                        "declared_background_occupancy": (
                            (area - colors[background]) / area if background is not None else None),
                        "dominant_color_fraction": max(colors.values()) / area,
                    })
                pairs.append({"split": split, "pair": index,
                              "output_input_area_ratio": areas["output"] / areas["input"]})
    technical = record.get("technical_results") or []
    terminal_response = technical[-1].get("response", {}) if technical else {}
    shortcuts = terminal_response.get("serious_concrete_shortcuts") or []
    blind = record.get("blind_results") or []
    evidence = record.get("final_episode_evidence") or {}
    plan = record.get("concrete_plan") or record.get("plan")
    # Legacy records often retain the wrapper instead of a standalone plan.
    if not plan:
        planning = record.get("planning_result") or {}
        plan = planning.get("plan") or planning.get("concrete_plan") or planning or None
    maximum = max((max(row["height"], row["width"]) for row in grids), default=0)
    return {
        "source": record.get("private_source", "unknown"),
        "route": record.get("private_route", "unknown"),
        "search_mode": record.get("search_mode") or (record.get("kernel") or {}).get("search_mode", "unknown"),
        "status": record.get("status"), "realized": bool(episode),
        "deterministic_valid": diagnostic.get("hard_valid") if episode else None,
        "plan_fidelity": (record.get("plan_fidelity") or {}).get("passed"),
        "technical_clear": strict_technical_clear(record),
        "technical_serious_shortcuts": shortcuts,
        "technical_verdict_contradiction": bool(shortcuts and terminal_response.get("status") == "pass"),
        "terminal_technical_hidden_match": technical[-1].get("hidden_output_match") if technical else None,
        "blind_reviews": len(blind),
        "final_episode_blind_reviews": evidence.get("final_episode_blind_reviews"),
        "raw_blind_alarm_count": evidence.get("raw_blind_alarm_count"),
        "effective_terminal_ambiguity": evidence.get("effective_terminal_ambiguity"),
        "evidence_clear": evidence_is_clear(record),
        "intrinsic_quality": record.get("intrinsic_quality"),
        "portfolio_placement": record.get("portfolio_placement"),
        "curriculum_status": record.get("curriculum_status"),
        "repair_count": record.get("repair_count", 0),
        "plan_json_characters": len(json.dumps(plan, ensure_ascii=False, sort_keys=True)) if plan else None,
        "train_pair_count": len(episode["train"]) if episode else None,
        "test_pair_count": len(episode["test"]) if episode else None,
        "maximum_dimension": maximum or None,
        "size_band": ("compact" if maximum <= 10 else "medium" if maximum <= 20 else "large") if maximum else "unrealized_or_invalid",
        "grids": grids, "pair_area_ratios": pairs,
        "editorial_judgments": record.get("editorial_judgments") or [],
        "mechanism_audits": record.get("mechanism_audits") or [],
        **({"mechanism_audit_status": "not_requested", "audit_feature_prevalence": None}
           if record.get("mechanism_audit_status") == "not_requested" else {}),
    }


def cohort_metrics(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = {key: task_metrics(row) for key, row in sorted(records.items())}
    return {
        "schema_version": 1, "non_gating": True, "tasks": rows,
        "selected_denominator": len(rows),
        "source_counts": dict(Counter(row["source"] for row in rows.values())),
        "search_mode_counts": dict(Counter(row["search_mode"] for row in rows.values())),
        "size_band_counts": dict(Counter(row["size_band"] for row in rows.values())),
        "stage_counts": {field: sum(row[field] is True for row in rows.values()) for field in (
            "realized", "deterministic_valid", "plan_fidelity", "technical_clear", "evidence_clear")},
        "technical_contradictions_for_independent_review": [
            key for key, row in rows.items() if row["technical_verdict_contradiction"]],
        "effective_ambiguity": {
            "confirmed": sum(row["effective_terminal_ambiguity"] is True for row in rows.values()),
            "clear": sum(row["effective_terminal_ambiguity"] is False for row in rows.values()),
            "unknown": sum(row["effective_terminal_ambiguity"] is None for row in rows.values()),
        },
        "limitations": [
            "Selected-candidate denominator includes failed candidates; proposal/shard/global funnels are separate saved selection artifacts.",
            "Absent background declarations remain unknown, not assumed black. Dominant-color fraction is not a background or ontology classification.",
            "Grid size, color use, pair ratios and serialized plan length are descriptive, not quality gates or token counts.",
            "Retained mechanism judgments use the existing research instruments and are not independently confirmed construct validity.",
        ],
    }
