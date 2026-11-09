"""Resolve non-destructive accepted-bank audit flags for active use.

The immutable registry is the historical ledger.  A later safety review may
clear or quarantine an entry without rewriting that history.  This module is
the single deterministic precedence rule for current reference consumers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .util import read_json


def _decision(value: Mapping[str, Any]) -> tuple[int, bool] | None:
    status = str(value.get("status", ""))
    if status == "clear_dominant_e19":
        return 100, True
    if status.startswith("quarantine_e19") or status == "quarantine_superseded_by_e19_repair":
        return 100, False
    if status.startswith("quarantine"):
        return 10, False
    return None


def active_bank_audit_decisions(bank: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(bank).resolve()
    flags = root / "audit_flags"
    resolved: dict[str, dict[str, Any]] = {}
    for path in sorted(flags.glob("*.json")) if flags.is_dir() else []:
        value = read_json(path)
        task_id = value.get("registry_task_id")
        decision = _decision(value) if isinstance(value, Mapping) else None
        if not isinstance(task_id, str) or decision is None:
            continue
        priority, active = decision
        prior = resolved.get(task_id)
        if prior is not None and int(prior["priority"]) > priority:
            continue
        resolved[task_id] = {
            "registry_task_id": task_id,
            "active": active,
            "status": value["status"],
            "priority": priority,
            "source_flag": str(path.relative_to(root)),
        }
    return resolved


def quarantined_registry_task_ids(bank: str | Path) -> list[str]:
    decisions = active_bank_audit_decisions(bank)
    return sorted(task_id for task_id, value in decisions.items() if not value["active"])


def filter_active_registry(
    registry: Mapping[str, Any], *, bank: str | Path
) -> dict[str, Any]:
    excluded = set(quarantined_registry_task_ids(bank))
    result = deepcopy(dict(registry))
    result["entries"] = [
        deepcopy(item)
        for item in registry.get("entries", [])
        if item.get("registry_task_id") not in excluded
    ]
    active_counts = Counter(item.get("commit_set_id") for item in result["entries"])
    if isinstance(result.get("commit_sets"), list):
        for item in result["commit_sets"]:
            if isinstance(item, dict) and isinstance(item.get("commit_set_id"), str):
                item["entry_count"] = active_counts[item["commit_set_id"]]
    result["active_bank_filter"] = {
        "historical_registry_entry_count": len(registry.get("entries", [])),
        "active_entry_count": len(result["entries"]),
        "excluded_quarantined_registry_task_ids": sorted(excluded),
        "historical_registry_mutated": False,
    }
    return result


__all__ = [
    "active_bank_audit_decisions",
    "filter_active_registry",
    "quarantined_registry_task_ids",
]
