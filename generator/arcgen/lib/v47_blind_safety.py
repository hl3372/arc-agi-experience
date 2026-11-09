"""Fail-safe normalization for blind-solver ambiguity results.

Historical V4 wrappers sometimes summarized a blind result as clear even when
the solver's own primary response reported a serious different-output rule.
That happened when the hidden author answer appeared in the solver's
``alternatives`` list while the solver's primary prediction was the competing
answer. New qualification and boardless paths use this additive helper rather
than changing any frozen historical pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def effective_serious_blind_concern(blind: Mapping[str, Any]) -> bool | None:
    """Return the effective unresolved serious-alternative result.

    A complete adjudication or a successfully revalidated repair may resolve a
    raw primary concern. Without either, conflicting wrapper and raw response
    flags are resolved conservatively: any explicit serious signal wins.
    """

    if blind.get("operational_failure") is True:
        return None

    adjudication = blind.get("ambiguity_adjudication") or blind.get("adjudication")
    if isinstance(adjudication, Mapping) and not adjudication.get("operational_failure"):
        decision = adjudication.get("decision")
        different = adjudication.get("comparably_simple_different_output_rule")
        if decision == "clear" and different is False:
            return False
        if decision in {
            "repairable",
            "serious_ambiguity",
            "ambiguous",
            "reject",
        } or different is True:
            return True
        # An incomplete adjudication cannot establish safety.
        return None

    repair_count = blind.get("repair_count")
    resolution = blind.get("effective_resolution_kind")
    record_status = blind.get("record_status") or blind.get("status")
    explicit = blind.get("effective_serious_different_output_alternative")
    if (
        isinstance(repair_count, int)
        and not isinstance(repair_count, bool)
        and repair_count > 0
        and record_status in {None, "blind_pass"}
        and resolution in {None, "targeted_repair_revalidated"}
        and explicit is False
    ):
        return False
    if resolution == "targeted_repair_revalidated" and explicit is False:
        return False

    response = blind.get("response")
    candidates = [
        explicit,
        blind.get("serious_different_output_alternative"),
        blind.get("primary_serious_different_output_alternative"),
        response.get("serious_different_output_alternative")
        if isinstance(response, Mapping)
        else None,
    ]
    booleans = [value for value in candidates if type(value) is bool]
    if any(booleans):
        return True
    if booleans:
        return False
    return None


__all__ = ["effective_serious_blind_concern"]
