"""Explicit, ordered feature-promotion state for ARCGEN V4.1.

Experimental runners only read this manifest.  The sole writer requires an
explicit user-decision flag and records the run and authorization text.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, Mapping
import fcntl
import time

from .util import atomic_write_json, content_hash, read_json


FEATURE_ORDER = (
    "palette_diversity",
    "supported_departure",
    "episode_joint_inference",
    "abstraction_led",
)
FEATURE_TO_FLAG = {
    "palette_diversity": "palette_scheduler",
    "supported_departure": "supported_departure_planning",
    "episode_joint_inference": "episode_joint_reservoir_stream",
    "abstraction_led": "abstraction_led_reservoir_stream",
}
FEATURE_STATUSES = {"proposed", "testing", "accepted", "rejected"}
TERMINAL_FEATURE_STATUSES = {"accepted", "rejected"}


class V41FeatureError(ValueError):
    pass


def validate_feature_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise V41FeatureError("V4.1 feature manifest must be an object")
    if value.get("manifest_id") != "arcgen-v4.1-feature-promotion-20260803":
        raise V41FeatureError("unexpected V4.1 feature manifest")
    if value.get("promotion_order") != list(FEATURE_ORDER):
        raise V41FeatureError("V4.1 promotion order changed")
    if value.get("automatic_promotion_allowed") is not False:
        raise V41FeatureError("automatic V4.1 promotion must remain disabled")
    features = value.get("features")
    if not isinstance(features, dict) or set(features) != set(FEATURE_ORDER):
        raise V41FeatureError("V4.1 feature manifest has the wrong feature set")
    for feature in FEATURE_ORDER:
        item = features[feature]
        if not isinstance(item, dict) or item.get("status") not in FEATURE_STATUSES:
            raise V41FeatureError(f"invalid status for {feature}")
        if not isinstance(item.get("decision_history"), list):
            raise V41FeatureError(f"{feature} decision history must be a list")
        if item["status"] == "accepted":
            if not isinstance(item.get("accepted_run_id"), str) or not item["accepted_run_id"]:
                raise V41FeatureError(f"accepted {feature} lacks its run ID")
            if not isinstance(item.get("authorization_record"), str) or not item["authorization_record"]:
                raise V41FeatureError(f"accepted {feature} lacks authorization")
        elif item.get("accepted_run_id") is not None:
            raise V41FeatureError(f"non-accepted {feature} has an accepted run ID")
    return deepcopy(value)


def load_feature_manifest(path: str | Path) -> dict[str, Any]:
    return validate_feature_manifest(read_json(Path(path)))


def prior_features(feature: str) -> tuple[str, ...]:
    if feature not in FEATURE_ORDER:
        raise V41FeatureError(f"unknown V4.1 feature {feature!r}")
    return FEATURE_ORDER[: FEATURE_ORDER.index(feature)]


def assert_prior_decisions_complete(manifest: Mapping[str, Any], feature: str) -> None:
    unresolved = [
        item for item in prior_features(feature)
        if manifest["features"][item]["status"] not in TERMINAL_FEATURE_STATUSES
    ]
    if unresolved:
        raise V41FeatureError(
            f"{feature} cannot start before explicit decisions for: {', '.join(unresolved)}"
        )


def accepted_feature_flags(manifest: Mapping[str, Any]) -> dict[str, bool]:
    return {
        FEATURE_TO_FLAG[feature]: manifest["features"][feature]["status"] == "accepted"
        for feature in FEATURE_ORDER
    }


@contextmanager
def _manifest_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def authorize_feature_decision(
    manifest_path: str | Path,
    *,
    feature: str,
    decision: str,
    run_dir: str | Path,
    authorization_record: str,
    explicit_user_authorization: bool,
    now: float | None = None,
) -> dict[str, Any]:
    """Apply one explicit user decision; never called by an experiment runner."""

    if not explicit_user_authorization:
        raise V41FeatureError("feature promotion requires explicit user authorization")
    if decision not in {"accepted", "rejected"}:
        raise V41FeatureError("feature decision must be accepted or rejected")
    if not isinstance(authorization_record, str) or not authorization_record.strip():
        raise V41FeatureError("an auditable user authorization record is required")
    if feature not in FEATURE_ORDER:
        raise V41FeatureError(f"unknown V4.1 feature {feature!r}")
    run_path = Path(run_dir).resolve()
    report_path = run_path / "report.json"
    if not report_path.is_file():
        raise V41FeatureError("promotion source run has no report.json")
    report = read_json(report_path)
    if report.get("feature") != feature or not report.get("completed"):
        raise V41FeatureError("promotion source is not a completed report for this feature")
    path = Path(manifest_path).resolve()
    with _manifest_lock(path):
        manifest = load_feature_manifest(path)
        assert_prior_decisions_complete(manifest, feature)
        current = manifest["features"][feature]
        if current["status"] in TERMINAL_FEATURE_STATUSES:
            raise V41FeatureError(f"{feature} already has a terminal user decision")
        moment = round(time.time() if now is None else now, 3)
        event = {
            "decision": decision,
            "run_id": report.get("run_id", run_path.name),
            "run_path": str(run_path),
            "report_sha256": content_hash(report_path.read_bytes()),
            "authorization_record": authorization_record.strip(),
            "authorized_at_epoch_seconds": moment,
            "report_recommendation": report.get("promotion_recommendation"),
        }
        current["status"] = decision
        current["accepted_run_id"] = event["run_id"] if decision == "accepted" else None
        current["authorization_record"] = authorization_record.strip()
        current["decision_history"].append(event)
        manifest["last_updated_at_epoch_seconds"] = moment
        atomic_write_json(path, validate_feature_manifest(manifest))
        return deepcopy(manifest)


__all__ = [
    "FEATURE_ORDER",
    "FEATURE_STATUSES",
    "FEATURE_TO_FLAG",
    "TERMINAL_FEATURE_STATUSES",
    "V41FeatureError",
    "accepted_feature_flags",
    "assert_prior_decisions_complete",
    "authorize_feature_decision",
    "load_feature_manifest",
    "prior_features",
    "validate_feature_manifest",
]
