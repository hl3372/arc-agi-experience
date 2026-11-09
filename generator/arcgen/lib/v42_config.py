"""Configuration and frozen-baseline checks for ARCGEN V4.2 experiments."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError
from .util import content_hash, read_json
from .v41_config import load_v41_config


V42_PIPELINE = "arcgen-v4.2-exp-logical-leap"
V42_DESIGN = "selector_routed_logical_leap_v1"
V41_BASELINE_MANIFEST_ID = "arcgen-v4.1-palette-structural-transfer-20260804"


def verify_v41_baseline(
    repository_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Verify the exact user-approved V4.1 authoring boundary.

    V4.2 is intentionally not allowed to reinterpret whichever V4.1 source
    happens to be current.  It branches from the active-file hashes recorded
    when the milestone was frozen.
    """

    root = Path(repository_root).resolve()
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get("manifest_id") != V41_BASELINE_MANIFEST_ID:
        raise ConfigError("unexpected ARCGEN V4.1 baseline manifest")
    failures: list[str] = []
    for relative, expected in manifest.get("frozen_active_files", {}).items():
        source = root / relative
        if not source.is_file():
            failures.append(f"missing {relative}")
            continue
        actual = sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"changed {relative}: {actual} != {expected}")
    if failures:
        raise ConfigError("frozen V4.1 baseline changed: " + " | ".join(failures))

    behavior = manifest.get("active_behavior", {})
    if behavior.get("palette_scheduler", {}).get("enabled") is not True:
        raise ConfigError("V4.1 baseline does not enable the accepted palette scheduler")
    structural = behavior.get("selector_routed_structural_transfer", {})
    if structural.get("enabled") is not True:
        raise ConfigError("V4.1 baseline does not include structural-transfer planning")
    if int(structural.get("maximum_invitations_per_eight_developed_tasks", -1)) != 3:
        raise ConfigError("V4.1 structural-transfer ceiling differs from the frozen baseline")
    if "logical-leap planning" not in manifest.get("inactive_historical_treatments", []):
        raise ConfigError("V4.1 manifest does not explicitly exclude logical-leap planning")
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": content_hash(manifest),
        "verified_file_count": len(manifest.get("frozen_active_files", {})),
        "active_behavior": deepcopy(behavior),
    }


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _fraction(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not 0 <= float(value) <= 1:
        raise ConfigError(f"{field} must be between zero and one")
    return float(value)


def load_v42_config(path: str | Path) -> dict[str, Any]:
    """Load the isolated V4.2 logical-leap pilot configuration.

    Internally the run continues to use the V4.1 experimental runtime so its
    proven information boundaries and downstream machinery remain unchanged.
    The public V4.2 identity is recorded separately and checked here.
    """

    config = load_v41_config(path, allow_experiment=True)
    root = Path(config["_repository_root"])
    if config.get("v4_2_pipeline") != V42_PIPELINE:
        raise ConfigError(f"v4_2_pipeline must be {V42_PIPELINE!r}")
    experimental = config.get("experimental", {})
    if experimental.get("design") != V42_DESIGN:
        raise ConfigError(f"experimental.design must be {V42_DESIGN!r}")
    if experimental.get("feature") != "supported_departure":
        raise ConfigError("V4.2 must remain isolated in the unpromoted semantic experiment slot")
    if config.get("pipeline") != "arcgen-v4.1-exp-supported-departure":
        raise ConfigError("V4.2 internal runtime pipeline differs from its frozen V4.1 parent")
    if config.get("_effective_features", {}).get("palette_scheduler") is not True:
        raise ConfigError("V4.2 must inherit the accepted palette scheduler")
    if config.get("run", {}).get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("V4.2 experimental tasks may not be committed automatically")

    manifest_value = config.get("paths", {}).get("v4_1_baseline_manifest")
    if not manifest_value:
        raise ConfigError("paths.v4_1_baseline_manifest is required")
    config["_v4_1_baseline_verification"] = verify_v41_baseline(root, manifest_value)

    settings = config.get("logical_leap")
    if not isinstance(settings, Mapping):
        raise ConfigError("logical_leap must be a mapping")
    if _integer(settings.get("independent_slates"), "logical_leap.independent_slates", minimum=1) != 1:
        raise ConfigError("the first V4.2 pilot is intentionally one slate")
    developed = _integer(
        settings.get("maximum_development_candidates"),
        "logical_leap.maximum_development_candidates",
        minimum=1,
    )
    if developed != 8:
        raise ConfigError("the qualified pilot develops exactly eight candidates")
    maximum = _integer(
        settings.get("maximum_challenge_nominations_per_slate"),
        "logical_leap.maximum_challenge_nominations_per_slate",
    )
    logical = _integer(
        settings.get("logical_leap_invitations_per_slate"),
        "logical_leap.logical_leap_invitations_per_slate",
    )
    structural = _integer(
        settings.get("structural_transfer_controls_per_slate"),
        "logical_leap.structural_transfer_controls_per_slate",
    )
    ordinary = _integer(
        settings.get("ordinary_planning_minimum"),
        "logical_leap.ordinary_planning_minimum",
    )
    if maximum != 3 or logical != 2 or structural != 1 or ordinary != 5:
        raise ConfigError("V4.2 pilot routing must remain 2 logical + 1 structural + at least 5 ordinary")
    if logical + structural > maximum or developed - maximum < ordinary:
        raise ConfigError("V4.2 routing ceilings cannot preserve the ordinary majority")
    if settings.get("paired_task_variants") is not False:
        raise ConfigError("V4.2 creates one task instance per kernel")
    if settings.get("selector_ranking_must_precede_nomination") is not True:
        raise ConfigError("quality selection and ranking must precede V4.2 routing")
    _integer(settings.get("audit_batch_size"), "logical_leap.audit_batch_size", minimum=1)
    _integer(
        settings.get("second_opinion_budget_per_slate"),
        "logical_leap.second_opinion_budget_per_slate",
    )
    exclusion_value = settings.get("replication_exclusion_manifest")
    if exclusion_value is not None:
        exclusion_path = Path(str(exclusion_value))
        if not exclusion_path.is_absolute():
            exclusion_path = root / exclusion_path
        exclusion_path = exclusion_path.resolve()
        if not exclusion_path.is_file():
            raise ConfigError(
                "logical_leap.replication_exclusion_manifest must name an existing file"
            )
        exclusion = read_json(exclusion_path)
        proposal_ids = exclusion.get("proposal_ids")
        if (
            not isinstance(proposal_ids, list)
            or not proposal_ids
            or any(not isinstance(item, str) or not item for item in proposal_ids)
            or len(proposal_ids) != len(set(proposal_ids))
        ):
            raise ConfigError(
                "the replication exclusion manifest must contain unique proposal_ids"
            )
    if _integer(
        settings.get("production_initial_logical_max_per_eight"),
        "logical_leap.production_initial_logical_max_per_eight",
    ) != 1:
        raise ConfigError("future production integration must begin at at most one logical invitation per eight")
    lower = _fraction(settings.get("rolling_target_lower"), "logical_leap.rolling_target_lower")
    upper = _fraction(settings.get("rolling_target_upper"), "logical_leap.rolling_target_upper")
    ceiling = _fraction(settings.get("rolling_soft_ceiling"), "logical_leap.rolling_soft_ceiling")
    if not lower < upper < ceiling or (lower, upper, ceiling) != (0.10, 0.20, 0.25):
        raise ConfigError("V4.2 rolling target must remain 10–20% with a 25% soft ceiling")
    config["logical_leap"] = deepcopy(dict(settings))
    if exclusion_value is not None:
        config["logical_leap"]["replication_exclusion_manifest"] = str(
            exclusion_path
        )
    return config


__all__ = [
    "V41_BASELINE_MANIFEST_ID",
    "V42_DESIGN",
    "V42_PIPELINE",
    "load_v42_config",
    "verify_v41_baseline",
]
