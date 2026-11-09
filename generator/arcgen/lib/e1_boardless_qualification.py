"""E1 boardless-disposition qualification harness.

This module is deliberately downstream of a selected, frozen production
baseline.  It does not run authoring models and it cannot commit tasks.  A
baseline-specific runner exports normalized pre-editorial evidence; this
harness freezes the boardless dispositions before a separate, non-gating
shadow-editorial result may be attached.

The adapter is version-neutral: V4.4, V4.5, V4.6, or a composite winner use
the same selected-baseline and evidence manifests.  Consequently E1 does not
need to import or modify any frozen pipeline implementation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from .util import atomic_write_json, atomic_write_text, content_hash, read_json
from .v47_blind_safety import effective_serious_blind_concern


PIPELINE = "arcgen-e1-boardless-qualification"
VERSION = "1.0"
VERIFIED = "verified_candidate"
NOT_VERIFIED = "not_verified_candidate"


class E1QualificationError(RuntimeError):
    """Raised when E1 isolation, provenance, or evidence is invalid."""


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E1QualificationError(f"expected a mapping: {path}")
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _require_bool(mapping: Mapping[str, Any], key: str, expected: bool) -> None:
    if mapping.get(key) is not expected:
        raise E1QualificationError(f"{key} must be {str(expected).lower()}")


def _verify_file_ref(value: Mapping[str, Any], *, base: Path, label: str) -> Path:
    raw_path = value.get("path")
    expected = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise E1QualificationError(f"{label}.path is required")
    if not isinstance(expected, str) or len(expected) != 64:
        raise E1QualificationError(f"{label}.sha256 must be a SHA-256 digest")
    path = _resolve(base, raw_path)
    if not path.is_file():
        raise E1QualificationError(f"{label} does not exist: {path}")
    observed = _sha(path)
    if observed != expected:
        raise E1QualificationError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return path


def load_e1_config(path: str | Path) -> dict[str, Any]:
    """Load an executable E1 configuration.

    The distributed configuration is a non-executable template.  Both
    ``template`` and ``enabled`` must be explicitly changed, which prevents a
    placeholder baseline from being used accidentally.
    """

    source = Path(path).resolve()
    value = _read_yaml_or_json(source)
    if value.get("pipeline") != PIPELINE or str(value.get("version")) != VERSION:
        raise E1QualificationError(f"config must target {PIPELINE} {VERSION}")
    if value.get("template") is not False or value.get("enabled") is not True:
        raise E1QualificationError(
            "E1 template is non-executable; create a run-specific config with "
            "template=false and enabled=true"
        )
    base = source.parent.parent if source.parent.name == "config" else source.parent
    run = value.get("run")
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise E1QualificationError("run.id is required")
    _require_bool(run, "automatically_commit_generated_tasks", False)
    _require_bool(run, "production_state_mutation_allowed", False)
    _require_bool(run, "shadow_editorial_gating", False)
    for key in (
        "selected_baseline_manifest",
        "selected_baseline_manifest_sha256",
        "evidence_manifest",
        "evidence_manifest_sha256",
        "output_directory",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise E1QualificationError(f"{key} is required")
    for key in ("selected_baseline_manifest_sha256", "evidence_manifest_sha256"):
        if len(value[key]) != 64:
            raise E1QualificationError(f"{key} must be a SHA-256 digest")
    value["_config_path"] = str(source)
    value["_repository_root"] = str(base)
    value["_baseline_path"] = str(_resolve(base, value["selected_baseline_manifest"]))
    value["_evidence_path"] = str(_resolve(base, value["evidence_manifest"]))
    value["_output_directory"] = str(_resolve(base, value["output_directory"]))
    if _sha(Path(value["_baseline_path"])) != value["selected_baseline_manifest_sha256"]:
        raise E1QualificationError("selected baseline manifest hash mismatch")
    if _sha(Path(value["_evidence_path"])) != value["evidence_manifest_sha256"]:
        raise E1QualificationError("evidence manifest hash mismatch")
    shadow = value.get("shadow_policy")
    if not isinstance(shadow, Mapping):
        raise E1QualificationError("shadow_policy is required")
    if shadow.get("reviewer_count") != 2:
        raise E1QualificationError("shadow review requires exactly two reviewers")
    _require_bool(shadow, "gating", False)
    _require_bool(shadow, "defer_until_all_dispositions_frozen", True)
    if shadow.get("generated_group_minimum") != 4 or shadow.get("generated_group_maximum") != 6:
        raise E1QualificationError("shadow generated groups must normally contain 4-6 tasks")
    return value


class NormalizedSelectedBaselineAdapter:
    """Validate and expose a version-neutral selected-baseline manifest."""

    REQUIRED_FROZEN_INPUTS = (
        "committed_registry",
        "reservoir_or_window",
        "novelty_memory",
        "fixed_anchor_manifest",
        "historical_reference_manifest",
    )

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.base = self.path.parent
        self.value = read_json(self.path)
        self._validate()

    def _validate(self) -> None:
        value = self.value
        if value.get("schema_version") != 1:
            raise E1QualificationError("selected baseline schema_version must be 1")
        baseline = value.get("baseline")
        if not isinstance(baseline, Mapping):
            raise E1QualificationError("selected baseline.baseline is required")
        for key in ("id", "kind", "runner_module", "runner_version"):
            if not isinstance(baseline.get(key), str) or not baseline[key]:
                raise E1QualificationError(f"baseline.{key} is required")
        for key in ("manifest", "config"):
            ref = baseline.get(key)
            if not isinstance(ref, Mapping):
                raise E1QualificationError(f"baseline.{key} file reference is required")
            _verify_file_ref(ref, base=self.base, label=f"baseline.{key}")
        optional = baseline.get("optional_supply", {})
        if not isinstance(optional, Mapping):
            raise E1QualificationError("baseline.optional_supply must be a mapping")
        if optional.get("enabled") is True:
            digest = optional.get("provider_schedule_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise E1QualificationError(
                    "enabled optional supply must freeze provider_schedule_sha256"
                )
            if type(optional.get("auxiliary_routes_enabled")) is not bool:
                raise E1QualificationError(
                    "optional supply must explicitly freeze auxiliary_routes_enabled"
                )
        frozen = value.get("frozen_inputs")
        if not isinstance(frozen, Mapping):
            raise E1QualificationError("frozen_inputs is required")
        for key in self.REQUIRED_FROZEN_INPUTS:
            ref = frozen.get(key)
            if not isinstance(ref, Mapping):
                raise E1QualificationError(f"frozen_inputs.{key} is required")
            _verify_file_ref(ref, base=self.base, label=f"frozen_inputs.{key}")
        if frozen["fixed_anchor_manifest"].get("count") != 6:
            raise E1QualificationError("the selected control must freeze exactly six anchors")
        prompts = value.get("prompts")
        if not isinstance(prompts, Mapping) or not prompts:
            raise E1QualificationError("at least one frozen prompt is required")
        for name, item in prompts.items():
            if not isinstance(item, Mapping):
                raise E1QualificationError(f"prompts.{name} must be a mapping")
            if not isinstance(item.get("version"), str) or not item["version"]:
                raise E1QualificationError(f"prompts.{name}.version is required")
            if not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64:
                raise E1QualificationError(f"prompts.{name}.sha256 is invalid")
            if item.get("path") is not None:
                _verify_file_ref(item, base=self.base, label=f"prompts.{name}")
        routes = value.get("model_routes")
        if not isinstance(routes, Mapping) or not routes:
            raise E1QualificationError("model_routes is required")
        for role, item in routes.items():
            if not isinstance(item, Mapping):
                raise E1QualificationError(f"model_routes.{role} must be a mapping")
            for key in ("provider", "model", "reasoning_effort"):
                if not isinstance(item.get(key), str) or not item[key]:
                    raise E1QualificationError(f"model_routes.{role}.{key} is required")
        policy = value.get("experimental_policy")
        if not isinstance(policy, Mapping):
            raise E1QualificationError("experimental_policy is required")
        _require_bool(policy, "routine_editorial_gating", False)
        _require_bool(policy, "shadow_editorial_enabled", True)
        _require_bool(policy, "shadow_editorial_gating", False)
        _require_bool(policy, "automatic_commit", False)
        _require_bool(policy, "production_state_mutation_allowed", False)

    @property
    def sha256(self) -> str:
        return _sha(self.path)

    @property
    def baseline_id(self) -> str:
        return str(self.value["baseline"]["id"])

    def frozen_snapshot(self) -> dict[str, Any]:
        return deepcopy(self.value)


def effective_blind_serious(blind: Mapping[str, Any]) -> bool | None:
    """Return the effective unresolved ambiguity after any adjudication.

    The original primary solver concern remains in the dossier, but a clear
    adjudication overrides it.  Conversely, an ambiguous adjudication overrides
    an optimistic top-level summary.
    """

    return effective_serious_blind_concern(blind)


def _path(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _empty(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 0


@dataclass(frozen=True)
class GateCriterion:
    code: str
    description: str
    predicate: Callable[[Mapping[str, Any]], bool]


GATE_CRITERIA: tuple[GateCriterion, ...] = (
    GateCriterion("deterministic_valid", "deterministic ARC validation passes", lambda e: _path(e, "deterministic.valid") is True),
    GateCriterion("deterministic_hard_valid", "hard deterministic diagnostics pass", lambda e: _path(e, "deterministic.hard_valid") is True),
    GateCriterion("realization_operational", "realization completed without operational failure", lambda e: _path(e, "realization.operational_failure") is False),
    GateCriterion("plan_fidelity", "programmatic plan fidelity passes", lambda e: _path(e, "plan_fidelity.passed") is True),
    GateCriterion("technical_operational", "technical verification completed", lambda e: _path(e, "technical.operational_failure") is False),
    GateCriterion("technical_pass", "technical verifier passes", lambda e: _path(e, "technical.passed") is True),
    GateCriterion("hidden_output_exact", "independently derived hidden output matches exactly", lambda e: _path(e, "technical.hidden_output_match") is True),
    GateCriterion("mechanically_correct", "technical response confirms mechanical correctness", lambda e: _path(e, "technical.response.mechanically_correct") is True),
    GateCriterion("test_supported", "technical response confirms the test is supported", lambda e: _path(e, "technical.response.test_supported") is True),
    GateCriterion("one_answer_supported", "one expected answer is supported", lambda e: _path(e, "technical.response.one_answer_supported") is True),
    GateCriterion("composition_preserved", "functional composition survived realization", lambda e: _path(e, "technical.response.composition_preserved") is True),
    GateCriterion("technical_plan_fidelity", "technical verifier finds no plan-fidelity failure", lambda e: _path(e, "technical.response.plan_fidelity") not in {None, "fail", False}),
    GateCriterion("evidence_sufficient", "training evidence is sufficient", lambda e: _path(e, "technical.response.evidence_sufficiency") == "sufficient"),
    GateCriterion("pair_economy", "pair economy has no failure", lambda e: _path(e, "technical.response.pair_economy") not in {None, "fail"}),
    GateCriterion("no_arbitrary_exceptions", "no arbitrary exceptions remain", lambda e: _empty(_path(e, "technical.response.arbitrary_exceptions"))),
    GateCriterion("no_unsupported_test_semantics", "no unsupported test semantics remain", lambda e: _empty(_path(e, "technical.response.unsupported_test_semantics"))),
    GateCriterion("no_unapproved_realizer_decisions", "realizer made no unapproved semantic decision", lambda e: _empty(_path(e, "technical.response.unapproved_realizer_decisions"))),
    GateCriterion("no_unresolved_pair_economy_flags", "no unresolved pair-economy flag remains", lambda e: _empty(_path(e, "technical.unresolved_pair_economy_flags"))),
    GateCriterion("blind_operational", "blind verification completed", lambda e: _path(e, "blind.operational_failure") is False),
    GateCriterion("no_effective_serious_alternative", "no adjudicated serious different-output alternative remains", lambda e: effective_blind_serious(_path(e, "blind") or {}) is False),
    GateCriterion("selector_archive_checked", "selector/archive novelty checks were performed", lambda e: _path(e, "novelty.selector_archive_checked") is True),
    GateCriterion("no_confirmed_same_procedure_duplicate", "no confirmed same-procedure duplicate exists", lambda e: _path(e, "novelty.confirmed_same_procedure_duplicate") is False),
)


def evaluate_boardless_candidate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the declarative E1 gate without consulting editorial judgments."""

    failures = [
        {"criterion": item.code, "reason": item.description}
        for item in GATE_CRITERIA
        if not item.predicate(evidence)
    ]
    status = VERIFIED if not failures else NOT_VERIFIED
    return {
        "boardless_disposition": status,
        "boardless_reasons": failures,
        "commit_eligible": False,
        "export_as_core_eligible": False,
    }


_FORBIDDEN_PRE_SHADOW_KEYS = {
    "editorial",
    "editorial_board",
    "boards",
    "board_judgments",
    "shadow_editorial",
    "intrinsic_quality",
    "portfolio_placement",
    "experimental_disposition",
}


def _assert_pre_shadow(value: Any, *, context: str = "evidence") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_PRE_SHADOW_KEYS:
                raise E1QualificationError(
                    f"pre-shadow {context} contains forbidden editorial field: {key}"
                )
            _assert_pre_shadow(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_pre_shadow(item, context=f"{context}[{index}]")


_ACCOUNTING_FIELDS = (
    "substantive_logical_calls",
    "transport_retry_calls",
    "cache_hits",
    "active_model_seconds",
    "queue_wait_seconds",
    "input_tokens",
    "output_tokens",
)


def _accounting(value: Mapping[str, Any] | None) -> dict[str, Any]:
    source = value or {}
    result: dict[str, Any] = {}
    for key in _ACCOUNTING_FIELDS:
        item = source.get(key, 0)
        if not isinstance(item, (int, float)) or item < 0:
            raise E1QualificationError(f"invalid accounting field {key}")
        result[key] = item
    roles = source.get("calls_by_role", {})
    if not isinstance(roles, Mapping) or any(
        not isinstance(key, str) or type(item) is not int or item < 0
        for key, item in roles.items()
    ):
        raise E1QualificationError("calls_by_role must contain non-negative integers")
    result["calls_by_role"] = dict(sorted(roles.items()))
    return result


def _add_accounting(*values: Mapping[str, Any]) -> dict[str, Any]:
    result = _accounting({})
    role_counts: dict[str, int] = {}
    for value in values:
        row = _accounting(value)
        for key in _ACCOUNTING_FIELDS:
            result[key] += row[key]
        for role, count in row["calls_by_role"].items():
            role_counts[role] = role_counts.get(role, 0) + count
    result["calls_by_role"] = dict(sorted(role_counts.items()))
    return result


def load_evidence_manifest(path: str | Path, *, baseline_sha256: str) -> dict[str, Any]:
    source = Path(path).resolve()
    value = read_json(source)
    if value.get("schema_version") != 1:
        raise E1QualificationError("evidence schema_version must be 1")
    if value.get("selected_baseline_manifest_sha256") != baseline_sha256:
        raise E1QualificationError("evidence was not produced from the selected baseline")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise E1QualificationError("evidence.sources must be non-empty")
    base = source.parent
    source_ids: set[str] = set()
    for item in sources:
        if not isinstance(item, Mapping) or not isinstance(item.get("source_id"), str):
            raise E1QualificationError("each source requires source_id")
        if item["source_id"] in source_ids:
            raise E1QualificationError("duplicate evidence source_id")
        source_ids.add(item["source_id"])
        _verify_file_ref(item, base=base, label=f"source {item['source_id']}")
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise E1QualificationError("evidence.candidates must be a list")
    candidate_ids: set[str] = set()
    allowed_candidate_keys = {
        "candidate_id",
        "source_id",
        "task",
        "evidence",
        "lineage_id",
        "search_mode",
    }
    for item in candidates:
        if not isinstance(item, Mapping):
            raise E1QualificationError("candidate evidence must be a mapping")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_ids:
            raise E1QualificationError("candidate IDs must be unique non-empty strings")
        candidate_ids.add(candidate_id)
        unexpected = set(item) - allowed_candidate_keys
        if unexpected:
            raise E1QualificationError(
                f"candidate {candidate_id} has non-normalized fields: {sorted(unexpected)}"
            )
        if item.get("source_id") not in source_ids:
            raise E1QualificationError(f"candidate {candidate_id} has unknown source_id")
        task_ref = item.get("task")
        if not isinstance(task_ref, Mapping):
            raise E1QualificationError(f"candidate {candidate_id} requires a frozen task")
        task_path = _verify_file_ref(
            task_ref, base=base, label=f"candidate {candidate_id}.task"
        )
        # Shadow execution occurs from its own run directory.  Persist an
        # unambiguous absolute task path while retaining the frozen content
        # digest, rather than making its correctness depend on current cwd.
        task_ref["path"] = str(task_path)
        evidence = item.get("evidence")
        if not isinstance(evidence, Mapping):
            raise E1QualificationError(f"candidate {candidate_id}.evidence is required")
        _assert_pre_shadow(evidence)
    accounting = value.get("accounting")
    if not isinstance(accounting, Mapping):
        raise E1QualificationError("evidence.accounting is required")
    production = _accounting(accounting.get("simulated_production"))
    if production["calls_by_role"].get("editorial_board", 0) != 0:
        raise E1QualificationError(
            "simulated production accounting must exclude editorial-board calls"
        )
    _accounting(accounting.get("independent_quality_panel"))
    return value


def _candidate_pre_shadow_hash(item: Mapping[str, Any]) -> str:
    return content_hash(
        {
            "candidate_id": item["candidate_id"],
            "source_id": item["source_id"],
            "task": item["task"],
            "evidence": item["evidence"],
        }
    )


def freeze_boardless_dispositions(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze E1 treatment outcomes before shadow review can be imported."""

    baseline = NormalizedSelectedBaselineAdapter(config["_baseline_path"])
    if baseline.sha256 != config["selected_baseline_manifest_sha256"]:
        raise E1QualificationError("selected baseline changed after config load")
    prompt_hashes = {
        str(item["sha256"]) for item in baseline.value["prompts"].values()
    }
    if config["shadow_policy"]["editorial_prompt_sha256"] not in prompt_hashes:
        raise E1QualificationError(
            "shadow editorial prompt is not one of the selected baseline's frozen prompts"
        )
    evidence_path = Path(config["_evidence_path"])
    if _sha(evidence_path) != config["evidence_manifest_sha256"]:
        raise E1QualificationError("evidence changed after config load")
    evidence = load_evidence_manifest(evidence_path, baseline_sha256=baseline.sha256)
    output = Path(config["_output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "boardless_dispositions.frozen.json"
    snapshot_path = output / "config.snapshot.json"
    snapshot = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "config_sha256": _sha(Path(config["_config_path"])),
        "selected_baseline_manifest_sha256": baseline.sha256,
        "selected_baseline": baseline.frozen_snapshot(),
        "evidence_manifest_sha256": _sha(evidence_path),
        "source_report_hashes": {
            item["source_id"]: item["sha256"] for item in evidence["sources"]
        },
        "automatic_commit": False,
        "production_state_mutation_allowed": False,
        "shadow_editorial_gating": False,
    }
    if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
        raise E1QualificationError("E1 run inputs changed during resume")
    if not snapshot_path.exists():
        atomic_write_json(snapshot_path, snapshot)
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        if (
            frozen.get("selected_baseline_manifest_sha256") != baseline.sha256
            or frozen.get("evidence_manifest_sha256") != _sha(evidence_path)
        ):
            raise E1QualificationError("frozen boardless dispositions do not match inputs")
        return frozen
    now = round(time.time(), 3)
    candidates = []
    for item in evidence["candidates"]:
        decision = evaluate_boardless_candidate(item["evidence"])
        candidates.append(
            {
                "candidate_id": item["candidate_id"],
                "source_id": item["source_id"],
                "task": deepcopy(item["task"]),
                "pre_shadow_record_hash": _candidate_pre_shadow_hash(item),
                **decision,
                "verified_at_epoch_seconds": now,
                "shadow_editorial_status": "not_attached",
                "canonical_status_unchanged": True,
                "production_terminal_timestamp_written": False,
            }
        )
    body = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "selected_baseline_id": baseline.baseline_id,
        "selected_baseline_manifest_sha256": baseline.sha256,
        "evidence_manifest_sha256": _sha(evidence_path),
        "source_report_hashes": snapshot["source_report_hashes"],
        "frozen_before_shadow": True,
        "shadow_judgments_consulted": False,
        "automatic_commit_occurred": False,
        "production_state_mutated": False,
        "candidates": candidates,
        "accounting": {
            "simulated_production": _accounting(
                evidence["accounting"].get("simulated_production")
            ),
            "independent_quality_panel": _accounting(
                evidence["accounting"].get("independent_quality_panel")
            ),
        },
    }
    body["freeze_content_sha256"] = content_hash(body)
    atomic_write_json(frozen_path, body)
    queue = {
        "schema_version": 1,
        "run_id": config["run"]["id"],
        "boardless_freeze_sha256": _sha(frozen_path),
        "fixed_anchor_manifest_sha256": baseline.value["frozen_inputs"][
            "fixed_anchor_manifest"
        ]["sha256"],
        "tasks": [
            {
                "candidate_id": item["candidate_id"],
                "task": item["task"],
                "pre_shadow_record_hash": item["pre_shadow_record_hash"],
            }
            for item in candidates
            if item["boardless_disposition"] == VERIFIED
        ],
        "arm_labels_hidden_downstream": True,
        "shadow_is_non_gating": True,
    }
    atomic_write_json(output / "shadow_queue.private.json", queue)
    build_e1_report(config)
    return body


def _shadow_task_disposition(judgments: Sequence[Mapping[str, Any]]) -> str:
    if any(item.get("serious_issue") is True for item in judgments):
        return "shadow_hard_concern"
    recommendations = [item.get("intrinsic_recommendation") for item in judgments]
    qualities = [item.get("overall_quality") for item in judgments]
    if recommendations and all(item == "core_quality" for item in recommendations):
        return "shadow_core_quality"
    if recommendations and all(item in {"core_quality", "reserve_quality"} for item in recommendations):
        return "shadow_reserve_quality"
    if any(item == "rejected_quality" for item in recommendations) or any(
        item == "below_real_bar" for item in qualities
    ):
        return "shadow_quality_concern"
    return "shadow_editorial_disagreement"


def _validate_shadow_result(
    value: Mapping[str, Any],
    *,
    frozen: Mapping[str, Any],
    baseline: NormalizedSelectedBaselineAdapter,
    policy: Mapping[str, Any],
) -> None:
    if value.get("schema_version") != 1:
        raise E1QualificationError("shadow result schema_version must be 1")
    if value.get("boardless_freeze_sha256") != frozen.get("_file_sha256"):
        raise E1QualificationError("shadow result targets a different disposition freeze")
    shadow_policy = value.get("policy")
    if not isinstance(shadow_policy, Mapping):
        raise E1QualificationError("shadow result policy is required")
    if shadow_policy.get("reviewer_count") != 2:
        raise E1QualificationError("shadow result must use exactly two boards")
    if shadow_policy.get("fixed_anchor_manifest_sha256") != baseline.value[
        "frozen_inputs"
    ]["fixed_anchor_manifest"]["sha256"]:
        raise E1QualificationError("shadow result used the wrong fixed anchors")
    required_prompt_hash = policy.get("editorial_prompt_sha256")
    if shadow_policy.get("editorial_prompt_sha256") != required_prompt_hash:
        raise E1QualificationError("shadow result used the wrong editorial prompt")
    if shadow_policy.get("generated_group_minimum") != 4 or shadow_policy.get(
        "generated_group_maximum"
    ) != 6:
        raise E1QualificationError("shadow result used a non-control board group size")
    routes = shadow_policy.get("reviewer_routes")
    if not isinstance(routes, list) or len(routes) != 2:
        raise E1QualificationError("two shadow reviewer routes are required")
    reviewer_ids = set()
    for route in routes:
        if not isinstance(route, Mapping):
            raise E1QualificationError("invalid shadow reviewer route")
        if route.get("provider") != "codex" or route.get("model") != "gpt-5.6-sol" or route.get(
            "reasoning_effort"
        ) != "max":
            raise E1QualificationError("shadow boards must match two Sol/max control boards")
        reviewer_ids.add(route.get("reviewer_id"))
    if len(reviewer_ids) != 2 or None in reviewer_ids:
        raise E1QualificationError("shadow boards must be independent")
    frozen_clear = {
        item["candidate_id"]
        for item in frozen["candidates"]
        if item["boardless_disposition"] == VERIFIED
    }
    results = value.get("results")
    if not isinstance(results, list):
        raise E1QualificationError("shadow results must be a list")
    result_id_rows = [
        item.get("candidate_id") for item in results if isinstance(item, Mapping)
    ]
    if len(result_id_rows) != len(set(result_id_rows)):
        raise E1QualificationError("shadow results contain duplicate candidate IDs")
    result_ids = set(result_id_rows)
    if result_ids != frozen_clear:
        raise E1QualificationError("shadow result set must equal the frozen clear queue")
    for item in results:
        if item.get("operational_failure") is True:
            if item.get("judgments") not in (None, []):
                raise E1QualificationError("failed shadow review cannot carry judgments")
            continue
        judgments = item.get("judgments")
        if not isinstance(judgments, list) or len(judgments) != 2:
            raise E1QualificationError("each successful shadow task requires two judgments")
        ids = {judgment.get("reviewer_id") for judgment in judgments}
        if ids != reviewer_ids:
            raise E1QualificationError("shadow task judgments do not match reviewer routes")


def attach_shadow_results(
    config: Mapping[str, Any], shadow_results_path: str | Path
) -> dict[str, Any]:
    """Attach immutable, explicitly non-gating shadow board measurements."""

    output = Path(config["_output_directory"])
    frozen_path = output / "boardless_dispositions.frozen.json"
    if not frozen_path.is_file():
        raise E1QualificationError("freeze boardless dispositions before attaching shadow review")
    before_bytes = frozen_path.read_bytes()
    frozen = read_json(frozen_path)
    frozen["_file_sha256"] = sha256(before_bytes).hexdigest()
    baseline = NormalizedSelectedBaselineAdapter(config["_baseline_path"])
    evidence_path = Path(config["_evidence_path"])
    if _sha(evidence_path) != config["evidence_manifest_sha256"]:
        raise E1QualificationError("evidence changed before shadow attachment")
    # Revalidate every frozen source report, not merely the outer evidence
    # manifest, before accepting a shadow measurement.
    load_evidence_manifest(evidence_path, baseline_sha256=baseline.sha256)
    source = Path(shadow_results_path).resolve()
    value = read_json(source)
    _validate_shadow_result(
        value,
        frozen=frozen,
        baseline=baseline,
        policy=config["shadow_policy"],
    )
    attached_path = output / "shadow_editorial.non_gating.json"
    rows = []
    for item in value["results"]:
        failed = item.get("operational_failure") is True
        judgments = [] if failed else deepcopy(item["judgments"])
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "status": "measurement_incomplete" if failed else "completed",
                "judgments": judgments,
                "shadow_disposition": (
                    "measurement_incomplete"
                    if failed
                    else _shadow_task_disposition(judgments)
                ),
                "intrinsic_quality": item.get("intrinsic_quality"),
                "portfolio_placement": item.get("portfolio_placement"),
                "reasons": deepcopy(item.get("reasons", [])),
                "completed_at_epoch_seconds": item.get("completed_at_epoch_seconds"),
                "non_gating": True,
            }
        )
    attached = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "run_id": config["run"]["id"],
        "boardless_freeze_sha256": frozen["_file_sha256"],
        "source_shadow_result": {"path": str(source), "sha256": _sha(source)},
        "shadow_is_non_gating": True,
        "boardless_dispositions_unchanged": True,
        "automatic_commit_occurred": False,
        "production_state_mutated": False,
        "policy": deepcopy(value["policy"]),
        "results": rows,
        "accounting": _accounting(value.get("accounting")),
    }
    if attached_path.exists() and read_json(attached_path) != attached:
        raise E1QualificationError("a different shadow result is already attached")
    if not attached_path.exists():
        atomic_write_json(attached_path, attached)
    if frozen_path.read_bytes() != before_bytes:
        raise E1QualificationError("shadow attachment mutated frozen dispositions")
    build_e1_report(config)
    return attached


def build_e1_report(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(config["_output_directory"])
    frozen_path = output / "boardless_dispositions.frozen.json"
    if not frozen_path.is_file():
        raise E1QualificationError("no frozen E1 dispositions")
    frozen = read_json(frozen_path)
    attached_path = output / "shadow_editorial.non_gating.json"
    shadow = read_json(attached_path) if attached_path.is_file() else None
    clear = [
        item for item in frozen["candidates"] if item["boardless_disposition"] == VERIFIED
    ]
    shadow_by_id = {
        item["candidate_id"]: item for item in (shadow or {}).get("results", [])
    }
    tasks = []
    for item in frozen["candidates"]:
        tasks.append(
            {
                **deepcopy(item),
                "shadow_editorial": deepcopy(shadow_by_id.get(item["candidate_id"])),
                "final_production_status": item["boardless_disposition"],
                "shadow_changed_production_status": False,
            }
        )
    production = _accounting(frozen["accounting"]["simulated_production"])
    independent = _accounting(frozen["accounting"]["independent_quality_panel"])
    shadow_accounting = _accounting((shadow or {}).get("accounting"))
    completed_shadow = [
        item for item in shadow_by_id.values() if item.get("status") == "completed"
    ]
    hard = [
        item for item in completed_shadow if item["shadow_disposition"] == "shadow_hard_concern"
    ]
    explicit_hard_audit = bool(completed_shadow) and all(
        all(type(judgment.get("serious_issue")) is bool for judgment in item["judgments"])
        for item in completed_shadow
    )
    below = [
        item
        for item in completed_shadow
        if item["shadow_disposition"] in {"shadow_quality_concern", "shadow_hard_concern"}
    ]
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "selected_baseline_id": frozen["selected_baseline_id"],
        "completed": shadow is not None,
        "boardless_dispositions_frozen_before_shadow": True,
        "shadow_editorial_non_gating": True,
        "automatic_commit_occurred": False,
        "production_state_mutated": False,
        "funnel": {
            "candidate_count": len(frozen["candidates"]),
            "verified_candidate_count": len(clear),
            "not_verified_candidate_count": len(frozen["candidates"]) - len(clear),
            "shadow_completed_count": len(completed_shadow),
            "shadow_incomplete_count": len(clear) - len(completed_shadow),
            "shadow_hard_concern_count": len(hard),
            "shadow_quality_or_hard_concern_count": len(below),
        },
        "tasks": tasks,
        "accounting": {
            "simulated_production": production,
            "shadow_editorial": shadow_accounting,
            "independent_quality_panel": independent,
            "total_experiment": _add_accounting(production, shadow_accounting, independent),
            "shadow_excluded_from_simulated_production": True,
            "no_total_wall_time_saving_claim": True,
        },
        "qualification": {
            "minimum_fresh_clear_target": 60,
            "pilot_minimum_clear_target": 24,
            "zero_missed_hard_defect_required": True,
            "maximum_convincing_below_real_bar_rate": 0.05,
            "sample_complete": len(clear) >= 60 and len(completed_shadow) == len(clear),
            # The frozen V4 board schema has no serious_issue field.  Do not
            # manufacture negative evidence from its absence: this remains
            # pending until an explicit disagreement/hard-defect audit exists.
            "hard_defect_gate_pass": (
                len(hard) == 0 if shadow is not None and explicit_hard_audit else None
            ),
            "explicit_shadow_hard_defect_audit_complete": explicit_hard_audit,
            "below_real_bar_rate": (
                round(len(below) / len(completed_shadow), 6) if completed_shadow else None
            ),
            "promotion_recommendation": "pending_user_review_and_complete_shadow_measurement",
        },
        "limitations": [
            "verified_candidate is neutral and is not intrinsic core quality",
            "completed-grid semantic duplicate equivalence still requires explicit user/commit review",
            "shadow failures are measurement gaps rather than treatment failures",
            "shadow calls may compete for resources, so this report does not claim total-run wall savings",
        ],
    }
    atomic_write_json(output / "report.json", report)
    lines = [
        "# E1 boardless editorial qualification",
        "",
        f"Baseline: `{report['selected_baseline_id']}`",
        "",
        f"- Candidates: {report['funnel']['candidate_count']}",
        f"- Neutral verified candidates: {report['funnel']['verified_candidate_count']}",
        f"- Non-clear candidates: {report['funnel']['not_verified_candidate_count']}",
        f"- Shadow reviews complete: {report['funnel']['shadow_completed_count']}",
        f"- Shadow hard concerns: {report['funnel']['shadow_hard_concern_count']}",
        "",
        "The treatment disposition was frozen before any shadow result was attached. "
        "Shadow boards are immutable measurements and never convert `verified_candidate` "
        "into `core_quality` or change commit eligibility.",
        "",
        "## Accounting",
        "",
        f"- Simulated production calls: {production['substantive_logical_calls']}",
        f"- Shadow editorial calls: {shadow_accounting['substantive_logical_calls']}",
        f"- Independent panel calls: {independent['substantive_logical_calls']}",
        "",
        "No task was committed and no production state was mutated.",
        "",
    ]
    atomic_write_text(output / "report.md", "\n".join(lines))
    return report


def validate_e1(config: Mapping[str, Any]) -> dict[str, Any]:
    baseline = NormalizedSelectedBaselineAdapter(config["_baseline_path"])
    if config["shadow_policy"]["editorial_prompt_sha256"] not in {
        str(item["sha256"]) for item in baseline.value["prompts"].values()
    }:
        raise E1QualificationError(
            "shadow editorial prompt is not one of the selected baseline's frozen prompts"
        )
    evidence = load_evidence_manifest(
        config["_evidence_path"], baseline_sha256=baseline.sha256
    )
    return {
        "pipeline": PIPELINE,
        "version": VERSION,
        "baseline_id": baseline.baseline_id,
        "candidate_count": len(evidence["candidates"]),
        "source_count": len(evidence["sources"]),
        "gate_criteria": [item.code for item in GATE_CRITERIA],
        "passing_status": VERIFIED,
        "core_status_never_emitted": True,
        "shadow_non_gating": True,
        "automatic_commit": False,
        "production_state_mutation_allowed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E1 boardless qualification harness")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "freeze", "report"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
    attach = sub.add_parser("attach-shadow")
    attach.add_argument("--config", required=True)
    attach.add_argument("--shadow-results", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_e1_config(args.config)
        if args.command == "validate":
            value = validate_e1(config)
        elif args.command == "freeze":
            value = freeze_boardless_dispositions(config)
        elif args.command == "attach-shadow":
            value = attach_shadow_results(config, args.shadow_results)
        else:
            value = build_e1_report(config)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except E1QualificationError as exc:
        print(f"E1 qualification error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
