"""Explicit study configuration for the consolidated Sol/Astra factory.

Loading reads only the requested YAML/JSON file. It neither loads a historical
runtime nor inspects a task bank. Freezing and checking referenced files are
separate, explicit operations performed once by the run coordinator.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import os
import re
import shutil
from typing import Any

from .util import atomic_write_json, content_hash
from .factory_policy import REDUCED_PROFILE, REDUCED_POLICY, REDUCED_EFFORTS

PIPELINE = "arcgen-factory"
VERSION = "1.0"
SUPPORTED_MODELS = frozenset({"gpt-6-astra", "gpt-5.6-sol"})
SUPPORTED_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
I1_PROMPT_SHA256 = "d57888cb31f0c5853db8f51ad9bbc63430901c836974fcae820aa60fdae27039"

# These are A0/S0 comparator settings, not measured optimal efforts. Purpose
# names stay distinct even when two purposes share a prompt, schema or role.
INITIAL_EFFORTS = {
    "idea_conditioned": "xhigh",
    "idea_unconditioned": "max",
    "idea_logical_leap": "xhigh",
    "idea_episode_joint": "xhigh",
    "shard_selection": "max",
    "global_selection": "max",
    "planning": "max",
    "realization": "max",
    "technical_verification": "high",
    "primary_blind": "xhigh",
    "conditional_blind": "xhigh",
    "ambiguity_adjudication": "high",
    "targeted_repair": "max",
    "editorial_board": "max",
    "logical_leap_audit": "max",
    "structural_transfer_audit": "max",
    "mechanism_audit": "max",
    "diversity_audit": "max",
    "transport_probe": "high",
}

DEFAULT_SOURCE_PATHS = {
    "historical_registry": "committed_tasks/registry.json",
    "active_registry": "committed_tasks/active_registry.json",
    "quarantine_registry": "committed_tasks/quarantine_registry.json",
    "basic_valid_registry": "committed_tasks/basic_valid_registry.json",
    "production_reservoir": "reservoirs/v4_control.json",
}
DEFAULT_REFERENCES = {
    "base_v4_control": "config/v4_control.yaml",
    "downstream_adapter": "config/v4_4_functional_scale_fresh_integrated.yaml",
    "i1_prompt": {
        "path": "prompts/e20_u_i1_diverge_discard_commit.md",
        "sha256": I1_PROMPT_SHA256,
    },
}
DEFAULTS = {
    "execution": {
        # Strict comparisons keep 48. Working releases can explicitly accept
        # a smaller frozen window after bounded infrastructure failures.
        "minimum_idea_kernels": 48,
    },
    "context": {
        "context_limit": 272000,
        "usable_fraction": 0.95,
        "response_headroom": 32768,
    },
    "supply": {
        "conditioned_calls": 6,
        "unconditioned_calls": 4,
        "logical_leap_calls": 1,
        "episode_joint_calls": 1,
        "ideas_per_call": 4,
        "eligible_window": 48,
        "modes": ["open", "concept_led", "composition_led"],
    },
    "selection": {
        "routine_cap": 12,
        "deferred_cap": 4,
        "shard_count": 4,
        "shard_promotion_cap": 5,
        "logical_leap_ceiling": 1,
        "episode_joint_ceiling": 1,
        "specialist_total_ceiling": 2,
        "invitation_free_minimum_share": 0.7,
        "deferred_reserved": False,
        "deferred_developed": False,
        "fill_with_unqualified": False,
    },
    "worker_policy": {
        "downstream_workers": 8,
        "upstream_workers": 4,
        "global_call_ceiling": 12,
        "downstream_idle_lending_to_upstream": True,
        "upstream_workers_may_run_downstream": False,
        "board_group_workers": 4,
        "audit_workers": 8,
        "active_downstream_slates": 2,
        "ready_queue_target": 1,
        "ready_queue_hard_maximum": 2,
        "max_downstream_before_new_plan": 2,
    },
    "reservoir": {
        "low_watermark": 48,
        "high_watermark": 96,
        "hard_capacity": 144,
        "prefetch_slates": 2,
        "lease_seconds": 259200,
    },
    "verification": {
        "corrected_e19_trigger": True,
        "maximum_targeted_repairs": 1,
        "unresolved_evidence_status": "evidence_hold",
        "escalation_calls_per_slate": 5,
        "independent_blind_required": True,
        "repair_requires_reverification": True,
        "final_episode_bound_evidence": True,
    },
    "policy": {
        "ordinary_unconditioned_i1": True,
        "i1_route": "ordinary_unconditioned",
        "bank_context_fields": ["title", "concept", "rule"],
        "silent_context_truncation": False,
        "fresh_profile_ideas": True,
        "source_blind_selection": True,
        "guaranteed_source_slots": False,
        "architect": False,
        "extra_episode_joint_nominator": False,
        "answer_aware_replacement": False,
        "safe_global_palette": True,
        "evidence_preserving_scale": True,
        "maximum_scale_invitations_per_slate": 1,
        "one_answer": True,
        "research_audits_gating": False,
        "same_procedure_publication_requires_override": True,
        "exact_duplicates_allowed": False,
        "palette_duplicates_allowed": False,
        "automatic_quarantine_reactivation": False,
    },
    "transport": {
        "provider": "codex_cli",
        "executable": "codex",
        "timeout_seconds": 1800,
        "maximum_attempts": 2,
        "codex_home": None,
    },
    "embedding": {
        "provider": "openrouter",
        "model": "openai/text-embedding-3-large",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "batch_size": 64,
        "timeout_seconds": 120,
        "retries": 2,
        "app_name": "arcgen-v4-control",
    },
    "run": {
        "id": None,
        "account": "main",
        "replicate": "main",
        "slates": 4,
        "maximum_slates": 6,
        "mock": False,
        "mock_responses": None,
        "automatic_task_commit": False,
        "automatic_feature_promotion": False,
        "production_state_mutation_allowed": False,
        "production_reservoir_mutation_allowed": False,
    },
    "commit": {
        "enabled": False,
        "commit_core": False,
        "commit_reserve": False,
        "commit_basic_valid": False,
        "authorization_record": None,
    },
}


class FactoryConfigError(ValueError):
    """An invalid profile, accepted-policy change or isolation violation."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise FactoryConfigError(f"{label} must be a mapping with string keys")
    return dict(value)


def _section(value: Any, defaults: Mapping[str, Any], label: str) -> dict[str, Any]:
    supplied = _mapping(value, label)
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise FactoryConfigError(f"unknown {label} fields: {', '.join(sorted(unknown))}")
    return {**deepcopy(dict(defaults)), **deepcopy(supplied)}


def _fixed_policy(section: Mapping[str, Any], defaults: Mapping[str, Any], label: str,
                  mutable: tuple[str, ...] = ()) -> None:
    for key, expected in defaults.items():
        if key not in mutable and (
            type(section[key]) is not type(expected) or section[key] != expected
        ):
            raise FactoryConfigError(f"{label}.{key} must be {expected!r} in this study")


def _path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FactoryConfigError(f"{label} must be a non-empty path")
    given = Path(value)
    return (given if given.is_absolute() else root / given).resolve()


def _identifier(value: Any, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise FactoryConfigError(f"{label} must be a single non-empty identifier")


def load_factory_config(path: str | Path) -> dict[str, Any]:
    """Resolve one new factory config without inspecting its referenced inputs.

    Relative paths are repository-relative for files in a ``config`` directory,
    otherwise relative to the config's directory. The returned ``design`` and
    ``scheduling`` dictionaries are mechanical projections for reusable E18
    helpers; no historical validator or experiment identity is used.
    """
    source = Path(path).resolve()
    text = source.read_text(encoding="utf-8")
    try:
        import yaml

        raw = _mapping(yaml.safe_load(text), "config")
    except ImportError:
        raw = _mapping(json.loads(text), "config")
    allowed = {"version", "pipeline", "seed", "model_profile", "paths", "references", "pipeline_profile", "contract_profile", *DEFAULTS}
    if raw.keys() - allowed:
        raise FactoryConfigError(f"unknown config fields: {sorted(raw.keys() - allowed)}")
    if str(raw.get("version")) != VERSION or raw.get("pipeline") != PIPELINE:
        raise FactoryConfigError(f"configuration must target {PIPELINE} {VERSION}")
    if type(raw.get("seed")) is not int:
        raise FactoryConfigError("seed must be an integer")
    root = source.parent.parent if source.parent.name == "config" else source.parent
    profile = _section(raw.get("model_profile"), {"model": None, "efforts": {}}, "model_profile")
    if not isinstance(profile["model"], str) or profile["model"] not in SUPPORTED_MODELS:
        raise FactoryConfigError("model_profile.model must explicitly be gpt-6-astra or gpt-5.6-sol")
    candidate = raw.get("pipeline_profile")
    if "pipeline_profile" in raw and candidate != REDUCED_PROFILE:
        raise FactoryConfigError("unknown pipeline_profile; omit for legacy or select astra_reduced_v1")
    if "contract_profile" in raw:
        from .factory_policy import AUTHORING_CONTRACT_PROFILE
        if candidate != REDUCED_PROFILE or raw["contract_profile"] != AUTHORING_CONTRACT_PROFILE:
            raise FactoryConfigError("contract_profile requires astra_reduced_v1 and astra_authoring_contract_r2")
    efforts = _section(profile["efforts"], REDUCED_EFFORTS if candidate else INITIAL_EFFORTS, "model_profile.efforts")
    if candidate and (profile["model"] != "gpt-6-astra" or efforts != REDUCED_EFFORTS):
        raise FactoryConfigError("astra_reduced_v1 requires its exact Astra role/effort profile")
    for purpose, effort in efforts.items():
        if not isinstance(effort, str) or effort not in SUPPORTED_EFFORTS:
            raise FactoryConfigError(f"unsupported effort for {purpose}: {effort!r}")

    config: dict[str, Any] = {"version": VERSION, "pipeline": PIPELINE, "seed": raw["seed"]}
    for name, defaults in DEFAULTS.items():
        config[name] = _section(raw.get(name, {}), defaults, name)
    minimum = config["execution"]["minimum_idea_kernels"]
    if type(minimum) is not int or minimum not in {32, 36, 40, 44, 48}:
        raise FactoryConfigError("minimum_idea_kernels must be 32, 36, 40, 44 or 48")
    for name in ("supply", "selection", "worker_policy", "reservoir", "verification", "embedding", "commit"):
        _fixed_policy(config[name], DEFAULTS[name], name)
    _fixed_policy(config["context"], DEFAULTS["context"], "context")
    _fixed_policy(config["policy"], DEFAULTS["policy"], "policy", ("ordinary_unconditioned_i1",))
    if type(config["policy"]["ordinary_unconditioned_i1"]) is not bool:
        raise FactoryConfigError("policy.ordinary_unconditioned_i1 must be boolean")
    _fixed_policy(config["transport"], DEFAULTS["transport"], "transport", ("executable", "timeout_seconds", "codex_home"))
    if not isinstance(config["transport"]["executable"], str) or not config["transport"]["executable"].strip():
        raise FactoryConfigError("transport.executable must be non-empty")
    if type(config["transport"]["timeout_seconds"]) is not int or config["transport"]["timeout_seconds"] < 1:
        raise FactoryConfigError("transport.timeout_seconds must be a positive integer")
    run = config["run"]
    _fixed_policy(run, DEFAULTS["run"], "run", ("id", "account", "replicate", "slates", "maximum_slates", "mock", "mock_responses"))
    for name in ("id", "account", "replicate"):
        _identifier(run[name], f"run.{name}")
    codex_home = config["transport"]["codex_home"]
    config["transport"]["codex_home"] = str(
        _path(root, codex_home, "transport.codex_home") if codex_home is not None
        else Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser().resolve()
    )
    if type(run["mock"]) is not bool:
        raise FactoryConfigError("run.mock must be boolean")
    if any(type(run[key]) is not int or run[key] < 1 for key in ("slates", "maximum_slates")):
        raise FactoryConfigError("run slate limits must be positive integers")
    if run["slates"] > run["maximum_slates"] or run["maximum_slates"] > 6:
        raise FactoryConfigError("run requires slates <= maximum_slates <= 6")
    if run["mock_responses"] is not None:
        run["mock_responses"] = str(_path(root, run["mock_responses"], "run.mock_responses"))
        if not run["mock"]:
            raise FactoryConfigError("mock_responses requires run.mock")

    paths = _section(raw.get("paths", {}), {**DEFAULT_SOURCE_PATHS, "output_directory": None, "cache": None}, "paths")
    output = _path(root, paths["output_directory"], "paths.output_directory")
    if output.name != run["id"]:
        raise FactoryConfigError("output directory name must equal run.id")
    resolved_paths = {name: _path(root, paths[name], f"paths.{name}") for name in DEFAULT_SOURCE_PATHS}
    protected = {root / name for name in ("config", "prompts", "arcgen.lib", "tests")}
    protected.update(value.parent for value in resolved_paths.values())
    if root.is_relative_to(output) or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise FactoryConfigError("output directory overlaps protected source or bank paths")
    cache_root = _path(root, paths["cache"], "paths.cache") if paths["cache"] else output / "cache"
    if not cache_root.is_relative_to(output) or cache_root == output:
        raise FactoryConfigError("cache must be isolated inside the run output directory")
    cache = cache_root / run["replicate"]
    resolved_paths.update(output_directory=output, cache=cache)

    references = {**deepcopy(DEFAULT_REFERENCES), **_mapping(raw.get("references", {}), "references")}
    resolved_references, expected_hashes = {}, {}
    for name, value in references.items():
        if isinstance(value, Mapping):
            if set(value) != {"path", "sha256"} or not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
                raise FactoryConfigError(f"references.{name} must contain path and a SHA-256")
            expected_hashes[name] = value["sha256"]
            value = value["path"]
        resolved_references[name] = _path(root, value, f"references.{name}")
    if expected_hashes.get("i1_prompt") != I1_PROMPT_SHA256:
        raise FactoryConfigError("the I1 reference must bind the exact accepted prompt")

    if candidate:
        if (minimum != 32 or not config["policy"]["ordinary_unconditioned_i1"]
                or config["transport"]["timeout_seconds"] != 1800):
            raise FactoryConfigError("astra_reduced_v1 requires minimum 32, accepted I1 and the 1800 s base transport timeout")
        config["pipeline_profile"] = candidate
        config["_pipeline_policy"] = deepcopy(REDUCED_POLICY)
    if "contract_profile" in raw:
        config["contract_profile"] = raw["contract_profile"]

    config["paths"] = {name: str(value) for name, value in resolved_paths.items()}
    config["references"] = references
    config["model_profile"] = {"model": profile["model"], "efforts": efforts}
    config["role_profile"] = {
        purpose: {"purpose": purpose, "model": profile["model"], "reasoning_effort": effort}
        for purpose, effort in efforts.items()
    }
    selection, supply, workers = config["selection"], config["supply"], config["worker_policy"]
    config["design"] = {
        "slates": run["slates"], "idea_calls_per_slate": 12,
        "ideas_per_call": supply["ideas_per_call"], "eligible_window_size": supply["eligible_window"],
        "shard_size": supply["eligible_window"] // selection["shard_count"],
        "ordinary_unconditioned_calls_per_slate": supply["unconditioned_calls"],
        "ordinary_bank_conditioned_calls_per_slate": supply["conditioned_calls"],
        "logical_leap_calls_per_slate": supply["logical_leap_calls"],
        "episode_joint_calls_per_slate": supply["episode_joint_calls"],
        "maximum_selected_per_slate": selection["routine_cap"],
        "maximum_plans_per_slate": selection["routine_cap"],
        "maximum_realizations_per_slate": selection["routine_cap"],
        "maximum_logical_selected_per_slate": selection["logical_leap_ceiling"],
        "maximum_episode_joint_selected_per_slate": selection["episode_joint_ceiling"],
        "maximum_specialized_selected_per_slate": selection["specialist_total_ceiling"],
        "minimum_invitation_free_fraction": selection["invitation_free_minimum_share"],
        "audit_batch_size": 6,
    }
    config["scheduling"] = {
        "global_workers": workers["downstream_workers"], "selection_cap": 6,
        "codex_cap": workers["upstream_workers"],
        "max_downstream_before_new_plan": workers["max_downstream_before_new_plan"],
        "escalation_calls_per_slate": config["verification"]["escalation_calls_per_slate"],
        "maximum_attempts_per_logical_call": config["transport"]["maximum_attempts"],
        "candidate_level_streaming": True, "cross_slate_overlap": True, "deferred_boards": True,
    }
    config["_config_path"] = source
    config["_repository_root"] = root
    config["_output_directory"] = str(output)
    config["_paths"] = resolved_paths
    config["_references"] = resolved_references
    config["_expected_reference_hashes"] = expected_hashes
    config["_profile_hash"] = content_hash(config["model_profile"])
    config["_config_sha256"] = sha256(text.encode("utf-8")).hexdigest()
    config["_run_identity"] = {
        "run_id": run["id"], "account": run["account"], "replicate": run["replicate"],
        "profile_hash": config["_profile_hash"], "config_sha256": config["_config_sha256"],
    }
    return config


def resolve_role(config: Mapping[str, Any], purpose: str) -> dict[str, str]:
    """Return a copy of one resolved binding; never fall back to another role."""
    try:
        return dict(config["role_profile"][purpose])
    except KeyError as exc:
        raise FactoryConfigError(f"unknown factory call purpose: {purpose}") from exc


def resolved_config_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """Produce a JSON-safe resolved manifest without another filesystem read."""
    def encode(value: Any) -> str:
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"not a JSON config value: {type(value).__name__}")

    return json.loads(json.dumps(config, default=encode))


def verify_factory_references(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Check only explicitly referenced files, on request during input freeze."""
    result = {}
    for name, path in config["_references"].items():
        source = Path(path)
        if not source.is_file():
            raise FactoryConfigError(f"reference missing: {name}: {source}")
        digest = sha256(source.read_bytes()).hexdigest()
        expected = config["_expected_reference_hashes"].get(name)
        if expected is not None and digest != expected:
            raise FactoryConfigError(f"reference changed: {name}")
        result[name] = {"path": str(source), "sha256": digest}
    return result


def snapshot_explicit_files(paths: Mapping[str, str | Path], destination: str | Path) -> dict[str, Any]:
    """Copy a named explicit file set into a new snapshot; never scan a tree.

    Keys are relative snapshot paths. Existing destinations are never replaced.
    The manifest retains original paths for a separately authorized restore.
    """
    target = Path(destination).resolve()
    sources = {}
    for name, path in _mapping(paths, "snapshot paths").items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in {"", "."} or relative.as_posix() != name or relative.parts[0] == "manifest.json":
            raise FactoryConfigError(f"unsafe snapshot name: {name}")
        source = Path(path).resolve()
        if not source.is_file():
            raise FactoryConfigError(f"snapshot source is not a file: {source}")
        sources[name] = (source, sha256(source.read_bytes()).hexdigest())
    if not sources:
        raise FactoryConfigError("snapshot requires at least one explicit file")
    if target.exists():
        raise FactoryConfigError(f"snapshot destination already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"schema_version": 1, "files": {}}
    for name, (source, digest) in sources.items():
        copied = target / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
        if sha256(copied.read_bytes()).hexdigest() != digest:
            raise FactoryConfigError(f"source changed while freezing: {source}")
        manifest["files"][name] = {"source": str(source), "sha256": digest, "bytes": copied.stat().st_size}
    manifest["content_hash"] = content_hash(manifest)
    atomic_write_json(target / "manifest.json", manifest)
    return manifest


def verify_snapshot(destination: str | Path, *, verify_sources: bool = False) -> dict[str, Any]:
    """Verify the named snapshot files; live-source drift is optional and explicit."""
    target = Path(destination).resolve()
    manifest = _mapping(json.loads((target / "manifest.json").read_text(encoding="utf-8")), "snapshot manifest")
    if manifest.get("schema_version") != 1 or manifest.get("content_hash") != content_hash({k: v for k, v in manifest.items() if k != "content_hash"}):
        raise FactoryConfigError("snapshot manifest is invalid or changed")
    for name, row in _mapping(manifest.get("files"), "snapshot files").items():
        copied = (target / name).resolve()
        if not copied.is_relative_to(target) or not copied.is_file() or sha256(copied.read_bytes()).hexdigest() != row["sha256"]:
            raise FactoryConfigError(f"snapshot file missing or changed: {name}")
        if verify_sources:
            source = Path(row["source"])
            if not source.is_file() or sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise FactoryConfigError(f"snapshot source changed: {source}")
    return manifest
