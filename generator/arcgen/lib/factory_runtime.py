"""Explicit factory stage bindings over the existing structured CLI transport.

Candidate and deferred-review helpers use the same keyword call protocol as
before. Construction does not load an experiment runtime or install parser
extensions. Supply and calibration can invoke an already-rendered prompt with
an explicit purpose through the same bounded attempt owner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
import re
import shlex
import shutil
import threading
import time
from typing import Any

from jsonschema import Draft202012Validator

from .e16_bank_context_tuning import IDEA_RESPONSE_SCHEMA, parse_e16_stage
from .e18_calls import E18AttemptInvoker
from .factory_config import resolve_role, resolved_config_manifest
from .factory_context import FactoryContextError, FactoryContextGuard
from .factory_payloads import TECHNICAL_PLAN_PROJECTION, technical_public_plan
from .factory_supply import parse_factory_selector
from .model import CallResult, ModelInvoker
from .schemas import parse_json_object
from .util import (
    PLACEHOLDER_RE, atomic_write_json, atomic_write_text, content_hash,
    load_prompt, read_json, render_prompt,
)
from .v4_schemas import OUTPUT_SCHEMAS as V4_SCHEMAS
from .v4_schemas import parse_stage_output as parse_v4
from .v41_schemas import OUTPUT_SCHEMAS as V41_SCHEMAS
from .v41_schemas import _transport, parse_stage_output as parse_v41
from .v41_structural_transfer import (
    STRUCTURAL_AUDIT_SCHEMA, parse_structural_extension,
)
from .v42_logical_leap import (
    LOGICAL_AUDIT_SCHEMA, LOGICAL_PLAN_SCHEMA, parse_logical_extension,
)
from .v43_episode_joint import STRICT_EXTENSION_SCHEMAS as V43_SCHEMAS
from .v43_episode_joint import parse_v43_extension
from .v44_functional_scale import SCALE_PLAN_SCHEMA, parse_scale_plan


class FactoryRuntimeError(RuntimeError):
    """A stage binding, payload or persisted invocation changed unexpectedly."""


# Only active factory templates are installed. Ordered composition is supplied
# by the reused candidate stages, including the final schema-selecting addendum.
PROMPT_SPECS = {
    "concrete_planner": ("v4_concrete_planner.md", "v4_concrete_plan"),
    "task_realizer": ("v4_task_realizer.md", "v4_task_realizer"),
    "technical_verifier": ("v4_technical_verifier.md", "v4_technical_verifier"),
    "blind_solver": ("v4_blind_solver.md", "v4_blind_solver"),
    "ambiguity_adjudicator": ("v4_ambiguity_adjudicator.md", "v4_ambiguity_adjudicator"),
    "targeted_repair": ("v4_targeted_repair.md", "v4_targeted_repair"),
    "editorial_board": ("v4_editorial_board.md", "v4_editorial_board"),
    "v44_quality_plan": ("v4_4_concrete_quality_planning_addendum.md", "v4_concrete_plan"),
    "logical_leap_planner_addendum": ("v4_2_logical_leap_planner_addendum.md", "v42_logical_leap_plan"),
    "v43_episode_joint_plan": ("v4_3_episode_joint_planning_addendum.md", "v43_episode_joint_plan"),
    "evidence_scale_addendum": ("v4_8_evidence_preserving_functional_scale_planning_addendum.md", "v44_functional_scale_plan"),
    "logical_leap_auditor": ("v4_2_logical_leap_auditor.md", "v42_logical_leap_audit"),
    "structural_transfer_auditor": ("v4_1_structural_transfer_auditor.md", "v41_structural_transfer_audit"),
    "mechanism_auditor": ("v4_1_mechanism_auditor.md", "v41_mechanism_audit"),
    "diversity_auditor": ("v4_1_diversity_auditor.md", "v41_diversity_audit"),
}

REPAIR_NORMALIZATION = "e18_repair_ledger_normalization"
REPAIR_NORMALIZATION_REASON = (
    "both ledgers described the authorized revision from the old plan, "
    "not a deviation from the returned repaired plan"
)
UPSTREAM_PURPOSES = frozenset({
    "idea_conditioned", "idea_unconditioned", "idea_logical_leap", "idea_episode_joint",
    "shard_selection", "global_selection", "full_window_selection",
})


def normalize_repair_ledgers(value: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the existing narrow normalization without mutating its input."""
    result = deepcopy(dict(value))
    if result.get("status") != "repaired":
        return result
    realization = result["repaired_realization"]
    fidelity = realization["authoring_metadata"]["plan_fidelity"]
    top, nested = realization["deviations"], fidelity["deviations"]
    if top == nested:
        return result
    if not isinstance(top, list) or not isinstance(nested, list) or not top or not nested:
        raise FactoryRuntimeError("repair returned one-sided or empty/nonempty deviation ledgers")
    if len(top) != len(nested):
        raise FactoryRuntimeError("repair deviation ledger cardinalities differ")
    flags = lambda rows: [row.get("semantic_or_evidential") for row in rows if isinstance(row, Mapping)]
    if len(flags(top)) != len(top) or sorted(flags(top)) != sorted(flags(nested)):
        raise FactoryRuntimeError("repair deviation ledgers are not semantically aligned")
    if any(
        not str(row.get(field) or "").strip()
        for row in [*top, *nested] for field in ("plan_field", "deviation", "reason")
    ):
        raise FactoryRuntimeError("repair deviation ledger is incomplete")
    realization["deviations"] = []
    fidelity["deviations"] = []
    result[REPAIR_NORMALIZATION] = {
        "applied": True,
        "reason": REPAIR_NORMALIZATION_REASON,
        "original_top_level": top,
        "original_nested": nested,
    }
    return result


def parse_targeted_repair(data: Any) -> dict[str, Any]:
    """Strict repair parsing with verifiable raw/parsed/cache round trips.

    Transport outputs use the unchanged V4 schema. Persisted parsed responses
    can contain our normalization ledger; reconstructing and re-parsing its
    original ledgers proves that this annotation was produced by this parser.
    """
    value = deepcopy(dict(data)) if isinstance(data, Mapping) else data
    if isinstance(value, dict) and REPAIR_NORMALIZATION in value:
        supplied = deepcopy(value)
        provenance = value.pop(REPAIR_NORMALIZATION)
        try:
            realization = value["repaired_realization"]
            fidelity = realization["authoring_metadata"]["plan_fidelity"]
            if realization["deviations"] != [] or fidelity["deviations"] != []:
                raise FactoryRuntimeError("normalized repair must have empty deviation ledgers")
            realization["deviations"] = provenance["original_top_level"]
            fidelity["deviations"] = provenance["original_nested"]
        except (KeyError, TypeError) as exc:
            raise FactoryRuntimeError("malformed repair normalization provenance") from exc
        normalized = normalize_repair_ledgers(parse_v4("v4_targeted_repair", value))
        if normalized != supplied:
            raise FactoryRuntimeError("repair normalization provenance does not reproduce the response")
        return normalized
    return normalize_repair_ledgers(parse_v4("v4_targeted_repair", value))


@dataclass(frozen=True)
class StageBinding:
    schema: dict[str, Any]
    parser: Callable[[Any], dict[str, Any]]
    purposes: tuple[str, ...]


def _probe_parser(schema: Mapping[str, Any], data: Any) -> dict[str, Any]:
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else dict(data)
    Draft202012Validator(schema).validate(value)
    return value


def build_stage_bindings() -> dict[str, StageBinding]:
    """Build every schema/parser/purpose binding before transport construction."""
    purposes = {
        "v4_shard_selector": ("shard_selection",),
        "v4_global_selector": ("global_selection",),
        "v4_concrete_plan": ("planning",),
        "v4_task_realizer": ("realization",),
        "v4_technical_verifier": ("technical_verification",),
        "v4_blind_solver": ("primary_blind", "conditional_blind"),
        "v4_ambiguity_adjudicator": ("ambiguity_adjudication",),
        "v4_targeted_repair": ("targeted_repair",),
        "v4_editorial_board": ("editorial_board",),
    }
    bindings = {
        stage: StageBinding(deepcopy(V4_SCHEMAS[stage]), partial(parse_v4, stage), roles)
        for stage, roles in purposes.items()
    }
    bindings["v4_targeted_repair"] = StageBinding(
        deepcopy(V4_SCHEMAS["v4_targeted_repair"]), parse_targeted_repair, ("targeted_repair",),
    )
    for stage in ("v4_shard_selector", "v4_global_selector"):
        bindings[stage] = StageBinding(
            deepcopy(V4_SCHEMAS[stage]), partial(parse_factory_selector, stage), purposes[stage],
        )
    for stage, schema, parser, purpose in (
        ("v42_logical_leap_plan", LOGICAL_PLAN_SCHEMA, partial(parse_logical_extension, "v42_logical_leap_plan"), "planning"),
        ("v42_logical_leap_audit", LOGICAL_AUDIT_SCHEMA, partial(parse_logical_extension, "v42_logical_leap_audit"), "logical_leap_audit"),
        ("v43_episode_joint_plan", V43_SCHEMAS["v43_episode_joint_plan"], partial(parse_v43_extension, "v43_episode_joint_plan"), "planning"),
        ("v44_functional_scale_plan", SCALE_PLAN_SCHEMA, parse_scale_plan, "planning"),
        ("v41_structural_transfer_audit", STRUCTURAL_AUDIT_SCHEMA, partial(parse_structural_extension, "v41_structural_transfer_audit"), "structural_transfer_audit"),
        ("v41_mechanism_audit", V41_SCHEMAS["v41_mechanism_audit"], partial(parse_v41, "v41_mechanism_audit"), "mechanism_audit"),
        ("v41_diversity_audit", V41_SCHEMAS["v41_diversity_audit"], partial(parse_v41, "v41_diversity_audit"), "diversity_audit"),
    ):
        bindings[stage] = StageBinding(_transport(schema), parser, (purpose,))
    bindings["e16_idea_search"] = StageBinding(
        deepcopy(IDEA_RESPONSE_SCHEMA), partial(parse_e16_stage, "e16_idea_search"),
        ("idea_conditioned", "idea_unconditioned", "idea_logical_leap", "idea_episode_joint"),
    )
    properties = {"probe_id": {"type": "string"}, "ok": {"type": "boolean"}}
    for stage, extra in (
        ("factory_transport_probe", {}),
        ("factory_transport_probe_nested", {"checks": {"type": "array", "items": {
            "type": "object", "properties": {"name": {"type": "string"}, "passed": {"type": "boolean"}},
            "required": ["name", "passed"], "additionalProperties": False,
        }}}),
    ):
        fields = {**properties, **extra}
        schema = {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
        bindings[stage] = StageBinding(schema, partial(_probe_parser, schema), ("transport_probe",))
    return bindings


def _immutable_json(path: Path, value: Any) -> None:
    if path.is_file():
        if read_json(path) != value:
            raise FactoryRuntimeError(f"factory runtime input changed on resume: {path.name}")
    else:
        atomic_write_json(path, value)


class FactoryPromptStore:
    def __init__(self, prompt_dir: Path, run_dir: Path, *, prompt_overrides: Mapping[str, tuple[str, str]] | None = None) -> None:
        self.entries = {}
        destination = run_dir / "prompts" / "templates"
        for name, (filename, stage) in {**PROMPT_SPECS, **deepcopy(dict(prompt_overrides or {}))}.items():
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise FactoryRuntimeError("prompt filename must stay within the prompt directory")
            text, version, digest = load_prompt(prompt_dir / filename)
            self.entries[name] = {"filename": filename, "text": text, "version": version, "hash": digest, "stage": stage}
            target = destination / filename
            if target.is_file() and target.read_text(encoding="utf-8") != text:
                raise FactoryRuntimeError(f"factory prompt changed on resume: {filename}")
            if not target.is_file():
                atomic_write_text(target, text)
        _immutable_json(run_dir / "prompts" / "manifest.json", {
            name: {key: value for key, value in entry.items() if key != "text"}
            for name, entry in self.entries.items()
        })

    def render(
        self, name: str, values: Mapping[str, Any], composed_with: Sequence[str] = (),
    ) -> tuple[str, str, str, str]:
        entries = [self.entries[key] for key in (name, *composed_with)]
        combined = "\n\n".join(entry["text"] for entry in entries)
        required = set(PLACEHOLDER_RE.findall(combined))
        if required != set(values):
            raise FactoryRuntimeError(
                f"{name} bindings differ: missing={sorted(required-set(values))}, extra={sorted(set(values)-required)}"
            )
        digest = content_hash({"templates": [entry["hash"] for entry in entries], "combined": combined}) if composed_with else entries[0]["hash"]
        return render_prompt(combined, dict(values)), "+".join(entry["version"] for entry in entries), digest, entries[-1]["stage"]


def _read_recipe(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FactoryRuntimeError(f"factory recipe must be a mapping: {path}")
    return value


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the semantic sections actually needed by reused stage helpers."""
    root = Path(config["_repository_root"])
    references = config["_references"]
    base = _read_recipe(Path(references["base_v4_control"]))
    palette = _read_recipe(root / "config/v4_1_control.yaml")["palette_experiment"]
    adapter = _read_recipe(Path(references["downstream_adapter"]))
    paths = {name: str((root / value).resolve()) for name, value in base["paths"].items()}
    paths.update({
        "runs": str(config["_output_directory"]),
        "cache": str(config["_paths"]["cache"]),
        "committed_task_registry": str(config["_paths"]["active_registry"]),
    })
    # Explicit fixture/input overrides are passed through the resolved paths;
    # no bank is read or validated during runtime construction.
    for name in ("annotations", "tasks", "prompts", "fixed_anchor_manifest"):
        if name in config["_paths"]:
            paths[name] = str(config["_paths"][name])
    audit_batch = int(adapter["fresh_scale_integration"]["audit_batch_size"])
    legacy_roles = {
        "idea_search": "idea_conditioned", "batch_selector": "global_selection",
        "concrete_planner": "planning", "task_realizer": "realization",
        "technical_verifier": "technical_verification", "blind_solver": "primary_blind",
        "ambiguity_adjudicator": "ambiguity_adjudication", "targeted_repair": "targeted_repair",
        "editorial_board": "editorial_board",
    }
    roles = {name: {key: val for key, val in resolve_role(config, purpose).items() if key != "purpose"} for name, purpose in legacy_roles.items()}
    return {
        "pipeline": config["pipeline"], "seed": int(config["seed"]) + 18_031,
        "run": {**deepcopy(config["run"]), "automatically_commit_generated_tasks": False},
        "paths": paths, "_repository_root": str(root), "_config_path": str(config["_config_path"]),
        "model": {
            **deepcopy(config["transport"]), "name": resolve_role(config, "planning")["model"],
            "reasoning_effort": resolve_role(config, "planning")["reasoning_effort"],
            "retries": 0, "concurrency": int(config["worker_policy"]["global_call_ceiling"]),
            "stage_overrides": {}, "roles": roles,
        },
        "embedding": deepcopy(config["embedding"]),
        "palette_experiment": palette, "repair_policy": deepcopy(adapter["repair_policy"]),
        "selection": {
            **deepcopy(base["selection"]),
            "max_plans": config["design"]["maximum_plans_per_slate"],
            "max_realizations": config["design"]["maximum_realizations_per_slate"],
            "global_selector_max_selected": config["selection"]["routine_cap"] + config["selection"]["deferred_cap"],
            "maximum_optional_development_suggestions": 3,
        },
        "functional_scale": {"escalation_calls_per_slate": config["scheduling"]["escalation_calls_per_slate"]},
        "logical_leap": {"audit_batch_size": audit_batch},
        "structural_transfer": {"audit_batch_size": audit_batch},
        "integrated_qualification": {"audit_batch_size": audit_batch},
    }


class FactoryModelInvoker(ModelInvoker):
    """Bind provider-slot acquisition at construction, leaving caches unchanged."""
    def __init__(self, config: dict[str, Any], run_dir: Path, *, pool: Any = None, lane: str, **kwargs: Any) -> None:
        transport_config = deepcopy(config)
        self.account_binding = _bind_account_launcher(transport_config, run_dir)
        super().__init__(transport_config, run_dir, **kwargs)
        self.pool, self.lane = pool, lane
        self._transport_context = threading.local()
        if pool is not None:
            self._semaphore = pool.bound(lane)

    def invoke(self, *, stage: str, call_id: str, **kwargs: Any) -> CallResult:
        context = self.pool.call_context(self.lane, f"{stage}/{call_id}") if self.pool is not None else nullcontext()
        previous = getattr(self._transport_context, "identity", None)
        self._transport_context.identity = (stage, call_id)
        try:
            with context:
                return super().invoke(stage=stage, call_id=call_id, **kwargs)
        finally:
            self._transport_context.identity = previous

    def _codex_response(self, prompt: str, schema_path: Path, *, timeout_seconds: int,
                        model_name: str, reasoning_effort: str) -> tuple[str, str, str]:
        # Delegate the one actual dispatch unchanged. Successful CLI observations
        # are needed to inspect reported model/usage, unexpected tools and context
        # compaction; an absence of those observations is not verification.
        response = super()._codex_response(
            prompt, schema_path, timeout_seconds=timeout_seconds,
            model_name=model_name, reasoning_effort=reasoning_effort,
        )
        stage, call_id = self._transport_context.identity
        directory = self.calls_dir / stage / call_id
        # These ordinal observations count returned CLI responses, not physical
        # attempts: failed CLI execution has only the existing failure log.
        observations = directory / "transport_observations"
        ordinal = 1 + len(list(observations.glob("response-*.stdout.log")))
        for stream, value in zip(("stdout", "stderr"), response[1:], strict=True):
            atomic_write_text(observations / f"response-{ordinal:02d}.{stream}.log", value)
            atomic_write_text(directory / f"transport.{stream}.log", value)
        return response


def _bind_account_launcher(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Select a Codex account only in the invoked process's environment.

    The existing transport remains responsible for CLI flags, temporary working
    directories, schemas, timing, retries and raw-response persistence. A tiny
    immutable launcher supplies the one environment dependency it cannot take
    directly; no account credential or global environment is read or changed.
    """
    configured = str(config["model"]["executable"])
    executable = shutil.which(configured)
    if executable is None:
        if not config["run"].get("mock"):
            raise FactoryRuntimeError(f"model executable not found: {configured!r}")
        executable = configured
    else:
        executable = str(Path(executable).resolve())
    codex_directory = Path(config["model"]["codex_home"])
    if not codex_directory.is_absolute():
        raise FactoryRuntimeError("transport.codex_home must be explicitly resolved")
    launcher = run_dir / "transport" / "codex-account"
    if Path(executable).resolve() == launcher.resolve():
        raise FactoryRuntimeError("account launcher cannot invoke itself")
    script = (
        "#!/bin/sh\nexec /usr/bin/env "
        + shlex.quote(f"CODEX_HOME={codex_directory}") + " "
        + shlex.quote(executable) + ' "$@"\n'
    )
    manifest = {
        "account": config["run"]["account"], "codex_home": str(codex_directory),
        "configured_executable": configured, "resolved_executable": executable,
        "launcher_path": str(launcher), "launcher_sha256": sha256(script.encode()).hexdigest(),
    }
    _immutable_json(run_dir / "transport" / "manifest.json", manifest)
    if launcher.exists() and launcher.read_text(encoding="utf-8") != script:
        raise FactoryRuntimeError("account launcher changed on resume")
    if not launcher.exists():
        atomic_write_text(launcher, script)
    launcher.chmod(0o700)
    config["model"]["executable"] = str(launcher)
    return manifest


class FactoryRuntime:
    def __init__(
        self, config: Mapping[str, Any], source_manifest: Mapping[str, Any], pool: Any = None,
        *, namespace: str = "downstream", lane: str = "downstream", model_invoker: Any = None,
        stage_bindings: Mapping[str, StageBinding] | None = None,
        contract_bindings: Mapping[str, Any] | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", namespace):
            raise FactoryRuntimeError("runtime namespace must be one directory name")
        if lane not in {"downstream", "selection_codex"}:
            raise FactoryRuntimeError("factory runtime lane must be downstream or selection_codex")
        if model_invoker is not None and pool is not None:
            raise FactoryRuntimeError("an injected transport must bind its own pool at construction")
        self.contract_bindings = deepcopy(dict(contract_bindings)) if contract_bindings is not None else None
        contract_stages, prompt_overrides = {}, {}
        contract_manifest = None
        if self.contract_bindings is not None:
            from .factory_policy import AUTHORING_CONTRACT_PROFILE, pipeline_policy
            binding = self.contract_bindings
            required = {"profile_id", "prompt_overrides", "stage_bindings", "prepare_values",
                        "validate_request", "validate_response", "materialize_palette"}
            if (set(binding) != required or config.get("contract_profile") != AUTHORING_CONTRACT_PROFILE
                    or binding.get("profile_id") != AUTHORING_CONTRACT_PROFILE or pipeline_policy(config) is None):
                raise FactoryRuntimeError("contract bindings require the explicit matching candidate contract profile")
            hooks = ("prepare_values", "validate_request", "validate_response", "materialize_palette")
            if any(not callable(binding[name]) for name in hooks):
                raise FactoryRuntimeError("candidate contract hooks must be callable")
            contract_stages = dict(binding["stage_bindings"])
            prompt_overrides = deepcopy(dict(binding["prompt_overrides"]))
            legacy_files = {filename for filename, _ in PROMPT_SPECS.values()}
            authoring_prompts = {"concrete_planner", "v44_quality_plan", "logical_leap_planner_addendum",
                                 "v43_episode_joint_plan", "evidence_scale_addendum", "task_realizer",
                                 "targeted_repair", "technical_verifier"}
            for name, (filename, stage) in prompt_overrides.items():
                if name not in authoring_prompts or filename in legacy_files or stage not in contract_stages:
                    raise FactoryRuntimeError("candidate prompt overrides require new files and explicit new stages")
            contract_manifest = {
                "profile_id": binding["profile_id"], "prompt_overrides": {name: list(row) for name, row in prompt_overrides.items()},
                "hooks": {name: f"{binding[name].__module__}.{binding[name].__qualname__}" for name in hooks},
            }
        elif "contract_profile" in config:
            raise FactoryRuntimeError("selected candidate contract requires explicit constructor bindings")
        if set(contract_stages) & set(stage_bindings or {}):
            raise FactoryRuntimeError("additional and contract stage bindings overlap")
        self._contract_stages = frozenset(contract_stages)
        # Isolated studies may add an explicitly named stage, never replace a
        # production binding or create an implicit model/effort route.
        self.bindings = build_stage_bindings()
        additional = {}
        for stage, binding in {**(stage_bindings or {}), **contract_stages}.items():
            if (stage in self.bindings or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stage)
                    or not isinstance(binding, StageBinding) or not callable(binding.parser)
                    or not binding.purposes):
                raise FactoryRuntimeError("additional stage binding must be new, named and explicit")
            Draft202012Validator.check_schema(binding.schema)
            for purpose in binding.purposes:
                resolve_role(config, purpose)
                if stage in contract_stages and purpose not in {"planning", "realization", "technical_verification", "targeted_repair"}:
                    raise FactoryRuntimeError("candidate authoring contract cannot replace independent blind, review or supply policy")
            copied = StageBinding(deepcopy(binding.schema), binding.parser, tuple(binding.purposes))
            self.bindings[stage] = copied
            additional[stage] = {
                "schema": copied.schema, "purposes": list(copied.purposes),
                "parser": f"{getattr(binding.parser, '__module__', type(binding.parser).__module__)}."
                          f"{getattr(binding.parser, '__qualname__', type(binding.parser).__qualname__)}",
            }
            if stage in contract_stages:
                from functools import partial
                parser = binding.parser
                function = parser.func if isinstance(parser, partial) else parser
                additional[stage]["parser_binding"] = {
                    "function": f"{function.__module__}.{function.__qualname__}",
                    "args": list(parser.args) if isinstance(parser, partial) else [],
                    "keywords": deepcopy(parser.keywords) if isinstance(parser, partial) else {},
                }
        self.factory_config = deepcopy(dict(config))
        self.context_guard = FactoryContextGuard.from_config(config)
        self.config = _runtime_config(config)
        self.source_manifest = resolved_config_manifest(source_manifest)
        self.namespace, self.lane = namespace, lane
        self.run_id = str(config["run"]["id"])
        self.run_dir = Path(config["_output_directory"]) / namespace
        self.lock = threading.RLock()
        self._call_locks: dict[str, threading.RLock] = {}
        self.state_path = self.run_dir / "state.json"
        _immutable_json(self.run_dir / "runtime_manifest.json", {
            "schema_version": 1, "namespace": namespace, "lane": lane,
            "factory_config": resolved_config_manifest(config), "stage_config": self.config,
            "source_manifest": self.source_manifest,
            "technical_plan_projection": ({"contract_profile": contract_manifest["profile_id"],
                                            "prepare_values": contract_manifest["hooks"]["prepare_values"]}
                                           if contract_manifest is not None else TECHNICAL_PLAN_PROJECTION),
            **({"additional_stage_bindings": additional} if additional else {}),
            **({"contract_bindings": contract_manifest} if contract_manifest is not None else {}),
        })
        self.prompts = FactoryPromptStore(Path(self.config["paths"]["prompts"]), self.run_dir, prompt_overrides=prompt_overrides)
        self.schemas = {stage: binding.schema for stage, binding in self.bindings.items()}
        self.parsers = {stage: binding.parser for stage, binding in self.bindings.items()}
        self.role_profile = {
            purpose: resolve_role(self.factory_config, purpose)
            for binding in self.bindings.values() for purpose in binding.purposes
        }
        self.transport = model_invoker if model_invoker is not None else FactoryModelInvoker(
            self.config, self.run_dir, pool=pool, lane=lane,
            output_schemas=self.schemas, output_parser=self.parse_output,
        )
        self.calls_dir = Path(self.transport.calls_dir)
        if self.calls_dir.resolve() != (self.run_dir / "model_calls").resolve():
            raise FactoryRuntimeError("injected transport artifacts must use this runtime namespace")
        self.invoker = E18AttemptInvoker(
            self.transport, self.run_dir / "attempt_journal.json",
            maximum_attempts=int(config["transport"]["maximum_attempts"]),
        )
        self.state = read_json(self.state_path) if self.state_path.is_file() else {
            "pipeline": config["pipeline"], "run_id": self.run_id, "namespace": namespace,
            "completed": False, "started_at_epoch_seconds": round(time.time(), 3),
            "source_manifest_hash": content_hash(self.source_manifest),
            "call_ledger": {}, "semantic_records": {}, "results": {},
        }
        if (
            self.state.get("run_id") != self.run_id
            or self.state.get("namespace") != namespace
            or self.state.get("source_manifest_hash") != content_hash(self.source_manifest)
        ):
            raise FactoryRuntimeError("runtime state identity differs from the frozen inputs")
        self.save()

    def save(self) -> None:
        with self.lock:
            atomic_write_json(self.state_path, self.state)

    def parse_output(self, stage: str, data: Any) -> dict[str, Any]:
        try:
            parser = self.parsers[stage]
        except KeyError as exc:
            raise FactoryRuntimeError(f"unbound factory stage: {stage}") from exc
        return parser(deepcopy(data))

    def purpose_for_call(self, stage: str, call_id: str) -> str:
        purposes = self.bindings[stage].purposes
        if len(purposes) == 1:
            return purposes[0]
        if stage == "v4_blind_solver":
            if call_id.endswith("-blind-primary"):
                return "primary_blind"
            if call_id.endswith(("-blind-second", "-blind-after-ambiguity-repair")):
                return "conditional_blind"
        raise FactoryRuntimeError(f"{stage}/{call_id} requires an explicit call purpose")

    def _episode_hashes(self, parent_ids: Sequence[str]) -> dict[str, str]:
        hashes = {}
        for task_id in parent_ids:
            record = self.state["semantic_records"].get(task_id, {})
            episode = record.get("complete_task")
            if not episode:
                realization = record.get("realization") or {}
                episode = deepcopy(realization.get("task"))
                outputs = (realization.get("authoring_metadata") or {}).get("expected_test_outputs")
                if episode and isinstance(outputs, list) and len(outputs) == len(episode["test"]):
                    for pair, output in zip(episode["test"], outputs, strict=True):
                        pair["output"] = deepcopy(output)
                else:
                    episode = None
            if episode:
                hashes[task_id] = content_hash(episode)
        return hashes

    def invoke_rendered(
        self, *, purpose: str, stage: str, call_id: str, prompt: str,
        prompt_template_version: str, prompt_template_hash: str,
        parent_ids: Sequence[str] = (), response_validator: Callable[[dict[str, Any]], None] | None = None,
        arm_id: str = "factory", role: str | None = None,
        request_context: Mapping[str, Any] | None = None,
    ) -> CallResult:
        if stage not in self.bindings or purpose not in self.bindings[stage].purposes:
            raise FactoryRuntimeError(f"purpose {purpose} is not bound to stage {stage}")
        if purpose != "transport_probe":
            required_lane = "selection_codex" if purpose in UPSTREAM_PURPOSES else "downstream"
            if self.lane != required_lane:
                raise FactoryRuntimeError(f"purpose {purpose} requires pool lane {required_lane}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,180}", call_id):
            raise FactoryRuntimeError("logical call ID must be one unchanged transport artifact name")
        context = deepcopy(request_context)
        if stage in self._contract_stages:
            self.contract_bindings["validate_request"](stage, prompt, deepcopy(context))
            supplied_validator = response_validator

            def validate_contract_response(value):
                self.contract_bindings["validate_response"](stage, value, deepcopy(context))
                if supplied_validator is not None:
                    supplied_validator(value)

            response_validator = validate_contract_response
        elif request_context is not None:
            raise FactoryRuntimeError("request context requires an explicitly bound contract stage")
        settings = self.role_profile[purpose]
        context_report = self.context_guard.inspect(prompt, purpose)
        atomic_write_json(self.run_dir / "context_preflight" / stage / f"{call_id}.json", context_report)
        if context_report["requires_context_decision"]:
            raise FactoryContextError(context_report)
        key = f"{arm_id}/{stage}/{call_id}"
        logical_id = f"{stage}/{call_id}"
        fingerprint = content_hash({
            "purpose": purpose, "stage": stage, "call_id": call_id, "prompt": prompt,
            "template_version": prompt_template_version, "template_hash": prompt_template_hash,
            "schema": self.schemas[stage], "settings": settings, "parent_ids": list(parent_ids),
            **({"contract_request_context": context} if stage in self._contract_stages else {}),
        })
        with self.lock:
            call_lock = self._call_locks.setdefault(logical_id, threading.RLock())
        with call_lock:
            with self.lock:
                previous = [row for row in self.state["call_ledger"].values() if row["stage"] == stage and row["call_id"] == call_id]
                if any(row["request_fingerprint"] != fingerprint for row in previous):
                    raise FactoryRuntimeError(f"logical call changed within this run: {logical_id}")
                if key not in self.state["call_ledger"]:
                    hashes = self._episode_hashes(parent_ids)
                    self.state["call_ledger"][key] = {
                        "arm_id": arm_id, "stage": stage, "role": role or purpose,
                        "purpose": purpose, "call_id": call_id, "parent_ids": list(parent_ids),
                        "model": settings["model"], "reasoning_effort": settings["reasoning_effort"],
                        "resolved_profile": deepcopy(settings),
                        "account": self.config["run"]["account"],
                        "codex_home": self.config["model"]["codex_home"],
                        "request_fingerprint": fingerprint, "status": "reserved",
                        "log_dir": str(self.calls_dir / stage / call_id),
                        "reserved_at_epoch_seconds": round(time.time(), 6),
                        "candidate_episode_hashes": hashes,
                        "candidate_episode_hash": next(iter(hashes.values())) if len(hashes) == 1 else None,
                    }
                    self.save()
            try:
                result = self.invoker.invoke(
                    stage=stage, call_id=call_id, prompt=prompt,
                    prompt_template_version=prompt_template_version, prompt_template_hash=prompt_template_hash,
                    parent_ids=list(parent_ids), response_validator=response_validator,
                    model_name=settings["model"], reasoning_effort=settings["reasoning_effort"],
                )
            except Exception as exc:
                with self.lock:
                    # Cache revalidation failure is not a new provider failure
                    # and must not rewrite the completed call's historical cost.
                    if self.state["call_ledger"][key].get("status") == "complete":
                        raise
                    attempts = self.invoker.value["calls"].get(logical_id, {}).get("attempts", [])
                    self.state["call_ledger"][key].update(
                        status="failed", error=str(exc), attempts=len(attempts), retry_calls=max(0, len(attempts)-1),
                        completed_at_epoch_seconds=round(time.time(), 6),
                    )
                    self.save()
                raise
            marker = read_json(result.log_dir / "result.json")
            with self.lock:
                prior = self.state["call_ledger"][key]
                if prior.get("status") == "complete":
                    if prior.get("response_content_hash") != content_hash(result.value):
                        raise FactoryRuntimeError(f"completed response changed on resume: {logical_id}")
                    return result
                self.state["call_ledger"][key].update(
                    status="complete", attempts=result.attempts, retry_calls=max(0, result.attempts-1),
                    cache_source=marker.get("cache_source"), cache_key=result.cache_key, log_dir=str(result.log_dir),
                    timing=marker.get("timing"), response_content_hash=content_hash(result.value),
                    completed_at_epoch_seconds=(marker.get("timing") or {}).get("completed_at_epoch_seconds", round(time.time(), 6)),
                    reused_success=marker.get("cache_source") is not None,
                )
                self.state["call_ledger"][key].pop("error", None)
                self.save()
            return result

    def call(
        self, *, prompt_name: str, composed_with: Sequence[str] = (), call_id: str,
        values: Mapping[str, Any], parent_ids: Sequence[str], role: str, arm_id: str,
        expected: Mapping[str, Any] | None = None,
        validator: Callable[[dict[str, Any]], None] | None = None,
        forbidden_prompt_fragments: Sequence[str] = (), purpose: str | None = None,
    ) -> dict[str, Any]:
        request_context = None
        uses_contract = self.contract_bindings is not None and prompt_name in self.contract_bindings["prompt_overrides"]
        if self.contract_bindings is not None:
            values, request_context = self.contract_bindings["prepare_values"](prompt_name, deepcopy(dict(values)))
            if not uses_contract:
                request_context = None  # Unchanged blind/review stages retain their original invocation contract.
        if prompt_name in {"technical_verifier", "blind_solver"}:
            public = values.get("PUBLIC_TASK")
            if not isinstance(public, Mapping) or any("output" in pair for pair in public.get("test", [])):
                raise FactoryRuntimeError("hidden output entered verifier public task")
        if prompt_name == "technical_verifier" and not uses_contract:
            values = {**values, "CONCRETE_PLAN": technical_public_plan(values["CONCRETE_PLAN"])}
        prompt, version, digest, stage = self.prompts.render(prompt_name, values, composed_with)
        if uses_contract and stage not in self._contract_stages:
            raise FactoryRuntimeError("candidate prompt composition selected a legacy stage")
        leaks = [fragment for fragment in forbidden_prompt_fragments if fragment.casefold() in prompt.casefold()]
        if leaks:
            raise FactoryRuntimeError(f"downstream prompt leaked private label(s): {leaks}")

        def validate(value: dict[str, Any]) -> None:
            for name, wanted in (expected or {}).items():
                if value.get(name) != wanted:
                    raise FactoryRuntimeError(f"{call_id} returned wrong {name}")
            if validator is not None:
                validator(value)

        return self.invoke_rendered(
            purpose=purpose or self.purpose_for_call(stage, call_id), stage=stage, call_id=call_id,
            prompt=prompt, prompt_template_version=version, prompt_template_hash=digest,
            parent_ids=parent_ids, response_validator=validate, arm_id=arm_id, role=role,
            request_context=request_context,
        ).value
