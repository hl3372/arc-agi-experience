"""V4.5 Claude Opus 5 authoring and crossed-evaluation experiment.

This module is intentionally disjoint from the frozen V4.5 implementation.
It has no API for task-bank, reservoir, reservation, feature, or rejection-
memory mutation.  A real run requires both explicit command authorization and
fresh effort-specific Claude model attestations.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from random import Random
from string import ascii_lowercase, digits
from typing import Any, Callable, Mapping, Sequence
import argparse
import json
import re
import sys
import threading
import time

from .claude_code import (
    ClaudeCodeCallResult,
    ClaudeCodeCapabilities,
    ClaudeCodeInvocationError,
    ClaudeCodeInvoker,
    attest_claude_role_efforts,
    inspect_claude_code,
    validate_claude_attestation,
)
from .lean_pipeline import serious_blind_alternative
from .lean_v3_pipeline import classify_v3_editorial_results
from .lean_v3_rendering import build_contact_sheet, render_episode
from .model import CallResult, ModelInvoker
from .util import (
    PLACEHOLDER_RE,
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    parallel_map,
    read_json,
    render_prompt,
)
from .v4_diagnostics import (
    check_plan_fidelity,
    complete_task,
    diagnose_arc_task_v4,
    diagnose_public_arc_task_v4,
)
from .v4_pipeline import V4ControlPipeline, v4_call_accounting
from .v4_schemas import OUTPUT_SCHEMAS as V4_OUTPUT_SCHEMAS
from .v4_schemas import parse_stage_output as parse_v4_stage_output
from .v41_experiments import _anchors, _portfolio_digest, _task_metrics
from .v41_integration import (
    _committed_distribution,
    _palette_target,
    transform_realization_palette,
)
from .v41_palette import schedule_backgrounds
from .v43_episode_joint import _reference_packages
from .v431_episode_joint_quality import GLOBAL_SELECTOR_SCHEMA, parse_v431_global_selector
from .v45_claude_config import (
    PIPELINE,
    VERSION,
    load_v45_claude_config,
    snapshot_v45_claude_config,
    validate_claude_environment,
    verify_v45_integrated_baseline,
)
from .validation import validate_arc_task


class ClaudeCrossEvalError(RuntimeError):
    """Raised when the isolated experiment violates its frozen design."""


PROMPTS: dict[str, tuple[str, str]] = {
    "idea_open": ("v4_idea_open.md", "v4_idea_search"),
    "idea_concept": ("v4_idea_concept.md", "v4_idea_search"),
    "idea_composition": ("v4_idea_composition.md", "v4_idea_search"),
    "selector": ("v4_4_global_quality_selector.md", "v44_kernel_selector"),
    "concrete_planner": ("v4_concrete_planner.md", "v4_concrete_plan"),
    "quality_plan": ("v4_4_concrete_quality_planning_addendum.md", "v4_concrete_plan"),
    "task_realizer": ("v4_task_realizer.md", "v4_task_realizer"),
    "technical_verifier": ("v4_technical_verifier.md", "v4_technical_verifier"),
    "blind_solver": ("v4_blind_solver.md", "v4_blind_solver"),
    "editorial_board": ("v4_editorial_board.md", "v4_editorial_board"),
}

AUTHOR_PROVIDERS = ("codex", "claude")
EVALUATOR_PROVIDERS = ("codex", "claude")
SOURCE_FORBIDDEN = (
    "claude-authored",
    "codex-authored",
    "claude author",
    "codex author",
    "author provider",
    "experimental arm",
)


def _transport_schema(value: Any) -> Any:
    unsupported = {
        "minLength", "maxLength", "minimum", "maximum", "minItems",
        "maxItems", "uniqueItems",
    }
    if isinstance(value, dict):
        return {
            key: _transport_schema(item)
            for key, item in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [_transport_schema(item) for item in value]
    return value


OUTPUT_SCHEMAS = {
    **deepcopy(V4_OUTPUT_SCHEMAS),
    "v44_kernel_selector": _transport_schema(GLOBAL_SELECTOR_SCHEMA),
}


def parse_cross_stage(stage: str, value: str | dict[str, Any]) -> dict[str, Any]:
    if stage == "v44_kernel_selector":
        return parse_v431_global_selector(value)
    return parse_v4_stage_output(stage, value)


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


class CrossPromptStore:
    def __init__(self, prompt_dir: Path, run_dir: Path) -> None:
        self.entries: dict[str, dict[str, str]] = {}
        destination = run_dir / "prompts" / "templates"
        destination.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "pipeline": PIPELINE,
            "templates": {},
            "production_prompts_modified": False,
        }
        for name, (filename, stage) in PROMPTS.items():
            text, version, digest = load_prompt(prompt_dir / filename)
            self.entries[name] = {
                "filename": filename,
                "stage": stage,
                "text": text,
                "version": version,
                "hash": digest,
            }
            target = destination / filename
            if target.exists() and target.read_text(encoding="utf-8") != text:
                raise ClaudeCrossEvalError(f"prompt changed during resume: {filename}")
            if not target.exists():
                atomic_write_text(target, text)
            manifest["templates"][name] = {
                "filename": filename,
                "stage": stage,
                "version": version,
                "sha256": digest,
                "placeholders": sorted(PLACEHOLDER_RE.findall(text)),
            }
        path = run_dir / "prompts" / "manifest.json"
        if path.exists() and read_json(path) != manifest:
            raise ClaudeCrossEvalError("prompt manifest changed during resume")
        if not path.exists():
            atomic_write_json(path, manifest)

    def render(
        self,
        name: str,
        values: Mapping[str, Any],
        *,
        composed_with: Sequence[str] = (),
    ) -> tuple[str, str, str, str]:
        names = (name, *composed_with)
        entries = [self.entries[item] for item in names]
        text = "\n\n".join(item["text"] for item in entries)
        required = set(PLACEHOLDER_RE.findall(text))
        if required != set(values):
            raise ClaudeCrossEvalError(
                f"prompt bindings differ for {'+'.join(names)}: "
                f"missing={sorted(required-set(values))}, extra={sorted(set(values)-required)}"
            )
        prompt = render_prompt(text, dict(values))
        return (
            prompt,
            "+".join(item["version"] for item in entries),
            content_hash({"templates": [item["hash"] for item in entries], "text": text}),
            entries[-1]["stage"],
        )


def _protected_state(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    keys = (
        "committed_task_registry",
        "reservoir",
        "v4_1_feature_manifest",
        "v4_2_feature_manifest",
        "v4_4_feature_manifest",
    )
    result = {}
    for key in keys:
        value = config.get("paths", {}).get(key)
        if not value:
            continue
        path = Path(value)
        if path.is_file():
            result[key] = {"path": str(path), "sha256": _sha(path)}
    baseline = Path(config["_v4_5_baseline_manifest"])
    result["v4_5_baseline_manifest"] = {
        "path": str(baseline),
        "sha256": _sha(baseline),
    }
    return result


def _assert_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    changed = [key for key in before if before[key]["sha256"] != after.get(key, {}).get("sha256")]
    if changed:
        raise ClaudeCrossEvalError(
            "experiment mutated protected production state: " + ", ".join(changed)
        )


def _implementation_manifest(root: Path, run_dir: Path) -> dict[str, Any]:
    relatives = (
        "arcgen.lib/claude_code.py",
        "arcgen.lib/v45_claude_config.py",
        "arcgen.lib/v45_claude_cross_eval.py",
        "config/v4_5_exp_claude_opus_cross_eval.yaml",
        "prompts/v4_5_claude_code_system.md",
        "experiments/v4_5_claude_opus_cross_eval_plan.md",
    )
    value = {
        "pipeline_version": VERSION,
        "branch_source": "arcgen-v4.5-integrated-authoring-20260808",
        "files": {relative: _sha(root / relative) for relative in relatives},
    }
    path = run_dir / "implementation_manifest.json"
    if path.exists() and read_json(path) != value:
        raise ClaudeCrossEvalError("experiment implementation changed during resume")
    if not path.exists():
        atomic_write_json(path, value)
    return value


class CrossEvalRuntime:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        require_attestation: bool,
        codex_mock_responses: Mapping[str, Any] | None = None,
        claude_mock_responses: Mapping[str, Any] | None = None,
    ) -> None:
        if config.get("pipeline") != PIPELINE:
            raise ClaudeCrossEvalError("an explicit Claude cross-eval config is required")
        self.config = config
        self.run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.state_path = self.run_dir / "state.json"
        root = Path(config["_repository_root"])
        _implementation_manifest(root, self.run_dir)
        snapshot = snapshot_v45_claude_config(config)
        snapshot_path = self.run_dir / "config.snapshot.json"
        if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
            raise ClaudeCrossEvalError("config changed during resume")
        if not snapshot_path.exists():
            atomic_write_json(snapshot_path, snapshot)
        self.prompts = CrossPromptStore(Path(config["paths"]["prompts"]), self.run_dir)
        system_text, system_version, system_hash = load_prompt(
            Path(config["paths"]["claude_system_prompt"])
        )
        atomic_write_text(self.run_dir / "prompts" / "claude_system.md", system_text)
        atomic_write_json(
            self.run_dir / "prompts" / "claude_system.json",
            {"version": system_version, "sha256": system_hash},
        )
        self.capabilities: ClaudeCodeCapabilities | None
        if config["run"]["mock"]:
            self.capabilities = None
        else:
            self.capabilities = inspect_claude_code(config["claude_code"]["executable"])
            if require_attestation:
                validate_claude_attestation(
                    Path(config["claude_code"]["attestation_path"]),
                    settings=config["claude_code"],
                    capabilities=self.capabilities,
                    required_efforts=config["claude_code"]["required_attestation_efforts"],
                    maximum_age_seconds=int(
                        config["claude_code"]["attestation_maximum_age_seconds"]
                    ),
                )
        self.codex = ModelInvoker(
            config,
            self.run_dir / "codex_provider",
            output_schemas=OUTPUT_SCHEMAS,
            output_parser=parse_cross_stage,
        )
        if codex_mock_responses is not None:
            self.codex.mock = True
            self.codex.mock_responses = deepcopy(dict(codex_mock_responses))
        self.claude = ClaudeCodeInvoker(
            config["claude_code"],
            self.run_dir / "claude_provider",
            cache_root=Path(config["paths"]["cache"]),
            output_schemas=OUTPUT_SCHEMAS,
            output_parser=parse_cross_stage,
            system_prompt=system_text,
            capabilities=self.capabilities,
            mock_responses=claude_mock_responses,
        )
        if self.state_path.exists():
            self.state = read_json(self.state_path)
        else:
            self.state = {
                "schema_version": 1,
                "pipeline": PIPELINE,
                "pipeline_version": VERSION,
                "run_id": config["run"]["id"],
                "completed": False,
                "non_gating": True,
                "auto_promote": False,
                "automatic_task_commit": False,
                "production_state_mutation_allowed": False,
                "started_at_epoch_seconds": round(time.time(), 3),
                "protected_state_before": _protected_state(config),
                "call_ledger": {},
            }
            self.save()

    def save(self) -> None:
        with self.lock:
            atomic_write_json(self.state_path, self.state)

    def call(
        self,
        *,
        provider: str,
        prompt_name: str,
        call_id: str,
        values: Mapping[str, Any],
        parent_ids: Sequence[str],
        role: str,
        private_purpose: str,
        composed_with: Sequence[str] = (),
        expected: Mapping[str, Any] | None = None,
        validator: Callable[[dict[str, Any]], None] | None = None,
        forbid_source_labels: bool = True,
    ) -> dict[str, Any]:
        if provider not in {"codex", "claude"}:
            raise ClaudeCrossEvalError(f"unknown provider {provider!r}")
        prompt, version, digest, stage = self.prompts.render(
            prompt_name, values, composed_with=composed_with
        )
        if forbid_source_labels:
            leaks = [item for item in SOURCE_FORBIDDEN if item in prompt.casefold()]
            if leaks:
                raise ClaudeCrossEvalError(f"source/provider label leaked into prompt: {leaks}")

        def validate(value: dict[str, Any]) -> None:
            for key, wanted in (expected or {}).items():
                if value.get(key) != wanted:
                    raise ClaudeCrossEvalError(f"{call_id} returned wrong {key}")
            if validator:
                validator(value)

        if provider == "codex":
            settings = self.config["model"]["roles"][role]
        else:
            settings = {
                "model": self.config["claude_code"]["model_alias"],
                "reasoning_effort": self.config["claude_code"]["role_efforts"][role],
            }
        key = f"{provider}/{private_purpose}/{stage}/{call_id}"
        with self.lock:
            prior = self.state["call_ledger"].get(key)
            if prior is None:
                self.state["call_ledger"][key] = {
                    "provider": provider,
                    "private_purpose": private_purpose,
                    "stage": stage,
                    "role": role,
                    "call_id": call_id,
                    "model": settings["model"],
                    "reasoning_effort": settings["reasoning_effort"],
                    "status": "reserved",
                    "substantive_experimental_call": True,
                    "prompt_hash": content_hash(prompt),
                }
                self.save()
        try:
            if provider == "codex":
                result: CallResult | ClaudeCodeCallResult = self.codex.invoke(
                    stage=stage,
                    call_id=call_id,
                    prompt=prompt,
                    prompt_template_version=version,
                    prompt_template_hash=digest,
                    parent_ids=list(parent_ids),
                    response_validator=validate,
                    model_name=settings["model"],
                    reasoning_effort=settings["reasoning_effort"],
                )
            else:
                result = self.claude.invoke(
                    stage=stage,
                    call_id=call_id,
                    prompt=prompt,
                    prompt_template_version=version,
                    prompt_template_hash=digest,
                    parent_ids=list(parent_ids),
                    response_validator=validate,
                    reasoning_effort=settings["reasoning_effort"],
                )
        except Exception as exc:
            failed_log_dir = (
                self.run_dir
                / ("claude_provider/claude_model_calls" if provider == "claude" else "codex_provider/model_calls")
                / stage
                / call_id
            )
            attempts = len(list(failed_log_dir.glob("attempt-*.timing.json")))
            with self.lock:
                self.state["call_ledger"][key].update(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "log_dir": str(failed_log_dir),
                        "attempts": attempts,
                        "retry_calls": max(0, attempts - 1),
                    }
                )
                self.save()
            raise
        result_record = read_json(result.log_dir / "result.json")
        with self.lock:
            self.state["call_ledger"][key].update(
                {
                    "status": "complete",
                    "attempts": result.attempts,
                    "retry_calls": max(0, result.attempts - 1),
                    "cache_source": result.cache_source,
                    "cache_key": result.cache_key,
                    "log_dir": str(result.log_dir),
                    "timing": result_record.get("timing"),
                    "resolved_models": (
                        list(result.resolved_models)
                        if isinstance(result, ClaudeCodeCallResult)
                        else [settings["model"]]
                    ),
                    "usage": result_record.get("usage", {}),
                    "model_usage": result_record.get("model_usage", {}),
                    "total_cost_usd": result_record.get("total_cost_usd"),
                }
            )
            self.save()
        return result.value


def _observed_terminal_failure(
    runtime: CrossEvalRuntime,
    *,
    provider: str,
    stage: str,
    call_id: str,
) -> dict[str, Any] | None:
    """Return a previously exhausted call failure used only for recovery.

    A terminal failure must name its preserved source run.  This prevents a
    clean recovery run from spending another hour repeating two known timeout
    attempts while keeping the missing experimental observation explicit.
    """

    for row in runtime.config["claude_cross_eval"].get(
        "observed_terminal_failures", []
    ):
        if (
            row.get("provider") == provider
            and row.get("stage") == stage
            and row.get("call_id") == call_id
        ):
            return deepcopy(dict(row))
    return None


def _json_after_marker(prompt: str, marker: str) -> Any:
    start = prompt.find(marker)
    if start < 0:
        raise ClaudeCrossEvalError(f"source prompt lacks marker {marker!r}")
    start = prompt.find("[", start + len(marker))
    if start < 0:
        raise ClaudeCrossEvalError("source prompt lacks calibration JSON")
    try:
        value, _ = json.JSONDecoder().raw_decode(prompt[start:])
    except json.JSONDecodeError as exc:
        raise ClaudeCrossEvalError("source prompt has malformed calibration JSON") from exc
    if not isinstance(value, list) or len(value) != 4 or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ClaudeCrossEvalError("source prompt calibration set is malformed")
    return value


def _source_calibrations(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(config["paths"]["frozen_generation_source_run"])
    pattern = "selection_graph/v4_reservoir_supply/*topup-01/model_calls/v4_idea_search/*/prompt.md"
    paths = sorted(root.glob(pattern))
    mode_patterns = {
        "open": "open-supply",
        "concept_led": "concept-led-supply",
        "composition_led": "composition-led-supply",
    }
    result: dict[str, dict[str, Any]] = {}
    for mode, needle in mode_patterns.items():
        source = next((path for path in paths if needle in str(path)), None)
        if source is None:
            raise ClaudeCrossEvalError(f"cannot locate frozen {mode} calibration prompt")
        text = source.read_text(encoding="utf-8")
        result[mode] = {
            "descriptions": _json_after_marker(text, "Calibration descriptions:"),
            "source_prompt": str(source),
            "source_prompt_sha256": _sha(source),
        }
    all_descriptions = [item for row in result.values() for item in row["descriptions"]]
    if len(all_descriptions) != len(set(all_descriptions)):
        raise ClaudeCrossEvalError("frozen calibration examples repeat across search modes")
    return result


def freeze_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze comparison inputs and state without invoking either model."""

    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "frozen_experiment_manifest.private.json"
    if path.exists():
        return read_json(path)
    calibrations = _source_calibrations(config)
    modes = list(config["claude_cross_eval"]["normal_search_modes"])
    idea_pairs = []
    private_mapping: dict[str, str] = {}
    for mode_index, mode in enumerate(modes, 1):
        shared_ids = [
            "q" + content_hash(
                {"seed": config["seed"], "mode": mode, "idea": index}
            )[:13]
            for index in range(1, 5)
        ]
        provider_ids = {}
        for provider in AUTHOR_PROVIDERS:
            ids = [
                "k" + content_hash(
                    {
                        "seed": config["seed"], "provider": provider,
                        "mode": mode, "idea": index,
                    }
                )[:13]
                for index in range(1, 5)
            ]
            provider_ids[provider] = ids
            private_mapping.update({task_id: provider for task_id in ids})
        idea_pairs.append(
            {
                "mode": mode,
                "mode_index": mode_index,
                "shared_call_id": f"paired-{mode.replace('_', '-')}-search",
                "shared_candidate_ids": shared_ids,
                "provider_candidate_ids": provider_ids,
                "calibration_descriptions": calibrations[mode]["descriptions"],
                "calibration_source_prompt": calibrations[mode]["source_prompt"],
                "calibration_source_prompt_sha256": calibrations[mode]["source_prompt_sha256"],
            }
        )
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "pipeline_version": VERSION,
        "seed": config["seed"],
        "v4_5_baseline": deepcopy(config["_v4_5_baseline_verification"]),
        "protected_state_before": _protected_state(config),
        "idea_pairs": idea_pairs,
        "private_candidate_author_mapping": private_mapping,
        "same_rendered_idea_prompt_within_each_pair": True,
        "same_calibration_descriptions": True,
        "same_output_schema": True,
        "same_role_effort": True,
        "codex_selector_is_gating": True,
        "claude_selector_is_shadow": True,
        "source_labels_hidden_downstream": True,
        "repairs_enabled": False,
        "automatic_task_commit": False,
        "production_state_mutation_allowed": False,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        run_dir / "frozen_experiment_manifest.public.json",
        {key: deepcopy(item) for key, item in value.items() if key != "private_candidate_author_mapping"},
    )
    return value


def _prompt_name_for_mode(mode: str) -> str:
    return {
        "open": "idea_open",
        "concept_led": "idea_concept",
        "composition_led": "idea_composition",
    }[mode]


def _generate_kernels(
    runtime: CrossEvalRuntime,
    design: Mapping[str, Any],
) -> list[dict[str, Any]]:
    path = runtime.run_dir / "kernel_supply_manifest.private.json"
    if path.exists():
        return read_json(path)["kernels"]
    specs = [
        (provider, pair)
        for pair in design["idea_pairs"]
        for provider in AUTHOR_PROVIDERS
    ]

    def generate(spec: tuple[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        provider, pair = spec
        shared_ids = list(pair["shared_candidate_ids"])
        shared_call_id = str(pair["shared_call_id"])

        def validate(value: dict[str, Any]) -> None:
            if [item["candidate_id"] for item in value["ideas"]] != shared_ids:
                raise ClaudeCrossEvalError("paired idea response changed the shared IDs")

        response = runtime.call(
            provider=provider,
            prompt_name=_prompt_name_for_mode(str(pair["mode"])),
            call_id=f"{provider}-{shared_call_id}",
            values={
                "CALIBRATION_DESCRIPTIONS": list(pair["calibration_descriptions"]),
                "SEARCH_CALL_ID": shared_call_id,
                "CANDIDATE_IDS": shared_ids,
            },
            parent_ids=[],
            role="idea_search",
            private_purpose=f"private-{provider}-authoring",
            expected={"search_call_id": shared_call_id, "search_mode": pair["mode"]},
            validator=validate,
        )
        mapped_ids = list(pair["provider_candidate_ids"][provider])
        rows = []
        for index, item in enumerate(response["ideas"]):
            kernel = deepcopy(item)
            kernel["candidate_id"] = mapped_ids[index]
            rows.append(
                {
                    "candidate_id": mapped_ids[index],
                    "private_author_provider": provider,
                    "search_mode": pair["mode"],
                    "kernel": kernel,
                    "shared_pair_call_id": shared_call_id,
                    "shared_prompt_hash": runtime.state["call_ledger"][
                        f"{provider}/private-{provider}-authoring/v4_idea_search/{provider}-{shared_call_id}"
                    ]["prompt_hash"],
                }
            )
        return rows

    groups = parallel_map(
        generate,
        specs,
        min(len(specs), int(runtime.config["model"]["concurrency"])),
    )
    kernels = [item for group in groups for item in group]
    expected = 2 * 3 * 4
    if len(kernels) != expected or len({item["candidate_id"] for item in kernels}) != expected:
        raise ClaudeCrossEvalError("paired idea generation did not produce 24 unique kernels")
    prompt_pairs = []
    for pair in design["idea_pairs"]:
        hashes = {
            provider: next(
                item["shared_prompt_hash"]
                for item in kernels
                if item["private_author_provider"] == provider
                and item["shared_pair_call_id"] == pair["shared_call_id"]
            )
            for provider in AUTHOR_PROVIDERS
        }
        if len(set(hashes.values())) != 1:
            raise ClaudeCrossEvalError("paired providers received different idea prompts")
        prompt_pairs.append(
            {"mode": pair["mode"], "prompt_hashes": hashes, "exact_match": True}
        )
    value = {
        "schema_version": 1,
        "kernel_count": len(kernels),
        "counts_by_author": dict(Counter(item["private_author_provider"] for item in kernels)),
        "counts_by_mode": dict(Counter(item["search_mode"] for item in kernels)),
        "paired_prompt_checks": prompt_pairs,
        "kernels": kernels,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        runtime.run_dir / "kernel_supply_manifest.blinded.json",
        {
            "schema_version": 1,
            "source_labels_removed": True,
            "kernels": [
                {
                    "candidate_id": item["candidate_id"],
                    "search_mode": item["search_mode"],
                    "kernel": item["kernel"],
                }
                for item in kernels
            ],
        },
    )
    return kernels


def _selector_response(
    runtime: CrossEvalRuntime,
    kernels: Sequence[Mapping[str, Any]],
    *,
    provider: str,
    purpose: str,
) -> dict[str, Any]:
    packages, accepted, reservations, rejected = _reference_packages(
        runtime.config, kernels
    )
    expected_ids = {str(item["candidate_id"]) for item in kernels}
    selector_id = "opaque-kernel-editor-" + content_hash(
        {"seed": runtime.config["seed"], "candidates": sorted(expected_ids)}
    )[:12]

    def validate(value: dict[str, Any]) -> None:
        assessed = {item["candidate_id"] for item in value["assessments"]}
        if assessed != expected_ids or set(value["quality_order"]) != expected_ids:
            raise ClaudeCrossEvalError("selector did not assess the complete frozen pool")

    return runtime.call(
        provider=provider,
        prompt_name="selector",
        call_id=f"{provider}-{selector_id}",
        values={
            "CANDIDATES_WITH_REFERENCES": packages,
            "ACCEPTED_BANK_DIGEST": accepted,
            "IN_FLIGHT_RESERVATIONS": reservations,
            "REJECTED_MEMORY_DIGEST": rejected,
            "SELECTOR_ID": selector_id,
        },
        parent_ids=sorted(expected_ids),
        role="batch_selector",
        private_purpose=purpose,
        expected={"selector_id": selector_id},
        validator=validate,
    )


def _select_kernels(
    runtime: CrossEvalRuntime,
    kernels: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    path = runtime.run_dir / "selected_kernel_manifest.private.json"
    if path.exists():
        return read_json(path)["selected"]
    # The exact same source-blind prompt is sent to both providers. Only the
    # Codex result is allowed to determine development; Claude is a judgment
    # comparison, not a hidden second gate.
    responses = parallel_map(
        lambda provider: (
            provider,
            _selector_response(
                runtime,
                kernels,
                provider=provider,
                purpose=(
                    "gating-source-blind-selection"
                    if provider == "codex"
                    else "shadow-source-blind-selection"
                ),
            ),
        ),
        list(EVALUATOR_PROVIDERS),
        2,
    )
    by_provider = dict(responses)
    codex = by_provider["codex"]
    claude = by_provider["claude"]
    rows = {str(item["candidate_id"]): item for item in kernels}
    assessments = {
        str(item["candidate_id"]): item for item in codex["assessments"]
    }
    max_per = int(runtime.config["claude_cross_eval"]["maximum_development_per_author"])
    max_total = int(runtime.config["claude_cross_eval"]["maximum_total_development"])
    chosen: list[str] = []
    counts = Counter()
    exclusions: list[dict[str, Any]] = []
    for candidate_id in codex["quality_order"]:
        assessment = assessments[candidate_id]
        author = str(rows[candidate_id]["private_author_provider"])
        siblings = [
            other
            for other in chosen
            if other in assessment["same_minimal_procedure_with"]
            or candidate_id in assessments[other]["same_minimal_procedure_with"]
        ]
        reason: str | None = None
        if assessment["recommendation"] != "develop":
            reason = assessment["recommendation"]
        elif siblings:
            reason = "same_minimal_procedure_sibling"
        elif counts[author] >= max_per:
            reason = "author_sampling_ceiling"
        elif len(chosen) >= max_total:
            reason = "global_development_ceiling"
        if reason is not None:
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "reason": reason,
                    "same_procedure_with_selected": siblings,
                }
            )
            continue
        chosen.append(candidate_id)
        counts[author] += 1
    selected = []
    per_author_priority = Counter()
    for priority, candidate_id in enumerate(chosen, 1):
        source = deepcopy(rows[candidate_id])
        author = source["private_author_provider"]
        per_author_priority[author] += 1
        brief = deepcopy(assessments[candidate_id]["development_brief"])
        if brief is None:
            raise ClaudeCrossEvalError("developed selector item has no editorial brief")
        brief["candidate_id"] = candidate_id
        selected.append(
            {
                **source,
                "selector_global_priority": priority,
                "author_priority": per_author_priority[author],
                "editorial_brief": brief,
                "selector_assessment": deepcopy(assessments[candidate_id]),
            }
        )
    codex_order = {candidate_id: index for index, candidate_id in enumerate(codex["quality_order"], 1)}
    claude_order = {candidate_id: index for index, candidate_id in enumerate(claude["quality_order"], 1)}
    selector_comparison = {
        "schema_version": 1,
        "same_prompt_hash": (
            runtime.state["call_ledger"][next(
                key for key in runtime.state["call_ledger"]
                if key.startswith("codex/gating-source-blind-selection/")
            )]["prompt_hash"]
            == runtime.state["call_ledger"][next(
                key for key in runtime.state["call_ledger"]
                if key.startswith("claude/shadow-source-blind-selection/")
            )]["prompt_hash"]
        ),
        "source_labels_hidden": True,
        "gating_provider": "codex",
        "shadow_provider": "claude",
        "selected_count": len(selected),
        "selected_by_private_author": dict(counts),
        "recommendation_agreement_count": sum(
            next(item for item in codex["assessments"] if item["candidate_id"] == candidate_id)["recommendation"]
            == next(item for item in claude["assessments"] if item["candidate_id"] == candidate_id)["recommendation"]
            for candidate_id in rows
        ),
        "mean_absolute_rank_difference": round(
            sum(abs(codex_order[item] - claude_order[item]) for item in rows) / len(rows), 3
        ),
        "codex_response": codex,
        "claude_shadow_response": claude,
    }
    value = {
        "schema_version": 1,
        "selected": selected,
        "exclusions": exclusions,
        "selector_comparison": selector_comparison,
        "source_balancing_applied_only_after_source_blind_quality_order": True,
        "hold_or_reject_never_promoted_for_balance": True,
    }
    atomic_write_json(path, value)
    atomic_write_json(runtime.run_dir / "selector_comparison.json", selector_comparison)
    atomic_write_json(
        runtime.run_dir / "selected_kernel_manifest.blinded.json",
        {
            "schema_version": 1,
            "source_labels_removed": True,
            "selected": [
                {
                    key: deepcopy(item)
                    for key, item in row.items()
                    if key not in {"private_author_provider", "shared_pair_call_id", "shared_prompt_hash"}
                }
                for row in selected
            ],
        },
    )
    return selected


def _plan_selected(
    runtime: CrossEvalRuntime,
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    path = runtime.run_dir / "planning_manifest.private.json"
    if path.exists():
        return {item["candidate_id"]: item for item in read_json(path)["records"]}

    def plan_one(source: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(source["candidate_id"])
        provider = str(source["private_author_provider"])
        brief = deepcopy(source["editorial_brief"])
        # The initial attribution study uses ordinary planning only. Optional
        # V4.5 coverage interventions would add a second variable.
        brief["optional_development_suggestion"] = None
        try:
            response = runtime.call(
                provider=provider,
                prompt_name="concrete_planner",
                composed_with=("quality_plan",),
                call_id=f"{task_id}-concrete-plan",
                values={
                    "CANDIDATE": deepcopy(source["kernel"]),
                    "EDITORIAL_BRIEF": brief,
                    "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(brief["route"]),
                    "CANDIDATE_ID": task_id,
                },
                parent_ids=[task_id],
                role="concrete_planner",
                private_purpose=f"private-{provider}-authoring",
                expected={"candidate_id": task_id},
            )
        except Exception as exc:
            # Candidate-level isolation is part of the experimental contract:
            # one provider timeout must be measured against that candidate, not
            # invalidate independently completed matched work.
            return {
                "candidate_id": task_id,
                "private_author_provider": provider,
                "search_mode": source["search_mode"],
                "kernel": deepcopy(source["kernel"]),
                "editorial_brief": brief,
                "selector_global_priority": source["selector_global_priority"],
                "plan": None,
                "status": "plan_operational_failure",
                "abort_reason": "",
                "failure_reason": str(exc),
            }
        return {
            "candidate_id": task_id,
            "private_author_provider": provider,
            "search_mode": source["search_mode"],
            "kernel": deepcopy(source["kernel"]),
            "editorial_brief": brief,
            "selector_global_priority": source["selector_global_priority"],
            "plan": response,
            "status": "plan_ready" if response.get("status") == "ready" else "plan_aborted",
            "abort_reason": response.get("abort_reason", ""),
        }

    records = parallel_map(
        plan_one,
        list(selected),
        min(len(selected), int(runtime.config["model"]["concurrency"])),
    ) if selected else []
    for item in records:
        directory = runtime.run_dir / "plans" / item["private_author_provider"]
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / f"{item['candidate_id']}.json", item)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "selected_count": len(selected),
            "ready_count": sum(item["status"] == "plan_ready" for item in records),
            "aborted_count": sum(item["status"] == "plan_aborted" for item in records),
            "operational_failure_count": sum(
                item["status"] == "plan_operational_failure" for item in records
            ),
            "records": records,
        },
    )
    return {item["candidate_id"]: item for item in records}


def _palette_assignments(
    runtime: CrossEvalRuntime,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int | None]:
    path = runtime.run_dir / "palette_assignments.json"
    if path.exists():
        return {key: value for key, value in read_json(path)["assignments"].items()}
    ready = [item["candidate_id"] for item in records if item["status"] == "plan_ready"]
    target, reference = _palette_target(runtime.config)
    committed = _committed_distribution(runtime.config)
    assignments = schedule_backgrounds(
        ready,
        seed=int(runtime.config["seed"]),
        current_counts=committed["counts"],
        target_distribution=target,
        eligible={task_id: True for task_id in ready},
    )
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "assignments": assignments,
            "target_distribution": {str(key): value for key, value in target.items()},
            "real_reference": reference,
            "committed_distribution": committed,
            "orthogonal_deterministic_v4_5_palette_policy": True,
        },
    )
    return assignments


def _realize_plans(
    runtime: CrossEvalRuntime,
    plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    path = runtime.run_dir / "realization_manifest.private.json"
    if path.exists():
        return {item["candidate_id"]: item for item in read_json(path)["records"]}
    assignments = _palette_assignments(runtime, list(plans.values()))

    def realize_one(record: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(record["candidate_id"])
        provider = str(record["private_author_provider"])
        if record["status"] != "plan_ready":
            reason = (
                "not_attempted_plan_failure"
                if record["status"] == "plan_operational_failure"
                else "not_attempted_plan_aborted"
            )
            return {**deepcopy(record), "realization_status": reason}
        plan = deepcopy(record["plan"])
        try:
            response = runtime.call(
                provider=provider,
                prompt_name="task_realizer",
                call_id=f"{task_id}-realization",
                values={"CONCRETE_PLAN": plan, "CANDIDATE_ID": task_id},
                parent_ids=[task_id],
                role="task_realizer",
                private_purpose=f"private-{provider}-authoring",
                expected={"candidate_id": task_id},
            )
            final_plan, final_response, palette = transform_realization_palette(
                candidate_id=task_id,
                plan=plan,
                realization=response,
                target_background=assignments.get(task_id),
                palette_config=runtime.config["palette_experiment"],
            )
            if final_response.get("status") != "realized":
                return {
                    **deepcopy(record),
                    "plan": final_plan,
                    "realization": final_response,
                    "palette_treatment": palette,
                    "realization_status": "refused",
                    "failure_reason": final_response.get("refusal_reason", "realizer refused"),
                }
            fidelity = check_plan_fidelity(final_plan, final_response)
            public_validation = validate_arc_task(
                final_response["task"], require_test_outputs=False
            )
            full = complete_task(final_response)
            full_validation = validate_arc_task(full, require_test_outputs=True)
            diagnostics = diagnose_arc_task_v4(full) if full_validation.valid else None
            failures = [
                *fidelity.get("objective_failures", []),
                *public_validation.errors,
                *full_validation.errors,
                *(diagnostics.get("errors", []) if diagnostics and not diagnostics["hard_valid"] else []),
            ]
            if failures:
                return {
                    **deepcopy(record),
                    "plan": final_plan,
                    "realization": final_response,
                    "palette_treatment": palette,
                    "plan_fidelity": fidelity,
                    "realization_status": "invalid",
                    "failure_reason": "; ".join(failures),
                }
            return {
                **deepcopy(record),
                "plan": final_plan,
                "realization": final_response,
                "complete_task": full,
                "palette_treatment": palette,
                "plan_fidelity": fidelity,
                "deterministic_diagnostics": diagnostics,
                "technical_diagnostics": diagnose_public_arc_task_v4(final_response["task"]),
                "task_metrics": _task_metrics(full),
                "realization_status": "realized",
            }
        except Exception as exc:
            return {
                **deepcopy(record),
                "realization_status": "operational_failure",
                "failure_reason": str(exc),
            }

    values = parallel_map(
        realize_one,
        list(plans.values()),
        min(len(plans), int(runtime.config["model"]["concurrency"])),
    ) if plans else []
    for record in values:
        if record["realization_status"] != "realized":
            continue
        task_id = record["candidate_id"]
        public_dir = runtime.run_dir / "tasks" / "anonymous"
        private_dir = runtime.run_dir / "tasks" / "private_dossiers"
        public_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(public_dir / f"{task_id}.arc.json", record["complete_task"])
        atomic_write_json(private_dir / f"{task_id}.authoring.json", record)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "records": values,
            "funnel": {
                "selected": len(plans),
                "plans_ready": sum(item["status"] == "plan_ready" for item in plans.values()),
                "realized": sum(item["realization_status"] == "realized" for item in values),
                "refused": sum(item["realization_status"] == "refused" for item in values),
                "invalid": sum(item["realization_status"] == "invalid" for item in values),
                "operational_failure": sum(item["realization_status"] == "operational_failure" for item in values),
            },
        },
    )
    return {item["candidate_id"]: item for item in values}


def _opaque_id(seed: int, namespace: str, task_id: str) -> str:
    rng = Random(int(content_hash({"seed": seed, "namespace": namespace, "task": task_id})[:15], 16))
    alphabet = ascii_lowercase + digits
    return "".join(rng.choice(alphabet) for _ in range(12))


def _technical_evaluation(
    record: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    hidden = record["realization"]["authoring_metadata"]["expected_test_outputs"]
    hidden_match = response["derived_test_outputs"] == hidden
    required = set(record["technical_diagnostics"].get("pair_economy_review_codes", []))
    assessed = {
        item["flag_code"]: item for item in response["diagnostic_flag_assessments"]
    }
    unresolved = [
        code
        for code in sorted(required)
        if code not in assessed or not assessed[code]["justified"]
    ]
    passed = (
        response["status"] == "pass"
        and response["mechanically_correct"]
        and response["test_supported"]
        and response["one_answer_supported"]
        and response["plan_fidelity"] != "fail"
        and response["evidence_sufficiency"] == "sufficient"
        and response["pair_economy"] != "fail"
        and response["composition_preserved"]
        and not response["arbitrary_exceptions"]
        and not response["unsupported_test_semantics"]
        and not response["unapproved_realizer_decisions"]
        and hidden_match
        and not unresolved
        and response["confidence"] != "low"
    )
    return {
        "response": deepcopy(dict(response)),
        "hidden_output_match": hidden_match,
        "unresolved_pair_economy_flags": unresolved,
        "passed": passed,
    }


def _crossed_technical(
    runtime: CrossEvalRuntime,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    path = runtime.run_dir / "crossed_technical_results.private.json"
    if path.exists():
        return read_json(path)["results"]
    specs = [
        (provider, task_id)
        for task_id, record in records.items()
        if record.get("realization_status") == "realized"
        for provider in EVALUATOR_PROVIDERS
    ]

    def one(spec: tuple[str, str]) -> tuple[str, str, dict[str, Any]]:
        provider, task_id = spec
        record = records[task_id]
        opaque = _opaque_id(runtime.config["seed"], f"technical-{provider}", task_id)
        verifier = _opaque_id(runtime.config["seed"], f"verifier-{provider}", task_id)
        try:
            response = runtime.call(
                provider=provider,
                prompt_name="technical_verifier",
                call_id=f"{provider}-technical-{task_id}",
                values={
                    "INTENDED_RULE": record["realization"]["authoring_metadata"]["intended_rule"],
                    "CONCRETE_PLAN": record["plan"],
                    "PUBLIC_TASK": record["realization"]["task"],
                    "DIAGNOSTICS": record["technical_diagnostics"],
                    "TASK_ID": opaque,
                    "VERIFIER_ID": verifier,
                },
                parent_ids=[task_id],
                role="technical_verifier",
                private_purpose=f"crossed-{provider}-technical",
                expected={"task_id": opaque, "verifier_id": verifier},
            )
        except Exception as exc:
            return task_id, provider, {
                "evaluator_provider": provider,
                "opaque_task_id": opaque,
                "operational_failure": True,
                "error": str(exc),
                "response": None,
                "hidden_output_match": None,
                "unresolved_pair_economy_flags": [],
                "passed": False,
                "author_provider_hidden": True,
                "hidden_output_withheld": True,
            }
        evaluation = _technical_evaluation(record, response)
        evaluation.update(
            {
                "evaluator_provider": provider,
                "opaque_task_id": opaque,
                "operational_failure": False,
                "author_provider_hidden": True,
                "hidden_output_withheld": True,
            }
        )
        return task_id, provider, evaluation

    rows = parallel_map(
        one,
        specs,
        min(len(specs), int(runtime.config["model"]["concurrency"])),
    ) if specs else []
    results: dict[str, dict[str, Any]] = {}
    for task_id, provider, result in rows:
        results.setdefault(task_id, {})[provider] = result
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "both_providers_evaluate_every_realized_task": True,
            "author_provider_hidden": True,
            "hidden_output_withheld_and_compared_programmatically": True,
            "results": results,
        },
    )
    return results


def _public_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train": deepcopy(task["train"]),
        "test": [{"input": deepcopy(pair["input"])} for pair in task["test"]],
    }


def _crossed_blind(
    runtime: CrossEvalRuntime,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    path = runtime.run_dir / "crossed_blind_results.private.json"
    if path.exists():
        return read_json(path)["results"]
    specs = [
        (provider, task_id)
        for task_id, record in records.items()
        if record.get("realization_status") == "realized"
        for provider in EVALUATOR_PROVIDERS
    ]

    def one(spec: tuple[str, str]) -> tuple[str, str, dict[str, Any]]:
        provider, task_id = spec
        record = records[task_id]
        public = _public_task(record["complete_task"])
        if any("output" in pair for pair in public["test"]):
            raise ClaudeCrossEvalError("hidden output entered blind prompt")
        opaque = _opaque_id(runtime.config["seed"], f"blind-{provider}", task_id)
        solver = _opaque_id(runtime.config["seed"], f"solver-{provider}", task_id)
        call_id = f"{provider}-blind-{task_id}"
        observed = _observed_terminal_failure(
            runtime,
            provider=provider,
            stage="v4_blind_solver",
            call_id=call_id,
        )
        try:
            if observed is not None:
                raise ClaudeCrossEvalError(
                    "previously exhausted terminal call: "
                    + str(observed.get("error", "unknown operational failure"))
                )
            response = runtime.call(
                provider=provider,
                prompt_name="blind_solver",
                call_id=call_id,
                values={"PUBLIC_TASK": public, "TASK_ID": opaque, "SOLVER_ID": solver},
                parent_ids=[task_id],
                role="blind_solver",
                private_purpose=f"crossed-{provider}-blind",
                expected={"task_id": opaque, "solver_id": solver},
            )
        except Exception as exc:
            return task_id, provider, {
                "evaluator_provider": provider,
                "response": None,
                "primary_matches_hidden": None,
                "serious_different_output_alternative": None,
                "operational_failure": True,
                "error": str(exc),
                "observed_terminal_failure": observed,
                "author_provider_hidden": True,
                "author_information_withheld": True,
                "hidden_output_withheld": True,
            }
        hidden = record["realization"]["authoring_metadata"]["expected_test_outputs"]
        return task_id, provider, {
            "evaluator_provider": provider,
            "response": response,
            "primary_matches_hidden": response["predicted_test_outputs"] == hidden,
            "serious_different_output_alternative": serious_blind_alternative(
                response, hidden
            ),
            "operational_failure": False,
            "author_provider_hidden": True,
            "author_information_withheld": True,
            "hidden_output_withheld": True,
        }

    rows = parallel_map(
        one,
        specs,
        min(len(specs), int(runtime.config["model"]["concurrency"])),
    ) if specs else []
    results: dict[str, dict[str, Any]] = {}
    for task_id, provider, result in rows:
        results.setdefault(task_id, {})[provider] = result
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "both_providers_evaluate_every_realized_task": True,
            "author_provider_hidden": True,
            "author_information_withheld": True,
            "results": results,
        },
    )
    return results


def _historical_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not config["claude_cross_eval"].get("historical_blind_calibration_enabled"):
        return []
    manifest = read_json(Path(config["paths"]["historical_case_manifest"]))
    if manifest.get("schema_version") != 2 or not isinstance(manifest.get("cases"), list):
        raise ClaudeCrossEvalError("historical calibration manifest is malformed")
    root = Path(config["_repository_root"])
    limit = int(config["claude_cross_eval"]["historical_blind_calibration_maximum"])
    cases = []
    for raw in manifest["cases"][:limit]:
        source = Path(str(raw["task_path"]))
        source = source if source.is_absolute() else root / source
        task = read_json(source)
        validation = validate_arc_task(task, require_test_outputs=True)
        if not validation.valid:
            raise ClaudeCrossEvalError(
                f"invalid historical task {raw['case_id']}: {'; '.join(validation.errors)}"
            )
        cases.append(
            {
                "case_id": raw["case_id"],
                "reference_label": raw["reference_label"],
                "answer_support_problem": raw["answer_support_problem"],
                "reference_certainty": raw["reference_certainty"],
                "reference_reason": raw["reference_reason"],
                "source_task_path": str(source.resolve()),
                "complete_task": task,
            }
        )
    return cases


def _historical_blind_calibration(
    runtime: CrossEvalRuntime,
) -> dict[str, Any]:
    path = runtime.run_dir / "historical_blind_calibration.private.json"
    if path.exists():
        return read_json(path)
    cases = _historical_cases(runtime.config)
    specs = [(provider, case) for case in cases for provider in EVALUATOR_PROVIDERS]

    def one(spec: tuple[str, Mapping[str, Any]]) -> tuple[str, str, dict[str, Any]]:
        provider, case = spec
        case_id = str(case["case_id"])
        public = _public_task(case["complete_task"])
        opaque = _opaque_id(runtime.config["seed"], f"history-{provider}", case_id)
        solver = _opaque_id(runtime.config["seed"], f"history-solver-{provider}", case_id)
        try:
            response = runtime.call(
                provider=provider,
                prompt_name="blind_solver",
                call_id=f"{provider}-historical-{case_id}",
                values={"PUBLIC_TASK": public, "TASK_ID": opaque, "SOLVER_ID": solver},
                parent_ids=[case_id],
                role="blind_solver",
                private_purpose=f"historical-{provider}-blind-calibration",
                expected={"task_id": opaque, "solver_id": solver},
            )
        except Exception as exc:
            return case_id, provider, {
                "response": None,
                "primary_matches_reference_output": None,
                "serious_different_output_alternative": None,
                "operational_failure": True,
                "error": str(exc),
            }
        hidden = [pair["output"] for pair in case["complete_task"]["test"]]
        return case_id, provider, {
            "response": response,
            "primary_matches_reference_output": response["predicted_test_outputs"] == hidden,
            "serious_different_output_alternative": serious_blind_alternative(
                response, hidden
            ),
            "operational_failure": False,
        }

    rows = parallel_map(
        one,
        specs,
        min(len(specs), int(runtime.config["model"]["concurrency"])),
    ) if specs else []
    by_case: dict[str, dict[str, Any]] = {}
    for case_id, provider, result in rows:
        by_case.setdefault(case_id, {})[provider] = result
    comparisons = []
    for case in cases:
        case_id = case["case_id"]
        result = by_case[case_id]
        comparisons.append(
            {
                "case_id": case_id,
                "reference_label": case["reference_label"],
                "answer_support_problem": case["answer_support_problem"],
                "reference_certainty": case["reference_certainty"],
                "reference_reason": case["reference_reason"],
                "codex": result["codex"],
                "claude": result["claude"],
                "serious_alternative_agreement": (
                    result["codex"]["serious_different_output_alternative"] is not None
                    and result["claude"]["serious_different_output_alternative"] is not None
                    and result["codex"]["serious_different_output_alternative"]
                    == result["claude"]["serious_different_output_alternative"]
                ),
            }
        )
    value = {
        "schema_version": 1,
        "case_count": len(cases),
        "same_cases_and_prompts_by_provider": True,
        "comparisons": comparisons,
    }
    atomic_write_json(path, value)
    return value


def _board_package(
    runtime: CrossEvalRuntime,
    records: Mapping[str, Mapping[str, Any]],
    group: Sequence[str],
    *,
    group_index: int,
    board_index: int,
) -> dict[str, Any]:
    anchors = _anchors(runtime.config)
    rng = Random(
        int(runtime.config["seed"]) + group_index * 1009 + board_index * 37
    )
    rows = [
        (
            f"anchor:{anchor.task_id}",
            anchor.rule_description,
            {"train": list(anchor.train), "test": list(anchor.test)},
        )
        for anchor in anchors
    ] + [
        (
            f"generated:{task_id}",
            records[task_id]["realization"]["authoring_metadata"]["intended_rule"],
            records[task_id]["complete_task"],
        )
        for task_id in group
    ]
    rng.shuffle(rows)
    used: set[str] = set()
    mapping: dict[str, str] = {}
    package = []
    for source, rule, episode in rows:
        while True:
            opaque = "".join(rng.choice(ascii_lowercase + digits) for _ in range(12))
            if opaque not in used:
                used.add(opaque)
                break
        mapping[opaque] = source
        package.append({"task_id": opaque, "rule_summary": rule, "episode": episode})
    board_id = "board-" + content_hash(
        {"seed": runtime.config["seed"], "group": group_index, "board": board_index}
    )[:14]
    return {
        "group_index": group_index,
        "board_index": board_index,
        "board_id": board_id,
        "generated_task_ids": list(group),
        "mapping": mapping,
        "package": package,
    }


def _crossed_boards(
    runtime: CrossEvalRuntime,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    path = runtime.run_dir / "crossed_editorial_results.private.json"
    if path.exists():
        return read_json(path)
    task_ids = sorted(
        [
            task_id
            for task_id, record in records.items()
            if record.get("realization_status") == "realized"
        ],
        key=lambda task_id: content_hash({"seed": runtime.config["seed"], "task": task_id}),
    )
    packages = []
    for offset in range(0, len(task_ids), 6):
        group = task_ids[offset : offset + 6]
        group_index = offset // 6 + 1
        for board_index in (1, 2):
            packages.append(
                _board_package(
                    runtime, records, group,
                    group_index=group_index, board_index=board_index,
                )
            )
    portfolio = _portfolio_digest(runtime.config)
    specs = [
        (provider, package)
        for package in packages
        for provider in EVALUATOR_PROVIDERS
    ]

    def one(spec: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
        provider, package = spec
        expected_ids = set(package["mapping"])

        def validate(value: dict[str, Any]) -> None:
            returned = [item["task_id"] for item in value["judgments"]]
            if len(returned) != len(set(returned)) or set(returned) != expected_ids:
                raise ClaudeCrossEvalError("editorial board did not judge every opaque item")

        try:
            response = runtime.call(
                provider=provider,
                prompt_name="editorial_board",
                call_id=f"{provider}-{package['board_id']}",
                values={
                    "BLINDED_TASKS": package["package"],
                    "PORTFOLIO_DIGEST": portfolio,
                    "BOARD_ID": package["board_id"],
                },
                parent_ids=list(package["generated_task_ids"]),
                role="editorial_board",
                private_purpose=f"crossed-{provider}-editorial",
                expected={"board_id": package["board_id"]},
                validator=validate,
            )
        except Exception as exc:
            return {
                "provider": provider,
                "group_index": package["group_index"],
                "board_index": package["board_index"],
                "board_id": package["board_id"],
                "generated_task_ids": package["generated_task_ids"],
                "mapping": package["mapping"],
                "response": None,
                "operational_failure": True,
                "error": str(exc),
            }
        return {
            "provider": provider,
            "group_index": package["group_index"],
            "board_index": package["board_index"],
            "board_id": package["board_id"],
            "generated_task_ids": package["generated_task_ids"],
            "mapping": package["mapping"],
            "response": response,
            "operational_failure": False,
        }

    calls = parallel_map(
        one,
        specs,
        min(len(specs), int(runtime.config["model"]["concurrency"])),
    ) if specs else []
    task_results: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        task_results[task_id] = {}
        for provider in EVALUATOR_PROVIDERS:
            judgments = []
            failures = []
            for result in calls:
                if result["provider"] != provider or task_id not in result["generated_task_ids"]:
                    continue
                if result.get("operational_failure"):
                    failures.append(
                        {
                            "board_id": result["board_id"],
                            "error": result.get("error", "editorial call failed"),
                        }
                    )
                    continue
                opaque = next(
                    key
                    for key, source in result["mapping"].items()
                    if source == f"generated:{task_id}"
                )
                judgment = next(
                    item
                    for item in result["response"]["judgments"]
                    if item["task_id"] == opaque
                )
                judgments.append({**deepcopy(judgment), "task_id": task_id})
            if len(judgments) != 2:
                task_results[task_id][provider] = {
                    "judgments": judgments,
                    "intrinsic_quality": "unreviewed_operational_failure",
                    "portfolio_placement": "not_applicable",
                    "intrinsic_reason": "Two completed editorial boards were not available.",
                    "portfolio_reason": "Editorial placement is unavailable.",
                    "operational_failure": True,
                    "failures": failures,
                }
                continue
            intrinsic, placement, intrinsic_reason, placement_reason = (
                classify_v3_editorial_results(judgments)
            )
            task_results[task_id][provider] = {
                "judgments": judgments,
                "intrinsic_quality": intrinsic,
                "portfolio_placement": placement,
                "intrinsic_reason": intrinsic_reason,
                "portfolio_reason": placement_reason,
                "operational_failure": False,
            }
    value = {
        "schema_version": 1,
        "author_provider_hidden": True,
        "same_packages_and_opaque_ids_by_evaluator": True,
        "fixed_anchor_panel_unchanged": True,
        "board_calls_per_evaluator_per_group": 2,
        "packages": packages,
        "calls": calls,
        "task_results": task_results,
    }
    atomic_write_json(path, value)
    return value


def _call_accounting(runtime: CrossEvalRuntime) -> dict[str, Any]:
    overall = v4_call_accounting(runtime.state["call_ledger"], run_dir=runtime.run_dir)
    providers: dict[str, Any] = {}
    for provider in EVALUATOR_PROVIDERS:
        ledger = {
            key: item
            for key, item in runtime.state["call_ledger"].items()
            if item["provider"] == provider
        }
        value = v4_call_accounting(ledger, run_dir=runtime.run_dir)
        role_counts = Counter(item["role"] for item in ledger.values())
        cost = sum(
            float(item["total_cost_usd"])
            for item in ledger.values()
            if type(item.get("total_cost_usd")) in {int, float}
        )
        token_counts = Counter()
        for item in ledger.values():
            usage = item.get("usage")
            if not isinstance(usage, Mapping):
                continue
            for key in (
                "input_tokens", "output_tokens", "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                if type(usage.get(key)) is int:
                    token_counts[key] += usage[key]
        providers[provider] = {
            **value,
            "logical_calls_by_role": dict(sorted(role_counts.items())),
            "reported_total_cost_usd": round(cost, 6),
            "reported_tokens": dict(sorted(token_counts.items())),
        }
    return {
        **overall,
        "by_provider": providers,
        "attestation_calls_excluded": True,
        "attestation_is_separately_logged": True,
    }


def _recovery_chain_accounting(runtime: CrossEvalRuntime) -> dict[str, Any]:
    """Account for live attempts across predecessor and final recovery runs."""

    root = Path(runtime.config["_repository_root"])
    directories: list[Path] = []
    for raw in runtime.config["claude_cross_eval"].get("recovery_source_runs", []):
        path = Path(raw)
        directories.append(path if path.is_absolute() else root / path)
    directories.append(runtime.run_dir)
    runs = []
    provider_seconds = Counter()
    total_attempts = 0
    total_failed_attempts = 0
    for directory in directories:
        state = read_json(directory / "state.json")
        ledger = deepcopy(state.get("call_ledger", {}))
        for item in ledger.values():
            if item.get("log_dir") or not item.get("stage") or not item.get("call_id"):
                continue
            provider_root = (
                "claude_provider/claude_model_calls"
                if item.get("provider") == "claude"
                else "codex_provider/model_calls"
            )
            item["log_dir"] = str(
                directory / provider_root / str(item["stage"]) / str(item["call_id"])
            )
        overall = v4_call_accounting(ledger, run_dir=directory)
        by_provider = {}
        for provider in EVALUATOR_PROVIDERS:
            subset = {
                key: item
                for key, item in ledger.items()
                if item.get("provider") == provider
            }
            accounting = v4_call_accounting(subset, run_dir=directory)
            by_provider[provider] = accounting
            provider_seconds[provider] += accounting["active_model_seconds"]
        total_attempts += overall["recorded_model_attempts"]
        total_failed_attempts += overall["failed_or_malformed_attempts"]
        runs.append(
            {
                "run": str(directory),
                "completed": bool(state.get("completed")),
                "accounting": overall,
                "by_provider": by_provider,
            }
        )
    total_seconds = sum(provider_seconds.values())
    return {
        "recovery_run_count": len(runs),
        "runs": runs,
        "recorded_live_model_attempts": total_attempts,
        "failed_or_malformed_attempts": total_failed_attempts,
        "active_model_seconds": round(total_seconds, 3),
        "active_model_minutes": round(total_seconds / 60, 3),
        "active_model_minutes_by_provider": {
            key: round(value / 60, 3)
            for key, value in sorted(provider_seconds.items())
        },
        "cache_backed_recovery_does_not_repeat_successful_live_attempts": True,
    }


def _task_report_rows(
    records: Mapping[str, Mapping[str, Any]],
    technical: Mapping[str, Mapping[str, Any]],
    blind: Mapping[str, Mapping[str, Any]],
    boards: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    board_results = boards.get("task_results", {})
    for task_id, record in sorted(records.items()):
        author = record["private_author_provider"]
        realized = record.get("realization_status") == "realized"
        tech = technical.get(task_id, {})
        solve = blind.get(task_id, {})
        editorial = board_results.get(task_id, {})
        experimental_consensus = "not_realized"
        if realized:
            technical_operational_failure = (
                len(tech) != 2
                or any(item.get("operational_failure") for item in tech.values())
            )
            blind_operational_failure = (
                len(solve) != 2
                or any(item.get("operational_failure") for item in solve.values())
            )
            editorial_operational_failure = (
                len(editorial) != 2
                or any(item.get("operational_failure") for item in editorial.values())
            )
            serious = any(
                item.get("serious_different_output_alternative") is True
                for item in solve.values()
            )
            both_tech = len(tech) == 2 and all(item.get("passed") for item in tech.values())
            intrinsic = [item.get("intrinsic_quality") for item in editorial.values()]
            if technical_operational_failure:
                experimental_consensus = "technical_operational_review"
            elif serious:
                experimental_consensus = "ambiguity_review"
            elif not both_tech:
                experimental_consensus = "technical_review"
            elif blind_operational_failure:
                experimental_consensus = "blind_operational_review"
            elif editorial_operational_failure:
                experimental_consensus = "editorial_operational_review"
            elif intrinsic and all(item == "core_quality" for item in intrinsic):
                experimental_consensus = "cross_evaluator_core"
            elif intrinsic and all(item in {"core_quality", "reserve_quality"} for item in intrinsic):
                experimental_consensus = "cross_evaluator_core_or_reserve"
            else:
                experimental_consensus = "editorial_disagreement_or_reject"
        rows.append(
            {
                "task_id": task_id,
                "private_author_provider": author,
                "search_mode": record["search_mode"],
                "selector_global_priority": record["selector_global_priority"],
                "plan_status": record["status"],
                "realization_status": record.get("realization_status"),
                "failure_reason": record.get("failure_reason", ""),
                "task_metrics": record.get("task_metrics"),
                "palette_treatment": record.get("palette_treatment"),
                "technical": deepcopy(tech),
                "blind": deepcopy(solve),
                "editorial": deepcopy(editorial),
                "experimental_consensus": experimental_consensus,
                "task_path": (
                    f"tasks/anonymous/{task_id}.arc.json" if realized else None
                ),
                "dossier_path": (
                    f"tasks/private_dossiers/{task_id}.authoring.json" if realized else None
                ),
            }
        )
    return rows


def _judge_comparison(
    task_rows: Sequence[Mapping[str, Any]],
    historical: Mapping[str, Any],
) -> dict[str, Any]:
    realized = [item for item in task_rows if item["realization_status"] == "realized"]
    return {
        "realized_task_count": len(realized),
        "technical_pass_agreement_count": sum(
            len(item["technical"]) == 2
            and not any(
                value.get("operational_failure")
                for value in item["technical"].values()
            )
            and item["technical"]["codex"]["passed"] == item["technical"]["claude"]["passed"]
            for item in realized
        ),
        "technical_hidden_output_exact_agreement_count": sum(
            len(item["technical"]) == 2
            and not any(
                value.get("operational_failure")
                for value in item["technical"].values()
            )
            and item["technical"]["codex"]["hidden_output_match"]
            == item["technical"]["claude"]["hidden_output_match"]
            for item in realized
        ),
        "blind_primary_output_agreement_count": sum(
            len(item["blind"]) == 2
            and not any(
                value.get("operational_failure") for value in item["blind"].values()
            )
            and item["blind"]["codex"]["response"]["predicted_test_outputs"]
            == item["blind"]["claude"]["response"]["predicted_test_outputs"]
            for item in realized
        ),
        "blind_serious_alternative_agreement_count": sum(
            len(item["blind"]) == 2
            and not any(
                value.get("operational_failure") for value in item["blind"].values()
            )
            and item["blind"]["codex"]["serious_different_output_alternative"]
            == item["blind"]["claude"]["serious_different_output_alternative"]
            for item in realized
        ),
        "editorial_intrinsic_agreement_count": sum(
            len(item["editorial"]) == 2
            and not any(
                value.get("operational_failure")
                for value in item["editorial"].values()
            )
            and item["editorial"]["codex"]["intrinsic_quality"]
            == item["editorial"]["claude"]["intrinsic_quality"]
            for item in realized
        ),
        "generated_task_operational_failures": {
            stage: {
                provider: sum(
                    item[stage].get(provider, {}).get("operational_failure") is True
                    for item in realized
                )
                for provider in EVALUATOR_PROVIDERS
            }
            for stage in ("technical", "blind", "editorial")
        },
        "historical_case_count": historical.get("case_count", 0),
        "historical_serious_alternative_agreement_count": sum(
            item["serious_alternative_agreement"]
            for item in historical.get("comparisons", [])
        ),
        "historical_problem_detection": {
            provider: {
                "known_problem_count": sum(
                    item["answer_support_problem"]
                    and item["reference_certainty"] == "high"
                    for item in historical.get("comparisons", [])
                ),
                "serious_alternative_found_count": sum(
                    item["answer_support_problem"]
                    and item["reference_certainty"] == "high"
                    and item[provider]["serious_different_output_alternative"] is True
                    for item in historical.get("comparisons", [])
                ),
                "valid_case_false_alarm_count": sum(
                    not item["answer_support_problem"]
                    and item["reference_certainty"] == "high"
                    and item[provider]["serious_different_output_alternative"] is True
                    for item in historical.get("comparisons", [])
                ),
                "operational_failure_count": sum(
                    item[provider].get("operational_failure") is True
                    for item in historical.get("comparisons", [])
                ),
            }
            for provider in EVALUATOR_PROVIDERS
        },
    }


def _render_review_pack(
    runtime: CrossEvalRuntime,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    directory = runtime.run_dir / "human_review"
    directory.mkdir(parents=True, exist_ok=True)
    realized = [
        item for item in records.values() if item.get("realization_status") == "realized"
    ]
    rng = Random(int(runtime.config["seed"]) + 9901)
    order = list(realized)
    rng.shuffle(order)
    mapping = []
    images = []
    appendix = ["# V4.5 Claude cross-eval answer appendix", ""]
    for index, record in enumerate(order, 1):
        alias = f"episode-{index:02d}"
        task_id = record["candidate_id"]
        image_path = directory / f"{alias}.png"
        render_episode(
            record["complete_task"],
            title=alias,
            subtitle="Author source hidden; review the grids before the appendix",
            rule=None,
            destination=image_path,
        )
        images.append((alias, image_path))
        mapping.append(
            {
                "alias": alias,
                "task_id": task_id,
                "private_author_provider": record["private_author_provider"],
            }
        )
        appendix.extend(
            [
                f"## {alias}",
                "",
                record["realization"]["authoring_metadata"]["intended_rule"],
                "",
            ]
        )
    contact = build_contact_sheet(images, directory / "contact_sheet.png", columns=2)
    atomic_write_json(directory / "provenance_key.private.json", {"mapping": mapping})
    atomic_write_text(directory / "answer_rule_appendix.md", "\n".join(appendix))
    atomic_write_text(
        directory / "review_sheet.md",
        "# Blinded V4.5 Claude authoring review\n\n"
        "Review each rendered episode before opening the rule appendix. For each, record "
        "conceptual interest, functional composition, discoverability, visual economy, "
        "fairness, and overall preference. Author source is intentionally hidden.\n",
    )
    return {
        "task_count": len(realized),
        "contact_sheet": str(contact) if contact else None,
        "review_sheet": str(directory / "review_sheet.md"),
        "answer_appendix": str(directory / "answer_rule_appendix.md"),
        "private_provenance_key": str(directory / "provenance_key.private.json"),
    }


def _authoring_metrics(task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for author in AUTHOR_PROVIDERS:
        rows = [item for item in task_rows if item["private_author_provider"] == author]
        realized = [item for item in rows if item["realization_status"] == "realized"]
        result[author] = {
            "selected": len(rows),
            "plans_ready": sum(item["plan_status"] == "plan_ready" for item in rows),
            "realized": len(realized),
            "deterministically_valid": len(realized),
            "both_technical_pass": sum(
                len(item["technical"]) == 2
                and all(value["passed"] for value in item["technical"].values())
                for item in realized
            ),
            "both_hidden_output_exact": sum(
                len(item["technical"]) == 2
                and all(value["hidden_output_match"] for value in item["technical"].values())
                for item in realized
            ),
            "no_evaluator_serious_alternative": sum(
                len(item["blind"]) == 2
                and not any(
                    value.get("operational_failure")
                    for value in item["blind"].values()
                )
                and not any(
                    value["serious_different_output_alternative"] is True
                    for value in item["blind"].values()
                )
                for item in realized
            ),
            "blind_operational_failure": sum(
                any(
                    value.get("operational_failure")
                    for value in item["blind"].values()
                )
                for item in realized
            ),
            "cross_evaluator_core": sum(
                item["experimental_consensus"] == "cross_evaluator_core"
                for item in realized
            ),
            "cross_evaluator_core_or_reserve": sum(
                item["experimental_consensus"]
                in {"cross_evaluator_core", "cross_evaluator_core_or_reserve"}
                for item in realized
            ),
            "mean_maximum_dimension": (
                round(
                    sum(item["task_metrics"]["maximum_dimension"] for item in realized)
                    / len(realized),
                    3,
                )
                if realized
                else None
            ),
        }
    return result


def _source_preference_matrix(task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for evaluator in EVALUATOR_PROVIDERS:
        result[evaluator] = {}
        for author in AUTHOR_PROVIDERS:
            rows = [
                item
                for item in task_rows
                if item["private_author_provider"] == author
                and item["realization_status"] == "realized"
                and evaluator in item["editorial"]
            ]
            counts = Counter(
                item["editorial"][evaluator]["intrinsic_quality"] for item in rows
            )
            result[evaluator][author] = {
                "reviewed": len(rows),
                "intrinsic_quality_counts": dict(sorted(counts.items())),
                "core_rate": (
                    round(counts["core_quality"] / len(rows), 6) if rows else None
                ),
            }
    return result


def _attestation_accounting(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["claude_code"]["attestation_path"])
    if not path.is_file():
        return {"present": False, "substantive_calls": 0}
    record = read_json(path)
    active_seconds = 0.0
    cost = 0.0
    for row in record.get("attestations", []):
        result_path = Path(row["call_log"]) / "result.json"
        if not result_path.is_file():
            continue
        result = read_json(result_path)
        timing = result.get("timing") or {}
        active_seconds += float(timing.get("model_execution_seconds") or 0)
        if type(result.get("total_cost_usd")) in {int, float}:
            cost += float(result["total_cost_usd"])
    return {
        "present": True,
        "substantive_calls": len(record.get("attestations", [])),
        "efforts": [item["effort"] for item in record.get("attestations", [])],
        "resolved_models": sorted(
            {
                model
                for item in record.get("attestations", [])
                for model in item.get("resolved_models", [])
            }
        ),
        "active_model_seconds": round(active_seconds, 3),
        "reported_total_cost_usd": round(cost, 6),
        "artifact": str(path),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    authoring = report["authoring_results"]
    calls = report["call_accounting"]
    recovery = report["recovery_chain_accounting"]
    lines = [
        "# ARCGEN V4.5 Claude Opus 5 cross-evaluation experiment",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "This is an isolated, non-gating experiment. It did not mutate the good-task "
        "bank, reservoir, reservations, rejection memory, feature state, or frozen V4.5 "
        "baseline.",
        "",
        "## Design",
        "",
        "GPT-5.6 Sol and Claude Opus received matched open, concept-led, and "
        "composition-led idea prompts. A provenance-blind Sol selector gated development; "
        "Claude ran the same selector as a shadow. Each provider planned and realized its "
        "own selected kernels with identical prompts and role-matched efforts. Every realized "
        "task was then independently checked and blindly solved by both providers and judged "
        "by two fixed-anchor boards from each provider.",
        "",
        "## Funnel by author provider",
        "",
        "| Author | Selected | Plans ready | Realized | Both technical pass | Cross-evaluator core | Core or reserve |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in AUTHOR_PROVIDERS:
        row = authoring[provider]
        lines.append(
            f"| {provider} | {row['selected']} | {row['plans_ready']} | {row['realized']} | "
            f"{row['both_technical_pass']} | {row['cross_evaluator_core']} | "
            f"{row['cross_evaluator_core_or_reserve']} |"
        )
    lines.extend(
        [
            "",
            "## Judge comparison",
            "",
            f"- Realized tasks jointly reviewed: {report['judge_comparison']['realized_task_count']}",
            f"- Technical pass agreements: {report['judge_comparison']['technical_pass_agreement_count']}",
            f"- Blind primary-output agreements: {report['judge_comparison']['blind_primary_output_agreement_count']}",
            f"- Blind serious-alternative agreements: {report['judge_comparison']['blind_serious_alternative_agreement_count']}",
            f"- Editorial intrinsic-status agreements: {report['judge_comparison']['editorial_intrinsic_agreement_count']}",
            "",
            "## Calls and time",
            "",
            f"- Experimental logical calls: {calls['substantive_logical_calls']}",
            f"- Active model minutes: {calls['active_model_minutes']}",
            f"- Measured peak concurrency: {calls['measured_peak_concurrent_calls']}",
            f"- Full recovery-chain active model minutes: {recovery['active_model_minutes']}",
            f"- Full recovery-chain failed or malformed attempts: {recovery['failed_or_malformed_attempts']}",
        ]
    )
    for provider in EVALUATOR_PROVIDERS:
        row = calls["by_provider"][provider]
        lines.append(
            f"- {provider}: {row['substantive_logical_calls']} calls, "
            f"{row['active_model_minutes']} active model minutes, "
            f"reported cost ${row['reported_total_cost_usd']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Per-task outcomes",
            "",
            "| Task | Private author | Mode | Realization | Experimental consensus |",
            "|---|---|---|---|---|",
        ]
    )
    for task in report["tasks"]:
        lines.append(
            f"| {task['task_id']} | {task['private_author_provider']} | "
            f"{task['search_mode']} | {task['realization_status']} | "
            f"{task['experimental_consensus']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The machine report exposes attribution only after all downstream prompts were "
            "completed. The human review pack remains blinded. No task is production-accepted "
            "or committed by this experiment, and a single slate is not sufficient to establish "
            "model superiority.",
            "",
            f"Review pack: `{report['human_review_pack']['contact_sheet']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    runtime: CrossEvalRuntime,
    records: Mapping[str, Mapping[str, Any]],
    technical: Mapping[str, Mapping[str, Any]],
    blind: Mapping[str, Mapping[str, Any]],
    boards: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> dict[str, Any]:
    task_rows = _task_report_rows(records, technical, blind, boards)
    review = _render_review_pack(runtime, records)
    protected_after = _protected_state(runtime.config)
    _assert_unchanged(runtime.state["protected_state_before"], protected_after)
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "pipeline_version": VERSION,
        "run_id": runtime.config["run"]["id"],
        "completed": True,
        "baseline": verify_v45_integrated_baseline(
            runtime.config["_repository_root"],
            runtime.config["_v4_5_baseline_manifest"],
        ),
        "design": {
            "idea_search_modes": runtime.config["claude_cross_eval"]["normal_search_modes"],
            "same_idea_prompts_and_calibrations": True,
            "codex_selector_gating": True,
            "claude_selector_shadow": True,
            "same_authoring_prompts_and_role_efforts": True,
            "crossed_technical_blind_and_editorial_evaluation": True,
            "author_source_hidden_downstream": True,
            "repairs_enabled": False,
        },
        "authoring_results": _authoring_metrics(task_rows),
        "judge_comparison": _judge_comparison(task_rows, historical),
        "source_preference_matrix": _source_preference_matrix(task_rows),
        "historical_blind_calibration": historical,
        "tasks": task_rows,
        "call_accounting": _call_accounting(runtime),
        "recovery_chain_accounting": _recovery_chain_accounting(runtime),
        "attestation_accounting": _attestation_accounting(runtime.config),
        "human_review_pack": review,
        "production_promotion_occurred": False,
        "automatic_task_commit_occurred": False,
        "production_state_mutated": False,
        "protected_state_before": runtime.state["protected_state_before"],
        "protected_state_after": protected_after,
        "recommendation": "await_blinded_human_review",
        "limitations": [
            "One matched authoring slate is an exploratory sample, not a stable model ranking.",
            "The source-blind selector is Sol-gating by design; Claude selection is shadow-only.",
            "Repairs are disabled to avoid provider-specific intervention confounds.",
            "Historical blind cases test ambiguity behavior but do not provide author-aware technical plans.",
        ],
    }
    atomic_write_json(runtime.run_dir / "report.json", report)
    atomic_write_text(runtime.run_dir / "report.md", _markdown(report))
    runtime.state["completed"] = True
    runtime.state["completed_at_epoch_seconds"] = round(time.time(), 3)
    runtime.state["protected_state_after"] = protected_after
    runtime.state["report_path"] = str(runtime.run_dir / "report.json")
    runtime.save()
    return report


def run_experiment(
    config: dict[str, Any],
    *,
    authorize_model_calls: bool,
) -> dict[str, Any]:
    if not authorize_model_calls and not config["run"]["mock"]:
        raise ClaudeCrossEvalError(
            "real model calls are not authorized; pass --authorize-model-calls explicitly"
        )
    design = freeze_experiment(config)
    runtime = CrossEvalRuntime(config, require_attestation=not config["run"]["mock"])
    report_path = runtime.run_dir / "report.json"
    if runtime.state.get("completed") and report_path.is_file():
        return read_json(report_path)
    kernels = _generate_kernels(runtime, design)
    selected = _select_kernels(runtime, kernels)
    plans = _plan_selected(runtime, selected)
    records = _realize_plans(runtime, plans)
    technical = _crossed_technical(runtime, records)
    blind = _crossed_blind(runtime, records)
    historical = _historical_blind_calibration(runtime)
    boards = _crossed_boards(runtime, records)
    return build_report(runtime, records, technical, blind, boards, historical)


def attest_model(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the explicitly authorized three-effort Opus identity check."""

    environment = validate_claude_environment(config)
    if environment.get("mock"):
        raise ClaudeCrossEvalError("live model attestation is unavailable in mock mode")
    system_text, _, _ = load_prompt(Path(config["paths"]["claude_system_prompt"]))
    run_dir = (
        Path(config["paths"]["runs"])
        / config["run"]["id"]
        / "claude_attestation_calls"
    )
    return attest_claude_role_efforts(
        settings=config["claude_code"],
        run_dir=run_dir,
        cache_root=Path(config["paths"]["cache"]),
        system_prompt=system_text,
        efforts=config["claude_code"]["required_attestation_efforts"],
        destination=Path(config["claude_code"]["attestation_path"]),
    )


def _load_report(run: str | Path) -> dict[str, Any]:
    path = Path(run)
    if path.is_dir():
        path = path / "report.json"
    value = read_json(path)
    if value.get("pipeline") != PIPELINE:
        raise ClaudeCrossEvalError("run is not a Claude cross-eval experiment")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m arcgen.lib.v45_claude_cross_eval",
        description="Isolated V4.5 Claude Opus authoring/cross-evaluation study",
    )
    parser.add_argument(
        "command", choices=("validate", "freeze", "attest-model", "run", "report")
    )
    parser.add_argument(
        "--config", default="config/v4_5_exp_claude_opus_cross_eval.yaml"
    )
    parser.add_argument("--run")
    parser.add_argument("--authorize-model-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            if not args.run:
                raise ClaudeCrossEvalError("--run is required for report")
            report = _load_report(args.run)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        config = load_v45_claude_config(args.config)
        if args.command == "validate":
            result = {
                "pipeline": PIPELINE,
                "baseline": config["_v4_5_baseline_verification"],
                "claude_environment": validate_claude_environment(config),
                "model_calls_made": 0,
                "ready_for_freeze": True,
                "ready_for_real_run": Path(
                    config["claude_code"]["attestation_path"]
                ).is_file(),
            }
        elif args.command == "freeze":
            result = freeze_experiment(config)
        elif args.command == "attest-model":
            if not args.authorize_model_calls:
                raise ClaudeCrossEvalError(
                    "attest-model makes three tiny calls; pass --authorize-model-calls"
                )
            result = attest_model(config)
        elif args.command == "run":
            result = run_experiment(
                config, authorize_model_calls=args.authorize_model_calls
            )
        else:  # pragma: no cover - argparse guards this
            raise ClaudeCrossEvalError("unsupported command")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ClaudeCrossEvalError",
    "CrossEvalRuntime",
    "CrossPromptStore",
    "OUTPUT_SCHEMAS",
    "attest_model",
    "build_report",
    "freeze_experiment",
    "main",
    "parse_cross_stage",
    "run_experiment",
]
