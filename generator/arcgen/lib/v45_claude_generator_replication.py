"""Source-blind Claude Opus 5 versus Sol generator-only replication.

The experiment keeps the frozen V4.5 production baseline and the first Claude
cross-evaluation run untouched.  Claude is used only for matched idea-search
calls.  A source-blind Sol selector gates development and the complete
downstream path is the ordinary Sol control, removing planner/realizer source
as a confound.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from string import ascii_lowercase, digits
from typing import Any, Callable, Mapping, Sequence
import argparse
import json
import sys
import threading
import time

from .claude_code import (
    ClaudeCodeCallResult,
    ClaudeCodeCapabilities,
    ClaudeCodeInvoker,
    inspect_claude_code,
    validate_claude_attestation,
)
from .config import ConfigError
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
from .v41_experiments import _portfolio_digest, _task_metrics
from .v41_integration import transform_realization_palette
from .v43_episode_joint import _reference_packages
from .v431_episode_joint_quality import choose_global_development
from .v45_claude_config import (
    load_v45_claude_config,
    validate_claude_environment,
    verify_v45_integrated_baseline,
)
from .v45_claude_cross_eval import (
    OUTPUT_SCHEMAS,
    PROMPTS,
    SOURCE_FORBIDDEN,
    CrossEvalRuntime,
    _assert_unchanged,
    _board_package,
    _json_after_marker,
    _opaque_id,
    _palette_assignments,
    _protected_state,
    _public_task,
    _technical_evaluation,
    parse_cross_stage,
)
from .validation import validate_arc_task


PIPELINE = "arcgen-v4.5-exp-claude-opus-5-generator-replication"
VERSION = "arcgen-v4.5-claude-opus-5-generator-replication-1.0"
AUTHOR_PROVIDERS = ("codex", "claude")
DOWNSTREAM_PROVIDER = "codex"


class GeneratorReplicationError(RuntimeError):
    """Raised when the isolated generator study violates its frozen design."""


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"generator replication config must be a mapping: {path}")
    return value


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_generator_replication_config(path: str | Path) -> dict[str, Any]:
    """Load a strict isolated config by extending the validated Claude study."""

    source = Path(path).resolve()
    raw = _read_yaml_or_json(source)
    if raw.get("pipeline") != PIPELINE or str(raw.get("version")) != "1.0":
        raise ConfigError(f"generator replication must use {PIPELINE} version 1.0")
    parent_path = Path(str(raw.get("parent_cross_eval_config", "")))
    if not parent_path.is_absolute():
        parent_path = (source.parent / parent_path).resolve()
    config = load_v45_claude_config(parent_path)
    settings = raw.get("generator_replication")
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        raise ConfigError("generator_replication must be explicitly enabled")
    for field in (
        "gating",
        "auto_promote",
        "production_state_mutation_allowed",
        "good_task_bank_mutation_allowed",
        "reservoir_mutation_allowed",
        "reservation_mutation_allowed",
        "rejection_memory_mutation_allowed",
        "automatic_task_commit",
    ):
        if settings.get(field) is not False:
            raise ConfigError(f"generator_replication.{field} must be false")
    exact = {
        "independent_slates": 3,
        "normal_search_modes": ["open", "concept_led", "composition_led"],
        "calls_per_mode_per_author_per_slate": 1,
        "ideas_per_call": 4,
        "candidates_per_author": 36,
        "maximum_development_per_slate": 4,
        "maximum_total_development": 12,
        "codex_selector_gating": True,
        "source_balancing_enabled": False,
        "downstream_provider": DOWNSTREAM_PROVIDER,
        "downstream_model_control_unchanged": True,
        "arm_labels_hidden_downstream": True,
        "same_prompts_and_schemas": True,
        "same_calibration_descriptions_within_pairs": True,
        "repairs_enabled": False,
        "editorial_boards_per_group": 2,
        "editorial_generated_group_size": 6,
    }
    for field, expected in exact.items():
        if settings.get(field) != expected:
            raise ConfigError(
                f"generator_replication.{field} must be {expected!r}"
            )
    for field in ("generation_concurrency", "downstream_concurrency"):
        if type(settings.get(field)) is not int or settings[field] < 1:
            raise ConfigError(f"generator_replication.{field} must be positive")
    run = raw.get("run")
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str):
        raise ConfigError("run.id is required")
    if run.get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("the generator replication cannot commit tasks")
    if run.get("mock") not in {True, False}:
        raise ConfigError("run.mock must be boolean")
    seed = raw.get("seed")
    if type(seed) is not int:
        raise ConfigError("seed must be an integer")

    config = deepcopy(config)
    config.pop("claude_cross_eval", None)
    config["pipeline"] = PIPELINE
    config["version"] = "1.0"
    config["seed"] = seed
    config["generator_replication"] = deepcopy(dict(settings))
    config["run"] = {**deepcopy(config["run"]), **deepcopy(dict(run))}
    config["claude_code"]["mock"] = bool(config["run"]["mock"])
    config["experimental"] = {
        "enabled": True,
        "kind": "claude_opus_5_generator_only_replication",
        "gating": False,
        "auto_promote": False,
        "arm_id": "private",
    }
    config["_config_path"] = str(source)
    config["_parent_cross_eval_config"] = str(parent_path)
    return config


def snapshot_generator_replication_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


class ReplicationPromptStore:
    """Freeze exact V4.5 prompt templates without changing the prior study."""

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
                raise GeneratorReplicationError(f"prompt changed on resume: {filename}")
            if not target.exists():
                atomic_write_text(target, text)
            manifest["templates"][name] = {
                "filename": filename,
                "stage": stage,
                "version": version,
                "sha256": digest,
                "placeholders": sorted(PLACEHOLDER_RE.findall(text)),
            }
        target = run_dir / "prompts" / "manifest.json"
        if target.exists() and read_json(target) != manifest:
            raise GeneratorReplicationError("prompt manifest changed on resume")
        if not target.exists():
            atomic_write_json(target, manifest)

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
            raise GeneratorReplicationError(
                f"prompt bindings differ for {'+'.join(names)}: "
                f"missing={sorted(required-set(values))}, "
                f"extra={sorted(set(values)-required)}"
            )
        prompt = render_prompt(text, dict(values))
        return (
            prompt,
            "+".join(item["version"] for item in entries),
            content_hash(
                {"templates": [item["hash"] for item in entries], "text": text}
            ),
            entries[-1]["stage"],
        )


def _implementation_manifest(root: Path, run_dir: Path) -> dict[str, Any]:
    relatives = (
        "arcgen.lib/v45_claude_generator_replication.py",
        "config/v4_5_exp_claude_opus_generator_replication.yaml",
        "experiments/v4_5_claude_opus_generator_replication_plan.md",
        "arcgen.lib/claude_code.py",
        "arcgen.lib/v45_claude_cross_eval.py",
        "prompts/v4_5_claude_code_system.md",
    )
    value = {
        "schema_version": 1,
        "pipeline_version": VERSION,
        "branch_source": "arcgen-v4.5-integrated-authoring-20260808",
        "files": {relative: _sha(root / relative) for relative in relatives},
    }
    path = run_dir / "implementation_manifest.json"
    if path.exists() and read_json(path) != value:
        raise GeneratorReplicationError("implementation changed during resume")
    if not path.exists():
        atomic_write_json(path, value)
    return value


class GeneratorReplicationRuntime(CrossEvalRuntime):
    """Use the tested dual-provider call method with isolated experiment state."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        require_attestation: bool,
        codex_mock_responses: Mapping[str, Any] | None = None,
        claude_mock_responses: Mapping[str, Any] | None = None,
    ) -> None:
        if config.get("pipeline") != PIPELINE:
            raise GeneratorReplicationError("explicit generator replication config required")
        self.config = config
        self.run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.state_path = self.run_dir / "state.json"
        root = Path(config["_repository_root"])
        _implementation_manifest(root, self.run_dir)
        snapshot = snapshot_generator_replication_config(config)
        snapshot_path = self.run_dir / "config.snapshot.json"
        if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
            raise GeneratorReplicationError("config changed during resume")
        if not snapshot_path.exists():
            atomic_write_json(snapshot_path, snapshot)
        self.prompts = ReplicationPromptStore(Path(config["paths"]["prompts"]), self.run_dir)
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
                    required_efforts=["max"],
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


def _calibration_slates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = Path(config["paths"]["frozen_generation_source_run"])
    paths = sorted(
        source.glob(
            "selection_graph/v4_reservoir_supply/*/model_calls/"
            "v4_idea_search/*/prompt.md"
        )
    )
    by_topup: dict[str, dict[str, Path]] = defaultdict(dict)
    mode_needles = {
        "open": "open-supply",
        "concept_led": "concept-led-supply",
        "composition_led": "composition-led-supply",
    }
    for path in paths:
        topup = path.parents[3].name
        for mode, needle in mode_needles.items():
            if needle in path.parent.name:
                by_topup[topup][mode] = path
                break
    complete = [
        (name, rows)
        for name, rows in sorted(by_topup.items())
        if set(rows) == set(mode_needles)
    ]
    needed = int(config["generator_replication"]["independent_slates"])
    if len(complete) < needed:
        raise GeneratorReplicationError(
            f"only {len(complete)} complete frozen calibration sets; need {needed}"
        )
    result = []
    for slate_index, (name, rows) in enumerate(complete[:needed], 1):
        modes = {}
        for mode in mode_needles:
            prompt_path = rows[mode]
            text = prompt_path.read_text(encoding="utf-8")
            modes[mode] = {
                "descriptions": _json_after_marker(text, "Calibration descriptions:"),
                "source_prompt": str(prompt_path),
                "source_prompt_sha256": _sha(prompt_path),
            }
        descriptions = [item for row in modes.values() for item in row["descriptions"]]
        if len(descriptions) != len(set(descriptions)):
            raise GeneratorReplicationError(
                f"calibration descriptions repeat inside slate {slate_index}"
            )
        result.append(
            {"slate_index": slate_index, "source_topup": name, "modes": modes}
        )
    return result


def freeze_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "frozen_experiment_manifest.private.json"
    if path.exists():
        return read_json(path)
    pairs = []
    private_mapping: dict[str, str] = {}
    for slate in _calibration_slates(config):
        slate_index = int(slate["slate_index"])
        for mode_index, mode in enumerate(
            config["generator_replication"]["normal_search_modes"], 1
        ):
            shared_ids = [
                "q" + content_hash(
                    {
                        "seed": config["seed"],
                        "slate": slate_index,
                        "mode": mode,
                        "idea": index,
                    }
                )[:13]
                for index in range(1, 5)
            ]
            provider_ids: dict[str, list[str]] = {}
            for provider in AUTHOR_PROVIDERS:
                ids = [
                    "k" + content_hash(
                        {
                            "seed": config["seed"],
                            "slate": slate_index,
                            "provider": provider,
                            "mode": mode,
                            "idea": index,
                        }
                    )[:13]
                    for index in range(1, 5)
                ]
                provider_ids[provider] = ids
                private_mapping.update({candidate_id: provider for candidate_id in ids})
            calibration = slate["modes"][mode]
            pairs.append(
                {
                    "slate_index": slate_index,
                    "mode": mode,
                    "mode_index": mode_index,
                    "shared_call_id": (
                        f"slate-{slate_index:02d}-paired-{mode.replace('_', '-')}-search"
                    ),
                    "shared_candidate_ids": shared_ids,
                    "provider_candidate_ids": provider_ids,
                    "calibration_descriptions": calibration["descriptions"],
                    "calibration_source_prompt": calibration["source_prompt"],
                    "calibration_source_prompt_sha256": calibration[
                        "source_prompt_sha256"
                    ],
                    "calibration_source_topup": slate["source_topup"],
                }
            )
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "pipeline_version": VERSION,
        "seed": config["seed"],
        "v4_5_baseline": deepcopy(config["_v4_5_baseline_verification"]),
        "protected_state_before": _protected_state(config),
        "idea_pairs": pairs,
        "private_candidate_author_mapping": private_mapping,
        "independent_slates": 3,
        "same_rendered_idea_prompt_within_each_pair": True,
        "same_calibration_descriptions_within_each_pair": True,
        "same_output_schema": True,
        "same_idea_effort": True,
        "source_blind_sol_selector": True,
        "all_downstream_roles_use_sol_control": True,
        "source_balancing_enabled": False,
        "repairs_enabled": False,
        "automatic_task_commit": False,
        "production_state_mutation_allowed": False,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        run_dir / "frozen_experiment_manifest.public.json",
        {
            key: deepcopy(item)
            for key, item in value.items()
            if key != "private_candidate_author_mapping"
        },
    )
    return value


def _prompt_name_for_mode(mode: str) -> str:
    return {
        "open": "idea_open",
        "concept_led": "idea_concept",
        "composition_led": "idea_composition",
    }[mode]


def generate_kernels(
    runtime: GeneratorReplicationRuntime,
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

        def validate(value: dict[str, Any]) -> None:
            if [item["candidate_id"] for item in value["ideas"]] != shared_ids:
                raise GeneratorReplicationError("paired idea response changed shared IDs")

        response = runtime.call(
            provider=provider,
            prompt_name=_prompt_name_for_mode(str(pair["mode"])),
            call_id=f"{provider}-{pair['shared_call_id']}",
            values={
                "CALIBRATION_DESCRIPTIONS": list(pair["calibration_descriptions"]),
                "SEARCH_CALL_ID": pair["shared_call_id"],
                "CANDIDATE_IDS": shared_ids,
            },
            parent_ids=[],
            role="idea_search",
            private_purpose=f"private-{provider}-idea-source",
            expected={
                "search_call_id": pair["shared_call_id"],
                "search_mode": pair["mode"],
            },
            validator=validate,
        )
        mapped_ids = list(pair["provider_candidate_ids"][provider])
        ledger_key = (
            f"{provider}/private-{provider}-idea-source/v4_idea_search/"
            f"{provider}-{pair['shared_call_id']}"
        )
        rows = []
        for index, item in enumerate(response["ideas"]):
            kernel = deepcopy(item)
            kernel["candidate_id"] = mapped_ids[index]
            rows.append(
                {
                    "candidate_id": mapped_ids[index],
                    "private_kernel_author_provider": provider,
                    "slate_index": pair["slate_index"],
                    "search_mode": pair["mode"],
                    "kernel": kernel,
                    "shared_pair_call_id": pair["shared_call_id"],
                    "shared_prompt_hash": runtime.state["call_ledger"][ledger_key][
                        "prompt_hash"
                    ],
                }
            )
        return rows

    groups = parallel_map(
        generate,
        specs,
        min(
            len(specs),
            int(runtime.config["generator_replication"]["generation_concurrency"]),
        ),
    )
    kernels = [item for group in groups for item in group]
    expected = 3 * 3 * 2 * 4
    if len(kernels) != expected or len({item["candidate_id"] for item in kernels}) != expected:
        raise GeneratorReplicationError("matched generation did not yield 72 unique kernels")
    prompt_checks = []
    for pair in design["idea_pairs"]:
        hashes = {
            provider: next(
                item["shared_prompt_hash"]
                for item in kernels
                if item["private_kernel_author_provider"] == provider
                and item["shared_pair_call_id"] == pair["shared_call_id"]
            )
            for provider in AUTHOR_PROVIDERS
        }
        if len(set(hashes.values())) != 1:
            raise GeneratorReplicationError("paired providers received different idea prompts")
        prompt_checks.append(
            {
                "slate_index": pair["slate_index"],
                "mode": pair["mode"],
                "prompt_hashes": hashes,
                "exact_match": True,
            }
        )
    value = {
        "schema_version": 1,
        "kernel_count": len(kernels),
        "counts_by_author": dict(
            Counter(item["private_kernel_author_provider"] for item in kernels)
        ),
        "counts_by_mode": dict(Counter(item["search_mode"] for item in kernels)),
        "counts_by_slate": {
            str(index): sum(item["slate_index"] == index for item in kernels)
            for index in range(1, 4)
        },
        "paired_prompt_checks": prompt_checks,
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
                    "slate_index": item["slate_index"],
                    "search_mode": item["search_mode"],
                    "kernel": item["kernel"],
                }
                for item in kernels
            ],
        },
    )
    return kernels


def select_kernels(
    runtime: GeneratorReplicationRuntime,
    kernels: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = runtime.run_dir / "selected_kernel_manifest.private.json"
    if path.exists():
        value = read_json(path)
        return value["selected"], value["slates"]
    prepared = []
    for slate_index in range(1, 4):
        rows = [deepcopy(dict(item)) for item in kernels if item["slate_index"] == slate_index]
        Random(int(runtime.config["seed"]) + slate_index * 45191).shuffle(rows)
        packages, accepted, reservations, rejected = _reference_packages(
            runtime.config, rows
        )
        prepared.append(
            {
                "slate_index": slate_index,
                "rows": rows,
                "packages": packages,
                "accepted": accepted,
                "reservations": reservations,
                "rejected": rejected,
            }
        )

    def select_one(spec: Mapping[str, Any]) -> dict[str, Any]:
        slate_index = int(spec["slate_index"])
        expected_ids = {item["candidate_id"] for item in spec["rows"]}
        selector_id = f"source-blind-sol-selector-{slate_index:02d}"

        def validate(value: dict[str, Any]) -> None:
            assessed = {item["candidate_id"] for item in value["assessments"]}
            if assessed != expected_ids or set(value["quality_order"]) != expected_ids:
                raise GeneratorReplicationError("selector did not cover frozen slate")

        response = runtime.call(
            provider="codex",
            prompt_name="selector",
            call_id=selector_id,
            values={
                "CANDIDATES_WITH_REFERENCES": spec["packages"],
                "ACCEPTED_BANK_DIGEST": spec["accepted"],
                "IN_FLIGHT_RESERVATIONS": spec["reservations"],
                "REJECTED_MEMORY_DIGEST": spec["rejected"],
                "SELECTOR_ID": selector_id,
            },
            parent_ids=sorted(expected_ids),
            role="batch_selector",
            private_purpose="gating-source-blind-selection",
            expected={"selector_id": selector_id},
            validator=validate,
        )
        chosen, exclusions = choose_global_development(
            response,
            spec["rows"],
            maximum=int(
                runtime.config["generator_replication"][
                    "maximum_development_per_slate"
                ]
            ),
        )
        return {
            "slate_index": slate_index,
            "selector_id": selector_id,
            "candidate_count": len(spec["rows"]),
            "response": response,
            "selected": chosen,
            "exclusions": exclusions,
            "source_labels_visible_to_selector": False,
            "source_balancing_applied": False,
        }

    slates = parallel_map(
        select_one,
        prepared,
        min(len(prepared), int(runtime.config["model"]["concurrency"])),
    )
    selected = []
    for slate in sorted(slates, key=lambda item: item["slate_index"]):
        for row in slate["selected"]:
            row = deepcopy(row)
            row["selector_global_priority"] = len(selected) + 1
            selected.append(row)
    maximum = int(runtime.config["generator_replication"]["maximum_total_development"])
    if len(selected) > maximum:
        raise GeneratorReplicationError("selector exceeded global development cap")
    value = {
        "schema_version": 1,
        "selected": selected,
        "selected_count": len(selected),
        "selected_by_private_source": dict(
            Counter(item["private_kernel_author_provider"] for item in selected)
        ),
        "slates": slates,
        "source_labels_visible_to_selectors": False,
        "source_balancing_enabled": False,
        "hold_or_reject_promoted_for_balance": False,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        runtime.run_dir / "selected_kernel_manifest.blinded.json",
        {
            "schema_version": 1,
            "source_labels_removed": True,
            "selected": [
                {
                    key: deepcopy(item)
                    for key, item in row.items()
                    if key
                    not in {
                        "private_kernel_author_provider",
                        "shared_pair_call_id",
                        "shared_prompt_hash",
                    }
                }
                for row in selected
            ],
        },
    )
    return selected, slates


def plan_selected(
    runtime: GeneratorReplicationRuntime,
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    path = runtime.run_dir / "planning_manifest.private.json"
    if path.exists():
        return {item["candidate_id"]: item for item in read_json(path)["records"]}

    def plan_one(source: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(source["candidate_id"])
        brief = deepcopy(source["editorial_brief"])
        brief["optional_development_suggestion"] = None
        base = {
            "candidate_id": task_id,
            "private_kernel_author_provider": source[
                "private_kernel_author_provider"
            ],
            "downstream_provider": DOWNSTREAM_PROVIDER,
            "slate_index": int(source["slate_index"]),
            "search_mode": source["search_mode"],
            "kernel": deepcopy(source["kernel"]),
            "editorial_brief": brief,
            "selector_global_priority": source["selector_global_priority"],
        }
        try:
            response = runtime.call(
                provider="codex",
                prompt_name="concrete_planner",
                composed_with=("quality_plan",),
                call_id=f"{task_id}-sol-control-plan",
                values={
                    "CANDIDATE": deepcopy(source["kernel"]),
                    "EDITORIAL_BRIEF": brief,
                    "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(brief["route"]),
                    "CANDIDATE_ID": task_id,
                },
                parent_ids=[task_id],
                role="concrete_planner",
                private_purpose="sol-control-downstream",
                expected={"candidate_id": task_id},
            )
        except Exception as exc:
            return {
                **base,
                "plan": None,
                "status": "plan_operational_failure",
                "abort_reason": "",
                "failure_reason": str(exc),
            }
        return {
            **base,
            "plan": response,
            "status": "plan_ready" if response.get("status") == "ready" else "plan_aborted",
            "abort_reason": response.get("abort_reason", ""),
        }

    records = parallel_map(
        plan_one,
        list(selected),
        min(
            len(selected),
            int(runtime.config["generator_replication"]["downstream_concurrency"]),
        ),
    ) if selected else []
    directory = runtime.run_dir / "plans"
    directory.mkdir(parents=True, exist_ok=True)
    for item in records:
        atomic_write_json(directory / f"{item['candidate_id']}.json", item)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "all_planners_use_sol_control": True,
            "records": records,
            "funnel": {
                "selected": len(selected),
                "ready": sum(item["status"] == "plan_ready" for item in records),
                "aborted": sum(item["status"] == "plan_aborted" for item in records),
                "operational_failure": sum(
                    item["status"] == "plan_operational_failure" for item in records
                ),
            },
        },
    )
    return {item["candidate_id"]: item for item in records}


def realize_plans(
    runtime: GeneratorReplicationRuntime,
    plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    path = runtime.run_dir / "realization_manifest.private.json"
    if path.exists():
        return {item["candidate_id"]: item for item in read_json(path)["records"]}
    assignments = _palette_assignments(runtime, list(plans.values()))

    def realize_one(record: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(record["candidate_id"])
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
                provider="codex",
                prompt_name="task_realizer",
                call_id=f"{task_id}-sol-control-realization",
                values={"CONCRETE_PLAN": plan, "CANDIDATE_ID": task_id},
                parent_ids=[task_id],
                role="task_realizer",
                private_purpose="sol-control-downstream",
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
                    "failure_reason": final_response.get(
                        "refusal_reason", "realizer refused"
                    ),
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
                *(
                    diagnostics.get("errors", [])
                    if diagnostics and not diagnostics["hard_valid"]
                    else []
                ),
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
                "technical_diagnostics": diagnose_public_arc_task_v4(
                    final_response["task"]
                ),
                "task_metrics": _task_metrics(full),
                "realization_status": "realized",
            }
        except Exception as exc:
            return {
                **deepcopy(record),
                "realization_status": "operational_failure",
                "failure_reason": str(exc),
            }

    records = parallel_map(
        realize_one,
        list(plans.values()),
        min(
            len(plans),
            int(runtime.config["generator_replication"]["downstream_concurrency"]),
        ),
    ) if plans else []
    public_dir = runtime.run_dir / "tasks" / "anonymous"
    private_dir = runtime.run_dir / "tasks" / "private_dossiers"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        if record["realization_status"] != "realized":
            continue
        task_id = record["candidate_id"]
        atomic_write_json(public_dir / f"{task_id}.arc.json", record["complete_task"])
        atomic_write_json(private_dir / f"{task_id}.authoring.json", record)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "all_realizers_use_sol_control": True,
            "records": records,
            "funnel": {
                "selected": len(plans),
                "plans_ready": sum(
                    item["status"] == "plan_ready" for item in plans.values()
                ),
                "realized": sum(
                    item["realization_status"] == "realized" for item in records
                ),
                "refused": sum(
                    item["realization_status"] == "refused" for item in records
                ),
                "invalid": sum(
                    item["realization_status"] == "invalid" for item in records
                ),
                "operational_failure": sum(
                    item["realization_status"] == "operational_failure"
                    for item in records
                ),
            },
        },
    )
    return {item["candidate_id"]: item for item in records}


def technical_verify(
    runtime: GeneratorReplicationRuntime,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    path = runtime.run_dir / "technical_results.private.json"
    if path.exists():
        return read_json(path)["results"]
    specs = [
        task_id
        for task_id, record in records.items()
        if record.get("realization_status") == "realized"
    ]

    def one(task_id: str) -> tuple[str, dict[str, Any]]:
        record = records[task_id]
        opaque = _opaque_id(runtime.config["seed"], "technical-sol", task_id)
        verifier = _opaque_id(runtime.config["seed"], "verifier-sol", task_id)
        try:
            response = runtime.call(
                provider="codex",
                prompt_name="technical_verifier",
                call_id=f"sol-technical-{task_id}",
                values={
                    "INTENDED_RULE": record["realization"]["authoring_metadata"][
                        "intended_rule"
                    ],
                    "CONCRETE_PLAN": record["plan"],
                    "PUBLIC_TASK": record["realization"]["task"],
                    "DIAGNOSTICS": record["technical_diagnostics"],
                    "TASK_ID": opaque,
                    "VERIFIER_ID": verifier,
                },
                parent_ids=[task_id],
                role="technical_verifier",
                private_purpose="sol-control-downstream",
                expected={"task_id": opaque, "verifier_id": verifier},
            )
        except Exception as exc:
            return task_id, {
                "operational_failure": True,
                "error": str(exc),
                "response": None,
                "hidden_output_match": None,
                "passed": False,
                "kernel_author_provider_hidden": True,
                "hidden_output_withheld": True,
            }
        result = _technical_evaluation(record, response)
        result.update(
            {
                "operational_failure": False,
                "kernel_author_provider_hidden": True,
                "hidden_output_withheld": True,
            }
        )
        return task_id, result

    rows = parallel_map(
        one,
        specs,
        min(
            len(specs),
            int(runtime.config["generator_replication"]["downstream_concurrency"]),
        ),
    ) if specs else []
    results = dict(rows)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "all_verifiers_use_sol_high": True,
            "kernel_author_provider_hidden": True,
            "hidden_output_compared_programmatically": True,
            "results": results,
        },
    )
    return results


def blind_solve(
    runtime: GeneratorReplicationRuntime,
    records: Mapping[str, Mapping[str, Any]],
    technical: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    path = runtime.run_dir / "blind_results.private.json"
    if path.exists():
        return read_json(path)["results"]
    specs = [task_id for task_id, result in technical.items() if result.get("passed")]

    def one(task_id: str) -> tuple[str, dict[str, Any]]:
        record = records[task_id]
        public = _public_task(record["complete_task"])
        if any("output" in pair for pair in public["test"]):
            raise GeneratorReplicationError("hidden output entered blind prompt")
        opaque = _opaque_id(runtime.config["seed"], "blind-sol", task_id)
        solver = _opaque_id(runtime.config["seed"], "solver-sol", task_id)
        try:
            response = runtime.call(
                provider="codex",
                prompt_name="blind_solver",
                call_id=f"sol-blind-{task_id}",
                values={"PUBLIC_TASK": public, "TASK_ID": opaque, "SOLVER_ID": solver},
                parent_ids=[task_id],
                role="blind_solver",
                private_purpose="sol-control-downstream",
                expected={"task_id": opaque, "solver_id": solver},
            )
        except Exception as exc:
            return task_id, {
                "operational_failure": True,
                "error": str(exc),
                "response": None,
                "primary_matches_hidden": None,
                "serious_different_output_alternative": None,
                "kernel_author_provider_hidden": True,
                "author_information_withheld": True,
                "hidden_output_withheld": True,
            }
        hidden = record["realization"]["authoring_metadata"]["expected_test_outputs"]
        return task_id, {
            "operational_failure": False,
            "response": response,
            "primary_matches_hidden": response["predicted_test_outputs"] == hidden,
            "serious_different_output_alternative": serious_blind_alternative(
                response, hidden
            ),
            "kernel_author_provider_hidden": True,
            "author_information_withheld": True,
            "hidden_output_withheld": True,
        }

    rows = parallel_map(
        one,
        specs,
        min(
            len(specs),
            int(runtime.config["generator_replication"]["downstream_concurrency"]),
        ),
    ) if specs else []
    results = dict(rows)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "all_solvers_use_sol_xhigh": True,
            "kernel_author_provider_hidden": True,
            "author_information_withheld": True,
            "results": results,
        },
    )
    return results


def editorial_review(
    runtime: GeneratorReplicationRuntime,
    records: Mapping[str, Mapping[str, Any]],
    technical: Mapping[str, Mapping[str, Any]],
    blind: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    path = runtime.run_dir / "editorial_results.private.json"
    if path.exists():
        return read_json(path)
    task_ids = sorted(
        [
            task_id
            for task_id, result in blind.items()
            if technical[task_id].get("passed")
            and not result.get("operational_failure")
            and result.get("serious_different_output_alternative") is not True
        ],
        key=lambda task_id: content_hash(
            {"seed": runtime.config["seed"], "task": task_id}
        ),
    )
    group_size = int(
        runtime.config["generator_replication"]["editorial_generated_group_size"]
    )
    packages = []
    for offset in range(0, len(task_ids), group_size):
        group = task_ids[offset : offset + group_size]
        group_index = offset // group_size + 1
        for board_index in (1, 2):
            packages.append(
                _board_package(
                    runtime,
                    records,
                    group,
                    group_index=group_index,
                    board_index=board_index,
                )
            )
    portfolio = _portfolio_digest(runtime.config)

    def one(package: Mapping[str, Any]) -> dict[str, Any]:
        expected_ids = set(package["mapping"])

        def validate(value: dict[str, Any]) -> None:
            returned = [item["task_id"] for item in value["judgments"]]
            if len(returned) != len(set(returned)) or set(returned) != expected_ids:
                raise GeneratorReplicationError("board did not judge every opaque task")

        try:
            response = runtime.call(
                provider="codex",
                prompt_name="editorial_board",
                call_id=f"sol-{package['board_id']}",
                values={
                    "BLINDED_TASKS": package["package"],
                    "PORTFOLIO_DIGEST": portfolio,
                    "BOARD_ID": package["board_id"],
                },
                parent_ids=list(package["generated_task_ids"]),
                role="editorial_board",
                private_purpose="sol-control-downstream",
                expected={"board_id": package["board_id"]},
                validator=validate,
            )
        except Exception as exc:
            return {
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
        packages,
        min(len(packages), int(runtime.config["model"]["concurrency"])),
    ) if packages else []
    task_results: dict[str, Any] = {}
    for task_id in task_ids:
        judgments = []
        failures = []
        for result in calls:
            if task_id not in result["generated_task_ids"]:
                continue
            if result["operational_failure"]:
                failures.append(
                    {"board_id": result["board_id"], "error": result["error"]}
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
            task_results[task_id] = {
                "judgments": judgments,
                "intrinsic_quality": "unreviewed_operational_failure",
                "portfolio_placement": "not_applicable",
                "intrinsic_reason": "Two completed Sol boards were unavailable.",
                "portfolio_reason": "Editorial placement is unavailable.",
                "operational_failure": True,
                "failures": failures,
            }
            continue
        intrinsic, placement, intrinsic_reason, placement_reason = (
            classify_v3_editorial_results(judgments)
        )
        task_results[task_id] = {
            "judgments": judgments,
            "intrinsic_quality": intrinsic,
            "portfolio_placement": placement,
            "intrinsic_reason": intrinsic_reason,
            "portfolio_reason": placement_reason,
            "operational_failure": False,
        }
    value = {
        "schema_version": 1,
        "kernel_author_provider_hidden": True,
        "fixed_anchor_panel_unchanged": True,
        "boards_use_sol_max": True,
        "generated_group_size": group_size,
        "packages": packages,
        "calls": calls,
        "task_results": task_results,
    }
    atomic_write_json(path, value)
    return value


def _task_rows(
    records: Mapping[str, Mapping[str, Any]],
    technical: Mapping[str, Mapping[str, Any]],
    blind: Mapping[str, Mapping[str, Any]],
    boards: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results = boards.get("task_results", {})
    rows = []
    for task_id, record in sorted(
        records.items(),
        key=lambda item: (
            item[1].get("slate_index", 0),
            item[1].get("selector_global_priority", 0),
        ),
    ):
        realized = record.get("realization_status") == "realized"
        tech = technical.get(task_id)
        solve = blind.get(task_id)
        editorial = results.get(task_id)
        if not realized:
            disposition = record.get("realization_status", record.get("status"))
        elif tech is None or tech.get("operational_failure"):
            disposition = "technical_operational_review"
        elif not tech.get("passed"):
            disposition = "technical_reject"
        elif solve is None or solve.get("operational_failure"):
            disposition = "blind_operational_review"
        elif solve.get("serious_different_output_alternative") is True:
            disposition = "ambiguous"
        elif editorial is None or editorial.get("operational_failure"):
            disposition = "editorial_operational_review"
        else:
            disposition = editorial["intrinsic_quality"]
        rows.append(
            {
                "task_id": task_id,
                "private_kernel_author_provider": record[
                    "private_kernel_author_provider"
                ],
                "downstream_provider": DOWNSTREAM_PROVIDER,
                "slate_index": record["slate_index"],
                "search_mode": record["search_mode"],
                "plan_status": record["status"],
                "realization_status": record.get("realization_status"),
                "failure_reason": record.get("failure_reason", ""),
                "task_metrics": record.get("task_metrics"),
                "palette_treatment": record.get("palette_treatment"),
                "technical": deepcopy(tech),
                "blind": deepcopy(solve),
                "editorial": deepcopy(editorial),
                "experimental_disposition": disposition,
                "task_path": (
                    f"tasks/anonymous/{task_id}.arc.json" if realized else None
                ),
                "dossier_path": (
                    f"tasks/private_dossiers/{task_id}.authoring.json"
                    if realized
                    else None
                ),
            }
        )
    return rows


def _selector_metrics(
    kernels: Sequence[Mapping[str, Any]],
    slates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sources = {
        item["candidate_id"]: item["private_kernel_author_provider"]
        for item in kernels
    }
    by_source: dict[str, dict[str, int]] = {
        source: {
            "kernels": sum(
                item["private_kernel_author_provider"] == source for item in kernels
            ),
            "develop_recommendations": 0,
            "hold_recommendations": 0,
            "reject_recommendations": 0,
            "derivative_flags": 0,
            "same_set_derivative_flags": 0,
            "same_procedure_mentions": 0,
            "selected_top_four": 0,
        }
        for source in AUTHOR_PROVIDERS
    }
    for slate in slates:
        selected_ids = {item["candidate_id"] for item in slate["selected"]}
        for item in slate["response"]["assessments"]:
            source = sources[item["candidate_id"]]
            by_source[source][f"{item['recommendation']}_recommendations"] += 1
            if item["derivative_of"] != "none":
                by_source[source]["derivative_flags"] += 1
            if item["derivative_of"] == "same_set":
                by_source[source]["same_set_derivative_flags"] += 1
            if item["same_minimal_procedure_with"]:
                by_source[source]["same_procedure_mentions"] += 1
            if item["candidate_id"] in selected_ids:
                by_source[source]["selected_top_four"] += 1
    for source, row in by_source.items():
        row["develop_rate_ppm"] = round(
            row["develop_recommendations"] / row["kernels"] * 1_000_000
        )
        row["selection_rate_ppm"] = round(
            row["selected_top_four"] / row["kernels"] * 1_000_000
        )
    return by_source


def _authoring_metrics(task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for source in AUTHOR_PROVIDERS:
        rows = [
            item
            for item in task_rows
            if item["private_kernel_author_provider"] == source
        ]
        realized = [item for item in rows if item["realization_status"] == "realized"]
        core = [item for item in rows if item["experimental_disposition"] == "core_quality"]
        reserve = [
            item for item in rows if item["experimental_disposition"] == "reserve_quality"
        ]
        result[source] = {
            "selected": len(rows),
            "plans_ready": sum(item["plan_status"] == "plan_ready" for item in rows),
            "realized": len(realized),
            "technical_exact_pass": sum(
                item["technical"] is not None
                and item["technical"].get("passed")
                and item["technical"].get("hidden_output_match")
                for item in realized
            ),
            "blind_completed": sum(
                item["blind"] is not None
                and not item["blind"].get("operational_failure")
                for item in realized
            ),
            "blind_primary_exact": sum(
                item["blind"] is not None
                and item["blind"].get("primary_matches_hidden") is True
                for item in realized
            ),
            "serious_different_output_alternative": sum(
                item["blind"] is not None
                and item["blind"].get("serious_different_output_alternative") is True
                for item in realized
            ),
            "core_quality": len(core),
            "reserve_quality": len(reserve),
            "core_or_reserve": len(core) + len(reserve),
            "core_per_selected": round(len(core) / len(rows), 6) if rows else None,
            "core_or_reserve_per_selected": (
                round((len(core) + len(reserve)) / len(rows), 6) if rows else None
            ),
        }
    return result


def _call_accounting(runtime: GeneratorReplicationRuntime) -> dict[str, Any]:
    overall = v4_call_accounting(runtime.state["call_ledger"], run_dir=runtime.run_dir)
    by_provider = {}
    for provider in AUTHOR_PROVIDERS:
        ledger = {
            key: item
            for key, item in runtime.state["call_ledger"].items()
            if item["provider"] == provider
        }
        by_provider[provider] = v4_call_accounting(ledger, run_dir=runtime.run_dir)
    by_role = {}
    for role in (
        "idea_search",
        "batch_selector",
        "concrete_planner",
        "task_realizer",
        "technical_verifier",
        "blind_solver",
        "editorial_board",
    ):
        ledger = {
            key: item
            for key, item in runtime.state["call_ledger"].items()
            if item["role"] == role
        }
        by_role[role] = v4_call_accounting(ledger, run_dir=runtime.run_dir)
    return {**overall, "by_provider": by_provider, "by_role": by_role}


def _render_review_pack(
    runtime: GeneratorReplicationRuntime,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    directory = runtime.run_dir / "human_review"
    directory.mkdir(parents=True, exist_ok=True)
    realized = [
        item for item in records.values() if item.get("realization_status") == "realized"
    ]
    order = list(realized)
    Random(int(runtime.config["seed"]) + 451999).shuffle(order)
    mapping = []
    images = []
    appendix = ["# Generator replication answer and rule appendix", ""]
    for index, record in enumerate(order, 1):
        alias = f"episode-{index:02d}"
        task_id = record["candidate_id"]
        image_path = directory / f"{alias}.png"
        render_episode(
            record["complete_task"],
            title=alias,
            subtitle="Kernel source hidden; inspect grids before the appendix",
            rule=None,
            destination=image_path,
        )
        images.append((alias, image_path))
        mapping.append(
            {
                "alias": alias,
                "task_id": task_id,
                "private_kernel_author_provider": record[
                    "private_kernel_author_provider"
                ],
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
        "# Blinded Opus-versus-Sol generator review\n\n"
        "Review every episode before opening the rule appendix. For each episode, "
        "record conceptual interest, functional composition, discoverability, visual "
        "economy, fairness, and overall preference. All selected kernels were planned, "
        "realized, and evaluated by the same Sol control; only their private idea source "
        "differs.\n",
    )
    return {
        "task_count": len(realized),
        "contact_sheet": str(contact) if contact else None,
        "review_sheet": str(directory / "review_sheet.md"),
        "answer_appendix": str(directory / "answer_rule_appendix.md"),
        "private_provenance_key": str(directory / "provenance_key.private.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    selector = report["selector_results"]
    authoring = report["downstream_results"]
    calls = report["call_accounting"]
    lines = [
        "# Claude Opus 5 generator-only replication",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "This is an isolated, non-gating V4.5 experiment. Only idea generation differs "
        "between arms; every selector and downstream authoring/evaluation call uses the "
        "same Sol control. No task was committed and production state was unchanged.",
        "",
        "## Design",
        "",
        "Three independent matched slates produced 36 Opus and 36 Sol kernels. Within "
        "each provider pair, prompts, calibration descriptions, schema, search mode, and "
        "max effort were identical. A source-blind Sol selector chose at most four ideas "
        "per slate with no source quota. All selected ideas then used Sol/max planning and "
        "realization, Sol/high technical verification, Sol/xhigh blind solving, and two "
        "Sol/max fixed-anchor boards.",
        "",
        "## Selection funnel",
        "",
        "| Kernel source | Kernels | Develop recommendations | Selected | Derivative flags | Same-procedure mentions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in AUTHOR_PROVIDERS:
        row = selector[source]
        lines.append(
            f"| {source} | {row['kernels']} | {row['develop_recommendations']} | "
            f"{row['selected_top_four']} | {row['derivative_flags']} | "
            f"{row['same_procedure_mentions']} |"
        )
    lines.extend(
        [
            "",
            "## Identical-Sol downstream funnel",
            "",
            "| Kernel source | Selected | Plans ready | Realized | Technical exact | Blind exact | Serious alternative | Core | Reserve |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source in AUTHOR_PROVIDERS:
        row = authoring[source]
        lines.append(
            f"| {source} | {row['selected']} | {row['plans_ready']} | {row['realized']} | "
            f"{row['technical_exact_pass']} | {row['blind_primary_exact']} | "
            f"{row['serious_different_output_alternative']} | {row['core_quality']} | "
            f"{row['reserve_quality']} |"
        )
    lines.extend(
        [
            "",
            "## Calls and time",
            "",
            f"- Logical calls: {calls['substantive_logical_calls']}",
            f"- Active model minutes: {calls['active_model_minutes']}",
            f"- Peak measured concurrency: {calls['measured_peak_concurrent_calls']}",
            f"- Claude calls (idea generation only): {calls['by_provider']['claude']['substantive_logical_calls']}",
            f"- Sol calls: {calls['by_provider']['codex']['substantive_logical_calls']}",
            "",
            "## Task outcomes",
            "",
            "| Task | Private kernel source | Slate | Mode | Outcome |",
            "|---|---|---:|---|---|",
        ]
    )
    for task in report["tasks"]:
        lines.append(
            f"| {task['task_id']} | {task['private_kernel_author_provider']} | "
            f"{task['slate_index']} | {task['search_mode']} | "
            f"{task['experimental_disposition']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            f"Recommendation: **{report['recommendation']}**.",
            "",
            "A blinded human review is still required before treating this sample as an "
            "author-model preference result.",
            "",
            f"Review pack: `{report['human_review_pack']['contact_sheet']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    runtime: GeneratorReplicationRuntime,
    kernels: Sequence[Mapping[str, Any]],
    slates: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    technical: Mapping[str, Mapping[str, Any]],
    blind: Mapping[str, Mapping[str, Any]],
    boards: Mapping[str, Any],
) -> dict[str, Any]:
    task_rows = _task_rows(records, technical, blind, boards)
    selector = _selector_metrics(kernels, slates)
    downstream = _authoring_metrics(task_rows)
    review = _render_review_pack(runtime, records)
    calls = _call_accounting(runtime)
    protected_after = _protected_state(runtime.config)
    _assert_unchanged(runtime.state["protected_state_before"], protected_after)
    claude_core = downstream["claude"]["core_quality"]
    codex_core = downstream["codex"]["core_quality"]
    claude_selected = selector["claude"]["selected_top_four"]
    codex_selected = selector["codex"]["selected_top_four"]
    if min(claude_selected, codex_selected) < 3:
        recommendation = "continue_testing_source_supply_imbalanced"
        interpretation = (
            "The source-blind selector supplied too few developed examples from at least "
            "one source for a stable downstream comparison. This is itself evidence about "
            "kernel competitiveness, but not enough to rank author models."
        )
    elif claude_core >= codex_core and claude_selected >= codex_selected:
        recommendation = "candidate_minority_reservoir_source_pending_human_review"
        interpretation = (
            "Opus kernels remained competitive after all post-selection authorship was "
            "standardized to Sol. The confounded signal from the first study therefore "
            "survived this replication, subject to blinded human review and further slates."
        )
    else:
        recommendation = "continue_testing_no_clear_noninferior_signal"
        interpretation = (
            "Under identical Sol downstream authorship, the current Opus kernel sample did "
            "not establish non-inferior core yield. This does not justify adding Opus to "
            "production supply yet."
        )
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
            "independent_slates": 3,
            "kernels_per_source": 36,
            "same_idea_prompts_calibrations_schema_and_effort": True,
            "source_blind_sol_selector": True,
            "source_balancing_enabled": False,
            "all_downstream_roles_use_sol_control": True,
            "repairs_enabled": False,
            "source_hidden_from_all_downstream_prompts": True,
        },
        "selector_results": selector,
        "downstream_results": downstream,
        "tasks": task_rows,
        "call_accounting": calls,
        "human_review_pack": review,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "production_promotion_occurred": False,
        "automatic_task_commit_occurred": False,
        "production_state_mutated": False,
        "protected_state_before": runtime.state["protected_state_before"],
        "protected_state_after": protected_after,
        "limitations": [
            "The experiment uses three slates, but human preference remains pending.",
            "Repairs are disabled to preserve generator attribution.",
            "Sol is both the source-blind selector and all downstream authoring control.",
            "Selected source counts are intentionally not balanced after quality ranking.",
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
        raise GeneratorReplicationError(
            "real model calls are not authorized; pass --authorize-model-calls"
        )
    design = freeze_experiment(config)
    runtime = GeneratorReplicationRuntime(
        config, require_attestation=not config["run"]["mock"]
    )
    report_path = runtime.run_dir / "report.json"
    if runtime.state.get("completed") and report_path.is_file():
        return read_json(report_path)
    kernels = generate_kernels(runtime, design)
    selected, slates = select_kernels(runtime, kernels)
    plans = plan_selected(runtime, selected)
    records = realize_plans(runtime, plans)
    technical = technical_verify(runtime, records)
    blind = blind_solve(runtime, records, technical)
    boards = editorial_review(runtime, records, technical, blind)
    return build_report(
        runtime, kernels, slates, records, technical, blind, boards
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m arcgen.lib.v45_claude_generator_replication",
        description="Isolated Opus-versus-Sol generator-only replication",
    )
    parser.add_argument("command", choices=("validate", "freeze", "run", "report"))
    parser.add_argument(
        "--config",
        default="config/v4_5_exp_claude_opus_generator_replication.yaml",
    )
    parser.add_argument("--run")
    parser.add_argument("--authorize-model-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            if not args.run:
                raise GeneratorReplicationError("--run is required for report")
            path = Path(args.run)
            if path.is_dir():
                path = path / "report.json"
            report = read_json(path)
            if report.get("pipeline") != PIPELINE:
                raise GeneratorReplicationError("run is not a generator replication")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        config = load_generator_replication_config(args.config)
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
        elif args.command == "run":
            result = run_experiment(
                config, authorize_model_calls=args.authorize_model_calls
            )
        else:  # pragma: no cover
            raise GeneratorReplicationError("unsupported command")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GeneratorReplicationError",
    "GeneratorReplicationRuntime",
    "PIPELINE",
    "VERSION",
    "freeze_experiment",
    "generate_kernels",
    "load_generator_replication_config",
    "run_experiment",
    "select_kernels",
]
