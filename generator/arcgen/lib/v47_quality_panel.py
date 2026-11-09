"""Provenance-blind cross-version quality panel for the V4.4/V4.5/V4.6 qualification.

The panel is deliberately downstream of the frozen production pipelines.  It
never mutates their state and never sees author rules, provider labels, version
labels, plans, or earlier judgments.  Previously committed core episodes are
mixed into every ballot as calibration controls.  Two independent ballots are
run routinely; a fresh third ballot is run for every production-core proposal
and every disagreement.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from .config import ConfigError
from .model import ModelInvocationError, ModelInvoker
from .v41_palette import palette_equivalent_hash
from .util import (
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    parallel_map,
    read_json,
    render_prompt,
)
from .validation import validate_arc_task
from .v47_blind_safety import effective_serious_blind_concern


PIPELINE = "arcgen-v4.7-baseline-quality-panel"
VERSION = "1.0"
STAGE = "v47_provenance_blind_quality"


class V47QualityError(RuntimeError):
    """Raised when review isolation or source integrity is violated."""


QUALITY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["reviewer_id", "ranking", "batch_observations"],
    "properties": {
        "reviewer_id": {"type": "string", "minLength": 1},
        "ranking": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "rank",
                    "task_id",
                    "novel_concept",
                    "functional_composition",
                    "overall_quality",
                    "discoverability",
                    "conceptual_economy",
                    "intrinsic_recommendation",
                    "serious_issue",
                    "serious_issue_evidence",
                    "rationale",
                ],
                "properties": {
                    "rank": {"type": "integer", "minimum": 1},
                    "task_id": {"type": "string", "minLength": 1},
                    "novel_concept": {"enum": ["weak", "solid", "standout"]},
                    "functional_composition": {"enum": ["thin", "solid", "standout"]},
                    "overall_quality": {"enum": ["below_real_bar", "real_quality", "exceptional"]},
                    "discoverability": {"enum": ["concern", "solid", "strong"]},
                    "conceptual_economy": {"enum": ["concern", "solid", "strong"]},
                    "intrinsic_recommendation": {
                        "enum": ["core_quality", "reserve_quality", "rejected_quality", "ambiguous"]
                    },
                    "serious_issue": {"type": "boolean"},
                    "serious_issue_evidence": {"type": "string"},
                    "rationale": {"type": "string", "minLength": 1},
                },
            },
        },
        "batch_observations": {"type": "string"},
    },
}


def _parse(_stage: str, raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("quality response must be an object")
    return value


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"quality config must be a mapping: {path}")
    return value


def _resolve(base: Path, value: str) -> str:
    path = Path(value)
    return str((path if path.is_absolute() else base / path).resolve())


def load_v47_quality_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = _read_yaml_or_json(source)
    if value.get("pipeline") != PIPELINE or str(value.get("version")) != VERSION:
        raise ConfigError(f"quality panel requires {PIPELINE} version {VERSION}")
    base = source.parent.parent if source.parent.name == "config" else source.parent
    paths = value.get("paths")
    if not isinstance(paths, Mapping):
        raise ConfigError("paths must be a mapping")
    for key in ("runs", "cache", "prompts", "committed_task_registry"):
        raw = paths.get(key)
        if not isinstance(raw, str) or not raw:
            raise ConfigError(f"paths.{key} is required")
        paths[key] = _resolve(base, raw)
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ConfigError("sources must be a non-empty list")
    for item in sources:
        if not isinstance(item, dict) or item.get("kind") not in {"v47_paired", "v46_hybrid"}:
            raise ConfigError("each source kind must be v47_paired or v46_hybrid")
        item["run"] = _resolve(base, str(item.get("run", "")))
    model = value.get("model")
    if not isinstance(model, Mapping):
        raise ConfigError("model must be a mapping")
    for key in ("provider", "executable", "name"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ConfigError(f"model.{key} is required")
    if model.get("name") != "gpt-5.6-sol":
        raise ConfigError("qualification quality panel must use GPT-5.6 Sol")
    for key in ("timeout_seconds", "retries", "concurrency"):
        if type(model.get(key)) is not int or model[key] < (0 if key == "retries" else 1):
            raise ConfigError(f"model.{key} is invalid")
    experiment = value.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ConfigError("experiment must be a mapping")
    for key in ("fresh_per_batch", "reference_per_batch", "historical_reference_count"):
        if type(experiment.get(key)) is not int or experiment[key] < 1:
            raise ConfigError(f"experiment.{key} must be positive")
    reviewers = experiment.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) != 2:
        raise ConfigError("exactly two routine reviewers are required")
    if {item.get("effort") for item in reviewers if isinstance(item, Mapping)} != {"max", "xhigh"}:
        raise ConfigError("routine reviewers must independently use max and xhigh")
    if experiment.get("adjudicator_effort") != "max":
        raise ConfigError("adjudicator effort must be max")
    run = value.get("run")
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str):
        raise ConfigError("run.id is required")
    if run.get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("quality panel cannot automatically commit tasks")
    if type(value.get("seed")) is not int:
        raise ConfigError("seed must be an integer")
    value["_config_path"] = str(source)
    value["_repository_root"] = str(base)
    return value


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _task_payload(path: Path) -> dict[str, Any]:
    task = read_json(path)
    result = validate_arc_task(task, require_test_outputs=True)
    if not result.valid:
        raise V47QualityError(f"invalid review task {path}: {'; '.join(result.errors)}")
    return task


def _pipeline_core(row: Mapping[str, Any]) -> bool:
    value = row.get("intrinsic_quality") or row.get("experimental_disposition")
    return value == "core_quality"


def _technical_clear(row: Mapping[str, Any]) -> bool:
    if "technical_pass" in row:
        return row.get("technical_pass") is True and row.get("hidden_output_exact") is not False
    technical = row.get("technical") or {}
    response = technical.get("response") or {}
    match = technical.get("hidden_output_match")
    return response.get("status") == "pass" and match is not False


def _blind_serious(row: Mapping[str, Any]) -> bool:
    if row.get("serious_ambiguity") is True:
        return True
    blind = row.get("blind") or {}
    # The V4.6 harness persists the primary solver's raw serious-alternative
    # flag even after an independent ambiguity adjudicator resolves it.  The
    # qualification gate must use the effective adjudicated outcome while the
    # original solver finding remains available in the dossier for audit.
    return effective_serious_blind_concern(blind) is True


def _load_v47_source(path: Path) -> list[dict[str, Any]]:
    report = read_json(path / "report.json")
    if report.get("pipeline") != "arcgen-v4.7-exp-v44-v45-paired":
        raise V47QualityError(f"unexpected paired source: {path}")
    records: list[dict[str, Any]] = []
    labels = {"v44_control": "v4.4", "v45_scale": "v4.5"}
    for arm, label in labels.items():
        for row in report["arms"][arm]["tasks"]:
            task_path = row.get("task_json")
            if not task_path:
                continue
            source_key = f"{report['run_id']}::{arm}::{row['task_id']}"
            records.append(
                {
                    "source_key": source_key,
                    "source_run": str(path),
                    "source_kind": "v47_paired",
                    "private_version": label,
                    "private_arm": arm,
                    "task_id": row["task_id"],
                    "lineage_id": row.get("paired_lineage_id") or row.get("source_candidate_id"),
                    "task_path": str(Path(task_path).resolve()),
                    "dossier_path": row.get("authoring_json"),
                    "task": _task_payload(Path(task_path)),
                    "pipeline_core": row.get("intrinsic_quality") == "core_quality",
                    "pipeline_intrinsic": row.get("intrinsic_quality"),
                    "portfolio_placement": row.get("portfolio_placement"),
                    "technical_clear": row.get("technical_pass") is True and row.get("hidden_output_exact") is not False,
                    "blind_serious": row.get("serious_ambiguity") is True,
                    "search_mode": row.get("search_mode"),
                }
            )
    return records


def _load_v46_source(path: Path) -> list[dict[str, Any]]:
    report = read_json(path / "report.json")
    if report.get("pipeline") != "arcgen-v4.6-exp-sol-opus-hybrid":
        raise V47QualityError(f"unexpected V4.6 source: {path}")
    records: list[dict[str, Any]] = []
    labels = {"sol_only": "v4.5", "hybrid": "v4.6"}
    for arm, label in labels.items():
        arm_report = report["arms"][arm]
        runtime = Path(arm_report["runtime_dir"])
        for row in arm_report["tasks"]:
            task_path = row.get("task_path")
            if task_path and not Path(str(task_path)).is_absolute():
                task_path = str(path / str(task_path))
            if not task_path:
                fallback = runtime / "tasks" / "anonymous" / f"{row['task_id']}.arc.json"
                task_path = str(fallback) if fallback.is_file() else None
            if not task_path or not Path(task_path).is_file():
                continue
            dossier = row.get("dossier_path")
            if dossier and not Path(str(dossier)).is_absolute():
                dossier = str(path / str(dossier))
            if not dossier:
                fallback = runtime / "tasks" / "private_dossiers" / f"{row['task_id']}.authoring.json"
                dossier = str(fallback) if fallback.is_file() else None
            source_key = f"{report['run_id']}::{arm}::{row['task_id']}"
            # Slot IDs in the historical harness are not provider-stable.  The
            # content hash below is the safe shared-lineage fallback; exact
            # duplicates are reviewed only once and their vote is propagated.
            records.append(
                {
                    "source_key": source_key,
                    "source_run": str(path),
                    "source_kind": "v46_hybrid",
                    "private_version": label,
                    "private_arm": arm,
                    "task_id": row["task_id"],
                    "lineage_id": row.get("paired_slot_id") or f"{row.get('slate_index')}::{row['task_id']}",
                    "task_path": str(Path(task_path).resolve()),
                    "dossier_path": dossier,
                    "task": _task_payload(Path(task_path)),
                    "pipeline_core": row.get("experimental_disposition") == "core_quality",
                    "pipeline_intrinsic": row.get("experimental_disposition"),
                    "portfolio_placement": (row.get("editorial") or {}).get("portfolio_placement"),
                    "technical_clear": _technical_clear(row),
                    "blind_serious": _blind_serious(row),
                    "search_mode": row.get("search_mode"),
                    "private_kernel_author_provider": row.get("private_kernel_author_provider"),
                }
            )
    return records


def _load_fresh(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for source in sources:
        path = Path(str(source["run"]))
        if not (path / "report.json").is_file():
            raise V47QualityError(f"source report is not complete: {path}")
        values.extend(
            _load_v47_source(path)
            if source["kind"] == "v47_paired"
            else _load_v46_source(path)
        )
    for record in values:
        record["task_hash"] = content_hash(record["task"])
    return values


def _historical_references(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry_path = Path(config["paths"]["committed_task_registry"])
    registry = read_json(registry_path)
    root = registry_path.parent
    # User-authorized commits intentionally set the live portfolio status to
    # ``include_now``.  For a provenance-blind calibration panel, however, a
    # task that the source pipeline marked as crowded or same-procedure must
    # not be allowed to overweight that mechanism merely because it was later
    # committed for preservation.  Prefer the source-era placement whenever
    # it was retained; old entries without that field fall back to the live
    # placement for backwards compatibility.
    candidates = [
        entry
        for entry in registry.get("entries", [])
        if entry.get("intrinsic_status") == "core_quality"
        and (
            entry.get("source_portfolio_status")
            if entry.get("source_portfolio_status") is not None
            else entry.get("portfolio_status")
        )
        == "include_now"
    ]
    # One exact episode and one source candidate lineage only.  This prevents
    # user-authorized siblings in the bank from overweighting a mechanism.
    unique: list[Mapping[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_palette_identities: set[str] = set()
    seen_lineages: set[tuple[str, str]] = set()
    for entry in candidates:
        digest = str(entry.get("arc_sha256"))
        lineage = (str(entry.get("source_run")), str(entry.get("source_candidate_id")))
        task_path = root / str(entry["task_path"])
        palette_identity = palette_equivalent_hash(read_json(task_path))
        if (
            digest in seen_hashes
            or palette_identity in seen_palette_identities
            or lineage in seen_lineages
        ):
            continue
        seen_hashes.add(digest)
        seen_palette_identities.add(palette_identity)
        seen_lineages.add(lineage)
        unique.append(entry)
    by_commit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in unique:
        by_commit[str(entry.get("commit_set_id"))].append(entry)
    rng = Random(int(config["seed"]) + 470701)
    for entries in by_commit.values():
        rng.shuffle(entries)
    groups = sorted(by_commit)
    rng.shuffle(groups)
    target = int(config["experiment"]["historical_reference_count"])
    selected: list[Mapping[str, Any]] = []
    while len(selected) < target and any(by_commit[group] for group in groups):
        for group in groups:
            if by_commit[group] and len(selected) < target:
                selected.append(by_commit[group].pop())
    result = []
    for entry in selected:
        path = root / str(entry["task_path"])
        result.append(
            {
                "reference_id": str(entry["registry_task_id"]),
                "task_path": str(path.resolve()),
                "task": _task_payload(path),
                "commit_set_id": entry.get("commit_set_id"),
                "task_hash": content_hash(read_json(path)),
            }
        )
    return result


def _opaque(rng: Random, used: set[str]) -> str:
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    while True:
        value = "T" + "".join(rng.choice(alphabet) for _ in range(11))
        if value not in used:
            used.add(value)
            return value


def _unique_payloads(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_hash[str(record["task_hash"])].append(record)
    representatives: list[dict[str, Any]] = []
    aliases: dict[str, list[str]] = {}
    for digest, members in sorted(by_hash.items()):
        representative = deepcopy(dict(members[0]))
        representative["duplicate_source_keys"] = [str(item["source_key"]) for item in members]
        representative["pipeline_core"] = any(bool(item["pipeline_core"]) for item in members)
        representative["technical_clear"] = all(bool(item["technical_clear"]) for item in members)
        representative["blind_serious"] = any(bool(item["blind_serious"]) for item in members)
        representatives.append(representative)
        aliases[digest] = representative["duplicate_source_keys"]
    return representatives, aliases


def _make_batches(
    fresh: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    fresh_per_batch: int,
    reference_per_batch: int,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rng = Random(seed)
    ordered = [deepcopy(dict(item)) for item in fresh]
    rng.shuffle(ordered)
    batch_count = max(1, math.ceil(len(ordered) / fresh_per_batch))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(batch_count)]
    lineages: list[set[str]] = [set() for _ in range(batch_count)]
    for record in ordered:
        lineage = str(record.get("lineage_id"))
        choices = [
            index
            for index, bucket in enumerate(buckets)
            if len(bucket) < fresh_per_batch and lineage not in lineages[index]
        ]
        if not choices:
            choices = [index for index, bucket in enumerate(buckets) if len(bucket) < fresh_per_batch]
        if not choices:
            buckets.append([])
            lineages.append(set())
            choices = [len(buckets) - 1]
        index = min(choices, key=lambda item: len(buckets[item]))
        buckets[index].append(record)
        lineages[index].add(lineage)
    used: set[str] = set()
    private_mapping: dict[str, dict[str, Any]] = {}
    batches: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        ref_start = (index * reference_per_batch) % len(references)
        chosen_refs = [references[(ref_start + offset) % len(references)] for offset in range(reference_per_batch)]
        items = []
        for record in bucket:
            opaque = _opaque(rng, used)
            private_mapping[opaque] = {"kind": "fresh", "task_hash": record["task_hash"]}
            items.append({"task_id": opaque, "task": record["task"]})
        for reference in chosen_refs:
            opaque = _opaque(rng, used)
            private_mapping[opaque] = {"kind": "reference", "reference_id": reference["reference_id"]}
            items.append({"task_id": opaque, "task": reference["task"]})
        rng.shuffle(items)
        batches.append(
            {
                "batch_id": f"{prefix}-{index + 1:03d}",
                "items": items,
                "task_ids": [item["task_id"] for item in items],
            }
        )
    return batches, private_mapping


def _core_vote(item: Mapping[str, Any]) -> bool:
    return bool(
        item.get("intrinsic_recommendation") == "core_quality"
        and item.get("novel_concept") in {"solid", "standout"}
        and item.get("functional_composition") in {"solid", "standout"}
        and "standout" in {item.get("novel_concept"), item.get("functional_composition")}
        and item.get("overall_quality") in {"real_quality", "exceptional"}
        and item.get("serious_issue") is False
    )


def _wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.96
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


class V47QualityPanel:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.template, self.template_version, self.template_hash = load_prompt(
            Path(config["paths"]["prompts"]) / "v4_7_provenance_blind_quality_ballot.md"
        )
        self.adj_template, self.adj_version, self.adj_hash = load_prompt(
            Path(config["paths"]["prompts"]) / "v4_7_provenance_blind_quality_adjudicator.md"
        )
        self.fresh_all = _load_fresh(config["sources"])
        self.fresh, self.payload_sources = _unique_payloads(self.fresh_all)
        self.references = _historical_references(config)
        self.batches, self.mapping = _make_batches(
            self.fresh,
            self.references,
            seed=int(config["seed"]) + 471001,
            fresh_per_batch=int(config["experiment"]["fresh_per_batch"]),
            reference_per_batch=int(config["experiment"]["reference_per_batch"]),
            prefix="routine",
        )
        self._freeze()

    def _freeze(self) -> None:
        snapshot = {key: deepcopy(value) for key, value in self.config.items() if not key.startswith("_")}
        snapshot["source_report_hashes"] = {
            str(item["run"]): _sha(Path(str(item["run"])) / "report.json") for item in self.config["sources"]
        }
        snapshot["committed_registry_sha256"] = _sha(Path(self.config["paths"]["committed_task_registry"]))
        path = self.run_dir / "config.snapshot.json"
        if path.exists() and read_json(path) != snapshot:
            raise V47QualityError("quality-panel inputs changed during resume")
        if not path.exists():
            atomic_write_json(path, snapshot)
        private = {
            "mapping": self.mapping,
            "payload_sources": self.payload_sources,
            "fresh_private_metadata": {
                record["task_hash"]: {key: value for key, value in record.items() if key != "task"}
                for record in self.fresh
            },
            "historical_references": [
                {key: value for key, value in item.items() if key != "task"} for item in self.references
            ],
        }
        private_path = self.run_dir / "assignment_ledger.private.json"
        if private_path.exists() and read_json(private_path) != private:
            raise V47QualityError("quality-panel assignment changed during resume")
        if not private_path.exists():
            atomic_write_json(private_path, private)
        prompt_manifest = {
            "schema_version": 1,
            "prompts": {
                "v4_7_provenance_blind_quality_ballot.md": {
                    "version": self.template_version,
                    "sha256": self.template_hash,
                },
                "v4_7_provenance_blind_quality_adjudicator.md": {
                    "version": self.adj_version,
                    "sha256": self.adj_hash,
                },
            },
        }
        prompt_path = self.run_dir / "prompts" / "manifest.json"
        if prompt_path.exists() and read_json(prompt_path) != prompt_manifest:
            raise V47QualityError("quality-panel prompts changed during resume")
        if not prompt_path.exists():
            atomic_write_json(prompt_path, prompt_manifest)
        repository_root = Path(self.config["_repository_root"])
        implementation = {
            "schema_version": 1,
            "pipeline": PIPELINE,
            "pipeline_version": VERSION,
            "files": {
                "arcgen.lib/v47_quality_panel.py": _sha(Path(__file__)),
                "config/v4_7_baseline_quality_panel.yaml": _sha(
                    Path(self.config["_config_path"])
                ),
                "prompts/v4_7_provenance_blind_quality_ballot.md": _sha(
                    repository_root
                    / "prompts/v4_7_provenance_blind_quality_ballot.md"
                ),
                "prompts/v4_7_provenance_blind_quality_adjudicator.md": _sha(
                    repository_root
                    / "prompts/v4_7_provenance_blind_quality_adjudicator.md"
                ),
            },
            "source_report_hashes": snapshot["source_report_hashes"],
            "committed_registry_sha256": snapshot["committed_registry_sha256"],
            "arm_labels_hidden": True,
            "author_rules_hidden": True,
        }
        implementation_path = self.run_dir / "implementation_manifest.json"
        if implementation_path.exists() and read_json(implementation_path) != implementation:
            raise V47QualityError("quality-panel implementation changed during resume")
        if not implementation_path.exists():
            atomic_write_json(implementation_path, implementation)
        public_batches = [
            {"batch_id": batch["batch_id"], "task_ids": batch["task_ids"], "arm_labels_hidden": True}
            for batch in self.batches
        ]
        atomic_write_json(self.run_dir / "batch_manifest.blinded.json", public_batches)

    def _invoker(self) -> ModelInvoker:
        return ModelInvoker(
            {
                "model": deepcopy(self.config["model"]),
                "paths": {"cache": self.config["paths"]["cache"]},
                "run": {"mock": bool(self.config["run"].get("mock"))},
            },
            self.run_dir,
            output_schemas={STAGE: QUALITY_SCHEMA},
            output_parser=_parse,
        )

    @staticmethod
    def _validate(value: Mapping[str, Any], reviewer_id: str, expected: set[str]) -> None:
        if value.get("reviewer_id") != reviewer_id:
            raise V47QualityError("reviewer changed its opaque assignment")
        ranking = value.get("ranking")
        if not isinstance(ranking, list):
            raise V47QualityError("reviewer omitted ranking")
        ids = [item.get("task_id") for item in ranking]
        ranks = [item.get("rank") for item in ranking]
        if set(ids) != expected or len(ids) != len(expected):
            raise V47QualityError("reviewer returned incomplete opaque IDs")
        if ranks != list(range(1, len(ranking) + 1)):
            raise V47QualityError("reviewer ranks must be consecutive")

    def _call(
        self,
        invoker: ModelInvoker,
        batch: Mapping[str, Any],
        *,
        reviewer_name: str,
        effort: str,
        adjudication: bool,
    ) -> dict[str, Any]:
        reviewer_id = f"{batch['batch_id']}-{reviewer_name}"
        expected = set(batch["task_ids"])
        template = self.adj_template if adjudication else self.template
        version = self.adj_version if adjudication else self.template_version
        digest = self.adj_hash if adjudication else self.template_hash
        started = time.time()
        try:
            result = invoker.invoke(
                stage=STAGE,
                call_id=reviewer_id,
                prompt=render_prompt(
                    template,
                    {"REVIEWER_ID": reviewer_id, "TASKS_JSON": batch["items"]},
                ),
                prompt_template_version=version,
                prompt_template_hash=digest,
                parent_ids=[],
                response_validator=lambda value: self._validate(value, reviewer_id, expected),
                model_name=self.config["model"]["name"],
                reasoning_effort=effort,
                timeout_seconds=int(self.config["model"]["timeout_seconds"]),
                retries=int(self.config["model"]["retries"]),
            )
            timing = read_json(result.log_dir / "result.json").get("timing", {})
            return {
                "batch_id": batch["batch_id"],
                "reviewer_id": reviewer_id,
                "effort": effort,
                "adjudication": adjudication,
                "status": "success",
                "timing": timing,
                "wall_seconds": round(time.time() - started, 3),
                "response": result.value,
            }
        except (ModelInvocationError, V47QualityError, ValueError, OSError) as exc:
            return {
                "batch_id": batch["batch_id"],
                "reviewer_id": reviewer_id,
                "effort": effort,
                "adjudication": adjudication,
                "status": "failed",
                "error": str(exc),
                "wall_seconds": round(time.time() - started, 3),
            }

    def _votes(self, calls: Sequence[Mapping[str, Any]], mapping: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for call in calls:
            if call.get("status") != "success":
                continue
            for item in call["response"]["ranking"]:
                source = mapping[item["task_id"]]
                key = (
                    str(source["task_hash"])
                    if source["kind"] == "fresh"
                    else f"reference::{source['reference_id']}"
                )
                votes[key].append(
                    {
                        "reviewer_id": call["reviewer_id"],
                        "effort": call["effort"],
                        "adjudication": call["adjudication"],
                        "core_vote": _core_vote(item),
                        "judgment": deepcopy(item),
                    }
                )
        return votes

    @staticmethod
    def _reference_comparisons(
        calls: Sequence[Mapping[str, Any]],
        mapping: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Count grid-only ballot wins against committed-core references.

        The comparison is descriptive and never changes the core ballot.  It
        makes the requested historical-quality comparison explicit while
        preserving the exact same blinded rankings and model-call budget.
        """

        comparisons: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for call in calls:
            if call.get("status") != "success":
                continue
            ranking = call["response"]["ranking"]
            ranks = {str(item["task_id"]): int(item["rank"]) for item in ranking}
            reference_ids = [
                task_id
                for task_id in ranks
                if mapping[task_id]["kind"] == "reference"
            ]
            for task_id, rank in ranks.items():
                source = mapping[task_id]
                if source["kind"] != "fresh":
                    continue
                digest = str(source["task_hash"])
                wins = sum(rank < ranks[reference_id] for reference_id in reference_ids)
                comparisons[digest].append(
                    {
                        "reviewer_id": call["reviewer_id"],
                        "effort": call["effort"],
                        "adjudication": call["adjudication"],
                        "reference_count": len(reference_ids),
                        "wins_over_committed_core_references": wins,
                    }
                )
        return comparisons

    def run(self, *, authorize_model_calls: bool) -> dict[str, Any]:
        if not authorize_model_calls and not self.config["run"].get("mock"):
            raise V47QualityError("real quality calls require --authorize-model-calls")
        report_path = self.run_dir / "report.json"
        if report_path.exists():
            prior = read_json(report_path)
            if prior.get("completed") is True:
                return prior
        invoker = self._invoker()
        routine_specs = [
            (batch, reviewer)
            for batch in self.batches
            for reviewer in self.config["experiment"]["reviewers"]
        ]
        started = time.time()
        routine = parallel_map(
            lambda spec: self._call(
                invoker,
                spec[0],
                reviewer_name=str(spec[1]["id"]),
                effort=str(spec[1]["effort"]),
                adjudication=False,
            ),
            routine_specs,
            min(len(routine_specs), int(self.config["model"]["concurrency"])),
        )
        routine_votes = self._votes(routine, self.mapping)
        by_hash = {str(record["task_hash"]): record for record in self.fresh}
        needs_adjudication = []
        for digest, record in by_hash.items():
            votes = routine_votes.get(digest, [])
            decisions = {bool(vote["core_vote"]) for vote in votes}
            if record["pipeline_core"] or len(decisions) != 1 or len(votes) != 2:
                needs_adjudication.append(record)
        adj_batches, adj_mapping = _make_batches(
            needs_adjudication,
            self.references,
            seed=int(self.config["seed"]) + 471901,
            fresh_per_batch=int(self.config["experiment"]["fresh_per_batch"]),
            reference_per_batch=int(self.config["experiment"]["reference_per_batch"]),
            prefix="adjudication",
        )
        atomic_write_json(
            self.run_dir / "adjudication_manifest.blinded.json",
            [{"batch_id": item["batch_id"], "task_ids": item["task_ids"]} for item in adj_batches],
        )
        atomic_write_json(self.run_dir / "adjudication_mapping.private.json", adj_mapping)
        adjudication = parallel_map(
            lambda batch: self._call(
                invoker,
                batch,
                reviewer_name="fresh-max-adjudicator",
                effort=str(self.config["experiment"]["adjudicator_effort"]),
                adjudication=True,
            ),
            adj_batches,
            min(len(adj_batches), int(self.config["model"]["concurrency"])),
        ) if adj_batches else []
        all_votes = self._votes([*routine, *adjudication], {**self.mapping, **adj_mapping})
        reference_comparisons = self._reference_comparisons(
            [*routine, *adjudication], {**self.mapping, **adj_mapping}
        )
        decisions: dict[str, Any] = {}
        for digest, record in sorted(by_hash.items()):
            votes = all_votes.get(digest, [])
            core_votes = sum(bool(item["core_vote"]) for item in votes)
            required = 2 if len(votes) >= 3 else len(votes)
            panel_core = len(votes) >= 2 and core_votes >= required
            strict = bool(
                record["pipeline_core"]
                and record["technical_clear"]
                and not record["blind_serious"]
                and panel_core
            )
            decisions[digest] = {
                "task_hash": digest,
                "source_keys": self.payload_sources[digest],
                "pipeline_core": record["pipeline_core"],
                "technical_clear": record["technical_clear"],
                "blind_serious": record["blind_serious"],
                "vote_count": len(votes),
                "core_votes": core_votes,
                "panel_core_consensus": panel_core,
                "strict_commit_eligible": strict,
                "votes": votes,
                "committed_core_reference_comparisons": reference_comparisons.get(
                    digest, []
                ),
            }
        # Propagate payload-level decisions to every arm record without
        # counting an exact duplicate twice.
        source_decisions = {
            source_key: decisions[digest]
            for digest, source_keys in self.payload_sources.items()
            for source_key in source_keys
        }
        versions: dict[str, Any] = {}
        for version in ("v4.4", "v4.5", "v4.6"):
            records = [record for record in self.fresh_all if record["private_version"] == version]
            pipeline_core = sum(bool(record["pipeline_core"]) for record in records)
            panel_core = sum(bool(source_decisions[record["source_key"]]["panel_core_consensus"]) for record in records)
            strict = sum(
                bool(
                    source_decisions[record["source_key"]]["strict_commit_eligible"]
                    and record["pipeline_core"]
                    and record["technical_clear"]
                    and not record["blind_serious"]
                )
                for record in records
            )
            comparison_rows = [
                comparison
                for record in records
                for comparison in source_decisions[record["source_key"]].get(
                    "committed_core_reference_comparisons", []
                )
            ]
            comparison_wins = sum(
                int(row["wins_over_committed_core_references"])
                for row in comparison_rows
            )
            comparison_opportunities = sum(
                int(row["reference_count"]) for row in comparison_rows
            )
            versions[version] = {
                "completed": len(records),
                "pipeline_core": pipeline_core,
                "panel_core": panel_core,
                "strict_commit_eligible": strict,
                "pipeline_core_rate": round(pipeline_core / len(records), 4) if records else None,
                "pipeline_core_wilson_95": _wilson(pipeline_core, len(records)),
                "panel_core_rate": round(panel_core / len(records), 4) if records else None,
                "panel_core_wilson_95": _wilson(panel_core, len(records)),
                "committed_core_reference_wins": comparison_wins,
                "committed_core_reference_opportunities": comparison_opportunities,
                "committed_core_reference_win_rate": round(
                    comparison_wins / comparison_opportunities, 4
                )
                if comparison_opportunities
                else None,
            }
        reference_votes = {
            key: value for key, value in all_votes.items() if key.startswith("reference::")
        }
        ref_ballots = [vote for votes in reference_votes.values() for vote in votes]
        calls = [*routine, *adjudication]
        model_seconds = sum(
            float(call.get("timing", {}).get("model_execution_seconds") or 0)
            for call in calls
            if call.get("status") == "success"
        )
        failed_calls = sum(call.get("status") != "success" for call in calls)
        report = {
            "schema_version": 1,
            "pipeline": PIPELINE,
            "pipeline_version": VERSION,
            "run_id": self.config["run"]["id"],
            # An incomplete report is deliberately resumable.  Successful
            # calls remain content-addressed cache hits; only failed calls are
            # re-attempted on the next invocation.
            "completed": failed_calls == 0,
            "fresh_source_record_count": len(self.fresh_all),
            "unique_fresh_episode_count": len(self.fresh),
            "exact_duplicate_record_count": len(self.fresh_all) - len(self.fresh),
            "historical_reference_count": len(self.references),
            "routine_batch_count": len(self.batches),
            "routine_call_count": len(routine),
            "adjudication_batch_count": len(adj_batches),
            "adjudication_call_count": len(adjudication),
            "successful_calls": sum(call.get("status") == "success" for call in calls),
            "failed_calls": failed_calls,
            "model_execution_minutes": round(model_seconds / 60.0, 3),
            "wall_clock_minutes": round((time.time() - started) / 60.0, 3),
            "historical_core_ballot_pass_rate": round(
                sum(bool(vote["core_vote"]) for vote in ref_ballots) / len(ref_ballots), 4
            ) if ref_ballots else None,
            "committed_core_reference_pairwise": {
                "wins": sum(
                    int(row["wins_over_committed_core_references"])
                    for rows in reference_comparisons.values()
                    for row in rows
                ),
                "opportunities": sum(
                    int(row["reference_count"])
                    for rows in reference_comparisons.values()
                    for row in rows
                ),
            },
            "versions": versions,
            "decisions": decisions,
            "source_decisions": source_decisions,
            "strict_commit_eligible_unique_count": sum(
                bool(item["strict_commit_eligible"]) for item in decisions.values()
            ),
            "routine_calls": routine,
            "adjudication_calls": adjudication,
            "arm_labels_hidden_downstream": True,
            "author_rules_hidden": True,
            "automatic_task_commit_occurred": False,
        }
        atomic_write_json(report_path, report)
        atomic_write_text(self.run_dir / "report.md", self._markdown(report))
        return report

    @staticmethod
    def _markdown(report: Mapping[str, Any]) -> str:
        lines = [
            "# Provenance-blind V4.4 / V4.5 / V4.6 quality panel",
            "",
            f"- Unique fresh episodes: {report['unique_fresh_episode_count']}",
            f"- Historical committed-core references: {report['historical_reference_count']}",
            f"- Calls: {report['successful_calls']} successful, {report['failed_calls']} failed",
            f"- Historical-core ballot calibration pass rate: {report['historical_core_ballot_pass_rate']}",
            f"- Strict unique commit-eligible episodes: {report['strict_commit_eligible_unique_count']}",
            "",
            "| Version | Completed | Pipeline core | Blind-panel core | Strict commit eligible | Wins vs prior cores |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for version, values in report["versions"].items():
            lines.append(
                f"| {version} | {values['completed']} | {values['pipeline_core']} | "
                f"{values['panel_core']} | {values['strict_commit_eligible']} | "
                f"{values['committed_core_reference_win_rate']} |"
            )
        lines.extend(
            [
                "",
                "Reviewers saw complete grids and expected outputs, but no author rule, plan, provider, version, arm, or prior judgment. Every production-core proposal and every routine disagreement received an independent third Sol/max ballot.",
                "",
            ]
        )
        return "\n".join(lines)


def validate_v47_quality(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = [str(item["run"]) for item in config["sources"] if not (Path(str(item["run"])) / "report.json").is_file()]
    return {
        "valid": not missing,
        "pipeline": PIPELINE,
        "run_id": config["run"]["id"],
        "pending_source_runs": missing,
        "reviewers": deepcopy(config["experiment"]["reviewers"]),
        "adjudicator_effort": config["experiment"]["adjudicator_effort"],
        "arm_labels_hidden": True,
        "author_rules_hidden": True,
        "automatic_task_commit": False,
    }


def _parser_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "report"))
    parser.add_argument("--config", default="config/v4_7_baseline_quality_panel.yaml")
    parser.add_argument("--run")
    parser.add_argument("--authorize-model-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser_cli().parse_args(argv)
    try:
        if args.command == "report":
            if not args.run:
                raise V47QualityError("--run is required")
            value = read_json(Path(args.run) / "report.json")
        else:
            config = load_v47_quality_config(args.config)
            value = (
                validate_v47_quality(config)
                if args.command == "validate"
                else V47QualityPanel(config).run(authorize_model_calls=args.authorize_model_calls)
            )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PIPELINE",
    "QUALITY_SCHEMA",
    "V47QualityError",
    "V47QualityPanel",
    "load_v47_quality_config",
    "validate_v47_quality",
]
