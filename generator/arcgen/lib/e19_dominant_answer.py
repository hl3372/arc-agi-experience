"""E19 dominant-answer gate calibration and accepted-bank re-adjudication.

The experiment is deliberately additive and non-mutating.  Historical registry
and dossier files are inputs; all responses and provisional resolutions live in
an isolated run directory until a separate, explicit resolution command is
implemented and invoked.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from .committed_tasks import load_committed_task_registry
from .e19_schemas import (
    OUTPUT_SCHEMAS,
    normalize_adjudication,
    normalize_dominant_review,
    parse_stage_output,
)
from .lean_pipeline import serious_blind_alternative
from .model import CallResult, ModelInvoker
from .util import (
    PLACEHOLDER_RE,
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    read_json,
    render_prompt,
)
from .validation import validate_arc_task
from .v4_pipeline import v4_call_accounting


PIPELINE = "arcgen-e19-dominant-answer-verification"
VERSION = "1.0.0"


class E19Error(ValueError):
    pass


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E19Error("E19 config must be a YAML object")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _read_yaml(config_path)
    if config.get("pipeline") != PIPELINE:
        raise E19Error(f"expected pipeline {PIPELINE!r}")
    root = config_path.parent.parent.resolve()
    config["_config_path"] = str(config_path)
    config["_repository_root"] = str(root)
    for key, raw in list(config.get("paths", {}).items()):
        value = Path(str(raw))
        if not value.is_absolute():
            value = root / value
        config["paths"][key] = str(value.resolve())
    registry = Path(config["paths"]["registry"])
    expected = config["expected_state"]
    if _sha(registry) != expected["registry_sha256"]:
        raise E19Error("committed registry differs from the frozen E19 input")
    verified = load_committed_task_registry(
        registry,
        asset_root=config["paths"].get("registry_asset_root"),
    )
    if len(verified["entries"]) != int(expected["registry_count"]):
        raise E19Error("committed registry count differs from the frozen E19 input")
    experiment = config["experiment"]
    forbidden = (
        not experiment.get("non_gating")
        or experiment.get("automatic_registry_mutation")
        or experiment.get("automatic_task_commit")
        or config.get("run", {}).get("automatic_task_commit")
        or config.get("run", {}).get("production_state_mutation_allowed")
    )
    if forbidden:
        raise E19Error("E19 must be non-gating and unable to mutate production state")
    return config


def _load_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    return load_committed_task_registry(
        config["paths"]["registry"],
        asset_root=config["paths"].get("registry_asset_root"),
    )


def public_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train": deepcopy(task["train"]),
        "test": [{"input": deepcopy(pair["input"])} for pair in task["test"]],
    }


def proposed_outputs(task: Mapping[str, Any]) -> list[Any]:
    return [deepcopy(pair["output"]) for pair in task["test"]]


def corrected_blind_trigger(
    response: Mapping[str, Any], hidden_outputs: Sequence[Any]
) -> dict[str, Any]:
    """Apply the proposed trigger without changing the historical predicate."""

    old_serious = serious_blind_alternative(dict(response), hidden_outputs)
    primary_present = isinstance(response.get("predicted_test_outputs"), list)
    primary_differs = primary_present and response.get("predicted_test_outputs") != list(
        hidden_outputs
    )
    raw_serious = response.get("serious_different_output_alternative") is True
    unsupported_guessing = response.get("test_requires_unsupported_guessing") is True
    reasons: list[str] = []
    if old_serious:
        reasons.append("listed_comparable_different_output")
    if primary_differs and raw_serious:
        reasons.append("wrong_primary_with_raw_serious_flag")
    if unsupported_guessing:
        reasons.append("unsupported_guessing")
    if raw_serious and not primary_present:
        reasons.append("internally_incomplete_serious_flag")
    return {
        "old_serious_predicate": old_serious,
        "primary_output_present": primary_present,
        "primary_differs_from_hidden": primary_differs,
        "raw_serious_flag": raw_serious,
        "unsupported_guessing_flag": unsupported_guessing,
        "proposed_review_required": bool(reasons),
        "trigger_reasons": reasons,
        "automatic_rejection": False,
    }


def _response_from_wrapper(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    response = value.get("response")
    if isinstance(response, Mapping) and "predicted_test_outputs" in response:
        return response
    if "predicted_test_outputs" in value:
        return value
    return None


def terminal_blind_responses(review: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Normalize the four committed-dossier blind record shapes.

    For sequential lists, only the final response applies to the final episode;
    earlier calls can refer to a superseded pre-repair task.  Dual-provider
    records preserve both terminal providers.
    """

    blind = review.get("blind_results")
    if isinstance(blind, list):
        for value in reversed(blind):
            response = _response_from_wrapper(value)
            if response is not None:
                return [response]
        return []
    if not isinstance(blind, Mapping):
        return []
    if "codex" in blind or "claude" in blind:
        responses = []
        for key in ("codex", "claude"):
            response = _response_from_wrapper(blind.get(key))
            if response is not None:
                responses.append(response)
        return responses
    response = _response_from_wrapper(blind)
    if response is not None:
        return [response]
    provider_results = blind.get("provider_results")
    if isinstance(provider_results, Mapping):
        responses = []
        for key in sorted(provider_results):
            response = _response_from_wrapper(provider_results[key])
            if response is not None:
                responses.append(response)
        return responses
    return []


def _review_path(root: Path, task_id: str) -> Path:
    return root / "committed_tasks" / "dossiers" / task_id / "review.json"


def replay_committed_bank(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = _load_registry(config)
    root = Path(config["_repository_root"])
    rows: list[dict[str, Any]] = []
    for entry in registry["entries"]:
        task_id = entry["registry_task_id"]
        review_path = _review_path(root, task_id)
        review = read_json(review_path)
        task = review.get("complete_task")
        if not isinstance(task, Mapping):
            task = read_json(Path(entry["absolute_task_path"]))
        hidden = proposed_outputs(task)
        responses = terminal_blind_responses(review)
        gates = [corrected_blind_trigger(response, hidden) for response in responses]
        proposed = any(item["proposed_review_required"] for item in gates)
        old = any(item["old_serious_predicate"] for item in gates)
        rows.append(
            {
                "registry_task_id": task_id,
                "intrinsic_status": entry["intrinsic_status"],
                "source_pipeline": entry["source_pipeline"],
                "task_hash": content_hash(task),
                "review_sha256": _sha(review_path),
                "terminal_response_count": len(responses),
                "old_review_required": old,
                "proposed_review_required": proposed,
                "trigger_reasons": sorted(
                    {reason for item in gates for reason in item["trigger_reasons"]}
                ),
                "response_gates": gates,
            }
        )
    audit = read_json(Path(config["paths"]["safety_audit"]))
    expected = {
        item["registry_task_id"]
        for item in audit["raw_serious_wrapper_conflicts"]
        + audit["terminal_unsupported_guessing_flags"]
    }
    actual = {item["registry_task_id"] for item in rows if item["proposed_review_required"]}
    value = {
        "schema_version": 1,
        "registry_sha256": _sha(config["paths"]["registry"]),
        "task_count": len(rows),
        "old_review_required_count": sum(item["old_review_required"] for item in rows),
        "proposed_review_required_count": len(actual),
        "newly_escalated_count": sum(
            item["proposed_review_required"] and not item["old_review_required"]
            for item in rows
        ),
        "expected_audit_ids": sorted(expected),
        "proposed_review_ids": sorted(actual),
        "exactly_reproduces_safety_audit": actual == expected,
        "simulated_conditional_review_rate": len(actual) / len(rows) if rows else None,
        "intrinsic_counts": dict(
            Counter(
                item["intrinsic_status"]
                for item in rows
                if item["proposed_review_required"]
            )
        ),
        "trigger_reason_counts": dict(
            Counter(reason for item in rows for reason in item["trigger_reasons"])
        ),
        "rows": rows,
    }
    if not value["exactly_reproduces_safety_audit"]:
        raise E19Error("proposed replay does not reproduce the frozen nine-task audit")
    return value


def _historical_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = read_json(Path(config["paths"]["historical_cases"]))
    if manifest.get("schema_version") != 2:
        raise E19Error("unsupported historical calibration manifest")
    root = Path(config["_repository_root"])
    cases = []
    for item in manifest["cases"]:
        path = Path(item["task_path"])
        if not path.is_absolute():
            path = root / path
        task = read_json(path)
        validation = validate_arc_task(task, require_test_outputs=True)
        if not validation.valid:
            raise E19Error(f"invalid historical task {item['case_id']}")
        cases.append(
            {
                "case_id": f"historical::{item['case_id']}",
                "source_kind": "historical_calibration",
                "source_id": item["case_id"],
                "task": task,
                "task_hash": content_hash(task),
                "reference_problem": item["answer_support_problem"],
                "reference_label": item["reference_label"],
                "reference_certainty": item["reference_certainty"],
                "reference_basis": item["reference_basis"],
                "reference_reason": item["reference_reason"],
            }
        )
    return cases


def _flagged_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["_repository_root"])
    registry = _load_registry(config)
    entries = {item["registry_task_id"]: item for item in registry["entries"]}
    audit = read_json(Path(config["paths"]["safety_audit"]))
    raw = audit["raw_serious_wrapper_conflicts"] + audit["terminal_unsupported_guessing_flags"]
    cases = []
    for flag in raw:
        task_id = flag["registry_task_id"]
        entry = entries[task_id]
        task = read_json(Path(entry["absolute_task_path"]))
        review = read_json(_review_path(root, task_id))
        cases.append(
            {
                "case_id": f"accepted::{task_id}",
                "source_kind": "accepted_review_required",
                "source_id": task_id,
                "task": task,
                "task_hash": content_hash(task),
                "intrinsic_status": entry["intrinsic_status"],
                "reference_problem": None,
                "reference_label": "re_review_required",
                "reference_certainty": "unresolved",
                "reference_basis": "committed_bank_safety_audit",
                "reference_reason": flag["status"],
                "existing_review": review,
            }
        )
    return cases


def build_calibration_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Merge exact episode duplicates while preserving all private labels."""

    by_hash: dict[str, dict[str, Any]] = {}
    for case in [*_historical_cases(config), *_flagged_cases(config)]:
        prior = by_hash.get(case["task_hash"])
        source = {
            "case_id": case["case_id"],
            "source_kind": case["source_kind"],
            "source_id": case["source_id"],
            "reference_problem": case.get("reference_problem"),
            "reference_label": case.get("reference_label"),
            "reference_certainty": case.get("reference_certainty"),
            "reference_basis": case.get("reference_basis"),
            "reference_reason": case.get("reference_reason"),
            "intrinsic_status": case.get("intrinsic_status"),
        }
        if prior is None:
            merged = deepcopy(case)
            merged["sources"] = [source]
            merged["accepted_task_ids"] = (
                [case["source_id"]]
                if case["source_kind"] == "accepted_review_required"
                else []
            )
            by_hash[case["task_hash"]] = merged
        else:
            prior["sources"].append(source)
            if case["source_kind"] == "accepted_review_required":
                prior["accepted_task_ids"].append(case["source_id"])
                prior["existing_review"] = case["existing_review"]
    return sorted(by_hash.values(), key=lambda item: item["task_hash"])


class E19Runtime:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root = Path(config["_repository_root"])
        self.run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.state_path = self.run_dir / "state.json"
        self.prompts: dict[str, dict[str, str]] = {}
        prompt_specs = {
            "dominant_review": (
                "e19_dominant_answer_reviewer.md",
                "e19_dominant_review",
            ),
            "blind_solver": ("v4_blind_solver.md", "v4_blind_solver"),
            "dominant_adjudicator": (
                "e19_dominant_answer_adjudicator.md",
                "e19_dominant_adjudication",
            ),
        }
        prompt_dest = self.run_dir / "prompts" / "templates"
        prompt_dest.mkdir(parents=True, exist_ok=True)
        prompt_manifest = {"pipeline": PIPELINE, "templates": {}}
        for name, (filename, stage) in prompt_specs.items():
            source = Path(config["paths"]["prompts"]) / filename
            text, version, digest = load_prompt(source)
            self.prompts[name] = {
                "filename": filename,
                "stage": stage,
                "text": text,
                "version": version,
                "hash": digest,
            }
            target = prompt_dest / filename
            if target.exists() and target.read_text(encoding="utf-8") != text:
                raise E19Error(f"prompt changed during resume: {filename}")
            if not target.exists():
                atomic_write_text(target, text)
            prompt_manifest["templates"][name] = {
                "filename": filename,
                "stage": stage,
                "version": version,
                "sha256": digest,
                "placeholders": sorted(PLACEHOLDER_RE.findall(text)),
            }
        manifest_path = self.run_dir / "prompts" / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != prompt_manifest:
            raise E19Error("prompt manifest changed during resume")
        if not manifest_path.exists():
            atomic_write_json(manifest_path, prompt_manifest)
        self._freeze_implementation()
        if self.state_path.exists():
            self.state = read_json(self.state_path)
        else:
            self.state = {
                "schema_version": 1,
                "pipeline": PIPELINE,
                "version": VERSION,
                "run_id": config["run"]["id"],
                "completed": False,
                "non_gating": True,
                "production_state_mutation_allowed": False,
                "started_at_epoch_seconds": round(time.time(), 3),
                "call_ledger": {},
                "responses": {"dominant": {}, "blind": {}, "adjudication": {}},
            }
            atomic_write_json(self.state_path, self.state)
        self.invoker = ModelInvoker(
            config,
            self.run_dir,
            output_schemas=OUTPUT_SCHEMAS,
            output_parser=parse_stage_output,
        )

    def _freeze_implementation(self) -> None:
        names = (
            "arcgen.lib/e19_dominant_answer.py",
            "arcgen.lib/e19_schemas.py",
            "config/e19_dominant_answer_verification.yaml",
            "experiments/E19_DOMINANT_ANSWER_VERIFICATION_CALIBRATION_PLAN.md",
            "prompts/e19_dominant_answer_reviewer.md",
            "prompts/e19_dominant_answer_adjudicator.md",
            "prompts/v4_blind_solver.md",
        )
        value = {
            "pipeline": PIPELINE,
            "version": VERSION,
            "files": {name: _sha(self.root / name) for name in names},
        }
        path = self.run_dir / "implementation_manifest.json"
        if path.exists() and read_json(path) != value:
            raise E19Error("E19 implementation changed during resume")
        if not path.exists():
            atomic_write_json(path, value)

    def save(self) -> None:
        with self.lock:
            atomic_write_json(self.state_path, self.state)

    def call(
        self,
        *,
        prompt_name: str,
        call_id: str,
        values: Mapping[str, Any],
        parent_ids: Sequence[str],
        role: str,
        expected: Mapping[str, str],
    ) -> dict[str, Any]:
        entry = self.prompts[prompt_name]
        required = set(PLACEHOLDER_RE.findall(entry["text"]))
        if required != set(values):
            raise E19Error(f"{prompt_name} prompt bindings differ")
        prompt = render_prompt(entry["text"], dict(values))

        def validate(value: dict[str, Any]) -> None:
            for key, wanted in expected.items():
                if value.get(key) != wanted:
                    raise E19Error(f"{call_id} returned wrong {key}")

        settings = self.config["model"]["roles"][role]
        ledger_key = f"{role}/{entry['stage']}/{call_id}"
        with self.lock:
            self.state["call_ledger"].setdefault(
                ledger_key,
                {
                    "role": role,
                    "stage": entry["stage"],
                    "call_id": call_id,
                    "model": settings["model"],
                    "reasoning_effort": settings["reasoning_effort"],
                    "status": "reserved",
                    "substantive_experimental_call": True,
                },
            )
            self.save()
        try:
            result: CallResult = self.invoker.invoke(
                stage=entry["stage"],
                call_id=call_id,
                prompt=prompt,
                prompt_template_version=entry["version"],
                prompt_template_hash=entry["hash"],
                parent_ids=list(parent_ids),
                response_validator=validate,
                model_name=settings["model"],
                reasoning_effort=settings["reasoning_effort"],
                timeout_seconds=int(
                    self.config["experiment"].get(f"{role}_timeout_seconds")
                    or self.config["model"]["timeout_seconds"]
                ),
                retries=int(self.config["experiment"]["retries"]),
            )
        except Exception as exc:
            with self.lock:
                self.state["call_ledger"][ledger_key].update(
                    {"status": "failed", "error": str(exc)}
                )
                self.save()
            raise
        result_meta = read_json(result.log_dir / "result.json")
        with self.lock:
            self.state["call_ledger"][ledger_key].update(
                {
                    "status": "complete",
                    "attempts": result.attempts,
                    "retry_calls": max(0, result.attempts - 1),
                    "cache_source": result.cache_source,
                    "cache_key": result.cache_key,
                    "log_dir": str(result.log_dir),
                    "timing": result_meta.get("timing"),
                }
            )
            self.save()
        return result.value

    def accounting(self) -> dict[str, Any]:
        return v4_call_accounting(self.state["call_ledger"], run_dir=self.run_dir)


def _opaque(seed: int, namespace: str, task_hash: str) -> str:
    return namespace + "-" + content_hash(
        {"seed": seed, "namespace": namespace, "task_hash": task_hash}
    )[:14]


def _freeze_inputs(
    runtime: E19Runtime, replay: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> None:
    registry = Path(runtime.config["paths"]["registry"])
    reservoir = Path(runtime.config["paths"]["production_reservoir"])
    value = {
        "schema_version": 1,
        "registry": {"path": str(registry), "sha256": _sha(registry)},
        "safety_audit": {
            "path": runtime.config["paths"]["safety_audit"],
            "sha256": _sha(runtime.config["paths"]["safety_audit"]),
        },
        "historical_cases": {
            "path": runtime.config["paths"]["historical_cases"],
            "sha256": _sha(runtime.config["paths"]["historical_cases"]),
        },
        "production_reservoir": {
            "path": str(reservoir),
            "sha256": _sha(reservoir),
        },
        "replay_hash": content_hash(replay),
        "case_hashes": {item["task_hash"]: content_hash(item["task"]) for item in cases},
    }
    path = runtime.run_dir / "frozen_input_manifest.json"
    if path.exists() and read_json(path) != value:
        raise E19Error("frozen E19 input changed during resume")
    if not path.exists():
        atomic_write_json(path, value)


def _write_case_manifest(runtime: E19Runtime, cases: Sequence[Mapping[str, Any]]) -> None:
    seed = int(runtime.config["seed"])
    private_rows = []
    assignments = []
    for case in cases:
        alias = _opaque(seed, "task", case["task_hash"])
        private_rows.append(
            {
                "opaque_task_id": alias,
                "task_hash": case["task_hash"],
                "sources": deepcopy(case["sources"]),
                "accepted_task_ids": deepcopy(case["accepted_task_ids"]),
                "public_task": public_task(case["task"]),
                "proposed_outputs": proposed_outputs(case["task"]),
            }
        )
        assignments.append(
            {
                "opaque_task_id": alias,
                "dominant_reviewer_id": _opaque(seed, "reviewer", case["task_hash"]),
                "blind_solver_id": _opaque(seed, "solver", case["task_hash"]),
                "adjudicator_id": _opaque(seed, "adjudicator", case["task_hash"]),
                "arm_labels_hidden": True,
                "provenance_hidden": True,
            }
        )
    atomic_write_json(
        runtime.run_dir / "calibration_case_manifest.private.json",
        {"schema_version": 1, "cases": private_rows},
    )
    atomic_write_json(
        runtime.run_dir / "assignment_ledger.private.json",
        {"schema_version": 1, "assignments": assignments},
    )


def _run_parallel(
    specs: Sequence[Any], workers: int, function: Callable[[Any], tuple[str, Any]]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(specs)))) as executor:
        futures = {executor.submit(function, spec): spec for spec in specs}
        for future in as_completed(futures):
            try:
                key, value = future.result()
                results[key] = value
            except Exception as exc:  # Model failures remain explicit and resumable.
                spec = futures[future]
                key = str(spec.get("task_hash") if isinstance(spec, Mapping) else spec)
                failures[key] = str(exc)
    if failures:
        raise E19Error(f"model-call failures remain: {failures}")
    return results


def run_reviews(runtime: E19Runtime, cases: Sequence[Mapping[str, Any]]) -> None:
    seed = int(runtime.config["seed"])
    with runtime.lock:
        dominant_done = runtime.state["responses"]["dominant"]
        blind_done = runtime.state["responses"]["blind"]
    dominant_specs = [item for item in cases if item["task_hash"] not in dominant_done]
    blind_specs = [item for item in cases if item["task_hash"] not in blind_done]

    def dominant(case: Mapping[str, Any]) -> tuple[str, Any]:
        task_hash = case["task_hash"]
        task_id = _opaque(seed, "task", task_hash)
        reviewer_id = _opaque(seed, "reviewer", task_hash)
        response = runtime.call(
            prompt_name="dominant_review",
            call_id=f"dominant-{task_hash[:16]}",
            values={
                "TASK_ID": task_id,
                "REVIEWER_ID": reviewer_id,
                "PUBLIC_TASK": public_task(case["task"]),
                "PROPOSED_OUTPUTS": proposed_outputs(case["task"]),
            },
            parent_ids=[task_hash],
            role="dominant_reviewer",
            expected={"task_id": task_id, "reviewer_id": reviewer_id},
        )
        normalized = normalize_dominant_review(
            response,
            proposed_outputs=proposed_outputs(case["task"]),
            training_pair_count=len(case["task"]["train"]),
        )
        return task_hash, {"response": response, "normalized": normalized}

    def blind(case: Mapping[str, Any]) -> tuple[str, Any]:
        task_hash = case["task_hash"]
        task_id = _opaque(seed, "blind-task", task_hash)
        solver_id = _opaque(seed, "solver", task_hash)
        response = runtime.call(
            prompt_name="blind_solver",
            call_id=f"blind-{task_hash[:16]}",
            values={
                "PUBLIC_TASK": public_task(case["task"]),
                "TASK_ID": task_id,
                "SOLVER_ID": solver_id,
            },
            parent_ids=[task_hash],
            role="fresh_blind_solver",
            expected={"task_id": task_id, "solver_id": solver_id},
        )
        gate = corrected_blind_trigger(response, proposed_outputs(case["task"]))
        return task_hash, {"response": response, "gate": gate}

    workers = int(runtime.config["model"]["concurrency"])
    for namespace, specs, function in (
        ("dominant", dominant_specs, dominant),
        ("blind", blind_specs, blind),
    ):
        if not specs:
            continue
        values = _run_parallel(specs, workers, function)
        with runtime.lock:
            runtime.state["responses"][namespace].update(values)
            runtime.save()
    atomic_write_json(
        runtime.run_dir / "dominance_reviews.private.json",
        {"schema_version": 1, "results": runtime.state["responses"]["dominant"]},
    )
    atomic_write_json(
        runtime.run_dir / "blind_reviews.private.json",
        {"schema_version": 1, "results": runtime.state["responses"]["blind"]},
    )


def _anonymized_reviews(
    runtime: E19Runtime, task_hash: str
) -> list[dict[str, Any]]:
    rows = [
        deepcopy(runtime.state["responses"]["dominant"][task_hash]["response"]),
        deepcopy(runtime.state["responses"]["blind"][task_hash]["response"]),
    ]
    rng = Random(int(runtime.config["seed"]) + int(task_hash[:12], 16))
    rng.shuffle(rows)
    return [{"review_id": f"review_{index}", "review": row} for index, row in enumerate(rows, 1)]


def _calibration_problem(case: Mapping[str, Any]) -> bool | None:
    labels = [
        source["reference_problem"]
        for source in case["sources"]
        if source.get("reference_problem") is not None
        and source.get("reference_certainty") == "high"
    ]
    if not labels:
        return None
    if len(set(labels)) != 1:
        return None
    return bool(labels[0])


def _needs_calibration_adjudication(
    runtime: E19Runtime, case: Mapping[str, Any]
) -> bool:
    if case["accepted_task_ids"]:
        return True
    reference = _calibration_problem(case)
    if reference is None:
        return False
    dominant = runtime.state["responses"]["dominant"][case["task_hash"]]["normalized"]
    predicted_problem = dominant["normalized_verdict"] == "comparable_different_answer"
    inconclusive = dominant["normalized_verdict"] == "inconclusive"
    blind = runtime.state["responses"]["blind"][case["task_hash"]]["gate"]
    blind_problem = blind["proposed_review_required"]
    return inconclusive or predicted_problem != reference or blind_problem != reference


def run_adjudications(runtime: E19Runtime, cases: Sequence[Mapping[str, Any]]) -> None:
    seed = int(runtime.config["seed"])
    done = runtime.state["responses"]["adjudication"]
    specs = [
        case
        for case in cases
        if _needs_calibration_adjudication(runtime, case) and case["task_hash"] not in done
    ]

    def adjudicate(case: Mapping[str, Any]) -> tuple[str, Any]:
        task_hash = case["task_hash"]
        task_id = _opaque(seed, "adjudication-task", task_hash)
        adjudicator_id = _opaque(seed, "adjudicator", task_hash)
        response = runtime.call(
            prompt_name="dominant_adjudicator",
            call_id=f"adjudicate-{task_hash[:16]}",
            values={
                "TASK_ID": task_id,
                "ADJUDICATOR_ID": adjudicator_id,
                "PUBLIC_TASK": public_task(case["task"]),
                "PROPOSED_OUTPUTS": proposed_outputs(case["task"]),
                "ANONYMOUS_REVIEWS": _anonymized_reviews(runtime, task_hash),
            },
            parent_ids=[task_hash],
            role="dominant_adjudicator",
            expected={"task_id": task_id, "adjudicator_id": adjudicator_id},
        )
        normalized = normalize_adjudication(
            response,
            proposed_outputs=proposed_outputs(case["task"]),
            training_pair_count=len(case["task"]["train"]),
        )
        return task_hash, {"response": response, "normalized": normalized}

    if specs:
        values = _run_parallel(
            specs, int(runtime.config["model"]["concurrency"]), adjudicate
        )
        with runtime.lock:
            runtime.state["responses"]["adjudication"].update(values)
            runtime.save()
    atomic_write_json(
        runtime.run_dir / "adjudications.private.json",
        {"schema_version": 1, "results": runtime.state["responses"]["adjudication"]},
    )


def _resolution_rows(
    runtime: E19Runtime, cases: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    by_id = {
        task_id: case
        for case in cases
        for task_id in case["accepted_task_ids"]
    }
    for task_id in sorted(by_id):
        case = by_id[task_id]
        task_hash = case["task_hash"]
        adjudication = runtime.state["responses"]["adjudication"].get(task_hash)
        if adjudication is None:
            disposition = "re_review_incomplete"
            decision = None
        else:
            decision = adjudication["normalized"]["normalized_decision"]
            disposition = {
                "clear_dominant": "clearance_proposed",
                "repairable": "targeted_repair_required",
                "quarantine": "quarantine_required",
                "inconclusive": "quarantine_pending_further_evidence",
            }[decision]
        rows.append(
            {
                "registry_task_id": task_id,
                "task_hash": task_hash,
                "intrinsic_status": next(
                    (
                        source.get("intrinsic_status")
                        for source in case["sources"]
                        if source.get("source_id") == task_id
                    ),
                    None,
                ),
                "fresh_dominant_review": deepcopy(
                    runtime.state["responses"]["dominant"].get(task_hash)
                ),
                "fresh_blind_review": deepcopy(
                    runtime.state["responses"]["blind"].get(task_hash)
                ),
                "adjudication": deepcopy(adjudication),
                "normalized_decision": decision,
                "provisional_disposition": disposition,
                "historical_registry_mutated": False,
                "repair_attempted": False,
            }
        )
    return rows


def _metrics(runtime: E19Runtime, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for case in cases:
        reference = _calibration_problem(case)
        dominant = runtime.state["responses"]["dominant"][case["task_hash"]]["normalized"]
        blind = runtime.state["responses"]["blind"][case["task_hash"]]["gate"]
        rows.append(
            {
                "task_hash": case["task_hash"],
                "reference_problem": reference,
                "dominant_verdict": dominant["normalized_verdict"],
                "blind_review_required": blind["proposed_review_required"],
                "accepted_task_ids": case["accepted_task_ids"],
            }
        )
    labeled = [item for item in rows if item["reference_problem"] is not None]
    positive = [item for item in labeled if item["reference_problem"]]
    negative = [item for item in labeled if not item["reference_problem"]]

    def reviewer_problem(item: Mapping[str, Any]) -> bool:
        return item["dominant_verdict"] == "comparable_different_answer"

    return {
        "unique_case_count": len(cases),
        "accepted_review_required_count": sum(bool(item["accepted_task_ids"]) for item in rows),
        "high_certainty_reference_count": len(labeled),
        "reference_positive_count": len(positive),
        "reference_negative_count": len(negative),
        "dominant_reviewer": {
            "defect_recall": (
                sum(reviewer_problem(item) for item in positive) / len(positive)
                if positive
                else None
            ),
            "false_alarm_rate": (
                sum(reviewer_problem(item) for item in negative) / len(negative)
                if negative
                else None
            ),
            "inconclusive_count": sum(
                item["dominant_verdict"] == "inconclusive" for item in labeled
            ),
            "verdict_counts": dict(Counter(item["dominant_verdict"] for item in rows)),
        },
        "fresh_blind": {
            "defect_recall": (
                sum(item["blind_review_required"] for item in positive) / len(positive)
                if positive
                else None
            ),
            "false_alarm_rate": (
                sum(item["blind_review_required"] for item in negative) / len(negative)
                if negative
                else None
            ),
            "review_required_count": sum(item["blind_review_required"] for item in rows),
        },
        "rows": rows,
    }


def _non_mutation(runtime: E19Runtime) -> dict[str, Any]:
    frozen = read_json(runtime.run_dir / "frozen_input_manifest.json")
    registry = Path(runtime.config["paths"]["registry"])
    reservoir = Path(runtime.config["paths"]["production_reservoir"])
    current = {"registry_sha256": _sha(registry), "reservoir_sha256": _sha(reservoir)}
    expected = {
        "registry_sha256": frozen["registry"]["sha256"],
        "reservoir_sha256": frozen["production_reservoir"]["sha256"],
    }
    value = {
        "schema_version": 1,
        "expected": expected,
        "current": current,
        "unchanged": current == expected,
        "historical_dossiers_mutated": False,
        "production_prompts_mutated": False,
    }
    if not value["unchanged"]:
        raise E19Error("production registry or reservoir changed during E19")
    atomic_write_json(runtime.run_dir / "non_mutation_attestation.json", value)
    return value


def _markdown(report: Mapping[str, Any]) -> str:
    replay = report["historical_replay"]
    metrics = report["calibration_metrics"]
    resolutions = report["nine_task_resolution_ledger"]["rows"]
    lines = [
        "# E19 dominant-answer verification calibration",
        "",
        "## Outcome",
        "",
        f"The zero-call replay examined {replay['task_count']} accepted dossiers and "
        f"routed {replay['proposed_review_required_count']} ({replay['simulated_conditional_review_rate']:.2%}) "
        "to conditional dominance review.",
        "",
        "## Calibration",
        "",
        f"- Unique reviewed episodes: {metrics['unique_case_count']}",
        f"- High-certainty historical references: {metrics['high_certainty_reference_count']}",
        f"- Dominant-review defect recall: {metrics['dominant_reviewer']['defect_recall']}",
        f"- Dominant-review false-alarm rate: {metrics['dominant_reviewer']['false_alarm_rate']}",
        f"- Fresh-blind defect recall: {metrics['fresh_blind']['defect_recall']}",
        f"- Fresh-blind false-alarm rate: {metrics['fresh_blind']['false_alarm_rate']}",
        "",
        "## Accepted-bank resolutions",
        "",
        "| Task | Decision | Provisional disposition |",
        "|---|---|---|",
    ]
    for row in resolutions:
        lines.append(
            f"| `{row['registry_task_id']}` | `{row['normalized_decision']}` | "
            f"`{row['provisional_disposition']}` |"
        )
    lines.extend(
        [
            "",
            "No production status or historical artifact was changed by this calibration run.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    runtime = E19Runtime(config)
    replay_path = runtime.run_dir / "historical_replay.json"
    replay = read_json(replay_path) if replay_path.exists() else replay_committed_bank(config)
    if not replay_path.exists():
        atomic_write_json(replay_path, replay)
    cases = build_calibration_cases(config)
    _freeze_inputs(runtime, replay, cases)
    _write_case_manifest(runtime, cases)
    config_snapshot = deepcopy(config)
    atomic_write_json(runtime.run_dir / "config.snapshot.json", config_snapshot)
    run_reviews(runtime, cases)
    run_adjudications(runtime, cases)
    resolutions = {"schema_version": 1, "rows": _resolution_rows(runtime, cases)}
    atomic_write_json(runtime.run_dir / "nine_task_resolution_ledger.json", resolutions)
    metrics = _metrics(runtime, cases)
    accounting = runtime.accounting()
    atomic_write_json(runtime.run_dir / "call_and_timing_ledger.json", accounting)
    non_mutation = _non_mutation(runtime)
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "completed": True,
        "non_gating": True,
        "production_promotion_occurred": False,
        "historical_replay": replay,
        "calibration_metrics": metrics,
        "nine_task_resolution_ledger": resolutions,
        "call_accounting": accounting,
        "non_mutation_attestation": non_mutation,
        "repair_phase_complete": False,
        "recommendation": "pending_repair_and_final_resolution",
    }
    with runtime.lock:
        runtime.state["completed"] = True
        runtime.state["completed_at_epoch_seconds"] = round(time.time(), 3)
        runtime.state["results"] = report
        runtime.save()
    atomic_write_json(runtime.run_dir / "report.json", report)
    atomic_write_text(runtime.run_dir / "report.md", _markdown(report))
    return report


def replay_only(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    runtime = E19Runtime(config)
    replay = replay_committed_bank(config)
    atomic_write_json(runtime.run_dir / "historical_replay.json", replay)
    cases = build_calibration_cases(config)
    _freeze_inputs(runtime, replay, cases)
    _write_case_manifest(runtime, cases)
    return replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("replay", "run"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    value = replay_only(args.config) if args.command == "replay" else run(args.config)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "E19Error",
    "PIPELINE",
    "VERSION",
    "build_calibration_cases",
    "corrected_blind_trigger",
    "load_config",
    "replay_committed_bank",
    "run",
    "terminal_blind_responses",
]
