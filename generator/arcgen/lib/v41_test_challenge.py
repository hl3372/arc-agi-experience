"""Single-instance, test-only ARCGEN V4.1 challenge experiment.

Unlike the first supported-departure study, this design never authors two
episodes from one kernel.  It freezes one already-realized ordinary episode,
keeps every training grid and the intended rule byte-for-byte equivalent, and
allows one Sol/max authoring call to replace only the test pair.  All later
quality checks remain ordinary V4 checks and receive no intervention label.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence
import json
import time

from jsonschema import Draft202012Validator

from .lean_v3_pipeline import classify_v3_editorial_results
from .lean_v3_rendering import build_contact_sheet, render_episode
from .schemas import SchemaError, parse_json_object
from .util import (
    PLACEHOLDER_RE,
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    parallel_map,
    read_json,
)
from .v4_diagnostics import diagnose_arc_task_v4, diagnose_public_arc_task_v4
from .v4_schemas import CANVAS_PAIR, GRID, TEST_STORYBOARD
from .validation import validate_arc_task
from .v41_experiments import V41Runtime, _anchors, _anonymous_id, _portfolio_digest, _task_metrics
from .v41_integration import _assert_hashes_unchanged, _protected_hashes
from .v41_supported_departure import (
    SPECIAL_LABEL_FRAGMENTS,
    SlateEscalationBudget,
    _record_stage,
    _run_blind,
    _run_technical,
)


class TestChallengeError(ValueError):
    pass


TEXT = {"type": "string", "minLength": 1}
MAYBE_TEXT = {"type": "string"}
BOOLEAN = {"type": "boolean"}


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": deepcopy(dict(properties)),
        "required": list(properties),
        "additionalProperties": False,
    }


CHALLENGE_CONTRACT = _object(
    {
        "small_setup_shift": TEXT,
        "why_more_demanding": TEXT,
        "same_rule_invariant": TEXT,
        "training_support": TEXT,
        "shallow_expectation_defeated": TEXT,
        "no_new_semantics_check": TEXT,
        "one_answer_support": TEXT,
    }
)
REVISED_TEST = _object(
    {
        "pair_number": {"type": "integer", "minimum": 1, "maximum": 3},
        "input": deepcopy(GRID),
        "expected_output": deepcopy(GRID),
        "storyboard": deepcopy(TEST_STORYBOARD),
        "canvas_plan": deepcopy(CANVAS_PAIR),
        "construction_witness": TEXT,
    }
)
TEST_REFINEMENT_SCHEMA = _object(
    {
        "candidate_id": TEXT,
        "decision": {"type": "string", "enum": ["use", "decline"]},
        "decision_reason": TEXT,
        "challenge_contract": {
            "anyOf": [deepcopy(CHALLENGE_CONTRACT), {"type": "null"}]
        },
        "revised_tests": {
            "anyOf": [
                {
                    "type": "array",
                    "items": deepcopy(REVISED_TEST),
                    "minItems": 1,
                    "maxItems": 3,
                },
                {"type": "null"},
            ]
        },
    }
)

AUDIT_JUDGMENT = _object(
    {
        "task_id": TEXT,
        "status": {
            "type": "string",
            "enum": [
                "meaningfully_harder_same_rule",
                "comparable",
                "merely_larger_or_cluttered",
                "unsupported",
            ],
        },
        "setup_shift": MAYBE_TEXT,
        "same_rule": BOOLEAN,
        "training_support": TEXT,
        "shallow_strategy_defeated": MAYBE_TEXT,
        "one_answer_supported": BOOLEAN,
        "reason": TEXT,
        "open_ended_signature": MAYBE_TEXT,
    }
)
TEST_AUDIT_SCHEMA = _object(
    {
        "audit_id": TEXT,
        "judgments": {
            "type": "array",
            "items": deepcopy(AUDIT_JUDGMENT),
            "minItems": 1,
            "maxItems": 16,
        },
        "non_gating_confirmation": BOOLEAN,
    }
)


def _transport(value: Any) -> Any:
    unsupported = {
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
    if isinstance(value, dict):
        return {key: _transport(item) for key, item in value.items() if key not in unsupported}
    if isinstance(value, list):
        return [_transport(item) for item in value]
    return value


STRICT_EXTENSION_SCHEMAS = {
    "v41_test_challenge_refinement": TEST_REFINEMENT_SCHEMA,
    "v41_test_challenge_audit": TEST_AUDIT_SCHEMA,
}


def _parse_extension(stage: str, data: str | Mapping[str, Any]) -> dict[str, Any]:
    value = parse_json_object(data) if isinstance(data, str) else dict(data)
    errors = sorted(
        Draft202012Validator(STRICT_EXTENSION_SCHEMAS[stage]).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{item}]" if isinstance(item, int) else f".{item}"
            for item in error.absolute_path
        )
        raise SchemaError(path, error.message)
    if stage == "v41_test_challenge_refinement":
        used = value["decision"] == "use"
        if (value["revised_tests"] is not None) != used:
            raise SchemaError("$.revised_tests", "revised tests are required exactly when used")
        if (value["challenge_contract"] is not None) != used:
            raise SchemaError("$.challenge_contract", "contract is required exactly when used")
    elif value["non_gating_confirmation"] is not True:
        raise SchemaError("$.non_gating_confirmation", "audit must confirm non-gating status")
    return value


def _install_extensions(runtime: V41Runtime) -> None:
    root = Path(runtime.config["_repository_root"])
    prompt_specs = {
        "test_challenge_refiner": (
            "v4_1_test_challenge_refiner.md",
            "v41_test_challenge_refinement",
        ),
        "test_challenge_auditor": (
            "v4_1_test_challenge_auditor.md",
            "v41_test_challenge_audit",
        ),
    }
    manifest = {"schema_version": 1, "prompts": {}}
    destination = runtime.run_dir / "prompts" / "templates"
    for name, (filename, stage) in prompt_specs.items():
        path = root / "prompts" / filename
        text, version, digest = load_prompt(path)
        required = set(PLACEHOLDER_RE.findall(text))
        runtime.prompts.entries[name] = {
            "filename": filename,
            "text": text,
            "version": version,
            "hash": digest,
            "stage": stage,
        }
        target = destination / filename
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise TestChallengeError(f"extension prompt changed during resume: {filename}")
        if not target.exists():
            atomic_write_text(target, text)
        manifest["prompts"][name] = {
            "filename": filename,
            "version": version,
            "sha256": digest,
            "stage": stage,
            "placeholder_names": sorted(required),
        }
    manifest["implementation"] = {
        "file": "arcgen.lib/v41_test_challenge.py",
        "sha256": sha256((root / "arcgen.lib" / "v41_test_challenge.py").read_bytes()).hexdigest(),
    }
    manifest_path = runtime.run_dir / "test_challenge_extension_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise TestChallengeError("test-challenge extension changed during resume")
    if not manifest_path.exists():
        atomic_write_json(manifest_path, manifest)

    original_parser = runtime.invoker.output_parser

    def parse(stage: str, data: Any) -> Any:
        if stage in STRICT_EXTENSION_SCHEMAS:
            return _parse_extension(stage, data)
        return original_parser(stage, data)

    runtime.invoker.output_parser = parse
    runtime.invoker.output_schemas.update(
        {stage: _transport(schema) for stage, schema in STRICT_EXTENSION_SCHEMAS.items()}
    )
    runtime.invoker._write_schemas()


def _source_records(config: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    settings = config.get("test_challenge")
    if not isinstance(settings, Mapping):
        raise TestChallengeError("test_challenge configuration is missing")
    root = Path(config["_repository_root"])
    source_run = (root / str(settings["source_run"])).resolve()
    state = read_json(source_run / "state.json")
    records = state.get("semantic_records")
    if not isinstance(records, Mapping):
        raise TestChallengeError("source run has no semantic records")
    arm = str(settings.get("source_arm", "control"))
    rows = [
        deepcopy(record)
        for record in records.values()
        if record.get("arm") == arm
        and record.get("complete_task")
        and isinstance(record.get("concrete_plan"), Mapping)
    ]
    rows.sort(
        key=lambda record: (
            int(record.get("slate_index", 99)),
            int(record.get("source_development_priority", 99)),
            str(record["task_id"]),
        )
    )
    maximum = int(settings.get("maximum_candidates", len(rows)))
    return source_run, rows[:maximum]


def _aliases(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    rng = Random(seed + 44101)
    used: set[str] = set()
    keys = [str(row["task_id"]) for row in rows]
    rng.shuffle(keys)
    return {key: _anonymous_id(rng, used) for key in keys}


def _source_manifest(
    config: Mapping[str, Any], source_run: Path, rows: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str], protected: Mapping[str, Any],
) -> dict[str, Any]:
    source_files = {}
    for relative in ("state.json", "report.json", "config.snapshot.json", "frozen_control_manifest.json"):
        path = source_run / relative
        if path.is_file():
            source_files[relative] = sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "feature": "supported_departure",
        "experimental_design": "test_only_single_instance_v2",
        "source_run": str(source_run),
        "source_file_hashes": source_files,
        "selected_source_count": len(rows),
        "sources": [
            {
                "source_task_id": row["task_id"],
                "opaque_task_id": aliases[str(row["task_id"])],
                "slate_index": row.get("slate_index"),
                "search_mode": row.get("search_mode"),
                "training_hash": content_hash(row["complete_task"]["train"]),
                "intended_rule_hash": content_hash(
                    row["realization"]["authoring_metadata"]["intended_rule"]
                ),
                "base_plan_hash": content_hash(row["concrete_plan"]),
                "base_task_hash": content_hash(row["complete_task"]),
            }
            for row in rows
        ],
        "single_instance_only": True,
        "control_episode_is_not_reauthored": True,
        "arm_labels_hidden_downstream": True,
        "protected_production_state": deepcopy(dict(protected)),
        "feature_manifest_snapshot": deepcopy(config["_feature_manifest_snapshot"]),
        "model_and_effort_settings": deepcopy(config["model"]["roles"]),
    }


def _initialize_records(
    runtime: V41Runtime, rows: Sequence[Mapping[str, Any]], aliases: Mapping[str, str]
) -> None:
    with runtime.lock:
        if "semantic_records" in runtime.state:
            return
        records = {}
        for source in rows:
            source_id = str(source["task_id"])
            task_id = aliases[source_id]
            records[task_id] = {
                "task_id": task_id,
                "source_candidate_id": source_id,
                "arm": "single",
                "slate_index": int(source.get("slate_index", 1)),
                "source_development_priority": int(source.get("source_development_priority", 99)),
                "search_mode": source["search_mode"],
                "kernel": deepcopy(source["kernel"]),
                "editorial_brief": deepcopy(source["editorial_brief"]),
                "base_concrete_plan": deepcopy(source["concrete_plan"]),
                "base_complete_task": deepcopy(source["complete_task"]),
                "base_realization": deepcopy(source["realization"]),
                "base_downstream_status": source.get("status"),
                "base_intrinsic_quality": source.get("intrinsic_quality"),
                "base_portfolio_placement": source.get("portfolio_placement"),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
        runtime.state["semantic_records"] = records
        runtime.save()


def _validate_refinement(
    task_id: str, base: Mapping[str, Any], value: Mapping[str, Any]
) -> None:
    if value["candidate_id"] != task_id:
        raise TestChallengeError("refiner returned a mismatched task ID")
    if value["decision"] == "decline":
        return
    revised = value["revised_tests"]
    assert isinstance(revised, list)
    original_tests = base["base_complete_task"]["test"]
    if len(revised) != len(original_tests):
        raise TestChallengeError("test refinement changed test-pair count")
    expected_numbers = list(range(1, len(revised) + 1))
    numbers = [item["pair_number"] for item in revised]
    if numbers != expected_numbers:
        raise TestChallengeError("revised test pair numbers are not consecutive")
    changed = False
    for item, original in zip(revised, original_tests):
        storyboard = item["storyboard"]
        if storyboard["pair_number"] != item["pair_number"]:
            raise TestChallengeError("storyboard pair number mismatch")
        if [len(item["input"]), len(item["input"][0])] != [
            storyboard["input_dimensions"]["height"],
            storyboard["input_dimensions"]["width"],
        ]:
            raise TestChallengeError("revised test input dimensions disagree with storyboard")
        if [len(item["expected_output"]), len(item["expected_output"][0])] != [
            storyboard["output_dimensions"]["height"],
            storyboard["output_dimensions"]["width"],
        ]:
            raise TestChallengeError("revised test output dimensions disagree with storyboard")
        changed = changed or item["input"] != original["input"] or item["expected_output"] != original["output"]
    if not changed:
        raise TestChallengeError("a used refinement did not change any test cell")


def _ordinary_plan(
    task_id: str, base_plan: Mapping[str, Any], refinement: Mapping[str, Any]
) -> dict[str, Any]:
    plan = deepcopy(dict(base_plan))
    plan["candidate_id"] = task_id
    if refinement["decision"] == "decline":
        return plan
    revised = refinement["revised_tests"]
    contract = refinement["challenge_contract"]
    assert isinstance(revised, list) and isinstance(contract, Mapping)
    plan["test_pairs"] = [deepcopy(item["storyboard"]) for item in revised]
    plan["canvas_plan"] = [
        deepcopy(item)
        for item in plan.get("canvas_plan", [])
        if not str(item.get("pair", "")).casefold().startswith("test")
    ] + [deepcopy(item["canvas_plan"]) for item in revised]
    overview = deepcopy(plan["episode_overview"])
    overview["intended_difficulty_and_inferability"] = (
        "The training episode is unchanged. The test asks for a more demanding "
        "application of the same rule through a small setup variation: "
        + str(contract["why_more_demanding"])
    )
    plan["episode_overview"] = overview
    audit = deepcopy(plan["cross_pair_concrete_audit"])
    audit["remaining_ambiguity_risk"] = (
        str(contract["one_answer_support"])
        + " No new-semantics audit: "
        + str(contract["no_new_semantics_check"])
    )
    plan["cross_pair_concrete_audit"] = audit
    plan["frozen_decisions"] = [
        "The intended rule, central insight, functional color roles, pair count, and every training storyboard and grid are frozen unchanged.",
        "Only the test construction is revised; it must use the same semantic roles, operations, and rule clauses established in training.",
        "The revised test input and expected output are exactly those specified in the test storyboard.",
        "The test must remain supported by one best answer without an arbitrary exception or test-only semantic clause.",
    ]
    plan["allowed_realizer_freedom"] = []
    plan["deterministic_realization_procedure"] = (
        "Training grids are copied exactly from the frozen ordinary episode. "
        "The test grids are copied exactly from the revised test construction."
    )
    return plan


def _realization_from_refinement(
    task_id: str, record: Mapping[str, Any], refinement: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_task = deepcopy(record["base_complete_task"])
    base_realization = deepcopy(record["base_realization"])
    plan = _ordinary_plan(task_id, record["base_concrete_plan"], refinement)
    if refinement["decision"] == "use":
        revised = refinement["revised_tests"]
        assert isinstance(revised, list)
        base_task["test"] = [
            {"input": deepcopy(item["input"]), "output": deepcopy(item["expected_output"])}
            for item in revised
        ]
    public = {
        "train": deepcopy(base_task["train"]),
        "test": [{"input": deepcopy(pair["input"])} for pair in base_task["test"]],
    }
    realization = base_realization
    realization["candidate_id"] = task_id
    realization["task"] = public
    metadata = realization["authoring_metadata"]
    metadata["expected_test_outputs"] = [deepcopy(pair["output"]) for pair in base_task["test"]]
    metadata["intended_rule"] = record["base_realization"]["authoring_metadata"]["intended_rule"]
    if refinement["decision"] == "use":
        witnesses = [item["construction_witness"] for item in refinement["revised_tests"]]
        metadata["construction_witness"] = (
            metadata.get("construction_witness", "")
            + "\nTest-only refinement witness: "
            + " | ".join(witnesses)
        )
        metadata["pair_construction_trace"] = [
            item
            for item in metadata.get("pair_construction_trace", [])
            if not str(item.get("pair", "")).casefold().startswith("test")
        ] + [
            {"pair": f"test_{item['pair_number']}", "trace": item["construction_witness"]}
            for item in refinement["revised_tests"]
        ]
    metadata["plan_fidelity"] = {
        "statement": "The intended rule and every training grid are frozen; only the test was optionally refined.",
        "preserved_frozen_decisions": deepcopy(plan["frozen_decisions"]),
        "allowed_freedoms_used": [],
        "deviations": [],
    }
    realization["deviations"] = []
    fidelity = {
        "training_grids_byte_identical": base_task["train"] == record["base_complete_task"]["train"],
        "intended_rule_identical": metadata["intended_rule"] == record["base_realization"]["authoring_metadata"]["intended_rule"],
        "training_pair_count_identical": len(base_task["train"]) == len(record["base_complete_task"]["train"]),
        "test_pair_count_identical": len(base_task["test"]) == len(record["base_complete_task"]["test"]),
        "only_test_may_change": True,
    }
    if not all(fidelity.values()):
        raise TestChallengeError(f"programmatic frozen-episode fidelity failed for {task_id}")
    return plan, realization, fidelity


def _refine(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]

    def one(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        record = deepcopy(records[task_id])
        intended_rule = record["base_realization"]["authoring_metadata"]["intended_rule"]
        try:
            response = runtime.call(
                prompt_name="test_challenge_refiner",
                call_id=f"{task_id}-test-refinement",
                values={
                    "CANDIDATE_ID": task_id,
                    "INTENDED_RULE": intended_rule,
                    "CONCRETE_PLAN": record["base_concrete_plan"],
                    "COMPLETE_EPISODE": record["base_complete_task"],
                },
                parent_ids=[task_id],
                role="concrete_planner",
                arm_id="private-test-authoring",
                expected={"candidate_id": task_id},
                validator=lambda value: _validate_refinement(task_id, record, value),
            )
            return task_id, response, None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [task_id for task_id, record in records.items() if "test_refinement" not in record]
    results = parallel_map(one, pending, min(len(pending), int(runtime.config["model"]["concurrency"]))) if pending else []
    for task_id, response, error in results:
        if error or response is None:
            _record_stage(runtime, task_id, "test_refinement", status="operational_failure", refinement_error=error)
            continue
        try:
            record = deepcopy(runtime.state["semantic_records"][task_id])
            plan, realization, fidelity = _realization_from_refinement(task_id, record, response)
            complete = {
                "train": deepcopy(realization["task"]["train"]),
                "test": [
                    {"input": deepcopy(pair["input"]), "output": deepcopy(output)}
                    for pair, output in zip(
                        realization["task"]["test"],
                        realization["authoring_metadata"]["expected_test_outputs"],
                    )
                ],
            }
            validation = validate_arc_task(complete, require_test_outputs=True)
            if not validation.valid:
                raise TestChallengeError("; ".join(validation.errors))
            diagnostics = diagnose_arc_task_v4(complete)
            if not diagnostics["hard_valid"]:
                raise TestChallengeError("; ".join(diagnostics.get("errors", [])))
            public_diagnostics = diagnose_public_arc_task_v4(realization["task"])
            _record_stage(
                runtime,
                task_id,
                "test_refinement",
                status="realized",
                test_refinement=response,
                concrete_plan=plan,
                realization=realization,
                complete_task=complete,
                deterministic_diagnostics=diagnostics,
                technical_diagnostics=public_diagnostics,
                plan_fidelity=fidelity,
            )
            destination = runtime.run_dir / "tasks" / "single"
            destination.mkdir(parents=True, exist_ok=True)
            atomic_write_json(destination / f"{task_id}.arc.json", complete)
            atomic_write_json(
                destination / f"{task_id}.authoring.json",
                {
                    "base_source_candidate_id": record["source_candidate_id"],
                    "test_refinement": response,
                    "concrete_plan": plan,
                    "realization": realization,
                    "programmatic_fidelity": fidelity,
                },
            )
        except Exception as exc:
            _record_stage(runtime, task_id, "test_refinement", status="realization_failure", refinement_error=str(exc))


def _board(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]
    eligible = sorted(
        [task_id for task_id, record in records.items() if record.get("status") in {"blind_pass", "editorial_complete"}],
        key=lambda task_id: content_hash({"seed": runtime.config["seed"], "task": task_id}),
    )
    anchors = _anchors(runtime.config)
    portfolio = _portfolio_digest(runtime.config)
    all_results = []
    for group_index in range(0, len(eligible), 6):
        group = eligible[group_index : group_index + 6]
        ordinal = group_index // 6 + 1

        def call(board_index: int) -> dict[str, Any]:
            rng = Random(int(runtime.config["seed"]) + ordinal * 131 + board_index * 19)
            used: set[str] = set()
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
            mapping = {}
            package = []
            for source, rule, episode in rows:
                opaque = _anonymous_id(rng, used)
                mapping[opaque] = source
                package.append({"task_id": opaque, "rule_summary": rule, "episode": episode})
            board_id = "board-" + content_hash({"group": ordinal, "board": board_index, "ids": sorted(mapping)})[:14]

            def validate(value: dict[str, Any]) -> None:
                returned = [item["task_id"] for item in value["judgments"]]
                if len(returned) != len(set(returned)) or set(returned) != set(mapping):
                    raise TestChallengeError("board did not judge every opaque task")

            response = runtime.call(
                prompt_name="editorial_board",
                call_id=board_id,
                values={"BLINDED_TASKS": package, "PORTFOLIO_DIGEST": portfolio, "BOARD_ID": board_id},
                parent_ids=list(group),
                role="editorial_board",
                arm_id="blinded-single-editorial",
                expected={"board_id": board_id},
                validator=validate,
                forbidden_prompt_fragments=(*SPECIAL_LABEL_FRAGMENTS, "test challenge", "test-only refinement"),
            )
            return {"group_index": ordinal, "board_index": board_index, "group": group, "mapping": mapping, "response": response}

        results = parallel_map(call, [1, 2], min(2, int(runtime.config["model"]["concurrency"])))
        all_results.extend(results)
        for task_id in group:
            judgments = []
            for result in results:
                opaque = next(key for key, source in result["mapping"].items() if source == f"generated:{task_id}")
                judgment = next(item for item in result["response"]["judgments"] if item["task_id"] == opaque)
                judgments.append({**deepcopy(judgment), "task_id": task_id})
            intrinsic, placement, intrinsic_reason, placement_reason = classify_v3_editorial_results(judgments)
            _record_stage(
                runtime,
                task_id,
                "editorial",
                status="editorial_complete",
                editorial_judgments=judgments,
                intrinsic_quality=intrinsic,
                portfolio_placement=placement,
                editorial_reason=intrinsic_reason + "; " + placement_reason,
            )
    atomic_write_json(runtime.run_dir / "editorial_board_results.json", all_results)


def _audit(runtime: V41Runtime) -> list[dict[str, Any]]:
    records = runtime.state["semantic_records"]
    completed = [record for record in records.values() if record.get("complete_task")]
    rng = Random(int(runtime.config["seed"]) + 8721)
    rng.shuffle(completed)
    judgments = []
    calls = []
    for offset in range(0, len(completed), 16):
        chunk = completed[offset : offset + 16]
        audit_id = f"test-challenge-audit-{offset // 16 + 1:02d}"
        packages = [
            {
                "task_id": record["task_id"],
                "episode": record["complete_task"],
                "concise_rule": record["realization"]["authoring_metadata"]["intended_rule"],
                "concrete_plan": record["concrete_plan"],
            }
            for record in chunk
        ]

        def validate(value: dict[str, Any]) -> None:
            ids = [item["task_id"] for item in value["judgments"]]
            if len(ids) != len(set(ids)) or set(ids) != {item["task_id"] for item in packages}:
                raise TestChallengeError("auditor did not cover every task")

        response = runtime.call(
            prompt_name="test_challenge_auditor",
            call_id=audit_id,
            values={"AUDIT_ID": audit_id, "TASK_PACKAGES": packages},
            parent_ids=[item["task_id"] for item in packages],
            role="editorial_board",
            arm_id="blinded-test-audit",
            expected={"audit_id": audit_id},
            validator=validate,
            forbidden_prompt_fragments=("private-test-authoring", "experimental arm", "treatment arm"),
        )
        calls.append(response)
        judgments.extend(response["judgments"])
    atomic_write_json(
        runtime.run_dir / "test_challenge_audit.json",
        {"arm_hidden": True, "non_gating": True, "calls": calls, "judgments": judgments},
    )
    return judgments


def _human_pack(runtime: V41Runtime) -> dict[str, Any]:
    destination = runtime.run_dir / "human_review"
    tasks_dir = destination / "tasks"
    images_dir = destination / "images"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    records = [record for record in runtime.state["semantic_records"].values() if record.get("complete_task")]
    rng = Random(int(runtime.config["seed"]) + 9931)
    rng.shuffle(records)
    used: set[str] = set()
    public = []
    key = []
    rendered = []
    review_form = []
    for index, record in enumerate(records, 1):
        anonymous = _anonymous_id(rng, used)
        atomic_write_json(tasks_dir / f"{anonymous}.json", record["complete_task"])
        image = render_episode(
            record["complete_task"],
            title=f"Task {index}",
            subtitle="Source and intended rule withheld",
            destination=images_dir / f"{anonymous}.png",
        )
        rendered.append((f"Task {index}", image))
        public.append({"review_index": index, "anonymous_id": anonymous, "task_json": f"tasks/{anonymous}.json", "image": f"images/{anonymous}.png"})
        key.append(
            {
                "review_index": index,
                "anonymous_id": anonymous,
                "task_id": record["task_id"],
                "source_candidate_id": record["source_candidate_id"],
                "refinement_decision": record["test_refinement"]["decision"],
                "intended_rule": record["realization"]["authoring_metadata"]["intended_rule"],
            }
        )
        review_form.append(
            {
                "review_index": index,
                "overall_preference_quality": None,
                "clear_conceptual_click": None,
                "fair_and_one_answer": None,
                "test_is_meaningfully_more_demanding_than_training": None,
                "test_shift_improves_or_distracts": None,
                "overcomplicated": None,
                "notes": "",
            }
        )
    contact = build_contact_sheet(rendered, destination / "review_sheet.png")
    atomic_write_json(destination / "manifest.json", {"single_instance_only": True, "source_hidden": True, "rule_withheld": True, "task_count": len(public), "contact_sheet": "review_sheet.png", "tasks": public})
    atomic_write_json(destination / "review_form.json", review_form)
    atomic_write_json(destination / "DO_NOT_OPEN_FIRST.answer_key.json", key)
    appendix = ["# Answer and rule appendix", "", "Open only after reviewing the single-instance task sheet.", ""]
    for item in key:
        appendix.extend([f"## Task {item['review_index']}", "", f"- Task ID: `{item['task_id']}`", f"- Refinement decision: `{item['refinement_decision']}`", f"- Rule: {item['intended_rule']}", ""])
    atomic_write_text(destination / "answer_rule_appendix.md", "\n".join(appendix))
    return {"directory": str(destination), "task_count": len(public), "single_instance_only": True, "contact_sheet": str(contact) if contact else None, "human_results_available": False}


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    realized = [row for row in records if row.get("complete_task")]
    technical = [row for row in records if row.get("status") in {"technical_pass", "blind_pass", "editorial_complete", "ambiguous"}]
    blind = [row for row in records if row.get("blind_results")]
    core = sum(row.get("intrinsic_quality") == "core_quality" for row in records)
    reserve = sum(row.get("intrinsic_quality") == "reserve_quality" for row in records)
    return {
        "candidate_count": count,
        "realized_count": len(realized),
        "deterministic_validity_rate": len(realized) / count if count else 0.0,
        "technical_pass_count": len(technical),
        "technical_pass_rate": len(technical) / count if count else 0.0,
        "blind_completed_count": len(blind),
        "serious_ambiguity_count": sum(row.get("status") == "ambiguous" for row in records),
        "serious_ambiguity_rate": sum(row.get("status") == "ambiguous" for row in records) / len(blind) if blind else 0.0,
        "core_count": core,
        "reserve_count": reserve,
        "core_yield": core / count if count else 0.0,
        "core_or_reserve_yield": (core + reserve) / count if count else 0.0,
    }


def _base_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    core = sum(row.get("base_intrinsic_quality") == "core_quality" for row in records)
    reserve = sum(row.get("base_intrinsic_quality") == "reserve_quality" for row in records)
    ambiguous = sum(row.get("base_downstream_status") == "ambiguous" for row in records)
    return {
        "candidate_count": count,
        "core_count": core,
        "reserve_count": reserve,
        "core_yield": core / count if count else 0.0,
        "core_or_reserve_yield": (core + reserve) / count if count else 0.0,
        "serious_ambiguity_count": ambiguous,
        "comparison_kind": "descriptive frozen-source aggregate; no second task instance was authored",
    }


def enrich_test_challenge_report(
    report: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    """Add concise cross-cutting analysis without changing any task decision."""

    value = deepcopy(dict(report))
    tasks = value.get("tasks", [])
    decision_audit: Counter[tuple[str, str]] = Counter()
    search_modes: Counter[str] = Counter()
    technical_failures = []
    ambiguity_holds = []
    for task in tasks:
        decision = str(task.get("refinement_decision") or "failed")
        audit_status = str((task.get("audit") or {}).get("status") or "not_audited")
        decision_audit[(decision, audit_status)] += 1
        search_modes[str(task.get("search_mode") or "unknown")] += 1
        record = task.get("artifact_record") or {}
        status = record.get("status")
        if status == "rejected_technical":
            result = (record.get("technical_results") or [{}])[-1]
            response = result.get("response") or {}
            technical_failures.append(
                {
                    "task_id": task["task_id"],
                    "refinement_decision": decision,
                    "verifier_status": response.get("status"),
                    "mechanically_correct": response.get("mechanically_correct"),
                    "hidden_output_match": result.get("hidden_output_match"),
                    "repair_instruction": response.get("repair_instruction", ""),
                    "semantic_grid_change_required": False
                    if response.get("mechanically_correct") is True
                    and result.get("hidden_output_match") is True
                    else None,
                }
            )
        if status == "ambiguous":
            blind = record.get("blind_results") or []
            ambiguity_holds.append(
                {
                    "task_id": task["task_id"],
                    "refinement_decision": decision,
                    "blind_opinion_count": len(blind),
                    "serious_alternative_by_opinion": [
                        item.get("serious_different_output_alternative") for item in blind
                    ],
                    "reason": record.get("ambiguity_reason", ""),
                    "adjudication_run": bool(record.get("ambiguity_adjudication")),
                }
            )

    used = sum(count for (decision, _), count in decision_audit.items() if decision == "use")
    used_genuine = decision_audit[("use", "meaningfully_harder_same_rule")]
    value.setdefault("efficacy", {}).update(
        {
            "decision_by_audit_status": {
                decision: {
                    status: decision_audit[(decision, status)]
                    for status in sorted({key[1] for key in decision_audit if key[0] == decision})
                }
                for decision in sorted({key[0] for key in decision_audit})
            },
            "used_refinement_count": used,
            "used_refinement_meaningfully_harder_count": used_genuine,
            "used_refinement_efficacy_rate": used_genuine / used if used else None,
            "unsupported_audit_count": sum(
                count for (_, status), count in decision_audit.items() if status == "unsupported"
            ),
        }
    )
    value["diversity"] = {
        "genuine_revised_instance_count": used_genuine,
        "minimum_before_mechanism_clustering": 8,
        "cluster_audit_run": False,
        "conclusion": "inconclusive: too few genuine revised instances for a responsible mechanism-diversity claim",
    }
    value["coverage_retention"] = {
        "source_search_mode_counts": dict(sorted(search_modes.items())),
        "all_training_grids_preserved": value.get("fidelity", {}).get(
            "all_training_grids_identical"
        ),
        "all_intended_rules_preserved": value.get("fidelity", {}).get(
            "all_intended_rules_identical"
        ),
        "ordinary_source_concepts_suppressed": 0,
        "paired_task_variants_authored": False,
    }
    value["failure_mode_analysis"] = {
        "technical_rejections": technical_failures,
        "ambiguity_holds": ambiguity_holds,
        "operational_or_parse_failures": value.get("funnel", {}).get(
            "operational_or_validation_failures", 0
        ),
        "unsupported_test_audit_count": value["efficacy"]["unsupported_audit_count"],
        "important_interpretation": (
            "A metadata-only verifier failure and an unadjudicated 1-versus-1 blind "
            "disagreement are reported separately from confirmed semantic invalidity."
        ),
    }
    started = state.get("started_at_epoch_seconds")
    ended = state.get("completed_at_epoch_seconds")
    if isinstance(started, (int, float)):
        if not isinstance(ended, (int, float)):
            ended = time.time()
        value["timing"] = {
            "wall_clock_minutes": round((float(ended) - float(started)) / 60.0, 3),
            "active_model_minutes": value.get("call_accounting", {}).get(
                "active_model_minutes"
            ),
            "measured_peak_concurrent_calls": value.get("call_accounting", {}).get(
                "measured_peak_concurrent_calls"
            ),
            "measured_effective_concurrency": value.get("call_accounting", {}).get(
                "measured_effective_concurrency"
            ),
        }
    return value


def _render_report(report: Mapping[str, Any]) -> str:
    efficacy = report["efficacy"]
    calls = report["call_accounting"]
    timing = report.get("timing", {})
    lines = [
        "# ARCGEN V4.1 — single-instance test-challenge calibration",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Recommendation: **{report['promotion_recommendation']}**",
        "",
        "This follow-up replaces the first experiment's independently authored A/B episodes. Each source contributes one final task only: its training grids and rule are frozen, and a narrow Sol/max authoring call may revise only the test or decline.",
        "",
        "## Outcome",
        "",
        f"- Frozen ordinary source episodes: {report['funnel']['source_episodes']}",
        f"- Refinements used / declined / failed: {report['funnel']['used']} / {report['funnel']['declined']} / {report['funnel']['operational_or_validation_failures']}",
        f"- Deterministically valid final tasks: {report['funnel']['realized']}",
        f"- Technical passes: {report['funnel']['technical_passes']}",
        f"- Blindly cleared: {report['funnel']['blind_passes']}",
        f"- Core / reserve / ambiguous / rejected: {report['funnel']['core']} / {report['funnel']['reserve']} / {report['funnel']['ambiguous']} / {report['funnel']['other']}",
        "",
        "## Efficacy",
        "",
        f"- Final-task audit counts: `{efficacy['audit_status_counts']}`",
        f"- Used refinements that were genuinely harder under the same rule: {efficacy.get('used_refinement_meaningfully_harder_count', 'n/a')}/{efficacy.get('used_refinement_count', 'n/a')}.",
        f"- Used-refinement efficacy rate: `{efficacy.get('used_refinement_efficacy_rate')}`.",
        f"- Decision × audit breakdown: `{efficacy.get('decision_by_audit_status')}`.",
        f"- Unsupported tests found by the non-gating audit: {efficacy.get('unsupported_audit_count', 0)}.",
        f"- Training grids preserved exactly: **{report['fidelity']['all_training_grids_identical']}**.",
        f"- Intended rules preserved exactly: **{report['fidelity']['all_intended_rules_identical']}**.",
        "",
        "The three declines are informative: the author was allowed to preserve an already demanding test rather than manufacture novelty. Audit labels describe the final episode, so a declined source can still have a genuinely demanding original test.",
        "",
        "## Quality reference",
        "",
        f"- Frozen source aggregate: `{report['quality_reference']['frozen_source']}`",
        f"- Final single-instance aggregate: `{report['quality_reference']['final_tasks']}`",
        "",
        "## Failure modes",
        "",
    ]
    failures = report.get("failure_mode_analysis", {})
    for item in failures.get("technical_rejections", []):
        lines.append(
            f"- `{item['task_id']}`: technical `{item['verifier_status']}` despite "
            f"mechanical correctness `{item['mechanically_correct']}` and hidden-output match "
            f"`{item['hidden_output_match']}`. Repair note: {item['repair_instruction']}"
        )
    for item in failures.get("ambiguity_holds", []):
        lines.append(
            f"- `{item['task_id']}`: held ambiguous after {item['blind_opinion_count']} blind "
            f"opinions with serious-alternative flags `{item['serious_alternative_by_opinion']}`; "
            f"adjudication run: `{item['adjudication_run']}`. {item['reason']}"
        )
    if not failures.get("technical_rejections") and not failures.get("ambiguity_holds"):
        lines.append("- No technical rejection or ambiguity hold.")
    lines.extend(
        [
            "",
            "## Diversity and coverage retention",
            "",
            f"- Genuine revised instances available for clustering: {report.get('diversity', {}).get('genuine_revised_instance_count', 'n/a')}.",
            f"- Diversity conclusion: {report.get('diversity', {}).get('conclusion', 'not assessed')}.",
            f"- Source-mode counts: `{report.get('coverage_retention', {}).get('source_search_mode_counts', {})}`.",
            "- No paired variants were authored, no ordinary source concept was suppressed, and no task received an editorial bonus for a harder test.",
            "",
            "## Calls and timing",
            "",
            f"- Experimental logical calls: {calls['experimental_calls']}.",
            f"- Active model-minutes: {calls['active_model_minutes']}.",
            f"- Wall-clock minutes: {timing.get('wall_clock_minutes', 'n/a')}.",
            f"- Peak / effective concurrency: {calls['measured_peak_concurrent_calls']} / {calls['measured_effective_concurrency']}.",
            f"- Transport retries: {calls['transport_retry_calls']}.",
            "",
            "## Task outcomes",
            "",
            "| Task | Search mode | Refiner | Audit | Final intrinsic status | Portfolio placement |",
            "|---|---|---|---|---|---|",
        ]
    )
    for task in report["tasks"]:
        record = task.get("artifact_record") or {}
        intrinsic = task.get("intrinsic_quality") or record.get("status") or "unreviewed"
        placement = task.get("portfolio_placement") or "not_applicable"
        audit = (task.get("audit") or {}).get("status", "not_audited")
        lines.append(
            f"| `{task['task_id']}` | `{task['search_mode']}` | "
            f"`{task['refinement_decision']}` | `{audit}` | `{intrinsic}` | `{placement}` |"
        )
    lines.extend(
        [
            "",
            "## Human review and promotion",
            "",
            f"- Single-instance review sheet: `{report['human_review']['contact_sheet']}`",
            "- No paired tasks appear in the review pack; rules and provenance are withheld from the first page.",
            f"- Promotion recommendation: **{report['promotion_recommendation']}**.",
            "- Human review is pending, and four genuine revised instances are below the eight-instance threshold for a responsible diversity conclusion.",
            "",
            "No production feature promotion or automatic good-task commit occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def run_test_challenge_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("experimental", {}).get("design") != "test_only_single_instance_v2":
        raise TestChallengeError("test-only single-instance configuration is required")
    if config["_feature_manifest_snapshot"]["features"]["supported_departure"]["status"] not in {"proposed", "testing"}:
        raise TestChallengeError("supported-departure feature is not awaiting evaluation")
    protected_before = _protected_hashes(config)
    source_run, source_rows = _source_records(config)
    if not source_rows:
        raise TestChallengeError("no frozen ordinary source episodes are available")
    aliases = _aliases(source_rows, int(config["seed"]))
    frozen = _source_manifest(config, source_run, source_rows, aliases, protected_before)
    runtime = V41Runtime(config, frozen)
    _install_extensions(runtime)
    if runtime.state.get("completed") and (runtime.run_dir / "report.json").is_file():
        return read_json(runtime.run_dir / "report.json")
    _initialize_records(runtime, source_rows, aliases)
    _refine(runtime)
    with runtime.lock:
        runtime.state.setdefault("escalation_usage_by_slate", {})
        runtime.save()

    def persist(index: int, used: int) -> None:
        with runtime.lock:
            runtime.state["escalation_usage_by_slate"][str(index)] = used
            runtime.save()

    slate_indices = sorted({int(row.get("slate_index", 1)) for row in runtime.state["semantic_records"].values()})
    limit = int(config["test_challenge"].get("second_opinion_budget_per_slate", 1))
    budgets = {
        (index, "single"): SlateEscalationBudget(
            limit,
            used=int(runtime.state["escalation_usage_by_slate"].get(str(index), 0)),
            on_change=lambda used, index=index: persist(index, used),
        )
        for index in slate_indices
    }
    _run_technical(runtime, budgets)
    _run_blind(runtime, budgets)
    _board(runtime)
    audit = _audit(runtime)
    human = _human_pack(runtime)
    records = list(runtime.state["semantic_records"].values())
    decisions = Counter(row.get("test_refinement", {}).get("decision", "failed") for row in records)
    audit_counts = Counter(item["status"] for item in audit)
    metrics = _metrics(records)
    base = _base_metrics(records)
    protected_after = _protected_hashes(config)
    _assert_hashes_unchanged(protected_before, protected_after)
    statuses = Counter(row.get("status", "unknown") for row in records)
    report = {
        "schema_version": 1,
        "pipeline": "arcgen-v4.1-exp-test-challenge-single-instance",
        "run_id": runtime.run_id,
        "feature": "supported_departure",
        "experimental_design": "test_only_single_instance_v2",
        "completed": True,
        "funnel": {
            "source_episodes": len(records),
            "used": decisions["use"],
            "declined": decisions["decline"],
            "operational_or_validation_failures": decisions["failed"],
            "realized": metrics["realized_count"],
            "technical_passes": sum(row.get("status") in {"technical_pass", "blind_pass", "editorial_complete", "ambiguous"} for row in records),
            "blind_passes": sum(row.get("status") in {"blind_pass", "editorial_complete"} for row in records),
            "core": metrics["core_count"],
            "reserve": metrics["reserve_count"],
            "ambiguous": statuses["ambiguous"],
            "other": len(records) - metrics["core_count"] - metrics["reserve_count"] - statuses["ambiguous"],
        },
        "efficacy": {
            "audit_status_counts": dict(sorted(audit_counts.items())),
            "meaningfully_harder_same_rule_count": audit_counts["meaningfully_harder_same_rule"],
            "refiner_decline_rate": decisions["decline"] / len(records),
        },
        "fidelity": {
            "all_training_grids_identical": all(row.get("plan_fidelity", {}).get("training_grids_byte_identical") for row in records if row.get("complete_task")),
            "all_intended_rules_identical": all(row.get("plan_fidelity", {}).get("intended_rule_identical") for row in records if row.get("complete_task")),
            "only_test_was_mutable": True,
            "single_instance_review": True,
        },
        "quality_reference": {"frozen_source": base, "final_tasks": metrics},
        "human_review": human,
        "tasks": [
            {
                "task_id": row["task_id"],
                "source_candidate_id": row["source_candidate_id"],
                "search_mode": row["search_mode"],
                "refinement_decision": row.get("test_refinement", {}).get("decision"),
                "audit": next((item for item in audit if item["task_id"] == row["task_id"]), None),
                "deterministic_valid": bool(row.get("deterministic_diagnostics", {}).get("hard_valid")),
                "technical_results": row.get("technical_results"),
                "blind_results": row.get("blind_results"),
                "editorial_judgments": row.get("editorial_judgments"),
                "intrinsic_quality": row.get("intrinsic_quality"),
                "portfolio_placement": row.get("portfolio_placement"),
                "grid_metrics": _task_metrics(row["complete_task"]) if row.get("complete_task") else None,
                "artifact_record": deepcopy(row),
            }
            for row in sorted(records, key=lambda item: item["task_id"])
        ],
        "call_accounting": runtime.accounting(),
        "promotion_recommendation": "continue_testing",
        "promotion_reason": "Await blinded human review; this revised design must be judged on its single final episodes.",
        "production_promotion_occurred": False,
        "good_task_bank_mutated_by_experiment": False,
        "protected_production_state_unchanged": protected_before == protected_after,
        "next_experiment_started": False,
        "unresolved_uncertainties": [
            "Human review is pending.",
            "The frozen-source aggregate is descriptive rather than a newly authored paired control.",
            "A single follow-up sample is insufficient for automatic feature promotion.",
        ],
    }
    report = enrich_test_challenge_report(report, runtime.state)
    atomic_write_json(runtime.run_dir / "quality_reference.json", report["quality_reference"])
    atomic_write_json(runtime.run_dir / "test_challenge_fidelity.json", report["fidelity"])
    atomic_write_json(runtime.run_dir / "promotion_recommendation.json", {"feature": "supported_departure", "recommendation": "continue_testing", "production_promotion_occurred": False})
    return runtime.finish(report, _render_report(report))


__all__ = [
    "STRICT_EXTENSION_SCHEMAS",
    "TestChallengeError",
    "_ordinary_plan",
    "_parse_extension",
    "_realization_from_refinement",
    "_validate_refinement",
    "enrich_test_challenge_report",
    "run_test_challenge_experiment",
]
