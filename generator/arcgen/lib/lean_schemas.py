"""Structured outputs for ``arcgen-lean-authoring-v1``.

This registry is deliberately separate from :mod:`arcgen.lib.schemas`.  The
legacy pipeline keeps its original stage names and models; the lean pipeline
passes this registry explicitly to the shared invocation wrapper.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .schemas import SchemaError, parse_grid, parse_json_object


TEXT = {"type": "string", "minLength": 1}
MAYBE_TEXT = {"type": "string"}
BOOLEAN = {"type": "boolean"}
AXIS_POTENTIAL = {"type": "string", "enum": ["weak", "solid", "standout"]}
CONCEPT_AXIS = {"type": "string", "enum": ["weak", "solid", "standout"]}
COMPOSITION_AXIS = {"type": "string", "enum": ["thin", "solid", "standout"]}
CONFIDENCE = {"type": "string", "enum": ["low", "medium", "high"]}
GRID = {
    "type": "array",
    "minItems": 1,
    "maxItems": 30,
    "items": {
        "type": "array",
        "minItems": 1,
        "maxItems": 30,
        "items": {"type": "integer", "minimum": 0, "maximum": 9},
    },
}


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": deepcopy(dict(properties)),
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items: Any, *, minimum: int = 0, maximum: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"type": "array", "items": deepcopy(items), "minItems": minimum}
    if maximum is not None:
        value["maxItems"] = maximum
    return value


IDEA = _object(
    {
        "candidate_id": TEXT,
        "search_mode": {
            "type": "string",
            "enum": ["open", "concept_led", "composition_led"],
        },
        "central_relation": TEXT,
        "why_coherent": TEXT,
        "novel_concept_potential": TEXT,
        "functional_composition_potential": TEXT,
        "why_multiple_training_examples_may_be_useful": TEXT,
        "possible_test_generalization": TEXT,
        "main_authoring_risk": TEXT,
    }
)
IDEA_SEARCH = _object(
    {
        "search_call_id": TEXT,
        "search_mode": {
            "type": "string",
            "enum": ["open", "concept_led", "composition_led"],
        },
        "ideas": _array(IDEA, minimum=4, maximum=4),
    }
)

SELECTOR_ASSESSMENT = _object(
    {
        "candidate_id": TEXT,
        "novel_concept_potential": AXIS_POTENTIAL,
        "functional_composition_potential": AXIS_POTENTIAL,
        "fair_episode_potential": AXIS_POTENTIAL,
        "derivative_of": {
            "type": "string",
            "enum": [
                "none",
                "real_task",
                "previous_generated",
                "same_batch",
                "portfolio_metaphor",
            ],
        },
        "duplicate_or_sibling_of": _array(TEXT, maximum=11),
        "overused_metaphor_or_attractor": MAYBE_TEXT,
        "decision": {"type": "string", "enum": ["select", "reject"]},
        "reason": TEXT,
    }
)
EDITORIAL_BRIEF = _object(
    {
        "candidate_id": TEXT,
        "development_priority": {"type": "integer", "minimum": 1, "maximum": 5},
        "route": {
            "type": "string",
            "enum": ["concept_led", "composition_led", "flagship", "wildcard"],
        },
        "preserve": TEXT,
        "develop": TEXT,
        "avoid": TEXT,
        "closest_similarity_concern": TEXT,
        "selection_reason": TEXT,
    }
)
DUPLICATE_GROUP = _object(
    {
        "candidate_ids": _array(TEXT, minimum=2, maximum=12),
        "shared_minimal_solution_procedure": TEXT,
        "strongest_candidate_id": TEXT,
    }
)
BATCH_SELECTOR = _object(
    {
        "selector_id": TEXT,
        "candidate_assessments": _array(
            SELECTOR_ASSESSMENT, minimum=12, maximum=12
        ),
        "within_batch_duplicate_groups": _array(DUPLICATE_GROUP, maximum=12),
        "selected": _array(EDITORIAL_BRIEF, maximum=5),
        "portfolio_observations": TEXT,
    }
)

FUNCTIONAL_ELEMENT = _object(
    {
        "element": TEXT,
        "role": TEXT,
        "interaction_with_other_elements": TEXT,
        "what_breaks_or_simplifies_if_removed": TEXT,
    }
)
TRAINING_PLAN = _object(
    {
        "pair_number": {"type": "integer", "minimum": 1, "maximum": 10},
        "what_it_teaches": TEXT,
        "why_it_is_needed": TEXT,
        "what_varies": TEXT,
        "what_remains_invariant": TEXT,
        "strongest_wrong_rule_it_discriminates_against": TEXT,
        "construction_plan": TEXT,
    }
)
TEST_PLAN = _object(
    {
        "construction": TEXT,
        "what_is_new": TEXT,
        "what_training_evidence_supports_it": TEXT,
        "why_the_output_is_uniquely_determined": TEXT,
    }
)
COMPETING_RULE = _object(
    {
        "alternative_rule": TEXT,
        "why_it_is_plausible": TEXT,
        "which_training_evidence_defeats_it": TEXT,
    }
)
COMPOSITION_AUDIT = _object(
    {
        "why_the_composition_is_functional": TEXT,
        "whether_the_task_decomposes_into_independent_subtasks": TEXT,
        "decorative_elements_to_avoid": _array(TEXT, maximum=12),
    }
)
AXIS_SELF_ASSESSMENT = _object(
    {
        "novel_concept": CONCEPT_AXIS,
        "functional_composition": COMPOSITION_AXIS,
    }
)
EPISODE_ARCHITECT = _object(
    {
        "candidate_id": TEXT,
        "status": {"type": "string", "enum": ["ready", "abort"]},
        "abort_reason": MAYBE_TEXT,
        "intended_rule": MAYBE_TEXT,
        "functional_elements": _array(FUNCTIONAL_ELEMENT, maximum=12),
        "training_pair_plan": _array(TRAINING_PLAN, maximum=10),
        "test_plan": {
            "anyOf": [TEST_PLAN, {"type": "null"}],
        },
        "competing_rule_audit": _array(COMPETING_RULE, maximum=10),
        "composition_audit": {
            "anyOf": [COMPOSITION_AUDIT, {"type": "null"}],
        },
        "realization_plan": MAYBE_TEXT,
        "axis_self_assessment": {
            "anyOf": [AXIS_SELF_ASSESSMENT, {"type": "null"}],
        },
    }
)

ARC_TRAIN_PAIR = _object({"input": GRID, "output": GRID})
ARC_TEST_INPUT = _object({"input": GRID})
PUBLIC_ARC_TASK = _object(
    {
        "train": _array(ARC_TRAIN_PAIR, minimum=1, maximum=10),
        "test": _array(ARC_TEST_INPUT, minimum=1, maximum=3),
    }
)
PAIR_TRACE = _object({"pair": TEXT, "trace": TEXT})
REALIZER_METADATA = _object(
    {
        "intended_rule": TEXT,
        "expected_test_outputs": _array(GRID, minimum=1, maximum=3),
        "intermediate_construction_witness": TEXT,
        "pair_construction_trace": _array(PAIR_TRACE, minimum=2, maximum=13),
    }
)
TASK_REALIZER = _object(
    {
        "candidate_id": TEXT,
        "task": PUBLIC_ARC_TASK,
        "authoring_metadata": REALIZER_METADATA,
    }
)

TRAINING_TRACE = _object(
    {
        "pair": TEXT,
        "follows_rule_exactly": BOOLEAN,
        "trace": TEXT,
        "issues": _array(TEXT, maximum=20),
    }
)
DIAGNOSTIC_ASSESSMENT = _object(
    {
        "flag_code": TEXT,
        "justified": BOOLEAN,
        "evidentially_necessary": BOOLEAN,
        "assessment": TEXT,
    }
)
TECHNICAL_VERIFIER = _object(
    {
        "task_id": TEXT,
        "verifier_id": TEXT,
        "status": {
            "type": "string",
            "enum": ["pass", "repairable", "fatal", "uncertain"],
        },
        "derived_test_outputs": _array(GRID, minimum=1, maximum=3),
        "training_pair_traces": _array(TRAINING_TRACE, minimum=1, maximum=10),
        "mechanically_correct": BOOLEAN,
        "test_supported": BOOLEAN,
        "pair_economy": {
            "type": "string",
            "enum": ["pass", "concern", "fail"],
        },
        "composition_preserved": BOOLEAN,
        "arbitrary_exceptions": _array(TEXT, maximum=20),
        "unsupported_test_semantics": _array(TEXT, maximum=20),
        "diagnostic_flag_assessments": _array(DIAGNOSTIC_ASSESSMENT, maximum=100),
        "repair_instruction": MAYBE_TEXT,
        "confidence": CONFIDENCE,
    }
)

BLIND_ALTERNATIVE = _object(
    {
        "rule": TEXT,
        "fits_all_training_examples": BOOLEAN,
        "comparably_simple": BOOLEAN,
        "well_supported": BOOLEAN,
        "predicted_test_outputs": _array(GRID, minimum=1, maximum=3),
        "reason": TEXT,
    }
)
BLIND_SOLVER = _object(
    {
        "task_id": TEXT,
        "solver_id": TEXT,
        "primary_inferred_rule": TEXT,
        "predicted_test_outputs": _array(GRID, minimum=1, maximum=3),
        "strongest_plausible_alternatives": _array(BLIND_ALTERNATIVE, maximum=5),
        "serious_different_output_alternative": BOOLEAN,
        "test_requires_unsupported_guessing": BOOLEAN,
        "fairness_assessment": TEXT,
        "confidence": CONFIDENCE,
    }
)

AMBIGUITY_ADJUDICATOR = _object(
    {
        "task_id": TEXT,
        "decision": {
            "type": "string",
            "enum": ["clear", "repairable", "serious_ambiguity"],
        },
        "intended_rule_reasonably_supported": BOOLEAN,
        "comparably_simple_different_output_rule": BOOLEAN,
        "reason": TEXT,
        "repair_instruction": MAYBE_TEXT,
    }
)

REPAIR_OUTPUT = _object(
    {
        "candidate_id": TEXT,
        "repair_type": {"type": "string", "enum": ["mechanical", "ambiguity"]},
        "status": {"type": "string", "enum": ["repaired", "unrepairable"]},
        "unrepairable_reason": MAYBE_TEXT,
        "repaired_task": {"anyOf": [TASK_REALIZER, {"type": "null"}]},
        "changes": _array(TEXT, maximum=10),
        "why_rule_episode_and_test_are_preserved": MAYBE_TEXT,
    }
)

EDITORIAL_ENTRY = _object(
    {
        "task_id": TEXT,
        "novel_concept": CONCEPT_AXIS,
        "functional_composition": COMPOSITION_AXIS,
        "overall_quality": {
            "type": "string",
            "enum": ["below_real_bar", "real_quality", "exceptional"],
        },
        "portfolio_relation": {
            "type": "string",
            "enum": ["distinct", "overrepresented_theme", "same_family"],
        },
        "decision": {"type": "string", "enum": ["core", "reserve", "reject"]},
        "reason": TEXT,
    }
)
EDITORIAL_BOARD = _object(
    {
        "board_id": TEXT,
        "judgments": _array(EDITORIAL_ENTRY, minimum=7, maximum=10),
        "batch_observations": TEXT,
    }
)


STRICT_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "lean_idea_search": IDEA_SEARCH,
    "lean_batch_selector": BATCH_SELECTOR,
    "lean_episode_architect": EPISODE_ARCHITECT,
    "lean_task_realizer": TASK_REALIZER,
    "lean_technical_verifier": TECHNICAL_VERIFIER,
    "lean_blind_solver": BLIND_SOLVER,
    "lean_ambiguity_adjudicator": AMBIGUITY_ADJUDICATOR,
    "lean_targeted_repair": REPAIR_OUTPUT,
    "lean_editorial_board": EDITORIAL_BOARD,
}


def _transport_schema(value: Any) -> Any:
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
        return {
            key: _transport_schema(item)
            for key, item in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [_transport_schema(item) for item in value]
    return value


OUTPUT_SCHEMAS = {
    stage: _transport_schema(schema)
    for stage, schema in STRICT_OUTPUT_SCHEMAS.items()
}


def _schema_validate(stage: str, value: dict[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(STRICT_OUTPUT_SCHEMAS[stage]).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{item}]" if isinstance(item, int) else f".{item}"
            for item in error.absolute_path
        )
        raise SchemaError(path, error.message)


def _validate_grids(stage: str, value: dict[str, Any]) -> None:
    grids: list[tuple[Any, str]] = []
    if stage in {"lean_task_realizer", "lean_targeted_repair"}:
        wrapper = value
        if stage == "lean_targeted_repair":
            if value["status"] == "unrepairable":
                if value["repaired_task"] is not None or value["changes"]:
                    raise SchemaError("$", "unrepairable repair must not contain a task or changes")
                if not value["unrepairable_reason"].strip():
                    raise SchemaError("$.unrepairable_reason", "must explain why repair failed")
                return
            wrapper = value["repaired_task"]
            if wrapper is None:
                raise SchemaError("$.repaired_task", "repaired status requires a task")
            if wrapper["candidate_id"] != value["candidate_id"]:
                raise SchemaError("$.repaired_task.candidate_id", "must match candidate_id")
        task = wrapper["task"]
        metadata = wrapper["authoring_metadata"]
        for split in ("train", "test"):
            for index, pair in enumerate(task[split]):
                grids.append((pair["input"], f"$.task.{split}[{index}].input"))
                if split == "train":
                    grids.append((pair["output"], f"$.task.train[{index}].output"))
        outputs = metadata["expected_test_outputs"]
        if len(outputs) != len(task["test"]):
            raise SchemaError(
                "$.authoring_metadata.expected_test_outputs",
                "must contain one output for each test input",
            )
        grids.extend(
            (grid, f"$.authoring_metadata.expected_test_outputs[{index}]")
            for index, grid in enumerate(outputs)
        )
    elif stage == "lean_technical_verifier":
        grids.extend(
            (grid, f"$.derived_test_outputs[{index}]")
            for index, grid in enumerate(value["derived_test_outputs"])
        )
    elif stage == "lean_blind_solver":
        grids.extend(
            (grid, f"$.predicted_test_outputs[{index}]")
            for index, grid in enumerate(value["predicted_test_outputs"])
        )
        for a_index, alternative in enumerate(value["strongest_plausible_alternatives"]):
            grids.extend(
                (
                    grid,
                    f"$.strongest_plausible_alternatives[{a_index}].predicted_test_outputs[{index}]",
                )
                for index, grid in enumerate(alternative["predicted_test_outputs"])
            )
    for grid, path in grids:
        parse_grid(grid, path)


def parse_stage_output(
    stage: str, data: str | bytes | bytearray | Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in STRICT_OUTPUT_SCHEMAS:
        raise SchemaError("$.stage", f"unknown lean stage {stage!r}")
    value = (
        parse_json_object(data)
        if isinstance(data, (str, bytes, bytearray))
        else dict(data)
    )
    _schema_validate(stage, value)
    if stage == "lean_idea_search":
        ids = [idea["candidate_id"] for idea in value["ideas"]]
        if len(ids) != len(set(ids)):
            raise SchemaError("$.ideas", "candidate IDs must be unique")
        if any(idea["search_mode"] != value["search_mode"] for idea in value["ideas"]):
            raise SchemaError("$.ideas", "every idea must preserve the wrapper search_mode")
    elif stage == "lean_batch_selector":
        assessment_ids = [
            item["candidate_id"] for item in value["candidate_assessments"]
        ]
        selected_ids = [item["candidate_id"] for item in value["selected"]]
        priorities = [item["development_priority"] for item in value["selected"]]
        if len(set(assessment_ids)) != len(assessment_ids):
            raise SchemaError("$.candidate_assessments", "candidate IDs must be unique")
        if len(set(selected_ids)) != len(selected_ids):
            raise SchemaError("$.selected", "selected candidate IDs must be unique")
        if sorted(priorities) != list(range(1, len(priorities) + 1)):
            raise SchemaError("$.selected", "development priorities must be contiguous from 1")
    elif stage == "lean_episode_architect":
        ready = value["status"] == "ready"
        if ready:
            for name in (
                "intended_rule",
                "realization_plan",
            ):
                if not value[name].strip():
                    raise SchemaError(f"$.{name}", "must be non-empty for a ready architecture")
            if (
                not value["functional_elements"]
                or not value["training_pair_plan"]
                or value["test_plan"] is None
                or value["composition_audit"] is None
                or value["axis_self_assessment"] is None
                or value["abort_reason"].strip()
            ):
                raise SchemaError("$", "ready architecture is incomplete or has an abort reason")
            numbers = [item["pair_number"] for item in value["training_pair_plan"]]
            if numbers != list(range(1, len(numbers) + 1)):
                raise SchemaError("$.training_pair_plan", "pair numbers must be contiguous from 1")
            axes = value["axis_self_assessment"]
            if axes["novel_concept"] == "weak" or axes["functional_composition"] == "thin":
                raise SchemaError(
                    "$.axis_self_assessment",
                    "ready architecture must self-assess at least solid on both axes",
                )
        elif not value["abort_reason"].strip():
            raise SchemaError("$.abort_reason", "abort status requires a reason")
    _validate_grids(stage, value)
    return value


__all__ = ["OUTPUT_SCHEMAS", "STRICT_OUTPUT_SCHEMAS", "parse_stage_output"]
