"""Strict structured-output models for the ARC generation benchmark.

The module deliberately uses only the standard library.  Model responses are
treated as untrusted input: JSON must be a single object, duplicate JSON keys
are rejected, every required field must be present, and unknown fields are
rejected.  Difficulty labels remain free text; there is no difficulty ontology
or task-family taxonomy in these schemas.

``OUTPUT_SCHEMAS`` contains self-contained JSON Schemas that can be written to
disk and supplied to ``codex --output-schema``.  ``parse_stage_output`` applies
the stronger Python-side checks that JSON Schema cannot express conveniently
(for example rectangular grids and cross-field consistency).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
import math
from typing import Any, ClassVar, Mapping, Sequence, TypeAlias


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
Grid: TypeAlias = tuple[tuple[int, ...], ...]


class SchemaError(ValueError):
    """Raised when a structured model response is malformed or incomplete."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json_object(raw: str | bytes | bytearray) -> dict[str, Any]:
    """Parse exactly one strict JSON object.

    Markdown fences, prose surrounding JSON, duplicate keys, NaN, and Infinity
    are rejected instead of being repaired or silently reused.
    """

    try:
        value = json.loads(
            raw,
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise SchemaError("$", f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaError("$", "top-level JSON value must be an object")
    return value


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise SchemaError(path, "all object keys must be strings")
    return value


def _keys(
    value: Mapping[str, Any],
    path: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise SchemaError(path, f"missing required field(s): {', '.join(missing)}")
    if unknown:
        raise SchemaError(path, f"unknown field(s): {', '.join(unknown)}")


def _array(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(path, "must be an array")
    if len(value) < minimum:
        raise SchemaError(path, f"must contain at least {minimum} item(s)")
    if maximum is not None and len(value) > maximum:
        raise SchemaError(path, f"must contain at most {maximum} item(s)")
    return value


def _text(value: Any, path: str, *, allow_blank: bool = False) -> str:
    if not isinstance(value, str):
        raise SchemaError(path, "must be a string")
    if not allow_blank and not value.strip():
        raise SchemaError(path, "must be a non-empty string")
    if "\x00" in value:
        raise SchemaError(path, "must not contain NUL characters")
    return value


def _nullable_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _text(value, path)


def _json_value(value: Any, path: str, *, depth: int = 0) -> JSONValue:
    """Validate and copy an explicitly open JSON payload such as provenance."""

    if depth > 50:
        raise SchemaError(path, "JSON value is nested too deeply")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SchemaError(path, "number must be finite")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{path}[{i}]", depth=depth + 1) for i, item in enumerate(value)]
    if isinstance(value, dict):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError(path, "all object keys must be strings")
            result[key] = _json_value(item, f"{path}.{key}", depth=depth + 1)
        return result
    raise SchemaError(path, f"unsupported JSON value type {type(value).__name__}")


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise SchemaError(path, "must be a boolean")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise SchemaError(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaError(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise SchemaError(path, f"must be <= {maximum}")
    return value


def _enum(value: Any, path: str, allowed: Sequence[str]) -> str:
    text = _text(value, path)
    if text not in allowed:
        raise SchemaError(path, f"must be one of: {', '.join(allowed)}")
    return text


def _text_array(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
    unique: bool = False,
) -> tuple[str, ...]:
    items = _array(value, path, minimum=minimum, maximum=maximum)
    result = tuple(_text(item, f"{path}[{index}]") for index, item in enumerate(items))
    if unique and len(set(result)) != len(result):
        raise SchemaError(path, "must not contain duplicate strings")
    return result


def parse_grid(value: Any, path: str = "$.grid") -> Grid:
    """Parse a rectangular ARC grid with dimensions and colors in range."""

    rows = _array(value, path, minimum=1, maximum=30)
    parsed: list[tuple[int, ...]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(rows):
        row = _array(raw_row, f"{path}[{row_index}]", minimum=1, maximum=30)
        parsed_row = tuple(
            _integer(cell, f"{path}[{row_index}][{column_index}]", minimum=0, maximum=9)
            for column_index, cell in enumerate(row)
        )
        if width is None:
            width = len(parsed_row)
        elif len(parsed_row) != width:
            raise SchemaError(
                f"{path}[{row_index}]",
                f"row width {len(parsed_row)} does not match first-row width {width}",
            )
        parsed.append(parsed_row)
    return tuple(parsed)


def _primitive(value: Any) -> JSONValue:
    if is_dataclass(value):
        return {field.name: _primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_primitive(item) for item in value]
    if isinstance(value, list):
        return [_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class StrictModel:
    """Mixin shared by immutable structured-output dataclasses."""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "StrictModel":
        raise NotImplementedError

    @classmethod
    def from_json(cls, raw: str | bytes | bytearray) -> "StrictModel":
        return cls.from_dict(parse_json_object(raw))

    def to_dict(self) -> dict[str, Any]:
        result = _primitive(self)
        assert isinstance(result, dict)
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )


GENERATION_MODES = ("open", "single_factor", "two_factor")
SEED_FACTOR_STATUSES = ("central", "supporting", "decorative", "absent", "uncertain")
ACTIVATION_STATUSES = ("central", "supporting", "decorative", "absent")
SIMILARITY_TYPES = ("duplicate", "surface_variant", "same_family", "partial_overlap")
CONFIDENCE_LEVELS = ("low", "medium", "high")
AMBIGUITY_DECISIONS = ("pass", "reject", "revise")
ANSWER_CLASSES = ("one_guess", "bounded_two_guess", "underdetermined")
QUALITY_FLAGS = ("invalid", "ambiguous", "arbitrary", "trivial")


@dataclass(frozen=True, slots=True)
class IdeaCandidate(StrictModel):
    candidate_id: str
    generation_mode: str
    seed_factors: tuple[str, ...]
    central_relation: str
    why_coherent: str
    training_evidence_strategy: str
    episode_level_inference: str
    where_novelty_resides: str
    test_generalization: str
    difference_from_calibration_examples: str
    feasibility_notes: str

    REQUIRED: ClassVar[tuple[str, ...]] = (
        "candidate_id",
        "generation_mode",
        "seed_factors",
        "central_relation",
        "why_coherent",
        "training_evidence_strategy",
        "episode_level_inference",
        "where_novelty_resides",
        "test_generalization",
        "difference_from_calibration_examples",
        "feasibility_notes",
    )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "IdeaCandidate":
        obj = _mapping(value, path)
        _keys(obj, path, cls.REQUIRED)
        mode = _enum(obj["generation_mode"], f"{path}.generation_mode", GENERATION_MODES)
        seeds = _text_array(obj["seed_factors"], f"{path}.seed_factors", maximum=2, unique=True)
        expected_seed_count = {"open": 0, "single_factor": 1, "two_factor": 2}[mode]
        if len(seeds) != expected_seed_count:
            raise SchemaError(
                f"{path}.seed_factors",
                f"{mode!r} requires exactly {expected_seed_count} seed factor(s)",
            )
        return cls(
            candidate_id=_text(obj["candidate_id"], f"{path}.candidate_id"),
            generation_mode=mode,
            seed_factors=seeds,
            central_relation=_text(obj["central_relation"], f"{path}.central_relation"),
            why_coherent=_text(obj["why_coherent"], f"{path}.why_coherent"),
            training_evidence_strategy=_text(
                obj["training_evidence_strategy"],
                f"{path}.training_evidence_strategy",
            ),
            episode_level_inference=_text(
                obj["episode_level_inference"],
                f"{path}.episode_level_inference",
            ),
            where_novelty_resides=_text(
                obj["where_novelty_resides"],
                f"{path}.where_novelty_resides",
            ),
            test_generalization=_text(obj["test_generalization"], f"{path}.test_generalization"),
            difference_from_calibration_examples=_text(
                obj["difference_from_calibration_examples"],
                f"{path}.difference_from_calibration_examples",
            ),
            feasibility_notes=_text(obj["feasibility_notes"], f"{path}.feasibility_notes"),
        )


@dataclass(frozen=True, slots=True)
class IdeaGenerationOutput(StrictModel):
    generation_call_id: str
    generation_mode: str
    seed_factors: tuple[str, ...]
    factor_combination_feasible: bool
    factor_combination_infeasibility_reason: str
    ideas: tuple[IdeaCandidate, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "IdeaGenerationOutput":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            (
                "generation_call_id",
                "generation_mode",
                "seed_factors",
                "factor_combination_feasible",
                "factor_combination_infeasibility_reason",
                "ideas",
            ),
        )
        mode = _enum(obj["generation_mode"], f"{path}.generation_mode", GENERATION_MODES)
        seeds = _text_array(obj["seed_factors"], f"{path}.seed_factors", maximum=2, unique=True)
        expected_seed_count = {"open": 0, "single_factor": 1, "two_factor": 2}[mode]
        if len(seeds) != expected_seed_count:
            raise SchemaError(f"{path}.seed_factors", f"{mode!r} requires exactly {expected_seed_count} factor(s)")
        feasible = _boolean(obj["factor_combination_feasible"], f"{path}.factor_combination_feasible")
        reason = _text(
            obj["factor_combination_infeasibility_reason"],
            f"{path}.factor_combination_infeasibility_reason",
            allow_blank=True,
        )
        raw = _array(
            obj["ideas"],
            f"{path}.ideas",
            minimum=4 if feasible else 0,
            maximum=4 if feasible else 0,
        )
        ideas = tuple(IdeaCandidate.from_dict(item, f"{path}.ideas[{i}]") for i, item in enumerate(raw))
        ids = [idea.candidate_id for idea in ideas]
        if len(set(ids)) != len(ids):
            raise SchemaError(f"{path}.ideas", "candidate_id values must be unique within a call")
        if any(idea.generation_mode != mode or idea.seed_factors != seeds for idea in ideas):
            raise SchemaError(f"{path}.ideas", "each idea must preserve the wrapper generation mode and seed factors")
        if feasible and reason.strip():
            raise SchemaError(
                f"{path}.factor_combination_infeasibility_reason",
                "must be empty when the factor combination is feasible",
            )
        if not feasible and not reason.strip():
            raise SchemaError(
                f"{path}.factor_combination_infeasibility_reason",
                "must be non-empty when the factor combination is infeasible",
            )
        return cls(
            generation_call_id=_text(obj["generation_call_id"], f"{path}.generation_call_id"),
            generation_mode=mode,
            seed_factors=seeds,
            factor_combination_feasible=feasible,
            factor_combination_infeasibility_reason=reason,
            ideas=ideas,
        )


@dataclass(frozen=True, slots=True)
class ObservedFactor(StrictModel):
    label: str
    manifestation: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "ObservedFactor":
        obj = _mapping(value, path)
        _keys(obj, path, ("label", "manifestation"))
        return cls(
            label=_text(obj["label"], f"{path}.label"),
            manifestation=_text(obj["manifestation"], f"{path}.manifestation"),
        )


@dataclass(frozen=True, slots=True)
class SeedFactorAssessment(StrictModel):
    factor: str
    status: str
    reason: str
    substantively_new_reinterpretation: bool
    reinterpretation_explanation: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "SeedFactorAssessment":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            (
                "factor",
                "status",
                "reason",
                "substantively_new_reinterpretation",
                "reinterpretation_explanation",
            ),
        )
        return cls(
            factor=_text(obj["factor"], f"{path}.factor"),
            status=_enum(obj["status"], f"{path}.status", SEED_FACTOR_STATUSES),
            reason=_text(obj["reason"], f"{path}.reason"),
            substantively_new_reinterpretation=_boolean(
                obj["substantively_new_reinterpretation"],
                f"{path}.substantively_new_reinterpretation",
            ),
            reinterpretation_explanation=_text(
                obj["reinterpretation_explanation"],
                f"{path}.reinterpretation_explanation",
            ),
        )


@dataclass(frozen=True, slots=True)
class DifficultyAnnotation(StrictModel):
    candidate_id: str
    observed_factors: tuple[ObservedFactor, ...]
    seed_factor_assessment: tuple[SeedFactorAssessment, ...]
    new_factor_labels: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "DifficultyAnnotation":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            ("candidate_id", "observed_factors", "seed_factor_assessment", "new_factor_labels"),
        )
        observed_raw = _array(
            obj["observed_factors"], f"{path}.observed_factors", minimum=1, maximum=5
        )
        observed = tuple(
            ObservedFactor.from_dict(item, f"{path}.observed_factors[{i}]")
            for i, item in enumerate(observed_raw)
        )
        if len({factor.label for factor in observed}) != len(observed):
            raise SchemaError(f"{path}.observed_factors", "factor labels must be unique")
        assessments_raw = _array(obj["seed_factor_assessment"], f"{path}.seed_factor_assessment", maximum=2)
        assessments = tuple(
            SeedFactorAssessment.from_dict(item, f"{path}.seed_factor_assessment[{i}]")
            for i, item in enumerate(assessments_raw)
        )
        if len({item.factor for item in assessments}) != len(assessments):
            raise SchemaError(f"{path}.seed_factor_assessment", "assessed factors must be unique")
        new_labels = _text_array(
            obj["new_factor_labels"], f"{path}.new_factor_labels", maximum=5, unique=True
        )
        if not set(new_labels) <= {factor.label for factor in observed}:
            raise SchemaError(
                f"{path}.new_factor_labels",
                "each newly introduced label must also appear in observed_factors",
            )
        return cls(
            candidate_id=_text(obj["candidate_id"], f"{path}.candidate_id"),
            observed_factors=observed,
            seed_factor_assessment=assessments,
            new_factor_labels=new_labels,
        )

    def assert_seed_factors(self, expected: Sequence[str]) -> None:
        actual = tuple(item.factor for item in self.seed_factor_assessment)
        if actual != tuple(expected):
            raise SchemaError(
                "$.seed_factor_assessment",
                "factors must exactly match supplied raw labels in supplied order",
            )


@dataclass(frozen=True, slots=True)
class DiversityPairwiseDecision(StrictModel):
    candidate_id_a: str
    candidate_id_b: str
    essentially_same: bool
    shared_minimal_solution_procedure: str
    material_difference: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "DiversityPairwiseDecision":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            (
                "candidate_id_a",
                "candidate_id_b",
                "essentially_same",
                "shared_minimal_solution_procedure",
                "material_difference",
            ),
        )
        a = _text(obj["candidate_id_a"], f"{path}.candidate_id_a")
        b = _text(obj["candidate_id_b"], f"{path}.candidate_id_b")
        if a == b:
            raise SchemaError(path, "a pairwise comparison requires two different candidates")
        return cls(
            candidate_id_a=a,
            candidate_id_b=b,
            essentially_same=_boolean(obj["essentially_same"], f"{path}.essentially_same"),
            shared_minimal_solution_procedure=_text(
                obj["shared_minimal_solution_procedure"],
                f"{path}.shared_minimal_solution_procedure",
            ),
            material_difference=_text(obj["material_difference"], f"{path}.material_difference"),
        )


@dataclass(frozen=True, slots=True)
class DuplicateCluster(StrictModel):
    candidate_ids: tuple[str, ...]
    representative_candidate_id: str
    reason: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "DuplicateCluster":
        obj = _mapping(value, path)
        _keys(obj, path, ("candidate_ids", "representative_candidate_id", "reason"))
        candidates = _text_array(obj["candidate_ids"], f"{path}.candidate_ids", minimum=2, unique=True)
        representative = _text(obj["representative_candidate_id"], f"{path}.representative_candidate_id")
        if representative not in candidates:
            raise SchemaError(f"{path}.representative_candidate_id", "must be a member of candidate_ids")
        return cls(
            candidate_ids=candidates,
            representative_candidate_id=representative,
            reason=_text(obj["reason"], f"{path}.reason"),
        )


@dataclass(frozen=True, slots=True)
class DuplicateRejection(StrictModel):
    candidate_id: str
    duplicate_of: str
    reason: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "DuplicateRejection":
        obj = _mapping(value, path)
        _keys(obj, path, ("candidate_id", "duplicate_of", "reason"))
        candidate = _text(obj["candidate_id"], f"{path}.candidate_id")
        duplicate_of = _text(obj["duplicate_of"], f"{path}.duplicate_of")
        if candidate == duplicate_of:
            raise SchemaError(path, "candidate_id cannot be a duplicate of itself")
        return cls(
            candidate_id=candidate,
            duplicate_of=duplicate_of,
            reason=_text(obj["reason"], f"{path}.reason"),
        )


@dataclass(frozen=True, slots=True)
class DiversityJudgeOutput(StrictModel):
    group_id: str
    pairwise_decisions: tuple[DiversityPairwiseDecision, ...]
    duplicate_clusters: tuple[DuplicateCluster, ...]
    keep_candidate_ids: tuple[str, ...]
    reject_as_duplicate: tuple[DuplicateRejection, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "DiversityJudgeOutput":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            (
                "group_id",
                "pairwise_decisions",
                "duplicate_clusters",
                "keep_candidate_ids",
                "reject_as_duplicate",
            ),
        )
        pairwise_raw = _array(obj["pairwise_decisions"], f"{path}.pairwise_decisions", minimum=1)
        pairwise = tuple(
            DiversityPairwiseDecision.from_dict(item, f"{path}.pairwise_decisions[{i}]")
            for i, item in enumerate(pairwise_raw)
        )
        unordered_pairs = [frozenset((item.candidate_id_a, item.candidate_id_b)) for item in pairwise]
        if len(set(unordered_pairs)) != len(unordered_pairs):
            raise SchemaError(f"{path}.pairwise_decisions", "candidate pairs must not be repeated")
        all_ids = {item.candidate_id_a for item in pairwise} | {item.candidate_id_b for item in pairwise}
        clusters_raw = _array(obj["duplicate_clusters"], f"{path}.duplicate_clusters")
        clusters = tuple(
            DuplicateCluster.from_dict(item, f"{path}.duplicate_clusters[{i}]")
            for i, item in enumerate(clusters_raw)
        )
        clustered_ids: set[str] = set()
        for cluster in clusters:
            overlap = clustered_ids & set(cluster.candidate_ids)
            if overlap:
                raise SchemaError(f"{path}.duplicate_clusters", f"candidate(s) appear in multiple clusters: {sorted(overlap)!r}")
            clustered_ids.update(cluster.candidate_ids)
        keep = _text_array(obj["keep_candidate_ids"], f"{path}.keep_candidate_ids", minimum=1, unique=True)
        rejected_raw = _array(obj["reject_as_duplicate"], f"{path}.reject_as_duplicate")
        rejected = tuple(
            DuplicateRejection.from_dict(item, f"{path}.reject_as_duplicate[{i}]")
            for i, item in enumerate(rejected_raw)
        )
        rejected_ids = [item.candidate_id for item in rejected]
        if len(set(rejected_ids)) != len(rejected_ids):
            raise SchemaError(f"{path}.reject_as_duplicate", "candidate_id values must be unique")
        if set(keep) & set(rejected_ids) or set(keep) | set(rejected_ids) != all_ids:
            raise SchemaError(path, "kept and rejected candidates must partition all pairwise candidate IDs")
        if any(item.duplicate_of not in keep for item in rejected):
            raise SchemaError(f"{path}.reject_as_duplicate", "duplicate_of must name a kept representative")
        for cluster in clusters:
            if cluster.representative_candidate_id not in keep:
                raise SchemaError(f"{path}.duplicate_clusters", "every representative must be kept")
            if any(
                candidate != cluster.representative_candidate_id and candidate not in rejected_ids
                for candidate in cluster.candidate_ids
            ):
                raise SchemaError(f"{path}.duplicate_clusters", "every non-representative cluster member must be rejected")
        return cls(
            group_id=_text(obj["group_id"], f"{path}.group_id"),
            pairwise_decisions=pairwise,
            duplicate_clusters=clusters,
            keep_candidate_ids=keep,
            reject_as_duplicate=rejected,
        )


@dataclass(frozen=True, slots=True)
class PortfolioReferenceAssessment(StrictModel):
    reference_id: str
    essentially_same: bool
    shared_minimal_solution_procedure: str
    material_difference: str
    preferred_representative: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        path: str = "$",
    ) -> "PortfolioReferenceAssessment":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            (
                "reference_id",
                "essentially_same",
                "shared_minimal_solution_procedure",
                "material_difference",
                "preferred_representative",
            ),
        )
        return cls(
            reference_id=_text(
                obj["reference_id"], f"{path}.reference_id"
            ),
            essentially_same=_boolean(
                obj["essentially_same"],
                f"{path}.essentially_same",
            ),
            shared_minimal_solution_procedure=_text(
                obj["shared_minimal_solution_procedure"],
                f"{path}.shared_minimal_solution_procedure",
            ),
            material_difference=_text(
                obj["material_difference"],
                f"{path}.material_difference",
            ),
            preferred_representative=_enum(
                obj["preferred_representative"],
                f"{path}.preferred_representative",
                ("candidate", "reference"),
            ),
        )


@dataclass(frozen=True, slots=True)
class PortfolioDiversityOutput(StrictModel):
    candidate_id: str
    reference_assessments: tuple[
        PortfolioReferenceAssessment, ...
    ]
    duplicate_reference_ids: tuple[str, ...]
    representative_id: str
    decision: str
    reason: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        path: str = "$",
    ) -> "PortfolioDiversityOutput":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            (
                "candidate_id",
                "reference_assessments",
                "duplicate_reference_ids",
                "representative_id",
                "decision",
                "reason",
            ),
        )
        candidate_id = _text(
            obj["candidate_id"], f"{path}.candidate_id"
        )
        raw_assessments = _array(
            obj["reference_assessments"],
            f"{path}.reference_assessments",
            minimum=1,
        )
        assessments = tuple(
            PortfolioReferenceAssessment.from_dict(
                item,
                f"{path}.reference_assessments[{index}]",
            )
            for index, item in enumerate(raw_assessments)
        )
        reference_ids = [
            item.reference_id for item in assessments
        ]
        if len(set(reference_ids)) != len(reference_ids):
            raise SchemaError(
                f"{path}.reference_assessments",
                "reference_id values must be unique",
            )
        if candidate_id in reference_ids:
            raise SchemaError(
                f"{path}.reference_assessments",
                "candidate_id cannot also be a reference_id",
            )
        duplicate_ids = _text_array(
            obj["duplicate_reference_ids"],
            f"{path}.duplicate_reference_ids",
            unique=True,
        )
        assessed_duplicates = {
            item.reference_id
            for item in assessments
            if item.essentially_same
        }
        if set(duplicate_ids) != assessed_duplicates:
            raise SchemaError(
                f"{path}.duplicate_reference_ids",
                "must exactly equal references assessed essentially_same",
            )
        decision = _enum(
            obj["decision"],
            f"{path}.decision",
            ("distinct", "reject_candidate", "replace_references"),
        )
        representative = _text(
            obj["representative_id"],
            f"{path}.representative_id",
        )
        by_id = {
            item.reference_id: item for item in assessments
        }
        if decision == "distinct":
            if duplicate_ids or representative != candidate_id:
                raise SchemaError(
                    path,
                    "distinct requires no duplicates and the candidate as "
                    "representative",
                )
        elif decision == "reject_candidate":
            if (
                not duplicate_ids
                or representative not in duplicate_ids
                or by_id[
                    representative
                ].preferred_representative
                != "reference"
            ):
                raise SchemaError(
                    path,
                    "reject_candidate requires a duplicate reference as "
                    "the preferred representative",
                )
        else:
            if (
                not duplicate_ids
                or representative != candidate_id
                or any(
                    by_id[reference_id].preferred_representative
                    != "candidate"
                    for reference_id in duplicate_ids
                )
            ):
                raise SchemaError(
                    path,
                    "replace_references requires the candidate to be "
                    "preferred over every duplicate reference",
                )
        return cls(
            candidate_id=candidate_id,
            reference_assessments=assessments,
            duplicate_reference_ids=duplicate_ids,
            representative_id=representative,
            decision=decision,
            reason=_text(obj["reason"], f"{path}.reason"),
        )


@dataclass(frozen=True, slots=True)
class NoveltyConflict(StrictModel):
    reference_task_id: str
    similarity_type: str
    shared_mechanism: str
    material_difference: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "NoveltyConflict":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            ("reference_task_id", "similarity_type", "shared_mechanism", "material_difference"),
        )
        return cls(
            reference_task_id=_text(obj["reference_task_id"], f"{path}.reference_task_id"),
            similarity_type=_enum(obj["similarity_type"], f"{path}.similarity_type", SIMILARITY_TYPES),
            shared_mechanism=_text(obj["shared_mechanism"], f"{path}.shared_mechanism"),
            material_difference=_text(obj["material_difference"], f"{path}.material_difference"),
        )


def _parse_conflicts(value: Any, path: str) -> tuple[NoveltyConflict, ...]:
    raw = _array(value, path, maximum=5)
    conflicts = tuple(NoveltyConflict.from_dict(item, f"{path}[{i}]") for i, item in enumerate(raw))
    if len({item.reference_task_id for item in conflicts}) != len(conflicts):
        raise SchemaError(path, "reference_task_id values must be unique")
    return conflicts


@dataclass(frozen=True, slots=True)
class NoveltyScoutOutput(StrictModel):
    candidate_id: str
    scout_id: str
    possible_conflicts: tuple[NoveltyConflict, ...]
    provisional_novelty_score: int
    reason: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "NoveltyScoutOutput":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            ("candidate_id", "scout_id", "possible_conflicts", "provisional_novelty_score", "reason"),
        )
        return cls(
            candidate_id=_text(obj["candidate_id"], f"{path}.candidate_id"),
            scout_id=_text(obj["scout_id"], f"{path}.scout_id"),
            possible_conflicts=_parse_conflicts(obj["possible_conflicts"], f"{path}.possible_conflicts"),
            provisional_novelty_score=_integer(
                obj["provisional_novelty_score"], f"{path}.provisional_novelty_score", minimum=0, maximum=4
            ),
            reason=_text(obj["reason"], f"{path}.reason"),
        )


@dataclass(frozen=True, slots=True)
class NoveltyAdjudication(StrictModel):
    candidate_id: str
    final_novelty_score: int
    passes_novelty: bool
    strongest_conflicts: tuple[NoveltyConflict, ...]
    reason: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "NoveltyAdjudication":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            ("candidate_id", "final_novelty_score", "passes_novelty", "strongest_conflicts", "reason"),
        )
        score = _integer(obj["final_novelty_score"], f"{path}.final_novelty_score", minimum=0, maximum=4)
        passes = _boolean(obj["passes_novelty"], f"{path}.passes_novelty")
        if passes != (score >= 3):
            raise SchemaError(f"{path}.passes_novelty", "must be true exactly when final_novelty_score is 3 or 4")
        return cls(
            candidate_id=_text(obj["candidate_id"], f"{path}.candidate_id"),
            final_novelty_score=score,
            passes_novelty=passes,
            strongest_conflicts=_parse_conflicts(obj["strongest_conflicts"], f"{path}.strongest_conflicts"),
            reason=_text(obj["reason"], f"{path}.reason"),
        )


@dataclass(frozen=True, slots=True)
class TrainingPairBlueprint(StrictModel):
    pair_number: int
    what_it_teaches: str
    why_it_is_needed: str
    what_varies: str
    what_remains_invariant: str
    plausible_wrong_rule_it_eliminates: str
    grid_construction_plan: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "TrainingPairBlueprint":
        obj = _mapping(value, path)
        required = (
            "pair_number",
            "what_it_teaches",
            "why_it_is_needed",
            "what_varies",
            "what_remains_invariant",
            "plausible_wrong_rule_it_eliminates",
            "grid_construction_plan",
        )
        _keys(obj, path, required)
        return cls(
            pair_number=_integer(obj["pair_number"], f"{path}.pair_number", minimum=1, maximum=10),
            what_it_teaches=_text(obj["what_it_teaches"], f"{path}.what_it_teaches"),
            why_it_is_needed=_text(obj["why_it_is_needed"], f"{path}.why_it_is_needed"),
            what_varies=_text(obj["what_varies"], f"{path}.what_varies"),
            what_remains_invariant=_text(obj["what_remains_invariant"], f"{path}.what_remains_invariant"),
            plausible_wrong_rule_it_eliminates=_text(
                obj["plausible_wrong_rule_it_eliminates"],
                f"{path}.plausible_wrong_rule_it_eliminates",
            ),
            grid_construction_plan=_text(obj["grid_construction_plan"], f"{path}.grid_construction_plan"),
        )


@dataclass(frozen=True, slots=True)
class TestPairBlueprint(StrictModel):
    pair_number: int
    construction_plan: str
    what_is_new: str
    what_training_evidence_supports_it: str
    why_the_output_is_determined: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        path: str = "$",
        *,
        allow_blank: bool = False,
    ) -> "TestPairBlueprint":
        obj = _mapping(value, path)
        required = (
            "pair_number",
            "construction_plan",
            "what_is_new",
            "what_training_evidence_supports_it",
            "why_the_output_is_determined",
        )
        _keys(obj, path, required)
        return cls(
            pair_number=_integer(
                obj["pair_number"], f"{path}.pair_number", minimum=1, maximum=3
            ),
            **{
                name: _text(obj[name], f"{path}.{name}", allow_blank=allow_blank)
                for name in required
                if name != "pair_number"
            },
        )


@dataclass(frozen=True, slots=True)
class AlternativeRuleBlueprint(StrictModel):
    rule: str
    how_the_training_pairs_rule_it_out: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "AlternativeRuleBlueprint":
        obj = _mapping(value, path)
        _keys(obj, path, ("rule", "how_the_training_pairs_rule_it_out"))
        return cls(
            rule=_text(obj["rule"], f"{path}.rule"),
            how_the_training_pairs_rule_it_out=_text(
                obj["how_the_training_pairs_rule_it_out"],
                f"{path}.how_the_training_pairs_rule_it_out",
            ),
        )


@dataclass(frozen=True, slots=True)
class PairBlueprint(StrictModel):
    candidate_id: str
    blueprint_viable: bool
    rejection_reasons: tuple[str, ...]
    intended_rule: str
    training_evidence_strategy: str
    joint_inference: str
    evidence_economy_notes: str
    training_pairs: tuple[TrainingPairBlueprint, ...]
    test_pairs: tuple[TestPairBlueprint, ...]
    plausible_alternative_rules: tuple[AlternativeRuleBlueprint, ...]
    realization_strategy: str
    fairness_risks: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "PairBlueprint":
        obj = _mapping(value, path)
        required = (
            "candidate_id",
            "blueprint_viable",
            "rejection_reasons",
            "intended_rule",
            "training_evidence_strategy",
            "joint_inference",
            "evidence_economy_notes",
            "training_pairs",
            "test_pairs",
            "plausible_alternative_rules",
            "realization_strategy",
            "fairness_risks",
        )
        _keys(obj, path, required)
        viable = _boolean(obj["blueprint_viable"], f"{path}.blueprint_viable")
        rejection_reasons = _text_array(
            obj["rejection_reasons"],
            f"{path}.rejection_reasons",
            minimum=0 if viable else 1,
            maximum=10,
        )
        if viable and rejection_reasons:
            raise SchemaError(f"{path}.rejection_reasons", "must be empty for a viable blueprint")
        pairs_raw = _array(
            obj["training_pairs"],
            f"{path}.training_pairs",
            minimum=1 if viable else 0,
            maximum=10,
        )
        pairs = tuple(
            TrainingPairBlueprint.from_dict(item, f"{path}.training_pairs[{i}]")
            for i, item in enumerate(pairs_raw)
        )
        numbers = tuple(pair.pair_number for pair in pairs)
        if numbers != tuple(range(1, len(pairs) + 1)):
            raise SchemaError(f"{path}.training_pairs", "pair_number values must be consecutive from 1")
        test_raw = _array(
            obj["test_pairs"],
            f"{path}.test_pairs",
            minimum=1 if viable else 0,
            maximum=3,
        )
        test_pairs = tuple(
            TestPairBlueprint.from_dict(
                item, f"{path}.test_pairs[{i}]", allow_blank=not viable
            )
            for i, item in enumerate(test_raw)
        )
        test_numbers = tuple(pair.pair_number for pair in test_pairs)
        if test_numbers != tuple(range(1, len(test_pairs) + 1)):
            raise SchemaError(
                f"{path}.test_pairs",
                "pair_number values must be consecutive from 1",
            )
        alternatives_raw = _array(obj["plausible_alternative_rules"], f"{path}.plausible_alternative_rules", maximum=10)
        alternatives = tuple(
            AlternativeRuleBlueprint.from_dict(item, f"{path}.plausible_alternative_rules[{i}]")
            for i, item in enumerate(alternatives_raw)
        )
        return cls(
            candidate_id=_text(obj["candidate_id"], f"{path}.candidate_id"),
            blueprint_viable=viable,
            rejection_reasons=rejection_reasons,
            intended_rule=_text(obj["intended_rule"], f"{path}.intended_rule", allow_blank=not viable),
            training_evidence_strategy=_text(
                obj["training_evidence_strategy"],
                f"{path}.training_evidence_strategy",
                allow_blank=not viable,
            ),
            joint_inference=_text(
                obj["joint_inference"],
                f"{path}.joint_inference",
                allow_blank=not viable,
            ),
            evidence_economy_notes=_text(
                obj["evidence_economy_notes"],
                f"{path}.evidence_economy_notes",
                allow_blank=not viable,
            ),
            training_pairs=pairs,
            test_pairs=test_pairs,
            plausible_alternative_rules=alternatives,
            realization_strategy=_text(
                obj["realization_strategy"],
                f"{path}.realization_strategy",
                allow_blank=not viable,
            ),
            fairness_risks=_text_array(obj["fairness_risks"], f"{path}.fairness_risks", maximum=10),
        )


@dataclass(frozen=True, slots=True)
class ARCExample(StrictModel):
    input: Grid
    output: Grid | None

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        path: str = "$",
        *,
        require_output: bool = True,
        forbid_output: bool = False,
    ) -> "ARCExample":
        obj = _mapping(value, path)
        if require_output and forbid_output:
            raise ValueError("require_output and forbid_output are mutually exclusive")
        if forbid_output:
            _keys(obj, path, ("input",))
        elif require_output:
            _keys(obj, path, ("input", "output"))
        else:
            _keys(obj, path, ("input",), ("output",))
        output = parse_grid(obj["output"], f"{path}.output") if "output" in obj else None
        return cls(input=parse_grid(obj["input"], f"{path}.input"), output=output)


@dataclass(frozen=True, slots=True)
class ARCTask(StrictModel):
    train: tuple[ARCExample, ...]
    test: tuple[ARCExample, ...]

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        path: str = "$",
        *,
        require_test_outputs: bool = True,
        forbid_test_outputs: bool = False,
        min_train_pairs: int = 1,
        max_train_pairs: int = 10,
        min_test_pairs: int = 1,
        max_test_pairs: int = 3,
    ) -> "ARCTask":
        obj = _mapping(value, path)
        _keys(obj, path, ("train", "test"))
        train_raw = _array(
            obj["train"], f"{path}.train", minimum=min_train_pairs, maximum=max_train_pairs
        )
        test_raw = _array(
            obj["test"], f"{path}.test", minimum=min_test_pairs, maximum=max_test_pairs
        )
        train = tuple(
            ARCExample.from_dict(item, f"{path}.train[{i}]", require_output=True)
            for i, item in enumerate(train_raw)
        )
        test = tuple(
            ARCExample.from_dict(
                item,
                f"{path}.test[{i}]",
                require_output=require_test_outputs,
                forbid_output=forbid_test_outputs,
            )
            for i, item in enumerate(test_raw)
        )
        return cls(train=train, test=test)


@dataclass(frozen=True, slots=True)
class ARCInput(StrictModel):
    """One solver-visible test item, intentionally lacking an output field."""

    input: Grid

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "ARCInput":
        obj = _mapping(value, path)
        _keys(obj, path, ("input",))
        return cls(input=parse_grid(obj["input"], f"{path}.input"))


@dataclass(frozen=True, slots=True)
class PublicARCTask(StrictModel):
    train: tuple[ARCExample, ...]
    test: tuple[ARCInput, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "PublicARCTask":
        obj = _mapping(value, path)
        _keys(obj, path, ("train", "test"))
        train_raw = _array(obj["train"], f"{path}.train", minimum=1, maximum=10)
        test_raw = _array(obj["test"], f"{path}.test", minimum=1, maximum=3)
        return cls(
            train=tuple(
                ARCExample.from_dict(item, f"{path}.train[{i}]", require_output=True)
                for i, item in enumerate(train_raw)
            ),
            test=tuple(
                ARCInput.from_dict(item, f"{path}.test[{i}]")
                for i, item in enumerate(test_raw)
            ),
        )


@dataclass(frozen=True, slots=True)
class PairConstructionTrace(StrictModel):
    pair: str
    trace: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "PairConstructionTrace":
        obj = _mapping(value, path)
        _keys(obj, path, ("pair", "trace"))
        return cls(
            pair=_text(obj["pair"], f"{path}.pair"),
            trace=_text(obj["trace"], f"{path}.trace"),
        )


@dataclass(frozen=True, slots=True)
class GenerationProvenance(StrictModel):
    candidate_id: str
    generation_call_id: str
    generation_mode: str
    seed_factors: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "GenerationProvenance":
        obj = _mapping(value, path)
        _keys(obj, path, ("candidate_id", "generation_call_id", "generation_mode", "seed_factors"))
        mode = _enum(obj["generation_mode"], f"{path}.generation_mode", GENERATION_MODES)
        factors = _text_array(obj["seed_factors"], f"{path}.seed_factors", maximum=2, unique=True)
        expected_count = {"open": 0, "single_factor": 1, "two_factor": 2}[mode]
        if len(factors) != expected_count:
            raise SchemaError(f"{path}.seed_factors", f"{mode!r} requires exactly {expected_count} factor(s)")
        return cls(
            candidate_id=_text(obj["candidate_id"], f"{path}.candidate_id"),
            generation_call_id=_text(obj["generation_call_id"], f"{path}.generation_call_id"),
            generation_mode=mode,
            seed_factors=factors,
        )


@dataclass(frozen=True, slots=True)
class AuthoringMetadata(StrictModel):
    intended_rule: str
    pair_blueprint: PairBlueprint
    expected_test_outputs: tuple[Grid, ...]
    intermediate_construction_representation: str
    pair_construction_trace: tuple[PairConstructionTrace, ...]
    generation_provenance: GenerationProvenance

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "AuthoringMetadata":
        obj = _mapping(value, path)
        required = (
            "intended_rule",
            "pair_blueprint",
            "expected_test_outputs",
            "intermediate_construction_representation",
            "pair_construction_trace",
            "generation_provenance",
        )
        _keys(obj, path, required)
        expected_raw = _array(
            obj["expected_test_outputs"],
            f"{path}.expected_test_outputs",
            minimum=1,
            maximum=3,
        )
        expected_outputs = tuple(
            parse_grid(item, f"{path}.expected_test_outputs[{i}]")
            for i, item in enumerate(expected_raw)
        )
        trace_raw = _array(
            obj["pair_construction_trace"],
            f"{path}.pair_construction_trace",
            minimum=2,
            maximum=13,
        )
        traces = tuple(
            PairConstructionTrace.from_dict(item, f"{path}.pair_construction_trace[{i}]")
            for i, item in enumerate(trace_raw)
        )
        if len({item.pair for item in traces}) != len(traces):
            raise SchemaError(f"{path}.pair_construction_trace", "pair labels must be unique")
        provenance = GenerationProvenance.from_dict(
            obj["generation_provenance"], f"{path}.generation_provenance"
        )
        return cls(
            intended_rule=_text(obj["intended_rule"], f"{path}.intended_rule"),
            pair_blueprint=PairBlueprint.from_dict(obj["pair_blueprint"], f"{path}.pair_blueprint"),
            expected_test_outputs=expected_outputs,
            intermediate_construction_representation=_text(
                obj["intermediate_construction_representation"],
                f"{path}.intermediate_construction_representation",
            ),
            pair_construction_trace=traces,
            generation_provenance=provenance,
        )


@dataclass(frozen=True, slots=True)
class AuthoredEpisode(StrictModel):
    """Public solver episode plus author-only metadata and expected output."""

    task: PublicARCTask
    authoring_metadata: AuthoringMetadata

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "AuthoredEpisode":
        obj = _mapping(value, path)
        _keys(obj, path, ("task", "authoring_metadata"))
        task = PublicARCTask.from_dict(obj["task"], f"{path}.task")
        metadata = AuthoringMetadata.from_dict(obj["authoring_metadata"], f"{path}.authoring_metadata")
        expected_trace_labels = {
            *(f"train_{i}" for i in range(1, len(task.train) + 1)),
            *(f"test_{i}" for i in range(1, len(task.test) + 1)),
        }
        actual_trace_labels = {item.pair for item in metadata.pair_construction_trace}
        if actual_trace_labels != expected_trace_labels:
            raise SchemaError(
                f"{path}.authoring_metadata.pair_construction_trace",
                f"must cover exactly {sorted(expected_trace_labels)!r}",
            )
        if len(metadata.pair_blueprint.training_pairs) != len(task.train):
            raise SchemaError(
                f"{path}.authoring_metadata.pair_blueprint.training_pairs",
                "blueprint and realized task must have the same training-pair count",
            )
        if len(metadata.pair_blueprint.test_pairs) != len(task.test):
            raise SchemaError(
                f"{path}.authoring_metadata.pair_blueprint.test_pairs",
                "blueprint and realized task must have the same test-pair count",
            )
        if len(metadata.expected_test_outputs) != len(task.test):
            raise SchemaError(
                f"{path}.authoring_metadata.expected_test_outputs",
                "must contain exactly one expected output for each public test input",
            )
        if not metadata.pair_blueprint.blueprint_viable:
            raise SchemaError(f"{path}.authoring_metadata.pair_blueprint", "realized blueprint must be viable")
        if metadata.generation_provenance.candidate_id != metadata.pair_blueprint.candidate_id:
            raise SchemaError(
                f"{path}.authoring_metadata.generation_provenance.candidate_id",
                "must match pair_blueprint.candidate_id",
            )
        return cls(task=task, authoring_metadata=metadata)

    def complete_task(self) -> ARCTask:
        """Return the author-only ARC task with the expected test output attached."""

        test = tuple(
            ARCExample(input=public.input, output=expected)
            for public, expected in zip(
                self.task.test,
                self.authoring_metadata.expected_test_outputs,
                strict=True,
            )
        )
        return ARCTask(train=self.task.train, test=test)


@dataclass(frozen=True, slots=True)
class RealizedTaskOutput(StrictModel):
    candidate_id: str
    task: PublicARCTask
    authoring_metadata: AuthoringMetadata

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "RealizedTaskOutput":
        obj = _mapping(value, path)
        _keys(obj, path, ("candidate_id", "task", "authoring_metadata"))
        candidate_id = _text(obj["candidate_id"], f"{path}.candidate_id")
        episode = AuthoredEpisode.from_dict(
            {"task": obj["task"], "authoring_metadata": obj["authoring_metadata"]},
            path,
        )
        if episode.authoring_metadata.pair_blueprint.candidate_id != candidate_id:
            raise SchemaError(
                f"{path}.authoring_metadata.pair_blueprint.candidate_id",
                "must match candidate_id",
            )
        return cls(
            candidate_id=candidate_id,
            task=episode.task,
            authoring_metadata=episode.authoring_metadata,
        )

    def episode(self) -> AuthoredEpisode:
        return AuthoredEpisode(task=self.task, authoring_metadata=self.authoring_metadata)

    def complete_task(self) -> ARCTask:
        return self.episode().complete_task()


@dataclass(frozen=True, slots=True)
class RepairLogEntry(StrictModel):
    defect: str
    change: str
    why_rule_is_preserved: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "RepairLogEntry":
        obj = _mapping(value, path)
        _keys(obj, path, ("defect", "change", "why_rule_is_preserved"))
        return cls(
            defect=_text(obj["defect"], f"{path}.defect"),
            change=_text(obj["change"], f"{path}.change"),
            why_rule_is_preserved=_text(obj["why_rule_is_preserved"], f"{path}.why_rule_is_preserved"),
        )


@dataclass(frozen=True, slots=True)
class RepairOutput(StrictModel):
    candidate_id: str
    repair_status: str
    unrepairable_reason: str
    repaired_task: AuthoredEpisode | None
    repair_log: tuple[RepairLogEntry, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "RepairOutput":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            ("candidate_id", "repair_status", "unrepairable_reason", "repaired_task", "repair_log"),
        )
        candidate_id = _text(obj["candidate_id"], f"{path}.candidate_id")
        status = _enum(obj["repair_status"], f"{path}.repair_status", ("repaired", "unrepairable"))
        reason = _text(obj["unrepairable_reason"], f"{path}.unrepairable_reason", allow_blank=True)
        raw_task = obj["repaired_task"]
        task = None if raw_task is None else AuthoredEpisode.from_dict(raw_task, f"{path}.repaired_task")
        log_raw = _array(obj["repair_log"], f"{path}.repair_log", maximum=20)
        log = tuple(RepairLogEntry.from_dict(item, f"{path}.repair_log[{i}]") for i, item in enumerate(log_raw))
        if status == "repaired":
            if task is None or not log or reason.strip():
                raise SchemaError(path, "repaired status requires a task and log, with empty unrepairable_reason")
            if task.authoring_metadata.pair_blueprint.candidate_id != candidate_id:
                raise SchemaError(f"{path}.repaired_task", "blueprint candidate_id must match candidate_id")
        elif task is not None or log or not reason.strip():
            raise SchemaError(path, "unrepairable status requires null task, empty log, and a nonempty reason")
        return cls(
            candidate_id=candidate_id,
            repair_status=status,
            unrepairable_reason=reason,
            repaired_task=task,
            repair_log=log,
        )


def _coordinate(value: Any, path: str) -> tuple[int, int]:
    raw = _array(value, path, minimum=2, maximum=2)
    return (
        _integer(raw[0], f"{path}[0]", minimum=0, maximum=29),
        _integer(raw[1], f"{path}[1]", minimum=0, maximum=29),
    )


@dataclass(frozen=True, slots=True)
class PairValidation(StrictModel):
    pair: str
    valid: bool
    rule_trace: str
    issues: tuple[str, ...]
    suspected_cell_discrepancies: tuple[tuple[int, int], ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "PairValidation":
        obj = _mapping(value, path)
        _keys(obj, path, ("pair", "valid", "rule_trace", "issues", "suspected_cell_discrepancies"))
        discrepancies_raw = _array(
            obj["suspected_cell_discrepancies"], f"{path}.suspected_cell_discrepancies", maximum=100
        )
        discrepancies = tuple(
            _coordinate(item, f"{path}.suspected_cell_discrepancies[{i}]")
            for i, item in enumerate(discrepancies_raw)
        )
        valid = _boolean(obj["valid"], f"{path}.valid")
        issues = _text_array(obj["issues"], f"{path}.issues", maximum=20)
        if valid and discrepancies:
            raise SchemaError(path, "a valid pair cannot contain suspected cell discrepancies")
        if not valid and not issues and not discrepancies:
            raise SchemaError(path, "an invalid pair must explain at least one issue")
        return cls(
            pair=_text(obj["pair"], f"{path}.pair"),
            valid=valid,
            rule_trace=_text(obj["rule_trace"], f"{path}.rule_trace"),
            issues=issues,
            suspected_cell_discrepancies=discrepancies,
        )


@dataclass(frozen=True, slots=True)
class RuleAwareValidation(StrictModel):
    candidate_id: str
    validator_id: str
    valid: bool
    pair_results: tuple[PairValidation, ...]
    all_pair_roles_instantiated: bool
    missing_or_misinstantiated_roles: tuple[str, ...]
    test_supported: bool
    test_only_semantics: tuple[str, ...]
    arbitrary_exceptions: tuple[str, ...]
    overall_issues: tuple[str, ...]
    confidence: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "RuleAwareValidation":
        obj = _mapping(value, path)
        required = (
            "candidate_id",
            "validator_id",
            "valid",
            "pair_results",
            "all_pair_roles_instantiated",
            "missing_or_misinstantiated_roles",
            "test_supported",
            "test_only_semantics",
            "arbitrary_exceptions",
            "overall_issues",
            "confidence",
        )
        _keys(obj, path, required)
        pair_raw = _array(obj["pair_results"], f"{path}.pair_results", minimum=2, maximum=13)
        pair_results = tuple(
            PairValidation.from_dict(item, f"{path}.pair_results[{i}]") for i, item in enumerate(pair_raw)
        )
        if len({pair.pair for pair in pair_results}) != len(pair_results):
            raise SchemaError(f"{path}.pair_results", "pair labels must be unique")
        valid = _boolean(obj["valid"], f"{path}.valid")
        roles = _boolean(obj["all_pair_roles_instantiated"], f"{path}.all_pair_roles_instantiated")
        missing_roles = _text_array(
            obj["missing_or_misinstantiated_roles"],
            f"{path}.missing_or_misinstantiated_roles",
            maximum=20,
        )
        if roles != (not missing_roles):
            raise SchemaError(
                f"{path}.all_pair_roles_instantiated",
                "must agree with missing_or_misinstantiated_roles",
            )
        test_supported = _boolean(obj["test_supported"], f"{path}.test_supported")
        test_only = _text_array(obj["test_only_semantics"], f"{path}.test_only_semantics", maximum=20)
        exceptions = _text_array(obj["arbitrary_exceptions"], f"{path}.arbitrary_exceptions", maximum=20)
        mechanically_valid = (
            all(pair.valid for pair in pair_results)
            and roles
            and test_supported
            and not test_only
            and not exceptions
        )
        if valid != mechanically_valid:
            raise SchemaError(
                f"{path}.valid",
                "must agree with pair validity, test support, and absence of arbitrary exceptions",
            )
        return cls(
            candidate_id=_text(obj["candidate_id"], f"{path}.candidate_id"),
            validator_id=_text(obj["validator_id"], f"{path}.validator_id"),
            valid=valid,
            pair_results=pair_results,
            all_pair_roles_instantiated=roles,
            missing_or_misinstantiated_roles=missing_roles,
            test_supported=test_supported,
            test_only_semantics=test_only,
            arbitrary_exceptions=exceptions,
            overall_issues=_text_array(obj["overall_issues"], f"{path}.overall_issues", maximum=20),
            confidence=_enum(obj["confidence"], f"{path}.confidence", CONFIDENCE_LEVELS),
        )

    def assert_pair_coverage(self, train_count: int, test_count: int) -> None:
        expected = {*(f"train_{i}" for i in range(1, train_count + 1)), *(f"test_{i}" for i in range(1, test_count + 1))}
        actual = {item.pair for item in self.pair_results}
        if actual != expected:
            raise SchemaError("$.pair_results", f"must cover exactly {sorted(expected)!r}")


@dataclass(frozen=True, slots=True)
class PlausibleRule(StrictModel):
    rule: str
    why_training_supports_it: str
    comparably_simple: bool
    predicted_test_outputs: tuple[Grid, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "PlausibleRule":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            ("rule", "why_training_supports_it", "comparably_simple", "predicted_test_outputs"),
        )
        predicted_raw = _array(
            obj["predicted_test_outputs"],
            f"{path}.predicted_test_outputs",
            minimum=1,
            maximum=3,
        )
        return cls(
            rule=_text(obj["rule"], f"{path}.rule"),
            why_training_supports_it=_text(
                obj["why_training_supports_it"], f"{path}.why_training_supports_it"
            ),
            comparably_simple=_boolean(obj["comparably_simple"], f"{path}.comparably_simple"),
            predicted_test_outputs=tuple(
                parse_grid(item, f"{path}.predicted_test_outputs[{i}]")
                for i, item in enumerate(predicted_raw)
            ),
        )


@dataclass(frozen=True, slots=True)
class BlindSolverOutput(StrictModel):
    task_id: str
    solver_id: str
    primary_inferred_rule: str
    predicted_test_outputs: tuple[Grid, ...]
    other_plausible_rules: tuple[PlausibleRule, ...]
    alternative_produces_different_test_output: bool
    fairness_assessment: str
    confidence: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "BlindSolverOutput":
        obj = _mapping(value, path)
        required = (
            "task_id",
            "solver_id",
            "primary_inferred_rule",
            "predicted_test_outputs",
            "other_plausible_rules",
            "alternative_produces_different_test_output",
            "fairness_assessment",
            "confidence",
        )
        _keys(obj, path, required)
        predicted_raw = _array(
            obj["predicted_test_outputs"],
            f"{path}.predicted_test_outputs",
            minimum=1,
            maximum=3,
        )
        predicted = tuple(
            parse_grid(item, f"{path}.predicted_test_outputs[{i}]")
            for i, item in enumerate(predicted_raw)
        )
        alternative_raw = _array(obj["other_plausible_rules"], f"{path}.other_plausible_rules", maximum=5)
        alternatives = tuple(
            PlausibleRule.from_dict(item, f"{path}.other_plausible_rules[{i}]")
            for i, item in enumerate(alternative_raw)
        )
        changed = _boolean(
            obj["alternative_produces_different_test_output"],
            f"{path}.alternative_produces_different_test_output",
        )
        for index, item in enumerate(alternatives):
            if len(item.predicted_test_outputs) != len(predicted):
                raise SchemaError(
                    f"{path}.other_plausible_rules[{index}].predicted_test_outputs",
                    "must contain one output for each test input",
                )
        actual_changed = any(item.predicted_test_outputs != predicted for item in alternatives)
        if changed != actual_changed:
            raise SchemaError(
                f"{path}.alternative_produces_different_test_output",
                "must agree with the supplied alternative predicted outputs",
            )
        return cls(
            task_id=_text(obj["task_id"], f"{path}.task_id"),
            solver_id=_text(obj["solver_id"], f"{path}.solver_id"),
            primary_inferred_rule=_text(obj["primary_inferred_rule"], f"{path}.primary_inferred_rule"),
            predicted_test_outputs=predicted,
            other_plausible_rules=alternatives,
            alternative_produces_different_test_output=changed,
            fairness_assessment=_text(obj["fairness_assessment"], f"{path}.fairness_assessment"),
            confidence=_enum(obj["confidence"], f"{path}.confidence", CONFIDENCE_LEVELS),
        )


EVIDENCE_SUPPORT_LEVELS = ("sufficient", "weak", "absent")


@dataclass(frozen=True, slots=True)
class DescriptionAwarePairCheck(StrictModel):
    pair: str
    follows_supplied_rule: bool
    instantiates_author_specification: bool
    issues: tuple[str, ...]
    suspected_cell_discrepancies: tuple[tuple[int, int], ...]

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], path: str = "$"
    ) -> "DescriptionAwarePairCheck":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            (
                "pair",
                "follows_supplied_rule",
                "instantiates_author_specification",
                "issues",
                "suspected_cell_discrepancies",
            ),
        )
        follows = _boolean(
            obj["follows_supplied_rule"], f"{path}.follows_supplied_rule"
        )
        instantiates = _boolean(
            obj["instantiates_author_specification"],
            f"{path}.instantiates_author_specification",
        )
        issues = _text_array(obj["issues"], f"{path}.issues", maximum=20)
        discrepancy_raw = _array(
            obj["suspected_cell_discrepancies"],
            f"{path}.suspected_cell_discrepancies",
            maximum=100,
        )
        discrepancies = tuple(
            _coordinate(item, f"{path}.suspected_cell_discrepancies[{index}]")
            for index, item in enumerate(discrepancy_raw)
        )
        if follows and instantiates and (issues or discrepancies):
            raise SchemaError(
                path,
                "a fully valid training pair cannot contain issues or discrepancies",
            )
        if (not follows or not instantiates) and not issues and not discrepancies:
            raise SchemaError(
                path,
                "a failed training-pair check must explain an issue or discrepancy",
            )
        return cls(
            pair=_text(obj["pair"], f"{path}.pair"),
            follows_supplied_rule=follows,
            instantiates_author_specification=instantiates,
            issues=issues,
            suspected_cell_discrepancies=discrepancies,
        )


@dataclass(frozen=True, slots=True)
class RuleClauseEvidence(StrictModel):
    rule_clause: str
    visible_support: str
    support: str

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], path: str = "$"
    ) -> "RuleClauseEvidence":
        obj = _mapping(value, path)
        _keys(obj, path, ("rule_clause", "visible_support", "support"))
        return cls(
            rule_clause=_text(obj["rule_clause"], f"{path}.rule_clause"),
            visible_support=_text(obj["visible_support"], f"{path}.visible_support"),
            support=_enum(
                obj["support"], f"{path}.support", EVIDENCE_SUPPORT_LEVELS
            ),
        )


@dataclass(frozen=True, slots=True)
class DescriptionAwareAlternative(StrictModel):
    rule: str
    fits_all_training_pairs: bool
    comparably_simple: bool
    comparably_well_supported: bool
    predicted_test_outputs: tuple[Grid, ...]
    assessment: str

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], path: str = "$"
    ) -> "DescriptionAwareAlternative":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            (
                "rule",
                "fits_all_training_pairs",
                "comparably_simple",
                "comparably_well_supported",
                "predicted_test_outputs",
                "assessment",
            ),
        )
        outputs_raw = _array(
            obj["predicted_test_outputs"],
            f"{path}.predicted_test_outputs",
            minimum=1,
            maximum=3,
        )
        return cls(
            rule=_text(obj["rule"], f"{path}.rule"),
            fits_all_training_pairs=_boolean(
                obj["fits_all_training_pairs"],
                f"{path}.fits_all_training_pairs",
            ),
            comparably_simple=_boolean(
                obj["comparably_simple"], f"{path}.comparably_simple"
            ),
            comparably_well_supported=_boolean(
                obj["comparably_well_supported"],
                f"{path}.comparably_well_supported",
            ),
            predicted_test_outputs=tuple(
                parse_grid(item, f"{path}.predicted_test_outputs[{index}]")
                for index, item in enumerate(outputs_raw)
            ),
            assessment=_text(obj["assessment"], f"{path}.assessment"),
        )


@dataclass(frozen=True, slots=True)
class DescriptionAwareCheckerOutput(StrictModel):
    fixture_id: str
    predicted_test_outputs: tuple[Grid, ...]
    training_pair_checks: tuple[DescriptionAwarePairCheck, ...]
    rule_clause_evidence: tuple[RuleClauseEvidence, ...]
    test_only_semantics: tuple[str, ...]
    serious_alternatives: tuple[DescriptionAwareAlternative, ...]
    distinct_serious_test_output_counts: tuple[int, ...]
    answer_class: str
    rule_is_mechanically_realized: bool
    rule_is_sufficiently_communicated: bool
    revision_recommended: bool
    revision_reason: str
    confidence: str

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], path: str = "$"
    ) -> "DescriptionAwareCheckerOutput":
        obj = _mapping(value, path)
        required = (
            "fixture_id",
            "predicted_test_outputs",
            "training_pair_checks",
            "rule_clause_evidence",
            "test_only_semantics",
            "serious_alternatives",
            "distinct_serious_test_output_counts",
            "answer_class",
            "rule_is_mechanically_realized",
            "rule_is_sufficiently_communicated",
            "revision_recommended",
            "revision_reason",
            "confidence",
        )
        _keys(obj, path, required)
        predicted_raw = _array(
            obj["predicted_test_outputs"],
            f"{path}.predicted_test_outputs",
            minimum=1,
            maximum=3,
        )
        predicted = tuple(
            parse_grid(item, f"{path}.predicted_test_outputs[{index}]")
            for index, item in enumerate(predicted_raw)
        )
        pair_raw = _array(
            obj["training_pair_checks"],
            f"{path}.training_pair_checks",
            minimum=1,
            maximum=10,
        )
        pair_checks = tuple(
            DescriptionAwarePairCheck.from_dict(
                item, f"{path}.training_pair_checks[{index}]"
            )
            for index, item in enumerate(pair_raw)
        )
        if len({item.pair for item in pair_checks}) != len(pair_checks):
            raise SchemaError(
                f"{path}.training_pair_checks", "pair labels must be unique"
            )
        evidence_raw = _array(
            obj["rule_clause_evidence"],
            f"{path}.rule_clause_evidence",
            minimum=1,
            maximum=20,
        )
        evidence = tuple(
            RuleClauseEvidence.from_dict(
                item, f"{path}.rule_clause_evidence[{index}]"
            )
            for index, item in enumerate(evidence_raw)
        )
        test_only = _text_array(
            obj["test_only_semantics"],
            f"{path}.test_only_semantics",
            maximum=20,
        )
        alternative_raw = _array(
            obj["serious_alternatives"],
            f"{path}.serious_alternatives",
            maximum=10,
        )
        alternatives = tuple(
            DescriptionAwareAlternative.from_dict(
                item, f"{path}.serious_alternatives[{index}]"
            )
            for index, item in enumerate(alternative_raw)
        )
        for index, alternative in enumerate(alternatives):
            if len(alternative.predicted_test_outputs) != len(predicted):
                raise SchemaError(
                    f"{path}.serious_alternatives[{index}].predicted_test_outputs",
                    "must contain one output for each test input",
                )
        mechanically_realized = _boolean(
            obj["rule_is_mechanically_realized"],
            f"{path}.rule_is_mechanically_realized",
        )
        expected_mechanical = all(
            item.follows_supplied_rule
            and item.instantiates_author_specification
            for item in pair_checks
        )
        if mechanically_realized != expected_mechanical:
            raise SchemaError(
                f"{path}.rule_is_mechanically_realized",
                "must agree with all training_pair_checks",
            )
        communicated = _boolean(
            obj["rule_is_sufficiently_communicated"],
            f"{path}.rule_is_sufficiently_communicated",
        )
        expected_communicated = (
            all(item.support == "sufficient" for item in evidence)
            and not test_only
        )
        if communicated != expected_communicated:
            raise SchemaError(
                f"{path}.rule_is_sufficiently_communicated",
                "must agree with rule-clause evidence and test-only semantics",
            )
        serious_outputs: list[set[Grid]] = [{output} for output in predicted]
        for alternative in alternatives:
            if (
                alternative.fits_all_training_pairs
                and alternative.comparably_simple
                and alternative.comparably_well_supported
            ):
                for test_index, output in enumerate(
                    alternative.predicted_test_outputs
                ):
                    serious_outputs[test_index].add(output)
        actual_counts = tuple(len(outputs) for outputs in serious_outputs)
        counts_raw = _array(
            obj["distinct_serious_test_output_counts"],
            f"{path}.distinct_serious_test_output_counts",
            minimum=len(predicted),
            maximum=len(predicted),
        )
        supplied_counts = tuple(
            _integer(
                item,
                f"{path}.distinct_serious_test_output_counts[{index}]",
                minimum=1,
                maximum=11,
            )
            for index, item in enumerate(counts_raw)
        )
        if supplied_counts != actual_counts:
            raise SchemaError(
                f"{path}.distinct_serious_test_output_counts",
                f"must equal {actual_counts!r} from primary and serious alternatives",
            )
        expected_answer_class = (
            "underdetermined"
            if not communicated or any(count > 2 for count in actual_counts)
            else (
                "one_guess"
                if all(count == 1 for count in actual_counts)
                else "bounded_two_guess"
            )
        )
        answer_class = _enum(
            obj["answer_class"], f"{path}.answer_class", ANSWER_CLASSES
        )
        if answer_class != expected_answer_class:
            raise SchemaError(
                f"{path}.answer_class",
                f"cross-field findings require {expected_answer_class!r}",
            )
        revision = _boolean(
            obj["revision_recommended"], f"{path}.revision_recommended"
        )
        expected_revision = not mechanically_realized or answer_class == "underdetermined"
        if revision != expected_revision:
            raise SchemaError(
                f"{path}.revision_recommended",
                "must be true exactly for a mechanical defect or underdetermination",
            )
        revision_reason = _text(
            obj["revision_reason"],
            f"{path}.revision_reason",
            allow_blank=True,
        )
        if revision != bool(revision_reason.strip()):
            raise SchemaError(
                f"{path}.revision_reason",
                "must be nonempty exactly when revision_recommended is true",
            )
        return cls(
            fixture_id=_text(obj["fixture_id"], f"{path}.fixture_id"),
            predicted_test_outputs=predicted,
            training_pair_checks=pair_checks,
            rule_clause_evidence=evidence,
            test_only_semantics=test_only,
            serious_alternatives=alternatives,
            distinct_serious_test_output_counts=supplied_counts,
            answer_class=answer_class,
            rule_is_mechanically_realized=mechanically_realized,
            rule_is_sufficiently_communicated=communicated,
            revision_recommended=revision,
            revision_reason=revision_reason,
            confidence=_enum(
                obj["confidence"], f"{path}.confidence", CONFIDENCE_LEVELS
            ),
        )

    def assert_context(self, train_count: int, test_count: int) -> None:
        expected_pairs = {
            f"train_{index}" for index in range(1, train_count + 1)
        }
        actual_pairs = {item.pair for item in self.training_pair_checks}
        if actual_pairs != expected_pairs:
            raise SchemaError(
                "$.training_pair_checks",
                f"must cover exactly {sorted(expected_pairs)!r}",
            )
        if len(self.predicted_test_outputs) != test_count:
            raise SchemaError(
                "$.predicted_test_outputs",
                f"must contain exactly {test_count} test output(s)",
            )


@dataclass(frozen=True, slots=True)
class EvaluatedAlternativeRule(StrictModel):
    rule: str
    fits_all_training_pairs: bool
    comparably_simple: bool
    comparably_well_supported: bool
    predicts_different_test_output: bool
    predicted_test_outputs: tuple[Grid, ...]
    assessment: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "EvaluatedAlternativeRule":
        obj = _mapping(value, path)
        _keys(
            obj,
            path,
            (
                "rule",
                "fits_all_training_pairs",
                "comparably_simple",
                "comparably_well_supported",
                "predicts_different_test_output",
                "predicted_test_outputs",
                "assessment",
            ),
        )
        predicted_raw = _array(
            obj["predicted_test_outputs"],
            f"{path}.predicted_test_outputs",
            minimum=1,
            maximum=3,
        )
        return cls(
            rule=_text(obj["rule"], f"{path}.rule"),
            fits_all_training_pairs=_boolean(
                obj["fits_all_training_pairs"], f"{path}.fits_all_training_pairs"
            ),
            comparably_simple=_boolean(obj["comparably_simple"], f"{path}.comparably_simple"),
            comparably_well_supported=_boolean(
                obj["comparably_well_supported"], f"{path}.comparably_well_supported"
            ),
            predicts_different_test_output=_boolean(
                obj["predicts_different_test_output"], f"{path}.predicts_different_test_output"
            ),
            predicted_test_outputs=tuple(
                parse_grid(item, f"{path}.predicted_test_outputs[{i}]")
                for i, item in enumerate(predicted_raw)
            ),
            assessment=_text(obj["assessment"], f"{path}.assessment"),
        )


@dataclass(frozen=True, slots=True)
class AmbiguityAdjudication(StrictModel):
    candidate_id: str
    intended_rule_reasonably_supported: bool
    intended_test_outputs: tuple[Grid, ...]
    serious_competing_alternative_predicts_different_output: bool
    distinct_serious_test_output_counts: tuple[int, ...]
    solvable_within_two_guesses: bool
    answer_class: str
    difficult_but_fair: bool
    underdetermined: bool
    training_pair_should_be_revised: bool
    evaluated_alternatives: tuple[EvaluatedAlternativeRule, ...]
    decision: str
    recommended_revision: str
    reason: str
    confidence: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "AmbiguityAdjudication":
        obj = _mapping(value, path)
        required = (
            "candidate_id",
            "intended_rule_reasonably_supported",
            "intended_test_outputs",
            "serious_competing_alternative_predicts_different_output",
            "distinct_serious_test_output_counts",
            "solvable_within_two_guesses",
            "answer_class",
            "difficult_but_fair",
            "underdetermined",
            "training_pair_should_be_revised",
            "evaluated_alternatives",
            "decision",
            "recommended_revision",
            "reason",
            "confidence",
        )
        _keys(obj, path, required)
        supported = _boolean(
            obj["intended_rule_reasonably_supported"],
            f"{path}.intended_rule_reasonably_supported",
        )
        intended_raw = _array(
            obj["intended_test_outputs"],
            f"{path}.intended_test_outputs",
            minimum=1,
            maximum=3,
        )
        intended_outputs = tuple(
            parse_grid(item, f"{path}.intended_test_outputs[{i}]")
            for i, item in enumerate(intended_raw)
        )
        serious_alternative = _boolean(
            obj["serious_competing_alternative_predicts_different_output"],
            f"{path}.serious_competing_alternative_predicts_different_output",
        )
        count_raw = _array(
            obj["distinct_serious_test_output_counts"],
            f"{path}.distinct_serious_test_output_counts",
            minimum=len(intended_outputs),
            maximum=len(intended_outputs),
        )
        supplied_output_counts = tuple(
            _integer(
                item,
                f"{path}.distinct_serious_test_output_counts[{i}]",
                minimum=1,
                maximum=11,
            )
            for i, item in enumerate(count_raw)
        )
        supplied_two_guess_solvability = _boolean(
            obj["solvable_within_two_guesses"],
            f"{path}.solvable_within_two_guesses",
        )
        answer_class = _enum(
            obj["answer_class"], f"{path}.answer_class", ANSWER_CLASSES
        )
        fair = _boolean(obj["difficult_but_fair"], f"{path}.difficult_but_fair")
        underdetermined = _boolean(obj["underdetermined"], f"{path}.underdetermined")
        revise = _boolean(obj["training_pair_should_be_revised"], f"{path}.training_pair_should_be_revised")
        decision = _enum(obj["decision"], f"{path}.decision", AMBIGUITY_DECISIONS)
        alternatives_raw = _array(obj["evaluated_alternatives"], f"{path}.evaluated_alternatives", maximum=10)
        alternatives = tuple(
            EvaluatedAlternativeRule.from_dict(item, f"{path}.evaluated_alternatives[{i}]")
            for i, item in enumerate(alternatives_raw)
        )
        serious_outputs: list[set[Grid]] = [
            {output} for output in intended_outputs
        ]
        actual_serious_alternative = False
        for index, item in enumerate(alternatives):
            if len(item.predicted_test_outputs) != len(intended_outputs):
                raise SchemaError(
                    f"{path}.evaluated_alternatives[{index}].predicted_test_outputs",
                    "must contain one output for each test input",
                )
            differs = item.predicted_test_outputs != intended_outputs
            if item.predicts_different_test_output != differs:
                raise SchemaError(
                    f"{path}.evaluated_alternatives[{index}].predicts_different_test_output",
                    "must agree with predicted_test_outputs versus intended_test_outputs",
                )
            serious = (
                item.fits_all_training_pairs
                and item.comparably_simple
                and item.comparably_well_supported
            )
            if serious:
                for test_index, output in enumerate(item.predicted_test_outputs):
                    serious_outputs[test_index].add(output)
                actual_serious_alternative = actual_serious_alternative or differs
        if serious_alternative != actual_serious_alternative:
            raise SchemaError(
                f"{path}.serious_competing_alternative_predicts_different_output",
                "must agree with evaluated_alternatives",
            )
        actual_output_counts = tuple(len(outputs) for outputs in serious_outputs)
        if supplied_output_counts != actual_output_counts:
            raise SchemaError(
                f"{path}.distinct_serious_test_output_counts",
                f"must equal {actual_output_counts!r} from intended and serious alternative outputs",
            )
        expected_two_guess_solvability = supported and all(
            count <= 2 for count in actual_output_counts
        )
        if supplied_two_guess_solvability != expected_two_guess_solvability:
            raise SchemaError(
                f"{path}.solvable_within_two_guesses",
                "must agree with intended-rule support and the two-output limit",
            )
        expected_answer_class = (
            "underdetermined"
            if not expected_two_guess_solvability
            else (
                "one_guess"
                if all(count == 1 for count in actual_output_counts)
                else "bounded_two_guess"
            )
        )
        if answer_class != expected_answer_class:
            raise SchemaError(
                f"{path}.answer_class",
                f"cross-field findings require answer_class {expected_answer_class!r}",
            )
        expected_underdetermined = not expected_two_guess_solvability
        if underdetermined != expected_underdetermined:
            raise SchemaError(
                f"{path}.underdetermined",
                "must be true exactly when the task is not solvable within two serious output guesses",
            )
        adverse_finding = underdetermined or not supplied_two_guess_solvability or not fair
        if revise and not adverse_finding:
            raise SchemaError(
                f"{path}.training_pair_should_be_revised",
                "may be true only when at least one ambiguity or fairness finding requires revision",
            )
        expected = "revise" if revise else ("reject" if adverse_finding else "pass")
        if decision != expected:
            raise SchemaError(f"{path}.decision", f"cross-field findings require decision {expected!r}")
        revision = _text(obj["recommended_revision"], f"{path}.recommended_revision", allow_blank=True)
        if revise != bool(revision.strip()):
            raise SchemaError(
                f"{path}.recommended_revision",
                "must be nonempty exactly when training_pair_should_be_revised is true",
            )
        return cls(
            candidate_id=_text(obj["candidate_id"], f"{path}.candidate_id"),
            intended_rule_reasonably_supported=supported,
            intended_test_outputs=intended_outputs,
            serious_competing_alternative_predicts_different_output=serious_alternative,
            distinct_serious_test_output_counts=supplied_output_counts,
            solvable_within_two_guesses=supplied_two_guess_solvability,
            answer_class=answer_class,
            difficult_but_fair=fair,
            underdetermined=underdetermined,
            training_pair_should_be_revised=revise,
            evaluated_alternatives=alternatives,
            decision=decision,
            recommended_revision=revision,
            reason=_text(obj["reason"], f"{path}.reason"),
            confidence=_enum(obj["confidence"], f"{path}.confidence", CONFIDENCE_LEVELS),
        )


@dataclass(frozen=True, slots=True)
class FactorActivationAssessment(StrictModel):
    factor: str
    status: str
    explanation: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "FactorActivationAssessment":
        obj = _mapping(value, path)
        _keys(obj, path, ("factor", "status", "explanation"))
        return cls(
            factor=_text(obj["factor"], f"{path}.factor"),
            status=_enum(obj["status"], f"{path}.status", ACTIVATION_STATUSES),
            explanation=_text(obj["explanation"], f"{path}.explanation"),
        )


@dataclass(frozen=True, slots=True)
class FactorActivationOutput(StrictModel):
    candidate_id: str
    seed_factor_activation: tuple[FactorActivationAssessment, ...]
    seeded_mode_success: bool | None
    observed_factors: tuple[ObservedFactor, ...]
    new_factor_labels: tuple[str, ...]
    overall_reason: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "FactorActivationOutput":
        obj = _mapping(value, path)
        required = (
            "candidate_id",
            "seed_factor_activation",
            "seeded_mode_success",
            "observed_factors",
            "new_factor_labels",
            "overall_reason",
        )
        _keys(obj, path, required)
        raw = _array(obj["seed_factor_activation"], f"{path}.seed_factor_activation", maximum=2)
        assessments = tuple(
            FactorActivationAssessment.from_dict(item, f"{path}.seed_factor_activation[{i}]")
            for i, item in enumerate(raw)
        )
        if len({item.factor for item in assessments}) != len(assessments):
            raise SchemaError(f"{path}.seed_factor_activation", "factors must be unique")
        raw_success = obj["seeded_mode_success"]
        if assessments:
            succeeds: bool | None = _boolean(raw_success, f"{path}.seeded_mode_success")
            actually_active = all(item.status in ("central", "supporting") for item in assessments)
            if succeeds != actually_active:
                raise SchemaError(
                    f"{path}.seeded_mode_success",
                    "must be true exactly when every seed factor is central or supporting",
                )
        elif raw_success is not None:
            raise SchemaError(
                f"{path}.seeded_mode_success",
                "must be null for an unseeded task",
            )
        else:
            succeeds = None
        observed_raw = _array(obj["observed_factors"], f"{path}.observed_factors")
        observed = tuple(
            ObservedFactor.from_dict(item, f"{path}.observed_factors[{i}]")
            for i, item in enumerate(observed_raw)
        )
        if len({item.label for item in observed}) != len(observed):
            raise SchemaError(f"{path}.observed_factors", "factor labels must be unique")
        new_labels = _text_array(
            obj["new_factor_labels"], f"{path}.new_factor_labels", unique=True
        )
        if not set(new_labels) <= {item.label for item in observed}:
            raise SchemaError(f"{path}.new_factor_labels", "each new label must also appear in observed_factors")
        return cls(
            candidate_id=_text(obj["candidate_id"], f"{path}.candidate_id"),
            seed_factor_activation=assessments,
            seeded_mode_success=succeeds,
            observed_factors=observed,
            new_factor_labels=new_labels,
            overall_reason=_text(obj["overall_reason"], f"{path}.overall_reason"),
        )

    def assert_seed_factors(self, expected: Sequence[str]) -> None:
        actual = tuple(item.factor for item in self.seed_factor_activation)
        if actual != tuple(expected):
            raise SchemaError("$.seed_factor_activation", "must exactly preserve supplied raw seed labels and order")


@dataclass(frozen=True, slots=True)
class RankingEntry(StrictModel):
    task_id: str
    rank: int
    rationale: str
    flags: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "RankingEntry":
        obj = _mapping(value, path)
        _keys(obj, path, ("task_id", "rank", "rationale", "flags"))
        flags_raw = _array(obj["flags"], f"{path}.flags", maximum=len(QUALITY_FLAGS))
        parsed_flags = tuple(_enum(item, f"{path}.flags[{i}]", QUALITY_FLAGS) for i, item in enumerate(flags_raw))
        if len(set(parsed_flags)) != len(parsed_flags):
            raise SchemaError(f"{path}.flags", "flags must be unique")
        return cls(
            task_id=_text(obj["task_id"], f"{path}.task_id"),
            rank=_integer(obj["rank"], f"{path}.rank", minimum=1, maximum=16),
            rationale=_text(obj["rationale"], f"{path}.rationale"),
            flags=parsed_flags,
        )


@dataclass(frozen=True, slots=True)
class QualityRankingOutput(StrictModel):
    ranker_id: str
    ranking: tuple[RankingEntry, ...]
    batch_observations: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "$") -> "QualityRankingOutput":
        obj = _mapping(value, path)
        _keys(obj, path, ("ranker_id", "ranking", "batch_observations"))
        raw = _array(obj["ranking"], f"{path}.ranking", minimum=1, maximum=16)
        ranking = tuple(RankingEntry.from_dict(item, f"{path}.ranking[{i}]") for i, item in enumerate(raw))
        if len({entry.task_id for entry in ranking}) != len(ranking):
            raise SchemaError(f"{path}.ranking", "task_id values must be unique")
        if tuple(entry.rank for entry in ranking) != tuple(range(1, len(ranking) + 1)):
            raise SchemaError(f"{path}.ranking", "entries must be ordered with consecutive ranks starting at 1")
        return cls(
            ranker_id=_text(obj["ranker_id"], f"{path}.ranker_id"),
            ranking=ranking,
            batch_observations=_text(obj["batch_observations"], f"{path}.batch_observations"),
        )


# -- JSON Schemas ---------------------------------------------------------


def _object_schema(properties: Mapping[str, Any], required: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required if required is not None else properties.keys()),
        "additionalProperties": False,
    }


TEXT_SCHEMA = {"type": "string", "minLength": 1}
GRID_SCHEMA: dict[str, Any] = {
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

ARC_PAIR_SCHEMA = _object_schema({"input": GRID_SCHEMA, "output": GRID_SCHEMA})
ARC_TASK_SCHEMA = _object_schema(
    {
        "train": {"type": "array", "minItems": 1, "maxItems": 10, "items": ARC_PAIR_SCHEMA},
        "test": {"type": "array", "minItems": 1, "maxItems": 3, "items": ARC_PAIR_SCHEMA},
    }
)

IDEA_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "generation_mode": {"type": "string", "enum": list(GENERATION_MODES)},
        "seed_factors": {"type": "array", "maxItems": 2, "uniqueItems": True, "items": TEXT_SCHEMA},
        "central_relation": TEXT_SCHEMA,
        "why_coherent": TEXT_SCHEMA,
        "training_evidence_strategy": TEXT_SCHEMA,
        "episode_level_inference": TEXT_SCHEMA,
        "where_novelty_resides": TEXT_SCHEMA,
        "test_generalization": TEXT_SCHEMA,
        "difference_from_calibration_examples": TEXT_SCHEMA,
        "feasibility_notes": TEXT_SCHEMA,
    }
)
IDEA_GENERATOR_SCHEMA = _object_schema(
    {
        "generation_call_id": TEXT_SCHEMA,
        "generation_mode": {"type": "string", "enum": list(GENERATION_MODES)},
        "seed_factors": {"type": "array", "maxItems": 2, "uniqueItems": True, "items": TEXT_SCHEMA},
        "factor_combination_feasible": {"type": "boolean"},
        "factor_combination_infeasibility_reason": {"type": "string"},
        "ideas": {"type": "array", "maxItems": 4, "items": IDEA_SCHEMA},
    }
)

OBSERVED_FACTOR_SCHEMA = _object_schema({"label": TEXT_SCHEMA, "manifestation": TEXT_SCHEMA})
SEED_ASSESSMENT_SCHEMA = _object_schema(
    {
        "factor": TEXT_SCHEMA,
        "status": {"type": "string", "enum": list(SEED_FACTOR_STATUSES)},
        "reason": TEXT_SCHEMA,
        "substantively_new_reinterpretation": {"type": "boolean"},
        "reinterpretation_explanation": TEXT_SCHEMA,
    }
)
DIFFICULTY_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "observed_factors": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": OBSERVED_FACTOR_SCHEMA,
        },
        "seed_factor_assessment": {"type": "array", "maxItems": 2, "items": SEED_ASSESSMENT_SCHEMA},
        "new_factor_labels": {
            "type": "array",
            "maxItems": 5,
            "uniqueItems": True,
            "items": TEXT_SCHEMA,
        },
    }
)

DIVERSITY_PAIRWISE_SCHEMA = _object_schema(
    {
        "candidate_id_a": TEXT_SCHEMA,
        "candidate_id_b": TEXT_SCHEMA,
        "essentially_same": {"type": "boolean"},
        "shared_minimal_solution_procedure": TEXT_SCHEMA,
        "material_difference": TEXT_SCHEMA,
    }
)
DUPLICATE_CLUSTER_SCHEMA = _object_schema(
    {
        "candidate_ids": {
            "type": "array",
            "minItems": 2,
            "uniqueItems": True,
            "items": TEXT_SCHEMA,
        },
        "representative_candidate_id": TEXT_SCHEMA,
        "reason": TEXT_SCHEMA,
    }
)
DUPLICATE_REJECTION_SCHEMA = _object_schema(
    {"candidate_id": TEXT_SCHEMA, "duplicate_of": TEXT_SCHEMA, "reason": TEXT_SCHEMA}
)
DIVERSITY_SCHEMA = _object_schema(
    {
        "group_id": TEXT_SCHEMA,
        "pairwise_decisions": {
            "type": "array",
            "minItems": 1,
            "items": DIVERSITY_PAIRWISE_SCHEMA,
        },
        "duplicate_clusters": {"type": "array", "items": DUPLICATE_CLUSTER_SCHEMA},
        "keep_candidate_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": TEXT_SCHEMA,
        },
        "reject_as_duplicate": {"type": "array", "items": DUPLICATE_REJECTION_SCHEMA},
    }
)

PORTFOLIO_REFERENCE_ASSESSMENT_SCHEMA = _object_schema(
    {
        "reference_id": TEXT_SCHEMA,
        "essentially_same": {"type": "boolean"},
        "shared_minimal_solution_procedure": TEXT_SCHEMA,
        "material_difference": TEXT_SCHEMA,
        "preferred_representative": {
            "type": "string",
            "enum": ["candidate", "reference"],
        },
    }
)
PORTFOLIO_DIVERSITY_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "reference_assessments": {
            "type": "array",
            "minItems": 1,
            "items": PORTFOLIO_REFERENCE_ASSESSMENT_SCHEMA,
        },
        "duplicate_reference_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": TEXT_SCHEMA,
        },
        "representative_id": TEXT_SCHEMA,
        "decision": {
            "type": "string",
            "enum": [
                "distinct",
                "reject_candidate",
                "replace_references",
            ],
        },
        "reason": TEXT_SCHEMA,
    }
)

NOVELTY_CONFLICT_SCHEMA = _object_schema(
    {
        "reference_task_id": TEXT_SCHEMA,
        "similarity_type": {"type": "string", "enum": list(SIMILARITY_TYPES)},
        "shared_mechanism": TEXT_SCHEMA,
        "material_difference": TEXT_SCHEMA,
    }
)
NOVELTY_SCOUT_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "scout_id": TEXT_SCHEMA,
        "possible_conflicts": {"type": "array", "maxItems": 5, "items": NOVELTY_CONFLICT_SCHEMA},
        "provisional_novelty_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "reason": TEXT_SCHEMA,
    }
)
NOVELTY_ADJUDICATION_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "final_novelty_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "passes_novelty": {"type": "boolean"},
        "strongest_conflicts": {"type": "array", "maxItems": 5, "items": NOVELTY_CONFLICT_SCHEMA},
        "reason": TEXT_SCHEMA,
    }
)

TRAIN_BLUEPRINT_SCHEMA = _object_schema(
    {
        "pair_number": {"type": "integer", "minimum": 1, "maximum": 10},
        "what_it_teaches": TEXT_SCHEMA,
        "why_it_is_needed": TEXT_SCHEMA,
        "what_varies": TEXT_SCHEMA,
        "what_remains_invariant": TEXT_SCHEMA,
        "plausible_wrong_rule_it_eliminates": TEXT_SCHEMA,
        "grid_construction_plan": TEXT_SCHEMA,
    }
)
TEST_BLUEPRINT_SCHEMA = _object_schema(
    {
        "pair_number": {"type": "integer", "minimum": 1, "maximum": 3},
        "construction_plan": {"type": "string"},
        "what_is_new": {"type": "string"},
        "what_training_evidence_supports_it": {"type": "string"},
        "why_the_output_is_determined": {"type": "string"},
    }
)
ALT_RULE_BLUEPRINT_SCHEMA = _object_schema(
    {"rule": TEXT_SCHEMA, "how_the_training_pairs_rule_it_out": TEXT_SCHEMA}
)
BLUEPRINT_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "blueprint_viable": {"type": "boolean"},
        "rejection_reasons": {"type": "array", "maxItems": 10, "items": TEXT_SCHEMA},
        "intended_rule": {"type": "string"},
        "training_evidence_strategy": {"type": "string"},
        "joint_inference": {"type": "string"},
        "evidence_economy_notes": {"type": "string"},
        "training_pairs": {
            "type": "array",
            "maxItems": 10,
            "items": TRAIN_BLUEPRINT_SCHEMA,
        },
        "test_pairs": {
            "type": "array",
            "maxItems": 3,
            "items": TEST_BLUEPRINT_SCHEMA,
        },
        "plausible_alternative_rules": {
            "type": "array",
            "maxItems": 10,
            "items": ALT_RULE_BLUEPRINT_SCHEMA,
        },
        "realization_strategy": {"type": "string"},
        "fairness_risks": {"type": "array", "maxItems": 10, "items": TEXT_SCHEMA},
    }
)

PUBLIC_TEST_PAIR_SCHEMA = _object_schema({"input": GRID_SCHEMA})
PUBLIC_ARC_TASK_SCHEMA = _object_schema(
    {
        "train": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": ARC_PAIR_SCHEMA,
        },
        "test": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": PUBLIC_TEST_PAIR_SCHEMA,
        },
    }
)
PAIR_CONSTRUCTION_TRACE_SCHEMA = _object_schema({"pair": TEXT_SCHEMA, "trace": TEXT_SCHEMA})
GENERATION_PROVENANCE_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "generation_call_id": TEXT_SCHEMA,
        "generation_mode": {"type": "string", "enum": list(GENERATION_MODES)},
        "seed_factors": {"type": "array", "maxItems": 2, "items": TEXT_SCHEMA},
    }
)
AUTHORING_METADATA_SCHEMA = _object_schema(
    {
        "intended_rule": TEXT_SCHEMA,
        "pair_blueprint": BLUEPRINT_SCHEMA,
        "expected_test_outputs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": GRID_SCHEMA,
        },
        "intermediate_construction_representation": TEXT_SCHEMA,
        "pair_construction_trace": {
            "type": "array",
            "minItems": 2,
            "maxItems": 13,
            "items": PAIR_CONSTRUCTION_TRACE_SCHEMA,
        },
        "generation_provenance": GENERATION_PROVENANCE_SCHEMA,
    }
)
AUTHORED_EPISODE_SCHEMA = _object_schema(
    {"task": PUBLIC_ARC_TASK_SCHEMA, "authoring_metadata": AUTHORING_METADATA_SCHEMA}
)
REALIZED_TASK_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "task": PUBLIC_ARC_TASK_SCHEMA,
        "authoring_metadata": AUTHORING_METADATA_SCHEMA,
    }
)
REPAIR_LOG_SCHEMA = _object_schema(
    {"defect": TEXT_SCHEMA, "change": TEXT_SCHEMA, "why_rule_is_preserved": TEXT_SCHEMA}
)
REPAIR_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "repair_status": {"type": "string", "enum": ["repaired", "unrepairable"]},
        "unrepairable_reason": {"type": "string"},
        "repaired_task": {"anyOf": [AUTHORED_EPISODE_SCHEMA, {"type": "null"}]},
        "repair_log": {"type": "array", "maxItems": 20, "items": REPAIR_LOG_SCHEMA},
    }
)

CELL_COORDINATE_SCHEMA = {
    "type": "array",
    "minItems": 2,
    "maxItems": 2,
    "items": {"type": "integer", "minimum": 0, "maximum": 29},
}
PAIR_VALIDATION_SCHEMA = _object_schema(
    {
        "pair": TEXT_SCHEMA,
        "valid": {"type": "boolean"},
        "rule_trace": TEXT_SCHEMA,
        "issues": {"type": "array", "maxItems": 20, "items": TEXT_SCHEMA},
        "suspected_cell_discrepancies": {
            "type": "array",
            "maxItems": 100,
            "items": CELL_COORDINATE_SCHEMA,
        },
    }
)
RULE_VALIDATION_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "validator_id": TEXT_SCHEMA,
        "valid": {"type": "boolean"},
        "pair_results": {
            "type": "array",
            "minItems": 2,
            "maxItems": 13,
            "items": PAIR_VALIDATION_SCHEMA,
        },
        "all_pair_roles_instantiated": {"type": "boolean"},
        "missing_or_misinstantiated_roles": {"type": "array", "maxItems": 20, "items": TEXT_SCHEMA},
        "test_supported": {"type": "boolean"},
        "test_only_semantics": {"type": "array", "maxItems": 20, "items": TEXT_SCHEMA},
        "arbitrary_exceptions": {"type": "array", "maxItems": 20, "items": TEXT_SCHEMA},
        "overall_issues": {"type": "array", "maxItems": 20, "items": TEXT_SCHEMA},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
    }
)

PLAUSIBLE_RULE_SCHEMA = _object_schema(
    {
        "rule": TEXT_SCHEMA,
        "why_training_supports_it": TEXT_SCHEMA,
        "comparably_simple": {"type": "boolean"},
        "predicted_test_outputs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": GRID_SCHEMA,
        },
    }
)
BLIND_SOLVER_SCHEMA = _object_schema(
    {
        "task_id": TEXT_SCHEMA,
        "solver_id": TEXT_SCHEMA,
        "primary_inferred_rule": TEXT_SCHEMA,
        "predicted_test_outputs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": GRID_SCHEMA,
        },
        "other_plausible_rules": {"type": "array", "maxItems": 5, "items": PLAUSIBLE_RULE_SCHEMA},
        "alternative_produces_different_test_output": {"type": "boolean"},
        "fairness_assessment": TEXT_SCHEMA,
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
    }
)
DESCRIPTION_AWARE_PAIR_CHECK_SCHEMA = _object_schema(
    {
        "pair": TEXT_SCHEMA,
        "follows_supplied_rule": {"type": "boolean"},
        "instantiates_author_specification": {"type": "boolean"},
        "issues": {"type": "array", "maxItems": 20, "items": TEXT_SCHEMA},
        "suspected_cell_discrepancies": {
            "type": "array",
            "maxItems": 100,
            "items": CELL_COORDINATE_SCHEMA,
        },
    }
)
RULE_CLAUSE_EVIDENCE_SCHEMA = _object_schema(
    {
        "rule_clause": TEXT_SCHEMA,
        "visible_support": TEXT_SCHEMA,
        "support": {
            "type": "string",
            "enum": list(EVIDENCE_SUPPORT_LEVELS),
        },
    }
)
DESCRIPTION_AWARE_ALTERNATIVE_SCHEMA = _object_schema(
    {
        "rule": TEXT_SCHEMA,
        "fits_all_training_pairs": {"type": "boolean"},
        "comparably_simple": {"type": "boolean"},
        "comparably_well_supported": {"type": "boolean"},
        "predicted_test_outputs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": GRID_SCHEMA,
        },
        "assessment": TEXT_SCHEMA,
    }
)
DESCRIPTION_AWARE_CHECKER_SCHEMA = _object_schema(
    {
        "fixture_id": TEXT_SCHEMA,
        "predicted_test_outputs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": GRID_SCHEMA,
        },
        "training_pair_checks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": DESCRIPTION_AWARE_PAIR_CHECK_SCHEMA,
        },
        "rule_clause_evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": RULE_CLAUSE_EVIDENCE_SCHEMA,
        },
        "test_only_semantics": {
            "type": "array",
            "maxItems": 20,
            "items": TEXT_SCHEMA,
        },
        "serious_alternatives": {
            "type": "array",
            "maxItems": 10,
            "items": DESCRIPTION_AWARE_ALTERNATIVE_SCHEMA,
        },
        "distinct_serious_test_output_counts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "integer", "minimum": 1, "maximum": 11},
        },
        "answer_class": {"type": "string", "enum": list(ANSWER_CLASSES)},
        "rule_is_mechanically_realized": {"type": "boolean"},
        "rule_is_sufficiently_communicated": {"type": "boolean"},
        "revision_recommended": {"type": "boolean"},
        "revision_reason": {"type": "string"},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
    }
)
EVALUATED_ALTERNATIVE_SCHEMA = _object_schema(
    {
        "rule": TEXT_SCHEMA,
        "fits_all_training_pairs": {"type": "boolean"},
        "comparably_simple": {"type": "boolean"},
        "comparably_well_supported": {"type": "boolean"},
        "predicts_different_test_output": {"type": "boolean"},
        "predicted_test_outputs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": GRID_SCHEMA,
        },
        "assessment": TEXT_SCHEMA,
    }
)
AMBIGUITY_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "intended_rule_reasonably_supported": {"type": "boolean"},
        "intended_test_outputs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": GRID_SCHEMA,
        },
        "serious_competing_alternative_predicts_different_output": {"type": "boolean"},
        "distinct_serious_test_output_counts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "integer", "minimum": 1, "maximum": 11},
        },
        "solvable_within_two_guesses": {"type": "boolean"},
        "answer_class": {"type": "string", "enum": list(ANSWER_CLASSES)},
        "difficult_but_fair": {"type": "boolean"},
        "underdetermined": {"type": "boolean"},
        "training_pair_should_be_revised": {"type": "boolean"},
        "evaluated_alternatives": {"type": "array", "maxItems": 10, "items": EVALUATED_ALTERNATIVE_SCHEMA},
        "decision": {"type": "string", "enum": list(AMBIGUITY_DECISIONS)},
        "recommended_revision": {"type": "string"},
        "reason": TEXT_SCHEMA,
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
    }
)

FACTOR_ASSESSMENT_SCHEMA = _object_schema(
    {
        "factor": TEXT_SCHEMA,
        "status": {"type": "string", "enum": list(ACTIVATION_STATUSES)},
        "explanation": TEXT_SCHEMA,
    }
)
FACTOR_ACTIVATION_SCHEMA = _object_schema(
    {
        "candidate_id": TEXT_SCHEMA,
        "seed_factor_activation": {
            "type": "array",
            "maxItems": 2,
            "items": FACTOR_ASSESSMENT_SCHEMA,
        },
        "seeded_mode_success": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
        "observed_factors": {
            "type": "array",
            "items": OBSERVED_FACTOR_SCHEMA,
        },
        "new_factor_labels": {
            "type": "array",
            "uniqueItems": True,
            "items": TEXT_SCHEMA,
        },
        "overall_reason": TEXT_SCHEMA,
    }
)

RANKING_ENTRY_SCHEMA = _object_schema(
    {
        "task_id": TEXT_SCHEMA,
        "rank": {"type": "integer", "minimum": 1, "maximum": 16},
        "rationale": TEXT_SCHEMA,
        "flags": {
            "type": "array",
            "maxItems": len(QUALITY_FLAGS),
            "uniqueItems": True,
            "items": {"type": "string", "enum": list(QUALITY_FLAGS)},
        },
    }
)
QUALITY_RANKING_SCHEMA = _object_schema(
    {
        "ranker_id": TEXT_SCHEMA,
        "ranking": {"type": "array", "minItems": 1, "maxItems": 16, "items": RANKING_ENTRY_SCHEMA},
        "batch_observations": TEXT_SCHEMA,
    }
)


def _codex_transport_schema(value: Any) -> Any:
    """Project strict schemas to the Codex response-format subset.

    Codex rejects several ordinary JSON Schema validation keywords (including
    ``uniqueItems``).  Those constraints remain enforced by the stage parser;
    the transport schema retains shapes, closed objects, required fields,
    enums, and nullability so generation is still structurally constrained.
    """

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
            key: _codex_transport_schema(item)
            for key, item in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [_codex_transport_schema(item) for item in value]
    return value


_STRICT_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "idea_generator": IDEA_GENERATOR_SCHEMA,
    "difficulty_annotator": DIFFICULTY_SCHEMA,
    "diversity_judge": DIVERSITY_SCHEMA,
    "portfolio_diversity_judge": PORTFOLIO_DIVERSITY_SCHEMA,
    "novelty_scout": NOVELTY_SCOUT_SCHEMA,
    "novelty_adjudicator": NOVELTY_ADJUDICATION_SCHEMA,
    "blueprint_author": BLUEPRINT_SCHEMA,
    "task_realizer": REALIZED_TASK_SCHEMA,
    "task_repair": REPAIR_SCHEMA,
    "ambiguity_repair": REPAIR_SCHEMA,
    "rule_aware_validator": RULE_VALIDATION_SCHEMA,
    "blind_solver": BLIND_SOLVER_SCHEMA,
    "description_aware_checker": DESCRIPTION_AWARE_CHECKER_SCHEMA,
    "ambiguity_adjudicator": AMBIGUITY_SCHEMA,
    "factor_activation_judge": FACTOR_ACTIVATION_SCHEMA,
    "quality_ranker": QUALITY_RANKING_SCHEMA,
}

OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    stage: _codex_transport_schema(schema)
    for stage, schema in _STRICT_OUTPUT_SCHEMAS.items()
}


STAGE_MODELS: dict[str, type[StrictModel]] = {
    "idea_generator": IdeaGenerationOutput,
    "difficulty_annotator": DifficultyAnnotation,
    "diversity_judge": DiversityJudgeOutput,
    "portfolio_diversity_judge": PortfolioDiversityOutput,
    "novelty_scout": NoveltyScoutOutput,
    "novelty_adjudicator": NoveltyAdjudication,
    "blueprint_author": PairBlueprint,
    "task_realizer": RealizedTaskOutput,
    "task_repair": RepairOutput,
    "ambiguity_repair": RepairOutput,
    "rule_aware_validator": RuleAwareValidation,
    "blind_solver": BlindSolverOutput,
    "description_aware_checker": DescriptionAwareCheckerOutput,
    "ambiguity_adjudicator": AmbiguityAdjudication,
    "factor_activation_judge": FactorActivationOutput,
    "quality_ranker": QualityRankingOutput,
}

_STAGE_ALIASES = {
    "ideas": "idea_generator",
    "idea_generation": "idea_generator",
    "difficulty": "difficulty_annotator",
    "diversity": "diversity_judge",
    "portfolio_diversity": "portfolio_diversity_judge",
    "novelty_adjudication": "novelty_adjudicator",
    "blueprint": "blueprint_author",
    "realizer": "task_realizer",
    "repair": "task_repair",
    "ambiguity_revision": "ambiguity_repair",
    "rule_validation": "rule_aware_validator",
    "informed_checker": "description_aware_checker",
    "ambiguity": "ambiguity_adjudicator",
    "factor_activation": "factor_activation_judge",
    "ranking": "quality_ranker",
}


def canonical_stage_name(stage: str) -> str:
    if not isinstance(stage, str) or not stage.strip():
        raise SchemaError("$.stage", "stage must be a non-empty string")
    canonical = _STAGE_ALIASES.get(stage, stage)
    if canonical not in STAGE_MODELS:
        raise SchemaError("$.stage", f"unknown stage {stage!r}; expected one of {sorted(STAGE_MODELS)!r}")
    return canonical


def parse_stage_output(
    stage: str,
    data: str | bytes | bytearray | Mapping[str, Any],
) -> StrictModel:
    """Strictly parse one model response for ``stage``.

    Callers should cache only after this function succeeds.  No extraction,
    partial parsing, default insertion, or malformed-response repair occurs.
    """

    canonical = canonical_stage_name(stage)
    if isinstance(data, (str, bytes, bytearray)):
        obj = parse_json_object(data)
    else:
        obj = dict(_mapping(data, "$"))
    return STAGE_MODELS[canonical].from_dict(obj)


__all__ = [
    "ACTIVATION_STATUSES",
    "AMBIGUITY_DECISIONS",
    "ANSWER_CLASSES",
    "ARC_PAIR_SCHEMA",
    "ARC_TASK_SCHEMA",
    "ARCExample",
    "ARCInput",
    "ARCTask",
    "AmbiguityAdjudication",
    "AuthoredEpisode",
    "AuthoringMetadata",
    "BlindSolverOutput",
    "DescriptionAwareCheckerOutput",
    "DifficultyAnnotation",
    "DiversityJudgeOutput",
    "FactorActivationOutput",
    "GENERATION_MODES",
    "GenerationProvenance",
    "GRID_SCHEMA",
    "Grid",
    "IdeaCandidate",
    "IdeaGenerationOutput",
    "NoveltyAdjudication",
    "NoveltyScoutOutput",
    "OUTPUT_SCHEMAS",
    "ObservedFactor",
    "PairBlueprint",
    "PortfolioDiversityOutput",
    "PublicARCTask",
    "QualityRankingOutput",
    "RealizedTaskOutput",
    "RepairOutput",
    "RuleAwareValidation",
    "STAGE_MODELS",
    "SchemaError",
    "StrictModel",
    "canonical_stage_name",
    "parse_grid",
    "parse_json_object",
    "parse_stage_output",
]
