"""Deterministic ARC-format and hidden-output validation.

These checks deliberately validate representation, not the semantic task rule.
They do not generate, complete, repair, or reinterpret grids.  Heuristics such
as identity outputs and large grids are warnings; only concrete ARC-format,
serialization, duplication, or leakage defects are errors.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import (
    ARCExample,
    ARCTask,
    AuthoredEpisode,
    Grid,
    RealizedTaskOutput,
    SchemaError,
    parse_json_object,
)


class ARCValidationError(ValueError):
    """Raised when a caller explicitly requires a valid ARC task."""


class HiddenOutputLeakageError(ARCValidationError):
    """Raised when a solver-facing view contains author-only information."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    severity: str

    def __post_init__(self) -> None:
        if self.severity not in ("error", "warning"):
            raise ValueError("severity must be 'error' or 'warning'")

    def render(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass(frozen=True, slots=True)
class ARCValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    issues: tuple[ValidationIssue, ...]
    train_count: int
    test_count: int


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _task_payload(value: Any) -> Any:
    if isinstance(value, RealizedTaskOutput):
        return value.complete_task().to_dict()
    if isinstance(value, AuthoredEpisode):
        return value.complete_task().to_dict()
    if isinstance(value, ARCTask):
        return value.to_dict()
    if isinstance(value, (str, bytes, bytearray)):
        return parse_json_object(value)
    if isinstance(value, Mapping):
        # Accept the explicit authored wrapper as a convenience, while still
        # validating the resulting standard ARC task strictly.
        if set(value) == {"candidate_id", "task", "authoring_metadata"}:
            return RealizedTaskOutput.from_dict(value).complete_task().to_dict()
        if set(value) == {"task", "authoring_metadata"}:
            return AuthoredEpisode.from_dict(value).complete_task().to_dict()
        if "arc_task" in value and isinstance(value["arc_task"], Mapping):
            return deepcopy(value["arc_task"])
        return deepcopy(dict(value))
    return value


def _issue(
    issues: list[ValidationIssue],
    code: str,
    path: str,
    message: str,
    severity: str = "error",
) -> None:
    issues.append(ValidationIssue(code=code, path=path, message=message, severity=severity))


def _validate_grid(value: Any, path: str, issues: list[ValidationIssue]) -> Grid | None:
    if not isinstance(value, list):
        _issue(issues, "grid_not_array", path, "grid must be an array of rows")
        return None
    if not 1 <= len(value) <= 30:
        _issue(issues, "grid_height", path, "grid height must be between 1 and 30")
    if not value:
        return None
    width: int | None = None
    parsed: list[tuple[int, ...]] = []
    structural_failure = False
    for row_index, row in enumerate(value):
        row_path = f"{path}[{row_index}]"
        if not isinstance(row, list):
            _issue(issues, "row_not_array", row_path, "grid row must be an array")
            structural_failure = True
            continue
        if not 1 <= len(row) <= 30:
            _issue(issues, "grid_width", row_path, "row width must be between 1 and 30")
        if width is None:
            width = len(row)
        elif len(row) != width:
            _issue(
                issues,
                "non_rectangular_grid",
                row_path,
                f"row width {len(row)} does not match first-row width {width}",
            )
        parsed_row: list[int] = []
        for column_index, cell in enumerate(row):
            cell_path = f"{row_path}[{column_index}]"
            if type(cell) is not int:
                _issue(issues, "non_integer_cell", cell_path, "ARC cell must be an integer, not a boolean or other value")
                structural_failure = True
                continue
            if not 0 <= cell <= 9:
                _issue(issues, "cell_out_of_range", cell_path, "ARC cell value must be between 0 and 9")
                structural_failure = True
            parsed_row.append(cell)
        parsed.append(tuple(parsed_row))
    if any(issue.severity == "error" and issue.path.startswith(path) for issue in issues):
        return None
    if structural_failure or width is None:
        return None
    return tuple(parsed)


def validate_grid(value: Any, path: str = "$.grid") -> tuple[str, ...]:
    """Return deterministic validation errors for one ARC grid."""

    issues: list[ValidationIssue] = []
    _validate_grid(value, path, issues)
    return tuple(issue.render() for issue in issues if issue.severity == "error")


def _pair_key(pair: Mapping[str, Any]) -> str | None:
    try:
        return json.dumps(pair, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None


def validate_arc_task(
    value: Any,
    *,
    require_test_outputs: bool = True,
    min_train_pairs: int = 1,
    max_train_pairs: int = 10,
    min_test_pairs: int = 1,
    max_test_pairs: int = 3,
) -> ARCValidationResult:
    """Validate a complete ARC task without changing it.

    The default accepts the repository's labeled reference tasks and complete
    generated authoring tasks.  Use ``require_test_outputs=False`` only for an
    intentionally solver-facing public view.
    """

    issues: list[ValidationIssue] = []
    try:
        task = _task_payload(value)
    except (SchemaError, ValueError, TypeError) as exc:
        _issue(issues, "malformed_authored_wrapper", "$", str(exc))
        task = None
    if not isinstance(task, dict):
        _issue(issues, "task_not_object", "$", "ARC task must be a JSON object")
        return _result(issues, 0, 0)
    non_string_root = [key for key in task if not isinstance(key, str)]
    if non_string_root:
        _issue(issues, "non_string_task_key", "$", "all task object keys must be strings")
    unknown_root = sorted(
        (key for key in task if not isinstance(key, str) or key not in {"train", "test"}),
        key=lambda item: repr(item),
    )
    missing_root = sorted({"train", "test"} - set(task))
    if missing_root:
        _issue(issues, "missing_task_field", "$", f"missing required field(s): {', '.join(missing_root)}")
    if unknown_root:
        _issue(
            issues,
            "unknown_task_field",
            "$",
            "unknown field(s): " + ", ".join(repr(item) for item in unknown_root),
        )

    split_values: dict[str, list[Any]] = {}
    count_limits = {
        "train": (min_train_pairs, max_train_pairs),
        "test": (min_test_pairs, max_test_pairs),
    }
    for split in ("train", "test"):
        raw = task.get(split)
        if not isinstance(raw, list):
            if split in task:
                _issue(issues, "split_not_array", f"$.{split}", f"{split} must be an array")
            split_values[split] = []
            continue
        split_values[split] = raw
        minimum, maximum = count_limits[split]
        if not minimum <= len(raw) <= maximum:
            _issue(
                issues,
                "pair_count",
                f"$.{split}",
                f"{split} must contain between {minimum} and {maximum} pair(s); found {len(raw)}",
            )

    normalized_pairs: list[tuple[str, int, dict[str, Any]]] = []
    parsed_grids: dict[tuple[str, int, str], Grid] = {}
    for split in ("train", "test"):
        for index, pair in enumerate(split_values[split]):
            pair_path = f"$.{split}[{index}]"
            if not isinstance(pair, dict):
                _issue(issues, "pair_not_object", pair_path, "pair must be an object")
                continue
            non_string_pair = [key for key in pair if not isinstance(key, str)]
            if non_string_pair:
                _issue(issues, "non_string_pair_key", pair_path, "all pair object keys must be strings")
            required = {"input", "output"} if split == "train" or require_test_outputs else {"input"}
            allowed = required
            missing = sorted(required - set(pair))
            unknown = sorted(
                (key for key in pair if not isinstance(key, str) or key not in allowed),
                key=lambda item: repr(item),
            )
            if missing:
                _issue(issues, "missing_pair_field", pair_path, f"missing required field(s): {', '.join(missing)}")
            if unknown:
                if split == "test" and not require_test_outputs and "output" in unknown:
                    _issue(issues, "hidden_test_output_leak", f"{pair_path}.output", "solver-facing test pair must not contain output")
                    unknown.remove("output")
                if unknown:
                    _issue(
                        issues,
                        "unknown_pair_field",
                        pair_path,
                        "unknown field(s): " + ", ".join(repr(item) for item in unknown),
                    )
            for field_name in ("input", "output"):
                if field_name in pair:
                    grid = _validate_grid(pair[field_name], f"{pair_path}.{field_name}", issues)
                    if grid is not None:
                        parsed_grids[(split, index, field_name)] = grid
            normalized_pairs.append((split, index, pair))

    # Exact pair duplication is always accidental in this benchmark.  Repeated
    # outputs alone remain legal; only complete episodes or identical inputs
    # are flagged.
    seen_pairs: dict[str, tuple[str, int]] = {}
    seen_inputs: dict[str, tuple[str, int, str | None]] = {}
    for split, index, pair in normalized_pairs:
        location = (split, index)
        key = _pair_key(pair)
        if key is not None and key in seen_pairs:
            prior_split, prior_index = seen_pairs[key]
            _issue(
                issues,
                "duplicate_pair",
                f"$.{split}[{index}]",
                f"duplicates $.{prior_split}[{prior_index}] exactly",
            )
        elif key is not None:
            seen_pairs[key] = location
        if "input" in pair:
            input_key = _pair_key({"input": pair["input"]})
            output_key = _pair_key({"output": pair.get("output")}) if "output" in pair else None
            if input_key is not None and input_key in seen_inputs:
                prior_split, prior_index, prior_output = seen_inputs[input_key]
                if prior_output != output_key:
                    _issue(
                        issues,
                        "conflicting_duplicate_input",
                        f"$.{split}[{index}].input",
                        f"duplicates $.{prior_split}[{prior_index}].input but has a different output",
                    )
            elif input_key is not None:
                seen_inputs[input_key] = (split, index, output_key)

    identity_count = 0
    for index in range(len(split_values["train"])):
        input_grid = parsed_grids.get(("train", index, "input"))
        output_grid = parsed_grids.get(("train", index, "output"))
        if input_grid is not None and input_grid == output_grid:
            identity_count += 1
    if split_values["train"] and identity_count == len(split_values["train"]):
        _issue(
            issues,
            "all_identity_training_pairs",
            "$.train",
            "all training outputs are identical to their inputs",
            "warning",
        )
    elif identity_count:
        _issue(
            issues,
            "identity_training_pair",
            "$.train",
            f"{identity_count} training pair(s) are identity transformations",
            "warning",
        )

    if any(
        len(grid) > 20 or len(grid[0]) > 20
        for grid in parsed_grids.values()
        if grid and grid[0]
    ):
        _issue(
            issues,
            "large_grid",
            "$",
            "one or more grids exceed 20 cells on an axis (still valid ARC format)",
            "warning",
        )

    if not round_trip_preserves(task):
        _issue(issues, "serialization_round_trip", "$", "JSON serialization and reload did not preserve the task exactly")
    return _result(issues, len(split_values["train"]), len(split_values["test"]))


def _result(issues: Sequence[ValidationIssue], train_count: int, test_count: int) -> ARCValidationResult:
    errors = tuple(issue.render() for issue in issues if issue.severity == "error")
    warnings = tuple(issue.render() for issue in issues if issue.severity == "warning")
    return ARCValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        issues=tuple(issues),
        train_count=train_count,
        test_count=test_count,
    )


def assert_valid_arc_task(value: Any, **kwargs: Any) -> ARCTask:
    """Return a normalized immutable task or raise with all deterministic errors."""

    result = validate_arc_task(value, **kwargs)
    if not result.valid:
        raise ARCValidationError("; ".join(result.errors))
    payload = _task_payload(value)
    return ARCTask.from_dict(
        payload,
        require_test_outputs=bool(kwargs.get("require_test_outputs", True)),
        min_train_pairs=int(kwargs.get("min_train_pairs", 1)),
        max_train_pairs=int(kwargs.get("max_train_pairs", 10)),
        min_test_pairs=int(kwargs.get("min_test_pairs", 1)),
        max_test_pairs=int(kwargs.get("max_test_pairs", 3)),
    )


def round_trip_preserves(value: Any) -> bool:
    """Whether canonical JSON serialization and reload preserve ``value``."""

    try:
        plain = _plain(value)
        serialized = json.dumps(
            plain,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        restored = json.loads(serialized, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        return restored == plain
    except (TypeError, ValueError, OverflowError):
        return False


def serialization_round_trip(value: Any) -> Any:
    """Serialize and reload a JSON-compatible value, raising on invalid data."""

    plain = _plain(value)
    serialized = json.dumps(
        plain,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    restored = json.loads(serialized, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if restored != plain:
        raise ARCValidationError("serialization round trip changed the value")
    return restored


def _public_source(value: Any) -> Mapping[str, Any]:
    if isinstance(value, RealizedTaskOutput):
        return value.task.to_dict()
    if isinstance(value, AuthoredEpisode):
        return value.task.to_dict()
    if isinstance(value, ARCTask):
        return value.to_dict()
    if isinstance(value, (str, bytes, bytearray)):
        parsed = parse_json_object(value)
    elif isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raise ARCValidationError("task must be an ARC object, authored wrapper, or JSON object")
    if "task" in parsed and isinstance(parsed["task"], Mapping):
        return parsed["task"]
    if "arc_task" in parsed and isinstance(parsed["arc_task"], Mapping):
        return parsed["arc_task"]
    return parsed


def solver_facing_task(value: Any) -> dict[str, Any]:
    """Project a task to the only information a blind solver may receive.

    The function uses an allow-list projection: training inputs/outputs and test
    inputs are copied; every other root, pair, authoring, rule, blueprint,
    provenance, factor, and expected-output field is discarded.
    """

    source = _public_source(value)
    if not isinstance(source.get("train"), (list, tuple)) or not isinstance(source.get("test"), (list, tuple)):
        raise ARCValidationError("task must contain train and test arrays")
    result: dict[str, Any] = {"train": [], "test": []}
    for index, pair in enumerate(source["train"]):
        if not isinstance(pair, Mapping) or "input" not in pair or "output" not in pair:
            raise ARCValidationError(f"$.train[{index}] must contain input and output")
        result["train"].append(
            {"input": deepcopy(_plain(pair["input"])), "output": deepcopy(_plain(pair["output"]))}
        )
    for index, pair in enumerate(source["test"]):
        if not isinstance(pair, Mapping) or "input" not in pair:
            raise ARCValidationError(f"$.test[{index}] must contain input")
        result["test"].append({"input": deepcopy(_plain(pair["input"]))})
    assert_no_hidden_test_leakage(result)
    return result


def assert_no_hidden_test_leakage(value: Any) -> None:
    """Assert that a solver-facing task contains no hidden output or metadata."""

    if isinstance(value, (str, bytes, bytearray)):
        try:
            task = parse_json_object(value)
        except SchemaError as exc:
            raise HiddenOutputLeakageError(str(exc)) from exc
    elif isinstance(value, Mapping):
        task = dict(value)
    elif isinstance(value, ARCTask):
        task = value.to_dict()
    else:
        raise HiddenOutputLeakageError("solver-facing task must be a JSON object")
    if set(task) != {"train", "test"}:
        extra = sorted(
            (key for key in task if not isinstance(key, str) or key not in {"train", "test"}),
            key=lambda item: repr(item),
        )
        missing = sorted({"train", "test"} - set(task))
        details = []
        if extra:
            details.append(f"author-only or unknown root fields present: {extra!r}")
        if missing:
            details.append(f"missing root fields: {missing!r}")
        raise HiddenOutputLeakageError("; ".join(details))
    for split in ("train", "test"):
        pairs = task[split]
        if not isinstance(pairs, list) or not pairs:
            raise HiddenOutputLeakageError(f"$.{split} must be a nonempty array")
        for index, pair in enumerate(pairs):
            path = f"$.{split}[{index}]"
            if not isinstance(pair, dict):
                raise HiddenOutputLeakageError(f"{path} must be an object")
            expected_keys = {"input", "output"} if split == "train" else {"input"}
            if set(pair) != expected_keys:
                if split == "test" and "output" in pair:
                    raise HiddenOutputLeakageError(f"{path}.output leaks the hidden expected test output")
                extra = sorted(
                    (key for key in pair if not isinstance(key, str) or key not in expected_keys),
                    key=lambda item: repr(item),
                )
                missing = sorted(expected_keys - set(pair))
                raise HiddenOutputLeakageError(
                    f"{path} has invalid fields (extra={extra!r}, missing={missing!r})"
                )
            for field_name in expected_keys:
                errors = validate_grid(pair[field_name], f"{path}.{field_name}")
                if errors:
                    raise HiddenOutputLeakageError("; ".join(errors))


def has_hidden_test_leakage(value: Any) -> bool:
    try:
        assert_no_hidden_test_leakage(value)
    except (HiddenOutputLeakageError, SchemaError):
        return True
    return False


def remove_hidden_test_outputs(value: Any) -> dict[str, Any]:
    """Compatibility alias for the allow-list solver projection."""

    return solver_facing_task(value)


def load_arc_task(path: str | Path, *, require_test_outputs: bool = True) -> ARCTask:
    """Load and deterministically validate one ARC JSON file."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ARCValidationError(f"cannot read {source}: {exc}") from exc
    result = validate_arc_task(raw, require_test_outputs=require_test_outputs)
    if not result.valid:
        raise ARCValidationError(f"{source}: {'; '.join(result.errors)}")
    return ARCTask.from_dict(parse_json_object(raw), require_test_outputs=require_test_outputs)


__all__ = [
    "ARCValidationError",
    "ARCValidationResult",
    "HiddenOutputLeakageError",
    "ValidationIssue",
    "assert_no_hidden_test_leakage",
    "assert_valid_arc_task",
    "has_hidden_test_leakage",
    "load_arc_task",
    "remove_hidden_test_outputs",
    "round_trip_preserves",
    "serialization_round_trip",
    "solver_facing_task",
    "validate_arc_task",
    "validate_grid",
]
