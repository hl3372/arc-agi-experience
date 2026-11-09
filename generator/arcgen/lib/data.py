"""Load the curated ARC bank and make reproducible benchmark selections.

The repository deliberately keeps natural-language annotations separate from
the canonical ARC grid files.  This module joins them without changing either
source format.  Difficulty-factor labels are retained verbatim apart from
leading/trailing and repeated whitespace; no ontology or synonym
normalisation is applied.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable, Iterator, Mapping, Sequence
import json
import re


DEFAULT_SPLIT_SEED = 20260722
EXPECTED_TASK_COUNT = 120


class DataValidationError(ValueError):
    """Raised when annotations and ARC task files cannot form a valid bank."""


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """One joined annotation/grid record in the benchmark's logical schema."""

    task_id: str
    rule_description: str
    difficulty_factors: tuple[str, ...]
    train: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]

    @property
    def rule(self) -> str:
        """Compatibility alias for repositories whose field is named ``rule``."""

        return self.rule_description

    @property
    def difficulty_text(self) -> str:
        """Return the open-vocabulary factors as plain text for retrieval."""

        return "; ".join(self.difficulty_factors)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy using the benchmark's logical names."""

        return {
            "task_id": self.task_id,
            "rule_description": self.rule_description,
            "difficulty_factors": list(self.difficulty_factors),
            "train": deepcopy(list(self.train)),
            "test": deepcopy(list(self.test)),
        }


@dataclass(frozen=True, slots=True)
class TaskSplit:
    """Disjoint calibration and hidden-anchor partitions."""

    calibration: tuple[TaskRecord, ...]
    hidden_anchors: tuple[TaskRecord, ...]
    seed: int

    @property
    def anchors(self) -> tuple[TaskRecord, ...]:
        """Short compatibility alias for ``hidden_anchors``."""

        return self.hidden_anchors

    @property
    def calibration_ids(self) -> tuple[str, ...]:
        return tuple(record.task_id for record in self.calibration)

    @property
    def hidden_anchor_ids(self) -> tuple[str, ...]:
        return tuple(record.task_id for record in self.hidden_anchors)

    def __iter__(self) -> Iterator[tuple[TaskRecord, ...]]:
        """Allow ``calibration, anchors = deterministic_split(...)``."""

        yield self.calibration
        yield self.hidden_anchors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exact split without exposing task descriptions."""

        return {
            "seed": self.seed,
            "calibration_task_ids": list(self.calibration_ids),
            "hidden_anchor_task_ids": list(self.hidden_anchor_ids),
        }


@dataclass(frozen=True, slots=True)
class FactorChoice:
    """An exact factor label and its task-level frequency."""

    label: str
    count: int
    task_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "count": self.count, "task_ids": list(self.task_ids)}


@dataclass(frozen=True, slots=True)
class FactorPlan:
    """Reproducible raw-label assignments for factor-seeded generation calls."""

    recurring: FactorChoice
    rare: FactorChoice
    single_factor_calls: tuple[tuple[str, ...], tuple[str, ...]]
    two_factor_calls: tuple[tuple[str, str], tuple[str, str]]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "recurring": self.recurring.to_dict(),
            "rare": self.rare.to_dict(),
            "single_factor_calls": [list(value) for value in self.single_factor_calls],
            "two_factor_calls": [list(value) for value in self.two_factor_calls],
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DataValidationError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except DataValidationError as exc:
        raise DataValidationError(f"invalid JSON object in {path}: {exc}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"JSON root must be an object: {path}")
    return value


def _json_files(directory: str | Path, label: str) -> tuple[Path, ...]:
    path = Path(directory)
    if not path.is_dir():
        raise DataValidationError(f"{label} directory does not exist or is not a directory: {path}")
    files = tuple(sorted(path.glob("*.json"), key=lambda item: item.name))
    if not files:
        raise DataValidationError(f"{label} directory contains no JSON files: {path}")
    duplicate_stems = [stem for stem, count in Counter(p.stem for p in files).items() if count > 1]
    if duplicate_stems:
        raise DataValidationError(
            f"{label} directory has duplicate task filenames: {', '.join(sorted(duplicate_stems))}"
        )
    return files


def _clean_raw_label(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise DataValidationError(f"{context} must be a string")
    # Whitespace cleanup is explicitly permitted; spelling, punctuation, case,
    # and the open-ended wording otherwise remain untouched.
    cleaned = " ".join(value.split())
    if not cleaned:
        raise DataValidationError(f"{context} must not be empty")
    return cleaned


def _extract_rule(annotation: Mapping[str, Any], path: Path) -> str:
    for field in ("rule_description", "rule", "description"):
        if field in annotation:
            value = annotation[field]
            if not isinstance(value, str) or not value.strip():
                raise DataValidationError(f"{path}: {field} must be a non-empty string")
            return value.strip()
    raise DataValidationError(
        f"{path}: annotation lacks rule_description (supported aliases: rule, description)"
    )


def _extract_factors(annotation: Mapping[str, Any], path: Path) -> tuple[str, ...]:
    raw = annotation.get("difficulty_factors")
    if raw is None:
        # A few annotation formats use labels as the shorter field name.
        raw = annotation.get("factors")
    if not isinstance(raw, list):
        raise DataValidationError(f"{path}: difficulty_factors must be a list")
    labels: list[str] = []
    for index, item in enumerate(raw):
        context = f"{path}: difficulty_factors[{index}]"
        if isinstance(item, str):
            label = _clean_raw_label(item, context=context)
        elif isinstance(item, Mapping):
            candidate = item.get("factor", item.get("label"))
            label = _clean_raw_label(candidate, context=f"{context}.factor")
        else:
            raise DataValidationError(f"{context} must be a string or object")
        labels.append(label)
    if not labels:
        raise DataValidationError(f"{path}: difficulty_factors must not be empty")
    duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicates:
        raise DataValidationError(
            f"{path}: duplicate exact difficulty-factor labels: {', '.join(duplicates)}"
        )
    return tuple(labels)


def _extract_task_id(value: Mapping[str, Any], path: Path, *, required: bool) -> str:
    task_id = value.get("task_id", value.get("id"))
    if task_id is None and not required:
        return path.stem
    if not isinstance(task_id, str) or not task_id.strip():
        raise DataValidationError(f"{path}: task_id must be a non-empty string")
    task_id = task_id.strip()
    if task_id != path.stem:
        raise DataValidationError(
            f"{path}: embedded task_id {task_id!r} does not match filename {path.stem!r}"
        )
    return task_id


def _extract_pairs(task: Mapping[str, Any], path: Path, split: str) -> tuple[dict[str, Any], ...]:
    raw = task.get(split)
    if not isinstance(raw, list) or not raw:
        raise DataValidationError(f"{path}: {split} must be a non-empty list")
    pairs: list[dict[str, Any]] = []
    for index, pair in enumerate(raw):
        if not isinstance(pair, dict):
            raise DataValidationError(f"{path}: {split}[{index}] must be an object")
        if "input" not in pair:
            raise DataValidationError(f"{path}: {split}[{index}] lacks input")
        if "output" not in pair:
            raise DataValidationError(f"{path}: {split}[{index}] lacks output")
        pairs.append(deepcopy(pair))
    return tuple(pairs)


def load_task_bank(
    annotation_dir: str | Path,
    task_dir: str | Path,
    *,
    expected_count: int | None = EXPECTED_TASK_COUNT,
) -> list[TaskRecord]:
    """Join annotation and grid files and validate their task-id correspondence.

    Args:
        annotation_dir: Directory containing one annotation JSON per task.
        task_dir: Directory containing the matching ARC ``train``/``test`` JSON.
        expected_count: Required joined count.  The benchmark default is exactly
            120; tests or downstream tools may explicitly pass ``None`` or a
            smaller fixture count.

    Returns:
        Records sorted by task id, independent of filesystem enumeration order.
    """

    if expected_count is not None and (isinstance(expected_count, bool) or expected_count < 1):
        raise DataValidationError("expected_count must be a positive integer or None")
    annotation_files = _json_files(annotation_dir, "annotation")
    task_files = _json_files(task_dir, "task")
    annotations = {path.stem: path for path in annotation_files}
    tasks = {path.stem: path for path in task_files}

    missing_tasks = sorted(annotations.keys() - tasks.keys())
    missing_annotations = sorted(tasks.keys() - annotations.keys())
    if missing_tasks or missing_annotations:
        details: list[str] = []
        if missing_tasks:
            details.append("annotations without grids: " + ", ".join(missing_tasks))
        if missing_annotations:
            details.append("grids without annotations: " + ", ".join(missing_annotations))
        raise DataValidationError("task-id sets do not match; " + "; ".join(details))

    if expected_count is not None and len(annotations) != expected_count:
        raise DataValidationError(
            f"expected exactly {expected_count} matching tasks, found {len(annotations)}"
        )

    records: list[TaskRecord] = []
    for task_id in sorted(annotations):
        annotation_path = annotations[task_id]
        task_path = tasks[task_id]
        annotation = _load_json_object(annotation_path)
        task = _load_json_object(task_path)
        embedded_id = _extract_task_id(annotation, annotation_path, required=False)
        if embedded_id != task_id:
            # _extract_task_id normally catches this; this keeps the invariant
            # explicit if an adapter is expanded later.
            raise DataValidationError(f"annotation id mismatch for {task_id}")
        if "task_id" in task or "id" in task:
            _extract_task_id(task, task_path, required=False)
        records.append(
            TaskRecord(
                task_id=task_id,
                rule_description=_extract_rule(annotation, annotation_path),
                difficulty_factors=_extract_factors(annotation, annotation_path),
                train=_extract_pairs(task, task_path, "train"),
                test=_extract_pairs(task, task_path, "test"),
            )
        )
    return records


def deterministic_split(
    records: Sequence[TaskRecord],
    seed: int = DEFAULT_SPLIT_SEED,
    calibration_count: int = 90,
) -> TaskSplit:
    """Split sorted records with a seeded shuffle into calibration and anchors."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DataValidationError("split seed must be an integer")
    if isinstance(calibration_count, bool) or not isinstance(calibration_count, int):
        raise DataValidationError("calibration_count must be an integer")
    ordered = sorted(records, key=lambda record: record.task_id)
    ids = [record.task_id for record in ordered]
    duplicates = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise DataValidationError(f"cannot split duplicate task ids: {', '.join(duplicates)}")
    if not 0 < calibration_count < len(ordered):
        raise DataValidationError(
            f"calibration_count must be between 1 and {max(1, len(ordered) - 1)}"
        )
    Random(seed).shuffle(ordered)
    calibration = tuple(ordered[:calibration_count])
    anchors = tuple(ordered[calibration_count:])
    if set(record.task_id for record in calibration) & set(record.task_id for record in anchors):
        raise AssertionError("internal error: calibration and anchor split overlap")
    return TaskSplit(calibration=calibration, hidden_anchors=anchors, seed=seed)


def factor_frequencies(records: Iterable[TaskRecord]) -> dict[str, int]:
    """Count exact raw labels by task, without merging spelling variants."""

    counts: Counter[str] = Counter()
    for record in records:
        # Count a factor at most once per task even if a caller constructed a
        # non-loader TaskRecord containing a duplicate label.
        counts.update(dict.fromkeys(record.difficulty_factors, 1))
    return dict(sorted(counts.items()))


_FACTOR_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _comparison_tokens(label: str) -> frozenset[str]:
    """Make conservative tokens used only to avoid obvious paired variants."""

    tokens: set[str] = set()
    for token in _FACTOR_WORD_RE.findall(label.casefold().replace("-", " ")):
        # Very light suffix stripping catches plural/hyphen spelling variants;
        # it is never used to alter or aggregate the label itself.
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.add(token)
    return frozenset(tokens)


def _obvious_near_variant(left: str, right: str) -> bool:
    left_tokens = _comparison_tokens(left)
    right_tokens = _comparison_tokens(right)
    if not left_tokens or not right_tokens:
        return left.casefold() == right.casefold()
    if left_tokens == right_tokens:
        return True
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return overlap >= 0.75


def _factor_choices(records: Sequence[TaskRecord]) -> tuple[list[FactorChoice], list[FactorChoice]]:
    task_ids_by_factor: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        for label in dict.fromkeys(record.difficulty_factors):
            task_ids_by_factor[label].append(record.task_id)
    choices = [
        FactorChoice(label, len(task_ids), tuple(sorted(task_ids)))
        for label, task_ids in sorted(task_ids_by_factor.items())
    ]
    recurring = [choice for choice in choices if choice.count >= 2]
    if not choices:
        raise DataValidationError("cannot sample factors from records with no labels")
    rare_count = min(choice.count for choice in choices)
    rare = [choice for choice in choices if choice.count == rare_count]
    if not recurring:
        raise DataValidationError("factor plan requires at least one label recurring across tasks")
    return recurring, rare


def _pick_compatible(
    pool: Sequence[FactorChoice],
    *,
    against: FactorChoice,
    rng: Random,
    excluded: set[str],
) -> FactorChoice:
    candidates = [
        choice
        for choice in pool
        if choice.label not in excluded and not _obvious_near_variant(choice.label, against.label)
    ]
    if not candidates:
        raise DataValidationError(
            f"could not find a factor sufficiently distinct from {against.label!r}"
        )
    return candidates[rng.randrange(len(candidates))]


def sample_factor_plan(
    calibration_records: Sequence[TaskRecord], seed: int = DEFAULT_SPLIT_SEED
) -> FactorPlan:
    """Choose reproducible recurring/rare labels and two-factor call seeds.

    The first two assignments directly implement the single-factor experiment:
    one exact recurring label and one exact rare (preferably singleton) label.
    Two-factor assignments use one recurring and one rare label per call while
    conservatively excluding obvious lexical variants within each pair.  The
    selected strings are returned exactly as present in the loaded records.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DataValidationError("factor-plan seed must be an integer")
    recurring_pool, rare_pool = _factor_choices(calibration_records)
    rng = Random(seed)
    # Inputs are sorted before seeded indexing, so results do not depend on the
    # caller's task ordering or dictionary insertion order.
    recurring = recurring_pool[rng.randrange(len(recurring_pool))]
    rare_candidates = [
        choice for choice in rare_pool if not _obvious_near_variant(choice.label, recurring.label)
    ]
    if not rare_candidates:
        raise DataValidationError("no rare factor is distinct from the sampled recurring factor")
    rare = rare_candidates[rng.randrange(len(rare_candidates))]

    # Reuse the primary recurring/rare contrast in the first two-factor call.
    # A second independently selected pair broadens the factor-seeded sample.
    excluded = {recurring.label, rare.label}
    try:
        recurring_two = _pick_compatible(
            recurring_pool, against=rare, rng=rng, excluded=excluded
        )
    except DataValidationError:
        # Small fixtures may contain only one recurring label.  Reusing it in
        # the other mode is preferable to inventing or rewriting a factor.
        recurring_two = recurring
    excluded.add(recurring_two.label)
    try:
        rare_two = _pick_compatible(
            rare_pool, against=recurring_two, rng=rng, excluded=excluded
        )
    except DataValidationError:
        compatible = [
            choice
            for choice in rare_pool
            if not _obvious_near_variant(choice.label, recurring_two.label)
        ]
        if not compatible:
            raise DataValidationError(
                f"no rare factor is distinct from {recurring_two.label!r}"
            )
        rare_two = compatible[rng.randrange(len(compatible))]

    return FactorPlan(
        recurring=recurring,
        rare=rare,
        single_factor_calls=((recurring.label,), (rare.label,)),
        two_factor_calls=(
            (recurring.label, rare.label),
            (recurring_two.label, rare_two.label),
        ),
        seed=seed,
    )
