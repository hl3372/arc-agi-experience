"""Deterministic, semantics-preserving palette treatment for ARCGEN V4.1.

The treatment operates after realization on a complete episode envelope.  It
never creates a second concept, never recolors pairs independently, and keeps a
machine-checkable inverse mapping for every public and hidden grid.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from math import log2
from typing import Any, Iterable, Mapping, Sequence
import re

from .data import TaskRecord
from .util import content_hash
from .validation import validate_arc_task


class V41PaletteError(ValueError):
    pass


def _all_complete_grids(task: Mapping[str, Any]) -> Iterable[list[list[int]]]:
    for split in ("train", "test"):
        for pair in task.get(split, []):
            yield pair["input"]
            if "output" in pair:
                yield pair["output"]


def _input_grids(task: Mapping[str, Any]) -> Iterable[list[list[int]]]:
    for split in ("train", "test"):
        for pair in task.get(split, []):
            yield pair["input"]


def _modal_color(grid: Sequence[Sequence[int]]) -> tuple[int, float]:
    counts = Counter(int(cell) for row in grid for cell in row)
    color, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return color, count / sum(counts.values())


def infer_episode_background(
    task: Mapping[str, Any],
    *,
    modal_fraction: float = 0.5,
    require_consistent_input_mode: bool = True,
) -> int | None:
    """Conservatively infer one background from solver-visible inputs only."""

    observations = [_modal_color(grid) for grid in _input_grids(task)]
    if not observations or any(fraction < modal_fraction for _, fraction in observations):
        return None
    colors = {color for color, _ in observations}
    if require_consistent_input_mode and len(colors) != 1:
        return None
    return observations[0][0] if len(colors) == 1 else None


def measure_background_distribution(
    tasks: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    modal_fraction: float,
    require_consistent_input_mode: bool,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    ids: dict[str, list[str]] = {}
    total = 0
    for task_id, task in tasks:
        total += 1
        background = infer_episode_background(
            task,
            modal_fraction=modal_fraction,
            require_consistent_input_mode=require_consistent_input_mode,
        )
        key = "none" if background is None else str(background)
        counts[key] += 1
        ids.setdefault(key, []).append(task_id)
    confident = total - counts.get("none", 0)
    frequencies = {
        str(color): counts.get(str(color), 0) / confident if confident else 0.0
        for color in range(10)
    }
    return {
        "heuristic": {
            "inputs_only": True,
            "modal_fraction_minimum": modal_fraction,
            "require_same_modal_color_across_inputs": require_consistent_input_mode,
            "outputs_excluded_from_background_inference": True,
        },
        "task_count": total,
        "confident_background_count": confident,
        "unresolved_count": counts.get("none", 0),
        "counts": dict(sorted(counts.items())),
        "frequencies_among_confident": frequencies,
        "task_ids_by_background": ids,
    }


def measure_real_bank_backgrounds(
    bank: Iterable[TaskRecord],
    *,
    modal_fraction: float,
    require_consistent_input_mode: bool,
) -> dict[str, Any]:
    tasks = (
        (
            record.task_id,
            {"train": list(record.train), "test": list(record.test)},
        )
        for record in bank
    )
    return measure_background_distribution(
        tasks,
        modal_fraction=modal_fraction,
        require_consistent_input_mode=require_consistent_input_mode,
    )


def soft_target_distribution(
    real_distribution: Mapping[str, Any],
    *,
    allowed_colors: Sequence[int],
    real_reference_weight: float,
    diversity_weight: float,
) -> dict[int, float]:
    if not allowed_colors or len(set(allowed_colors)) != len(allowed_colors):
        raise V41PaletteError("allowed background colors must be unique")
    if any(type(color) is not int or not 0 <= color <= 9 for color in allowed_colors):
        raise V41PaletteError("allowed background colors must be ARC colors")
    if 0 not in allowed_colors:
        raise V41PaletteError("black must remain eligible")
    if abs(real_reference_weight + diversity_weight - 1.0) > 1e-9:
        raise V41PaletteError("palette weights must sum to one")
    uniform = 1.0 / len(allowed_colors)
    real = real_distribution["frequencies_among_confident"]
    values = {
        color: real_reference_weight * float(real.get(str(color), 0.0))
        + diversity_weight * uniform
        for color in allowed_colors
    }
    total = sum(values.values())
    if total <= 0:
        raise V41PaletteError("palette target has no probability mass")
    return {color: value / total for color, value in values.items()}


def choose_preferred_background(
    *,
    seed: int,
    assignment_id: str,
    current_counts: Mapping[str, int],
    target_distribution: Mapping[int, float],
    allowed_targets: Sequence[int] | None = None,
) -> int:
    candidates = set(target_distribution) if allowed_targets is None else set(allowed_targets)
    if not candidates or not candidates <= set(target_distribution):
        raise V41PaletteError("candidate-specific palette targets are empty or invalid")
    assigned = sum(int(current_counts.get(str(color), 0)) for color in target_distribution)
    desired_total = assigned + 1
    deficits = {
        color: desired_total * weight - int(current_counts.get(str(color), 0))
        for color, weight in target_distribution.items()
    }
    return min(
        candidates,
        key=lambda color: (
            -deficits[color],
            content_hash({"seed": seed, "assignment_id": assignment_id, "color": color}),
            color,
        ),
    )


def schedule_backgrounds(
    task_ids: Sequence[str],
    *,
    seed: int,
    current_counts: Mapping[str, int],
    target_distribution: Mapping[int, float],
    eligible: Mapping[str, bool] | None = None,
    allowed_targets_by_task: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, int | None]:
    counts = Counter({str(key): int(value) for key, value in current_counts.items()})
    assignments: dict[str, int | None] = {}
    for task_id in task_ids:
        if eligible is not None and not eligible.get(task_id, False):
            assignments[task_id] = None
            continue
        color = choose_preferred_background(
            seed=seed,
            assignment_id=task_id,
            current_counts=counts,
            target_distribution=target_distribution,
            allowed_targets=(
                allowed_targets_by_task.get(task_id)
                if allowed_targets_by_task is not None
                else None
            ),
        )
        assignments[task_id] = color
        counts[str(color)] += 1
    return assignments


def palette_permutation(source_background: int, target_background: int) -> dict[int, int]:
    if not 0 <= source_background <= 9 or not 0 <= target_background <= 9:
        raise V41PaletteError("palette colors must be in 0..9")
    mapping = {color: color for color in range(10)}
    mapping[source_background], mapping[target_background] = target_background, source_background
    validate_bijection(mapping)
    return mapping


def validate_bijection(mapping: Mapping[int, int]) -> None:
    if set(mapping) != set(range(10)) or set(mapping.values()) != set(range(10)):
        raise V41PaletteError("palette mapping must be a global ARC-color bijection")


def inverse_palette(mapping: Mapping[int, int]) -> dict[int, int]:
    validate_bijection(mapping)
    return {value: key for key, value in mapping.items()}


def remap_grid(grid: Sequence[Sequence[int]], mapping: Mapping[int, int]) -> list[list[int]]:
    validate_bijection(mapping)
    return [[mapping[int(cell)] for cell in row] for row in grid]


def remap_complete_task(task: Mapping[str, Any], mapping: Mapping[int, int]) -> dict[str, Any]:
    result = deepcopy(dict(task))
    for split in ("train", "test"):
        for pair in result[split]:
            pair["input"] = remap_grid(pair["input"], mapping)
            if "output" in pair:
                pair["output"] = remap_grid(pair["output"], mapping)
    validation = validate_arc_task(result, require_test_outputs=True)
    if not validation.valid:
        raise V41PaletteError("mapped task is invalid: " + "; ".join(validation.errors))
    return result


ARC_COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "fuchsia", 7: "orange", 8: "azure", 9: "maroon",
}


def used_colors(task: Mapping[str, Any]) -> set[int]:
    return {int(cell) for grid in _all_complete_grids(task) for row in grid for cell in row}


def _remap_text(text: str, mapping: Mapping[int, int]) -> str:
    """Update explicit color claims without changing coordinate/count numbers."""

    tokens = {color: f"__ARC_COLOR_{color}__" for color in range(10)}
    value = text
    # Named ARC colors are unambiguously palette claims in authoring prose.
    for color, name in ARC_COLOR_NAMES.items():
        value = re.sub(rf"\b{re.escape(name)}\b", tokens[color], value, flags=re.IGNORECASE)
    # Numeric claims are changed only when explicitly introduced as colors,
    # codes, values, backgrounds, or fill/empty values. Coordinates and counts
    # remain untouched.
    patterns = (
        r"(?P<prefix>\bcolor\s*[-:=]?\s*)(?P<color>[0-9])\b",
        r"(?P<prefix>\b(?:cell\s+)?(?:value|code|digit)\s*[-:=]?\s*)(?P<color>[0-9])\b",
        r"(?P<prefix>\bbackground\s*(?:color)?\s*[-:=]?\s*)(?P<color>[0-9])\b",
        r"(?P<prefix>\b(?:empty|blank|zero-valued)\s*=\s*)(?P<color>[0-9])\b",
        r"(?P<prefix>\b(?:filled|fill|initialized|initialize|remain|remains|become|becomes|replace|replaced|write|written)\s+(?:with|as|to)?\s*)(?P<color>[0-9])\b",
        r"(?P<prefix>\b(?:fill|filled|allocate|allocated|create|created|initialize|initialized)\b[^.!?\n]{0,120}?\b(?:with|as|to)\s*(?:color\s*)?)(?P<color>[0-9])\b",
        r"(?P<prefix>\b(?:all|every)\s+(?:(?:other|remaining|unspecified|unlisted)\s+)?cells?\s+(?:are|is|remain|remains|stay|stays)\s*(?:color\s*)?)(?P<color>[0-9])\b",
        r"(?P<prefix>\b(?:from|changing\s+from|changes?\s+from)\s*)(?P<color>[0-9])(?=\s+(?:to|into)\s+[0-9])",
    )
    for pattern in patterns:
        value = re.sub(
            pattern,
            lambda match: match.group("prefix") + tokens[int(match.group("color"))],
            value,
            flags=re.IGNORECASE,
        )
    # Source-background prose often uses these compact forms.  They are not
    # coordinates or counts and can therefore be rewritten deterministically.
    for color, token in tokens.items():
        value = re.sub(
            rf"\b{color}(?=\s+(?:background|floor|empty|blank|canvas)\b)",
            token,
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            rf"\b{color}(?=-to-[0-9]\b)", token, value, flags=re.IGNORECASE
        )
    value = re.sub(r"\bzero(?:-|\s)?filled\b", "filled with color " + tokens[0], value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:zeros|zeroes)\b", "color " + tokens[0] + " cells", value, flags=re.IGNORECASE)
    for color, token in tokens.items():
        mapped = mapping[color]
        # Prefer the numeric identifier: unlike a natural-language color name,
        # it preserves the exact ARC construction contract.
        value = value.replace(token, str(mapped))
    return value


def stale_palette_claims(value: Any, *, source_color: int, target_color: int) -> list[str]:
    """Find likely source-background claims left after a non-identity rewrite.

    The detector deliberately targets role-bearing phrases, never bare digits,
    so coordinates, dimensions, times, and counts do not become false alarms.
    """

    if source_color == target_color:
        return []
    text = str(value)
    name = ARC_COLOR_NAMES[source_color]
    number = re.escape(str(source_color))
    patterns = (
        rf"\b{re.escape(name)}\b",
        rf"\bcolor\s*[-:=]?\s*{number}\b",
        rf"\b(?:fill|filled|allocate|allocated|create|created|initialize|initialized)\b[^.!?\n]{{0,120}}?\b(?:with|as|to)\s*(?:color\s*)?{number}\b",
        rf"\b(?:all|every)\s+(?:(?:other|remaining|unspecified|unlisted)\s+)?cells?\s+(?:are|is|remain|remains|stay|stays)\s*(?:color\s*)?{number}\b",
        rf"\b{number}\s+(?:background|floor|empty|blank|canvas)\b",
        rf"\b(?:from|changing\s+from|changes?\s+from)\s*{number}\s+(?:to|into)\s+[0-9]\b",
        rf"\b{number}-to-[0-9]\b",
        r"\bzero(?:-|\s)?filled\b" if source_color == 0 else r"(?!)",
        r"\b(?:zeros|zeroes)\b" if source_color == 0 else r"(?!)",
    )
    matches = []
    for pattern in patterns:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            matches.append(found.group(0))
    return matches


def remap_authoring_contract(value: Any, mapping: Mapping[int, int], *, field: str = "$") -> Any:
    """Materialize a final-palette contract without exposing arm or mapping."""

    validate_bijection(mapping)
    if isinstance(value, str):
        return _remap_text(value, mapping)
    if isinstance(value, list):
        return [remap_authoring_contract(item, mapping, field=f"{field}[]") for item in value]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key in {"background_color", "color"} and type(item) is int and 0 <= item <= 9:
                result[key] = mapping[item]
            else:
                result[key] = remap_authoring_contract(item, mapping, field=f"{field}.{key}")
        return result
    return deepcopy(value)


def _public_and_hidden(complete_task: Mapping[str, Any]) -> tuple[dict[str, Any], list[list[list[int]]]]:
    public = deepcopy(dict(complete_task))
    hidden = []
    for pair in public["test"]:
        if "output" not in pair:
            raise V41PaletteError("complete palette source lacks a test output")
        hidden.append(pair.pop("output"))
    return public, hidden


def palette_equivalent_hash(task: Mapping[str, Any]) -> str:
    """Canonicalize first-seen colors so global reskins share one identity."""

    renaming: dict[int, int] = {}
    next_color = 0
    normalized = deepcopy(dict(task))
    for grid in _all_complete_grids(normalized):
        for row_index, row in enumerate(grid):
            for column_index, value in enumerate(row):
                color = int(value)
                if color not in renaming:
                    renaming[color] = next_color
                    next_color += 1
                row[column_index] = renaming[color]
    return content_hash(normalized)


_LITERAL_NUMERIC_SEMANTICS = re.compile(
    r"(?:numeric|arithmetic|magnitude|ordered|ordinal)\s+(?:value|meaning|color)|"
    r"(?:add|sum|subtract|multiply|divide|modulo|greater|less)\s+(?:the\s+)?color|"
    # A global swap can preserve nominal color roles, but it does not preserve
    # predicates over the ARC identifiers themselves.  In particular, moving
    # a color-0 background would make "nonzero cells" denote a different set.
    # Rewriting such predicates into an equivalent disjunction is deliberately
    # outside this conservative surface-only treatment, so these tasks opt out.
    r"\bnon[- ]?zero\b|"
    r"\bzero[- ]?valued\b|"
    r"\bpositive[- ]?(?:valued\s+)?(?:cell|cells|color|colors|value|values)\b|"
    r"\b(?:cell|cells|color|colors|value|values)\b[^.!?\n]{0,40}"
    r"(?:!=|==|not\s+equal\s+to|equal\s+to|greater\s+than|less\s+than)\s*[0-9]\b",
    re.IGNORECASE,
)


def palette_safety_assessment(
    *,
    complete_task: Mapping[str, Any],
    declared_background: int | None,
    authoring_text: Any,
    modal_fraction: float,
    require_consistent_input_mode: bool,
) -> dict[str, Any]:
    inferred = infer_episode_background(
        complete_task,
        modal_fraction=modal_fraction,
        require_consistent_input_mode=require_consistent_input_mode,
    )
    reasons: list[str] = []
    if declared_background is None:
        reasons.append("authoring contract declares no single meaningful background")
    if inferred is None:
        reasons.append("conservative input-only heuristic cannot infer one background")
    if declared_background is not None and inferred is not None and declared_background != inferred:
        reasons.append("declared and conservatively inferred backgrounds disagree")
    text = str(authoring_text)
    if _LITERAL_NUMERIC_SEMANTICS.search(text):
        reasons.append(
            "authoring contract assigns numeric semantics or membership predicates to color IDs"
        )
    if len(used_colors(complete_task)) == 10 and authoring_text:
        reasons.append(
            "all ARC colors are used, so no background swap can avoid rewriting "
            "potentially ambiguous literal construction claims"
        )
    resolved_background = (
        declared_background
        if declared_background is not None
        and inferred is not None
        and declared_background == inferred
        else None
    )
    return {
        "background_color": resolved_background,
        "declared_background_color": declared_background,
        "inferred_background_color": inferred,
        "palette_permutation_safe": not reasons,
        "reasons": reasons or ["one nominal background role is consistent across solver-visible inputs"],
    }


def make_episode_envelope(
    *,
    task_id: str,
    complete_task: Mapping[str, Any],
    intended_rule: str,
    concrete_plan: Mapping[str, Any] | None,
    source: str,
    modal_fraction: float,
    require_consistent_input_mode: bool,
) -> dict[str, Any]:
    validation = validate_arc_task(dict(complete_task), require_test_outputs=True)
    if not validation.valid:
        raise V41PaletteError(f"source task {task_id} is invalid")
    overview = (concrete_plan or {}).get("episode_overview") or {}
    declared = overview.get("background_color")
    if declared is None and concrete_plan is None:
        declared = infer_episode_background(
            complete_task,
            modal_fraction=modal_fraction,
            require_consistent_input_mode=require_consistent_input_mode,
        )
    roles = deepcopy(overview.get("functional_color_roles", []))
    if not roles:
        colors = sorted({cell for grid in _all_complete_grids(complete_task) for row in grid for cell in row})
        roles = [
            {
                "role": "background" if color == declared else f"functional color {color}",
                "color": color,
                "functional_meaning": "Recovered from the completed episode; consult the intended rule.",
            }
            for color in colors
        ]
    safety = palette_safety_assessment(
        complete_task=complete_task,
        declared_background=declared,
        authoring_text={"intended_rule": intended_rule, "concrete_plan": concrete_plan},
        modal_fraction=modal_fraction,
        require_consistent_input_mode=require_consistent_input_mode,
    )
    public, hidden = _public_and_hidden(complete_task)
    return {
        "task_id": task_id,
        "source": source,
        "complete_task": deepcopy(dict(complete_task)),
        "public_task": public,
        "hidden_test_outputs": hidden,
        "intended_rule": intended_rule,
        "concrete_plan": deepcopy(dict(concrete_plan)) if concrete_plan is not None else None,
        "background_color": safety["background_color"],
        "palette_permutation_safe": safety["palette_permutation_safe"],
        "color_roles": roles,
        "safety_assessment": safety,
        "exact_task_hash": content_hash(complete_task),
        "palette_equivalent_identity_hash": palette_equivalent_hash(complete_task),
    }


def apply_scheduled_palette(
    envelope: Mapping[str, Any], target_background: int | None
) -> dict[str, Any]:
    result = deepcopy(dict(envelope))
    source = envelope.get("background_color")
    evidence = {
        "applied": False,
        "source_background_color": source,
        "requested_background_color": target_background,
        "realized_background_color": source,
        "mapping": {str(color): color for color in range(10)},
        "inverse_round_trip_exact": True,
        "reason": "",
    }
    if target_background is None:
        evidence["reason"] = "scheduler opt-out"
        result["palette_treatment"] = evidence
        return result
    if not envelope.get("palette_permutation_safe") or source is None:
        evidence["reason"] = "task is not safe for deterministic palette permutation"
        result["palette_treatment"] = evidence
        return result
    mapping = palette_permutation(int(source), int(target_background))
    inverse = inverse_palette(mapping)
    mapped_task = remap_complete_task(envelope["complete_task"], mapping)
    restored = remap_complete_task(mapped_task, inverse)
    if restored != envelope["complete_task"]:
        raise V41PaletteError("inverse palette round-trip changed the task")
    if palette_equivalent_hash(mapped_task) != envelope["palette_equivalent_identity_hash"]:
        raise V41PaletteError("global recoloring changed palette-equivalent task identity")
    public, hidden = _public_and_hidden(mapped_task)
    effective_plan = remap_authoring_contract(envelope.get("concrete_plan"), mapping)
    effective_rule = remap_authoring_contract(envelope["intended_rule"], mapping)
    stale = [
        *stale_palette_claims(
            effective_rule,
            source_color=int(source),
            target_color=int(target_background),
        ),
        *stale_palette_claims(
            effective_plan,
            source_color=int(source),
            target_color=int(target_background),
        ),
    ]
    if stale:
        result["palette_permutation_safe"] = False
        result["safety_assessment"] = {
            **deepcopy(result["safety_assessment"]),
            "palette_permutation_safe": False,
            "reasons": [
                "deterministic rewrite left likely stale source-background claims: "
                + ", ".join(sorted(set(stale)))
            ],
        }
        evidence.update(
            {
                "reason": "scheduler opt-out after stale authoring-claim scan",
                "stale_authoring_claims": sorted(set(stale)),
            }
        )
        result["palette_treatment"] = evidence
        return result
    result.update(
        {
            "complete_task": mapped_task,
            "public_task": public,
            "hidden_test_outputs": hidden,
            "background_color": mapping[int(source)],
            "color_roles": [
                {**deepcopy(role), "color": mapping[int(role["color"])]}
                for role in envelope["color_roles"]
            ],
            "exact_task_hash": content_hash(mapped_task),
            "palette_equivalent_identity_hash": palette_equivalent_hash(mapped_task),
            "effective_concrete_plan": effective_plan,
            "effective_intended_rule": effective_rule,
        }
    )
    evidence.update(
        {
            "applied": int(source) != int(target_background),
            "realized_background_color": mapping[int(source)],
            "mapping": {str(key): value for key, value in mapping.items()},
            "reason": "one global bijection applied to every public and hidden grid and structured color role",
        }
    )
    result["palette_treatment"] = evidence
    return result


def background_metrics(counts: Mapping[str, int], reference: Mapping[str, Any]) -> dict[str, Any]:
    confident = sum(int(counts.get(str(color), 0)) for color in range(10))
    frequencies = {
        str(color): int(counts.get(str(color), 0)) / confident if confident else 0.0
        for color in range(10)
    }
    entropy = -sum(value * log2(value) for value in frequencies.values() if value > 0)
    nonblack = {key: value for key, value in frequencies.items() if key != "0"}
    dominant_nonblack = max(nonblack, key=nonblack.get) if nonblack else None
    reference_frequency = reference["frequencies_among_confident"]
    return {
        "counts": {str(color): int(counts.get(str(color), 0)) for color in range(10)},
        "confident_count": confident,
        "frequencies": frequencies,
        "black_background_share": frequencies["0"],
        "entropy_bits": round(entropy, 6),
        "dominant_nonblack_color": dominant_nonblack,
        "dominant_nonblack_share": nonblack.get(dominant_nonblack, 0.0) if dominant_nonblack else 0.0,
        "l1_distance_from_real_reference": round(
            sum(abs(frequencies[str(color)] - float(reference_frequency[str(color)])) for color in range(10)),
            6,
        ),
    }


def simulate_future_scheduling(
    *,
    seed: int,
    starting_counts: Mapping[str, int],
    target_distribution: Mapping[int, float],
    slate_count: int,
    tasks_per_slate: int,
) -> dict[str, Any]:
    counts = Counter({str(key): int(value) for key, value in starting_counts.items()})
    slates = []
    for slate in range(1, slate_count + 1):
        ids = [f"future-{slate:02d}-{index:02d}" for index in range(1, tasks_per_slate + 1)]
        assignments = schedule_backgrounds(
            ids,
            seed=seed,
            current_counts=counts,
            target_distribution=target_distribution,
        )
        for color in assignments.values():
            counts[str(color)] += 1
        slates.append({"slate": slate, "assignments": assignments, "counts_after": dict(sorted(counts.items()))})
    return {"slates": slates, "final_counts": dict(sorted(counts.items()))}


__all__ = [
    "V41PaletteError",
    "apply_scheduled_palette",
    "background_metrics",
    "choose_preferred_background",
    "infer_episode_background",
    "inverse_palette",
    "make_episode_envelope",
    "measure_background_distribution",
    "measure_real_bank_backgrounds",
    "palette_equivalent_hash",
    "palette_permutation",
    "palette_safety_assessment",
    "remap_complete_task",
    "remap_grid",
    "remap_authoring_contract",
    "schedule_backgrounds",
    "simulate_future_scheduling",
    "soft_target_distribution",
    "stale_palette_claims",
    "used_colors",
    "validate_bijection",
]
