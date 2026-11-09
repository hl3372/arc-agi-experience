"""High-precision deterministic checks for concrete-plan contradictions.

E8 aims to prevent expensive realization of plans that already contradict
their own structured facts.  These checks are deliberately narrow: they do
not judge difficulty, novelty, scale, prose style, or semantic quality, and
they never turn a free-form concern into an automatic abort unless it conflicts
with a concrete structured value or exact row witness.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from .util import atomic_write_json, atomic_write_text, read_json

PIPELINE = "arcgen-e8-plan-contradiction-prequalification"
VERSION = "1.0"

# Only dimensions explicitly tied to the complete input canvas are safe to
# compare with the structured pair dimensions.  A bare ``7x7`` very often
# describes an embedded object, panel, or mask rather than the whole grid.
WHOLE_INPUT_DIMENSION_RE = re.compile(
    r"(?:"
    r"\b(?:on|fill|make|create|build|start\s+with)\s+(?:an?\s+|the\s+)?"
    r"(?:zero\s+)?(?P<h1>\d{1,2})\s*[x×]\s*(?P<w1>\d{1,2})\s+"
    r"(?:input\s+)?(?:canvas|grid)\b"
    r"|"
    r"\b(?:the\s+)?(?:input|input\s+grid|input\s+canvas)\s+(?:is|has\s+dimensions?)\s+"
    r"(?P<h2>\d{1,2})\s*[x×]\s*(?P<w2>\d{1,2})\b"
    r")",
    re.IGNORECASE,
)
ZERO_GRID_RE = re.compile(
    r"\bstart\s+with\s+(?:an?\s+)?\d{1,2}\s*[x×]\s*\d{1,2}\s+zero\s+grid\b",
    re.IGNORECASE,
)
BACKGROUND_FILL_RE = re.compile(
    r"\bcolou?r\s+([0-9])\s+fills\s+(?:the\s+)?(?:canvas|exterior|background)\b",
    re.IGNORECASE,
)
AUTHORITATIVE_ROWS_RE = re.compile(
    r"\bauthoritative\s+(?:input|output)\s+rows[^:]*\s+(?:are|:)\s*:\s*([^.]*)",
    re.IGNORECASE,
)
PAYLOAD_COUNT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twenty-six|twenty-seven|twenty-eight|twenty-nine|thirty)\s+payload\s+tiles\b",
    re.IGNORECASE,
)
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    **{f"twenty-{name}": 20 + value for name, value in list({
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }.items())},
    "thirty": 30,
}


class E8PlanError(RuntimeError):
    """Raised when the historical E8 census cannot be reproduced safely."""


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _number(value: str) -> int:
    return int(value) if value.isdigit() else NUMBER_WORDS[value.casefold()]


def _issue(code: str, field: str, evidence: str) -> dict[str, str]:
    return {"code": code, "field": field, "evidence": evidence}


def _authoritative_rows(text: str) -> list[str] | None:
    match = AUTHORITATIVE_ROWS_RE.search(text)
    if not match:
        return None
    rows = [row.strip() for row in match.group(1).split("/")]
    if not rows or any(not row or any(cell not in "0123456789" for cell in row) for row in rows):
        return None
    return rows


def _check_storyboard_pair(
    pair: Mapping[str, Any],
    *,
    split: str,
    background_color: int | None,
    payload_colors: set[int],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    pair_number = int(pair.get("pair_number") or 0)
    prefix = f"{split}_pairs[{pair_number}]"
    construction = str(pair.get("exact_input_construction") or "")
    placements = str(pair.get("precise_placements_quantities_relations_or_states") or "")
    dimensions = pair.get("input_dimensions") or {}
    height = dimensions.get("height")
    width = dimensions.get("width")
    declared = WHOLE_INPUT_DIMENSION_RE.search(construction)
    if declared:
        declared_height = int(declared.group("h1") or declared.group("h2"))
        declared_width = int(declared.group("w1") or declared.group("w2"))
    else:
        declared_height = declared_width = None
    if declared and (declared_height, declared_width) != (height, width):
        issues.append(
            _issue(
                "structured_text_dimension_conflict",
                f"{prefix}.input_dimensions",
                f"structured={height}x{width}; exact construction says {declared_height}x{declared_width}",
            )
        )
    if ZERO_GRID_RE.search(construction) and background_color not in {None, 0}:
        issues.append(
            _issue(
                "zero_grid_background_conflict",
                f"{prefix}.exact_input_construction",
                f"construction initializes a zero grid but structured background_color={background_color}",
            )
        )
    rows = _authoritative_rows(construction)
    if rows is not None:
        widths = {len(row) for row in rows}
        if len(rows) != height or widths != {width}:
            issues.append(
                _issue(
                    "authoritative_rows_dimension_conflict",
                    f"{prefix}.exact_input_construction",
                    f"authoritative rows are {len(rows)}x{sorted(widths)} but structured dimensions are {height}x{width}",
                )
            )
        if payload_colors:
            observed = sum(int(cell) in payload_colors for row in rows for cell in row)
            claims = {_number(match.group(1)) for match in PAYLOAD_COUNT_RE.finditer(construction + " " + placements)}
            for claim in sorted(claims):
                if claim != observed:
                    issues.append(
                        _issue(
                            "authoritative_payload_conservation_conflict",
                            f"{prefix}.precise_placements_quantities_relations_or_states",
                            f"exact rows contain {observed} payload-color cells but prose claims {claim} payload tiles",
                        )
                    )
    return issues


def check_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("status") != "ready":
        return {
            "checked": False,
            "reason": "plan is not ready",
            "issues": [],
            "deterministic_abort_recommended": False,
        }
    issues: list[dict[str, str]] = []
    overview = plan.get("episode_overview") or {}
    training = plan.get("training_pairs") or []
    tests = plan.get("test_pairs") or []
    expected_train = overview.get("training_pair_count")
    expected_test = overview.get("test_pair_count")
    if expected_train != len(training):
        issues.append(
            _issue(
                "training_pair_count_conflict",
                "episode_overview.training_pair_count",
                f"structured count={expected_train}; records={len(training)}",
            )
        )
    if expected_test != len(tests):
        issues.append(
            _issue(
                "test_pair_count_conflict",
                "episode_overview.test_pair_count",
                f"structured count={expected_test}; records={len(tests)}",
            )
        )
    for split, rows in (("training", training), ("test", tests)):
        numbers = [row.get("pair_number") for row in rows if isinstance(row, Mapping)]
        expected = list(range(1, len(rows) + 1))
        if numbers != expected:
            issues.append(
                _issue(
                    f"{split}_pair_number_conflict",
                    f"{split}_pairs",
                    f"pair numbers={numbers}; expected={expected}",
                )
            )
    background = overview.get("background_color")
    treatment = str(overview.get("background_treatment") or "")
    prose_backgrounds = {int(match.group(1)) for match in BACKGROUND_FILL_RE.finditer(treatment)}
    if background is not None and any(color != background for color in prose_backgrounds):
        issues.append(
            _issue(
                "background_treatment_structured_conflict",
                "episode_overview.background_treatment",
                f"structured background_color={background}; treatment names {sorted(prose_backgrounds)} as canvas/exterior fill",
            )
        )
    roles = overview.get("functional_color_roles") or []
    named_backgrounds = {
        int(role["color"])
        for role in roles
        if isinstance(role, Mapping)
        and "background" in str(role.get("role") or "").casefold()
        and isinstance(role.get("color"), int)
    }
    if background is not None and any(color != background for color in named_backgrounds):
        issues.append(
            _issue(
                "background_role_structured_conflict",
                "episode_overview.functional_color_roles",
                f"structured background_color={background}; named background roles use {sorted(named_backgrounds)}",
            )
        )
    payload_colors = {
        int(role["color"])
        for role in roles
        if isinstance(role, Mapping)
        and "payload identity" in str(role.get("role") or "").casefold()
        and isinstance(role.get("color"), int)
    }
    for pair in training:
        if isinstance(pair, Mapping):
            issues.extend(
                _check_storyboard_pair(
                    pair,
                    split="training",
                    background_color=background,
                    payload_colors=payload_colors,
                )
            )
    for pair in tests:
        if isinstance(pair, Mapping):
            issues.extend(
                _check_storyboard_pair(
                    pair,
                    split="test",
                    background_color=background,
                    payload_colors=payload_colors,
                )
            )
    canvas = plan.get("canvas_plan") or []
    expected_labels = [
        *[f"train_{index}" for index in range(1, len(training) + 1)],
        *[f"test_{index}" for index in range(1, len(tests) + 1)],
    ]
    actual_labels = [row.get("pair") for row in canvas if isinstance(row, Mapping)]
    if actual_labels != expected_labels:
        issues.append(
            _issue(
                "canvas_plan_pair_coverage_conflict",
                "canvas_plan",
                f"labels={actual_labels}; expected={expected_labels}",
            )
        )
    return {
        "checked": True,
        "issue_count": len(issues),
        "issues": issues,
        "deterministic_abort_recommended": bool(issues),
        "scope": "objective_structured_or_exact-witness contradictions only",
    }


def _plan_from_authoring(authoring: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("concrete_plan", "plan"):
        value = authoring.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def historical_census(repository_root: Path, output: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    registry_path = root / "committed_tasks/registry.json"
    registry = read_json(registry_path)
    cases = []
    for entry in registry.get("entries") or []:
        dossier = registry_path.parent / str(entry.get("dossier_path") or "")
        authoring_path = dossier.parent / "authoring.json"
        if not authoring_path.is_file():
            continue
        authoring = read_json(authoring_path)
        plan = _plan_from_authoring(authoring)
        if plan is None:
            continue
        cases.append(
            {
                "case_id": str(entry["registry_task_id"]),
                "source": "committed_core_or_reserve",
                "expected_late_failure": False,
                "authoring_path": str(authoring_path.resolve()),
                "authoring_sha256": _sha(authoring_path),
                "result": check_plan(plan),
            }
        )
    known = (
        (
            "known-realizer-refusal-q9hagaywjm9x",
            root
            / "runs/e2-live-streaming-cache-preflight-20260812/final_state_exports/authoring/q9hagaywjm9x.json",
        ),
        (
            "known-technical-plan-conflict-a2u23mk1w3pg",
            root
            / "runs/v4-8-evidence-scale-paired-01-20260812/execution/v45_original_scale/tasks/single/a2u23mk1w3pg.authoring.json",
        ),
    )
    for case_id, authoring_path in known:
        authoring = read_json(authoring_path)
        plan = _plan_from_authoring(authoring)
        assert plan is not None
        cases.append(
            {
                "case_id": case_id,
                "source": "known_late_failure",
                "expected_late_failure": True,
                "authoring_path": str(authoring_path.resolve()),
                "authoring_sha256": _sha(authoring_path),
                "result": check_plan(plan),
            }
        )
    controls = [row for row in cases if not row["expected_late_failure"]]
    failures = [row for row in cases if row["expected_late_failure"]]
    false_flags = [row for row in controls if row["result"]["deterministic_abort_recommended"]]
    caught = [row for row in failures if row["result"]["deterministic_abort_recommended"]]
    result = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "completed": True,
        "model_calls": 0,
        "committed_registry": {"path": str(registry_path), "sha256": _sha(registry_path)},
        "committed_plan_count": len(controls),
        "known_late_failure_count": len(failures),
        "known_late_failures_caught": len(caught),
        "committed_plan_false_flag_count": len(false_flags),
        "false_flag_case_ids": [row["case_id"] for row in false_flags],
        "known_failure_results": failures,
        "issue_code_counts": dict(
            sorted(
                Counter(
                    issue["code"]
                    for row in cases
                    for issue in row["result"]["issues"]
                ).items()
            )
        ),
        "cases": cases,
        "automatic_abort_enabled": False,
        "production_prompt_or_schema_changed": False,
        "recommendation": (
            "candidate_for_paired_planner_study"
            if len(caught) == len(failures) and not false_flags
            else "refine_deterministic_checks_before_paired_study"
        ),
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "report.json", result)
    atomic_write_text(
        output / "report.md",
        "\n".join(
            [
                "# E8 concrete-plan contradiction prequalification",
                "",
                f"Committed plans checked: {len(controls)}",
                f"Known late failures caught: {len(caught)}/{len(failures)}",
                f"Committed-plan false flags: {len(false_flags)}",
                f"Recommendation: `{result['recommendation']}`",
                "",
                "Only objective structured-field and exact-witness contradictions were checked. No model call, automatic abort, prompt change, or production mutation occurred.",
                "",
            ]
        ),
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census", choices=("census",))
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = historical_census(Path(args.repository_root), Path(args.output))
        print(value["recommendation"])
        return 0
    except Exception as exc:  # noqa: BLE001 - fail-closed audit CLI
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["E8PlanError", "check_plan", "historical_census"]
