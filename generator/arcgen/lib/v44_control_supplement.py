"""Fresh ordinary-control completion supplement for the V4.4 study.

The original matched experiment intentionally reused a frozen control supply.
Across repeated slates that produced too few independent control kernels.  This
non-production supplement generates fresh ordinary concept-led kernels from
the same calibration contexts and sends only the strongest through the exact
V4.4 downstream path.  It adds no abstraction instruction and cannot promote
features or commit tasks.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence
import time

from .util import atomic_write_json, atomic_write_text, content_hash, parallel_map, read_json
from .v41_experiments import V41Runtime
from .v41_supported_departure import _realize_records, _run_blind, _run_technical
from .v41_test_challenge import _board as _single_instance_boards
from .v43_episode_joint import _arm_quality
from .v44_abstraction_led import (
    AbstractionLedError,
    _audit_status,
    _budgets,
    _human_pack,
    _initialize_records,
    _install_extensions,
    _mechanism_audit,
    _plan_records,
    _select_kernels,
)
from .v44_config import V44_PIPELINE, load_v44_config


V44_CONTROL_SUPPLEMENT = "arcgen-v4.4-control-completion-supplement"


class V44ControlSupplementError(AbstractionLedError):
    pass


def _verify_source(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config["_repository_root"])).resolve()
    settings = config["control_supplement"]
    source = Path(str(settings["calibration_source_run"]))
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    expected = settings.get("source_artifact_sha256")
    if not source.is_dir() or not isinstance(expected, Mapping) or not expected:
        raise V44ControlSupplementError("a checksummed calibration source run is required")
    verified = {}
    for relative, digest in expected.items():
        path = source / str(relative)
        if not path.is_file():
            raise V44ControlSupplementError(f"missing control source artifact {relative}")
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise V44ControlSupplementError(
                f"control source artifact changed: {relative}: {actual} != {digest}"
            )
        verified[str(relative)] = actual
    return {
        "source_run": str(source),
        "artifact_sha256": verified,
        "snapshot_id": content_hash({"source": str(source), "artifacts": verified}),
    }


def load_v44_control_supplement_config(path: str | Path) -> dict[str, Any]:
    config = load_v44_config(path)
    settings = config.get("control_supplement")
    if not isinstance(settings, Mapping):
        raise V44ControlSupplementError("control_supplement settings are required")
    if settings.get("pipeline") != V44_CONTROL_SUPPLEMENT:
        raise V44ControlSupplementError(
            f"control_supplement.pipeline must be {V44_CONTROL_SUPPLEMENT}"
        )
    fixed = {
        "generation_calls": 6,
        "ideas_per_call": 4,
        "independent_shards": 3,
    }
    for key, expected in fixed.items():
        if settings.get(key) != expected:
            raise V44ControlSupplementError(f"control_supplement.{key} must be {expected}")
    maximum = settings.get("maximum_development")
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise V44ControlSupplementError(
            "control_supplement.maximum_development must be between 1 and 8"
        )
    if settings.get("automatic_promotion") is not False:
        raise V44ControlSupplementError("control supplement cannot promote itself")
    if settings.get("automatic_task_commit") is not False:
        raise V44ControlSupplementError("control supplement cannot commit tasks")
    if config["run"].get("automatically_commit_generated_tasks") is not False:
        raise V44ControlSupplementError("automatic task commitment must remain disabled")
    config["_v44_control_supplement_source"] = _verify_source(config)
    return config


def _candidate_id(seed: int, slate: int, pair: int, index: int) -> str:
    digest = sha256(
        f"v44-control-supplement:{seed}:{slate}:{pair}:{index}".encode()
    ).hexdigest()
    return "c" + digest[:13]


def _freeze_design(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    path = run_dir / "frozen_control_supplement_design.private.json"
    if path.exists():
        return read_json(path)
    source = Path(config["_v44_control_supplement_source"]["source_run"])
    source_design = read_json(source / "frozen_experiment_design.private.json")
    calls = []
    mapping = {}
    for item in source_design["paired_calls"]:
        slate = int(item["slate_index"])
        pair = int(item["pair_index"])
        ids = [
            _candidate_id(int(config["seed"]), slate, pair, index)
            for index in range(1, 5)
        ]
        mapping.update({candidate_id: "control" for candidate_id in ids})
        calls.append(
            {
                "slate_index": slate,
                "pair_index": pair,
                "calibration_task_ids": list(item["calibration_task_ids"]),
                "calibration_descriptions": list(item["calibration_descriptions"]),
                "control_call_id": f"v44-control-supplement-{slate:02d}-{pair:02d}",
                "control_candidate_ids": ids,
            }
        )
    if len(calls) != 6:
        raise V44ControlSupplementError("source must provide six matched calibration calls")
    value = {
        "schema_version": 1,
        "pipeline": V44_CONTROL_SUPPLEMENT,
        "calibration_source": deepcopy(config["_v44_control_supplement_source"]),
        "calls": calls,
        "private_candidate_arm_mapping": mapping,
        "ordinary_concept_prompt_unchanged": True,
        "abstraction_instruction_absent": True,
        "same_downstream_path_as_v44": True,
        "automatic_promotion": False,
        "automatic_task_commit": False,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        run_dir / "frozen_control_supplement_design.public.json",
        {key: deepcopy(item) for key, item in value.items() if key != "private_candidate_arm_mapping"},
    )
    return value


def _generate(runtime: V41Runtime, design: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = runtime.run_dir / "kernel_supply_manifest.private.json"
    if path.exists():
        return read_json(path)["kernels"]

    def one(call: Mapping[str, Any]) -> list[dict[str, Any]]:
        ids = list(call["control_candidate_ids"])
        call_id = str(call["control_call_id"])

        def validate(value: dict[str, Any]) -> None:
            if [item["candidate_id"] for item in value["ideas"]] != ids:
                raise V44ControlSupplementError("ordinary idea call changed assigned IDs")

        response = runtime.call(
            prompt_name="idea_concept",
            call_id=call_id,
            values={
                "CALIBRATION_DESCRIPTIONS": list(call["calibration_descriptions"]),
                "SEARCH_CALL_ID": call_id,
                "CANDIDATE_IDS": ids,
            },
            parent_ids=[],
            role="idea_search",
            arm_id="private-control-idea-supply",
            expected={"search_call_id": call_id, "search_mode": "concept_led"},
            validator=validate,
        )
        return [
            {
                "candidate_id": item["candidate_id"],
                "slate_index": int(call["slate_index"]),
                "pair_index": int(call["pair_index"]),
                "private_arm": "control",
                "kernel": deepcopy(item),
                "calibration_task_ids": list(call["calibration_task_ids"]),
                "generation_call_id": call_id,
            }
            for item in response["ideas"]
        ]

    rows = parallel_map(
        one,
        list(design["calls"]),
        min(len(design["calls"]), int(runtime.config["model"]["concurrency"])),
    )
    kernels = [item for group in rows for item in group]
    if len(kernels) != 24 or len({item["candidate_id"] for item in kernels}) != 24:
        raise V44ControlSupplementError("fresh control supply must contain 24 kernels")
    value = {
        "schema_version": 1,
        "fresh_ordinary_control_supply": True,
        "abstraction_instruction_absent": True,
        "new_control_idea_calls": 6,
        "kernel_count": 24,
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
                    "kernel": item["kernel"],
                }
                for item in kernels
            ],
        },
    )
    return kernels


def _markdown(report: Mapping[str, Any]) -> str:
    funnel = report["funnel"]
    lines = [
        "# V4.4 fresh ordinary-control completion supplement",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Recommendation: **{report['recommendation']}**",
        "",
        "This supplement used the unchanged ordinary concept-led prompt with the same "
        "calibration contexts as V4.4. It supplied missing fresh controls only; it did not "
        "change or promote the abstraction treatment.",
        "",
        "## Funnel",
        "",
        f"- Fresh kernels / shard-qualified / developed: {funnel['kernels']} / {funnel['shard_qualified']} / {funnel['developed']}",
        f"- Plans ready / realized: {funnel['plans_ready']} / {funnel['realized']}",
        f"- Technical / blind clear: {funnel['technical_pass']} / {funnel['blind_clear']}",
        f"- Core / reserve / ambiguous / other: {funnel['core']} / {funnel['reserve']} / {funnel['ambiguous']} / {funnel['other']}",
        f"- Genuine abstraction observed naturally: {report['naturally_genuine_abstraction_count']}",
        "",
        "## Tasks",
        "",
        "| Task | Audit | Technical | Blind | Intrinsic | Portfolio |",
        "|---|---|---|---|---|---|",
    ]
    for item in report["tasks"]:
        lines.append(
            f"| `{item['task_id']}` | `{item['mechanism_status']}` | `{item['technical_pass']}` | "
            f"`{item['blind_clear']}` | `{item['intrinsic_quality'] or item['terminal_status']}` | "
            f"`{item['portfolio_placement'] or 'not_applicable'}` |"
        )
    lines.extend(
        [
            "",
            f"Logical calls: {report['call_accounting']['experimental_calls']}  ",
            f"Active model-minutes: {report['call_accounting']['active_model_minutes']}  ",
            f"Wall-clock minutes: {report['timing']['wall_clock_minutes']}",
            "",
            "No production promotion, reservoir mutation, or good-task-bank mutation occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def run_v44_control_supplement(config: dict[str, Any]) -> dict[str, Any]:
    if config["control_supplement"]["pipeline"] != V44_CONTROL_SUPPLEMENT:
        raise V44ControlSupplementError("isolated control supplement config is required")
    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    start_path = run_dir / "experiment_start.json"
    if not start_path.exists():
        atomic_write_json(start_path, {"started_at_epoch_seconds": round(time.time(), 3)})
    design = _freeze_design(config, run_dir)
    runtime = V41Runtime(config, design)
    _install_extensions(runtime)
    if runtime.state.get("completed") and (run_dir / "report.json").is_file():
        return read_json(run_dir / "report.json")
    kernels = _generate(runtime, design)
    selected, selection = _select_kernels(runtime, kernels)
    maximum = int(config["control_supplement"]["maximum_development"])
    selected = selected[:maximum]
    atomic_write_json(
        run_dir / "control_supplement_development_sample.private.json",
        {
            "schema_version": 1,
            "quality_order_preserved": True,
            "maximum_development": maximum,
            "candidate_ids": [item["candidate_id"] for item in selected],
            "sample": selected,
        },
    )
    _initialize_records(runtime, selected)
    _plan_records(runtime)
    _realize_records(runtime)
    budgets = _budgets(runtime)
    _run_technical(runtime, budgets)
    _run_blind(runtime, budgets)
    _single_instance_boards(runtime)
    audit = _mechanism_audit(runtime)
    human = _human_pack(runtime, audit)
    audit_by_id = {item["task_id"]: item for item in audit}
    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda item: int(item["selector_global_priority"]),
    )
    status = Counter(item.get("status") for item in records)
    core = sum(item.get("intrinsic_quality") == "core_quality" for item in records)
    reserve = sum(item.get("intrinsic_quality") == "reserve_quality" for item in records)
    realized = sum(bool(item.get("complete_task")) for item in records)
    technical = sum(
        bool(item.get("technical_results"))
        and item["technical_results"][-1].get("passed") is True
        for item in records
    )
    blind = sum(item.get("status") in {"blind_pass", "editorial_complete"} for item in records)
    ready = sum(item.get("concrete_plan", {}).get("status") == "ready" for item in records)
    genuine = sum(
        _audit_status(item) == "genuine" for item in audit
    )
    accounting = runtime.accounting()
    started = read_json(start_path)["started_at_epoch_seconds"]
    completed = round(time.time(), 3)
    report = {
        "schema_version": 1,
        "pipeline": V44_CONTROL_SUPPLEMENT,
        "run_id": runtime.run_id,
        "completed": True,
        "calibration_source": deepcopy(config["_v44_control_supplement_source"]),
        "funnel": {
            "kernels": len(kernels),
            "shard_qualified": sum(item["selected_count"] for item in selection["shards"]),
            "developed": len(records),
            "plans_ready": ready,
            "realized": realized,
            "technical_pass": technical,
            "blind_clear": blind,
            "core": core,
            "reserve": reserve,
            "ambiguous": status["ambiguous"],
            "other": len(records) - core - reserve - status["ambiguous"],
        },
        "ordinary_control_quality": _arm_quality(records),
        "naturally_genuine_abstraction_count": genuine,
        "mechanism_audit_non_gating": True,
        "human_review": human,
        "call_accounting": accounting,
        "timing": {
            "started_at_epoch_seconds": started,
            "completed_at_epoch_seconds": completed,
            "wall_clock_minutes": round((completed - started) / 60.0, 3),
        },
        "tasks": [
            {
                "task_id": item["task_id"],
                "mechanism_status": _audit_status(audit_by_id[item["task_id"]])
                if item["task_id"] in audit_by_id
                else "not_audited",
                "terminal_status": item.get("status"),
                "technical_pass": bool(item.get("technical_results"))
                and item["technical_results"][-1].get("passed") is True,
                "blind_clear": item.get("status") in {"blind_pass", "editorial_complete"},
                "intrinsic_quality": item.get("intrinsic_quality"),
                "portfolio_placement": item.get("portfolio_placement"),
                "task_json": str(run_dir / "tasks" / "control" / f"{item['task_id']}.arc.json")
                if item.get("complete_task")
                else None,
            }
            for item in records
        ],
        "recommendation": (
            "control_sample_complete" if realized >= 4 else "continue_control_sampling"
        ),
        "production_promotion_occurred": False,
        "good_task_bank_mutated": False,
        "production_reservoir_mutated": False,
    }
    runtime.state["completed"] = True
    runtime.state["results"] = report
    runtime.save()
    atomic_write_json(run_dir / "report.json", report)
    atomic_write_text(run_dir / "report.md", _markdown(report))
    return report


__all__ = [
    "V44_CONTROL_SUPPLEMENT",
    "V44ControlSupplementError",
    "load_v44_control_supplement_config",
    "run_v44_control_supplement",
]
