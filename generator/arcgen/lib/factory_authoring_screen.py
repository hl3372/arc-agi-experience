"""Bounded fixed-kernel planning-effort study, using ordinary factory stages.

No generation, selector, publication, feature promotion or production-state
writer. Old outcomes are private references, never seeded as new stage results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import time

from .e18_artifacts import (attach_curriculum_statuses, materialize_terminal_artifacts,
                            render_tasks, strict_technical_clear, verify_terminal_artifact_bundle,
                            verify_render_artifacts)
from .e18_candidate_runtime import plan_one, run_slate
from .e18_evidence_gate import evidence_is_clear
from .e18_pre_e20_pool import DedicatedCallPool
from .factory import call_accounting, freeze_inputs, run_lock
from .factory_config import load_factory_config, resolved_config_manifest, verify_snapshot
from .factory_evidence import apply_final_evidence, recover_terminal_evidence_metadata
from .factory_metrics import cohort_metrics
from .factory_release import _terminal
from .factory_review import FactoryReviews
from .factory_runtime import FactoryRuntime
from .util import atomic_write_json, atomic_write_text, content_hash, read_json

TASKS = ("e189236754bce50", "e18b7bbc50342d9", "e187000a4b1404f",
         "e1826886d0ed38a", "e18bc002636aca1", "e1854f3468fb7ab")
INITIAL_FIELDS = ("task_id", "source_candidate_id", "lineage_id", "arm", "private_source",
                  "private_conditioned", "private_source_call_id", "private_idea_effort", "private_route",
                  "slate_index", "source_development_priority", "search_mode", "kernel", "editorial_brief", "novelty")


def initial_record(record):
    return {key: deepcopy(record[key]) for key in INITIAL_FIELDS} | {
        "status": "selected", "stage_history": [], "repair_count": 0, "candidate_revision": 0}


def planning_request(record):
    """Ask the unchanged stage helper to construct its request, without a call."""
    class Captured(Exception):
        pass
    def capture(**kwargs):
        raise Captured(kwargs)
    try:
        plan_one(SimpleNamespace(state={"semantic_records": {record["task_id"]: initial_record(record)}}, call=capture), record["task_id"])
    except Captured as result:
        return result.args[0]
    raise ValueError("planning stage did not emit exactly one request")


def check_profiles(config, reference):
    wanted = deepcopy(reference["model_profile"])
    if wanted["model"] != "gpt-6-astra" or wanted["efforts"]["planning"] != "xhigh":
        raise ValueError("requires an Astra xhigh-planning reference")
    wanted["efforts"]["planning"] = "high"
    if config["model_profile"] != wanted:
        raise ValueError("only planning effort may change")
    roles = deepcopy(reference["role_profile"])
    roles["planning"]["reasoning_effort"] = "high"
    if config["role_profile"] != roles:
        raise ValueError("resolved role routing differs beyond planning effort")
    for key in ("selection", "supply", "policy", "verification", "worker_policy", "seed", "transport"):
        if config[key] != reference[key]:
            raise ValueError(f"authoring study changes frozen {key}")


def prepare(config, source, task_ids=TASKS):
    root, source = Path(config["_output_directory"]), Path(source).resolve()
    if root.resolve() == source or root.resolve().is_relative_to(source) or source.is_relative_to(root.resolve()):
        raise ValueError("source and study directories must be separate")
    if not 1 <= len(task_ids) <= 6 or len(task_ids) != len(set(task_ids)):
        raise ValueError("one to six distinct frozen source candidates required")
    with run_lock(root):
        if (root / "prepared.json").is_file():
            saved = read_prepared(config)
            if saved["task_ids"] != list(task_ids) or saved["source_run"] != str(source):
                raise ValueError("frozen authoring membership/source changed")
            return saved
        source_manifest = read_json(source / "input_manifest.json")
        reference = load_factory_config(source_manifest["config"]["_config_path"])
        source_report, old = _terminal(reference)
        check_profiles(config, reference)
        if content_hash(Path(config["_paths"]["active_registry"]).read_bytes()) != content_hash((source / "protected_state/active_registry.json").read_bytes()):
            raise ValueError("editorial bank context must match the source snapshot")
        frozen = freeze_inputs(config)
        pool = DedicatedCallPool(root / "worker_pool.json")
        runtime = FactoryRuntime(config, frozen, pool)
        assignments = read_json(source / "downstream/palette_assignments.dynamic.json")["assignments"]
        portfolios = [spec["values"]["PORTFOLIO_DIGEST"]
                      for path in sorted((source / "downstream/reviews").glob("*/prepared.json"))
                      for group in read_json(path)["board_groups"] for spec in group["calls"]]
        if not portfolios or len({content_hash(value) for value in portfolios}) != 1:
            raise ValueError("need one reproducibly frozen source board portfolio context")
        rows = []
        for task_id in task_ids:
            record = old.state["semantic_records"][task_id]
            request = planning_request(record)
            prompt, version, digest, stage = runtime.prompts.render(request["prompt_name"], request["values"], request["composed_with"])
            calls = [call for call in old.state["call_ledger"].values() if call["purpose"] == "planning"
                     and call["call_id"] == request["call_id"] and call["stage"] == stage and call["status"] == "complete"]
            if len(calls) != 1:
                raise ValueError("missing unique successful frozen planning reference")
            old_dir = source / "downstream/model_calls" / stage / request["call_id"]
            old_request = read_json(old_dir / "request.json")
            if (prompt != (old_dir / "prompt.md").read_text(encoding="utf-8")
                    or digest != old_request["prompt_template_hash"] or version != old_request["prompt_template_version"]):
                raise ValueError("planner prompt or composition changed beyond effort")
            if runtime.context_guard.inspect(prompt, "planning")["requires_context_decision"]:
                raise ValueError("planning packet exceeds context budget")
            rows.append({"task_id": task_id, "initial": initial_record(record), "reference": deepcopy(record),
                         "reference_planning_call": deepcopy(calls[0]), "palette_assignment": assignments[task_id],
                         "prompt_hash": content_hash(prompt), "template_hash": digest, "stage": stage,
                         "schema_hash": content_hash(runtime.bindings[stage].schema)})
            atomic_write_text(root / "inputs" / (task_id + ".md"), prompt)
        pool.finish()
        saved = {"schema_version": 1, "kind": "planning_high", "task_ids": list(task_ids), "source_run": str(source),
                 "source_report_hash": content_hash(source_report), "source_state_hash": content_hash(old.state),
                 "frozen_input_hash": content_hash(frozen), "assignments": rows,
                 "portfolio_digest": portfolios[0],
                 "authoring_candidate_cap": 6, "automatic_commit": False, "automatic_promotion": False}
        saved["content_hash"] = content_hash(saved)
        atomic_write_json(root / "prepared.json", saved)
        return saved


def read_prepared(config, *, live=True):
    root = Path(config["_output_directory"])
    saved, frozen = read_json(root / "prepared.json"), read_json(root / "input_manifest.json")
    if (saved["content_hash"] != content_hash({key: value for key, value in saved.items() if key != "content_hash"})
            or saved["frozen_input_hash"] != content_hash(frozen)
            or frozen["config_identity"] != content_hash(resolved_config_manifest(config))):
        raise ValueError("frozen authoring inputs/profile changed")
    verify_snapshot(root / "frozen_source", verify_sources=live)
    verify_snapshot(root / "protected_state", verify_sources=live)
    for row in saved["assignments"]:
        if content_hash((root / "inputs" / (row["task_id"] + ".md")).read_bytes()) != row["prompt_hash"]:
            raise ValueError("frozen planner prompt changed")
    return saved


def verify_terminal(config, runtime, report):
    root = Path(config["_output_directory"])
    read_prepared(config)
    records = runtime.state["semantic_records"]
    for artifact in report["artifacts"]["rows"]:
        verify_terminal_artifact_bundle(config, artifact, records[artifact["task_id"]])
    if set(records) != {row["task_id"] for row in report["artifacts"]["rows"]}:
        raise ValueError("candidate artifact membership differs")
    verify_render_artifacts(config, report["renders"], runtime)
    pool = read_json(root / "worker_pool.json")
    queue = read_json(runtime.run_dir / "reviews/queue.json")
    if (pool["active_calls"] or any(pool["waiting_by_category"].values())
            or pool["peak_active"] > 12 or pool["peak_downstream"] > 8 or queue["waiting"]
            or any(not row["status"].startswith("complete") for row in queue["batches"])):
        raise ValueError("nonterminal pool or review queue")
    if any(call["status"] not in {"complete", "failed"} or any(row["outcome"] == "started" for row in call["attempts"])
           for call in runtime.invoker.value["calls"].values()):
        raise ValueError("nonterminal model attempt")
    if any(evidence_is_clear(row) and row.get("final_episode_evidence", {}).get("status") not in {"evidence_clear", "evidence_clear_after_review"} for row in records.values()):
        raise ValueError("clear outcome lacks final-version evidence")
    return {"verified": True, "candidate_bundles": len(records), "no_active_calls_or_review_entries": True}


def run(config, *, authorize_model_calls=False):
    root = Path(config["_output_directory"])
    with run_lock(root):
        saved = read_prepared(config, live=not (root / "complete.json").is_file())
        if (root / "complete.json").is_file():
            report = read_json(root / "report.json")
            if content_hash(report) != read_json(root / "complete.json")["report_hash"]:
                raise ValueError("completed report changed")
            return report
        if not config["run"]["mock"] and not authorize_model_calls:
            raise ValueError("live authoring study requires --authorize-model-calls")
        pool = DedicatedCallPool(root / "worker_pool.json")
        runtime = FactoryRuntime(config, read_json(root / "input_manifest.json"), pool)
        if not runtime.state["semantic_records"]:
            runtime.state["semantic_records"] = {row["task_id"]: deepcopy(row["initial"]) for row in saved["assignments"]}
            runtime.save()
        if set(runtime.state["semantic_records"]) != set(saved["task_ids"]):
            raise ValueError("live candidate membership changed")
        recover_terminal_evidence_metadata(runtime)
        sessions = read_json(root / "sessions.json") if (root / "sessions.json").is_file() else []
        session = {"started_at": time.time(), "status": "running"}
        sessions.append(session)
        atomic_write_json(root / "sessions.json", sessions)
        reviews = FactoryReviews(runtime, portfolio_digest=saved["portfolio_digest"])
        try:
            slates = defaultdict(list)
            for row in saved["assignments"]:
                slates[row["initial"]["slate_index"]].append(row["task_id"])
            palettes = {row["task_id"]: row["palette_assignment"] for row in saved["assignments"]}
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(run_slate, config, runtime, slate=slate, task_ids=ids, assignments=palettes)
                           for slate, ids in sorted(slates.items())]
                for future in futures:
                    future.result()
            for key in sorted(runtime.state["semantic_records"]):
                apply_final_evidence(runtime, key)
            while not reviews.tick(final=True):
                time.sleep(0.25)
            pool.finish()
            attach_curriculum_statuses(runtime)
            records = runtime.state["semantic_records"]
            report = {"status": "prepared", "kind": "planning_high", "prepared_hash": saved["content_hash"],
                      "counts": {"selected": len(records), "realized": sum(bool(row.get("complete_task")) for row in records.values()),
                                 "technical_clear": sum(strict_technical_clear(row) for row in records.values()),
                                 "evidence_clear": sum(evidence_is_clear(row) for row in records.values()),
                                 "core": sum(row.get("intrinsic_quality") == "core_quality" for row in records.values()),
                                 "reserve": sum(row.get("intrinsic_quality") == "reserve_quality" for row in records.values()),
                                 "basic_valid": sum(row.get("curriculum_status") == "basic_valid" for row in records.values())},
                      "statuses": dict(Counter(row["status"] for row in records.values())),
                      "descriptive_metrics": cohort_metrics(records),
                      "artifacts": materialize_terminal_artifacts(config, runtime), "renders": render_tasks(config, runtime),
                      "tasks_committed": 0, "production_promoted": False,
                      "recommendation": "inconclusive", "independent_review": "pending",
                      "limitations": ["Six deliberately selected kernels; not a representative yield estimate.",
                                      "Saved controls are noncontemporaneous. Reduced original-slate budget competition and different board peers are confounds."]}
            accounting = call_accounting([runtime])
            atomic_write_json(root / "call_timing_ledger.json", accounting)
            report["calls"] = {key: value for key, value in accounting.items() if key not in {"calls", "attempts"}}
            session.update(status="complete", ended_at=time.time())
            atomic_write_json(root / "sessions.json", sessions)
            report["wall_sessions"] = sessions
            atomic_write_json(root / "prepared_report.json", report)
            report["terminal_verification"] = verify_terminal(config, runtime, report)
            report["status"] = "complete"
            atomic_write_json(root / "report.json", report)
            atomic_write_text(root / "report.md", "# Planning-high pilot\n\nCompleted frozen-candidate path. Independent interpretation pending.\n\n" + json.dumps(report["counts"], indent=2) + "\n\nNo task commitment or production promotion.\n")
            atomic_write_json(root / "complete.json", {"report_hash": content_hash(report), "verified_at": time.time()})
            return report
        except BaseException as exc:
            session.update(status="incomplete", ended_at=time.time(), error={"type": type(exc).__name__, "message": str(exc)[:2000]})
            atomic_write_json(root / "sessions.json", sessions)
            raise
        finally:
            reviews.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--authorize-model-calls", action="store_true")
    args = parser.parse_args(argv)
    config = load_factory_config(args.config)
    if args.command == "prepare":
        if not args.source:
            parser.error("prepare requires --source")
        value = prepare(config, args.source)
        print(json.dumps({"status": "prepared", "task_ids": value["task_ids"]}))
    else:
        value = run(config, authorize_model_calls=args.authorize_model_calls)
        print(json.dumps({key: value[key] for key in ("status", "counts", "calls")}, indent=2))


if __name__ == "__main__":
    main()
