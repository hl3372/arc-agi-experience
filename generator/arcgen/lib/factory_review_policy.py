"""One actual board's grade and evidence binding, independent of publication consent."""
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import re

from .factory_policy import review_policy, validate_review_policy
from .util import content_hash, read_json, load_prompt, render_prompt

EVIDENCE_FIELDS = (
    "complete_task", "concrete_plan", "realization", "answer_evidence",
    "technical_results", "blind_results", "ambiguity_adjudication",
    "final_episode_evidence", "deterministic_diagnostics", "plan_fidelity", "repair_count",
)


def evidence_hash(record):
    return content_hash({key: record.get(key) for key in EVIDENCE_FIELDS})


def one_board_grade(judgment):
    if (judgment["novel_concept"] == "weak" or judgment["functional_composition"] == "thin"
            or judgment["overall_quality"] == "below_real_bar"
            or judgment["intrinsic_recommendation"] == "rejected_quality"):
        return "rejected_quality"
    if ((judgment["novel_concept"] == "standout" or judgment["functional_composition"] == "standout")
            and judgment["overall_quality"] in ("real_quality", "exceptional")
            and judgment["intrinsic_recommendation"] == "core_quality"):
        return "core_quality"
    return "reserve_quality"


def single_board_update(judgment, task_id):
    grade = one_board_grade(judgment)
    if judgment["same_minimal_procedure"] or judgment["portfolio_recommendation"] == "same_procedure_duplicate":
        placement = "same_procedure_duplicate"
    elif judgment["addition_is_excessive"] or judgment["portfolio_recommendation"] == "hold_crowded":
        placement = "hold_crowded"
    elif grade == "rejected_quality":
        placement = "not_applicable"
    else:
        placement = "include_now"
    return {"editorial_judgments": [{**deepcopy(judgment), "task_id": task_id}],
            "intrinsic_quality": grade, "portfolio_placement": placement,
            "editorial_reason": "Preselected actual board 1; retained individual quality floors and placement rules."}


def board_binding(policy, batch, group, result, task_id):
    validate_review_policy(policy)
    spec, = group["calls"]
    if spec["board_index"] != 1 or result["board_index"] != 1 or result["mapping"] != spec["mapping"]:
        raise ValueError("required preselected board identity changed")
    alias = next(key for key, source in spec["mapping"].items() if source == "generated:" + task_id)
    judgment = next(row for row in result["response"]["judgments"] if row["task_id"] == alias)
    update = single_board_update(judgment, task_id)
    binding = {
        "policy": deepcopy(policy), "batch_id": batch["id"], "group_index": group["group_index"],
        "call_id": spec["call_id"], "alias": alias, "spec_hash": content_hash(spec),
        "episode_hash": content_hash(batch["records"][task_id]["complete_task"]),
        "evidence_hash": evidence_hash(batch["records"][task_id]),
        "response_hash": content_hash(result["response"]),
        "judgments_hash": content_hash(update["editorial_judgments"]),
    }
    binding["binding_hash"] = content_hash(binding)
    return {**update, "editorial_review_binding": binding}


def required_boards(record, policy=None):
    """Default callers retain two boards. A record cannot choose its own policy."""
    validate_review_policy(policy)
    rows = record.get("editorial_judgments")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        return False
    if policy is None:
        return len(rows) == 2
    binding = record.get("editorial_review_binding") or {}
    return (len(rows) == 1 and binding.get("policy") == policy
            and binding.get("binding_hash") == content_hash({k: v for k, v in binding.items() if k != "binding_hash"})
            and binding.get("episode_hash") == content_hash(record.get("complete_task"))
            and binding.get("evidence_hash") == evidence_hash(record)
            and binding.get("judgments_hash") == content_hash(rows))


def verify_recorded_review(config, runtime, record):
    """Pure file reader: validate the actual saved call, package and frozen policy.

    Called by terminal verification and again at every publication handoff. No
    runtime construction, writes, locks, provider retries or historical relabeling.
    """
    policy = review_policy(config)
    if policy is None:
        return None
    if not required_boards(record, policy):
        raise ValueError("required actual board or its episode/evidence binding is missing or changed")
    root = Path(config["_output_directory"])
    manifest = read_json(root / "input_manifest.json")
    frozen = manifest.get("config") or {}
    if (review_policy(frozen) != policy or manifest.get("config_identity") != content_hash(frozen)
            or frozen.get("_run_identity") != config.get("_run_identity")
            or frozen.get("role_profile") != config.get("role_profile")):
        raise ValueError("review policy was not frozen for this run")
    runtime_manifest = read_json(root / "downstream/runtime_manifest.json")
    if runtime_manifest["factory_config"] != frozen:
        raise ValueError("review runtime differs from the frozen run")
    binding = record["editorial_review_binding"]
    for field in ("batch_id", "call_id"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", binding[field]):
            raise ValueError("unsafe review artifact identity")
    directory = root / "downstream/reviews" / binding["batch_id"]
    batch = read_json(directory / "prepared.json")
    if (batch.get("review_policy") != policy
            or batch.get("prepared_hash") != content_hash({k: v for k, v in batch.items() if k != "prepared_hash"})):
        raise ValueError("frozen review package or policy changed")
    group, = [group for group in batch["board_groups"] if group["group_index"] == binding["group_index"]]
    spec, = group["calls"]
    task_id = record["task_id"]
    if (spec["board_index"] != 1 or spec["call_id"] != binding["call_id"]
            or content_hash(spec) != binding["spec_hash"]
            or task_id not in group["task_ids"]
            or evidence_hash(batch["records"][task_id]) != evidence_hash(record)):
        raise ValueError("review group or final evidence changed")
    saved = read_json(directory / "report.json")
    if spec["call_id"] in saved["failures"] or not saved["status"].startswith("complete"):
        raise ValueError("required board failed or did not finish")
    result = saved["results"][spec["call_id"]]
    expected = board_binding(policy, batch, group, result, task_id)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("record does not match its actual board disposition")
    calls = runtime.state.get("call_ledger", {})
    calls = calls.values() if isinstance(calls, Mapping) else calls
    call, = [row for row in calls if row.get("stage") == policy["stage"] and row.get("call_id") == spec["call_id"]]
    if (call.get("status") != "complete" or call.get("purpose") != "editorial_board"
            or call.get("model") != policy["model"] or call.get("reasoning_effort") != policy["reasoning_effort"]
            or call.get("account") != frozen["run"]["account"]
            or call.get("parent_ids") != spec["parent_ids"]
            or call.get("candidate_episode_hashes", {}).get(task_id) != binding["episode_hash"]
            or call.get("response_content_hash") != binding["response_hash"]):
        raise ValueError("required board call identity or episode changed")
    logs = root / "downstream/model_calls" / policy["stage"] / spec["call_id"]
    request, response = read_json(logs / "request.json"), read_json(logs / "parsed_response.json")
    marker = read_json(logs / "result.json")
    raw = (logs / "raw_response.txt").read_text()
    prompt = (logs / "prompt.md").read_text()
    from .v4_schemas import parse_stage_output as parse_v4_stage, OUTPUT_SCHEMAS
    # The unchanged prompt is taken from the frozen source, not today's file.
    template, version, digest = load_prompt(root / "frozen_source/prompts/v4_editorial_board.md")
    schema = OUTPUT_SCHEMAS[policy["stage"]]
    if (response != result["response"] or parse_v4_stage(policy["stage"], raw) != response
            or response.get("board_id") != spec["call_id"]
            or marker.get("parsed_response_hash") != content_hash(response)
            or marker.get("raw_response_hash") != content_hash(raw)
            or prompt != render_prompt(template, spec["values"])
            or request.get("prompt_hash") != content_hash(prompt)
            or request.get("prompt_template_hash") != digest or request.get("prompt_template_version") != version
            or request.get("schema_hash") != content_hash(schema)
            or request.get("parent_ids") != spec["parent_ids"]
            or request.get("model", {}).get("name") != policy["model"]
            or request.get("model", {}).get("reasoning_effort") != policy["reasoning_effort"]):
        raise ValueError("required board response, prompt or schema changed")
    return {"policy": policy, "call_id": spec["call_id"], "response_hash": binding["response_hash"]}


def verify_dossier_calls(config, runtime, record, destination):
    """Require every successful relevant call, including conditional/repair history.

    Seven is the reduced minimum, but a directory count cannot substitute for
    the required purposes or for copying the actual frozen calls.
    """
    root = Path(config["_output_directory"])
    required = []
    for namespace, parent in (("upstream", record["source_candidate_id"]), ("downstream", record["task_id"])):
        ledger = read_json(root / namespace / "state.json")["call_ledger"]
        required.extend(row for row in ledger.values() if row.get("status") == "complete" and (
            parent in row.get("parent_ids", []) or (namespace == "upstream"
            and row.get("call_id") == record["private_source_call_id"] and row.get("stage") == "e16_idea_search")))
    design = read_json(root / "frozen_supply_design.private.json")
    spec, = [spec for spec in design["specs"] if spec["call_id"] == record["private_source_call_id"]]
    if record["source_candidate_id"] not in spec["candidate_ids"]:
        raise ValueError("candidate's idea provenance differs from frozen supply")
    purposes = {row["purpose"] for row in required}
    if (not {"full_window_selection", "planning", "realization", "technical_verification", "primary_blind", "editorial_board"} <= purposes
            or len(purposes & {"idea_conditioned", "idea_unconditioned", "idea_logical_leap", "idea_episode_joint"}) != 1):
        raise ValueError("reduced candidate lacks required successful stage evidence")
    copied = {}
    for path in (Path(destination) / "model_calls").rglob("request.json"):
        request = read_json(path)
        copied[(request.get("stage"), request.get("call_id"))] = path.parent
    for call in required:
        directory = copied.get((call["stage"], call["call_id"]))
        if directory is None or content_hash(read_json(directory / "parsed_response.json")) != call["response_content_hash"]:
            raise ValueError("candidate dossier omits or changes an actual successful call")
    return len(required)
