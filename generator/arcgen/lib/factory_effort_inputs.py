"""Freeze bounded role-effort panels from authentic, completed saved calls.

This module never renders a prompt, constructs a model runtime, or reads the
live task bank. Explicit packet ingestion and the small inventory resolver use
the same preflight. A failed historical verifier attempt remains a valid fixed
input; it is not silently dropped from the preregistered safety panel. Reference
answers stay private and are not output-equality requirements or ground truth.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from .factory_payloads import technical_public_plan
from .factory_runtime import build_stage_bindings
from .util import atomic_write_json, atomic_write_text, content_hash, read_json


SCREEN = {
    "idea_conditioned": (6, 1, 1), "idea_unconditioned": (6, 1, 1),
    "idea_logical_leap": (3, 1, 1), "idea_episode_joint": (3, 1, 1),
    "shard_selection": (8, 1, 2), "global_selection": (8, 1, 2),
    "planning": (8, 1, 2), "realization": (8, 1, 2),
    "technical_verification": (24, 1, 2), "primary_blind": (24, 1, 2),
    "conditional_blind": (8, 1, 2), "ambiguity_adjudication": (8, 1, 2),
    "targeted_repair": (8, 1, 2), "editorial_board": (4, 2, 2),
    "logical_leap_audit": (4, 1, 2), "structural_transfer_audit": (4, 1, 2),
    "mechanism_audit": (4, 1, 2),
}
ACTIVE_PURPOSES = tuple(SCREEN)
MANIFEST_NAME = "manifest.private.json"
INVENTORY = "experiments/ASTRA_ROLE_EFFORT_INPUT_INVENTORY_20260905.json"
RARE = {"conditional_blind", "ambiguity_adjudication", "targeted_repair"}
AUDITS = {"logical_leap_audit", "structural_transfer_audit", "mechanism_audit"}
ORDINARY_MODES = {"open", "concept_led", "composition_led"}
IDENTITY_FIELDS = {
    "search_call_id", "search_mode", "selector_id", "candidate_id", "task_id",
    "solver_id", "verifier_id", "board_id", "audit_id", "repair_type",
    "requested_properties", "non_gating_confirmation",
}


class FactoryEffortInputsError(RuntimeError):
    def __init__(self, message: str, *, missing_coverage: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.missing_coverage = dict(missing_coverage or {})


def _read(path: Path) -> Any:
    try:
        return read_json(path)
    except ValueError as exc:
        raise FactoryEffortInputsError(str(exc)) from exc


def _inside(root: Path, name: str) -> Path:
    if Path(name).is_absolute():
        raise FactoryEffortInputsError(f"artifact path must be relative: {name}")
    path = (root / name).resolve()
    if path == root.resolve() or not path.is_relative_to(root.resolve()):
        raise FactoryEffortInputsError(f"unsafe artifact path: {name}")
    return path


def _bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FactoryEffortInputsError(f"missing saved call artifact: {path}") from exc


def _requested(purposes: Sequence[str] | None) -> list[str]:
    requested = list(ACTIVE_PURPOSES if purposes is None else purposes)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(ACTIVE_PURPOSES):
        raise FactoryEffortInputsError("choose distinct active substantive purposes")
    return [purpose for purpose in ACTIVE_PURPOSES if purpose in requested]


def _inventory(config: Mapping[str, Any], path: Path | None) -> tuple[Path, dict[str, Any]]:
    path = Path(path or Path(config["_repository_root"]) / INVENTORY).resolve()
    inventory = _read(path)
    found = {row["purpose"]: (row["matched_inputs"], row["calls_per_input_per_effort"],
                              row["independent_repeat_inputs"]) for row in inventory["roles"]}
    if found != SCREEN or any(row["screen_efforts"] != ["high", "xhigh", "max"] for row in inventory["roles"]):
        raise FactoryEffortInputsError("inventory differs from the approved bounded screen")
    return path, inventory


def _json_after(prompt: str, marker: str) -> Any:
    if prompt.count(marker) != 1:
        raise FactoryEffortInputsError(f"cannot establish saved payload boundary: {marker.strip()}")
    try:
        return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0]
    except json.JSONDecodeError as exc:
        raise FactoryEffortInputsError(f"invalid saved payload after {marker.strip()}") from exc


def _public_boundary(prompt: str, purpose: str) -> dict[str, Any]:
    if purpose not in {"technical_verification", "primary_blind", "conditional_blind"}:
        return {}
    marker = "Solver-visible episode:\n" if purpose == "technical_verification" else "Solver-facing ARC episode:\n"
    task = _json_after(prompt, marker)
    if (not isinstance(task, dict) or set(task) != {"train", "test"}
            or not task["train"] or not task["test"]
            or any(set(pair) != {"input", "output"} for pair in task["train"])
            or any(set(pair) != {"input"} for pair in task["test"])):
        raise FactoryEffortInputsError("saved verifier/solver payload is not public-only")
    metadata = {"public_task_hash": content_hash(task), "public_only_boundary_verified": True}
    if purpose == "technical_verification":
        plan = _json_after(prompt, "Frozen concrete storyboard:\n")
        if technical_public_plan(plan) != plan:
            raise FactoryEffortInputsError("technical packet contains an unprojected answer-bearing plan")
        metadata["public_plan_hash"] = content_hash(plan)
    return metadata


def _purpose(ledger: Mapping[str, Any], bindings: Mapping[str, Any]) -> str | None:
    stage, call_id = ledger["stage"], ledger["call_id"]
    if ledger.get("purpose"):
        return str(ledger["purpose"])
    if stage == "v4_blind_solver":
        if call_id.endswith("-blind-primary"):
            return "primary_blind"
        if call_id.endswith(("-blind-second", "-blind-after-ambiguity-repair")):
            return "conditional_blind"
        return None
    binding = bindings.get(stage)
    return binding.purposes[0] if binding and len(binding.purposes) == 1 else None


def _board_mapping(mapping: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result = {}
    for opaque, value in mapping.items():
        if isinstance(value, str) and ":" in value:
            kind, private_id = value.split(":", 1)
        elif isinstance(value, Mapping):
            kind, private_id = value.get("kind"), value.get("private_id")
        else:
            raise FactoryEffortInputsError("unsupported saved board mapping")
        if kind not in {"anchor", "generated", "case"} or not private_id:
            raise FactoryEffortInputsError("incomplete saved board mapping")
        result[opaque] = {"kind": "anchor" if kind == "anchor" else "case", "private_id": private_id}
    return result


def _historical_assignment(packet: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(packet["calibration_root"]).resolve()
    matches = [row for row in _read(root / "assignment_ledger.private.json")["assignments"]
               if row["assignment_id"] == packet["assignment_id"]]
    if len(matches) != 1:
        raise FactoryEffortInputsError("missing authoritative historical assignment")
    assignment = matches[0]
    status = _read(root / "call_ledger.private.json")["calls"].get(assignment["assignment_id"], {}).get("status")
    if (assignment.get("phase") != "panel" or status not in {"complete", "failed", "timeout", "timed_out"}
            or any(assignment[key] != packet[key] for key in ("purpose", "stage", "call_id", "expected", "account"))
            or _inside(root, assignment["namespace"]) != Path(packet["run_directory"]).resolve()):
        raise FactoryEffortInputsError("historical packet differs from its terminal saved assignment")
    if (assignment["purpose"] == "editorial_board"
            and _read(_inside(root, assignment["mapping_path"])) != packet.get("board_mapping")):
        raise FactoryEffortInputsError("historical board mapping differs from its exact saved assignment")
    return {**assignment, "_terminal_status": status}


def _load_packet(packet: Mapping[str, Any], bindings: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = Path(packet["run_directory"]).resolve()
    stage, call_id, purpose = (str(packet[key]) for key in ("stage", "call_id", "purpose"))
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in (stage, call_id)):
        raise FactoryEffortInputsError("unsafe stage/call identifier")
    binding = bindings.get(stage)
    if binding is None or purpose not in ACTIVE_PURPOSES or purpose not in binding.purposes:
        raise FactoryEffortInputsError("saved packet purpose/stage is not an active factory binding")
    state = _read(root / "state.json")
    entries = [row for row in state.get("call_ledger", {}).values()
               if row.get("stage") == stage and row.get("call_id") == call_id]
    assignment = _historical_assignment(packet) if packet.get("source_kind") == "historical_calibration" else None
    failed_historical = (assignment is not None and purpose in {"technical_verification", "primary_blind", "editorial_board"}
                         and len(entries) == 1 and entries[0].get("status") in {"failed", "timeout", "timed_out"})
    if len(entries) != 1 or (entries[0].get("status") != "complete" and not failed_historical):
        raise FactoryEffortInputsError(f"saved call is not uniquely complete: {root}/{stage}/{call_id}")
    ledger = entries[0]
    if assignment is not None and assignment["_terminal_status"] != ledger["status"]:
        raise FactoryEffortInputsError("historical terminal status disagrees across saved ledgers")
    resolved = _purpose(ledger, bindings)
    if resolved is not None and resolved != purpose:
        raise FactoryEffortInputsError("packet purpose contradicts the actual saved call purpose")
    if resolved is None and packet.get("source_kind") != "historical_calibration":
        raise FactoryEffortInputsError("ambiguous saved call requires authentic assignment provenance")
    directory = _inside(root, f"model_calls/{stage}/{call_id}")
    names = ["prompt.md", "request.json"]
    if not failed_historical:
        names += ["parsed_response.json", "result.json", "raw_response.txt"]
    files = {name: _bytes(_inside(directory, name)) for name in names}
    if failed_historical:
        for path in sorted(directory.iterdir()):
            if path.is_file() and (path.name in {"parse_failures.json", "raw_response.txt", "parsed_response.json", "result.json"}
                                   or re.fullmatch(r"attempt-\d+\.(raw_response\.txt|failure\.log|timing\.json)", path.name)):
                files["failed_attempt_" + path.name] = _bytes(path)
    files["schema.json"] = _bytes(_inside(root, f"schemas/{stage}.schema.json"))
    request, schema = (json.loads(files[name]) for name in ("request.json", "schema.json"))
    response = {} if failed_historical else json.loads(files["parsed_response.json"])
    result = {} if failed_historical else json.loads(files["result.json"])
    if (request.get("stage") != stage or request.get("call_id") != call_id
            or request.get("prompt_hash") != content_hash(files["prompt.md"])
            or request.get("schema_hash") != content_hash(schema)
            or schema != binding.schema):
        raise FactoryEffortInputsError("saved prompt/request/schema contract does not match")
    if not failed_historical and (result.get("parsed_response_hash") != content_hash(response)
            or result.get("raw_response_hash") != content_hash(files["raw_response.txt"])
            or result.get("cache_key") != request.get("cache_key")
            or ledger.get("response_content_hash", content_hash(response)) != content_hash(response)):
        raise FactoryEffortInputsError("saved successful response evidence does not match")
    if not failed_historical:
        try:
            binding.parser(deepcopy(response))
        except Exception as exc:
            raise FactoryEffortInputsError(f"saved response no longer satisfies its parser: {call_id}: {exc}") from exc
    model = request["model"]
    if ledger.get("model") != model["name"] or ledger.get("reasoning_effort") != model["reasoning_effort"]:
        raise FactoryEffortInputsError("saved model/effort provenance disagrees")
    if assignment is not None and (model["name"] != assignment["model"] or model["reasoning_effort"] != assignment["effort"]
                                   or request["parent_ids"] != assignment["parent_ids"]):
        raise FactoryEffortInputsError("historical request differs from its exact prepared assignment")
    for key, name in (("prompt_sha256", "prompt.md"), ("request_sha256", "request.json"),
                      ("parsed_response_sha256", "parsed_response.json")):
        if packet.get(key) and packet[key] != content_hash(files[name]):
            raise FactoryEffortInputsError(f"preregistered inventory artifact changed: {call_id}/{name}")
    prompt = files["prompt.md"].decode("utf-8")
    boundary = _public_boundary(prompt, purpose)
    if assignment is not None and purpose in {"technical_verification", "primary_blind"}:
        frozen = _read(_inside(Path(packet["calibration_root"]), assignment["packet_path"]))
        if content_hash(frozen["PUBLIC_TASK"]) != boundary["public_task_hash"]:
            raise FactoryEffortInputsError("historical prompt changed its prepared public episode")
        if purpose == "technical_verification" and content_hash(frozen["CONCRETE_PLAN"]) != boundary["public_plan_hash"]:
            raise FactoryEffortInputsError("historical prompt changed its prepared public plan")
    expected = deepcopy(assignment["expected"]) if failed_historical else {
        key: deepcopy(value) for key, value in response.items() if key in IDENTITY_FIELDS}
    if any(expected.get(key) != value for key, value in packet.get("expected", {}).items()):
        raise FactoryEffortInputsError("saved output identity disagrees with its authentic input packet")
    membership = None
    for field, id_field in (("ideas", "candidate_id"), ("candidate_assessments", "candidate_id"), ("judgments", "task_id")):
        if field in response:
            ids = [row[id_field] for row in response[field]]
            if len(ids) != len(set(ids)):
                raise FactoryEffortInputsError("saved response repeats an input member")
            membership = {"field": field, "id_field": id_field, "ids": ids}
    if purpose == "editorial_board":
        episodes = _json_after(prompt, "Blinded episodes and concise rules:\n")
        ids = [item["task_id"] for item in episodes]
        if assignment is not None:
            original = _read(_inside(Path(packet["calibration_root"]), assignment["packet_path"]))
            if original["BLINDED_TASKS"] != episodes:
                raise FactoryEffortInputsError("historical board prompt changed its prepared opaque episode set")
        if failed_historical:
            membership = {"field": "judgments", "id_field": "task_id", "ids": ids}
        if membership is None or set(ids) != set(membership["ids"]):
            raise FactoryEffortInputsError("board response does not cover its exact opaque episode set")
        mapping = _board_mapping(packet.get("board_mapping", {}))
        anchors = [row["private_id"] for row in mapping.values() if row["kind"] == "anchor"]
        if set(mapping) != set(ids) or len(anchors) != 6 or len(set(anchors)) != 6:
            raise FactoryEffortInputsError("board packet requires its authentic mapping and all six fixed anchors")
        if any(not item["episode"].get("test") or any(set(pair) != {"input", "output"}
               for pair in item["episode"]["train"] + item["episode"]["test"]) for item in episodes):
            raise FactoryEffortInputsError("board packet must preserve complete episodes including authored test outputs")
        comparative = {mapping[item["task_id"]]["kind"] + ":" + mapping[item["task_id"]]["private_id"]:
                       content_hash(item["episode"]) for item in episodes}
    if purpose.startswith("idea_") and (membership is None or len(membership["ids"]) != 4):
        raise FactoryEffortInputsError("generator effort input must retain the four-kernel contract")
    if packet.get("member_ids") is not None and (membership is None or set(packet["member_ids"]) != set(membership["ids"])):
        raise FactoryEffortInputsError("saved response membership differs from its frozen source packet")
    source = {"run_directory": str(root), "call_id": call_id, "stage": stage,
              "model": model["name"], "effort": model["reasoning_effort"],
              "account": ledger.get("account", packet.get("account")),
              "account_provenance": "saved_call_ledger" if ledger.get("account") else
                                    "saved_assignment" if packet.get("account") else "not_recorded",
              "source_kind": packet.get("source_kind", "explicit_completed_call"),
              "request_path": str(directory / "request.json"), "request_sha256": content_hash(files["request.json"]),
              "response_path": None if failed_historical else str(directory / "parsed_response.json"),
              "response_sha256": None if failed_historical else content_hash(files["parsed_response.json"]),
              "request_model_settings": model, "ledger_status": ledger["status"], "source_status": ledger["status"],
              "prior_attempt_succeeded": not failed_historical,
              "source_call_ledger_entry": deepcopy(ledger), **boundary}
    for key in ("source_alias", "candidate_episode_hash", "candidate_episode_hashes", "evidence_version",
                "group_episode_hash", "private_case_id", "search_mode", "private_conditioned", "route"):
        if key in ledger or key in packet:
            source[key] = deepcopy(ledger.get(key, packet.get(key)))
    identity = {"purpose": purpose, "prompt_sha256": content_hash(files["prompt.md"]), "schema_hash": content_hash(schema)}
    input_id = purpose + "-" + content_hash(identity)[:20]
    row = {"input_id": input_id, "purpose": purpose, "stage": stage, "call_id": call_id,
           "prompt_template_version": request["prompt_template_version"],
           "prompt_template_hash": request["prompt_template_hash"],
           "parent_ids": request["parent_ids"], "schema_hash": content_hash(schema),
           "expected": expected, "category": packet.get("category", stage), "source": source,
           "repeat_selected": bool(packet.get("repeat_selected", False)), **boundary}
    if membership is not None:
        row["response_membership"] = membership
    for key in ("group_id", "board_index", "stratum", "batch_ordinal", "board_mapping", "episode_hash", "evidence_version"):
        if key in packet:
            row[key] = deepcopy(packet[key])
    if purpose == "editorial_board":
        row["board_mapping"] = mapping
        row["comparative_set_hash"] = content_hash(comparative)
        row["anchor_set_hash"] = content_hash({key: value for key, value in comparative.items() if key.startswith("anchor:")})
    if purpose in AUDITS and membership is not None:
        row["batch_size"] = len(membership["ids"])
    if purpose.startswith("idea_"):
        row["stratum"] = response["search_mode"]
    reference = packet.get("reference")
    if reference is not None:
        files["reference.private.json"] = (json.dumps(reference, ensure_ascii=False, indent=2) + "\n").encode()
    # Save a stable, bounded ledger entry, not the changing whole run state.
    files["source_ledger.private.json"] = (json.dumps(ledger, ensure_ascii=False, indent=2) + "\n").encode()
    return row, files


def _coverage(rows: Sequence[Mapping[str, Any]], requested: Sequence[str]) -> dict[str, Any]:
    coverage = {}
    for purpose, (units, calls, repeats) in SCREEN.items():
        selected = [row for row in rows if row["purpose"] == purpose]
        groups = defaultdict(list)
        for row in selected:
            groups[row.get("group_id") or row["input_id"]].append(row)
        problems = []
        if len(groups) != units:
            problems.append(f"requires {units} independent inputs/groups; found {len(groups)}")
        if len(selected) != units * calls:
            problems.append(f"requires {units * calls} saved call packets; found {len(selected)}")
        for group in groups.values():
            if len(group) != calls or (purpose == "editorial_board" and {row.get("board_index") for row in group} != {1, 2}):
                problems.append("both independent board calls must remain paired")
            if len({bool(row["repeat_selected"]) for row in group}) != 1:
                problems.append("board pair repeat membership differs")
            if purpose == "editorial_board":
                comparatives = [sorted((item.get("kind"), item.get("private_id", item.get("id")))
                                       for item in row.get("board_mapping", {}).values()) for row in group]
                if comparatives and any(value != comparatives[0] for value in comparatives):
                    problems.append("paired boards do not contain the same private comparative set")
                if len({row.get("comparative_set_hash") for row in group}) != 1:
                    problems.append("paired boards do not contain the same complete episode versions")
        repeat_count = sum(bool(group[0]["repeat_selected"]) for group in groups.values())
        if repeat_count != repeats:
            problems.append(f"requires {repeats} preregistered repeated inputs/groups; found {repeat_count}")
        if purpose in {"idea_conditioned", "idea_unconditioned"}:
            if Counter(row.get("stratum") for row in selected) != Counter({mode: 2 for mode in ORDINARY_MODES}):
                problems.append("ordinary generator requires two exact blocks per each of three modes")
        if purpose in {"idea_logical_leap", "idea_episode_joint"} and any(row.get("stratum") != "concept_led" for row in selected):
            problems.append("specialist generator must retain concept_led mode")
        if purpose == "editorial_board":
            if len({row.get("anchor_set_hash") for row in selected}) > 1:
                problems.append("board groups do not retain the same six fixed anchor episodes")
            case_sets = [tuple(sorted(item["private_id"] for item in group[0]["board_mapping"].values()
                                      if item["kind"] != "anchor")) for group in groups.values()]
            if len(set(case_sets)) != len(case_sets):
                problems.append("reordered copies of one board comparative group are not independent inputs")
        coverage[purpose] = {"status": "not_frozen" if purpose not in requested else "missing" if problems else "complete",
                             "required_inputs": units, "available_inputs": len(groups),
                             "required_call_packets": units * calls, "frozen_call_packets": len(selected),
                             "required_repeat_inputs": repeats, "repeat_inputs": repeat_count,
                             "missing": problems if purpose in requested else [],
                             "origin_model_call_counts": dict(Counter(row["source"]["model"] for row in selected)),
                             "source_status_counts": dict(Counter(row["source"].get("source_status", "complete") for row in selected))}
    return coverage


def _safe_output(config: Mapping[str, Any], output: Path, sources: Sequence[Path]) -> None:
    protected = [Path(config["_repository_root"]).resolve()]
    protected += [Path(value).resolve() for key, value in config.get("_paths", {}).items()
                  if key not in {"output_directory", "cache"} and isinstance(value, (str, Path))]
    if output in protected:
        raise FactoryEffortInputsError("output is a protected root")
    for source in [*sources, *protected[1:]]:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise FactoryEffortInputsError("effort output overlaps a source or protected input")


def freeze_effort_inputs(config: Mapping[str, Any], packets: Sequence[Mapping[str, Any]],
                         output_directory: str | Path, *, purposes: Sequence[str] | None = None,
                         inventory_path: Path | None = None,
                         _resolver_identity: str | None = None) -> dict[str, Any]:
    """Ingest explicit complete-call descriptors; never accept rendered replacements.

    Required descriptor fields: run_directory, stage, call_id, purpose. Boards
    additionally require group_id, board_index and their original board_mapping.
    repeat_selected is preregistered per descriptor. Panels must be complete.
    """
    requested = _requested(purposes)
    inventory_file, inventory = _inventory(config, inventory_path)
    root = Path(output_directory).resolve()
    _safe_output(config, root, [Path(row["run_directory"]).resolve() for row in packets])
    identity = content_hash({"packets": [dict(row, run_directory=str(Path(row["run_directory"]).resolve())) for row in packets],
                             "purposes": requested, "inventory_hash": content_hash(_bytes(inventory_file))})
    if (root / MANIFEST_NAME).exists():
        saved = verify_effort_inputs(root)
        if saved["ingestion_identity"] != identity:
            raise FactoryEffortInputsError("effort inputs are already frozen from a different request")
        return saved
    bindings = build_stage_bindings()
    loaded = [_load_packet(packet, bindings) for packet in packets if packet["purpose"] in requested]
    rows = [row for row, _ in loaded]
    if len({row["input_id"] for row in rows}) != len(rows):
        raise FactoryEffortInputsError("duplicate exact input cannot fill an independent panel slot")
    coverage = _coverage(rows, requested)
    missing = {key: value for key, value in coverage.items() if value["status"] == "missing"}
    if missing:
        raise FactoryEffortInputsError("requested effort batch is incomplete", missing_coverage=missing)
    artifacts = {}
    for row, files in loaded:
        prefix = "packets/" + row["input_id"] + "/"
        for name, data in files.items():
            path = _inside(root, prefix + name)
            if path.exists():
                if _bytes(path) != data:
                    raise FactoryEffortInputsError("refusing to overwrite a partially frozen artifact")
            else:
                atomic_write_text(path, data.decode("utf-8"))
            artifacts[prefix + name] = content_hash(data)
        row.update(prompt_path=prefix + "prompt.md", prompt_sha256=artifacts[prefix + "prompt.md"],
                   schema_path=prefix + "schema.json", schema_sha256=artifacts[prefix + "schema.json"],
                   request_path=prefix + "request.json")
        if "parsed_response.json" in files:
            row["response_path"] = prefix + "parsed_response.json"
        if "reference.private.json" in files:
            row["reference_path"] = prefix + "reference.private.json"
    manifest = {"schema_version": 1, "inventory_id": inventory["inventory_id"],
                "inventory_sha256": content_hash(_bytes(inventory_file)), "ingestion_identity": identity,
                "resolver_identity": _resolver_identity,
                "output_directory": str(root), "required_purposes": requested, "coverage": coverage,
                "inputs": rows, "artifacts": artifacts, "provider_calls_made": 0,
                "reconstructed_prompts": False, "prior_call_credit": 0,
                "reference_labels_are_ground_truth": False,
                "reference_limitations": ["Historical categories and decisions are fallible reference evidence, not independent ground truth.",
                                          "Disputed or unknown negative coverage requires separately reviewed held-out confirmation; this freezer does not relabel or extend the historical panel."],
                "target_screen_calls": sum((SCREEN[p][0] + SCREEN[p][2]) * SCREEN[p][1] * 3 for p in requested)}
    manifest["manifest_hash"] = content_hash(manifest)
    atomic_write_json(root / MANIFEST_NAME, manifest)
    return verify_effort_inputs(root)


def verify_effort_inputs(directory: str | Path) -> dict[str, Any]:
    """Verify immutable local bytes and coverage, with no mutable source reads."""
    root = Path(directory).resolve()
    manifest = _read(root / MANIFEST_NAME)
    if manifest.get("manifest_hash") != content_hash({key: value for key, value in manifest.items() if key != "manifest_hash"}):
        raise FactoryEffortInputsError("frozen effort manifest changed")
    requested = _requested(manifest["required_purposes"])
    if (manifest.get("schema_version") != 1 or manifest.get("provider_calls_made") != 0
            or manifest.get("target_screen_calls") != sum((SCREEN[p][0] + SCREEN[p][2]) * SCREEN[p][1] * 3 for p in requested)):
        raise FactoryEffortInputsError("frozen bounded screen contract changed")
    rows = manifest["inputs"]
    if len({row["input_id"] for row in rows}) != len(rows) or any(row["purpose"] not in requested for row in rows):
        raise FactoryEffortInputsError("frozen effort input identities are invalid")
    coverage = _coverage(rows, requested)
    if coverage != manifest["coverage"] or any(coverage[p]["status"] != "complete" for p in requested):
        raise FactoryEffortInputsError("frozen effort coverage is incomplete")
    for name, digest in manifest["artifacts"].items():
        if content_hash(_bytes(_inside(root, name))) != digest:
            raise FactoryEffortInputsError(f"frozen effort artifact changed: {name}")
    bindings = build_stage_bindings()
    for row in rows:
        required_paths = [row["request_path"], row["prompt_path"], row["schema_path"]]
        required_paths += [row[key] for key in ("reference_path", "response_path") if key in row]
        if any(path not in manifest["artifacts"] for path in required_paths):
            raise FactoryEffortInputsError("untracked frozen provenance artifact")
        for kind in ("prompt", "schema"):
            if manifest["artifacts"].get(row[kind + "_path"]) != row[kind + "_sha256"]:
                raise FactoryEffortInputsError("untracked frozen input artifact")
        schema = _read(_inside(root, row["schema_path"]))
        binding = bindings.get(row["stage"])
        if not binding or row["purpose"] not in binding.purposes or schema != binding.schema or content_hash(schema) != row["schema_hash"]:
            raise FactoryEffortInputsError("frozen input schema/purpose binding changed")
        _public_boundary(_bytes(_inside(root, row["prompt_path"])).decode("utf-8"), row["purpose"])
    return manifest


def _historical_packets(root: Path | None, requested: Sequence[str], inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    if root is None:
        return []
    root = root.resolve()
    ledger = _read(root / "call_ledger.private.json")["calls"]
    panel = _read(root / "panel/manifest.private.json")
    cases = {row["case_id"]: row for row in panel["cases"]}
    repeated = inventory["preregistered_repeat_policy"]["historical_technical_and_primary_repeat_private_case_ids"]
    packets = []
    for row in _read(root / "assignment_ledger.private.json")["assignments"]:
        if (row["purpose"] not in requested or row.get("phase") != "panel" or row["model"] != "gpt-6-astra"
                or ledger.get(row["assignment_id"], {}).get("status") not in {"complete", "failed", "timeout", "timed_out"}):
            continue
        case = cases.get(row.get("case_id"), {})
        if row["purpose"] in {"technical_verification", "primary_blind"} and not case:
            raise FactoryEffortInputsError("historical assignment lacks its preregistered reference case")
        packet = {"run_directory": str(_inside(root, row["namespace"])), "stage": row["stage"],
                  "call_id": row["call_id"], "purpose": row["purpose"], "account": row["account"],
                  "source_kind": "historical_calibration", "expected": row["expected"],
                  "calibration_root": str(root), "assignment_id": row["assignment_id"],
                  "category": case.get("category", "historical_anchor_board"),
                  "private_case_id": case.get("private_case_id"),
                  "repeat_selected": case.get("private_case_id") in repeated,
                  "reference": {"case": case, "assignment": row, "packet": _read(_inside(root, row["packet_path"]))}}
        if row["purpose"] == "editorial_board":
            mapping = _read(_inside(root, row["mapping_path"]))
            packet.update(group_id="historical-" + content_hash(str(root))[:8] + f"-{row['group_index']}",
                          board_index=row["board_index"], board_mapping=mapping,
                          repeat_selected=True)
        packets.append(packet)
    return packets


def _inventory_packets(config: Mapping[str, Any], inventory: Mapping[str, Any], purpose: str) -> list[dict[str, Any]]:
    entries = inventory["source_inventory"]["authentic_fixed_input_candidates"].get(purpose, [])
    sources = inventory["source_inventory"]["sources"]
    repeats = inventory["preregistered_repeat_policy"]
    packets = []
    for index, entry in enumerate(entries, 1):
        root = Path(config["_repository_root"]) / sources[entry["source"]]["root"]
        packet = {**entry, "run_directory": str(root.resolve()), "purpose": purpose,
                  "source_alias": entry["source"], "source_kind": "preregistered_historical_inventory",
                  "category": entry.get("conditional_subtype", entry.get("saved_repair_type", purpose)),
                  "reference": entry, "member_ids": entry.get("task_ids") if purpose in AUDITS else None,
                  "repeat_selected": entry["call_id"] in repeats["exact_rare_repeat_calls"].get(purpose, [])}
        if purpose in AUDITS:
            packet.update(batch_ordinal=index, repeat_selected=index in repeats["audit_repeat_saved_batch_ordinals"],
                          category=f"historical_batch_{entry['saved_batch_size']}")
        packets.append(packet)
    return packets


def _fresh_packets(roots: Sequence[Path], requested: Sequence[str]) -> list[dict[str, Any]]:
    packets, bindings = [], build_stage_bindings()
    for root_value in roots:
        root = Path(root_value).resolve()
        if (_read(root / "factory_state.json").get("status") != "complete"
                or _read(root / "report.json").get("status") != "complete"):
            raise FactoryEffortInputsError(f"fresh source run is not terminal: {root}")
        spec_path = root / "frozen_supply_design.private.json"
        specs = {row["call_id"]: row for row in _read(spec_path)["specs"]} if spec_path.is_file() else {}
        board_specs, audit_specs = {}, {}
        for path in sorted((root / "downstream/reviews").glob("*/prepared.json")):
            prepared = _read(path)
            for group in prepared["board_groups"]:
                for spec in group["calls"]:
                    board_specs[spec["call_id"]] = (prepared, group, spec)
            for spec in prepared["audit_specs"]:
                audit_specs[spec["call_id"]] = spec
        for namespace in ("upstream", "downstream"):
            directory = root / namespace
            if not (directory / "state.json").is_file():
                continue
            state = _read(directory / "state.json")
            for ledger in state.get("call_ledger", {}).values():
                purpose = _purpose(ledger, bindings)
                if purpose not in requested or ledger.get("status") != "complete":
                    continue
                call_id = ledger["call_id"]
                packet = {"run_directory": str(directory), "purpose": purpose, "stage": ledger["stage"],
                          "call_id": call_id, "source_kind": "completed_fresh_factory",
                          "category": "fresh_" + ledger["stage"], "origin_model": ledger["model"]}
                if purpose.startswith("idea_"):
                    if call_id not in specs:
                        raise FactoryEffortInputsError("fresh generator lacks its frozen source recipe")
                    spec = specs[call_id]
                    packet.update(search_mode=spec["search_mode"], private_conditioned=spec["private_conditioned"],
                                  member_ids=spec["candidate_ids"], stratum=spec["search_mode"],
                                  expected={"search_call_id": call_id, "search_mode": spec["search_mode"]}, reference=spec)
                if purpose == "editorial_board":
                    if call_id not in board_specs:
                        raise FactoryEffortInputsError("fresh board lacks its authoritative prepared pair")
                    prepared, group, spec = board_specs[call_id]
                    packet.update(group_id="fresh-" + content_hash(str(root))[:8] + "-" + prepared["id"] + f"-{group['group_index']}",
                                  board_index=spec["board_index"], board_mapping=spec["mapping"],
                                  reference={"prepared_hash": prepared["prepared_hash"], "group": group},
                                  member_ids=[row["task_id"] for row in spec["values"]["BLINDED_TASKS"]])
                if purpose in AUDITS:
                    if call_id not in audit_specs:
                        raise FactoryEffortInputsError("fresh audit lacks its authentic prepared batch")
                    spec = audit_specs[call_id]
                    packet.update(reference=spec, member_ids=spec["task_ids"], expected=spec["expected"])
                packets.append(packet)
    return packets


def _choose_fresh(packets: Sequence[dict[str, Any]], purpose: str, count: int) -> list[dict[str, Any]]:
    candidates = [deepcopy(row) for row in packets if row["purpose"] == purpose]
    # This ordering uses input bytes and origin, never saved success/quality labels.
    def key(row: Mapping[str, Any]) -> tuple[bool, str]:
        prompt = Path(row["run_directory"]) / "model_calls" / row["stage"] / row["call_id"] / "prompt.md"
        return row.get("origin_model") != "gpt-6-astra", content_hash(_bytes(prompt))
    candidates.sort(key=key)
    if purpose in {"idea_conditioned", "idea_unconditioned"}:
        selected = [row for mode in sorted(ORDINARY_MODES) for row in [r for r in candidates if r.get("stratum") == mode][:2]]
    elif purpose == "editorial_board":
        groups = defaultdict(list)
        for row in candidates:
            groups[row["group_id"]].append(row)
        selected = [row for group in list(groups.values())[:count] for row in sorted(group, key=lambda row: row["board_index"])]
    elif purpose == "planning":
        # Include every available route before filling remaining slots by hash.
        by_stage = {}
        for row in candidates:
            by_stage.setdefault(row["stage"], row)
        selected = list(by_stage.values())[:count]
        selected += [row for row in candidates if row not in selected][:max(0, count - len(selected))]
    else:
        selected = candidates[:count]
    for index, row in enumerate(selected):
        if purpose == "editorial_board":
            group_order = list(dict.fromkeys(item["group_id"] for item in selected))
            row["repeat_selected"] = row["group_id"] in group_order[:SCREEN[purpose][2]]
        else:
            row["repeat_selected"] = index < SCREEN[purpose][2]
    return selected


def build_effort_inputs(config: Mapping[str, Any], source_runs: Sequence[Path],
                        historical_calibration: Path | None, output_directory: str | Path, *,
                        purposes: Sequence[str] | None = None, inventory_path: Path | None = None) -> dict[str, Any]:
    """Resolve approved inventory, historical packets and explicitly named fresh runs.

    Fresh Astra inputs are preferred; explicit S0 roots may fill genuine scarcity
    (notably global pools). Their originating-model strata remain in coverage.
    No historical creative prompts or shuffled duplicates fill missing A0 slots.
    """
    requested = _requested(purposes)
    inventory_file, inventory = _inventory(config, inventory_path)
    resolver_identity = content_hash({"source_runs": [str(Path(path).resolve()) for path in source_runs],
                                      "historical_calibration": str(Path(historical_calibration).resolve()) if historical_calibration else None,
                                      "purposes": requested, "inventory_sha256": content_hash(_bytes(inventory_file))})
    if (Path(output_directory) / MANIFEST_NAME).exists():
        saved = verify_effort_inputs(output_directory)
        if saved.get("resolver_identity") != resolver_identity:
            raise FactoryEffortInputsError("effort inputs are already frozen from different source arguments")
        return saved
    fresh_purposes = [p for p in requested if p not in RARE | {"technical_verification", "primary_blind"}]
    fresh = _fresh_packets(source_runs, fresh_purposes) if fresh_purposes else []
    historical = _historical_packets(historical_calibration, [p for p in requested if p in {
        "technical_verification", "primary_blind", "editorial_board"}], inventory) if historical_calibration else []
    packets = []
    for purpose in requested:
        if purpose in RARE:
            selected = _inventory_packets(config, inventory, purpose)
        elif purpose in {"technical_verification", "primary_blind"}:
            selected = [row for row in historical if row["purpose"] == purpose]
        elif purpose in AUDITS:
            selected = _choose_fresh(fresh, purpose, SCREEN[purpose][0])
            # A whole historical cohort is preferable to an unregistered mixed panel.
            if len(selected) < SCREEN[purpose][0]:
                selected = _inventory_packets(config, inventory, purpose)
        elif purpose == "editorial_board":
            selected = [row for row in historical if row["purpose"] == purpose]
            needed = max(0, SCREEN[purpose][0] - len({row["group_id"] for row in selected}))
            extra = _choose_fresh(fresh, purpose, needed)
            repeated = {row["group_id"] for row in selected if row["repeat_selected"]}
            extra_repeats = list(dict.fromkeys(row["group_id"] for row in extra))[:max(0, 2 - len(repeated))]
            for row in extra:
                row["repeat_selected"] = row["group_id"] in extra_repeats
            selected += extra
        else:
            selected = _choose_fresh(fresh, purpose, SCREEN[purpose][0])
        packets.extend(selected)
    return freeze_effort_inputs(config, packets, output_directory, purposes=requested,
                                inventory_path=inventory_path, _resolver_identity=resolver_identity)
