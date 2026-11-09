"""Bind terminal evidence to the episode that was actually reviewed.

This is an accounting/integrity projection, not a new model or quality gate.
The existing conditional evidence policy still decides clear versus held.
An old judgment cannot condemn a repaired episode, and a clear label cannot
override a concrete unresolved judgment about the *current* episode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .e18_artifacts import strict_technical_clear
from .e18_evidence_gate import E18EvidenceError, _gate, _persist_metadata, evidence_is_clear
from .util import content_hash, read_json
from arcgen.astra_usage_diversity_20260913 import adjudication as difficulty_adjudication



def _response(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    value = row.get("response") or row
    return dict(value) if isinstance(value, Mapping) else {}


def _bound_calls(task_id: str, response: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    digest = content_hash(response)
    return [row for row in calls if row.get("status") == "complete"
            and task_id in (row.get("parent_ids") or [])
            and row.get("response_content_hash") == digest
            and (row.get("candidate_episode_hash") or (row.get("candidate_episode_hashes") or {}).get(task_id))]


def _episode_version(task_id: str, call: Mapping[str, Any]) -> str:
    return str(call.get("candidate_episode_hash") or (call.get("candidate_episode_hashes") or {}).get(task_id))


def _right_twice_clear(record: Mapping[str, Any], evidence: Mapping[str, Any], final_blind: Sequence[Mapping[str, Any]]) -> bool:
    """Gate study (2026-10-15, arcgen/batch/gate.py right_twice): an idea drawn into the right-twice gate, cleared there
    because its first two current-episode blind answers both give the intended test outputs. Other ideas never match."""
    if (record.get("study") or {}).get("gate") != "right_twice" or evidence.get("resolution") != "right_twice_clear":
        return False
    expected = [pair["output"] for pair in (record.get("complete_task") or {}).get("test", [])]
    return bool(expected) and len(final_blind) >= 2 and all(
        _response(row).get("predicted_test_outputs") == expected for row in final_blind[:2])


def project_final_evidence(
    task_id: str, record: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Require successful, version-bound evidence; never reinterpret old banks.

    Call ledgers hash parsed responses independently of the wrapper's later
    derived fields. Legacy imported records with no binding remain unknown.
    """
    episode = record.get("complete_task")
    episode_hash = content_hash(episode) if episode else None
    relevant = [row for row in calls if task_id in (row.get("parent_ids") or [])]

    def versions(response: Mapping[str, Any]) -> set[str]:
        return {_episode_version(task_id, row) for row in _bound_calls(task_id, response, relevant)}

    technical = list(record.get("technical_results") or [])
    blind = list(record.get("blind_results") or [])
    final_blind = [row for row in blind if episode_hash in versions(_response(row))]
    terminal_technical_bound = bool(
        technical and episode_hash in versions(_response(technical[-1]))
    )
    progress = record.get('difficulty_versions', {}).get(episode_hash, {})
    adjudication = progress.get('adjudication') or record.get("ambiguity_adjudication") or {}
    current_adjudication_calls = [r for r in relevant if r.get('status') == 'complete'
        and r.get('stage') == difficulty_adjudication.STAGE and _episode_version(task_id, r) == episode_hash]

    adjudication_versions = versions(adjudication) if adjudication else set()
    adjudication_current = bool(episode_hash and episode_hash in adjudication_versions)
    adjudication_stale = bool(adjudication_versions and not adjudication_current)
    evidence = record.get("answer_evidence") or {}
    reasons: list[str] = []
    status = str(evidence.get("status") or "not_evaluated")
    effective_ambiguity: bool | None = None
    raw_alarm_count = sum(
        bool(_response(row).get("serious_different_output_alternative")) for row in blind
    )
    current_adjudication_ambiguous = adjudication_current and (
        adjudication.get("decision") in {"ambiguous", "serious_ambiguity"}
        or adjudication.get("comparably_simple_different_output_rule") is True
    )
    current_adjudication_clear = adjudication_current and (
        difficulty_adjudication.is_clear(adjudication)
        and len(current_adjudication_calls) == 1
    )
    if current_adjudication_clear:
        try:
            difficulty_adjudication.validate_response(adjudication, {
                'TASK_ID': task_id, 'EPISODE_HASH': episode_hash, 'COMPLETE_TASK': episode,
                'CONCERNS': difficulty_adjudication.concerns(final_blind)})
        except (ValueError, TypeError, KeyError):
            current_adjudication_clear = False
    if len(current_adjudication_calls) > 1:
        status = 'evidence_hold'
        reasons.append('multiple_current_version_adjudications')
    elif not episode_hash:
        status = "not_evaluated"
        reasons.append("no_completed_episode")
    elif current_adjudication_ambiguous:
        status = "ambiguous"
        effective_ambiguity = True
        reasons.append("current_episode_has_unresolved_different_output_rule")
    elif evidence_is_clear(record):
        if not terminal_technical_bound or not strict_technical_clear(record):
            reasons.append("final_episode_missing_clear_bound_technical_evidence")
        if not final_blind:
            reasons.append("final_episode_missing_bound_blind_evidence")
        marker = record.get('difficulty_repair')
        if marker and marker['source_episode_hash'] != episode_hash:
            from arcgen.astra_usage_diversity_20260913.gate import _repair_values
            from arcgen.astra_usage_diversity_20260913.contracts import Bindings
            from arcgen.astra_usage_diversity_20260913 import recheck
            try:
                response = technical[-1]['response']
                values, _ = Bindings().prepare_values('difficulty_technical_recheck', {
                    'TASK_ID': task_id, 'VERIFIER_ID': response['verifier_id'],
                    'CONCRETE_PLAN': record['concrete_plan'], 'PUBLIC_TASK': record['realization']['task'],
                    'DIAGNOSTICS': record['technical_diagnostics'], 'REPAIR_CHALLENGE': _repair_values(record)})
                if not recheck.cleared(response, values):
                    reasons.append('original_repair_target_not_demonstrably_removed')
            except (KeyError, TypeError, ValueError):
                reasons.append('repair_target_evidence_missing_or_changed')
        if adjudication and not adjudication_versions:
            reasons.append("adjudication_episode_binding_missing")
        if final_blind:
            # A clean second answer alone does not dismiss a primary concern.
            # Only a current clear adjudication or reverified changed episode
            # resolves it under the existing policy, or, for ideas drawn into
            # the right-twice gate, two right answers on this episode.
            try:
                current_concern = any(_gate(record, row)["proposed_review_required"]
                                      for row in final_blind)
            except E18EvidenceError:
                current_concern = True
                reasons.append("final_episode_missing_hidden_output_metadata")
            if current_concern and not current_adjudication_clear and not _right_twice_clear(record, evidence, final_blind):
                reasons.append("current_episode_blind_concern_not_adjudicated")
        if evidence.get("resolution") == "adjudicated_clear" and not current_adjudication_clear:
            reasons.append("clear_adjudication_is_not_bound_to_final_episode")
        if reasons:
            status = "evidence_hold"
        else:
            effective_ambiguity = False
            reasons.append("existing_clear_policy_supported_by_final_episode_evidence")
    else:
        reasons.append("existing_pipeline_has_not_cleared_evidence")
        if status not in {"ambiguous", "evidence_hold", "not_evaluated"}:
            status = "evidence_hold"
    return {
        "schema_version": 1,
        "episode_hash": episode_hash,
        "status": status,
        "reasons": reasons,
        "effective_terminal_ambiguity": effective_ambiguity,
        "raw_blind_alarm_count": raw_alarm_count,
        "final_episode_blind_reviews": len(final_blind),
        "terminal_technical_bound": terminal_technical_bound,
        "adjudication_current": adjudication_current,
        "stale_adjudication_retained_in_dossier": adjudication_stale,
    }


def recover_terminal_evidence_metadata(runtime):
    """Finish only missing terminal metadata using exact version-bound saved responses."""
    report = {'recovered': [], 'held': [], 'provider_calls': 0}
    terminal = {'evidence_clear', 'evidence_clear_after_review', 'evidence_hold', 'ambiguous'}
    with runtime.lock:
        ledger = runtime.state.get('call_ledger') or {}
        calls = list(ledger.values()) if isinstance(ledger, Mapping) else list(ledger)
        for tid, record in runtime.state['semantic_records'].items():
            prior = record.get('answer_evidence') or {}
            if record.get('status') != 'blind_pass' or prior.get('status') in terminal:
                continue
            candidate = deepcopy(record); errors = []
            relevant = [r for r in calls if tid in r.get('parent_ids', [])]
            current_hash = content_hash(candidate['complete_task']) if candidate.get('complete_task') else None
            versions = candidate.setdefault('difficulty_versions', {})
            by_version = {}
            for call in relevant:
                if call.get('status') != 'complete' or call.get('stage') != difficulty_adjudication.STAGE:
                    continue
                version = _episode_version(tid, call)
                by_version.setdefault(version, []).append(call)
                try:
                    path = Path(runtime.calls_dir) / call['stage'] / call['call_id'] / 'parsed_response.json'
                    value = difficulty_adjudication.parse(read_json(path))
                    if (content_hash(value) != call.get('response_content_hash') or value['task_id'] != tid
                            or value['episode_hash'] != version or version not in versions):
                        raise ValueError('saved adjudication identity/version/history changed')
                    old = versions[version].get('adjudication')
                    if old is not None and old != value:
                        raise ValueError('saved adjudication differs from semantic history')
                    versions[version]['adjudication'] = value
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append('saved_adjudication_unavailable_or_changed: ' + str(exc))
            if any(len(rows) != 1 for rows in by_version.values()):
                errors.append('multiple_saved_adjudications_for_one_version')
            progress = versions.get(current_hash, {})
            current_adj = progress.get('adjudication')
            if current_adj: candidate['ambiguity_adjudication'] = current_adj
            from arcgen.astra_usage_diversity_20260913.gate import current_blind
            blind = current_blind(tid, candidate, relevant) if current_hash else []
            outcome = progress.get('outcome', {})
            resolution = outcome.get('resolution') if outcome.get('kind') == 'clear' else None
            if resolution is None and len(blind) == 1 and not _gate(candidate, blind[0])['proposed_review_required'] and not candidate.get('difficulty_repair'):
                resolution = 'ordinary_primary_clear'
            if not resolution: errors.append('terminal_resolution_not_established_by_saved_evidence')
            desired = 'evidence_clear' if resolution == 'ordinary_primary_clear' else 'evidence_clear_after_review'
            candidate['answer_evidence'] = {'status': desired, 'resolution': resolution}
            projection = project_final_evidence(tid, candidate, relevant)
            clear = not errors and projection['status'] in ('evidence_clear', 'evidence_clear_after_review')
            record['difficulty_versions'] = versions
            if current_adj: record['ambiguity_adjudication'] = deepcopy(current_adj)
            _persist_metadata(runtime, tid,
                evidence_status=desired if clear else 'ambiguous' if projection['status'] == 'ambiguous' else 'evidence_hold',
                route={'episode_hash': current_hash, 'resolution': resolution if clear else 'terminal_metadata_recovery_unresolved',
                    'difficulty_outcome': deepcopy(outcome) or None,
                    'conditional_review_used': resolution != 'ordinary_primary_clear',
                    'ambiguity_adjudication': deepcopy(current_adj) if current_adj else record.get('ambiguity_adjudication'),
                    'repair_count': int(record.get('repair_count') or 0),
                    'metadata_recovery': {'saved_evidence_only': True, 'provider_calls': 0,
                                         'reasons': errors + ([] if clear else projection['reasons'])}},
                force_terminal_hold=not clear)
            report['recovered' if clear else 'held'].append(tid)
    return report


def apply_final_evidence(runtime: Any, task_id: str) -> dict[str, Any]:
    """Persist the projection before editorial admission or terminal export."""
    with runtime.lock:
        record = runtime.state["semantic_records"][task_id]
        calls = runtime.state.get("call_ledger") or []
        if isinstance(calls, Mapping):
            calls = list(calls.values())
        result = project_final_evidence(task_id, record, calls)
        record["final_episode_evidence"] = result
        if evidence_is_clear(record) and result["status"] not in {
            "evidence_clear", "evidence_clear_after_review"
        }:
            record.setdefault("answer_evidence_history", []).append(
                deepcopy(record["answer_evidence"])
            )
            record["answer_evidence"] = {
                **record["answer_evidence"],
                "status": result["status"],
                "resolution": "final_episode_binding_unresolved",
            }
            record["pre_evidence_hold_status"] = record.get("status")
            record["status"] = "evidence_hold"
        from arcgen.astra_usage_diversity_20260913.gate import performance
        record['blind_performance'] = performance(task_id, record, calls)
        if (record.get('study') or {}).get('gate'):  # gate study 2026-10-15: right-twice fields (arcgen/batch/gate.py)
            from arcgen.batch.gate import _gate_study_fields
            record['blind_performance'].update(_gate_study_fields(task_id, record, calls, record.get('answer_evidence') or {}))
        study = record.get('study') or {}
        if study.get('line') or 'drawn' in (study.get('weak_overlays') or {}).values() or study.get('structure_call'):
            # weakness run 2026-10-20: the line's checks on the final episode (arcgen/batch/lines.py)
            from arcgen.batch.lines import fields as _weakness_fields
            record['blind_performance'].update(_weakness_fields(task_id, record, calls))
        if study.get('slot_control') or 'ONE' in (study.get('idea_modifiers') or ()):
            # ONE (2026-10-23): the final episode's training pair count (arcgen/batch/slots.py)
            from arcgen.batch.slots import fields as _slot_fields
            record['blind_performance'].update(_slot_fields(record))
        runtime.save()
        return deepcopy(result)
