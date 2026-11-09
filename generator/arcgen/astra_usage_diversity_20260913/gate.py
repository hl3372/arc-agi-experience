"""Bounded evidence review, preserving every response and episode version."""
from copy import deepcopy
from arcgen.lib.e18_evidence_gate import _gate, _persist_metadata
from arcgen.lib.e18_artifacts import strict_technical_clear
from arcgen.lib.util import content_hash
from arcgen.lib.v41_supported_departure import (
    _blind_call, _wrap_blind, _record_stage, _repair_call,
    _store_repaired_realization, _technical_evaluation,
)
from .common import BudgetError, IntegrityError
from . import adjudication, recheck


def _record(runtime, tid):
    return deepcopy(runtime.state['semantic_records'][tid])


def _calls(runtime):
    rows = runtime.state.get('call_ledger', {})
    return list(rows.values()) if isinstance(rows, dict) else list(rows)


def current_blind(tid, record, calls):
    from arcgen.lib.factory_evidence import _bound_calls, _episode_version
    version = content_hash(record['complete_task']); out = []; seen = set()
    for wrapped in record.get('blind_results', []):
        response = wrapped['response']; rh = content_hash(response)
        if rh not in seen and any(_episode_version(tid, call) == version for call in _bound_calls(tid, response, calls)):
            out.append(deepcopy(wrapped)); seen.add(rh)
    return out


def technical_bound(tid, record, calls):
    from arcgen.lib.factory_evidence import _bound_calls, _episode_version
    technical = record.get('technical_results', [])
    return bool(technical and strict_technical_clear(record) and any(
        _episode_version(tid, call) == content_hash(record['complete_task'])
        for call in _bound_calls(tid, technical[-1]['response'], calls)))


def performance(tid, record, calls):
    from arcgen.lib.factory_evidence import _bound_calls, _episode_version
    if not record.get('complete_task'):
        return {'scope': 'production_blind_checks', 'primary_result': 'unavailable', 'any_all_test_success': None}
    version = content_hash(record['complete_task'])
    blind = current_blind(tid, record, calls)
    expected = [p['output'] for p in record['complete_task']['test']]
    results = []
    for wrapped in blind:
        response = wrapped['response']; outputs = response['predicted_test_outputs']
        refs = [r for r in _bound_calls(tid, response, calls) if _episode_version(tid, r) == version]
        results.append({'response_hash': content_hash(response), 'all_test_success': outputs == expected,
            'per_test_matches': [i < len(outputs) and outputs[i] == grid for i, grid in enumerate(expected)],
            'calls': [{k: c.get(k) for k in ('stage', 'call_id', 'purpose', 'model', 'reasoning_effort', 'attempts')} for c in refs]})
    evidence = record.get('answer_evidence', {})
    verified = evidence.get('status') in ('evidence_clear', 'evidence_clear_after_review')
    solved = any(r['all_test_success'] for r in results) if results else None
    adj = record.get('difficulty_versions', {}).get(version, {}).get('adjudication', {})
    return {'scope': 'production_blind_checks', 'episode_hash': version,
        'primary_result': 'unavailable' if not results else 'matched' if results[0]['all_test_success'] else 'mismatched',
        'any_all_test_success': solved, 'valid_logical_blind_calls': len(results), 'results': results,
        'failure_kind': adj.get('failure_kind', 'unknown'),
        'label': 'verified; unsolved at tested budget' if verified and solved is False else None,
        'intrinsic_difficulty': 'not_established'}


def annotate_performance(runtime, tid):
    with runtime.lock:
        record = runtime.state['semantic_records'][tid]
        record['blind_performance'] = performance(tid, record, _calls(runtime))
        runtime.save()


def _version(runtime, tid):
    with runtime.lock:
        record = runtime.state['semantic_records'][tid]
        version = content_hash(record['complete_task'])
        versions = record.setdefault('difficulty_versions', {})
        if version not in versions:
            if len(versions) >= 2:
                raise IntegrityError('difficulty review permits at most two episode versions')
            versions[version] = {'episode_hash': version, 'policy': adjudication.POLICY,
                'episode': deepcopy(record['complete_task']), 'public_contract': deepcopy(record['concrete_plan']['public_contract'])}
            runtime.save()
        return version, deepcopy(versions[version])


def _save_version(runtime, tid, version, **fields):
    with runtime.lock:
        runtime.state['semantic_records'][tid]['difficulty_versions'][version].update(deepcopy(fields))
        runtime.save()


def _error_kind(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, BudgetError): return 'budget'
        seen.add(id(exc)); exc = exc.__cause__ or exc.__context__
    return 'operational'


def _finish(runtime, tid, kind, reason, *, resolution=None):
    version, progress = _version(runtime, tid)
    adj = progress.get('adjudication', {})
    rival = adjudication.confirmed_rival(adj)
    if kind == 'clear' and rival:
        raise IntegrityError('a confirmed current-version rival cannot clear')
    evidence_status = 'evidence_clear_after_review' if kind == 'clear' else 'ambiguous' if rival else 'evidence_hold'
    outcome = {'kind': kind, 'reason': reason, 'resolution': resolution, 'confirmed_rival': rival,
               'episode_hash': version, 'status': evidence_status}
    _save_version(runtime, tid, version, outcome=outcome)
    before = _record(runtime, tid).get('answer_evidence', {})
    if before.get('difficulty_outcome') != outcome:
        _record_stage(runtime, tid, 'blind', status='blind_pass' if kind == 'clear' else 'ambiguous' if rival else 'evidence_hold')
        _persist_metadata(runtime, tid, evidence_status=evidence_status,
            route={'episode_hash': version, 'difficulty_outcome': outcome, 'hold_kind': None if kind == 'clear' else kind,
                'conditional_review_used': True, 'resolution': resolution or reason,
                'ambiguity_adjudication': deepcopy(adj) or None,
                'repair_count': _record(runtime, tid).get('repair_count', 0)},
            force_terminal_hold=kind != 'clear' and not rival)
    annotate_performance(runtime, tid)


def _adjudication_values(runtime, tid):
    record = _record(runtime, tid); calls = _calls(runtime)
    if not technical_bound(tid, record, calls):
        raise IntegrityError('adjudication requires current-version independently clear technical evidence')
    blind = current_blind(tid, record, calls)
    if not blind: raise IntegrityError('adjudication has no bound public blind response')
    # The same projection applies to every arm, including comparators. Preserve
    # exact commitments and derivations, but keep the private author-targeting
    # narrative out of the deciding review. Production records remain intact.
    specification = deepcopy(record['concrete_plan'])
    specification['private_execution']['cross_pair_concrete_audit']['relevant_and_incidental_variation'] = (
        'Author targeting account withheld; assess the supplied commitments, grids and evidence.')
    return {'TASK_ID': tid, 'EPISODE_HASH': content_hash(record['complete_task']),
        'AUTHOR_SPECIFICATION': {'intended_rule': record['realization']['authoring_metadata']['intended_rule'],
                                 'concrete_plan': specification},
        'COMPLETE_TASK': record['complete_task'],
        'BLIND_RESULTS': [{'response': row['response']} for row in blind],
        'CONCERNS': adjudication.concerns(blind),
        'TECHNICAL_EVIDENCE': record['technical_results'][-1]['response']}


def _append_blind(runtime, tid, response, status='blind_needs_resolution'):
    current = _record(runtime, tid); results = list(current.get('blind_results', []))
    if not any(row['response'] == response for row in results):
        wrapped = _wrap_blind(current, response)
        wrapped['e18_corrected_evidence_trigger'] = _gate(current, wrapped)
        results.append(wrapped)
    _record_stage(runtime, tid, 'blind', status=status, blind_results=results)


def _repair_values(record):
    marker = record['difficulty_repair']; old = record['difficulty_versions'][marker['source_episode_hash']]['episode']
    new = record['complete_task']
    changed = ['train_' + str(i + 1) for i, pair in enumerate(new['train']) if i >= len(old['train']) or old['train'][i] != pair]
    return {'policy': adjudication.POLICY, 'original_episode_hash': marker['source_episode_hash'],
            'changed_training_pairs': changed, 'concerns': deepcopy(marker['challenges'])}


def _recheck_repair(runtime, tid):
    version, progress = _version(runtime, tid)
    record = _record(runtime, tid); challenge = _repair_values(record)
    if not challenge['changed_training_pairs']:
        _finish(runtime, tid, 'defect', 'repair_did_not_change_public_teaching'); return False
    values = {'TASK_ID': tid, 'VERIFIER_ID': 'verifier-' + content_hash({'task': tid, 'suffix': 'v' + version[:12] + '-technical-after-ambiguity-repair'})[:12],
        'CONCRETE_PLAN': record['concrete_plan'], 'PUBLIC_TASK': record['realization']['task'],
        'DIAGNOSTICS': record['technical_diagnostics'], 'REPAIR_CHALLENGE': challenge}
    if not progress.get('technical_recheck_saved'):
        try:
            response = runtime.call(prompt_name='difficulty_technical_recheck',
                call_id=f'{tid}-v{version[:12]}-technical-after-ambiguity-repair', values=values,
                parent_ids=[tid], role='technical_verifier', arm_id='private-single',
                expected={'task_id': tid, 'verifier_id': values['VERIFIER_ID']})
        except Exception as exc:
            _finish(runtime, tid, _error_kind(exc), 'technical_recheck_unavailable'); return False
        evaluation = _technical_evaluation(_record(runtime, tid), response)
        # Persist even non-pass/uncertain responses before interpreting them.
        latest = _record(runtime, tid); history = list(latest.get('technical_results', []))
        if not any(row['response'] == response for row in history): history.append(evaluation)
        _record_stage(runtime, tid, 'technical', status='blind_needs_resolution', technical_results=history)
        _save_version(runtime, tid, version, technical_recheck_saved=True)
    latest = _record(runtime, tid); technical = latest['technical_results'][-1]
    prepared, _ = runtime.q2.prepare_values('difficulty_technical_recheck', values)
    if not technical_bound(tid, latest, _calls(runtime)) or not recheck.cleared(technical['response'], prepared):
        _finish(runtime, tid, 'defect' if technical['response']['status'] == 'fatal' else 'unresolved', 'repair_target_not_verified_removed')
        return False
    _save_version(runtime, tid, version, repair_challenge_verified=True)
    if not current_blind(tid, latest, _calls(runtime)):
        try:
            response = _blind_call(runtime, tid, 'v' + version[:12] + '-blind-after-ambiguity-repair')
        except Exception as exc:
            _finish(runtime, tid, _error_kind(exc), 'repaired_blind_unavailable'); return False
        _append_blind(runtime, tid, response)
    return True


def _repair(runtime, tid, budget, version, adj):
    record = _record(runtime, tid)
    if record.get('repair_count', 0):
        _finish(runtime, tid, 'unresolved', 'one_repair_already_used'); return
    if not budget.reserve_once(f'{tid}:{version}:difficulty-repair', 3):
        _finish(runtime, tid, 'budget', 'repair_package_unfunded'); return
    with runtime.lock:
        runtime.state['semantic_records'][tid]['difficulty_repair'] = {
            'source_episode_hash': version, 'challenges': adjudication.repair_challenges(adj)}
        runtime.save()
    try:
        repaired = _repair_call(runtime, tid, repair_type='ambiguity', instruction=adj['repair_instruction'], evidence=adj)
    except Exception as exc:
        _finish(runtime, tid, _error_kind(exc), 'repair_unavailable'); return
    if repaired['status'] != 'repaired':
        _finish(runtime, tid, 'defect', 'repair_declined'); return
    # Native rule/test/answer and D1 fidelity checks still apply.
    okay, reason = _store_repaired_realization(runtime, tid, repaired['repaired_plan'], repaired['repaired_realization'], repair_type='ambiguity')
    if not okay:
        _finish(runtime, tid, 'defect', 'repair_fidelity_failure: ' + reason); return
    if content_hash(_record(runtime, tid)['complete_task']) == version:
        _finish(runtime, tid, 'defect', 'repair_did_not_change_episode'); return
    if _recheck_repair(runtime, tid):
        _review(runtime, tid, budget)


def _review(runtime, tid, budget):
    version, progress = _version(runtime, tid)
    if progress.get('outcome'):
        outcome = progress['outcome']
        _finish(runtime, tid, outcome['kind'], outcome['reason'], resolution=outcome['resolution']); return
    record = _record(runtime, tid); blind = current_blind(tid, record, _calls(runtime))
    if not technical_bound(tid, record, _calls(runtime)) or not blind:
        _finish(runtime, tid, 'unresolved', 'missing_current_technical_or_blind_evidence'); return
    primary_gate = _gate(record, blind[0])
    if not primary_gate['proposed_review_required']:
        _finish(runtime, tid, 'clear', 'verified_repaired_episode', resolution='rule_preserving_repair_reverified_clear'); return
    route = 'direct' if set(primary_gate['trigger_reasons']) == {'primary_answer_mismatch'} else 'second_then_adjudicate'
    if not budget.reserve_once(f'{tid}:{version}:difficulty-review:{route}', 1 if route == 'direct' else 2):
        _finish(runtime, tid, 'budget', 'required_review_route_unfunded'); return
    _save_version(runtime, tid, version, route=route)
    if route == 'second_then_adjudicate' and len(blind) == 1:
        try:
            response = _blind_call(runtime, tid, 'v' + version[:12] + '-blind-second')
        except Exception as exc:
            _finish(runtime, tid, _error_kind(exc), 'required_second_blind_unavailable'); return
        _append_blind(runtime, tid, response, 'blind_second_complete')
    if len(current_blind(tid, _record(runtime, tid), _calls(runtime))) > 2:
        raise IntegrityError('too many public blind calls for one episode')
    progress = _record(runtime, tid)['difficulty_versions'][version]
    values = _adjudication_values(runtime, tid)
    if not progress.get('adjudication'):
        try:
            adj = runtime.call(prompt_name='ambiguity_adjudicator',
                call_id=f'{tid}-v{version[:12]}-ambiguity-adjudication', values=values,
                parent_ids=[tid], role='ambiguity_adjudicator', arm_id='private-single',
                expected={'task_id': tid, 'episode_hash': version})
        except Exception as exc:
            _finish(runtime, tid, _error_kind(exc), 'adjudication_unavailable'); return
        _save_version(runtime, tid, version, adjudication=adj)
    else:
        adj = progress['adjudication']
    adjudication.validate_response(adj, values)
    with runtime.lock:
        current = runtime.state['semantic_records'][tid]
        current['ambiguity_adjudication'] = deepcopy(adj)
        runtime.save()
    if adj['decision'] == 'clear':
        _finish(runtime, tid, 'clear', 'concerns_concretely_resolved', resolution='adjudicated_clear')
    elif adj['decision'] == 'repairable':
        _repair(runtime, tid, budget, version, adj)
    else:
        _finish(runtime, tid, 'defect' if adj['decision'] == 'defect' else 'unresolved', adj['decision'])


def resolve(runtime, tid, budget):
    record = _record(runtime, tid)
    marker = record.get('difficulty_repair')
    if marker and content_hash(record['complete_task']) != marker['source_episode_hash']:
        version, progress = _version(runtime, tid)
        if progress.get('outcome'):
            outcome = progress['outcome']
            _finish(runtime, tid, outcome['kind'], outcome['reason'], resolution=outcome['resolution']); return
        if not _recheck_repair(runtime, tid): return
    _review(runtime, tid, budget)


def recover_progress(runtime):
    """Reconcile only saved r5 substeps before native scheduler reconciliation."""
    for tid in list(runtime.state['semantic_records']):
        record = _record(runtime, tid)
        if not record.get('complete_task'): continue
        version = content_hash(record['complete_task'])
        outcome = record.get('difficulty_versions', {}).get(version, {}).get('outcome')
        if outcome and record.get('answer_evidence', {}).get('difficulty_outcome') != outcome:
            _finish(runtime, tid, outcome['kind'], outcome['reason'], resolution=outcome['resolution'])
        elif (record.get('difficulty_repair', {}).get('source_episode_hash') not in (None, version)
              and record.get('status') == 'realized' and record.get('blind_results')):
            # A crash inside the native repair store leaves `realized`. Its
            # generic technical barrier would otherwise wait on this very task,
            # although r5 resumes its dedicated repair recheck in the blind DAG.
            _record_stage(runtime, tid, 'difficulty_repair_resume', status='blind_needs_resolution')
