"""Bounded evidence review, preserving every response and episode version."""
from copy import deepcopy
from arcgen.lib.e18_evidence_gate import _gate, _persist_metadata
from arcgen.lib.e18_artifacts import strict_technical_clear
from arcgen.lib.util import content_hash
from arcgen.lib.v41_supported_departure import (
    _blind_call, _wrap_blind, _record_stage, _repair_call,
    _store_repaired_realization, _technical_evaluation,
)
from .common import BudgetError, IntegrityError, HERE
from .config import SCIENCE
from . import lines, slots
from arcgen.lib.schemas import SchemaError

# diversity v1: up to three rule-preserving repairs with brakes; r23/default keep the single repair.
V1 = SCIENCE.get('prompt_set', 'r23') == 'v1'
MAX_REPAIRS = 3 if V1 else 1
MAX_ADDED_PAIRS = 2  # v1 repairs may replace pairs freely but add at most this many beyond the planner's count
PAIR_LIMIT_NOTE = ('\nThis task already has the most training pairs repairs may add. Replace an existing training pair '
                   'instead of adding one; if that cannot resolve the concern, return unrepairable.')
from arcgen.astra_usage_diversity_20260913 import adjudication, recheck


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
        'intrinsic_difficulty': 'not_established',
        **_gate_study_fields(tid, record, calls, evidence), **lines.fields(tid, record, calls), **slots.fields(record)}


def _gate_study_fields(tid, record, calls, evidence):
    # Runs with science.gate_draw only (every idea carries study.gate), so earlier records are unchanged.
    arm = record.get('study', {}).get('gate')
    if arm is None:
        return {}
    e = right_twice_evidence(tid, record, calls) or {}
    cleared = evidence.get('resolution') == 'right_twice_clear'
    return {'gate_arm': arm, 'gate_resolution': evidence.get('resolution'),
            'right_twice_eligible': bool(e and e['both_right'] and e['serious_rival'] and not e['paint_only_rival']),
            'contested': bool(cleared and e.get('second_flagged')),
            'rival_test_outputs': e.get('rival_test_outputs', []) if cleared else []}


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
            if len(versions) >= MAX_REPAIRS + 1:
                raise IntegrityError('difficulty review exceeded its episode-version limit')
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
    validation = False
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, BudgetError): return 'budget'
        validation = validation or isinstance(exc, SchemaError)
        seen.add(id(exc)); exc = exc.__cause__ or exc.__context__
    return 'validation' if V1 and validation else 'operational'


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


def _paint_only(task, intended, other):
    """A rival that changes exactly the cells the intended answer changes, only with other colours: a colour choice,
    which a repeated call cannot vouch for."""
    if not other or len(other) != len(intended) or other == intended:
        return False
    for pair, mine, theirs in zip(task['test'], intended, other):
        grid = pair['input']
        if not (len(mine) == len(theirs) == len(grid) and all(len(a) == len(b) == len(c) for a, b, c in zip(mine, theirs, grid))):
            return False
        if any((a != c) != (b != c) for ra, rb, rc in zip(mine, theirs, grid) for a, b, c in zip(ra, rb, rc)):
            return False
    return True


def right_twice_evidence(tid, record, calls):
    """Gate study (2026-10-15): whether a flagged
    episode qualifies for the right-twice clear. Both of the first two blind calls give the intended test outputs, the
    first call names a serious different-output rival (a guessing-only flag keeps the old route), and no listed rival is
    a colour choice on the intended cells. Returns None without a finished episode or two blind calls."""
    if not record.get('complete_task'):  # a development that ended before its episode (plan abort, failed realization)
        return None
    blind = current_blind(tid, record, calls)
    if len(blind) < 2:
        return None
    expected = [p['output'] for p in record['complete_task']['test']]
    responses = [row['response'] for row in blind[:2]]
    rivals = [alt.get('predicted_test_outputs') for r in responses for alt in r.get('strongest_plausible_alternatives') or []]
    rivals = [x for i, x in enumerate(rivals) if x and x != expected and x not in rivals[:i]]
    return {'both_right': all(r['predicted_test_outputs'] == expected for r in responses),
            'serious_rival': responses[0].get('serious_different_output_alternative') is True,
            'paint_only_rival': any(_paint_only(record['complete_task'], expected, x) for x in rivals),
            'second_flagged': responses[1].get('serious_different_output_alternative') is True
                              or responses[1].get('test_requires_unsupported_guessing') is True,
            'rival_test_outputs': rivals}


def right_twice(tid, record, calls):
    """For an idea drawn into the right-twice gate (science.gate_draw), a qualifying flagged episode clears without
    adjudication or repair; the blind responses, with their rivals, stay on the record."""
    if record.get('study', {}).get('gate') != 'right_twice':
        return False
    e = right_twice_evidence(tid, record, calls)
    return bool(e and e['both_right'] and e['serious_rival'] and not e['paint_only_rival'])


DEPART_POLICY = (HERE / 'prompts/v1/depart.adjudicator.md').read_text().split('-->', 1)[1].strip()
DEPART_REPAIR_NOTE = (HERE / 'prompts/v1/depart.repair_note.md').read_text().split('-->', 1)[1].strip()
WITHHELD = 'Withheld from review for this episode (departure package): judge from the grids and the public contract.'


DEPART_POLICY_VERSION = 'b17-depart-adjudicator-2'


def departure_invariant(record):
    """Whether a drawn idea's plan names its reserved situation in the public invariant "No training pair shows ..."."""
    contract = (record.get('concrete_plan') or {}).get('public_contract') or {}
    return any(isinstance(x, str) and x.strip().startswith('No training pair shows') for x in contract.get('invariants') or [])


def departure_review(tid, record, calls):
    """Departure package (2026-10-16). For an idea drawn into it:
    'policy' when the episode exists, its plan names the reserved situation in the public invariant ("No training pair
    shows ...") and at least one current blind call gave the intended answer; otherwise the reason the review policy does
    not apply (that adjudication is the control's). None for every other idea."""
    if record.get('study', {}).get('depart') != 'test':
        return None
    if not record.get('complete_task'):
        return 'no_episode'
    if not departure_invariant(record):
        return 'no_invariant'
    expected = [p['output'] for p in record['complete_task']['test']]
    if not any(row['response']['predicted_test_outputs'] == expected for row in current_blind(tid, record, calls)):
        return 'no_right_answer'
    return 'policy'


def _withhold_departure_arguments(plan):
    """For a drawn idea the adjudicator judges the reserved situation from the grids and the public contract: the plan's
    arguments for it (the test audits, the shortcut rebuttals, the remaining-risk note) are withheld."""
    private = plan.get('private_execution') or {}
    for row in private.get('test_outputs') or []:
        for key in ('test_only_semantics_audit', 'why_one_output_is_best_supported'):
            if key in row:
                row[key] = WITHHELD
    audit = private.get('cross_pair_concrete_audit') or {}
    if 'remaining_ambiguity_risk' in audit:
        audit['remaining_ambiguity_risk'] = WITHHELD
    for row in audit.get('strongest_simple_shortcuts') or []:
        if 'evidence_that_defeats_it' in row:
            row['evidence_that_defeats_it'] = WITHHELD


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
    review = departure_review(tid, record, calls)
    # weakness run (2026-10-20): the line's and overlay's review policies, each when its invariant is in the plan; DEPART v4
    # keeps the departure package's condition and withholding (weakness_lines.json)
    names, _ = lines.review(record, blind)
    if review == 'policy' or lines.withholds(names):  # otherwise the adjudicator sees exactly what a control episode's would
        _withhold_departure_arguments(specification)
    # ONE (2026-10-23): the slot's policy when the episode has its one training pair (slots.review)
    slot_names, _ = slots.review(record)
    texts = (([DEPART_POLICY] if review == 'policy' else []) + ([lines.policy_text(names)] if names else [])
             + ([slots.policy_text(slot_names)] if slot_names else []))
    policy = {'REVIEW_POLICY': '\n\n'.join(texts)} if V1 else {}
    return {**policy, 'TASK_ID': tid, 'EPISODE_HASH': content_hash(record['complete_task']),
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


def _pair_limit(record):
    """The planner's own training-pair count plus what repairs may add; the planner's choice is never capped."""
    return record['planning_result']['concrete_plan']['public_contract']['episode_overview']['training_pair_count'] + MAX_ADDED_PAIRS


def _repair(runtime, tid, budget, version, adj, clarifying=None):
    """An adjudicated repair, or (weakness run) a clarifying repair: clarifying = (instruction, challenges, evidence) when
    both blind calls took a declared shortcut (lines.clarify)."""
    record = _record(runtime, tid)
    if record.get('repair_count', 0) >= MAX_REPAIRS:
        _finish(runtime, tid, 'unresolved', 'one_repair_already_used' if MAX_REPAIRS == 1 else 'repair_limit_reached'); return
    if not budget.reserve_once(f'{tid}:{version}:difficulty-repair', 3):
        _finish(runtime, tid, 'budget', 'repair_package_unfunded'); return
    with runtime.lock:
        runtime.state['semantic_records'][tid]['difficulty_repair'] = {
            'source_episode_hash': version,
            'challenges': clarifying[1] if clarifying else adjudication.repair_challenges(adj),
            **({'route': 'shortcut_capture'} if clarifying else {})}
        runtime.save()
    try:
        # Later repairs are new calls on the current version, not a changed request under the first repair's ID.
        call_id = f'{tid}-v{version[:12]}-ambiguity-repair' if record.get('repair_count', 0) else None
        instruction = clarifying[0] if clarifying else adj['repair_instruction']
        if V1 and len(record['complete_task']['train']) >= _pair_limit(record):
            instruction += PAIR_LIMIT_NOTE
        repaired = _repair_call(runtime, tid, repair_type='ambiguity', instruction=instruction,
                                evidence=clarifying[2] if clarifying else adj, call_id=call_id)
    except Exception as exc:
        kind = _error_kind(exc)
        _finish(runtime, tid, kind, 'repair_validation_failure' if kind == 'validation' else 'repair_unavailable'); return
    if repaired['status'] != 'repaired':
        _finish(runtime, tid, 'defect', 'repair_declined'); return
    if V1 and len(repaired['repaired_realization']['task']['train']) > _pair_limit(record):
        _finish(runtime, tid, 'defect', 'repair_added_too_many_pairs'); return
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
    if route == 'direct' and lines.on_points(record, blind[0]['response']):
        # weakness run (2026-10-20, DESIGN §5.2): a first answer on a declared shortcut gets a second call, so that two
        # answers on it can be routed to a clarifying repair rather than to clearance
        route = 'second_then_adjudicate'
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
    review = departure_review(tid, _record(runtime, tid), _calls(runtime))
    if review and progress.get('departure_review') != review and not progress.get('adjudication'):
        _save_version(runtime, tid, version, departure_review=review,
                      departure_policy_version=DEPART_POLICY_VERSION if review == 'policy' else None)
    current = _record(runtime, tid)
    names, reason = lines.review(current, current_blind(tid, current, _calls(runtime)))
    if names is not None and not progress.get('adjudication'):
        line_review = names or reason
        if progress.get('line_review') != line_review:
            _save_version(runtime, tid, version, line_review=line_review,
                          line_policy_version=lines.POLICY_VERSION if names else None)
    slot_names, slot_reason = slots.review(current)  # ONE (2026-10-23): which slot policy the adjudicator gets, recorded
    if slot_names is not None and not progress.get('adjudication'):
        slot_review = slot_names or slot_reason
        if progress.get('slot_review') != slot_review:
            _save_version(runtime, tid, version, slot_review=slot_review,
                          slot_policy_version=slots.POLICY_VERSION if slot_names else None)
    # at most one clarifying repair per task (review S2): a capture on a later version goes to adjudication
    earlier = any(v.get('shortcut_capture') for h, v in (current.get('difficulty_versions') or {}).items() if h != version)
    capture = None if progress.get('adjudication') or earlier else lines.capture_route(current, current_blind(tid, current, _calls(runtime)))
    if capture:  # both blind calls on the same declared shortcut: a clarifying repair, never clearance (DESIGN §5.2)
        if progress.get('shortcut_capture') != capture:
            _save_version(runtime, tid, version, shortcut_capture=capture)
        instruction, challenges, evidence = lines.clarify(current, capture, current_blind(tid, current, _calls(runtime)))
        _repair(runtime, tid, budget, version, None, clarifying=(instruction, challenges, evidence)); return
    if not progress.get('adjudication') and right_twice(tid, _record(runtime, tid), _calls(runtime)):
        _finish(runtime, tid, 'clear', 'right_twice', resolution='right_twice_clear'); return
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
        if lines.is_decoy(_record(runtime, tid)):  # a JUDGE decoy stays the negative it was planted as (review S3)
            _finish(runtime, tid, 'unresolved', 'decoy_not_repaired'); return
        _repair(runtime, tid, budget, version, adj)
    else:
        _finish(runtime, tid, 'defect' if adj['decision'] == 'defect' else 'unresolved', adj['decision'])


def _resolve(runtime, tid, budget):
    record = _record(runtime, tid)
    marker = record.get('difficulty_repair')
    if marker and content_hash(record['complete_task']) != marker['source_episode_hash']:
        version, progress = _version(runtime, tid)
        if progress.get('outcome'):
            outcome = progress['outcome']
            _finish(runtime, tid, outcome['kind'], outcome['reason'], resolution=outcome['resolution']); return
        if not _recheck_repair(runtime, tid): return
    _review(runtime, tid, budget)


def resolve(runtime, tid, budget):
    result = _resolve(runtime, tid, budget)
    # Native operational/semantic completion releases only undispatched slots.
    # UsagePaused and interrupts propagate and retain packages for exact resume.
    budget.release_task(tid)
    return result


# ------------------------------------------------------------------ the default-reading call (STRUCTURE_SPEC D)


def _default_call(runtime, tid, version, request):
    """One default-reading call: the blind solver prompt and one sentence asking for the most straightforward answer,
    on the chosen training pairs and the test inputs (no plan, no hints); low-effort Sol (call_overrides.default_reading).
    The call ID ends "-default-reading", so it is a conditional call charged to the task's own package."""
    from arcgen.lib.util import render_prompt
    from .structures import default_task
    record = _record(runtime, tid)
    key = request['key']
    public = default_task(record, request)
    opaque = 'blind-' + content_hash({'seed': runtime.config['seed'], 'task': tid, 'default_reading': key})[:12]
    solver_id = 'solver-' + content_hash({'task': tid, 'suffix': 'default-reading-' + key})[:12]
    values, _ = runtime.q2.prepare_values('blind_solver', {'PUBLIC_TASK': public, 'TASK_ID': opaque, 'SOLVER_ID': solver_id})
    rows = lines.default_templates()
    template = '\n\n'.join(r[0] for r in rows)
    prompt = render_prompt(template, values)
    context = {'profile_id': runtime.q2.profile_id, 'prompt_name': 'default_reading', 'values': deepcopy(values),
               'q2_template_hashes': [r[2] for r in rows]}
    return runtime.invoke_rendered(purpose='conditional_blind', stage=lines.DEFAULT_STAGE,
        call_id=f'{tid}-v{version[:12]}-{key}-default-reading', prompt=prompt,
        prompt_template_version='+'.join(r[1] for r in rows),
        prompt_template_hash=content_hash({'templates': [r[2] for r in rows], 'combined': template}),
        parent_ids=[tid], arm_id=f"private-{record['arm']}", role='blind_solver', _review_context=context).value


def default_readings(runtime, tid, budget):
    """The recorded default-reading calls of one verified task (STRUCTURE_SPEC D, S2, S9; science.weakness.default_reading):
    SETTLE without its settling pairs, ELIMINATE without each rival's killer pair, PARSE on the pairs its naive parse fits,
    each compared with the declared rival by the 80% rule. Record only: never a gate, never a repair; a failed call is
    recorded and the task goes on. Funded as one package of at most three calls per task; resumable (a saved result for
    the current episode is kept)."""
    if not (SCIENCE.get('weakness') or {}).get('default_reading'):
        return None
    record = _record(runtime, tid)
    evidence = (record.get('answer_evidence') or {}).get('status')
    if not record.get('complete_task') or evidence not in ('evidence_clear', 'evidence_clear_after_review'):
        return None
    from . import structures
    version = content_hash(record['complete_task'])
    try:
        requests = structures.default_requests(record)
    except Exception as exc:  # record only: a record that cannot be read is noted, never a halt of the window
        value = {'episode_hash': version, 'requests': 0, 'results': [], 'status': 'request_error',
                 'detail': (_error_kind(exc) + ': ' + str(exc).strip())[:300]}
        with runtime.lock:
            runtime.state['semantic_records'][tid].setdefault('weakness', {})['default_readings'] = deepcopy(value)
            runtime.save()
        return value
    if not requests:
        return None
    saved = (record.get('weakness') or {}).get('default_readings') or {}
    if saved.get('episode_hash') == version and saved.get('status') in ('complete', 'unfunded'):
        return saved
    value = {'episode_hash': version, 'requests': len(requests), 'results': [], 'status': 'complete'}
    if not budget.reserve_once(f'{tid}:default-reading', len(requests)):
        value['status'] = 'unfunded'
    else:
        for request in requests:
            try:
                response = _default_call(runtime, tid, version, request)
                value['results'].append(structures.default_result(record, request, response))
            except Exception as exc:  # recorded; UsagePaused, DispatchPaused and interrupts propagate for an exact resume
                value['results'].append({'key': request['key'], 'basis': request['basis'], 'error': _error_kind(exc),
                                         'detail': str(exc).strip()[:300]})
        budget.release_task(tid)
    with runtime.lock:
        runtime.state['semantic_records'][tid].setdefault('weakness', {})['default_readings'] = deepcopy(value)
        runtime.save()
    return value


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
