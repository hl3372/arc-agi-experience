"""Episode, test and content slots; SPREAD; chain-aware repair; in-batch reference waves; single-task development groups
with resizable pools; the time and token estimate."""
from copy import deepcopy
import json
import threading
import time
import pytest
from arcgen.batch import admission, cli, config, contracts, idea_v1, properties, scheduler, supply, waves
from arcgen.batch.common import IntegrityError, digest


def test_spread_and_new_slots_have_their_texts():
    p = idea_v1.PROMPTS
    assert 'SPREAD' in idea_v1.ARMS and contracts.known_primary('SPREAD') and 'SPREAD' in contracts.V1_BRIDGES
    assert (p / 'cards/SPREAD.md').read_text().count('Seek a task') == 1
    for m in ('KEY', 'CHAINED', 'MINIMAL', 'DISTRIBUTED', 'DERIVED', 'MEASURE'):
        assert m in config.MODIFIER_NAMES and m in contracts.SLOT_BRIDGES
        assert (p / 'modifiers' / f'{m}.md').read_text().startswith('<!-- prompt-version:')


def test_reference_profile_carries_the_audit_exclusions():
    excl = config.SCIENCE['modifier_exclusions']
    assert 'MONO' in excl['LIGHT'] and {'BUSY', 'PACKED'} <= set(excl['LIGHT'])  # single-cell answers; LIGHT failing at 30x30
    assert 'CONSTRUCT' in excl['BUSY'] and 'AIM' in excl['DIGEST']  # tedious huge answers; AIM drifting into search
    assert config.SCIENCE['batch_references'] is True and config.OPERATIONS['group_size'] == 1


def _rows(studies):
    return [{'source_candidate_id': f'k{i:04d}', 'private_source': 'ordinary_unconditioned', 'study': deepcopy(s)}
            for i, s in enumerate(studies)]


def test_episode_slots_fix_presentation_and_many_pair_ideas_skip_many():
    studies = ([{'primary_arm': 'SEE', 'idea_modifiers': ['CHAINED']}] * 100 + [{'primary_arm': 'SEE', 'idea_modifiers': ['MINIMAL']}] * 100
               + [{'primary_arm': 'SPREAD'}] * 200 + [{'primary_arm': 'SEE', 'idea_modifiers': ['DISTRIBUTED']}] * 200
               + [{'primary_arm': 'SEE'}] * 300)
    routed = admission.configured_routes({}, _rows(studies))
    fixed = [r for r in routed if set(r['study'].get('idea_modifiers', ())) & {'CHAINED', 'MINIMAL'}]
    assert all(r['study']['colour_roles'] == 'v1' and r['study']['pose'] is False for r in fixed)
    many = [r for r in routed if r['study']['primary_arm'] == 'SPREAD' or 'DISTRIBUTED' in r['study'].get('idea_modifiers', ())]
    assert not any(r['study'].get('overlay') == 'MANY' for r in many)
    assert any(r['study'].get('overlay') == 'XFER' for r in many)
    plain = [r for r in routed if r['study'] .get('primary_arm') == 'SEE' and not r['study'].get('idea_modifiers')]
    assert any(r['study'].get('overlay') == 'MANY' for r in plain)
    assert {r['study']['colour_roles'] for r in plain} == {'vary', 'v1'}


def _chain(start, steps):
    states = [start]
    for _ in range(steps):
        states.append([[min(9, c + 1) for c in row] for row in states[-1]])
    return [{'input': states[i], 'output': states[i + 1]} for i in range(steps)]


def _episode(train):
    pairs = [{'pair_number': i + 1, 'exact_input_construction': json.dumps(p['input'])} for i, p in enumerate(train)]
    plan = {'public_contract': {'semantic_core': {'central_rule': 'Add one.'}, 'test_inputs': [{'ordinal': 1}],
                                'invariants': ['same size'], 'allowed_realizer_freedom': 'none',
                                'episode_overview': {'training_pair_count': len(train), 'test_pair_count': 1},
                                'training_pairs': pairs},
            'private_execution': {'test_outputs': [[[5]]]}}
    real = {'task': {'train': train, 'test': [{'input': [[4]]}]}, 'authoring_metadata': {'expected_test_outputs': [[[5]]]}}
    return plan, real


def test_ambiguity_repair_may_rebuild_a_chain_and_only_a_chain():
    from arcgen.lib.astra_contract_bindings_v3 import validate_repair_scope, chained_episode, chain_rebuild
    old = _chain([[0, 1]], 3); new = _chain([[2, 0]], 4)  # a different start state, one more step
    assert chained_episode(old) and chained_episode(new) and chain_rebuild(old, new)
    plan, real = _episode(old); plan2, real2 = _episode(new)
    original = {'concrete_plan': plan, 'realization': real}
    ok = {'repair_type': 'ambiguity', 'repaired_plan': plan2, 'repaired_realization': real2}
    validate_repair_scope(original, ok)
    broken = deepcopy(new); broken[2]['input'] = [[7, 7]]  # no longer one chain
    plan3, real3 = _episode(broken)
    with pytest.raises(Exception, match='at most one training pair'):
        validate_repair_scope(original, {'repair_type': 'ambiguity', 'repaired_plan': plan3, 'repaired_realization': real3})
    loose = [{'input': [[i]], 'output': [[i + 2]]} for i in range(3)]  # not a chain: a full rebuild stays refused
    plan4, real4 = _episode(loose); plan5, real5 = _episode([{'input': [[i]], 'output': [[i + 2]]} for i in range(5, 8)])
    with pytest.raises(Exception, match='at most one training pair'):
        validate_repair_scope({'concrete_plan': plan4, 'realization': real4},
                              {'repair_type': 'ambiguity', 'repaired_plan': plan5, 'repaired_realization': real5})
    assert not chain_rebuild(old, _chain([[2, 0]], 6))  # at most one more pair


def test_slot_versions_of_the_arms_share_their_grid_checks():
    g = lambda rows: [list(r) for r in rows]
    task = {'train': [{'input': g(['10', '00']), 'output': g(['11', '00'])}, {'input': g(['11', '00']), 'output': g(['11', '10'])}],
            'test': [{'input': g(['11', '10']), 'output': g(['11', '11'])}]}
    c = properties.checks(task)
    assert c['CHAINED'] == c['STAGES'] is True and c['MINIMAL'] == c['CONTRAST'] and c['DISTRIBUTED'] == c['SPREAD'] is False
    rep = properties.report([{'complete_task': task, 'study': {'primary_arm': 'SEE', 'idea_modifiers': ['CHAINED', 'KEY']}}])
    assert rep['by_property']['CHAINED'] == {'verified': 1, 'pass': 1} and rep['by_property']['KEY'] == {'verified': 1}
    assert 'CHAINED' not in rep['format_checks_all_verified']  # aliases are not counted twice


def _supply(wid, account, wave):
    return {'id': wid, 'account': account, 'kind': 'fresh_supply', 'reference_wave': wave}


def _queue(root, w, ids):
    path = waves.queue_path(root, w); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'rows': [{'source_candidate_id': k, 'kernel': {'central_relation': 'Relation of ' + k}} for k in ids]}))


def test_reference_waves_block_then_freeze_the_earlier_selections(tmp_path):
    from arcgen.batch.retrieval import ReferenceIndex
    from support.fixtures import Vectors, references
    a1, a2, b1 = _supply('w1', 'main', 'A'), _supply('w2', 'main', 'A'), _supply('w3', 'main', 'B')
    ws = [a1, a2, b1]
    assert not waves.blocked(tmp_path, ws, a1) and waves.blocked(tmp_path, ws, b1)
    _queue(tmp_path, a1, ['k1', 'k2'])
    assert waves.blocked(tmp_path, ws, b1)
    _queue(tmp_path, a2, ['k3'])
    assert not waves.blocked(tmp_path, ws, b1)
    base = ReferenceIndex(references(), Vectors(), provider_identity={'synthetic': True})
    assert waves.index(tmp_path, ws, a1, base) is base
    idx = waves.index(tmp_path, ws, b1, base)
    assert [r['reference_id'] for r in idx.batch_references] == ['batch:k1', 'batch:k2', 'batch:k3']
    assert set(idx.records) == set(base.records) | {'batch:k1', 'batch:k2', 'batch:k3'}
    assert idx.snapshot_hash == digest(sorted(references() + idx.batch_references, key=lambda r: r['reference_id']))
    again = waves.index(tmp_path, ws, b1, base)  # frozen: a resumed run sees the same references
    assert again.batch_references == idx.batch_references
    _queue(tmp_path, a2, ['k3', 'k4'])  # an earlier queue that changed after the freeze is refused
    with pytest.raises(IntegrityError):
        waves.references(tmp_path, ws, b1)
    near = idea_v1.neighbours(idx, [{'title': 'Relation of k1', 'idea': 'x', 'recipe': 'y'}])
    assert all(r['text'].startswith('(selected earlier in this batch)') for r in near[0] if r['reference_id'].startswith('batch:'))


def test_later_wave_waits_for_earlier_selection_and_sees_it(tmp_path, monkeypatch):
    from arcgen.lib.factory_runtime import FactoryModelInvoker
    from arcgen.batch.allocation import design
    from arcgen.batch.budget import Budget
    from arcgen.batch.common import account_pool
    from arcgen.astra_usage_diversity_20260913 import selection
    from arcgen.astra_usage_diversity_20260913.common import read
    from support.bank import prepare
    from arcgen.batch.retrieval import ReferenceIndex
    from support.fixtures import Vectors
    from test_v1_ideas import _sketches, _commit
    monkeypatch.setattr(scheduler, 'CONTINUOUS', True)
    monkeypatch.setattr(scheduler, 'POLICY', {**scheduler.POLICY, 'continuous_supply': True, 'supply_per_account': 2,
                                              'group_preps_per_account': 4, 'ready_per_account': scheduler.POLICY['downstream_per_account'],
                                              'supply_plus_ready_per_account': 2 + scheduler.POLICY['downstream_per_account']})
    inputs = prepare(tmp_path)
    windows, requests = [], {}
    for n, (wid, account, wave) in enumerate((('one-w1', 'main', 'A'), ('two-w2', 'main', 'B'))):
        call = supply.make_call(arm='SEM', bank_setting='loop', context='c01', seed=777 + n, account=account, window=wid,
                                idea_model='gpt-6.1-sol', effort='high')
        windows.append({'id': wid, 'phase': 'screen', 'account': account, 'context': 'c01', 'seed': 777 + n, 'calls': [call],
                        'maximum_developments': 0, 'minimum_selector_eligible_kernels': 1, 'kind': 'fresh_supply', 'supply_only': True,
                        'editorial_mode': 'bypass', 'protected': False, 'pairing_index': n, 'reference_wave': wave})
        requests[wid] = {call['opaque_id']: supply.idea_request(call, None, inputs)}
    slow = windows[0]['calls'][0]['opaque_id']  # without the wave wait, the B window would finish first
    cap = design()['limits']; cap['phases'][0]['phase'] = 'screen'
    budget = Budget(tmp_path, cap)
    index = ReferenceIndex(read(inputs['selector_context'])['references'], Vectors(), provider_identity={'synthetic': True})

    def fake(invoker, stage, cid):
        req = read(invoker.calls_dir / stage / cid / 'request.json')
        if stage == idea_v1.STAGE_SKETCH:
            if cid.startswith(slow):
                time.sleep(2.0)
            return json.dumps(_sketches())
        if stage == idea_v1.STAGE_COMMIT:
            return json.dumps(_commit())
        if stage == selection.STAGE:
            ids = req['parent_ids']
            return json.dumps({'selector_id': cid, 'window_observations': None,
                'candidate_assessments': [{'candidate_id': k, 'eligibility': 'plausible', 'conceptual_potential': 'solid',
                    'coherence_potential': 'solid', 'fair_realization_potential': 'plausible', 'redundant_with': [],
                    'same_procedure_family_with': [], 'bank_comparisons': [], 'useful_contribution': 'Fresh insight ' + k,
                    'reason': 'Coherent synthetic relation.'} for k in ids],
                'ranked_plausible': [{'candidate_id': k, 'development_priority': i + 1, 'development_brief': 'Realize fairly.',
                    'optional_development_suggestion': None} for i, k in enumerate(ids)]})
        raise AssertionError(stage)

    monkeypatch.setattr(FactoryModelInvoker, '_mock_response', fake)
    pools = {a: account_pool(tmp_path / 'accounts' / a / 'worker_pool.json') for a in ('main',)}
    try:
        scheduler.run_windows(tmp_path, 'screen', windows, requests, inputs, index=index, pools=pools, budget=budget,
                              production=inputs['production'], manifest={'fixture': 'waves'}, mock=True)
    finally:
        for pool in pools.values(): pool.finish()
    events = [json.loads(l) for l in (tmp_path / 'timing/screen.events.jsonl').read_text().splitlines()]
    a_selected = next(i for i, e in enumerate(events) if e.get('kind') == 'stage_finished' and e.get('stage') == 'selector'
                      and e.get('window') == 'one-w1')
    b_ideas = next(i for i, e in enumerate(events) if e.get('kind') == 'stage_started' and e.get('stage') == 'ideas'
                   and e.get('window') == 'two-w2')
    assert a_selected < b_ideas  # wave B started only after wave A's selection was frozen
    frozen_refs = read(tmp_path / 'design/waves/B/batch_references.private.json')['references']
    a_queue = read(tmp_path / 'accounts/main/one-w1/eligible_queue.private.json')['rows']
    assert [r['reference_id'] for r in frozen_refs] == ['batch:' + r['source_candidate_id'] for r in a_queue] and frozen_refs
    packed = read(tmp_path / 'accounts/main/two-w2/selector_inputs.private.json')
    base = read(inputs['selector_context'])['references']
    assert packed['private_retrieval']['snapshot_hash'] == digest(sorted(base + frozen_refs, key=lambda r: r['reference_id']))


def test_scheduler_spec_tolerates_a_new_pool_size_only(tmp_path):
    path = tmp_path / 'phases/x.scheduler.private.json'; path.parent.mkdir(parents=True)
    spec = {'phase': 'x', 'policy': {'downstream_per_account': 5, 'ready_per_account': 5, 'supply_plus_ready_per_account': 8,
                                     'provider_pool': 'native_16_downstream_7_upstream_23_total', 'supply_per_account': 3}, 'windows': [1]}
    scheduler._freeze_spec(path, spec)
    bigger = {**spec, 'policy': {**spec['policy'], 'downstream_per_account': 24, 'ready_per_account': 24,
                                 'supply_plus_ready_per_account': 27, 'provider_pool': 'native_22_downstream_8_upstream_30_total'}}
    scheduler._freeze_spec(path, bigger)
    assert (tmp_path / 'phases/x.policy_history.jsonl').read_text().count('\n') == 1
    with pytest.raises(IntegrityError):
        scheduler._freeze_spec(path, {**bigger, 'policy': {**bigger['policy'], 'supply_per_account': 4}})
    with pytest.raises(IntegrityError):
        scheduler._freeze_spec(path, {**spec, 'windows': [2]})


def test_pools_resize_on_resume_only_when_allowed(tmp_path):
    from arcgen.lib.e18_pre_e20_pool import DedicatedCallPool, DedicatedPoolError
    path = tmp_path / 'pool.json'
    DedicatedCallPool(path, downstream_workers=16, upstream_workers=7, global_ceiling=23, codex_cap=23).finish()
    with pytest.raises(DedicatedPoolError):
        DedicatedCallPool(path, downstream_workers=22, upstream_workers=8, global_ceiling=30, codex_cap=30)
    pool = DedicatedCallPool(path, downstream_workers=22, upstream_workers=8, global_ceiling=30, codex_cap=30, allow_resize=True)
    pool.finish()
    state = json.loads(path.read_text())
    assert state['downstream_workers'] == 22 and any(e['kind'] == 'resume_resized' for e in state['events'])


def test_worker_split_and_estimate():
    shape = cli.split(30, 1)
    assert shape == {'pool': {'downstream_workers': 22, 'upstream_workers': 8}, 'development_windows': 24}
    assert cli.split(30, 3)['development_windows'] == 8
    spec = json.loads(cli.TEMPLATE_SPEC.read_text())
    e = cli.estimate(spec['rows'], spec['windows'], 30)
    assert e['expected_verified_tasks'] == round(56 * cli.VERIFIED_PER_CALL) and 6 <= e['hours'] <= 9
    assert e['model_calls'] == e['expected_verified_tasks'] * cli.CALLS_PER_VERIFIED and e['memory_gb'] == 7.8


def test_reference_spec_has_waves_and_every_arm_with_a_card():
    spec = json.loads(cli.TEMPLATE_SPEC.read_text())
    assert {w['reference_wave'] for w in spec['windows'].values()} == {'A', 'B'}
    assert 'SPREAD' in {r[1] for r in spec['rows']}


def test_scaled_specs_keep_reference_waves_in_order_and_design_checks_it(monkeypatch):
    spec = json.loads(cli.TEMPLATE_SPEC.read_text())
    s = cli.scale_spec(spec, 3)
    assert len(s['rows']) == 3 * len(spec['rows'])
    for account in ('main',):
        order = [w['reference_wave'] for w in s['windows'].values() if w['account'] == account]
        assert order == sorted(order) and set(order) == {'A', 'B'}
    from arcgen.batch import allocation
    proposal = {'revision': 'x', 'windows': {'w1': {'account': 'main', 'wave': 'A', 'context': 'c01', 'reference_wave': 'B'},
                                             'w2': {'account': 'main', 'wave': 'A', 'context': 'c01', 'reference_wave': 'A'}}}
    calls = [{'window': w, 'account': 'main', 'wave': 'A', 'context': 'c01', 'seed': 1, 'arm': 'SEE', 'invitation': None,
              'bank_setting': 'loop', 'effort': 'high', 'model': 'gpt-6.1-sol', 'commit': 'explore', 'private_id': 'call-' + w,
              'candidate_ids': [f'k{w}{i}' for i in range(6)]} for w in ('w1', 'w2')]
    proposal['calls'] = calls; proposal['coverage'] = {}
    monkeypatch.setattr(allocation, 'read', lambda path: {**proposal, 'revision': allocation.REVISION})
    monkeypatch.setitem(config.SCIENCE, 'modifier_assignment', 'per_call')
    with pytest.raises(IntegrityError, match='reference-wave order'):
        allocation.design()


def test_mixed_wave_order_finishes_with_the_later_wave_last(tmp_path, monkeypatch):
    """Per-account order A, B, A (an old `--size 2+` layout): the blocked B window lets the A window behind it go
    first, then runs. Before the fix this spun forever with only the first window submitted (code review 2026-10-09)."""
    from arcgen.lib.factory_runtime import FactoryModelInvoker
    from arcgen.batch.allocation import design
    from arcgen.batch.budget import Budget
    from arcgen.batch.common import account_pool
    from arcgen.astra_usage_diversity_20260913 import selection
    from arcgen.astra_usage_diversity_20260913.common import read
    from support.bank import prepare
    from arcgen.batch.retrieval import ReferenceIndex
    from support.fixtures import Vectors
    from test_v1_ideas import _sketches, _commit
    monkeypatch.setattr(scheduler, 'CONTINUOUS', True)
    monkeypatch.setattr(scheduler, 'POLICY', {**scheduler.POLICY, 'continuous_supply': True, 'supply_per_account': 1,
                                              'group_preps_per_account': 4, 'ready_per_account': scheduler.POLICY['downstream_per_account'],
                                              'supply_plus_ready_per_account': 1 + scheduler.POLICY['downstream_per_account']})
    inputs = prepare(tmp_path)
    windows, requests = [], {}
    for n, (wid, wave) in enumerate((('one-w1', 'A'), ('one-w2', 'B'), ('one-w3', 'A'))):
        call = supply.make_call(arm='SEM', bank_setting='loop', context='c01', seed=777 + n, account='main', window=wid,
                                idea_model='gpt-6.1-sol', effort='high')
        windows.append({'id': wid, 'phase': 'screen', 'account': 'main', 'context': 'c01', 'seed': 777 + n, 'calls': [call],
                        'maximum_developments': 0, 'minimum_selector_eligible_kernels': 1, 'kind': 'fresh_supply', 'supply_only': True,
                        'editorial_mode': 'bypass', 'protected': False, 'pairing_index': n, 'reference_wave': wave})
        requests[wid] = {call['opaque_id']: supply.idea_request(call, None, inputs)}
    cap = design()['limits']; cap['phases'][0]['phase'] = 'screen'
    budget = Budget(tmp_path, cap)
    index = ReferenceIndex(read(inputs['selector_context'])['references'], Vectors(), provider_identity={'synthetic': True})

    def fake(invoker, stage, cid):
        req = read(invoker.calls_dir / stage / cid / 'request.json')
        if stage == idea_v1.STAGE_SKETCH:
            return json.dumps(_sketches())
        if stage == idea_v1.STAGE_COMMIT:
            return json.dumps(_commit())
        if stage == selection.STAGE:
            ids = req['parent_ids']
            return json.dumps({'selector_id': cid, 'window_observations': None,
                'candidate_assessments': [{'candidate_id': k, 'eligibility': 'plausible', 'conceptual_potential': 'solid',
                    'coherence_potential': 'solid', 'fair_realization_potential': 'plausible', 'redundant_with': [],
                    'same_procedure_family_with': [], 'bank_comparisons': [], 'useful_contribution': 'Fresh insight ' + k,
                    'reason': 'Coherent synthetic relation.'} for k in ids],
                'ranked_plausible': [{'candidate_id': k, 'development_priority': i + 1, 'development_brief': 'Realize fairly.',
                    'optional_development_suggestion': None} for i, k in enumerate(ids)]})
        raise AssertionError(stage)

    monkeypatch.setattr(FactoryModelInvoker, '_mock_response', fake)
    pools = {a: account_pool(tmp_path / 'accounts' / a / 'worker_pool.json') for a in ('main',)}
    done = []
    t = threading.Thread(target=lambda: done.append(scheduler.run_windows(tmp_path, 'screen', windows, requests, inputs, index=index,
        pools=pools, budget=budget, production=inputs['production'], manifest={'fixture': 'waves'}, mock=True)), daemon=True)
    t.start(); t.join(120)
    if not done:
        (tmp_path / 'STOP_DISPATCH.json').write_text('{}'); t.join(10)
    for pool in pools.values(): pool.finish()
    events = [json.loads(l) for l in (tmp_path / 'timing/screen.events.jsonl').read_text().splitlines()]
    submitted = [e['window'] for e in events if e.get('kind') == 'supply_submitted']
    assert done and submitted == ['one-w1', 'one-w3', 'one-w2']
