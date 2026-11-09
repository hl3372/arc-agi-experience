"""Scheduling: saved windows open first, idea windows fill idle slots at the start, development takes the idea slots once an account's idea
work has ended, kernels link to tasks under any planner, and a degenerate test pair fails before verification."""
import json
import time
from types import SimpleNamespace
import pytest
from arcgen.batch import cli, scheduler, supply, idea_v1
from arcgen.batch import runtime as batch_runtime


def _fake(slow=None, plan_seconds=0.0):
    """Mock model responses: idea loop, selector (all plausible), and plans that abort after plan_seconds."""
    from arcgen.lib import astra_authoring_schemas_v3 as s
    from arcgen.astra_usage_diversity_20260913 import selection
    from arcgen.astra_usage_diversity_20260913.common import read
    from support.fixtures import episode
    from test_v1_ideas import _sketches, _commit
    from test_carry_over import _selector_response
    slow = slow or {}

    def fake(invoker, stage, cid):
        req = read(invoker.calls_dir / stage / cid / 'request.json')
        if stage == idea_v1.STAGE_SKETCH:
            for key, seconds in slow.items():
                if cid.startswith(key):
                    time.sleep(seconds)
            return json.dumps(_sketches())
        if stage == idea_v1.STAGE_COMMIT:
            return json.dumps(_commit())
        if stage == selection.STAGE:
            return json.dumps(_selector_response(req['parent_ids'], cid))
        tid = req['parent_ids'][0]; plan, _ = episode(tid)
        plan.update(status='abort', abort_reason='Concrete unresolved semantic defect in this test.', public_contract=None, private_execution=None)
        time.sleep(plan_seconds)
        if stage == s.PLAN_STAGE: return json.dumps(plan)
        if stage == s.SCALE_STAGE:
            return json.dumps({'candidate_id': tid, 'scale_decision': 'abort', 'scale_reason': plan['abort_reason'], 'concrete_plan': None})
        raise AssertionError(stage)
    return fake


def _run(tmp_path, monkeypatch, windows, requests, inputs, accounts=('main',), **policy):
    from arcgen.lib.factory_runtime import FactoryModelInvoker
    from arcgen.batch.allocation import design
    from arcgen.batch.budget import Budget
    from arcgen.batch.common import account_pool
    from arcgen.astra_usage_diversity_20260913.common import read
    from arcgen.batch.retrieval import ReferenceIndex
    from support.fixtures import Vectors
    monkeypatch.setattr(scheduler, 'CONTINUOUS', True)
    supply_n = policy.get('supply_per_account', 1)
    down = policy.pop('downstream_per_account', scheduler.POLICY['downstream_per_account'])
    monkeypatch.setattr(scheduler, 'POLICY', {**scheduler.POLICY, 'continuous_supply': True, 'supply_per_account': supply_n,
                                              'group_preps_per_account': 4, 'downstream_per_account': down, 'ready_per_account': down,
                                              'supply_plus_ready_per_account': supply_n + down, **policy})
    cap = design()['limits']; cap['phases'][0]['phase'] = 'screen'
    budget = Budget(tmp_path, cap)
    index = ReferenceIndex(read(inputs['selector_context'])['references'], Vectors(), provider_identity={'synthetic': True})
    pools = {a: account_pool(tmp_path / 'accounts' / a / 'worker_pool.json') for a in accounts}
    try:
        scheduler.run_windows(tmp_path, 'screen', windows, requests, inputs, index=index, pools=pools, budget=budget,
                              production=inputs['production'], manifest={'fixture': 'b15'}, mock=True)
    finally:
        for pool in pools.values(): pool.finish()
    return [json.loads(l) for l in (tmp_path / 'timing/screen.events.jsonl').read_text().splitlines()]


def _fresh(inputs, wid, account, n, wave=None):
    call = supply.make_call(arm='SEM', bank_setting='loop', context='c01', seed=777 + n, account=account, window=wid,
                            idea_model='gpt-6.1-sol', effort='high')
    w = {'id': wid, 'phase': 'screen', 'account': account, 'context': 'c01', 'seed': 777 + n, 'calls': [call],
         'maximum_developments': 0, 'minimum_selector_eligible_kernels': 1, 'kind': 'fresh_supply', 'supply_only': True,
         'editorial_mode': 'bypass', 'protected': False, 'pairing_index': n}
    if wave:
        w['reference_wave'] = wave
    return w, {call['opaque_id']: supply.idea_request(call, None, inputs)}


def _groups(source, count, account=None):
    return [{'id': f'{source}-develop-{i}', 'phase': 'screen', 'account': account, 'source_account': 'main', 'context': 'c01',
             'seed': 900 + i, 'calls': [], 'maximum_developments': 1, 'kind': 'eligible_group', 'source_window': source,
             'group_ordinal': i, 'editorial_mode': 'bypass', 'research_calls_per_account': {'main': 0}}
            for i in range(count)]


def _index(events, kind, window=None, stage=None):
    return next(i for i, e in enumerate(events) if e.get('kind') == kind and (window is None or e.get('window') == window)
                and (stage is None or e.get('stage') == stage))


def test_saved_windows_open_before_idea_windows(tmp_path, monkeypatch):
    """Frozen order w1 (A), w2 (B), s1 (saved, A), s2 (saved, B), one supply window at a time: the saved wave-A window
    goes first; the saved wave-B window waits for wave A like any window of its wave, then goes before the idea window."""
    from support.bank import prepare
    from test_carry_over import _saved
    inputs = prepare(tmp_path)
    w1, r1 = _fresh(inputs, 'one-w1', 'main', 0, 'A')
    w2, r2 = _fresh(inputs, 'one-w2', 'main', 1, 'B')
    s1, _ = _saved(tmp_path, inputs, 'one-saved-01', wave='A')
    s2, _ = _saved(tmp_path, inputs, 'one-saved-02', wave='B')
    s1['pairing_index'], s2['pairing_index'] = 2, 3
    monkeypatch.setattr('arcgen.lib.factory_runtime.FactoryModelInvoker._mock_response', _fake())
    events = _run(tmp_path, monkeypatch, [w1, w2, s1, s2], {'one-w1': r1, 'one-w2': r2, 'one-saved-01': {}, 'one-saved-02': {}},
                  inputs)
    submitted = [e['window'] for e in events if e.get('kind') == 'supply_submitted']
    assert submitted == ['one-saved-01', 'one-w1', 'one-saved-02', 'one-w2']


def test_idle_accounts_open_extra_idea_windows_until_development_starts(tmp_path, monkeypatch):
    """One idea window normally, three while the account has no development: w1-w3 start together; once w1's groups
    develop (slowly), w4 waits until the account is back under one window, i.e. until w2 and w3 have selected."""
    from support.bank import prepare
    inputs = prepare(tmp_path)
    windows, requests = [], {}
    for n in range(4):
        w, r = _fresh(inputs, f'one-w{n + 1}', 'main', n)
        windows.append(w); requests[w['id']] = r
    windows += _groups('one-w1', 2)
    requests.update({g['id']: {} for g in windows[4:]})
    slow = {windows[1]['calls'][0]['opaque_id']: 5.0, windows[2]['calls'][0]['opaque_id']: 5.0}
    monkeypatch.setattr('arcgen.lib.factory_runtime.FactoryModelInvoker._mock_response', _fake(slow, plan_seconds=12.0))
    events = _run(tmp_path, monkeypatch, windows, requests, inputs, accounts=('main',), idle_supply_windows_per_account=3)
    assert _index(events, 'supply_submitted', 'one-w3') < _index(events, 'stage_finished', 'one-w1', 'ideas')
    first_development = _index(events, 'development_submitted', 'one-w1-develop-0')
    w4 = _index(events, 'supply_submitted', 'one-w4')
    assert first_development < w4
    assert w4 > _index(events, 'stage_finished', 'one-w2', 'selector') and w4 > _index(events, 'stage_finished', 'one-w3', 'selector')
    counts = [e for e in events if e.get('kind') == 'window_counts']
    assert max(e['supply'] for e in counts) == 3


def test_development_takes_idea_slots_once_idea_work_ends(tmp_path, monkeypatch):
    """One development window per account while ideas run; after the account's last selection its pool lends the idea
    slots and its development windows grow by the idea workers, so w1's three groups develop together."""
    from support.bank import prepare
    from arcgen.astra_usage_diversity_20260913.common import read
    inputs = prepare(tmp_path)
    w1, r1 = _fresh(inputs, 'one-w1', 'main', 0)
    groups = _groups('one-w1', 3)
    monkeypatch.setattr('arcgen.lib.factory_runtime.FactoryModelInvoker._mock_response', _fake(plan_seconds=2.0))
    events = _run(tmp_path, monkeypatch, [w1, *groups], {'one-w1': r1, **{g['id']: {} for g in groups}}, inputs,
                  accounts=('main',), downstream_per_account=1)
    lent = _index(events, 'idea_slots_to_development')
    assert lent > _index(events, 'stage_finished', 'one-w1', 'selector')
    assert max(e['downstream'] for e in events if e.get('kind') == 'window_counts') >= 3
    pool = read(tmp_path / 'accounts/main/worker_pool.json')
    assert any(e['kind'] == 'upstream_closed' for e in pool['events'])


def test_pool_lends_idea_slots_to_development_only_after_closing():
    import tempfile
    from pathlib import Path
    from arcgen.lib.e18_pre_e20_pool import DedicatedCallPool
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'pool.json'
        pool = DedicatedCallPool(path, downstream_workers=2, upstream_workers=1, global_ceiling=3, codex_cap=3)
        for i in range(2):
            with pool.call_context('downstream', f'd{i}'): pool.acquire('downstream')
        assert pool._can_acquire('downstream') == (False, False) and pool._can_acquire('selection_codex')[0]
        pool.close_upstream(reason='test')
        assert pool._can_acquire('downstream') == (True, False)
        with pool.call_context('downstream', 'd2'): pool.acquire('downstream')
        state = pool.snapshot()
        assert state['peak_downstream'] == 3 and sum(bool(c.get('upstream_slot')) for c in state['active_calls'].values()) == 1
        assert not pool._can_acquire('selection_codex')[0]  # every slot is busy
        for i in range(3):
            with pool.call_context('downstream', f'd{i}'): pool.release('downstream')
        pool.finish()
        again = DedicatedCallPool(path, downstream_workers=2, upstream_workers=1, global_ceiling=3, codex_cap=3)
        assert 'upstream_closed' not in again.state  # a new session closes it again itself
        again.finish()


def test_new_batches_size_idle_idea_windows_to_fill_the_slots(monkeypatch):
    monkeypatch.setattr(cli, 'mem_available_mb', lambda: 64000)
    ops, total, _ = cli.concurrency(1, workers=24, windows=15)
    assert total == 24 and ops['supply_windows'] == 3 and ops['idle_supply_windows'] == 6
    ops, _, _ = cli.concurrency(1, workers=24, windows=4)
    assert ops['idle_supply_windows'] == 4  # never more than the batch's windows
    from arcgen.batch.config import validate
    value = json.loads(cli.TEMPLATE_PROFILE.read_text()); value['operations'].update(ops)
    validate(value)
    value['operations']['idle_supply_windows'] = 1
    with pytest.raises(Exception, match='idle_supply_windows'):
        validate(value)


def test_kernels_link_to_their_task_under_any_planner():
    from arcgen.batch.batch_report import link_kernels
    kernels = [{'kernel_id': k, 'admitted': True} for k in ('k1', 'k2', 'k3')]
    tasks = [{'kernel_id': 'k1', 'task_id': 't1', 'planner': 'sol'},
             {'kernel_id': 'k2', 'task_id': 't2b', 'planner': 'sol_max'}, {'kernel_id': 'k2', 'task_id': 't2a', 'planner': 'astra'}]
    out = {k['kernel_id']: k for k in link_kernels(kernels, tasks)}
    assert out['k1']['task_id'] == 't1' and out['k2']['task_id'] == 't2a'  # the Astra version of a triplet, as before
    assert out['k3']['queued_unstarted'] and not out['k1']['queued_unstarted'] and out['k1']['development_started']


def test_degenerate_test_pairs():
    g = lambda *rows: [list(map(int, r)) for r in rows]
    move = {'input': g('100', '000'), 'output': g('010', '000')}
    task = lambda *tests, train=(move,): {'train': list(train), 'test': list(tests)}
    assert batch_runtime.degenerate_tests(task({'input': g('001', '000'), 'output': g('000', '001')})) == []
    same = {'input': g('001', '000'), 'output': g('001', '000')}
    assert batch_runtime.degenerate_tests(task(same)) == ['test 1 answer equals its input and no training pair leaves its input unchanged']
    assert batch_runtime.degenerate_tests(task(same, train=(move, {'input': g('001'), 'output': g('001')}))) == []  # a shown no-op
    blank = {'input': g('120', '000'), 'output': g('000', '000')}
    assert batch_runtime.degenerate_tests(task(move, blank)) == [
        'test 2 answer is blank (the input cleared to 0, its most common colour)']
    assert batch_runtime.degenerate_tests(task({'input': g('120', '000'), 'output': g('00', '00')})) == []  # smaller: legal
    assert batch_runtime.degenerate_tests(task({'input': g('120', '000'), 'output': g('22')})) == []  # one colour, not the ground
    v1 = SimpleNamespace(q2=SimpleNamespace(prompt_set='v1')); r23 = SimpleNamespace(q2=SimpleNamespace(prompt_set='r23'))
    assert batch_runtime.Runtime.episode_defect(v1, task(same)).startswith('degenerate test: test 1 answer equals')
    assert batch_runtime.Runtime.episode_defect(r23, task(same)) == ''


def test_a_degenerate_test_fails_at_realization_before_any_verification(tmp_path, monkeypatch):
    import sys, pathlib
    from support.pilot import run_case
    from arcgen.batch import config
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    monkeypatch.setattr(batch_runtime, 'degenerate_tests', lambda task: ['test 1 answer is blank (fixture)'])
    first, calls, *_ = run_case(tmp_path, monkeypatch, 'clear', 2, False, arm='H_DISC', planner='sol')  # plan and realization only
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert record['status'] == 'realization_failure' and record['realization_error'] == 'degenerate test: test 1 answer is blank (fixture)'
    assert not record.get('technical_results') and not record.get('blind_results')


def test_colourful_ideas_keep_one_colouring_but_may_turn():
    """COLOURFUL fixes each colour job to one colour (v1-b15 screen), so it never draws per-pair colour variation; pose stays."""
    from copy import deepcopy
    from arcgen.batch import admission
    rows = [{'source_candidate_id': f'k{i:04d}', 'private_source': 'ordinary_unconditioned',
             'study': {'primary_arm': 'SEE', **({'idea_modifiers': ['COLOURFUL', 'KEY']} if i % 2 else {})}} for i in range(400)]
    routed = admission.configured_routes({}, deepcopy(rows))
    colourful = [r for r in routed if 'COLOURFUL' in r['study'].get('idea_modifiers', ())]
    others = [r for r in routed if not r['study'].get('idea_modifiers')]
    assert len(colourful) == len(others) == 200
    assert all(r['study']['colour_roles'] == 'v1' for r in colourful) and any(r['study']['pose'] for r in colourful)
    assert {r['study']['colour_roles'] for r in others} == {'vary', 'v1'}
