"""Continuous supply: development groups start as soon as their own window's selection is frozen, while later idea
windows are still running (no barrier behind the whole supply phase)."""
import json
import threading
import time
from arcgen.lib.factory_runtime import FactoryModelInvoker
from arcgen.lib import astra_authoring_schemas_v3 as s
from arcgen.batch import scheduler, supply, idea_v1
from arcgen.batch.allocation import design
from arcgen.batch.budget import Budget
from arcgen.batch.common import account_pool
from arcgen.astra_usage_diversity_20260913 import selection
from arcgen.astra_usage_diversity_20260913.common import read
from support.bank import prepare
from arcgen.batch.retrieval import ReferenceIndex
from support.fixtures import Vectors
from support.fixtures import episode
from test_v1_ideas import _sketches, _commit


def test_development_overlaps_later_idea_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, 'CONTINUOUS', True)
    monkeypatch.setattr(scheduler, 'POLICY', {**scheduler.POLICY, 'continuous_supply': True, 'supply_per_account': 1,
                                              'group_preps_per_account': 4, 'ready_per_account': scheduler.POLICY['downstream_per_account'],
                                              'supply_plus_ready_per_account': 1 + scheduler.POLICY['downstream_per_account']})
    inputs = prepare(tmp_path)
    windows, requests = [], {}
    for n, (wid, account) in enumerate((('one-w1', 'main'), ('one-w2', 'main'), ('two-w1', 'main'))):
        call = supply.make_call(arm='SEM', bank_setting='loop', context='c01', seed=777 + n, account=account, window=wid,
                                idea_model='gpt-6.1-sol', effort='high')
        windows.append({'id': wid, 'phase': 'screen', 'account': account, 'context': 'c01', 'seed': 777 + n, 'calls': [call],
                        'maximum_developments': 0, 'minimum_selector_eligible_kernels': 1, 'kind': 'fresh_supply', 'supply_only': True,
                        'editorial_mode': 'bypass', 'protected': False, 'pairing_index': n})
        requests[wid] = {call['opaque_id']: supply.idea_request(call, None, inputs)}
    # both accounts stay busy with slower idea windows, as in a real batch: the first window's groups have nowhere idle to go
    slow = {windows[1]['calls'][0]['opaque_id']: 2.0, windows[2]['calls'][0]['opaque_id']: 4.0}
    for i in range(2):  # development groups of the first window only
        gid = f'one-w1-develop-{i}'
        windows.append({'id': gid, 'phase': 'screen', 'account': None, 'source_account': 'main', 'context': 'c01',
                        'seed': 900 + i, 'calls': [], 'maximum_developments': 3, 'kind': 'eligible_group', 'source_window': 'one-w1',
                        'group_ordinal': i, 'editorial_mode': 'bypass', 'research_calls_per_account': {'main': 0}})
        requests[gid] = {}
    cap = design()['limits']; cap['phases'][0]['phase'] = 'screen'
    budget = Budget(tmp_path, cap)
    index = ReferenceIndex(read(inputs['selector_context'])['references'], Vectors(), provider_identity={'synthetic': True})
    lock = threading.Lock()

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
            ids = req['parent_ids']
            return json.dumps({'selector_id': cid, 'window_observations': None,
                'candidate_assessments': [{'candidate_id': k, 'eligibility': 'plausible', 'conceptual_potential': 'solid',
                    'coherence_potential': 'solid', 'fair_realization_potential': 'plausible', 'redundant_with': [],
                    'same_procedure_family_with': [], 'bank_comparisons': [], 'useful_contribution': 'Fresh insight ' + k,
                    'reason': 'Coherent synthetic relation.'} for k in ids],
                'ranked_plausible': [{'candidate_id': k, 'development_priority': i + 1, 'development_brief': 'Realize fairly.',
                    'optional_development_suggestion': None} for i, k in enumerate(ids)]})
        tid = req['parent_ids'][0]; plan, _ = episode(tid)
        plan.update(status='abort', abort_reason='Concrete unresolved semantic defect in this test.', public_contract=None, private_execution=None)
        if stage == s.PLAN_STAGE: return json.dumps(plan)
        if stage == s.SCALE_STAGE:
            return json.dumps({'candidate_id': tid, 'scale_decision': 'abort', 'scale_reason': plan['abort_reason'], 'concrete_plan': None})
        raise AssertionError(stage)

    monkeypatch.setattr(FactoryModelInvoker, '_mock_response', fake)
    pools = {a: account_pool(tmp_path / 'accounts' / a / 'worker_pool.json') for a in ('main',)}
    try:
        scheduler.run_windows(tmp_path, 'screen', windows, requests, inputs, index=index, pools=pools, budget=budget,
                              production=inputs['production'], manifest={'fixture': 'continuous'}, mock=True)
    finally:
        for pool in pools.values(): pool.finish()
    events = [json.loads(l) for l in (tmp_path / 'timing/screen.events.jsonl').read_text().splitlines()]
    first_group = next(i for i, e in enumerate(events) if e.get('kind') == 'development_submitted' and e.get('window', '').startswith('one-w1-develop'))
    second_ideas = next(i for i, e in enumerate(events) if e.get('kind') == 'stage_finished' and e.get('stage') == 'ideas'
                        and e.get('window') == 'one-w2')
    assert first_group < second_ideas  # development began while the second window was still writing its ideas
    assert sum(e.get('kind') == 'window_terminal' for e in events) == 5  # three idea windows and both groups finished
