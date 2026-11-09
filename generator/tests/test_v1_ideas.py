"""v1 idea stage: sketch -> nearest bank tasks -> commit, through the selector into development; resume is call-free."""
import json
import threading
from arcgen.lib.factory_runtime import FactoryModelInvoker
from arcgen.lib import astra_authoring_schemas_v3 as s
from arcgen.batch import scheduler, supply, idea_v1, config
from arcgen.batch.allocation import design
from arcgen.batch.budget import Budget
from arcgen.batch.common import account_pool, IDEA_MODELS
from arcgen.astra_usage_diversity_20260913 import selection
from arcgen.astra_usage_diversity_20260913.common import read, digest
from support.bank import prepare
from arcgen.batch.retrieval import ReferenceIndex
from support.fixtures import Vectors
from support.fixtures import episode


def test_v1_profile_binds_the_loop_models_and_arms():
    assert config.SCIENCE['idea_stage'] == 'v1_loop' and supply.V1_IDEAS
    assert IDEA_MODELS == ('gpt-6.1-sol', 'gpt-6-sol')
    calls = [c for w in design()['windows'] for c in w['calls']]
    assert len(calls) == 56 and {c['bank_setting'] for c in calls} == {'loop'} and not any(c['invitation'] for c in calls)
    assert {c['arm'] for c in calls} <= set(idea_v1.ARMS)
    # explore windows (the reference design): sketch, explore and commit turns per idea call
    assert all(c.get('commit') == 'explore' for c in calls)
    assert design()['limits']['phases'][0]['categories']['ideas'] == 3 * 56
    req = supply.idea_request(calls[0], None, None)
    assert req['stage'] == idea_v1.STAGE_SKETCH and 'What we are looking for' in req['prompt']
    for banned in ('Calibration descriptions', 'nominate', 'coverage reminder', 'latest supply repeatedly', 'Favor direct visual'):
        assert banned not in req['prompt']


def _sketches(n=14):
    return {'sketches': [{'id': f's{i:02d}', 'title': f'Idea {i}', 'idea': f'A relation number {i}.',
                          'recipe': f'Do step {i} to every region.'} for i in range(1, n + 1)]}


def _commit():
    return {'sketch_verdicts': [{'id': f's{i:02d}', 'verdict': 'drop' if i % 3 else 'keep', 'closest_existing': 'x'} for i in range(1, 15)],
            'kernels': [{'source': 's03' if i < 2 else 'replacement', 'central_relation': f'Committed relation {i}.',
                         'recipe': f'Recipe {i}.', 'what_is_new': f'New insight {i}.', 'why_coherent': 'Coherent.',
                         'possible_test_generalization': 'A new situation.', 'main_authoring_risk': 'Ambiguity.'} for i in range(4)]}


def test_v1_idea_loop_through_selector_and_development(tmp_path, monkeypatch):
    inputs = prepare(tmp_path)
    wid = 'one-w0'
    calls = [supply.make_call(arm=arm, bank_setting='loop', context='c01', seed=777, account='main', window=wid,
                              idea_model=model, effort='xhigh') for arm, model in [('SEM', 'gpt-6.1-sol'), ('PER', 'gpt-6-sol')]]
    windows = [{'id': wid, 'phase': 'screen', 'account': 'main', 'context': 'c01', 'seed': 777, 'calls': calls,
                'maximum_developments': 0, 'minimum_selector_eligible_kernels': 1, 'kind': 'fresh_supply', 'supply_only': True,
                'editorial_mode': 'bypass', 'protected': False, 'pairing_index': 0}]
    requests = {wid: {c['opaque_id']: supply.idea_request(c, None, inputs) for c in calls}}
    for i in range(3):
        gid = f'{wid}-develop-{i}'
        windows.append({'id': gid, 'phase': 'screen', 'account': None, 'source_account': 'main', 'context': 'c01',
                        'seed': 900 + i, 'calls': [], 'maximum_developments': 3, 'kind': 'eligible_group', 'source_window': wid,
                        'group_ordinal': i, 'editorial_mode': 'bypass', 'research_calls_per_account': {'main': 0}})
        requests[gid] = {}
    cap = design()['limits']; cap['phases'][0]['phase'] = 'screen'
    budget = Budget(tmp_path, cap)
    index = ReferenceIndex(read(inputs['selector_context'])['references'], Vectors(), provider_identity={'synthetic': True})
    seen, lock = [], threading.Lock()

    def fake(invoker, stage, cid):
        with lock: seen.append((stage, cid))
        req = read(invoker.calls_dir / stage / cid / 'request.json')
        if stage == idea_v1.STAGE_SKETCH:
            return json.dumps(_sketches())
        if stage == idea_v1.STAGE_COMMIT:
            prompt = (invoker.calls_dir / stage / cid / 'prompt.md').read_text()
            assert 'Most similar existing tasks:' in prompt and prompt.count('### s') == 14
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
            prompt = (invoker.calls_dir / stage / cid / 'prompt.md').read_text()
            assert 'Optional presentation invitation' in prompt
            return json.dumps({'candidate_id': tid, 'scale_decision': 'abort', 'scale_reason': plan['abort_reason'], 'concrete_plan': None})
        raise AssertionError(stage)

    monkeypatch.setattr(FactoryModelInvoker, '_mock_response', fake)

    def run():
        pools = {a: account_pool(tmp_path / 'accounts' / a / 'worker_pool.json') for a in ('main',)}
        try:
            return scheduler.run_windows(tmp_path, 'screen', windows, requests, inputs, index=index, pools=pools, budget=budget,
                                         production=inputs['production'], manifest={'fixture': 'v1-ideas'}, mock=True)
        finally:
            for pool in pools.values(): pool.finish()

    terminal = run()
    source = read(tmp_path / f'accounts/main/{wid}/source_rows.private.json')
    assert len(source) == 8 and all(r['kernel']['search_mode'] == 'v1_loop' and r['kernel']['recipe'] for r in source.values())
    assert {r['private_source'] for r in source.values()} == {'ordinary_unconditioned'}
    assert {r['study']['generation_model'] for r in source.values()} == {'gpt-6.1-sol', 'gpt-6-sol'}
    loops = list((tmp_path / f'accounts/main/{wid}/idea_loop').glob('*.committed.private.json'))
    assert len(loops) == 2
    stages = [st for st, _ in seen]
    assert stages.count(idea_v1.STAGE_SKETCH) == 2 and stages.count(idea_v1.STAGE_COMMIT) == 2 and 'e16_idea_search' not in stages
    assert budget.summary()['developments'] == 8
    baseline = (len(seen), budget.summary()['logical_calls'], digest(terminal))
    assert run() == terminal and (len(seen), budget.summary()['logical_calls'], digest(terminal)) == baseline


def test_kept_draft_commit_takes_only_kept_sketches():
    """Kept-draft calls commit every kept sketch (up to six) and may not invent new ideas."""
    import pytest
    from arcgen.lib.v4_schemas import SchemaError
    verdicts = [{'id': f's{i}', 'verdict': 'keep' if i < 5 else 'drop', 'closest_existing': 'x'} for i in range(14)]
    kernel = lambda src: {'source': src, **{f: 'text' for f in idea_v1.KERNEL_FIELDS if f != 'source'}}
    ok = {'sketch_verdicts': verdicts, 'kernels': [kernel(f's{i}') for i in range(5)]}
    assert len(idea_v1.parse_commit_kept(ok)['kernels']) == 5
    for bad in ([kernel('replacement')], [kernel('s7')], [kernel('s1'), kernel('s1')]):
        with pytest.raises(SchemaError):
            idea_v1.parse_commit_kept({'sketch_verdicts': verdicts, 'kernels': bad})
    call = {'opaque_id': 'p1', 'commit': 'kept', 'opaque_kernel_ids': [f'k{i}' for i in range(6)]}
    value = {'search_call_id': 'p1', 'search_mode': idea_v1.SEARCH_MODE,
             'ideas': [{'candidate_id': f'k{i}', 'central_relation': 'r'} for i in range(3)]}
    idea_v1.validate_ideas(call, value)  # a prefix of the six slots is valid for kept-draft calls


def test_kept_windows_get_six_idea_slots():
    from arcgen.batch.allocations.make_allocation import build
    spec = {'revision': 'r', 'seed': 1, 'windows': {'w1': {'account': 'main', 'wave': 'A', 'context': 'c01'},
                                                   'w2': {'account': 'main', 'wave': 'A', 'context': 'c01', 'commit': 'kept'}},
            'rows': [['w1', 'OPEN', None, 'L', 'sol', 'high'], ['w2', 'OPEN', None, 'L', 'sol', 'high']]}
    out = build(spec, {'sol': 'gpt-6.1-sol'}, {}, lambda arm, inv: True)
    assert [len(c['candidate_ids']) for c in out['calls']] == [4, 6] and out['calls'][1]['commit'] == 'kept'


def _kept_commit(kept=5):
    return {'sketch_verdicts': [{'id': f's{i:02d}', 'verdict': 'keep' if i <= kept else 'drop', 'closest_existing': 'x'} for i in range(1, 15)],
            'kernels': [{'source': f's{i:02d}', 'central_relation': f'Kept relation {i}.', 'recipe': f'Recipe {i}.',
                         'what_is_new': f'New insight {i}.', 'why_coherent': 'Coherent.', 'possible_test_generalization': 'A new situation.',
                         'main_authoring_risk': 'Ambiguity.'} for i in range(1, kept + 1)]}


def test_kept_draft_window_runs_end_to_end(tmp_path, monkeypatch):
    """A window mixing an ordinary call (four ideas) and a kept-draft call (six slots, five kept) runs through ideas,
    selection and development (the v1-b6 failure: the window contract assumed four slots per call)."""
    inputs = prepare(tmp_path)
    wid = 'one-w0'
    calls = [supply.make_call(arm=arm, bank_setting='loop', context='c01', seed=777, account='main', window=wid,
                              idea_model='gpt-6.1-sol', effort='xhigh') for arm in ('SEM', 'OPEN')]
    calls[1]['commit'] = 'kept'
    calls[1]['opaque_kernel_ids'] += ['k' + calls[1]['opaque_id'][1:13] + f'{i:06d}' for i in range(2)]
    windows = [{'id': wid, 'phase': 'screen', 'account': 'main', 'context': 'c01', 'seed': 777, 'calls': calls,
                'maximum_developments': 0, 'minimum_selector_eligible_kernels': 1, 'kind': 'fresh_supply', 'supply_only': True,
                'editorial_mode': 'bypass', 'protected': False, 'pairing_index': 0}]
    requests = {wid: {c['opaque_id']: supply.idea_request(c, None, inputs) for c in calls}}
    for i in range(3):
        gid = f'{wid}-develop-{i}'
        windows.append({'id': gid, 'phase': 'screen', 'account': None, 'source_account': 'main', 'context': 'c01',
                        'seed': 900 + i, 'calls': [], 'maximum_developments': 3, 'kind': 'eligible_group', 'source_window': wid,
                        'group_ordinal': i, 'editorial_mode': 'bypass', 'research_calls_per_account': {'main': 0}})
        requests[gid] = {}
    cap = design()['limits']; cap['phases'][0]['phase'] = 'screen'
    budget = Budget(tmp_path, cap)
    index = ReferenceIndex(read(inputs['selector_context'])['references'], Vectors(), provider_identity={'synthetic': True})
    seen, lock = [], threading.Lock()

    def fake(invoker, stage, cid):
        with lock: seen.append(stage)
        req = read(invoker.calls_dir / stage / cid / 'request.json')
        if stage == idea_v1.STAGE_SKETCH:
            return json.dumps(_sketches())
        if stage == idea_v1.STAGE_COMMIT:
            return json.dumps(_commit())
        if stage == idea_v1.STAGE_COMMIT_KEPT:
            assert 'Do not write new ideas in this step' in (invoker.calls_dir / stage / cid / 'prompt.md').read_text()
            return json.dumps(_kept_commit())
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
                              production=inputs['production'], manifest={'fixture': 'v1-kept'}, mock=True)
    finally:
        for pool in pools.values(): pool.finish()
    source = read(tmp_path / f'accounts/main/{wid}/source_rows.private.json')
    assert len(source) == 9  # four ordinary ideas plus five kept drafts
    assert seen.count(idea_v1.STAGE_COMMIT) == 1 and seen.count(idea_v1.STAGE_COMMIT_KEPT) == 1
    assert budget.summary()['developments'] == 9


def test_explore_call_runs_end_to_end(tmp_path, monkeypatch):
    """An explore call: 14 sketches, then 6 more aimed beyond where they landed (with their own bank neighbours), then a
    kept-draft commit over all 20 that may take round-two ideas but invents nothing."""
    inputs = prepare(tmp_path)
    wid = 'one-w0'
    calls = [supply.make_call(arm=arm, bank_setting='loop', context='c01', seed=777, account='main', window=wid,
                              idea_model='gpt-6.1-sol', effort='xhigh') for arm in ('SEM', 'OPEN')]
    calls[1]['commit'] = 'explore'
    calls[1]['opaque_kernel_ids'] += ['k' + calls[1]['opaque_id'][1:13] + f'{i:06d}' for i in range(2)]
    windows = [{'id': wid, 'phase': 'screen', 'account': 'main', 'context': 'c01', 'seed': 777, 'calls': calls,
                'maximum_developments': 0, 'minimum_selector_eligible_kernels': 1, 'kind': 'fresh_supply', 'supply_only': True,
                'editorial_mode': 'bypass', 'protected': False, 'pairing_index': 0}]
    requests = {wid: {c['opaque_id']: supply.idea_request(c, None, inputs) for c in calls}}
    assert 'Step 1 of 3' in requests[wid][calls[1]['opaque_id']]['prompt']
    for i in range(3):
        gid = f'{wid}-develop-{i}'
        windows.append({'id': gid, 'phase': 'screen', 'account': None, 'source_account': 'main', 'context': 'c01',
                        'seed': 900 + i, 'calls': [], 'maximum_developments': 3, 'kind': 'eligible_group', 'source_window': wid,
                        'group_ordinal': i, 'editorial_mode': 'bypass', 'research_calls_per_account': {'main': 0}})
        requests[gid] = {}
    cap = design()['limits']; cap['phases'][0]['phase'] = 'screen'
    budget = Budget(tmp_path, cap)
    index = ReferenceIndex(read(inputs['selector_context'])['references'], Vectors(), provider_identity={'synthetic': True})
    seen, lock = [], threading.Lock()
    kernel = lambda src, i: {'source': src, 'central_relation': f'Relation {i}.', 'recipe': f'Recipe {i}.', 'what_is_new': 'New.',
                             'why_coherent': 'Coherent.', 'possible_test_generalization': 'A new situation.', 'main_authoring_risk': 'Ambiguity.'}

    def fake(invoker, stage, cid):
        with lock: seen.append(stage)
        req = read(invoker.calls_dir / stage / cid / 'request.json')
        prompt = (invoker.calls_dir / stage / cid / 'prompt.md').read_text()
        if stage == idea_v1.STAGE_SKETCH:
            return json.dumps(_sketches())
        if stage == idea_v1.STAGE_COMMIT:
            return json.dumps(_commit())
        if stage == idea_v1.STAGE_EXPLORE:
            assert prompt.count('### s') == 14 and 'Most similar existing tasks:' in prompt
            return json.dumps({'sketches': [{'id': f'n{i:02d}', 'title': f'Far idea {i}', 'idea': f'A distant relation {i}.',
                                             'recipe': f'Do far step {i}.'} for i in range(1, 7)]})
        if stage == idea_v1.STAGE_COMMIT_KEPT:
            assert prompt.count('### s') == 14 and prompt.count('### x') == 6 and 'Step 3 of 3' in prompt
            verdicts = [{'id': f's{i:02d}', 'verdict': 'keep' if i <= 3 else 'drop', 'closest_existing': 'x'} for i in range(1, 15)]
            verdicts += [{'id': f'x{i:02d}', 'verdict': 'keep' if i <= 2 else 'drop', 'closest_existing': 'x'} for i in range(1, 7)]
            return json.dumps({'sketch_verdicts': verdicts,
                               'kernels': [kernel(s, i) for i, s in enumerate(['s01', 's02', 's03', 'x01', 'x02'])]})
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
                              production=inputs['production'], manifest={'fixture': 'v1-explore'}, mock=True)
    finally:
        for pool in pools.values(): pool.finish()
    source = read(tmp_path / f'accounts/main/{wid}/source_rows.private.json')
    assert len(source) == 9  # four ordinary ideas plus five kept drafts, two of them from the explore round
    assert seen.count(idea_v1.STAGE_EXPLORE) == 1 and seen.count(idea_v1.STAGE_COMMIT_KEPT) == 1
    loop = tmp_path / f'accounts/main/{wid}/idea_loop/{calls[1]["opaque_id"]}.explore.private.json'
    assert len(read(loop)['sketches']) == 6 and len(read(loop)['neighbours']) == 6
    assert budget.summary()['developments'] == 9


def test_explore_calls_get_a_third_idea_turn_in_the_budget():
    from arcgen.batch.allocations.make_allocation import build
    spec = {'revision': 'r', 'seed': 1, 'windows': {'w1': {'account': 'main', 'wave': 'A', 'context': 'c01', 'commit': 'kept'},
                                                   'w2': {'account': 'main', 'wave': 'A', 'context': 'c01', 'commit': 'explore'}},
            'rows': [['w1', 'OPEN', None, 'L', 'sol', 'high'], ['w2', 'OPEN', None, 'L', 'sol', 'high']]}
    out = build(spec, {'sol': 'gpt-6.1-sol'}, {}, lambda arm, inv: True)
    assert [c.get('commit') for c in out['calls']] == ['kept', 'explore'] and [len(c['candidate_ids']) for c in out['calls']] == [6, 6]
    from arcgen.batch import allocation
    calls = [{'bank_setting': 'loop', 'account': 'main', 'commit': c['commit']} for c in out['calls']]
    assert sum(allocation.TURNS[c['bank_setting']] + (c.get('commit') == 'explore') for c in calls) == 5


def test_exact_modifier_assignment_hits_each_share():
    """With modifier_assignment exact (the v1 reference), each modifier lands on exactly round(rate x calls) calls."""
    from arcgen.batch import config
    if config.SCIENCE.get('modifier_assignment') != 'exact':
        import pytest; pytest.skip('profile uses per-call draws')
    calls = [c for w in design()['windows'] if w['kind'] == 'fresh_supply' for c in w['calls']]
    for m, rate in config.SCIENCE['idea_modifiers'].items():
        assert sum(m in c['idea_modifiers'] for c in calls) == round(rate * len(calls))
