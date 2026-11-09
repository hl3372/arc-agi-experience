"""Carrying a stopped run's ideas into the next batch, and oversized selector windows (2026-10-10, after v1-b13).

Saved-ideas windows send ideas that never met a selector through this batch's selector with no idea call; a window
whose complete selector prompt is over budget gets fewer accepted neighbours per candidate, down to a floor."""
from hashlib import sha256
import json
import pytest
from arcgen.batch import allocation, config, idea_v1, saved_ideas, scheduler, supply, waves
from arcgen.batch.common import IntegrityError, SETTINGS, digest
from arcgen.paths import resolve


def _rows(inputs, wid, seeds):
    """Source rows built the way a run builds them: mock idea-loop commits through supply.source_rows."""
    from test_v1_ideas import _commit
    calls = [supply.make_call(arm='SEM', bank_setting='loop', context='c01', seed=s, account='main', window=wid,
                              idea_model='gpt-6.1-sol', effort='high') for s in seeds]
    w = {'id': wid, 'context': 'c01', 'calls': calls, 'wave': 'A'}
    requests = {c['opaque_id']: supply.idea_request(c, None, inputs) for c in calls}
    responses = {}
    for n, c in enumerate(calls):
        value = idea_v1.as_ideas(c, idea_v1.parse_commit(_commit()))
        for i, k in enumerate(value['ideas']):
            k['central_relation'] += f' (variant {n}.{i})'
        responses[c['opaque_id']] = value
    return supply.source_rows(w, requests, responses)


def _saved(tmp_path, inputs, wid, account='main', wave='A'):
    rows = _rows(inputs, 'old-' + wid, (901, 902))
    path = tmp_path / f'{wid}.json'
    path.write_text(json.dumps({'provenance': {'id': wid, 'source_run': 'runs/old', 'source_window': 'old-' + wid},
                                'source_rows': rows}))
    window = {'id': wid, 'phase': 'screen', 'account': account, 'context': 'c01', 'seed': 4242, 'calls': [],
              'kind': 'saved_ideas', 'supply_only': True, 'maximum_developments': 0, 'minimum_selector_eligible_kernels': 1,
              'editorial_mode': 'bypass', 'protected': False, 'pairing_index': 0, 'reference_wave': wave,
              'saved_ideas': str(path), 'saved_ideas_sha256': sha256(path.read_bytes()).hexdigest(),
              'saved_kernel_ids': sorted(rows)}
    return window, rows


def _selector_response(ids, cid):
    return {'selector_id': cid, 'window_observations': None,
            'candidate_assessments': [{'candidate_id': k, 'eligibility': 'plausible', 'conceptual_potential': 'solid',
                'coherence_potential': 'solid', 'fair_realization_potential': 'plausible', 'redundant_with': [],
                'same_procedure_family_with': [], 'bank_comparisons': [], 'useful_contribution': 'Fresh insight ' + k,
                'reason': 'Coherent synthetic relation.'} for k in ids],
            'ranked_plausible': [{'candidate_id': k, 'development_priority': i + 1, 'development_brief': 'Realize fairly.',
                'optional_development_suggestion': None} for i, k in enumerate(ids)]}


def test_saved_ideas_refuse_a_changed_file_or_membership(tmp_path):
    from support.bank import prepare
    window, rows = _saved(tmp_path, prepare(tmp_path), 'one-saved-01')
    assert sorted(saved_ideas.load(window)['source_rows']) == window['saved_kernel_ids'] and len(rows) == 8
    with pytest.raises(IntegrityError, match='disagree'):
        saved_ideas.load({**window, 'saved_kernel_ids': window['saved_kernel_ids'][1:]})
    moved = dict(rows); first = next(iter(moved)); moved[first] = {**moved[first], 'study': {**moved[first]['study'], 'window': 'elsewhere'}}
    elsewhere = tmp_path / 'elsewhere.json'
    elsewhere.write_text(json.dumps({'provenance': {'source_run': 'runs/old', 'source_window': 'old-one-saved-01'}, 'source_rows': moved}))
    with pytest.raises(IntegrityError, match='generated elsewhere'):
        saved_ideas.load({**window, 'saved_ideas': str(elsewhere), 'saved_ideas_sha256': sha256(elsewhere.read_bytes()).hexdigest()})
    with open(window['saved_ideas'], 'a') as f: f.write(' ')
    with pytest.raises(IntegrityError, match='changed after'):
        saved_ideas.load(window)
    assert supply.window_kernel_ids({**window, 'saved_kernel_ids': ['a', 'b']}) == ['a', 'b']
    with pytest.raises(IntegrityError):
        supply.window_kernel_ids({**window, 'calls': [{'opaque_kernel_ids': ['c']}]})


def test_saved_ideas_meet_the_selector_without_idea_calls_and_feed_later_waves(tmp_path, monkeypatch):
    """A saved-ideas window in wave A is selected with no idea call; a wave-B idea window waits for it and sees its
    selections as in-batch references."""
    from arcgen.lib.factory_runtime import FactoryModelInvoker
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
    saved, rows = _saved(tmp_path, inputs, 'one-saved-01')
    call = supply.make_call(arm='SEM', bank_setting='loop', context='c01', seed=778, account='main', window='two-w2',
                            idea_model='gpt-6.1-sol', effort='high')
    fresh = {'id': 'two-w2', 'phase': 'screen', 'account': 'main', 'context': 'c01', 'seed': 778, 'calls': [call],
             'maximum_developments': 0, 'minimum_selector_eligible_kernels': 1, 'kind': 'fresh_supply', 'supply_only': True,
             'editorial_mode': 'bypass', 'protected': False, 'pairing_index': 1, 'reference_wave': 'B'}
    requests = {'one-saved-01': {}, 'two-w2': {call['opaque_id']: supply.idea_request(call, None, inputs)}}
    cap = allocation.design()['limits']; cap['phases'][0]['phase'] = 'screen'
    budget = Budget(tmp_path, cap)
    index = ReferenceIndex(read(inputs['selector_context'])['references'], Vectors(), provider_identity={'synthetic': True})
    stages = []

    def fake(invoker, stage, cid):
        req = read(invoker.calls_dir / stage / cid / 'request.json')
        stages.append((str(invoker.calls_dir), stage))
        if stage == idea_v1.STAGE_SKETCH:
            return json.dumps(_sketches())
        if stage == idea_v1.STAGE_COMMIT:
            return json.dumps(_commit())
        if stage == selection.STAGE:
            return json.dumps(_selector_response(req['parent_ids'], cid))
        raise AssertionError(stage)

    monkeypatch.setattr(FactoryModelInvoker, '_mock_response', fake)
    pools = {a: account_pool(tmp_path / 'accounts' / a / 'worker_pool.json') for a in ('main',)}
    try:
        scheduler.run_windows(tmp_path, 'screen', [saved, fresh], requests, inputs, index=index, pools=pools, budget=budget,
                              production=inputs['production'], manifest={'fixture': 'carry'}, mock=True)
    finally:
        for pool in pools.values(): pool.finish()
    d = tmp_path / 'accounts/main/one-saved-01'
    assert [s for p, s in stages if 'one-saved-01' in p] == [selection.STAGE]  # selector only, no idea call
    assert read(d / 'source_rows.private.json') == rows
    packed = read(d / 'selector_inputs.private.json')
    assert sorted(packed['raw_proposal_ids']) == saved['saved_kernel_ids'] and 'context_reduction' not in packed
    queue = read(d / 'eligible_queue.private.json')['rows']
    assert queue and {r['source_candidate_id'] for r in queue} <= set(rows)
    assert all(r['study']['original_selector_window'] == 'one-saved-01' and r['study']['window'] == 'old-one-saved-01'
               for r in queue)  # selected here, generated in the source run
    refs = read(tmp_path / 'design/waves/B/batch_references.private.json')['references']
    assert {r['reference_id'] for r in refs} == {'batch:' + r['source_candidate_id'] for r in queue}


def test_an_oversized_window_gets_fewer_neighbours_down_to_the_floor(tmp_path, monkeypatch):
    from arcgen.astra_usage_diversity_20260913.common import read
    from support.bank import prepare
    from arcgen.batch.retrieval import ReferenceIndex, _measure
    from support.fixtures import Vectors
    inputs = prepare(tmp_path, n=SETTINGS['selector']['accepted_neighbors_per_candidate'] + 20)  # deeper than k
    window, rows = _saved(tmp_path, inputs, 'one-saved-01')
    index = ReferenceIndex(read(inputs['selector_context'])['references'], Vectors(), provider_identity={'synthetic': True})
    sel = SETTINGS['selector']; top = sel['accepted_neighbors_per_candidate']
    size = lambda p: _measure(*supply._selector_prompt(window, p)[:3])[0]['estimated_prompt_tokens']
    monkeypatch.setitem(sel, 'minimum_accepted_neighbors', 1)
    full = supply.selector_inputs(window, rows, inputs, index)
    assert 'context_reduction' not in full and full['private_retrieval']['accepted_k'] == top
    supply.verify_selector_inputs(window, rows, inputs, full)
    sizes = {}
    for k in (top - 5, top - 4):
        monkeypatch.setitem(sel, 'accepted_neighbors_per_candidate', k)
        sizes[k] = size(supply.selector_inputs(window, rows, inputs, index))
    monkeypatch.setitem(sel, 'accepted_neighbors_per_candidate', top)
    assert sizes[top - 5] < sizes[top - 4] < size(full)
    # over budget at top - 4 and above, within it at top - 5: the loop stops at top - 5, records it, and the request fits
    monkeypatch.setitem(sel, 'prompt_token_budget', (sizes[top - 5] + sizes[top - 4]) // 2)
    packed = supply.selector_inputs(window, rows, inputs, index)
    assert packed['context_reduction']['accepted_k'] == packed['private_retrieval']['accepted_k'] == top - 5
    assert all(len(p['accepted_reference_ids']) == top - 5 for p in packed['payload']['packages'].values())
    supply.verify_selector_inputs(window, rows, inputs, packed, (), index)  # the reduction reproduces from the index
    with pytest.raises(IntegrityError, match='does not reproduce'):  # and cannot be verified without it
        supply.verify_selector_inputs(window, rows, inputs, packed)
    supply.selector_request(window, packed)
    for bad, why in ((0, 'floor'), (top, 'floor'), (top - 6, 'does not reproduce')):  # below the floor, no change, a deeper cut
        tampered = json.loads(json.dumps({k: v for k, v in packed.items() if k != 'packed_hash'}))
        tampered['context_reduction']['accepted_k'] = tampered['private_retrieval']['accepted_k'] = bad
        tampered['packed_hash'] = digest(tampered)
        with pytest.raises(IntegrityError, match=why):
            supply.verify_selector_inputs(window, rows, inputs, tampered, (), index)
    with pytest.raises(IntegrityError, match='different frozen'):  # any selector setting is part of the input identity
        monkeypatch.setitem(sel, 'minimum_accepted_neighbors', top - 4)
        supply.verify_selector_inputs(window, rows, inputs, packed, (), index)
    # the floor binds: at top - 4 the prompt is still over, nothing below is tried, and the request refuses as before
    floored = supply.selector_inputs(window, rows, inputs, index)
    assert floored['context_reduction']['accepted_k'] == top - 4
    with pytest.raises(IntegrityError, match='exceeds its frozen budget'):
        supply.selector_request(window, floored)
    monkeypatch.setitem(sel, 'minimum_accepted_neighbors', top)  # no floor below k (the default): never reduced
    same = supply.selector_inputs(window, rows, inputs, index)
    assert 'context_reduction' not in same
    with pytest.raises(IntegrityError, match='exceeds its frozen budget'):
        supply.selector_request(window, same)


def test_design_makes_saved_idea_windows_selector_windows(tmp_path, monkeypatch):
    from arcgen.astra_usage_diversity_20260913.common import read
    from support.bank import prepare
    window, rows = _saved(tmp_path, prepare(tmp_path), 'one-saved-01')
    proposal = {'revision': allocation.REVISION, 'coverage': {},
                'windows': {'w1': {'account': 'main', 'wave': 'A', 'context': 'c01', 'reference_wave': 'B'}},
                'calls': [{'window': 'w1', 'account': 'main', 'wave': 'A', 'context': 'c01', 'seed': 1, 'arm': 'SEE',
                           'invitation': None, 'bank_setting': 'loop', 'effort': 'high', 'model': 'gpt-6.1-sol', 'commit': 'explore',
                           'private_id': 'call-w1', 'candidate_ids': [f'kw1{i}' for i in range(6)]}],
                'saved_idea_windows': {'one-saved-01': {'account': 'main', 'wave': 'A', 'context': 'c01', 'seed': 5,
                    'ideas': window['saved_ideas'], 'ideas_sha256': window['saved_ideas_sha256'], 'reference_wave': 'A'}}}
    monkeypatch.setattr(allocation, 'read', lambda path: proposal if path == resolve(config.SCIENCE['allocation']) else read(path))
    monkeypatch.setitem(config.SCIENCE, 'modifier_assignment', 'per_call')
    plan = allocation.design()
    w = next(x for x in plan['windows'] if x['id'] == 'one-saved-01')
    assert (w['kind'], w['calls'], w['saved_kernel_ids'], w['reference_wave']) == ('saved_ideas', [], sorted(rows), 'A')
    assert plan['limits']['phases'][0]['categories']['selectors'] == 2  # the idea window and the saved ideas
    groups = [g for g in plan['potential_groups'] if g['source_window'] == 'one-saved-01']
    assert len(groups) == -(-allocation.development_capacity(len(rows)) // allocation.GROUP_SIZE)
    assert waves.earlier(plan['windows'], plan['windows'][0]) == [w]  # wave B idea windows wait for it
    no_wave = {k: v for k, v in proposal['saved_idea_windows']['one-saved-01'].items() if k != 'reference_wave'}
    proposal['saved_idea_windows']['one-saved-01'] = no_wave
    with pytest.raises(IntegrityError, match='every saved ideas window needs one'):
        allocation.design()
    proposal['saved_idea_windows']['one-saved-01'] = {**no_wave, 'reference_wave': 'A', 'ideas_sha256': '0' * 64}
    with pytest.raises(IntegrityError, match='bound hash'):
        allocation.design()


def test_design_accepts_saved_windows_only(tmp_path, monkeypatch):
    """Rescue r1 (2026-10-25): a spec of saved windows only, with no idea call, designs under a profile that keeps calls
    bare (v1.json's modifier_bare): nothing to draw or keep bare, one selector per saved-ideas window, development ceilings
    from the saved rows. Without idea windows a saved window takes no reference wave."""
    from arcgen.astra_usage_diversity_20260913.common import read
    from support.bank import prepare
    from support.carry import saved_queue, saved_ideas
    assert config.SCIENCE.get('modifier_bare', {}).get('*') and config.SCIENCE['modifier_assignment'] == 'exact'
    queue, ideas = 'batches/queues/test-saved-only-queue.json', 'batches/queues/test-saved-only-ideas.json'
    inputs = prepare(tmp_path)
    saved_queue(allocation.ROOT / queue, inputs); saved_ideas(allocation.ROOT / ideas, inputs)
    queued, rows = len(read(allocation.ROOT / queue)['rows']), read(allocation.ROOT / ideas)['source_rows']
    sha = lambda path: sha256((allocation.ROOT / path).read_bytes()).hexdigest()
    proposal = {'revision': allocation.REVISION, 'coverage': {}, 'windows': {}, 'calls': [],
                'saved_windows': {'one-saved-01': {'account': 'main', 'wave': 'A', 'context': 'c01', 'seed': 5, 'queue': queue,
                                                     'queue_sha256': sha(queue)}},
                'saved_idea_windows': {'two-carried-01': {'account': 'main', 'wave': 'A', 'context': 'c02', 'seed': 6,
                                                            'ideas': ideas, 'ideas_sha256': sha(ideas)}}}
    monkeypatch.setattr(allocation, 'read', lambda path: proposal if path == resolve(config.SCIENCE['allocation']) else read(path))
    plan = allocation.design()
    assert [(w['kind'], w['calls']) for w in plan['windows']] == [('saved_queue', []), ('saved_ideas', [])]
    categories = plan['limits']['phases'][0]['categories']
    assert (categories['ideas'], categories['selectors']) == (0, 1)
    assert plan['limits']['maximum_developments'] == allocation.development_capacity(queued) + allocation.development_capacity(len(rows))
    assert {g['source_window'] for g in plan['potential_groups']} == {'one-saved-01', 'two-carried-01'}
    for spec in (proposal['saved_windows']['one-saved-01'], proposal['saved_idea_windows']['two-carried-01']):
        spec['reference_wave'] = 'A'
    with pytest.raises(IntegrityError, match='reference wave needs batch references, idea windows'):
        allocation.design()


def test_commit_finds_carried_ideas_in_their_source_run(tmp_path, monkeypatch):
    """A task selected from saved ideas: its selector call is in this run, its idea calls in the ideas' source run."""
    from arcgen.batch import commit

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)); return path

    monkeypatch.setattr(commit, 'ROOT', tmp_path)
    run, old = tmp_path / 'runs/b14', tmp_path / 'runs/b13'
    write(run / 'config.json', {'revision': 'r14'})
    ideas = write(tmp_path / 'queues/ideas.json', {'provenance': {'source_run': 'runs/b13', 'source_window': 'two-supply-08'}})
    queue = write(tmp_path / 'queues/queue.json', {'provenance': {'source_run': 'runs/b13'}})
    windows = [{'id': 'one-carried-01', 'kind': 'saved_ideas', 'saved_ideas': 'queues/ideas.json', 'saved_kernel_ids': ['k1'],
                'saved_ideas_sha256': sha256(ideas.read_bytes()).hexdigest(),
                'account': 'main', 'cohort': 'r14'},
               {'id': 'one-saved-01', 'kind': 'saved_queue', 'saved_queue': 'queues/queue.json', 'account': 'main', 'cohort': 'r14'},
               {'id': 'one-supply-01', 'kind': 'fresh_supply', 'account': 'main', 'cohort': 'r14'}]
    write(run / 'design/batch.private.json', {'windows': windows})
    namespaces = {}
    for w in windows:
        write(run / 'accounts/main' / w['id'] / 'window.private.json', {'window': w})
        group = {'id': w['id'] + '-develop-01', 'source_account': 'main', 'source_window': w['id'], 'cohort': 'r14'}
        write(run / 'accounts/main' / group['id'] / 'window.private.json', {'window': group})
        namespaces[w['kind']] = run / 'accounts/main' / group['id'] / 'downstream'
    for wid in ('two-supply-08', 'two-supply-03'):
        (old / 'accounts/main' / wid).mkdir(parents=True)
    c = commit.Commit(run, 'c', 'ok')
    carried = {'source_candidate_id': 'k1', 'study': {'window': 'two-supply-08'}}
    assert c.upstream(namespaces['saved_ideas'], carried) == (
        old.resolve(), run / 'accounts/main/one-carried-01', old.resolve() / 'accounts/main/two-supply-08')
    for bad in ({**carried, 'study': {'window': 'two-supply-03'}}, {**carried, 'source_candidate_id': 'k2'}):
        with pytest.raises(ValueError, match='Saved idea provenance'):
            c.upstream(namespaces['saved_ideas'], bad)
    queued = {'source_candidate_id': 'k3', 'study': {'saved_queue_origin': {'window': 'two-supply-03'}}}
    assert c.upstream(namespaces['saved_queue'], queued) == (old.resolve(), *[old.resolve() / 'accounts/main/two-supply-03'] * 2)
    fresh = {'source_candidate_id': 'k4', 'study': {'window': 'one-supply-01'}}
    assert c.upstream(namespaces['fresh_supply'], fresh) == (run, *[run / 'accounts/main/one-supply-01'] * 2)
    assert c.saved_runs() == [old.resolve()] and c.run_of(old.resolve() / 'accounts/x/y/z') == old.resolve()
    with pytest.raises(ValueError, match='outside this run'):
        c.run_of(tmp_path / 'runs/other/accounts/x')
    ideas.write_text(ideas.read_text() + ' ')
    with pytest.raises(ValueError, match='changed after they were bound'):
        c.upstream(namespaces['saved_ideas'], carried)


def test_commit_reads_calls_of_a_source_run_moved_aside(tmp_path, monkeypatch):
    """2026-10-18: v1-b17-t1 crashed, was moved to v1-b17-t1.crashed-20261016, and a fresh v1-b17-t1 reused its call
    ids. v1-b17-t2 developed the crashed run's saved queue; its records keep the original call paths. The commit must
    read those calls from the moved run, never from the fresh run now at the original path."""
    from arcgen.batch import commit

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)); return path

    monkeypatch.setattr(commit, 'ROOT', tmp_path)
    run, moved, fresh = tmp_path / 'runs/t2', tmp_path / 'runs/t1.crashed', tmp_path / 'runs/t1'
    write(run / 'config.json', {'revision': 'r2', 'default_run': 'runs/t2'})
    write(moved / 'config.json', {'revision': 'r1', 'default_run': 'runs/t1'})
    write(fresh / 'config.json', {'revision': 'r1', 'default_run': 'runs/t1'})
    write(tmp_path / 'queues/queue.json', {'provenance': {'source_run': 'runs/t1.crashed'}})
    write(run / 'design/batch.private.json', {'windows': [{'id': 'one-saved-01', 'kind': 'saved_queue', 'saved_queue': 'queues/queue.json'}]})
    call = 'accounts/main/two-supply-02/upstream-x/model_calls/v1_idea_sketch/p1-sketch'
    for root, text in ((moved, 'crashed'), (fresh, 'fresh')):
        write(root / call / 'raw_response.txt', text)
    c = commit.Commit(run, 'c', 'ok')
    assert c.moved_runs() == {fresh.resolve(): moved.resolve()}
    local = c.local(str(fresh.resolve() / call))
    assert local == moved.resolve() / call and (local / 'raw_response.txt').read_text() == '"crashed"'
    assert c.run_of(local) == moved.resolve()
    assert c.local(str(run.resolve() / 'accounts/a/b')) == run.resolve() / 'accounts/a/b'
    with pytest.raises(ValueError, match='outside this run'):
        c.run_of(c.local(str(tmp_path / 'runs/other/accounts/x')))
