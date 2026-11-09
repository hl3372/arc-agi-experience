"""Domain A pilot harness: public names map to the world, budgets hold, invalid experiments cost budget, scoring works."""
import json
import random
import pytest
from research_envs.discover.world import World
from research_envs.discover import harness as H, fairness as F, baselines as Bl
from research_envs.construct import sandbox
from tests.test_baselines import TOGGLE

needs_landlock = pytest.mark.skipif(sandbox.landlock_abi() < sandbox.MIN_ABI, reason='needs Linux with Landlock ABI >= 6')


def test_public_map_round_trip_and_errors():
    w = World(TOGGLE)
    pm = H.PublicMap(w)
    assert set(pm.labels) == {'A', 'B'} and pm.svars == {'s1': 'x'}
    tr = pm.run({'start': {'s1': 1}, 'actions': ['A', 'B', 'B']})
    assert tr == [[1], [1], [2], [3]]
    for bad in ({'start': {'s9': 1}, 'actions': []}, {'start': {'s1': 99}, 'actions': []},
                {'start': {}, 'actions': ['Z']}, {'start': {}, 'actions': ['A'] * 7}):
        with pytest.raises(ValueError):
            pm.run(bad)


def test_scripted_pilot_through_the_harness():
    """A scripted 'model': runs a few probes, then answers queries with a learner fitted on its own traces."""
    w = World(TOGGLE)
    pm = H.PublicMap(w)
    qs = F.sample_queries(w, random.Random(0), n_interp=10, n_comp=5)
    for q in qs[:5]:
        q['disagree'] = [0]
    state = {'turn': 0}

    def call(prompt, schema, rec, name, effort='high'):
        if name.endswith('queries'):
            data = [(pm.to_world(h)[0], pm.to_world(h)[1], [tuple(x) for x in h['trace']]) for h in state['hist']]
            m = Bl.SymbolicHidden(data)
            state['expected'] = [m.predict(q['start'], q['inputs']) == F.answer(w, q) for q in qs]
            ans = {f'Q{i + 1}': list(m.predict(q['start'], q['inputs'])) for i, q in enumerate(qs)}
            return {'answers_json': json.dumps(ans), 'law': 'scripted'}
        state['turn'] += 1
        exps = [{'start': {'s1': s}, 'actions': ['A', 'B', 'B', 'A', 'B']} for s in range(4)] + \
               [{'start': {'s1': 2}, 'actions': ['B', 'B']}, {'start': {'s1': 0}, 'actions': ['NOPE']}]
        state['hist'] = [{'start': e['start'], 'actions': e['actions'], 'trace': pm.run(e)} for e in exps[:5]]
        return {'experiments_json': json.dumps(exps), 'belief': 'x', 'done': True}

    r = H.run_pilot(w, 'a device', qs, budget=6, calls=None, call=call)
    assert r['experiments_used'] == 6                       # the invalid experiment still costs budget
    assert any(isinstance(h['trace'], str) and h['trace'].startswith('invalid') for h in r['history'])
    assert r['accuracy'] is not None and r['critical_accuracy'] is not None
    assert r['hits'] == state['expected']                   # the harness scores exactly what was answered


@needs_landlock
def test_tolerant_parsing_and_predictor_program():
    w = World(TOGGLE)
    pm = H.PublicMap(w)
    assert pm.run({'start': {'s1': 1}, 'actions': ['a', ' B ', ['B']]}) == pm.run({'start': {'s1': 1}, 'actions': ['A', 'B', 'B']})
    assert H._norm_answer(True, ['r1']) == [1] and H._norm_answer({'r1': 3}, ['r1']) == [3] and H._norm_answer(5, ['r1']) == [5]
    qs = F.sample_queries(w, random.Random(0), n_interp=10, n_comp=5)
    truths = {f'Q{i + 1}': list(F.answer(w, q)) for i, q in enumerate(qs)}
    prog = 'import json\nprint(json.dumps(%s))' % json.dumps(truths)   # a "program" that knows the answers

    def call(prompt, schema, rec, name, effort='high'):
        if name.endswith('queries'):
            return {'answers_json': '{}', 'law': 'x', 'predictor_program': prog}
        return {'experiments_json': '[]', 'belief': '', 'done': True}
    r = H.run_pilot(w, 'a device', qs, budget=0, calls=None, call=call)
    assert r['scored_from'] == 'program' and r['accuracy'] == 1.0 and r['direct_accuracy'] == 0.0


def test_time_series_answers_take_the_final_readouts():
    assert H._norm_answer([8, 9, 14], ['r1']) == [14]
    assert H._norm_answer([[1, 0], [2, 1]], ['r1', 'r2']) == [2, 1]
    assert H._norm_answer([3, 1], ['r1', 'r2']) == [3, 1]          # exactly n values: the final readouts as asked


def test_predictor_program_is_never_run_unsandboxed(monkeypatch):
    """Without Landlock the predictor program is not run: the pilot is scored from its direct answers, and the
    record says why."""
    monkeypatch.setattr(sandbox, 'landlock_abi', lambda: -1)
    w = World(TOGGLE)
    qs = F.sample_queries(w, random.Random(0), n_interp=4, n_comp=0)
    answers = {f'Q{i + 1}': list(F.answer(w, q)) for i, q in enumerate(qs)}

    def call(prompt, schema, rec, name, effort='high'):
        if name.endswith('queries'):
            return {'answers_json': json.dumps(answers), 'law': 'x', 'predictor_program': 'print("{}")'}
        return {'experiments_json': '[]', 'belief': '', 'done': True}
    r = H.run_pilot(w, 'a device', qs, budget=0, calls=None, call=call)
    assert r['scored_from'] == 'answers' and r['accuracy'] == 1.0
    assert not r['program']['ran'] and 'Landlock' in r['program']['sandbox_unavailable']
