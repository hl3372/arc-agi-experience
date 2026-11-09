"""End-to-end dry run of one task with the model calls stubbed: every stage runs, the dossier is final, and a rerun
resumes from the cache without rebuilding. Also checks the battery's structure-preserving moves and time caps."""
import json
import random
import time
import pytest
from research_envs import common
from research_envs.construct import pipeline, llm, search
from research_envs.construct.evaluate import Spec
from research_envs.construct.sandbox import landlock_abi, MIN_ABI

# these tests run the stub planner's witness program, which only ever runs inside the Landlock sandbox
needs_landlock = pytest.mark.skipif(landlock_abi() < MIN_ABI, reason=f'needs Linux with Landlock ABI >= {MIN_ABI}')

SPEC = {'object': 'graph', 'kind': 'optimize', 'params': {'n': 8},
        'constraints': [{'name': 'max_deg_3', 'expr': 'all(deg(v) <= 3 for v in V)'},
                        {'name': 'no_triangle', 'expr': 'triangles() == 0'}],
        'objective': {'expr': 'len(E)', 'direction': 'max'}}
WITNESS = 'import json\nprint(json.dumps({"n": 8, "edges": [[i, i ^ (1 << b)] for i in range(8) for b in range(3) if i < i ^ (1 << b)]}))'  # the cube
PROGRAM = 'import json\nprint(json.dumps({"n": 8, "edges": [[i, (i + 1) % 8] for i in range(8)]}))'


def fake_call(prompt, schema, record_dir, name, effort='high', timeout=1800, retries=2):
    fake_call.calls.append(name)
    cand = {'title': 'cubic triangle-free', 'statement': 'max edges, degree <= 3, no triangle', 'spec_json': json.dumps(SPEC),
            'sibling_params_json': '{"n": 5}', 'search_hint_json': '{}', 'demand_rationale': '', 'construction_idea': '',
            'novelty_note': ''}
    return {'ideas': {'candidates': [cand]},
            'select': {'chosen_index': 0, 'reason': 'only one'},
            'plan': {'spec_json': json.dumps(SPEC), 'statement': cand['statement'], 'witness_program': WITNESS,
                     'witness_is_construction': True, 'degenerate_exprs': ['len(E) == 0'], 'sibling_params_json': '{"n": 5}',
                     'search_hint_json': '{}', 'notes': ''},
            'restate': {'spec_json': json.dumps(SPEC), 'ambiguities': ''},
            'naturalness': {'score': 4, 'about': 'x', 'arbitrary_parts': [], 'reasons': ''},
            }.get(name) or ({'object_json': json.dumps({'n': 8, 'edges': [[0, 1]]}), 'note': ''} if name.startswith('pilot_object')
                            else {'program': PROGRAM, 'note': ''})


@needs_landlock
def test_run_task_end_to_end_and_resume(tmp_path, monkeypatch):
    fake_call.calls = []
    monkeypatch.setattr(llm, 'call', fake_call)
    monkeypatch.setattr(pipeline, 'B', 300)
    monkeypatch.setattr(pipeline, 'PROCESSES', 4)
    rd = tmp_path / 'round'
    (rd / 'tasks').mkdir(parents=True)
    req = {'id': 't1', 'object': 'graph', 'demand': None}
    t = pipeline.run_task(req, rd, '')
    assert t['final'] and t['class'] in ('accepted', 'calibration', 'frontier', 'rejected')
    assert t['audit']['witness']['valid'] and t['reference'] == 12
    assert t['naturalness']['rater']['score'] == 4 and t['naturalness']['mechanical']['constraints'] == 2
    assert [o['class'] for o in t['outcomes']][:1] == ['valid_below_reference']
    assert (rd / 'tasks' / 't1' / 'build.json').exists()
    # a rerun with no final dossier reuses the cached build
    d = json.loads((rd / 'tasks' / 't1' / 'dossier.json').read_text()); d['final'] = False
    (rd / 'tasks' / 't1' / 'dossier.json').write_text(json.dumps(d))
    monkeypatch.setattr(pipeline, 'build', lambda *a: (_ for _ in ()).throw(AssertionError('rebuilt')))
    t2 = pipeline.run_task(req, rd, '')
    assert t2['final'] and t2['class'] == t['class']


@needs_landlock
def test_round_in_task_processes_with_a_failing_task(tmp_path):
    """The real round path: each task in its own spawned process (preview bug: evaluation in threads cannot use
    SIGALRM, so every pilot failed). One task fails; the other completes with pilot outcomes; no summary is written."""
    import os, subprocess, sys
    from pathlib import Path
    here = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'RE_LLM_STUB': str(here / 'tests' / 'llm_stub.py'), 'RE_BUDGET': '300', 'RE_ROOT': str(tmp_path)}
    code = ("import sys; sys.path.insert(0, %r)\nfrom research_envs.construct import pipeline\n"
            "if __name__ == '__main__':\n"
            "    pipeline.run_round('dry', [{'id': 'ok1', 'object': 'graph', 'demand': None},"
            " {'id': 'bad1', 'object': 'graph', 'demand': 'D1'}], workers=2)\n") % str(here)
    script = tmp_path / 'dry.py'; script.write_text(code)
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    rd = tmp_path / 'runs' / 'dry'
    assert not (rd / 'summary.json').exists()
    assert (rd / 'tasks' / 'bad1' / 'failure.txt').exists()
    t = json.loads((rd / 'tasks' / 'ok1' / 'dossier.json').read_text())
    assert t['final'] and t['outcomes'], t.get('pilots')
    assert {o['class'] for o in t['outcomes']} <= {'valid_below_reference', 'matches_reference', 'invalid', 'beats_reference'}
    assert t['outcomes'][0]['class'] == 'valid_below_reference'
    rows = {x['id']: x['class'] for x in json.loads((rd / 'progress.json').read_text())}
    assert rows['bad1'] == 'pending' and rows['ok1'] == t['class']


def test_structured_moves_preserve_degrees_and_set_sizes():
    rng = random.Random(0)
    p = search.Problem(Spec(SPEC), max_evals=10)
    chosen = frozenset(p.index[e] for e in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (4, 6)])
    degs = lambda ch: sorted(sum(v in p.universe[i] for i in ch) for v in range(8))
    swaps = [c for c in (search.neighbour(p, chosen, rng) for _ in range(300)) if len(c) == len(chosen) and c != chosen]
    assert swaps and any(degs(c) == degs(chosen) for c in swaps)
    sp = search.Problem(Spec({'object': 'set_system', 'kind': 'satisfy', 'params': {'m': 7},
                              'constraints': [{'name': 't', 'expr': 'True'}]}), {'set_sizes': [3]}, max_evals=10)
    ch = frozenset(range(4))
    for _ in range(100):
        ch = search.neighbour(sp, ch, rng)
        assert all(len(sp.universe[i]) == 3 for i in ch)


def test_time_cap_stops_a_method():
    t0 = time.time()
    r = search.run_method(('anneal', SPEC, None, 0, 10 ** 9, 2))
    assert r['time_capped'] and time.time() - t0 < 10


def test_file_slots_are_exclusive_across_processes(tmp_path, monkeypatch):
    import multiprocessing as mp
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    slots = common.FileSlots('t', 1)
    log = tmp_path / 'log'

    def hold(i):
        with slots:
            with open(log, 'a') as f:
                f.write(f'in{i}\n')
            time.sleep(0.5)
            with open(log, 'a') as f:
                f.write(f'out{i}\n')
    ps = [mp.get_context('fork').Process(target=hold, args=(i,)) for i in range(3)]
    [p.start() for p in ps]; [p.join() for p in ps]
    lines = log.read_text().split()
    assert all(lines[k][:2] == 'in' and lines[k + 1] == 'out' + lines[k][2:] for k in range(0, 6, 2)), lines


def test_local_search_finds_regular_graphs():
    """Battery bug (preview p5): all(deg(v) == d) graded as a count of failing vertices, so no single edge improved it
    and local search never found a regular graph. Graded as total distance, it does."""
    raw = {'object': 'graph', 'kind': 'optimize', 'params': {'n': 12},
           'constraints': [{'name': 'conn', 'expr': 'connected()'}, {'name': 'reg', 'expr': 'all(deg(v) == 3 for v in V)'}],
           'objective': {'expr': 'triangles()', 'direction': 'min'}}
    assert search.run_method(('hill_climb', raw, None, 1, 5000))['valid']


def test_blind_spec_that_cannot_evaluate_is_retried_and_never_makes_loopholes(monkeypatch, tmp_path):
    """Preview p4: the blind spec iterated over triangles() (a count), errored on every object, and every pilot was
    called a loophole. Now: one retry with the error; errors are never a rejection on the merits."""
    from research_envs.construct import sibling, probes
    broken = {**SPEC, 'constraints': [{'name': 'bad', 'expr': 'all(len(T) == 3 for T in triangles())'}]}
    answers = iter([{'spec_json': json.dumps(broken), 'ambiguities': ''}, {'spec_json': json.dumps(SPEC), 'ambiguities': ''}])
    names = []
    monkeypatch.setattr(llm, 'call', lambda prompt, schema, rec, name, **k: (names.append((name, 'could not be evaluated' in prompt)), next(answers))[1])
    cube = {'n': 8, 'edges': [[i, i ^ (1 << b)] for i in range(8) for b in range(3) if i < i ^ (1 << b)]}
    task = {'object': 'graph', 'statement': 's', 'spec': SPEC, 'witness': cube, 'sibling': {'n': 5}, 'hint': {}}
    rs, blind = pipeline.blind_restatement(task, tmp_path)
    assert names == [('restate', False), ('restate_2', True)] and blind == SPEC and rs['disagree_probes'] == 0
    assert not pipeline._blind_rejects(Spec(broken), cube)


def test_objectives_are_compared_only_on_valid_objects():
    """Preview p5: two specs with the same constraints and objectives that differ only off the valid set disagreed on
    22k sibling objects."""
    from research_envs.construct import sibling
    a = {'object': 'graph', 'kind': 'optimize', 'params': {'n': 5},
         'constraints': [{'name': 'tf', 'expr': 'triangles() == 0'}], 'objective': {'expr': 'len(E)', 'direction': 'max'}}
    b = {**a, 'objective': {'expr': 'len(E) - triangles()', 'direction': 'max'}}
    assert sibling.enumerate_all(a, other=b, processes=2)['disagree'] == 0


def test_literal_blind_specs_skip_the_sibling_comparison():
    """Round 1 r1_d2_g1: ours says degree (n-1)//3, the statement (at n = 31) says 10; a literal blind spec cannot
    be compared at the sibling size."""
    lit = {**SPEC, 'constraints': [{'name': 'd', 'expr': 'all(deg(v) == 3 for v in V)'}]}
    par = {**SPEC, 'constraints': [{'name': 'd', 'expr': 'all(deg(v) == n // 3 + 1 for v in V)'}]}
    assert not pipeline._parametric(lit) and pipeline._parametric(par)


@needs_landlock
def test_round2_switches_registry_and_improver(tmp_path):
    """Round 2 switches in task processes: the registry stops two tasks taking the same family, and the improver
    records a best known value that pilots are judged against."""
    import os, subprocess, sys
    from pathlib import Path
    here = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'RE_LLM_STUB': str(here / 'tests' / 'llm_stub.py'), 'RE_BUDGET': '300', 'RE_ROOT': str(tmp_path),
           'RE_ROUND_REGISTRY': '1', 'RE_IMPROVE': '1'}
    code = ("import sys; sys.path.insert(0, %r)\nfrom research_envs.construct import pipeline\n"
            "if __name__ == '__main__':\n"
            "    pipeline.run_round('dry2', [{'id': 'a1', 'object': 'graph', 'demand': None},"
            " {'id': 'a2', 'object': 'graph', 'demand': None}], workers=1)\n") % str(here)
    script = tmp_path / 'dry2.py'; script.write_text(code)
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    rd = tmp_path / 'runs' / 'dry2'
    t1 = json.loads((rd / 'tasks' / 'a1' / 'dossier.json').read_text())
    t2 = json.loads((rd / 'tasks' / 'a2' / 'dossier.json').read_text())
    assert t1['final'] and t1['best_known']['objective'] >= t1['reference']
    assert all('vs_best' in o for o in t1['outcomes'] if o['class'] != 'invalid')
    # the stub always proposes the same idea: the second task retries once with the round's families to avoid
    assert t2['class'] == 'rejected' and 'duplicates a family' in t2['reasons'][0]
    assert (rd / 'requests' / 'a2' / 'prechecks_2.json').exists()
