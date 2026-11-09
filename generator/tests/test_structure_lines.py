"""Structure lines (2026-10-22): a second slot axis drawn per call on top of a content line or an arm, its planner text composed
last and yielding to the line above, with a recorded "Structure: none" opt-out; the host matrix and the size limits at
admission; a separate `alternatives` record (whole rival grids per pair and test) with units and surfaces, re-written by
repairs; the per-line checks, capture and routing; the recorded default-reading call; QUIET and the new strata. Nothing
changes when the new keys are absent."""
from copy import deepcopy
from types import SimpleNamespace
import json, re
import pytest
from arcgen.batch import admission, config, contracts, gate, idea_v1, lines, routing, structures, supply
from arcgen.batch.common import IntegrityError, read
from arcgen.lib.astra_authoring_contract_v3 import pair_specific_test_narrative
from arcgen.lib.util import render_prompt
from test_weakness_lines import (WEAK, FLOORS, rows, _profile, _lplan, _record, _task, _row, _g, _pt, _emit, _Proxy,
                                 _shortcut)
from test_targeted import _Fake, _planner_kw

STRUCTS = ('ELIMINATE', 'RECOMBINE', 'PARSE', 'VARY', 'SHAPE', 'RELEVANCE')


@pytest.fixture
def weak(monkeypatch):
    w = deepcopy(WEAK)
    w['structures'] = {s: {} for s in STRUCTS}
    monkeypatch.setitem(config.SCIENCE, 'weakness', w)
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    return config.SCIENCE['weakness']


def srows(n, start=0, **study):
    return rows(n, start, **study)


# ---------------------------------------------------------------- registry, texts and config


def test_the_structure_registry_and_texts():
    assert config.STRUCTURES == STRUCTS and tuple(lines.STRUCTS) == STRUCTS
    assert lines.verify_registry() and {f'lines/{s}.planner.md' for s in STRUCTS} <= set(lines.verify_names())
    assert 'ideas/structures/SHAPE.md' in lines.verify_names() and 'lines/default_reading.md' in lines.verify_names()
    assert not set(STRUCTS) & set(config.LINE_NAMES) and not set(STRUCTS) & set(config.MODIFIER_NAMES)
    for name in STRUCTS:
        text = (lines.LINE_DIR / f'{name}.planner.md').read_text()
        head, body = text.split('-->', 1)
        assert '<!-- prompt-version: weakness-v1-structure-' in head
        # S5: composed last but yielding to the content line, with a recorded opt-out
        assert 'where the two conflict, the line above holds' in body and '"Structure: none' in body, name
        assert not re.search(r'\b[0-9a-f]{8}\b', body)
        s = lines.STRUCTS[name]
        assert s['kind'] == 'structure' and s['opt_out'] == {'field': 'invariants', 'prefix': 'Structure: none'}
        assert set(s['rungs']) == {'R1', 'R2'}  # S5: RECOMBINE R1/R2 only; every structure line here at R1-R2
    assert lines.STRUCTS['ELIMINATE']['not_on'] == ['SETTLE', 'JUDGE', 'COINCIDE', 'FORCED']  # the host matrix (S5)
    assert lines.STRUCTS['RECOMBINE']['not_on'] == ['PRECISE', 'FORCED'] and lines.STRUCTS['RECOMBINE']['counted_as'] == {'DEPART': 'DEPART'}
    assert lines.STRUCTS['SHAPE']['not_on'] == ['PROCESS', 'SETTLE', 'TRACE', 'PRECISE', 'FORCED'] and lines.STRUCTS['SHAPE']['idea']
    assert lines.STRUCTS['VARY']['replaces_overlay'] == 'ROLES' and lines.STRUCTS['RELEVANCE']['counted_as'] == {'RELATION': 'RELATION'}
    assert lines.STRUCTS['RECOMBINE']['policy_needs_check'] and lines.STRUCTS['RELEVANCE']['grids']['objects'] == {'contrast_grids': 2}
    assert 'shape detail' not in (lines.LINE_DIR / 'RELEVANCE.planner.md').read_text()  # S8: same colour, size and shape
    assert lines.LINES['QUIET']['grids']['pairs'] == {'min': 2, 'max': 10, 'minority': False}  # S10: >= 2 pairs from grids
    assert lines.LINES['PROCESS']['relax']['contact']['rungs'] == ['R1', 'R2'] and 'distortion' in lines.LINES['RELATION']['relax']
    assert lines.LINES['SETTLE']['default_reading'] == 'settling'
    for template in ('Survivor: a rule; each rival rule fails on one pair.',
                     'Combination: two laws each act apart in the pairs and commute; the test combines them.',
                     'Unit (holes as things): an enclosed hole.', 'Roles: a key and its targets.', 'Varied: colours and layout.',
                     'Output frame: the number of things.', 'Relevant: dots touching a wall act.', 'Structure: none — a reason.',
                     'Quiet edit: strays are removed; it acts beside the obvious change.'):
        assert not pair_specific_test_narrative(template) and 'first test' not in template, template


def test_registry_rejects_malformed_structure_entries(monkeypatch):
    good = deepcopy(lines.STRUCTS['ELIMINATE'])
    for bad in ({**good, 'kind': 'line'}, {**good, 'emits': ['points']}, {**good, 'not_on': ['NOPE']},
                {**good, 'rungs': []}, {**good, 'rungs': ['R9']}, {**good, 'opt_out': {'field': 'risk', 'prefix': 'x'}},
                {**good, 'grids': {'alternatives': {'rule': 'nope'}}}, {**good, 'grids': {}}, {**good, 'colour': 1},
                {**good, 'replaces_overlay': 'NOPE'}, {**good, 'default_reading': 'other'},
                {**good, 'plan': {'shortcut_count': [{'prefix': 'Rival', 'min': 3, 'max': 2}]}},
                {k: v for k, v in good.items() if k != 'realizer'}):
        monkeypatch.setitem(lines.STRUCTS, 'NEWSTRUCT', bad)
        with pytest.raises(IntegrityError):
            lines.verify_registry()
    monkeypatch.setitem(lines.STRUCTS, 'NEWSTRUCT', good)
    assert lines.verify_registry()
    with pytest.raises(IntegrityError, match='lines/NEWSTRUCT.planner.md'):
        lines.verify_names()


def test_structure_config_is_validated():
    w = _profile()['science']['weakness']
    config.validate(_profile(weakness={**w, 'structures': {'ELIMINATE': {}, 'SHAPE': {}}}))
    p = _profile(weakness={**w, 'structures': {'PARSE': {}}, 'default_reading': True})
    p['science']['call_overrides']['default_reading'] = ['sol', 'low']
    config.validate(p)
    config.validate(_profile(weakness={**w, 'overlays': {}, 'structures': {'PARSE': {}}}, colour_roles='v1', colour_vary_rate=None))
    with pytest.raises(config.ConfigError, match='VARY'):  # VARY varies role colours per idea, as ROLES does
        config.validate(_profile(weakness={**w, 'overlays': {}, 'structures': {'VARY': {}}}, colour_roles='v1', colour_vary_rate=None))
    for bad in ({**w, 'structures': {'NOPE': {}}}, {**w, 'structures': {'PARSE': {'share': 1}}}, {**w, 'structures': ['PARSE']},
                {**w, 'default_reading': True}, {**w, 'default_reading': 'yes'}, {**w, 'extra': 1}):
        with pytest.raises(config.ConfigError):
            config.validate(_profile(weakness=bad))


# ---------------------------------------------------------------- spec rows, admission, host matrix and size


def test_spec_rows_carry_a_structure(weak):
    from arcgen.batch.allocations.make_allocation import build
    def spec(*targets):
        return {'revision': 'r', 'seed': 1, 'windows': {'w': {'account': 'main', 'wave': 'A', 'context': 'c01'}},
                'rows': [['w', 'SEE', None, 'U', 'sol', 'high', t] for t in targets]}
    made = build(spec({'skill': 'MARKS', 'structure': 'ELIMINATE', 'size': 'small'}, {'structure': 'SHAPE', 'size': 'R2'},
                      {'skill': 'MARKS', 'structure_control': True, 'size': 'small'}, {'structure_control': True, 'size': 'R2'}),
                 config.SCIENCE['models'], {}, lambda a, i: True)
    assert [c.get('structure') for c in made['calls']] == ['ELIMINATE', 'SHAPE', None, None]
    assert [c.get('structure_control') for c in made['calls']] == [None, None, True, True]
    plain = build(spec({'skill': 'MARKS'}), config.SCIENCE['models'], {}, lambda a, i: True)['calls'][0]
    assert 'structure' not in plain  # rows without one keep their identity
    for bad, match in (({'structure': 'NOPE'}, 'not a structure line'), ({'control': True, 'structure': 'PARSE'}, 'matched-control'),
                       ({'skill': 'SETTLE', 'structure': 'ELIMINATE'}, 'host matrix'), ({'structure': 'RECOMBINE', 'size': 'R3'}, 'allows only'),
                       ({'skill': 'PROCESS', 'structure': 'SHAPE'}, 'host matrix'),
                       ({'structure': 'PARSE', 'structure_control': True}, 'structure_control'),
                       ({'control': True, 'structure_control': True}, 'structure_control')):
        with pytest.raises(ValueError, match=match):
            build(spec(bad), config.SCIENCE['models'], {}, lambda a, i: True)
    call = supply.make_call(arm='SEE', bank_setting='loop', context='c01', seed=1, account='main', window='w', effort='high',
                            idea_model=config.SCIENCE['models']['sol'])
    assert 'structure_call' not in json.dumps(call)


def test_admission_draws_the_structure_with_the_host_matrix_and_size_limits(weak, monkeypatch):
    monkeypatch.delitem(config.SCIENCE, 'rung_draw', raising=False)
    weak['lines']['MARKS']['rungs'] = {'R1': .25, 'R2': .25, 'R3': .25, 'R4': .25}
    out = admission.configured_routes(None, srows(200, idea_modifiers=['MARKS'], structure_call='ELIMINATE'))
    assert all(r['study']['structure'] == 'ELIMINATE' and r['study']['line'] == 'MARKS' for r in out)
    assert {r['study'].get('rung') for r in out} <= {'R1', 'R2'}  # S5: the line's mix limited to R1-R2
    assert {r['study'].get('rung') for r in out} == {'R1', 'R2'}
    large = admission.configured_routes(None, srows(40, idea_modifiers=['MARKS'], structure_call='RECOMBINE', size_arm='large'))
    assert {r['study']['rung'] for r in large} == {'R2'}
    with pytest.raises(IntegrityError):
        admission.configured_routes(None, srows(1, idea_modifiers=['MARKS'], structure_call='RECOMBINE', size_rung='R3', size_arm='large'))
    host = admission.configured_routes(None, srows(4, idea_modifiers=['SETTLE'], structure_call='ELIMINATE'))
    assert all('structure' not in r['study'] and r['study']['structure_excluded'] == {'structure': 'ELIMINATE', 'reason': 'host',
                                                                                      'host': 'SETTLE'} for r in host)
    counted = admission.configured_routes(None, srows(4, idea_modifiers=['RELATION'], structure_call='RELEVANCE'))
    assert all(r['study']['structure'] == 'RELEVANCE' and r['study']['structure_counted_as'] == 'RELATION' for r in counted)
    arm = admission.configured_routes(None, srows(4, structure_call='SHAPE', size_arm='small')) + \
        admission.configured_routes(None, srows(4, 10, structure_call='SHAPE', size_arm='large'))
    assert [r['study']['rung'] for r in arm] == ['R1'] * 4 + ['R2'] * 4 and all('line' not in r['study'] for r in arm)
    control = admission.configured_routes(None, srows(4, 20, structure_control=True, size_arm='large', size_rung='R2'))
    assert [r['study']['rung'] for r in control] == ['R2'] * 4 and all('structure' not in r['study'] for r in control)
    with pytest.raises(IntegrityError, match='matched control'):
        admission.configured_routes(None, srows(1, line_control=True, structure_call='PARSE'))
    weak['structures'].pop('PARSE')
    with pytest.raises(IntegrityError):
        admission.configured_routes(None, srows(1, structure_call='PARSE'))
    plain = admission.configured_routes(None, srows(20, idea_modifiers=['MARKS']))
    assert not any(k.startswith('structure') for r in plain for k in r['study'])  # absent keys stay absent


def test_vary_is_drawn_as_one_with_roles_and_structure_ideas_take_no_other_lines(weak, monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'colour_roles', 'mix')
    monkeypatch.setitem(config.SCIENCE, 'colour_vary_rate', 0.0)
    weak['overlays'] = {'ROLES': {'share': 1.0}}
    roles = admission.configured_routes(None, srows(20, idea_modifiers=['MARKS']))
    assert all(r['study']['weak_overlays'] == {'ROLES': 'drawn'} for r in roles)
    vary = admission.configured_routes(None, srows(20, idea_modifiers=['MARKS'], structure_call='VARY'))
    assert all('weak_overlays' not in r['study'] and r['study']['colour_roles'] == 'vary' for r in vary)
    fixed = admission.configured_routes(None, srows(4, idea_modifiers=['MARKS', 'COLOURFUL'], structure_call='VARY'))
    assert all(r['study']['colour_roles'] != 'vary' for r in fixed)  # fixed colours: VARY varies other dimensions
    monkeypatch.setitem(config.SCIENCE, 'teach_draw', {'plain': 1.0, 'excluded': []})
    monkeypatch.setitem(config.SCIENCE, 'depart_draw', {'test': 1.0, 'excluded': []})
    arm = admission.configured_routes(None, srows(10, structure_call='PARSE') + srows(10, 10, structure_control=True))
    assert all('teach' not in r['study'] and 'depart' not in r['study'] for r in arm)
    ordinary = admission.configured_routes(None, srows(10))
    assert all(r['study']['teach'] == 'plain' for r in ordinary)


def test_the_contact_stratum_keeps_its_levels(weak):
    weak['lines']['PROCESS'] = {'rungs': {'R1': .2, 'R2': .2, 'R3': .3, 'R4': .3}, 'relaxed': {'contact': .3}}
    out = admission.configured_routes(None, srows(200, idea_modifiers=['PROCESS'], fence_call='contact'))
    assert all(r['study']['fence'] == 'contact' and r['study'].get('rung') in (None, 'R1', 'R2') for r in out)
    strict = admission.configured_routes(None, srows(200, idea_modifiers=['PROCESS'], fence_call='strict'))
    assert {'R3', 'R4'} <= {r['study'].get('rung') for r in strict}


# ---------------------------------------------------------------- composition and plan fields


def test_the_structure_text_is_composed_last_after_the_line_and_its_fence():
    b = contracts.Bindings()
    texts = [r[0] for r in b.planning_templates('SEE', (), line='SETTLE', overlay='EDGE', fence='two_pairs', structure='RECOMBINE')]
    assert texts[-1].split('-->', 1)[1].strip().startswith('Structure for this idea: familiar laws in a new combination.')
    assert texts[-2].split('-->', 1)[1].strip().startswith('Relaxed fence for this idea: two training pairs')
    assert texts[-3].split('-->', 1)[1].strip().startswith('Settled point for this idea (binding')
    routed = [r[0] for r in b.planning_templates('SEE', ('evidence_scale_addendum',), structure='SHAPE')]
    assert 'Structure for this idea: infer the output' in routed[-2]  # a route addendum stays last
    assert [r[2] for r in b.planning_templates('SEE', ())] == [r[2] for r in b.planning_templates('SEE', (), structure=None)]
    with pytest.raises(Exception):
        b.planning_templates('SEE', (), structure='NOPE')
    with pytest.raises(Exception):
        b.planning_templates('SEE', (), structure='SETTLE')


def _plan(invariants=(), rules=(), notes=('a.', 'b.', 'c.')):
    return _lplan(notes=notes, invariants=list(invariants), rules=list(rules))


def test_plan_fields_for_every_structure():
    elim = _plan(['Survivor: move by the count; each rival rule fails on one pair.'], ['Rival 1: move two.', 'Rival 2: move to the wall.'])
    assert lines.plan_issues(None, None, elim, structure='ELIMINATE') == []
    assert lines.plan_issues(None, None, _plan(['Survivor: x; each rival rule fails on one pair.'], ['Rival 1: a.']), structure='ELIMINATE') == [
        '1 strongest_simple_shortcuts entries began "Rival", where 2 to 3 should']
    assert lines.plan_issues(None, None, {**elim, 'public_contract': {**elim['public_contract'],
                                                                       'training_pairs': elim['public_contract']['training_pairs'][:2]}},
                             structure='ELIMINATE') == ['it had 2 training pairs for 2 rivals, where at least 3 are needed']
    rec = _plan(['Combination: recolour and shift each act apart in the pairs and commute; the test combines them.'],
                ['Single law: recolour only.', 'Single law: shift only.', 'Replay: replay the closest pair.'])
    assert lines.plan_issues(None, None, rec, structure='RECOMBINE') == []
    assert any('"Replay:"' in i for i in lines.plan_issues(None, None, _plan(rec['public_contract']['invariants'][1:],
                                                                             ['Single law: a.', 'Single law: b.']), structure='RECOMBINE'))
    parse = _plan(['Unit (holes as things): an enclosed hole acts as one thing.'], ['Naive parse: single-colour pieces.'])
    assert lines.plan_issues(None, None, parse, structure='PARSE') == []
    assert lines.form('PARSE', lines.declared('PARSE', parse))['label'] == 'holes as things'
    vary = _plan(['Roles: a key and its targets, by position.'], ['Nearest surface (colours): x.', 'Nearest surface (layout): y.'])
    assert lines.plan_issues(None, None, vary, structure='VARY') == ['it had no invariant beginning "Varied:"']
    vary['public_contract']['invariants'].append('Varied: colours and the key layout.')
    assert lines.plan_issues(None, None, vary, structure='VARY') == []
    assert lines.plan_issues(None, None, _plan(['Output frame: the count of dots.']), structure='SHAPE') == []
    assert lines.plan_issues(None, None, _plan(), structure='SHAPE') == ['it had no invariant beginning "Output frame:"']
    rel = _plan(['Relevant: dots touching a wall act.'], ['Use everything: every dot.', 'Ignore all: no dot.'])
    assert lines.plan_issues(None, None, rel, structure='RELEVANCE') == []
    for name in STRUCTS:  # an honest opt-out is recorded and never re-planned
        none = _plan(['Structure: none — the line above fixes one law.'])
        assert lines.plan_issues(None, None, none, structure=name) == [] and lines.declared(name, none)['opted_out'] is True
    both = lines.plan_issues('PROCESS', None, _plan(['Process: things fall.'], ['Shortcut: x.']), structure='SHAPE')
    assert both == ['it had no invariant beginning "Conserved:"', 'it had no invariant beginning "Output frame:"']
    check = lines.plan_check('PROCESS', None, elim, structure='ELIMINATE')
    assert check['structure'] == 'ELIMINATE' and check['structure_declared']['shortcut_counts'] == {'Rival': 2}
    assert set(lines.plan_check('PROCESS', None, elim)) == {'line', 'overlay', 'fence', 'first_issues', 'replanned', 'used',
                                                            'final_issues', 'declared', 'form'}  # unchanged without one


def test_a_plan_missing_its_structure_fields_is_planned_once_more(tmp_path, monkeypatch):
    from arcgen.batch.runtime import Runtime
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    tid = 'dtest0000000000000901'
    record = _record({'structure': 'ELIMINATE'})
    bare = {**_plan(rules=['Rival 1: a.']), 'candidate_id': tid}
    good = {**_plan(['Survivor: x; each rival rule fails on one pair.'], ['Rival 1: a.', 'Rival 2: b.']), 'candidate_id': tid}
    fake = _Fake(tmp_path, {f'{tid}-concrete-plan': bare, f'{tid}-replan-concrete-plan': good})
    assert Runtime._plan_with_colour(fake, _planner_kw(tid, record), record, 'v1', False) == good
    first, second = fake.calls
    assert first['request_context']['q2_structure'] == 'ELIMINATE' and 'q2_line' not in first['request_context']
    assert first['prompt'].rstrip().endswith(lines.body('lines/ELIMINATE.planner.md').rstrip())
    assert '\n\nLine check on an earlier plan for this idea: it had no invariant beginning "Survivor:"' in second['prompt']
    receipt = read(tmp_path / 'line_check' / f'{tid}.json')
    assert receipt['structure'] == 'ELIMINATE' and receipt['final_issues'] == [] and receipt['used'] == 'second'


# ---------------------------------------------------------------- records, tables and checks


def _alt(aid, kind, train=(), test=(), line='ELIMINATE', note='n'):
    return {'id': aid, 'line': line, 'kind': kind, 'note': note, 'train': list(train), 'test': list(test)}


def _srec(task, study, alts=None, units=None, surfaces=None, objects=None, plan=None, mapping=None, points=None):
    m = {v: k for k, v in (mapping or {}).items()}  # the realizer wrote the original palette
    orig = lambda g: [[m.get(c, c) for c in row] for row in g] if g is not None else None
    entry = {'source': 'realizer', 'call_id': 'r', 'palette': 'original', 'problems': [], 'train': None,
             **{k: None for k in lines.LEDGERS}, 'objects': objects, 'points': points,
             'alternatives': [{**a, 'train': [orig(g) for g in a['train']], 'test': [orig(g) for g in a['test']]} for a in alts or []] or None,
             'units': units, 'surfaces': surfaces,
             'inputs': {'train': [orig(p['input']) for p in task['train']], 'test': [orig(p['input']) for p in task['test']]}}
    return {'study': study, 'complete_task': deepcopy(task), 'concrete_plan': plan,
            'palette_treatment': {'mapping': {str(k): v for k, v in (mapping or {}).items()}}, 'weakness': {'emissions': [entry]}}


def _two(g):
    """The grid with two adjacent cells at the row's end changed: an object-level difference."""
    out = deepcopy(g); out[0][5] = out[0][6] = 9
    return out


T = _task([(_row(1), _row(0, 1)), (_row(2), _row(0, 2)), (_row(3), _row(0, 3))], [(_row(4), _row(0, 4))])
OUT = [p['output'] for p in T['train']]
TRUTH = T['test'][0]['output']
R1 = _alt('r1', 'rival', [OUT[0], _two(OUT[1]), OUT[2]], [_two(TRUTH)])
R2 = _alt('r2', 'rival', [OUT[0], OUT[1], _two(OUT[2])], [_two(TRUTH)])
ELIM = {'structure': 'ELIMINATE'}


def _check(rec):
    return lines.episode_check(rec, rec['complete_task'])['structure']


def test_eliminate_killers_come_from_the_grids():
    ok = _check(_srec(T, ELIM, [R1, R2]))
    assert ok['status'] == 'pass' and ok['alternatives']['killers'] == [2, 3] and ok['alternatives']['distinct_killers']
    assert [r['status'] for r in ok['alternatives']['records']] == ['pass', 'pass']
    mapped = _srec(T, ELIM, [R1, R2], mapping={i: (i + 3) % 10 for i in range(10)})
    assert _check(mapped)['status'] == 'pass'  # realizer grids in the final palette
    same = _alt('r2', 'rival', [OUT[0], _two(OUT[1]), OUT[2]], [_two(TRUTH)])
    assert _check(_srec(T, ELIM, [R1, same]))['status'] == 'fail'  # killers must differ
    twice = _alt('r2', 'rival', [_two(OUT[0]), OUT[1], _two(OUT[2])], [_two(TRUTH)])
    assert _check(_srec(T, ELIM, [R1, twice]))['alternatives']['records'][1]['killer'] is None
    strawman = _alt('r2', 'rival', [OUT[0], OUT[1], _two(OUT[2])], [TRUTH])  # a rival the test does not separate
    assert _check(_srec(T, ELIM, [R1, strawman]))['status'] == 'fail'
    one_cell = lambda c: [[9 if j == c else x for j, x in enumerate(OUT[2][0])]]
    open_ground = _alt('r2', 'rival', [OUT[0], OUT[1], one_cell(6)], [_two(TRUTH)])  # one cell far from the change
    near = _alt('r2', 'rival', [OUT[0], OUT[1], one_cell(1)], [_two(TRUTH)])  # one cell beside it: not visible as an object
    assert _check(_srec(T, ELIM, [R1, open_ground]))['status'] == 'pass' and _check(_srec(T, ELIM, [R1, near]))['status'] == 'fail'
    blank = _alt('r2', 'rival', [OUT[0], None, _two(OUT[2])], [_two(TRUTH)])  # S3: a blank rival is not a pass
    assert _check(_srec(T, ELIM, [R1, blank]))['status'] == 'incomplete'
    assert _check(_srec(T, ELIM, [R1]))['status'] == 'fail'  # one rival: the single-rival form is SETTLE
    assert _check(_srec(T, ELIM))['status'] == 'incomplete'
    short = _task([(_row(1), _row(0, 1)), (_row(2), _row(0, 2))], [(_row(4), _row(0, 4))])
    r1 = _alt('r1', 'rival', [OUT[0], _two(OUT[1])], [_two(TRUTH)])
    r2 = _alt('r2', 'rival', [_two(OUT[0]), OUT[1]], [_two(TRUTH)])
    assert _check(_srec(short, ELIM, [r1, r2]))['alternatives']['pairs_enough'] is False
    plan = _plan(['Survivor: x; each rival rule fails on one pair.'], ['Rival 1: a.', 'Rival 2: b.', 'Rival 3: c.'])
    assert _check(_srec(T, ELIM, [R1, R2], plan=plan))['status'] == 'incomplete'  # a declared rival left out is missing


def test_recombine_orders_single_laws_and_replay():
    law1 = _alt('l1', 'single law', [OUT[0], _two(OUT[1]), _two(OUT[2])], [_two(TRUTH)], line='RECOMBINE')
    law2 = _alt('l2', 'single law', [_two(OUT[0]), OUT[1], OUT[2]], [[[5] * 7]], line='RECOMBINE')
    orders = [_alt(o, 'order', [], [TRUTH], line='RECOMBINE') for o in ('o1', 'o2')]
    replay = _alt('p1', 'replay', [], [[[6] * 7]], line='RECOMBINE')
    study = {'structure': 'RECOMBINE'}
    ok = _check(_srec(T, study, [*orders, law1, law2, replay]))
    assert ok['status'] == 'pass' and ok['alternatives']['orders']['orders_equal_truth']
    assert ok['alternatives']['laws']['pairs_no_single_law'] == []
    wrong_order = [orders[0], _alt('o2', 'order', [], [_two(TRUTH)], line='RECOMBINE')]
    assert _check(_srec(T, study, [*wrong_order, law1, law2, replay]))['status'] == 'fail'  # S6: order-independent only
    alone = _alt('l2', 'single law', [_two(OUT[0]), _two(OUT[1]), _two(OUT[2])], [[[5] * 7]], line='RECOMBINE')
    assert _check(_srec(T, study, [*orders, law1, alone, replay]))['status'] == 'fail'  # each law alone reproduces a pair
    twin = _alt('l2', 'single law', [_two(OUT[0]), OUT[1], OUT[2]], [_two(TRUTH)], line='RECOMBINE')
    assert _check(_srec(T, study, [*orders, law1, twin, replay]))['status'] == 'fail'  # one law in two words
    assert _check(_srec(T, study, [*orders, law1, law2]))['status'] == 'incomplete'
    still = _alt('p1', 'replay', [], [TRUTH], line='RECOMBINE')
    assert _check(_srec(T, study, [*orders, law1, law2, still]))['status'] == 'fail'


def test_parse_naive_parse_and_unit_labels():
    task = _task([([[1, 2, 0, 0]], [[3, 3, 0, 0]]), ([[1, 0, 2, 0]], [[3, 0, 3, 0]])], [([[0, 1, 2, 0]], [[0, 3, 3, 0]])])
    naive = _alt('n1', 'naive parse', [[[3, 3, 0, 0]], [[3, 0, 0, 0]]], [[[0, 3, 0, 0]]], line='PARSE')
    units = [{'grid': 'train_1', 'labels': [[1, 1, 0, 0]]}, {'grid': 'train_2', 'labels': [[1, 0, 2, 0]]},
             {'grid': 'test_1', 'labels': [[0, 1, 1, 0]]}]
    ok = _check(_srec(task, {'structure': 'PARSE'}, [naive], units=units))
    assert ok['status'] == 'pass' and ok['alternatives']['fitting_pairs'] == [1] and ok['alternatives']['differing_pairs'] == [2]
    u = ok['alternatives']['units']
    assert u['complete'] and u['differs_from_naive'] and [g['units'] for g in u['grids']] == [1, 2, 1]
    assert [g['naive_components'] for g in u['grids']] == [2, 2, 2] and [g['same_partition'] for g in u['grids']] == [False, True, False]
    fits_all = _alt('n1', 'naive parse', [p['output'] for p in task['train']], [[[0, 3, 0, 0]]], line='PARSE')
    assert _check(_srec(task, {'structure': 'PARSE'}, [fits_all], units=units))['status'] == 'fail'
    nowhere = _alt('n1', 'naive parse', [[[3, 0, 0, 0]], [[3, 0, 0, 0]]], [[[0, 3, 0, 0]]], line='PARSE')
    assert _check(_srec(task, {'structure': 'PARSE'}, [nowhere], units=units))['status'] == 'fail'  # S2: not a strawman
    partial = _check(_srec(task, {'structure': 'PARSE'}, [naive], units=units[:1]))
    assert partial['status'] == 'pass' and partial['alternatives']['units']['missing'] == ['train_2', 'test_1']  # recorded only


def test_vary_counts_answer_relevant_dimensions_and_records_surfaces():
    dims = [_alt('v1', 'role colours', [], [_two(TRUTH)], line='VARY'), _alt('v2', 'key layout', [], [[[5] * 7]], line='VARY')]
    surfaces = [{'grid': f'train_{i}', 'roles': [{'role': 'mover', 'colour': i}], 'key_box': [0, i, 0, i + 1], 'orientation': None,
                 'scale': None} for i in (1, 2, 3)] + [{'grid': 'test_1', 'roles': [{'role': 'mover', 'colour': 4}],
                                                         'key_box': None, 'orientation': None, 'scale': None}]
    ok = _check(_srec(T, {'structure': 'VARY'}, dims, surfaces=surfaces))
    assert ok['status'] == 'pass' and ok['alternatives']['relevant_dimensions'] == ['key layout', 'role colours']
    s = ok['alternatives']['surfaces']
    assert s['complete'] and s['varied']['role_colours'] and s['varied']['key_box'] and not s['varied']['orientation']
    idle = [dims[0], _alt('v2', 'key layout', [], [TRUTH], line='VARY')]  # a dimension that does not matter is not counted
    assert _check(_srec(T, {'structure': 'VARY'}, idle, surfaces=surfaces))['status'] == 'fail'
    mapped = _srec(T, {'structure': 'VARY'}, dims, surfaces=surfaces, mapping={i: (i + 1) % 10 for i in range(10)})
    roles = lines.tables(mapped)['surfaces']
    assert sorted(v['roles'][0]['colour'] for v in roles.values()) == [2, 3, 4, 5]  # surface colours in the final palette


def test_shape_is_fully_mechanical():
    box = lambda h, w, c=1: [[c] * w for _ in range(h)]
    test_in = [[0] * 5 for _ in range(5)]; test_in[0][0] = test_in[0][1] = test_in[1][0] = 2
    task = {'train': [{'input': box(5, 5, 0), 'output': box(1, 1)}, {'input': box(5, 5, 0), 'output': box(2, 2)},
                      {'input': box(5, 5, 0), 'output': box(3, 3)}], 'test': [{'input': test_in, 'output': box(4, 3)}]}
    ok = _check({'study': {'structure': 'SHAPE'}, 'complete_task': task})
    assert ok['status'] == 'pass' and ok['frame']['distinct_training_sizes'] == 3 and ok['frame']['tests'][0]['naive'] is None
    assert ok['frame']['quantity_check'] == 'not machine-read'  # S9: recorded, out of the pass rate
    assert structures.naive_frames(task, 1) == {(5, 5): 'test input size', (1, 1): 'train_1 output size',
                                                (2, 2): 'train_2 output size', (3, 3): 'train_3 output size'}
    boxed = deepcopy(task); boxed['test'][0]['output'] = box(2, 2)
    assert _check({'study': {'structure': 'SHAPE'}, 'complete_task': boxed})['status'] == 'fail'
    same = deepcopy(task); same['train'][2]['output'] = box(2, 2); same['test'][0]['output'] = box(4, 4)
    assert _check({'study': {'structure': 'SHAPE'}, 'complete_task': same})['status'] == 'fail'  # three sizes needed
    rec = {'study': {'structure': 'SHAPE'}, 'complete_task': task}
    rows_ = lines.point_results(rec, {'predicted_test_outputs': [box(5, 5)]})
    assert rows_ == [{'part': 'structure', 'id': 'frame', 'kind': 'test input size', 'test': 1, 'eligible': True, 'captured': True,
                      'share': None, 'equals_alternative': True}]
    assert lines.on_points(rec, {'predicted_test_outputs': [box(4, 4)]}) == []  # a wrong answer of another size
    assert lines.on_points(rec, {'predicted_test_outputs': [box(2, 2)]}) == ['structure']  # the largest object's box


def test_relevance_readings_and_object_contrast_in_two_pairs():
    use = _alt('u1', 'use everything', [_two(OUT[0]), OUT[1], OUT[2]], [_two(TRUTH)], line='RELEVANCE')
    ignore = _alt('i1', 'ignore all', [OUT[0], _two(OUT[1]), OUT[2]], [[[5] * 7]], line='RELEVANCE')
    obj = lambda g, o, out, colour=3: {'grid': g, 'object': o, 'colour': colour, 'size': 1, 'shape': 'dot', 'region': 'middle',
                                       'relation': 'touches' if out == 'acts' else 'apart', 'outcome': out}
    two = [obj('train_1', 'a', 'acts'), obj('train_1', 'b', 'inert'), obj('train_2', 'c', 'acts'), obj('train_2', 'd', 'inert'),
           obj('test_1', 'e', 'acts')]
    ok = _check(_srec(T, {'structure': 'RELEVANCE'}, [use, ignore], objects=two))
    assert ok['status'] == 'pass' and ok['objects']['contrast_grids'] == 2
    one = two[:2] + [obj('train_2', 'c', 'acts', 4), obj('train_2', 'd', 'inert', 5), two[-1]]
    assert _check(_srec(T, {'structure': 'RELEVANCE'}, [use, ignore], objects=one))['objects']['status'] == 'fail'
    decoration = _alt('u1', 'use everything', OUT, [TRUTH], line='RELEVANCE')  # inert things that are mere decoration
    assert _check(_srec(T, {'structure': 'RELEVANCE'}, [decoration, ignore], objects=two))['status'] == 'fail'
    assert _check(_srec(T, {'structure': 'RELEVANCE'}, [use], objects=two))['status'] == 'incomplete'


def test_structure_records_stay_apart_from_the_lines_points_and_merge_per_input():
    point = _pt('p1', [[0, 0, 4], [0, 1, 0]], kind='settled point')
    rec = _srec(T, {'line': 'MARKS', 'structure': 'ELIMINATE'}, [R1, R2], points=[point])
    table = lines.tables(rec)
    assert [p['id'] for p in table['points']] == ['p1'] and sorted(table['alternatives']) == ['r1', 'r2']  # S1: never mixed
    check = lines.episode_check(rec, rec['complete_task'])
    assert set(check) == {'line', 'structure'} and 'alternatives' not in check['line']
    # a repair re-writes the pairs it changes: train_2 replaced, the others kept from the realizer
    new = deepcopy(T); new['train'][1] = {'input': _row(5), 'output': _row(0, 5)}
    repaired = {'source': 'repair', 'call_id': 'x', 'palette': 'final', 'problems': [], 'train': None,
                **{k: None for k in lines.LEDGERS}, 'units': None, 'surfaces': None,
                'alternatives': [_alt('r1', 'rival', [None, _two(_row(0, 5)), None], [_two(TRUTH)]),
                                 _alt('r2', 'rival', [None, _row(0, 5), None], [_two(TRUTH)])],
                'inputs': {'train': [p['input'] for p in new['train']], 'test': [p['input'] for p in new['test']]}}
    rec['weakness']['emissions'].append(repaired)
    rec['complete_task'] = new
    after = lines.episode_check(rec, new)['structure']
    assert after['status'] == 'pass' and after['alternatives']['killers'] == [2, 3]
    first = lines.episode_check(rec, T, lines.tables(rec, first=True))['structure']
    assert first['status'] == 'pass'  # the first-version checks read the realizer's records alone


def test_weakness_outputs_with_structure_records_are_lenient():
    from support.fixtures import episode
    plan, realized = episode('dtest0000000000000911', train=3)
    good = {'training_outputs': None, 'points': None, 'overlay_points': None, 'objects': None, 'assembly': None, 'killer': None,
            'alternatives': [R1], 'units': [{'grid': 'train_1', 'labels': [[1, 0]]}],
            'surfaces': [{'grid': 'test_1', 'roles': [{'role': 'a', 'colour': 1}], 'key_box': [0, 0, 1, 1], 'orientation': None, 'scale': None}]}
    extra = lines.parse_realizer({**realized, 'weakness_outputs': good})['weakness_outputs']
    assert extra['alternatives'] == [R1] and '_problems' not in extra
    bad = {**good, 'alternatives': [R1, {**R2, 'test': 'x'}], 'surfaces': [{'grid': 'x'}]}
    extra = lines.parse_realizer({**realized, 'weakness_outputs': bad})['weakness_outputs']
    assert extra['alternatives'] == [R1] and extra['surfaces'] is None and len(extra['_problems']) == 2
    props = lines.REALIZER_SCHEMA['properties']['weakness_outputs']['anyOf'][0]['properties']
    assert {'alternatives', 'units', 'surfaces'} <= set(props) and 'alternatives' in lines.REPAIR_SCHEMA['properties'][
        'weakness_outputs']['anyOf'][0]['properties']
    assert lines.emission_problems({'structure': 'PARSE'}, {**realized, 'status': 'realized'}, {'alternatives': [R1]}) == ['units missing']


# ---------------------------------------------------------------- capture, routing, notes and fields


def test_grid_capture_uses_the_80_percent_rule():
    truth = [[0] * 5]
    alt = [[1, 1, 1, 1, 1]]
    assert structures.grid_capture(alt, truth, [[1, 1, 1, 1, 0]])['captured'] is True  # 4 of 5
    assert structures.grid_capture(alt, truth, [[1, 1, 1, 0, 0]])['captured'] is False
    assert structures.grid_capture(alt, truth, truth) is None
    other = [[1, 1], [1, 1]]  # a rival in another frame
    assert structures.grid_capture(other, truth, [[1, 1], [1, 0]]) == {'eligible': True, 'captured': False, 'share': 0.75}
    assert structures.grid_capture(other, truth, [[1, 1], [1, 1]])['captured'] is True
    assert structures.grid_capture(other, truth, [[2] * 5])['captured'] is False


def test_structure_capture_routes_to_a_clarifying_repair_with_the_killer(monkeypatch):
    rec = _srec(T, ELIM, [R1, R2], plan=_plan(['Survivor: x; each rival rule fails on one pair.'], ['Rival 1: move far.', 'Rival 2: b.']))
    rival = {'response': {'predicted_test_outputs': [_two(TRUTH)], 'primary_inferred_rule': 'far'}}
    right = {'response': {'predicted_test_outputs': [TRUTH], 'primary_inferred_rule': 'two'}}
    rows_ = lines.point_results(rec, rival['response'])
    assert {(r['part'], r['id'], r['captured'], r['equals_alternative']) for r in rows_} == {('structure', 'r1', True, True),
                                                                                           ('structure', 'r2', True, True)}
    assert lines.on_points(rec, rival['response']) == ['structure'] and lines.on_points(rec, right['response']) == []
    assert lines.capture_route(rec, [rival, rival]) == 'structure' and lines.capture_route(rec, [rival, right]) is None
    instruction, (challenge,), evidence = lines.clarify(rec, 'structure', [rival, rival])
    assert challenge['training_pair_ids'] == ['train_2'] and evidence['name'] == 'ELIMINATE'
    assert evidence['shortcut_rule'] == 'n' and lines.body('lines/ELIMINATE.clarify_note.md').split('{')[0] in instruction
    assert 'rule out: n. Make the pair' in instruction
    orders = _srec(T, {'structure': 'RECOMBINE'}, [_alt('o1', 'order', [], [_two(TRUTH)], line='RECOMBINE')])
    assert lines.point_results(orders, rival['response']) == []  # the order records are the truth by design, never a rival


def test_review_policies_and_repair_notes_for_structures():
    plan = _plan(['Survivor: x; each rival rule fails on one pair.'], ['Rival 1: a.', 'Rival 2: b.'])
    rec = _srec(T, ELIM, [R1, R2], plan=plan)
    blind = [{'response': {'predicted_test_outputs': [TRUTH]}}]
    assert lines.review(rec, blind) == (['ELIMINATE'], None)
    assert lines.body('lines/ELIMINATE.repair_note.md') in lines.repair_notes(rec)
    emit = lines.repair_notes(rec)[-1]
    assert '- alternatives:' in emit and '- units:' not in emit and '- points:' not in emit
    none = _srec(T, ELIM, plan=_plan(['Structure: none — one law.']))
    assert lines.review(none, blind) == ([], 'no_invariant') and lines.body('lines/ELIMINATE.repair_note.md') not in lines.repair_notes(none)
    rplan = _plan(['Combination: a and b each act apart in the pairs and commute; the test combines them.'],
                  ['Single law: a.', 'Single law: b.', 'Replay: c.'])
    failing = _srec(T, {'structure': 'RECOMBINE'}, [_alt('o1', 'order', [], [TRUTH], line='RECOMBINE')], plan=rplan)
    assert lines.review(failing, blind) == ([], 'check_not_passed')  # policy_needs_check (S6)
    parse = _srec(T, {'line': 'MARKS', 'structure': 'PARSE'}, plan=_plan(['Unit (lines): a line.'], ['Naive parse: cells.']))
    emit = lines.repair_notes(parse)[-1]
    assert '- alternatives:' in emit and '- units:' in emit and '- surfaces:' not in emit


def test_fields_carry_the_structure_only_when_drawn(monkeypatch):
    monkeypatch.setattr(gate, 'current_blind', lambda tid, record, calls: [])
    plan = _plan(['Survivor: x; each rival rule fails on one pair.'], ['Rival 1: a.', 'Rival 2: b.'])
    rec = _srec(T, {**ELIM, 'structure_call': 'ELIMINATE'}, [R1, R2], plan=plan)
    rec['weakness']['plan_check'] = lines.plan_check(None, None, plan, structure='ELIMINATE')
    rec['blind_results'] = [{'response': {'predicted_test_outputs': [_two(TRUTH)]}}]
    w = lines.fields('t', rec, [])['weakness']
    assert w['structure'] == 'ELIMINATE' and w['structure_tag'] == 'kept' and w['line'] is None and w['line_tag'] is None
    assert w['first_blind']['on_points'] == ['structure'] and w['final_version']['structure']['status'] == 'pass'
    assert [a['id'] for a in w['structure_alternatives']] == ['r1', 'r2'] and w['structure_alternatives'][0]['test'] == [_two(TRUTH)]
    opted = deepcopy(rec); opted['concrete_plan'] = _plan(['Structure: none — one law.'])
    opted['weakness']['plan_check'] = lines.plan_check(None, None, opted['concrete_plan'], structure='ELIMINATE')
    assert lines.fields('t', opted, [])['weakness']['structure_tag'] == 'opted_out'
    excluded = {'study': {'line': 'SETTLE', 'structure_call': 'ELIMINATE', 'structure_excluded': {'structure': 'ELIMINATE'}}}
    w = lines.fields('t', excluded, [])['weakness']
    assert w['structure'] is None and w['structure_excluded'] == {'structure': 'ELIMINATE'} and w['structure_tag'] is None
    plain = lines.fields('t', {'study': {'line': 'SETTLE'}}, [])['weakness']
    assert not any(k.startswith('structure') for k in plain) and 'default_readings' not in plain  # unchanged without one


# ---------------------------------------------------------------- the default-reading call


def test_default_reading_requests_and_results():
    rec = _srec(T, ELIM, [R1, R2], plan=_plan(['Survivor: x; each rival rule fails on one pair.'], ['Rival 1: a.', 'Rival 2: b.']))
    reqs = structures.default_requests(rec)
    assert [(r['key'], r['train'], r['basis']) for r in reqs] == [('rival-r1', [1, 3], 'ELIMINATE'), ('rival-r2', [1, 2], 'ELIMINATE')]
    assert structures.default_task(rec, reqs[0]) == {'train': [T['train'][0], T['train'][2]], 'test': [{'input': T['test'][0]['input']}]}
    hit = structures.default_result(rec, reqs[0], {'predicted_test_outputs': [_two(TRUTH)]})
    assert hit['captured'] and hit['equals_rival'] and not hit['equals_truth']
    near = deepcopy(_two(TRUTH)); near[0][6] = 0  # one of two rival cells: half, not 80%
    assert not structures.default_result(rec, reqs[0], {'predicted_test_outputs': [near]})['captured']
    right = structures.default_result(rec, reqs[0], {'predicted_test_outputs': [TRUTH]})
    assert right['equals_truth'] and not right['captured']
    task = _task([([[1, 2, 0, 0]], [[3, 3, 0, 0]]), ([[1, 0, 2, 0]], [[3, 0, 3, 0]])], [([[0, 1, 2, 0]], [[0, 3, 3, 0]])])
    naive = _alt('n1', 'naive parse', [[[3, 3, 0, 0]], [[3, 0, 0, 0]]], [[[0, 3, 0, 0]]], line='PARSE')
    preq, = structures.default_requests(_srec(task, {'structure': 'PARSE'}, [naive]))
    assert preq['key'] == 'naive' and preq['train'] == [1] and preq['expected'] == [[[0, 3, 0, 0]]]  # S2: the fitting pairs
    from test_weakness_lines import _settle_record, OK_TRAIN, OK_POINT
    settle = _settle_record(OK_TRAIN, OK_POINT)
    sreq, = structures.default_requests(settle)
    assert sreq['key'] == 'settle' and sreq['train'] == [1, 2] and sreq['expected'] == [[[4, 0, 0, 0, 0, 0, 0]]]
    assert structures.default_requests(_srec(T, {'line': 'MARKS'})) == []  # only SETTLE, ELIMINATE and PARSE


def test_the_default_reading_call_is_routed_bound_and_validated(weak, monkeypatch):
    from arcgen.astra_broad_arm_program_20260908.runtime import category_for
    assert category_for('conditional_blind', 't-v1-rival-r1-default-reading') == 'conditional_evidence_repair'
    assert lines.default_binding() == {}  # off unless the run records default readings
    weak['default_reading'] = True
    assert list(lines.default_binding()) == [lines.DEFAULT_STAGE]
    assert lines.default_binding()[lines.DEFAULT_STAGE].purposes == ('conditional_blind',)
    assert lines.DEFAULT_STAGE not in contracts.Bindings().stage_bindings  # never an authoring contract stage
    monkeypatch.setitem(routing.CALLS, 'default_reading', routing.binding(config.SCIENCE['models']['sol'], 'low'))
    assert routing.resolve('conditional_blind', 't-v1abc-rival-r1-default-reading') == {
        'model': config.SCIENCE['models']['sol'], 'reasoning_effort': 'low'}
    with pytest.raises(ValueError):
        routing.resolve('conditional_blind', 't-v1abc-default-reading-x')
    b = contracts.Bindings()
    public = {'train': T['train'][:2], 'test': [{'input': T['test'][0]['input']}]}
    values = {'PUBLIC_TASK': public, 'TASK_ID': 'blind-x', 'SOLVER_ID': 'solver-y'}
    rows_ = lines.default_templates()
    prompt = render_prompt('\n\n'.join(r[0] for r in rows_), values)
    assert prompt.rstrip().endswith(lines.body('lines/default_reading.md'))
    context = {'profile_id': b.profile_id, 'prompt_name': 'default_reading', 'values': values, 'q2_template_hashes': [r[2] for r in rows_]}
    b.validate_request(lines.DEFAULT_STAGE, prompt, context)
    for bad in ({**context, 'values': {**values, 'PUBLIC_TASK': {**public, 'test': [T['test'][0]]}}},  # a hidden answer
                {**context, 'prompt_name': 'blind_solver'}, {**context, 'values': {**values, 'PLAN': 1}}):
        with pytest.raises(Exception):
            b.validate_request(lines.DEFAULT_STAGE, prompt, bad)
    with pytest.raises(Exception):
        b.validate_request(lines.DEFAULT_STAGE, prompt + ' hint', context)
    b.validate_response(lines.DEFAULT_STAGE, {'task_id': 'blind-x', 'solver_id': 'solver-y', 'predicted_test_outputs': [TRUTH]}, context)
    with pytest.raises(Exception):
        b.validate_response(lines.DEFAULT_STAGE, {'task_id': 'blind-x', 'solver_id': 'solver-y', 'predicted_test_outputs': []}, context)


def test_default_readings_are_recorded_never_gate_and_resume(weak, monkeypatch):
    rec = _srec(T, ELIM, [R1, R2], plan=_plan(['Survivor: x; each rival rule fails on one pair.'], ['Rival 1: a.', 'Rival 2: b.']))
    rec['answer_evidence'] = {'status': 'evidence_clear'}
    calls, reserved = [], []

    class Budget:
        def reserve_once(self, key, count):
            reserved.append((key, count)); return True

        def release_task(self, tid):
            reserved.append(('release', tid))
    state = {'semantic_records': {'t': rec}}
    rt = SimpleNamespace(state=state, lock=__import__('threading').RLock(), save=lambda: None)

    def call(runtime, tid, version, request):
        calls.append(request['key'])
        if request['key'] == 'rival-r2':
            raise RuntimeError('provider failed')
        return {'predicted_test_outputs': [_two(TRUTH)]}
    monkeypatch.setattr(gate, '_default_call', call)
    assert gate.default_readings(rt, 't', Budget()) is None  # off unless science.weakness.default_reading
    weak['default_reading'] = True
    value = gate.default_readings(rt, 't', Budget())
    assert calls == ['rival-r1', 'rival-r2'] and reserved == [('t:default-reading', 2), ('release', 't')]
    assert value['results'][0]['captured'] and value['results'][1]['error'] == 'operational'
    assert rec['weakness']['default_readings'] == value and rec['answer_evidence'] == {'status': 'evidence_clear'}
    assert gate.default_readings(rt, 't', Budget()) == value and calls == ['rival-r1', 'rival-r2']  # resumed: no new call
    held = deepcopy(rec); held['answer_evidence'] = {'status': 'evidence_hold'}; held['weakness'].pop('default_readings')
    rt.state['semantic_records']['h'] = held
    assert gate.default_readings(rt, 'h', Budget()) is None  # verified tasks only

    class Paused(BaseException):
        pass
    monkeypatch.setattr(gate, '_default_call', lambda *a: (_ for _ in ()).throw(Paused()))
    fresh = deepcopy(rec); fresh['weakness'].pop('default_readings')
    rt.state['semantic_records']['p'] = fresh
    with pytest.raises(Paused):  # a pause propagates for an exact resume
        gate.default_readings(rt, 'p', Budget())
    assert 'default_readings' not in fresh['weakness']
    monkeypatch.setattr(structures, 'default_requests', lambda record: {}['missing'])
    broken = deepcopy(rec); broken['weakness'].pop('default_readings')
    rt.state['semantic_records']['b'] = broken
    value = gate.default_readings(rt, 'b', Budget())  # a record that cannot be read is noted; the window goes on
    assert value['status'] == 'request_error' and broken['weakness']['default_readings'] == value


# ---------------------------------------------------------------- the idea stage (SHAPE)


def test_shape_is_drawn_at_idea_time():
    call = supply.make_call(arm='SEE', bank_setting='loop', context='c01', seed=1, account='main', window='w', effort='high',
                            idea_model=config.SCIENCE['models']['sol'])
    plain = idea_v1.sketch_request({**call, 'commit': 'explore'})
    shaped = idea_v1.sketch_request({**call, 'commit': 'explore', 'structure': 'SHAPE'})
    text = lines.body('ideas/structures/SHAPE.md')
    assert text not in plain['prompt'] and text in shaped['prompt'] and 'The requirements below apply' in shaped['prompt']
    other = idea_v1.sketch_request({**call, 'commit': 'explore', 'structure': 'PARSE'})
    assert other['prompt'] == plain['prompt']  # the other structure lines act at planning only
    with pytest.raises(IntegrityError):
        idea_v1.sketch_request({**call, 'commit': 'explore', 'structure': 'NOPE'})


# ---------------------------------------------------------------- end to end


SURVIVOR = 'Survivor: the marked cell moves two columns right; each rival rule fails on one pair.'


def _elim_plan(plan):
    plan['public_contract']['invariants'] = list(plan['public_contract']['invariants']) + [SURVIVOR]
    shortcuts = _shortcut('Rival 1: the marked cell moves three columns right except near the start.') + \
        _shortcut('Rival 2: the marked cell moves to the third column after its own.')
    plan['private_execution']['cross_pair_concrete_audit']['strongest_simple_shortcuts'] = shortcuts


def _far(out):
    """The answer with the moved cell one column further: a 2-cell object-level difference."""
    grid = deepcopy(out)
    r, c = next((r, c) for r, row in enumerate(grid) for c, x in enumerate(row) if x == 1)
    grid[r][c], grid[r][c + 1] = 0, 1
    return grid


def _rivals(train, tests):
    outs = [p['output'] for p in train]
    return [{'id': f'r{k}', 'line': 'ELIMINATE', 'kind': 'rival', 'note': f'rival {k}',
             'train': [_far(o) if i == k else o for i, o in enumerate(outs)], 'test': [_far(t) for t in tests]} for k in (1, 2)]


def test_an_eliminate_idea_runs_end_to_end_with_repair_and_default_readings(tmp_path, monkeypatch, weak):
    """An arm idea with ELIMINATE through the mock pipeline: the structure text last in the planner prompt, the rival
    records from the realizer, a repair that re-writes the changed pair's rival grids, the ELIMINATE policy at
    adjudication, and after the task clears one recorded default-reading call per rival without its killer pair."""
    from support import pilot as pilot_fixture
    from arcgen.batch import runtime
    from arcgen.lib.factory_runtime import FactoryModelInvoker
    weak['overlays'] = {}
    weak['default_reading'] = True
    monkeypatch.delitem(config.SCIENCE, 'rung_draw', raising=False)
    monkeypatch.setitem(config.SCIENCE['call_overrides'], 'default_reading', ['sol', 'low'])
    monkeypatch.setitem(routing.CALLS, 'default_reading', routing.binding(config.SCIENCE['models']['sol'], 'low'))
    original = pilot_fixture.episode

    def episode(tid, **kw):
        plan, realized = original(tid, **{**kw, 'train': 3})
        _elim_plan(plan)
        if realized.get('authoring_metadata'):
            realized['authoring_metadata']['plan_fidelity']['preserved_frozen_decisions'] = list(plan['public_contract']['invariants'])
        return plan, realized
    monkeypatch.setattr(pilot_fixture, 'episode', episode)
    monkeypatch.setattr(pilot_fixture, 's', SimpleNamespace(PLAN_STAGE=pilot_fixture.s.PLAN_STAGE, REALIZER_STAGE=lines.REALIZER_STAGE))
    from arcgen.batch import repair
    monkeypatch.setattr(pilot_fixture, 'repair', SimpleNamespace(STAGE=lines.REPAIR_STAGE, REUSE_TESTS=repair.REUSE_TESTS,
                                                                 REUSE_ANSWERS=repair.REUSE_ANSWERS, VERSION=repair.VERSION))
    base = pilot_fixture.limits

    def roomy(account='main', primary=4):
        value = base(account, primary)
        for cats in [value['phases'][0]['categories'], value['accounts'][account]['categories']]:
            cats['conditional_evidence_repair'] = 12
        value['logical_cap'] = value['phases'][0]['logical_cap'] = value['phases'][0]['astra_logical_cap'] = \
            value['accounts'][account]['logical_cap'] = 30
        return value
    monkeypatch.setattr(pilot_fixture, 'limits', roomy)
    runtimes, made = [], runtime.make_runtime

    def capture_runtime(*a, **kw):
        rt = made(*a, **kw); runtimes.append(rt); return rt
    monkeypatch.setattr(runtime, 'make_runtime', capture_runtime)
    empty = {k: None for k in ('training_outputs', 'points', 'overlay_points', 'objects', 'assembly', 'killer', 'units', 'surfaces')}
    readings = []

    def wrap(answer):
        def respond(invoker, stage, cid):
            if stage == lines.DEFAULT_STAGE:
                with pytest.raises(AssertionError):  # the fixture records the call, then knows no such stage
                    answer(invoker, stage, cid)
                prompt = (invoker.calls_dir / stage / cid / 'prompt.md').read_text()
                record = runtimes[-1].state['semantic_records'][cid.split('-')[0]]
                rival = lines.tables(record)['alternatives']['r1' if '-rival-r1-' in cid else 'r2']
                test = record['complete_task']['test'][0]
                readings.append((cid, json.loads(prompt.split('Solver-facing ARC episode:', 1)[1].split('\n\nReturn', 1)[0])))
                answer_ = rival['test'][lines._key(test['input'])] if '-rival-r1-' in cid else test['output']
                return json.dumps({'task_id': re.search(r'Task ID: (\S+)', prompt).group(1),
                                   'solver_id': re.search(r'Solver ID is `([^`]+)`', prompt).group(1),
                                   'primary_inferred_rule': 'Straightforward.', 'predicted_test_outputs': [answer_],
                                   'strongest_plausible_alternatives': [], 'serious_different_output_alternative': False,
                                   'test_requires_unsupported_guessing': False, 'fairness_assessment': 'Fine.', 'confidence': 'high'})
            text = answer(invoker, stage, cid)
            if stage not in (lines.REALIZER_STAGE, lines.REPAIR_STAGE):
                return text
            value = json.loads(text)
            if stage == lines.REALIZER_STAGE:
                task = value['task']
                value['weakness_outputs'] = {**empty, 'alternatives': _rivals(task['train'],
                                                                                value['authoring_metadata']['expected_test_outputs'])}
            else:
                task = value['repaired_realization']['task']
                tests = value['repaired_realization']['authoring_metadata']['expected_test_outputs']
                full = _rivals(task['train'], tests if isinstance(tests, list) else [])
                for a in full:  # only the pair the fixture's repair changes (train_1) is re-written
                    a['train'] = [g if i == 0 else None for i, g in enumerate(a['train'])]
                    a['test'] = [None]
                value['weakness_outputs'] = {**empty, 'changed_pairs': [], 'alternatives': full}
                value['weakness_outputs'].pop('training_outputs')
            return json.dumps(value)
        return respond
    first, calls, *_ = pilot_fixture.run_case(tmp_path, _Proxy(monkeypatch, wrap), 'repair', 10, True, arm='SEE', planner='sol',
                                              study={'structure_call': 'ELIMINATE'})
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    stages = [st for st, _ in calls]
    assert stages.count(lines.DEFAULT_STAGE) == 2 and stages[-2:] == [lines.DEFAULT_STAGE] * 2
    prompts = lambda pattern: [p.read_text() for p in sorted(tmp_path.glob(f'accounts/**/model_calls/{pattern}/*/prompt.md'))]
    plan, = prompts('astra_concrete_plan_v3')
    assert plan.rstrip().endswith(lines.body('lines/ELIMINATE.planner.md').rstrip())
    realizer, = prompts(lines.REALIZER_STAGE)
    assert 'Rival records: give weakness_outputs.alternatives' in realizer
    adjudication, = prompts('*adjudicat*')
    assert lines.policy_text(['ELIMINATE']) in adjudication
    repair_prompt, = prompts(lines.REPAIR_STAGE)
    assert lines.body('lines/ELIMINATE.repair_note.md') in repair_prompt and '- alternatives: every structure record again' in repair_prompt
    assert record['study']['structure'] == 'ELIMINATE' and 'line' not in record['study']
    w = record['blind_performance']['weakness']
    assert w['structure'] == 'ELIMINATE' and w['structure_tag'] == 'kept' and w['line'] is None
    assert w['first_version']['structure']['status'] == 'pass' and w['final_version']['structure']['status'] == 'pass'
    assert w['final_version']['structure']['alternatives']['killers'] == [2, 3] and w['plan_check']['final_issues'] == []
    d = record['weakness']['default_readings']
    assert d['status'] == 'complete' and [(r['key'], r['train'], r['captured'], r['equals_truth']) for r in d['results']] == [
        ('rival-r1', [1, 3], True, False), ('rival-r2', [1, 2], False, True)]
    assert w['default_readings'] == d
    final = record['complete_task']
    for cid, public in readings:  # the killer pair is left out, and no answer, plan or hint is shown
        drop = 2 if '-rival-r1-' in cid else 3
        assert public['train'] == [p for i, p in enumerate(final['train'], 1) if i != drop]
        assert public['test'] == [{'input': t['input']} for t in final['test']]
    ledger = first['runtime'].state['call_ledger'].values()
    assert {c['reasoning_effort'] for c in ledger if c['stage'] == lines.DEFAULT_STAGE} == {'low'}
    assert record['pipeline_usable_before_union_audit'] is True and record['answer_evidence']['status'] == 'evidence_clear_after_review'


# ---------------------------------------------------------------- content additions: QUIET, contact, distortion


def test_quiet_line_plan_fields_and_grid_checks():
    inv = 'Quiet edit: strays of one cell are removed; it acts beside the obvious change.'
    assert lines.plan_issues('QUIET', None, _plan([inv], ['Obvious only: recolour the shapes only.'])) == []
    assert lines.plan_issues('QUIET', None, _plan([inv])) == ['no strongest_simple_shortcuts entry began "Obvious only:"']
    assert lines.plan_issues('QUIET', None, _lplan(risk='Not carried: no quiet law fits.')) == []
    names = [n.split('/')[-1] for n in lines.realizer_components({'line': 'QUIET'})]
    assert names == ['outputs.realizer.md', 'QUIET.realizer.md'] and lines.emits({'line': 'QUIET'}) == ('pairs', 'points')
    texts = [r[0] for r in contracts.Bindings().planning_templates('SEE', (), line='QUIET', structure='VARY')]
    assert 'Quiet edit for this idea (binding' in texts[-2] and 'Structure for this idea: the same roles' in texts[-1]
    quiet = lambda outs: _srec(T, {'line': 'QUIET'}, points=[_pt('p1', [[0, 6, 9]], kind='obvious only')])
    rec = quiet(None)
    obvious = [_two(OUT[0]), _two(OUT[1]), OUT[2]]  # S10: the obvious-only answer differs in at least two pairs
    rec['weakness']['emissions'][0]['train'] = [[p['input'], o] for p, o in zip(T['train'], obvious)]
    check = lines.episode_check(rec, rec['complete_task'])['line']
    assert check['status'] == 'pass' and [p['pair'] for p in check['pairs']['differing_pairs']] == [1, 2]
    rec['weakness']['emissions'][0]['train'] = [[p['input'], o] for p, o in zip(T['train'], [_two(OUT[0]), OUT[1], OUT[2]])]
    assert lines.episode_check(rec, rec['complete_task'])['line']['status'] == 'fail'  # one pair is not a law
    two_points = _srec(T, {'line': 'QUIET'}, points=[_pt('p1', [[0, 6, 9]]), _pt('p2', [[0, 5, 9]])])
    two_points['weakness']['emissions'][0]['train'] = [[p['input'], o] for p, o in zip(T['train'], obvious)]
    assert lines.episode_check(two_points, two_points['complete_task'])['line']['points']['status'] == 'fail'  # one per test


def test_contact_and_distortion_strata():
    b = contracts.Bindings()
    contact = [r[0] for r in b.planning_templates('SEE', (), line='PROCESS', fence='contact')]
    assert contact[-1].split('-->', 1)[1].strip().startswith('Relaxed fence for this idea: contact.')
    assert contact[-2].split('-->', 1)[1].strip() == lines.body('lines/PROCESS.planner.md')
    distortion = [r[0] for r in b.planning_templates('SEE', (), line='RELATION', fence='distortion', structure='RELEVANCE')]
    assert distortion[-2].split('-->', 1)[1].strip().startswith('Relaxed fence for this idea: distortion.')
    assert 'rigid-only' in distortion[-2] and 'Structure for this idea: which look-alikes matter' in distortion[-1]
    assert lines.spec('PROCESS', 'contact')['relaxation']['rungs'] == ['R1', 'R2']
    assert lines.spec('RELATION', 'distortion')['grids'] == lines.LINES['RELATION']['grids']  # RELATION's own checks
    assert lines.LINES['PROCESS']['relax']['interaction_once']['plan'] == {'pairs': [2, 2]}  # re-enabled as it was
