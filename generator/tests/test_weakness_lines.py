"""Weakness lines (2026-10-20): lines as registry data; pinned line modifiers with scoped novelty; the line's planner text composed last,
with its relaxed fence after it; fixed-prefix plan fields checked in code with one shared re-plan; review policies and
repair notes keyed on protected invariants; the declared points, ledgers and shortcut grids the realizer and repairer
emit, checked on every episode version; point capture and the clarifying repair; size arms with the full-scale rung; the
matched control; the EDGE and ROLES draws. Nothing changes when science.weakness is absent."""
from copy import deepcopy
from types import SimpleNamespace
import json, pathlib, re, sys
import pytest
from arcgen.batch import admission, config, contracts, gate, idea_v1, lines, supply
from arcgen.batch.common import IntegrityError, read
from arcgen.lib.astra_authoring_contract_v3 import pair_specific_test_narrative
from arcgen.lib.util import render_prompt
from test_targeted import _Fake, _planner_kw, _call

V1 = idea_v1.PROMPTS.parent
FLOORS = {'R1': {'output_cells': 150, 'computed_cells': 30, 'escalation': 0},
          'R2': {'output_cells': 300, 'computed_cells': 80, 'escalation': 1.3},
          'R3': {'output_cells': 450, 'computed_cells': 150, 'escalation': 1.3},
          'R4': {'output_cells': 800, 'computed_cells': 240, 'escalation': 0}}
MIX = {'R1': 0.25, 'R2': 0.25, 'R3': 0.10, 'R4': 0.15}  # v2 §7: MARKS, EXEMPLAR, RELATION, REFRAME
NO_R4 = {'R1': 0.25, 'R2': 0.30, 'R3': 0.20}  # DEPART, SETTLE, COINCIDE
WEAK = {'lines': {name: {'rungs': dict(NO_R4 if lines.LINES[name].get('r4') is False else MIX)} for name in config.LINE_NAMES},
        'overlays': {'EDGE': {'share': 0.15, 'excluded': ['BOARD']}, 'ROLES': {'share': 0.20}}}
WEAK['lines']['SETTLE']['relaxed'] = {'two_pairs': 0.1, 'in_pair_minority': 0.1, 'single_cell': 0.1, 'ordinal': 0.1}
drawn = lambda s, name: (s.get('weak_overlays') or {}).get(name) == 'drawn'


@pytest.fixture
def weak(monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'weakness', deepcopy(WEAK))
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    return config.SCIENCE['weakness']


def rows(n, start=0, **study):
    return [{'source_candidate_id': f'k{i:05d}', 'private_source': 'ordinary_unconditioned',
             'study': {'primary_arm': 'SEE', **deepcopy(study)}} for i in range(start, start + n)]


def share(out, f):
    return sum(bool(f(r['study'])) for r in out) / len(out)


# ---------------------------------------------------------------- texts and the registry


def test_every_line_has_its_texts_with_a_weakness_header_and_no_task_ids():
    assert set(config.LINE_MODIFIERS) <= set(config.MODIFIER_NAMES) <= set(idea_v1.MODIFIERS)
    assert set(config.LINE_MODIFIERS) == {'DEPART', 'SETTLE', 'MARKS', 'INGRID', 'PRECISE', 'TRACE', 'PROCESS', 'RELATION',
                                          'FORCED', 'REFRAME', 'COINCIDE', 'JUDGE', 'QUIET'}  # QUIET: 2026-10-22
    assert not set(config.LINE_NAMES) & contracts.SLOT_BRIDGES  # line texts are composed last, never as slot lines
    assert not set(config.LINE_NAMES) & set(idea_v1.ARMS)  # EXEMPLAR v2 is INGRID: no line shares an arm's name
    texts = [V1 / n for n in lines.verify_names() if not n.startswith('depart.')]
    texts += [idea_v1.PROMPTS / f for f in ('step2_explore_line.md', 'step3_kept_line.md', 'step3_kept_settle.md')]
    texts += [V1 / 'selector_weakness.md']
    for p in texts:
        head, body = p.read_text().split('-->', 1)
        assert '<!-- prompt-version: weakness-v1-' in head and body.strip(), p
        assert not re.search(r'\b[0-9a-f]{8}\b', body), p  # no task IDs in generation prompts
        assert 'control arm' not in body.lower() and 'treatment arm' not in body.lower()  # private realizer labels
    for m in config.LINE_MODIFIERS:
        assert (idea_v1.PROMPTS / 'modifiers' / f'{m}.md').read_text().split('-->', 1)[1].strip().startswith(
            'Additional requirement for this call:')
    for name in config.LINE_NAMES:  # every line's planner text carries the large-scene guard (owner, 2026-10-20)
        text = (lines.LINE_DIR / f'{name}.planner.md').read_text()
        assert 'last of the binding lines' in text and 'if a scale line above asks for a large scene' in text.lower(), name
    outputs = (lines.LINE_DIR / 'outputs.realizer.md').read_text()  # screen hole 9: records never block a realization
    assert 'never refuse, and never change a grid or a frozen decision, because of them' in outputs
    assert 'at least four' not in (lines.LINE_DIR / 'TRACE.realizer.md').read_text()
    assert 'both of its outcomes shown' in (lines.LINE_DIR / 'TRACE.planner.md').read_text()  # screen hole 4
    assert 'one point per place in each test where a settled point acts' in (lines.LINE_DIR / 'SETTLE.realizer.md').read_text()
    assert 'in at least two training pairs and in the test' in (lines.LINE_DIR / 'EDGE.planner.md').read_text()  # hole 7
    assert 'K5' not in (lines.LINE_DIR / 'COINCIDE.planner.md').read_text() and 'second test' not in \
        (lines.LINE_DIR / 'COINCIDE.planner.md').read_text().lower()  # draft 4: one test input per prompt at eval


def test_the_registry_drives_the_lines(monkeypatch):
    registry = json.loads((config.HERE / 'weakness_lines.json').read_text())
    assert tuple(n for n, v in registry['lines'].items() if v['kind'] == 'line') == config.LINE_MODIFIERS
    assert config.LINE_NAMES == config.LINE_MODIFIERS + (config.LINE_CONTROL,) and config.WEAKNESS_OVERLAYS == ('EDGE', 'ROLES')
    assert lines.verify_registry() and len(lines.verify_names()) > 60
    assert {n for n, s in lines.LINES.items() if s.get('r4') is False} == {'DEPART', 'SETTLE', 'COINCIDE', 'JUDGE'}  # v2 §6.6, §11
    assert 'second_test' not in lines.LINES['DEPART']['relax'] and 'growth' not in lines.LINES['DEPART']['relax']  # draft 4
    assert 'stacked' not in lines.LINES['SETTLE']['relax']  # at most two settled points
    for bad in ({'kind': 'line', 'emits': ['pairs'], 'realizer': 'x'},  # a pairs shortcut needs its shortcut
                {'kind': 'line', 'emits': ['points']},  # an emission needs its realizer addendum
                {'kind': 'line', 'plan': {'nope': 1}}, {'kind': 'line', 'colour': 'x'}, {'kind': 'other'},
                {'kind': 'line', 'opt_out': {'field': 'elsewhere', 'prefix': 'No:'}}, {'kind': 'overlay'},
                {'kind': 'line', 'grids': {'points': {'max_total': 1, 'nope': 2}}},
                {'kind': 'line', 'relax': {'x': {'plan': {}}}}, {'kind': 'line', 'retag_on_break': 'NOPE'}):
        monkeypatch.setitem(lines.LINES, 'NEWLINE', bad)
        with pytest.raises(IntegrityError):
            lines.verify_registry()
    monkeypatch.setitem(lines.LINES, 'NEWLINE', {'kind': 'line', 'invariant': 'New:', 'note': 'lines/NEWLINE.repair_note.md'})
    assert lines.verify_registry()
    with pytest.raises(IntegrityError, match='lines/NEWLINE.planner.md'):  # its files must exist
        lines.verify_names()


def test_dictated_invariants_pass_the_plan_validator():
    """The departure build's bug: an invariant naming "the first test" made every drawn plan fail validation."""
    found = []
    texts = [lines.LINE_DIR / f'{n}.planner.md' for n in (*config.LINE_NAMES, *config.WEAKNESS_OVERLAYS)]
    texts += [V1 / r['text'] for s in lines.LINES.values() for r in (s.get('relax') or {}).values()]
    texts += [contracts.RUNG_DIR / 'ANY.R4.planner.md']
    for path in texts:
        text = path.read_text()
        for template in re.findall(r'"((?:No training pair shows|Settled point|Marks|Worked instance|Decisions|Forced|Trace|'
                                   r'Process|Conserved|Relation|Frame|Coincidence|Judgement|Edge|Regular field|Quiet edit)[^"]*)"', text):
            filled = re.sub(r'<[^>]+>', 'a wall', template)
            assert not pair_specific_test_narrative(filled) and 'first test' not in filled, (path.name, filled)
            found.append(template.split(':')[0].split(' (')[0])
    for name, s in {**lines.LINES, **lines.OVERLAYS}.items():
        if s.get('invariant'):
            assert any(f.startswith(s['invariant'].split(':')[0].split(' (')[0]) for f in found), name
    assert found.count('Conserved') >= 5 and 'Regular field' in found and 'Judgement' in found


def test_the_scoped_novelty_steps_and_the_weakness_selector(monkeypatch):
    plain, _ = _call()
    plain = {**plain, 'commit': 'explore'}
    sketches = [{'id': 's01', 'title': 'T', 'idea': 'I.', 'recipe': 'R.'}]
    near = [[{'reference_id': 'committed:x', 'similarity': 0.5, 'text': 'A task.'}]]

    def steps(call):
        req = idea_v1.sketch_request(call)
        return idea_v1.explore_request(call, req, sketches, near)['prompt'], idea_v1.commit_request(call, req, sketches, near)['prompt']
    monkeypatch.delitem(config.SCIENCE, 'skill_novelty', raising=False)
    explore, commit = steps({**plain, 'idea_modifiers': ['SETTLE']})  # without the run flag: the standard steps
    assert 'would be far from the ones shown' in explore and 'sharing the property is not redundancy' not in commit
    monkeypatch.setitem(config.SCIENCE, 'skill_novelty', True)
    for line in ('MARKS', 'TRACE', 'COINCIDE'):
        explore, commit = steps({**plain, 'idea_modifiers': [line]})
        assert 'still carry this call\'s property' in explore and 'sharing the property is not redundancy' in commit
        assert 'Do not rescue a sketch by adding a twist' in commit
    explore, settle = steps({**plain, 'idea_modifiers': ['SETTLE']})
    assert 'A familiar reading together with the visible condition that breaks it is this call\'s property' in settle
    assert 'adding a twist' not in settle
    line, other = (idea_v1.PROMPTS / 'step3_kept_line.md').read_text(), (idea_v1.PROMPTS / 'step3_kept_settle.md').read_text()
    assert [a for a, b in zip(line.splitlines(), other.splitlines()) if a != b] == line.splitlines()[:1] + [
        x for x in line.splitlines() if x.startswith('- Drop every sketch')]  # only the header and the rescue clause differ
    explore, commit = steps({**plain, 'idea_modifiers': ['QUANTITY']})  # a blind-spot skill keeps its own steps
    assert 'still centre on this call\'s skill' in explore and 'sharing the skill is not redundancy' in commit
    sketch = idea_v1.sketch_request({**plain, 'idea_modifiers': ['MARKS']})['prompt']
    assert sketch.index('## Direction for this call') < sketch.index('a few small marks')
    targeted, weakness = (V1 / 'selector_targeted.md').read_text(), (V1 / 'selector_weakness.md').read_text()
    start = weakness.index('- Many candidates in this run are built to carry a task property')
    end = weakness.index('not a convention the solver must be told.\n', start) + len('not a convention the solver must be told.\n')
    assert weakness[:start].replace('weakness-v1-selector-weakness-2', 'diversity-v1-selector-targeted-2') + weakness[end:] == targeted
    for words in ('every step depends on the one before', 'a visible process', 'look-alikes do not share', 'another frame',
                  'a preference careful people share'):  # review S7: every line's property is named as by design
        assert words in weakness[start:end]
    seen = []

    def fake(path):
        seen.append(pathlib.Path(path).name); raise RuntimeError('stop')
    monkeypatch.setattr(supply, 'load_prompt', fake)
    for weak_run in (True, False):
        if weak_run:
            monkeypatch.setitem(config.SCIENCE, 'weakness', deepcopy(WEAK))
        else:
            monkeypatch.delitem(config.SCIENCE, 'weakness')
        with pytest.raises(RuntimeError):
            supply._selector_prompt({'id': 'w'}, {'payload': {}, 'packed_hash': 'x'})
    assert seen == ['selector_weakness.md', 'selector_skill.md']


# ---------------------------------------------------------------- config, pins, allocation


def _profile(**science):
    value = json.loads((config.PROFILES / 'v1.json').read_text())
    s = value['science']
    s.update(idea_modifiers={'SETTLE': 0.3, 'MARKS': 0.3, 'IGNORE': 0.1}, modifier_slots={'line': ['SETTLE', 'MARKS'], 'relevance': ['IGNORE']},
             modifier_cap=2, modifier_exclusions={}, rungs=FLOORS, skill_novelty=True,
             weakness={'lines': {'SETTLE': {'rungs': NO_R4, 'relaxed': {'two_pairs': 0.2}}, 'MARKS': {'rungs': MIX},
                                 'CONTROL': {'rungs': MIX}},
                       'overlays': {'EDGE': {'share': 0.15, 'excluded': ['BOARD']}, 'ROLES': {'share': 0.2}}})
    for k in ('modifier_arm_caps', 'modifier_bare'):
        s.pop(k, None)
    for key, v in science.items():
        if v is None:
            s.pop(key, None)
        else:
            s[key] = v
    return value


def test_weakness_config_is_validated():
    config.validate(_profile())
    w = _profile()['science']['weakness']
    config.validate(_profile(weakness={**w, 'lines': {**w['lines'], 'SETTLE': {'rungs': {}}}}))  # no size line is allowed
    config.validate(_profile(weakness={k: v for k, v in w.items() if k != 'overlays'}))
    bad = [dict(weakness={**w, 'lines': {**w['lines'], 'NOPE': {'rungs': {}}}}),
           dict(weakness={**w, 'lines': {k: v for k, v in w['lines'].items() if k != 'MARKS'}}),  # a line among the modifiers
           dict(weakness={**w, 'lines': {**w['lines'], 'DEPART': {'rungs': {}}}}),  # a line not among the modifiers
           dict(weakness={**w, 'lines': {**w['lines'], 'SETTLE': {'rungs': {'R1': 0.6, 'R2': 0.5}}}}),
           dict(weakness={**w, 'lines': {**w['lines'], 'SETTLE': {'rungs': {'R5': 0.1}}}}),
           dict(weakness={**w, 'lines': {**w['lines'], 'SETTLE': {'rungs': {'R4': 0.1}}}}),  # SETTLE never takes R4
           dict(weakness={**w, 'lines': {**w['lines'], 'SETTLE': {'ladder': {}}}}),
           dict(weakness={**w, 'lines': {**w['lines'], 'SETTLE': {'rungs': {}, 'relaxed': {'stacked': 0.1}}}}),
           dict(weakness={**w, 'lines': {**w['lines'], 'SETTLE': {'rungs': {}, 'relaxed': {'two_pairs': 0.3, 'ordinal': 0.2}}}}),
           dict(weakness={**w, 'lines': {**w['lines'], 'MARKS': {'rungs': {}, 'relaxed': {'two_pairs': 0.1}}}}),
           dict(weakness={**w, 'overlays': {'EDGE': {'share': 0.6}, 'ROLES': {'share': 0.5}}}),
           dict(weakness={**w, 'overlays': {'EDGE': {'share': True}}}), dict(weakness={**w, 'overlays': {'NOPE': {'share': 0.1}}}),
           dict(weakness={**w, 'overlays': {'EDGE': {'share': 0.1, 'excluded': 'BOARD'}}}), dict(weakness={**w, 'other': 1}),
           dict(weakness={**w, 'size_arms': {'small': {'R1': 0.5}}}),  # replaced by the per-line mixes
           dict(colour_roles='choice', colour_vary_rate=None),  # ROLES rides on the per-idea colour instruction of the mix
           dict(rungs=None),  # ladders need rung floors
           dict(rungs={k: v for k, v in FLOORS.items() if k != 'R4'}),  # a mix with R4 needs R4's floors
           dict(modifier_assignment='per_call'),
           dict(idea_modifiers={'SETTLE': 0.3, 'MARKS': 0.3, 'IGNORE': 0.1, 'BUSY': 0.1},  # MARKS must exclude busy scenes
                modifier_slots={'line': ['SETTLE', 'MARKS'], 'relevance': ['IGNORE'], 'scene': ['BUSY']})]
    for b in bad:
        with pytest.raises(config.ConfigError):
            config.validate(_profile(**b))
    config.validate(_profile(idea_modifiers={'SETTLE': 0.3, 'MARKS': 0.3, 'IGNORE': 0.1, 'BUSY': 0.1},
                             modifier_slots={'line': ['SETTLE', 'MARKS'], 'relevance': ['IGNORE'], 'scene': ['BUSY']},
                             modifier_exclusions={'MARKS': ['BUSY']}))
    lineless = _profile(weakness=None, colour_roles='mix')  # scoped novelty accepts weakness lines in place of skills
    lineless['science']['rungs'] = {k: v for k, v in FLOORS.items() if k != 'R4'}
    config.validate(lineless)
    config.validate(_profile(colour_roles='choice', colour_vary_rate=None, weakness={**w, 'overlays': {'EDGE': {'share': 0.2}}}))
    with pytest.raises(config.ConfigError):
        config.validate(_profile(weakness=None, idea_modifiers={'IGNORE': 0.1}, modifier_slots={'relevance': ['IGNORE']}))
    assert config.ALL_RUNGS == ('R1', 'R2', 'R3', 'R4') and config.SIZE_ARMS == {'small': ('R1',), 'large': ('R2', 'R3', 'R4')}
    assert config.RUNGS == ('R1', 'R2', 'R3')  # families and general rung draws keep R1-R3
    with pytest.raises(config.ConfigError):  # R4 floors only in a weakness run
        config.validate(_profile(weakness=None, colour_roles='mix'))
    without = _profile(weakness=None, colour_roles='mix')
    without['science']['rungs'] = {k: v for k, v in FLOORS.items() if k != 'R4'}
    config.validate(without)
    for draw in ({'rung': 'R4', 'rate': 0.2, 'excluded': []}, {'ladder': {'R3': 0.2, 'R4': 0.1}, 'excluded': []}):
        with pytest.raises(config.ConfigError):  # the general rung draw never takes R4
            config.validate(_profile(rung_draw=draw))


def test_spec_rows_pin_lines_mark_controls_and_fix_size_and_fence(monkeypatch):
    from arcgen.batch.allocations.make_allocation import build
    from arcgen.batch.allocation import assign_modifiers
    spec = {'revision': 'r', 'seed': 1, 'windows': {'w': {'account': 'main', 'wave': 'A', 'context': 'c01', 'commit': 'explore'}},
            'rows': [['w', 'SEE', None, 'L', 'sol', 'high'],
                     ['w', 'GEO', None, 'L', 'sol', 'high', {'skill': 'SETTLE', 'size': 'R3', 'fence': 'two_pairs'}],
                     ['w', 'SEE', None, 'L', 'sol', 'high', {'skill': ['MARKS', 'IGNORE'], 'size': 'small'}],
                     ['w', 'GEO', None, 'L', 'sol', 'high', {'control': True, 'size': 'R4'}]]}
    calls = build(spec, {'sol': 'gpt-6.1-sol'}, {}, lambda a, i: True)['calls']
    assert [c.get('skill') for c in calls] == [None, 'SETTLE', ['MARKS', 'IGNORE'], None]
    assert [c.get('control') for c in calls] == [None, None, None, True]
    assert [c.get('size') for c in calls] == [None, 'R3', 'small', 'R4'] and calls[1]['fence'] == 'two_pairs'
    old = build({**spec, 'rows': spec['rows'][:1]}, {'sol': 'gpt-6.1-sol'}, {}, lambda a, i: True)['calls'][0]
    assert old['private_id'] == calls[0]['private_id']  # rows without the new keys keep their IDs
    for bad in ({'control': True, 'skill': 'SETTLE'}, {'control': 'yes'}, {'size': 'huge'}, {'skill': []},
                {'skill': ['MARKS', 'MARKS']}, {'fence': 3}):
        with pytest.raises(ValueError):
            build({**spec, 'rows': [['w', 'GEO', None, 'L', 'sol', 'high', bad]]}, {'sol': 'm'}, {}, lambda a, i: True)
    pins = [{'opaque_id': f'c{k}', 'arm': arm, **({'pinned_modifiers': p} if p else {})} for k, (arm, p) in enumerate(
        [('GEO', ['SETTLE']), ('SEE', ['MARKS', 'IGNORE']), ('GEO', None), ('SEE', ['MARKS'])])]
    rates = {'SETTLE': 0.25, 'MARKS': 0.5, 'IGNORE': 0.25}
    assert assign_modifiers(pins, rates, {'line': ['SETTLE', 'MARKS'], 'relevance': ['IGNORE']}, 2, {'MARKS': ['BOARD']}) == {}
    assert [c['idea_modifiers'] for c in pins] == [['SETTLE'], ['IGNORE', 'MARKS'], [], ['MARKS']]
    monkeypatch.setattr(supply, 'idea_validator', lambda call, req: (lambda value: None))
    base = {'arm': 'GEO', 'effort': 'high', 'recipe_id': 'r', 'bank_setting': 'loop', 'account': 'main', 'private_id': 'p',
            'replicate': 1, 'idea_model': 'm', 'comparison_block': None, 'comparison_role': None}
    priv = {'source_class': 'ordinary_unconditioned', 'conditioned': False, 'bank_pack_hash': None, 'base_search': 'v1_loop',
            'pinning': 'none', 'author_pack_base_hash': None, 'pin_manifest_hash': None}
    window = {'id': 'w', 'context': 'c01', 'wave': 'A', 'calls': [
        {**base, 'opaque_id': 'a', 'line_control': True, 'size_arm': 'R4'}, {**base, 'opaque_id': 'b'},
        {**base, 'opaque_id': 'c', 'size_arm': 'small', 'fence': 'strict'}]}
    out = supply.source_rows(window, {k: {'private': priv} for k in 'abc'}, {k: {'ideas': [{'candidate_id': 'k' + k}]} for k in 'abc'})
    assert out['ka']['study']['line_control'] is True and (out['ka']['study']['size_arm'], out['ka']['study']['size_rung']) == ('large', 'R4')
    assert not {'line_control', 'size_arm', 'size_rung', 'fence_call'} & set(out['kb']['study'])
    assert out['kc']['study']['size_arm'] == 'small' and out['kc']['study']['fence_call'] == 'strict'


# ---------------------------------------------------------------- admission draws


def test_admission_sets_line_fence_size_arm_and_rung(weak):
    large = admission.configured_routes(None, rows(3000, idea_modifiers=['MARKS'], size_arm='large'))
    assert {r['study']['line'] for r in large} == {'MARKS'} and {r['study']['size_arm'] for r in large} == {'large'}
    for rung, p in (('R2', .25 / .5), ('R3', .10 / .5), ('R4', .15 / .5)):  # the line's mix renormalised inside the arm
        assert abs(share(large, lambda s: s.get('rung') == rung) - p) < 0.03, rung
    small = admission.configured_routes(None, rows(3000, idea_modifiers=['MARKS'], size_arm='small'))
    assert abs(share(small, lambda s: s.get('rung') == 'R1') - 0.5) < 0.03 and not any(s['study'].get('rung') in ('R2', 'R3', 'R4') for s in small)
    fixed = admission.configured_routes(None, rows(50, line_control=True, size_arm='large', size_rung='R4'))
    assert {(r['study']['line'], r['study']['rung'], r['study']['size_arm']) for r in fixed} == {('CONTROL', 'R4', 'large')}
    drawn_ = admission.configured_routes(None, rows(4000, idea_modifiers=['DEPART']))  # no arm fixed: the line's whole mix
    for rung, p in (('R1', .25), ('R2', .30), ('R3', .20), ('R4', 0), (None, .25)):
        assert abs(share(drawn_, lambda s: s.get('rung') == rung) - p) < 0.03, rung
    assert all(r['study']['size_arm'] == ('large' if r['study'].get('rung') in ('R2', 'R3', 'R4') else 'small') for r in drawn_)
    with pytest.raises(IntegrityError, match='R4'):  # DEPART, SETTLE and COINCIDE never take the full-scale rung
        admission.configured_routes(None, rows(1, idea_modifiers=['SETTLE'], size_arm='large', size_rung='R4'))
    settle = admission.configured_routes(None, rows(4000, idea_modifiers=['SETTLE']))
    for fence in ('two_pairs', 'in_pair_minority', 'single_cell', 'ordinal'):  # the relaxed fences, .10 each
        assert abs(share(settle, lambda s: s.get('fence') == fence) - 0.10) < 0.02, fence
    assert abs(share(settle, lambda s: s.get('fence') == 'strict') - 0.60) < 0.03
    called = admission.configured_routes(None, rows(20, idea_modifiers=['SETTLE'], fence_call='ordinal'))
    assert {r['study']['fence'] for r in called} == {'ordinal'}  # a spec row's fence holds for every idea of the call
    relaxed = admission.configured_routes(None, rows(1200, idea_modifiers=['SETTLE'], fence_call='relaxed'))
    for fence in ('two_pairs', 'in_pair_minority', 'single_cell', 'ordinal'):  # a relaxed call: every idea takes one form
        assert abs(share(relaxed, lambda s: s['fence'] == fence) - 0.25) < 0.04, fence
    strict = admission.configured_routes(None, rows(20, idea_modifiers=['SETTLE'], fence_call='strict'))
    assert {r['study']['fence'] for r in strict} == {'strict'}
    assert not any('fence' in r['study'] for r in admission.configured_routes(None, rows(5, idea_modifiers=['REFRAME'], fence_call='strict')))
    assert {r['study']['fence'] for r in admission.configured_routes(None, rows(5, idea_modifiers=['TRACE'], fence_call='strict'))} == {'strict'}
    for line, fence in (('REFRAME', 'relaxed'), ('TRACE', 'two_pairs'), ('TRACE', 'relaxed')):  # TRACE has no shares here
        with pytest.raises(IntegrityError):
            admission.configured_routes(None, rows(1, idea_modifiers=[line], fence_call=fence))
    assert not any('fence' in r['study'] for r in admission.configured_routes(None, rows(50, idea_modifiers=['MARKS'])))
    with pytest.raises(IntegrityError):
        admission.configured_routes(None, rows(1, idea_modifiers=['SETTLE'], fence_call='binary'))
    plain = admission.configured_routes(None, rows(200))
    assert not any({'line', 'size_arm', 'fence'} & set(r['study']) for r in plain)
    assert admission.configured_routes(None, rows(3000, idea_modifiers=['MARKS'], size_arm='large')) == large  # seeded
    with pytest.raises(IntegrityError):
        admission.configured_routes(None, rows(1, idea_modifiers=['SETTLE', 'MARKS']))


def test_line_ideas_take_no_teaching_line_departure_draw_many_overlay_or_general_rung(weak, monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'teach_draw', {'plain': 1.0})
    monkeypatch.setitem(config.SCIENCE, 'depart_draw', {'test': 1.0})
    monkeypatch.setitem(config.SCIENCE, 'rung_draw', {'ladder': {'R3': 1.0}, 'excluded': []})
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 1.0)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'kinds', ['MANY', 'XFER'])
    line = admission.configured_routes(None, rows(400, idea_modifiers=['SETTLE'], size_arm='small'))
    other = admission.configured_routes(None, rows(400, 400))
    assert not any('teach' in r['study'] or 'depart' in r['study'] for r in line)
    assert all(r['study']['teach'] == 'plain' and r['study']['depart'] == 'test' for r in other)
    assert not any('overlay' in r['study'] for r in line) and {r['private_route'] for r in line} == {'ordinary_current'}  # S9
    assert {r['study']['overlay'] for r in other} == {'MANY', 'XFER'}
    control = admission.configured_routes(None, rows(100, 800, line_control=True, size_arm='small'))
    assert not any('overlay' in r['study'] for r in control)
    assert not any(r['study'].get('rung') in ('R2', 'R3') for r in line)  # its own small arm, not the run's R3 ladder
    assert all(r['study']['rung'] == 'R3' for r in other)


def test_edge_and_roles_draws(weak, monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'canvas_draw', {'margin': 0.5, 'both': 0.5})
    out = admission.configured_routes(None, rows(6000, idea_modifiers=['SETTLE']))
    assert abs(share(out, lambda s: drawn(s, 'EDGE')) - 0.15) < 0.02
    assert abs(share(out, lambda s: drawn(s, 'ROLES')) - 0.20) < 0.02
    assert not any(drawn(r['study'], 'EDGE') and drawn(r['study'], 'ROLES') for r in out)  # one overlay per idea
    assert all(r['study']['colour_roles'] == 'vary' for r in out if drawn(r['study'], 'ROLES'))
    assert all(r['study'].get('canvas') in (None, 'square') for r in out if drawn(r['study'], 'EDGE'))
    assert {r['study'].get('canvas') for r in out if not drawn(r['study'], 'EDGE')} == {'margin', 'both'}
    assert {r['study']['weak_overlays']['EDGE'] for r in out} == {'drawn', 'control'}  # undrawn eligible ideas are the control
    assert {lines.overlay_of(r['study']) for r in out} == {'EDGE', 'ROLES', None}
    board = admission.configured_routes(None, rows(300, primary_arm='BOARD'))  # an excluded arm takes no EDGE draw
    assert not any('EDGE' in r['study']['weak_overlays'] for r in board) and any(drawn(r['study'], 'ROLES') for r in board)
    contrast = admission.configured_routes(None, rows(300, primary_arm='CONTRAST'))  # fixed colours: no ROLES draw
    assert not any('ROLES' in r['study']['weak_overlays'] for r in contrast)
    assert all(r['study'].get('colour_roles') in (None, 'v1') for r in contrast)
    monkeypatch.delitem(config.SCIENCE, 'weakness')
    assert not any('weak_overlays' in r['study'] for r in admission.configured_routes(None, rows(300)))


# ---------------------------------------------------------------- planner composition and the plan check


def test_the_line_is_composed_last_after_its_overlay_with_its_fence_after_it():
    b = contracts.Bindings()
    texts = [r[0] for r in b.planning_templates('SEE', (), family='SIM', rung='R2', canvas='square', teach='plain', depart='test',
                                                line='SETTLE', overlay='EDGE', fence='two_pairs')]
    assert texts[-1].split('-->', 1)[1].strip().startswith('Relaxed fence for this idea: two training pairs')
    assert texts[-2].split('-->', 1)[1].strip().startswith('Settled point for this idea (binding')
    assert 'Border for this idea' in texts[-3] and 'Departure for this idea' in texts[-4] and 'Teaching for this idea' in texts[-5]
    routed = [r[0] for r in b.planning_templates('SEE', ('evidence_scale_addendum',), line='CONTROL', overlay='ROLES')]
    assert 'Episode shape for this idea' in routed[-2] and 'Colour roles for this idea' in routed[-3]  # the route addendum stays last
    assert [r[2] for r in b.planning_templates('SEE', ())] == [r[2] for r in b.planning_templates('SEE', (), line=None, overlay=None)]
    r4 = [r[0] for r in b.planning_templates('SEE', (), rung='R4', line='MARKS')]
    assert 'at least 800' in r4[-2] and 'Regular field:' in r4[-2] and 'Marks for this idea' in r4[-1]
    for bad in ({'line': 'NOPE'}, {'overlay': 'SETTLE'}, {'line': 'EDGE'}, {'line': 'MARKS', 'fence': 'two_pairs'},
                {'fence': 'two_pairs'}, {'family': 'SIM', 'rung': 'R4'}, {'rung': 'R4'}, {'family': 'SIM', 'rung': 'R4', 'line': 'MARKS'},
                {'rung': 'R4', 'line': 'JUDGE'}, {'rung': 'R4', 'line': 'SETTLE'}):
        with pytest.raises(Exception):
            b.planning_templates('SEE', (), **bad)


def _lplan(tid='t', notes=('Consistent: a.', 'Settles: b.', 'Consistent: c.'), invariants=(), rules=(), risk='Little.',
           test_note='Same kind.', train=(4, 4), test=(4, 4), roles=(), why='Supported.'):
    dims = lambda a: {'height': a[0], 'width': a[1]}
    pairs = [{'input_dimensions': dims(train), 'output_dimensions': dims(train), 'what_it_helps_communicate': n} for n in notes]
    return {'candidate_id': tid, 'status': 'ready',
            'public_contract': {'training_pairs': pairs, 'invariants': ['Coordinates are zero-indexed.', *invariants],
                                'test_inputs': [{'input_dimensions': dims(test), 'what_is_new_relative_to_training': test_note}],
                                'episode_overview': {'functional_color_roles': list(roles)}},
            'private_execution': {'test_outputs': [{'output_dimensions': dims(test), 'why_one_output_is_best_supported': why}],
                                  'cross_pair_concrete_audit': {'strongest_simple_shortcuts': [{'rule': r} for r in rules],
                                                                'remaining_ambiguity_risk': risk}}}


SETTLE_INV = ('Settled point: moving each piece two steps gives every training output except in the pair(s) showing a wall, '
              'where it differs visibly; nothing else sets those pairs apart.')


def _grow(plan, size=(6, 6)):
    grown = deepcopy(plan)
    grown['public_contract']['test_inputs'][0]['input_dimensions'] = {'height': size[0], 'width': size[1]}
    return grown


def test_plan_fields_for_every_line_and_fence():
    good = _lplan(invariants=[SETTLE_INV], rules=['Shortcut: move two steps.'])
    assert lines.plan_issues('SETTLE', None, good) == []
    assert lines.plan_issues('SETTLE', None, _lplan(rules=['Shortcut: x.']))[0].startswith('it had no invariant')
    assert any('pairs began "Settles:"' in i for i in lines.plan_issues('SETTLE', None, _lplan(
        notes=('Settles: a.', 'Settles: b.', 'Consistent: c.', 'Consistent: d.'), invariants=[SETTLE_INV], rules=['Shortcut: x.'])))
    assert lines.plan_issues('SETTLE', None, _lplan(notes=('Settles: a.', 'Settles: b.', 'Consistent: c.', 'Consistent: d.',
                                                           'Consistent: e.'), invariants=[SETTLE_INV], rules=['Shortcut: x.'])) == []
    two = _lplan(notes=('Settles: a.', 'Consistent: b.'), invariants=[SETTLE_INV], rules=['Shortcut: x.'])
    assert any('2 training pairs, not 3 to 5' in i for i in lines.plan_issues('SETTLE', None, two))
    assert lines.plan_issues('SETTLE', None, two, 'two_pairs') == []  # the relaxed fence's own counts
    assert any('not 2' in i for i in lines.plan_issues('SETTLE', None, good, 'two_pairs'))
    inside = _lplan(notes=('Settles inside: a.', 'Settles inside: b.', 'Consistent: c.'), invariants=[SETTLE_INV], rules=['Shortcut: x.'])
    assert lines.plan_issues('SETTLE', None, inside, 'in_pair_minority') == [] and lines.plan_issues('SETTLE', None, inside)
    assert lines.plan_issues('SETTLE', None, _lplan(risk='Not carried: no default reading fits.')) == []  # an honest opt-out
    dep = 'No training pair shows a key on the left; the test does, and every law its answer uses there acts in the pairs.'
    depart = _lplan(notes=('a.', 'b.'), invariants=[dep], rules=['Shortcut: replay train_2.'],
                    test_note='Departure: turned frame; the key sits on the left; closest pair train_2.')
    assert lines.plan_issues('DEPART', None, depart) == [] and lines.plan_issues('DEPART', None, _grow(depart, (4, 6))) == []  # 1.5x
    assert any('1.5' in i for i in lines.plan_issues('DEPART', None, _grow(depart, (5, 5))))  # one growth rule: area 1.5x
    assert lines.plan_issues('DEPART', None, _lplan(test_note='No departure: none settles.')) == []
    assert lines.form('DEPART', lines.declared('DEPART', depart))['label'] == 'turned frame'
    marks = lambda vals, inv='Marks: loose dots set how many copies are drawn; each is at most three cells, apart from other material, and their value differs between pairs.': _lplan(
        notes=tuple(f'Marks set: {v}.' for v in vals), invariants=[inv], rules=['Shortcut: use two steps.'])
    assert lines.plan_issues('MARKS', None, marks(['2 steps', '3 steps', '4 steps'])) == []
    assert any('distinct values after "Marks set:"' in i for i in lines.plan_issues('MARKS', None, marks(['2 steps', '3 steps', '3 steps'])))
    assert lines.plan_issues('MARKS', None, marks(['on', 'off', 'on']), 'binary') == []  # a switch needs two values only
    assert lines.plan_issues('MARKS', None, marks(['left', 'right', 'left'], 'Marks: a stub sets the direction; each is at most '
                             'three cells, apart from other material, and their value differs between pairs; the value is used as it is.')) == []
    assert lines.form('MARKS', lines.declared('MARKS', marks(['1', '2', '3'])))['label'] == 'count'
    inv = ('Worked instance (in place): each input holds one worked instance of its own change; what it shows differs between '
           "pairs, and no training pair's instance shows the change the test's does.")
    ingrid = lambda test, i=inv: _lplan(notes=('Instance shows: a turn.', 'Instance shows: a flip.', 'Instance shows: a shift.'),
                                        invariants=[i], rules=['Shortcut: borrow.'], test_note=f'Instance shows: {test}.')
    assert lines.plan_issues('INGRID', None, ingrid('a doubling')) == []
    assert any("equals a training pair's" in i for i in lines.plan_issues('INGRID', None, ingrid('a flip')))
    assert lines.form('INGRID', lines.declared('INGRID', ingrid('a doubling')))['label'] == 'in place'
    assert lines.form('INGRID', lines.declared('INGRID', ingrid('x', inv.replace('(in place)', '(progression)')), 'progression'))['label'] == 'progression'
    assert lines.plan_issues('PRECISE', None, _lplan(invariants=['Decisions: stops and crossings; each kind acts in at least two '
                                                                 'training pairs.'])) == []
    assert lines.plan_issues('FORCED', None, _lplan(invariants=['Forced: x; every step follows from what is already placed.'],
                                                    rules=['Shortcut: by position.'])) == []
    assert lines.plan_issues('TRACE', None, _lplan(invariants=['Trace: a route; at a crossing it goes straight.'], rules=['Shortcut: nearest.'])) == []
    process = _lplan(invariants=['Process: things fall until they rest.'], rules=['Shortcut: each place alone.'])
    assert lines.plan_issues('PROCESS', None, process) == ['it had no invariant beginning "Conserved:"']
    process['public_contract']['invariants'].append('Conserved: the number of non-background cells is the same in input and output.')
    assert lines.plan_issues('PROCESS', None, process) == []
    assert lines.plan_issues('RELATION', None, _lplan(invariants=['Relation: what a thing touches decides.'], rules=['Shortcut: colour.'])) == []
    assert lines.plan_issues('REFRAME', None, _lplan(invariants=['Frame: diagonals from the marker.'], rules=['Shortcut: rows.'])) == []
    coincide = _lplan(invariants=['Coincidence: in every pair size and count agree; the test separates them, and a tie settles it.'],
                      rules=['Shortcut: by size.'], why='Killer: no single answer, since two things tie under the rival.')
    assert lines.plan_issues('COINCIDE', None, coincide) == [] and lines.declared('COINCIDE', coincide)['private_value'] == 'K1'
    assert lines.form('COINCIDE', lines.declared('COINCIDE', coincide))['label'] == 'K1'
    assert any('Killer:' in i for i in lines.plan_issues('COINCIDE', None, {**coincide, 'private_execution': {
        **coincide['private_execution'], 'test_outputs': [{'why_one_output_is_best_supported': 'Killer: second test.'}]}}))  # no K5
    control = _lplan(notes=('a.', 'b.', 'c.'), rules=['Fill every gap.'])
    assert lines.plan_issues('CONTROL', None, control) == [] and lines.declared_shortcut('CONTROL', control) == 'Fill every gap.'
    assert lines.plan_issues('CONTROL', None, _grow(control, (12, 12))) == []  # draft 4: the control takes the lines' sizes
    assert lines.plan_issues('CONTROL', None, _lplan(notes=('a.', 'b.'), rules=['x']))  # three to five pairs
    edge = _lplan(notes=('Cut by the border: a piece.', 'b.', 'c.'), invariants=['Edge: things cut by the grid\'s border move like '
                  'whole ones; they occur in the training pairs and the test.'], rules=['Edge shortcut: ignore cut things.'])
    assert lines.plan_issues(None, 'EDGE', edge) == [] and len(lines.plan_issues(None, 'EDGE', _lplan())) == 3
    assert lines.plan_issues(None, 'EDGE', _lplan(invariants=['Edge: none, because the border is a wall here.'])) == []
    roles = _lplan(roles=[{'role': 'ink', 'color': 3, 'functional_meaning': 'Changes role: a wall in train_1, a piece in train_2.'}])
    assert lines.declared('ROLES', roles)['role_meanings'] == ['ink'] and lines.plan_issues(None, 'ROLES', _lplan()) == []
    regular = _lplan(invariants=[SETTLE_INV, 'Regular field: distance bands from the marker.'])
    assert lines.declared('SETTLE', regular)['regular_field'].startswith('Regular field:')


def _record(study, **extra):
    return {'private_route': 'ordinary_current', 'selection_note': {'useful_contribution': 'Distinct.'},
            'study': {'primary_arm': 'SEE', 'colour_roles': 'v1', 'pose': False, **study}, **extra}


def test_a_plan_missing_its_line_fields_is_planned_once_more_and_resumes(tmp_path, monkeypatch):
    from arcgen.batch.runtime import Runtime
    from arcgen.batch import rungs
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    tid = 'dtest0000000000000101'
    record = _record({'line': 'SETTLE', 'idea_modifiers': ['SETTLE'], 'fence': 'two_pairs'})
    bare = _lplan(tid, notes=('Settles: a.', 'Consistent: b.'), rules=['Shortcut: x.'])
    good = _lplan(tid, notes=('Settles: a.', 'Consistent: b.'), invariants=[SETTLE_INV], rules=['Shortcut: move two steps.'])
    fake = _Fake(tmp_path, {f'{tid}-concrete-plan': bare, f'{tid}-replan-concrete-plan': good})
    value = Runtime._plan_with_colour(fake, _planner_kw(tid, record), record, 'v1', False)
    assert value == good and [c['call_id'] for c in fake.calls] == [f'{tid}-concrete-plan', f'{tid}-replan-concrete-plan']
    first, second = fake.calls
    assert 'Settled point for this idea (binding' in first['prompt'] and 'Relaxed fence for this idea: two training pairs' in first['prompt']
    assert first['request_context']['q2_fence'] == 'two_pairs' and first['request_context']['q2_line'] == 'SETTLE'
    assert second['prompt'].startswith(first['prompt'] + '\n\nLine check on an earlier plan for this idea: it had no invariant')
    assert second['prompt_template_version'].endswith('+line-replan-1')
    receipt = read(tmp_path / 'line_check' / f'{tid}.json')
    assert receipt['replanned'] and receipt['used'] == 'second' and receipt['final_issues'] == [] and receipt['first_issues']
    assert receipt['fence'] == 'two_pairs' and receipt['form']['text'] == SETTLE_INV
    assert not (tmp_path / 'rung_check' / f'{tid}.json').exists()
    again = _Fake(tmp_path, fake.answers)  # resume: the same calls and the same receipt
    assert Runtime._plan_with_colour(again, _planner_kw(tid, record), record, 'v1', False) == good
    assert [c['prompt'] for c in again.calls] == [c['prompt'] for c in fake.calls]
    strict = _record({'line': 'SETTLE', 'idea_modifiers': ['SETTLE']})
    for n, plan in enumerate((_lplan(invariants=[SETTLE_INV], rules=['Shortcut: x.']), _lplan(risk='Not carried: no reading.'))):
        t = f'dtest000000000000020{n}'
        f = _Fake(tmp_path, {f'{t}-concrete-plan': {**plan, 'candidate_id': t}})
        Runtime._plan_with_colour(f, _planner_kw(t, strict), strict, 'v1', False)
        assert len(f.calls) == 1 and read(tmp_path / 'line_check' / f'{t}.json')['replanned'] is False
    t = 'dtest0000000000000301'  # a rung shortfall and a line miss share the one re-plan
    ruled = _record({'line': 'SETTLE', 'idea_modifiers': ['SETTLE'], 'rung': 'R1'})
    f = _Fake(tmp_path, {f'{t}-concrete-plan': _lplan(t, rules=['Shortcut: x.']), f'{t}-replan-concrete-plan': _lplan(t)})
    Runtime._plan_with_colour(f, _planner_kw(t, ruled), ruled, 'v1', False)
    note = f.calls[1]['prompt'][len(f.calls[0]['prompt']):]
    assert note.startswith('\n\nScale check on an earlier plan') and '\n\nLine check on an earlier plan' in note
    assert f.calls[1]['prompt_template_version'].endswith('+rung-replan-1+line-replan-1') and len(f.calls) == 2
    t = 'dtest0000000000000302'  # a rung-only re-plan is unchanged
    plain = _record({'rung': 'R1'})
    f = _Fake(tmp_path, {f'{t}-concrete-plan': _lplan(t), f'{t}-replan-concrete-plan': _lplan(t)})
    Runtime._plan_with_colour(f, _planner_kw(t, plain), plain, 'v1', False)
    assert f.calls[1]['prompt'] == f.calls[0]['prompt'] + '\n\n' + rungs.feedback('R1', rungs.measures(_lplan(t)))
    assert f.calls[1]['prompt_template_version'].endswith('+rung-replan-1') and '+line-replan-1' not in f.calls[1]['prompt_template_version']
    assert not (tmp_path / 'line_check' / f'{t}.json').exists() and 'q2_line' not in f.calls[0]['request_context']
    b = contracts.Bindings()
    c = f.calls[0]
    with pytest.raises(Exception, match='re-plan note without a rung'):
        b.validate_request(c['stage'], c['prompt'] + '\n\nnote', {**{k: v for k, v in c['request_context'].items() if k != 'q2_rung'},
                                                                 'q2_replan': 'note'})


@pytest.mark.parametrize('line', ['MARKS', 'INGRID'])
def test_value_fields_resume_without_a_frozen_mismatch(tmp_path, line):
    """Review S1: the line-check receipt is frozen as JSON; its pair values have string keys, so a resume that plans the
    same task again compares equal instead of raising "frozen artifact changed"."""
    from arcgen.batch.runtime import Runtime
    tid = 'dtest00000000000006' + ('01' if line == 'MARKS' else '02')
    plan = {'MARKS': _lplan(tid, notes=('Marks set: 2.', 'Marks set: 3.', 'Marks set: 4.'), rules=['Shortcut: two.'],
                            invariants=['Marks: loose dots set how many copies are drawn; each is at most three cells, apart '
                                        'from other material, and their value differs between pairs.']),
            'INGRID': _lplan(tid, notes=('Instance shows: a turn.', 'Instance shows: a flip.', 'Instance shows: a shift.'),
                             test_note='Instance shows: a doubling.', rules=['Shortcut: borrow.'],
                             invariants=["Worked instance (in place): each input holds one worked instance of its own change; "
                                         "what it shows differs between pairs, and no training pair's instance shows the change "
                                         "the test's does."])}[line]
    record = _record({'line': line, 'idea_modifiers': [line]})
    fake = _Fake(tmp_path, {f'{tid}-concrete-plan': plan})
    Runtime._plan_with_colour(fake, _planner_kw(tid, record), record, 'v1', False)
    receipt = read(tmp_path / 'line_check' / f'{tid}.json')
    assert set(receipt['declared']['pair_values']) == {'1', '2', '3'} and receipt['final_issues'] == []
    again = _Fake(tmp_path, fake.answers)
    assert Runtime._plan_with_colour(again, _planner_kw(tid, record), record, 'v1', False) == plan  # no IntegrityError


# ---------------------------------------------------------------- realizer and repair stages


def test_realizer_components_and_stages(weak, monkeypatch):
    study = lambda line=None, edge=None, roles=None, fence=None: {'line': line, 'fence': fence, 'weak_overlays': {
        **({'EDGE': edge} if edge else {}), **({'ROLES': roles} if roles else {})}}
    names = lambda s: [n.split('/')[-1] for n in lines.realizer_components(s)]
    for line in config.LINE_NAMES:
        assert names(study(line)) == ['outputs.realizer.md', f'{line}.realizer.md'], line
    assert names(study(roles='drawn')) == [] and names(study()) == []
    assert names(study('SETTLE', edge='drawn')) == ['outputs.realizer.md', 'SETTLE.realizer.md', 'EDGE.realizer.md']
    assert lines.emits(study('COINCIDE')) == ('pairs', 'points', 'killer') and lines.emits(study('FORCED', edge='drawn')) == (
        'points', 'assembly', 'overlay_points')
    assert lines.REALIZER_STAGE in contracts.Bindings().stage_bindings and lines.REPAIR_STAGE in contracts.Bindings().stage_bindings
    monkeypatch.delitem(config.SCIENCE, 'weakness')
    assert lines.REALIZER_STAGE not in contracts.Bindings().stage_bindings  # nothing changes without the science key
    monkeypatch.setitem(config.SCIENCE, 'weakness', deepcopy(WEAK))
    from support.fixtures import episode
    from arcgen.lib.e2_live_streaming import PRIVATE_LABELS
    plan, realized = episode('dtest0000000000000401', train=3)
    b = contracts.Bindings()
    values, context = b.prepare_values('task_realizer', {'CONCRETE_PLAN': plan, 'CANDIDATE_ID': plan['candidate_id']})
    rows_ = b.realizer_templates({'line': 'SETTLE'})
    prompt = render_prompt('\n\n'.join(r[0] for r in rows_), values)
    assert not [f for f in PRIVATE_LABELS if f in prompt.casefold()]
    context.update(q2_line='SETTLE', q2_overlay=None, q2_template_hashes=[r[2] for r in rows_])
    b.validate_request(lines.REALIZER_STAGE, prompt, context)
    with pytest.raises(Exception):
        b.validate_request(lines.REALIZER_STAGE, prompt + ' ', context)
    with pytest.raises(Exception):
        b.validate_request(lines.REALIZER_STAGE, prompt, {**context, 'q2_line': 'DEPART'})
    point = {'id': 'p1', 'kind': 'settled point', 'note': 'The wall stops it.', 'test': 1, 'at': None, 'taught_by': [],
             'cells': [[0, 0, 5]]}
    outputs = {'training_outputs': [p['output'] for p in realized['task']['train']], 'points': [point], 'overlay_points': None,
               'objects': None, 'assembly': None, 'killer': None}
    value = lines.parse_realizer({**realized, 'weakness_outputs': outputs})
    b.validate_response(lines.REALIZER_STAGE, value, context)
    assert lines.parse_realizer(realized)['weakness_outputs'] is None  # a missing record is recorded as missing, never fatal
    lines.parse_realizer({**realized, 'weakness_outputs': None})
    # review B2: a malformed or oversized measurement record never fails the realization: bad parts are dropped and recorded
    many = [{**point, 'id': f'p{i}', 'note': ''} for i in range(250)]  # empty notes are fine; past the cap, truncated
    for bad, kept, problem in (({**outputs, 'points': [{**point, 'cells': [[0, 0]]}, point]}, 1, 'points: dropped 1 of 2'),
                               ({**outputs, 'killer': {'family': 'second_test', 'invariant': None, 'finished_sites': None}}, 1, 'killer: dropped'),
                               ({**outputs, 'points': many}, 200, 'points: kept the first 200 of 250'),
                               ({**outputs, 'points': [{**point, 'test': 9}]}, None, 'points: dropped 1 of 1'),
                               ({**outputs, 'training_outputs': [[[]]]}, 1, 'training_outputs: dropped')):
        parsed = lines.parse_realizer({**realized, 'weakness_outputs': bad})
        b.validate_response(lines.REALIZER_STAGE, parsed, context)  # the ordinary realization is untouched
        extra = parsed['weakness_outputs']
        assert (len(extra['points']) if extra['points'] else None) == kept and any(x.startswith(problem) for x in extra['_problems'])
    with pytest.raises(Exception):
        lines.parse_realizer({**{k: v for k, v in realized.items() if k != 'task'}, 'weakness_outputs': outputs})  # still strict
    wrong = deepcopy(value); wrong['authoring_metadata']['intended_rule'] = 'changed'
    with pytest.raises(Exception):
        b.validate_response(lines.REALIZER_STAGE, wrong, context)  # the ordinary realization checks still apply


# ---------------------------------------------------------------- checks on the grids


def _g(rows_):
    return [list(r) for r in rows_]


def _task(train, test):
    return {'train': [{'input': _g(i), 'output': _g(o)} for i, o in train], 'test': [{'input': _g(i), 'output': _g(o)} for i, o in test]}


def _pt(pid, cells, test=1, kind='point', at=None, taught=()):
    return {'id': pid, 'kind': kind, 'note': 'n', 'test': test, 'at': at, 'taught_by': list(taught), 'cells': cells}


def _emit(**kw):
    return {'weakness': {'emissions': [{'source': 'realizer', 'call_id': 'r', 'palette': 'original', 'problems': [], 'train': None,
                                        **{k: None for k in lines.LEDGERS}, **kw}]}}


def _row(*vals, width=7):
    return [list(vals) + [0] * (width - len(vals))]


SETTLE_TASK = _task([(_row(1), _row(0, 1)), (_row(2), _row(0, 2)), (_row(3), _row(0, 3, 0, 0, 0, 0, 3))],  # refuted on open ground
                    [(_row(4), _row(0, 4))])


def _settle_record(train_shortcuts, points, mapping=None, study=None, task=None):
    task = deepcopy(task or SETTLE_TASK)
    m = {v: k for k, v in (mapping or {}).items()}  # the realizer wrote the original palette
    orig = lambda g: [[m.get(c, c) for c in row] for row in g]
    return {'study': study or {'line': 'SETTLE', 'size_arm': 'small'}, 'complete_task': task,
            'palette_treatment': {'mapping': {str(k): v for k, v in (mapping or {}).items()}},
            **_emit(train=[[orig(p['input']), orig(o)] for p, o in zip(task['train'], train_shortcuts)],
                    points=[{**p, 'cells': [[r, c, m.get(x, x)] for r, c, x in p['cells']]} for p in points])}


OK_TRAIN = [_row(0, 1), _row(0, 2), _row(0, 3)]
OK_POINT = [_pt('p1', [[0, 0, 4], [0, 1, 0]], kind='settled point')]


def test_settle_checks_points_and_palette():
    ok = _settle_record(OK_TRAIN, OK_POINT)
    check = lines.episode_check(ok, ok['complete_task'])['line']
    assert check['status'] == 'pass' and [p['pair'] for p in check['pairs']['differing_pairs']] == [3]
    assert check['points']['points'] == [{'id': 'p1', 'kind': 'settled point', 'test': 1, 'differing_cells': 2, 'listed_cells': 2}]
    mapped = _settle_record(OK_TRAIN, OK_POINT, mapping={i: (i + 5) % 10 for i in range(10)})
    assert lines.episode_check(mapped, mapped['complete_task'])['line']['status'] == 'pass'  # realizer grids in the final palette
    assert lines.final_points(mapped)[0]['cells'] == [[0, 0, 4], [0, 1, 0]]
    every = _settle_record([_row(0, 1), _row(0, 2, 0, 0, 0, 0, 9), _row(0, 3)], OK_POINT)  # refuted in two of three pairs
    assert lines.episode_check(every, every['complete_task'])['line']['pairs']['status'] == 'fail'
    cluttered = _settle_record([_row(0, 1), _row(0, 2), _row(0, 3)], OK_POINT, task=_task(  # refuted next to the rule's change
        [(_row(1), _row(0, 1)), (_row(2), _row(0, 2)), (_row(3), _row(0, 3, 3))], [(_row(4), _row(0, 4))]))
    pairs = lines.episode_check(cluttered, cluttered['complete_task'])['line']['pairs']
    assert pairs['differing_pairs'][0]['open_ground'] is False and pairs['status'] == 'fail'  # DESIGN §3: only on open ground
    none = _settle_record([_row(0, 1), _row(0, 2), _row(0, 3, 0, 0, 0, 0, 3)], OK_POINT)  # refuted nowhere
    assert lines.episode_check(none, none['complete_task'])['line']['status'] == 'fail'
    idle = _settle_record(OK_TRAIN, [_pt('p1', [[0, 1, 4]])])  # the point's alternative equals the truth: it changes nothing
    assert lines.episode_check(idle, idle['complete_task'])['line']['points']['status'] == 'fail'
    three = _settle_record(OK_TRAIN, [_pt(f'p{i}', [[0, 0, i]], kind=f'point {i}') for i in (1, 2, 3)])  # at most two settled points
    assert lines.episode_check(three, three['complete_task'])['line']['points']['status'] == 'fail'
    places = _settle_record(OK_TRAIN, [_pt(f'p{i}', [[0, 0, i]], kind='blocked path') for i in (1, 2, 3, 5)])  # screen hole 1:
    check = lines.episode_check(places, places['complete_task'])['line']['points']  # one settled point, one entry per place
    assert check['status'] == 'pass' and check['kinds_per_test'] == {'1': 1} and check['per_test'] == {'1': 4}
    missing = _settle_record(OK_TRAIN[:2], OK_POINT)
    missing['weakness']['emissions'][0]['train'] = missing['weakness']['emissions'][0]['train'][:2]
    assert lines.episode_check(missing, missing['complete_task'])['line']['status'] == 'incomplete'
    assert lines.strawman_task(ok) == {'train': ok['complete_task']['train'][:2], 'test': [{'input': _row(4)}]}
    assert lines.strawman_task(every) is None


def test_the_relaxed_settle_forms_have_their_own_checks():
    pair = lambda i, o: {'input': _g(i), 'output': _g(o)}
    task = {'train': [pair([[1, 1, 1, 0, 0]], [[2, 2, 2, 0, 0]]), pair([[1, 1, 1, 1, 0]], [[2, 2, 2, 2, 0]])],
            'test': [pair([[1, 1, 0, 0, 0]], [[2, 2, 0, 0, 0]])]}
    minority = {'study': {'line': 'SETTLE', 'fence': 'in_pair_minority'}, 'complete_task': task,
                **_emit(train=[[task['train'][0]['input'], [[2, 2, 1, 0, 0]]], [task['train'][1]['input'], [[2, 2, 2, 1, 0]]]],
                        points=[_pt('p1', [[0, 1, 1]])])}
    check = lines.episode_check(minority, task)['line']
    assert check['pairs']['status'] == 'pass' and all(d['minority_of_changes'] for d in check['pairs']['differing_pairs'])
    assert lines.episode_check({**minority, 'study': {'line': 'SETTLE'}}, task)['line']['pairs']['status'] == 'fail'  # strict rule
    assert lines.strawman_task(minority) is None  # no consistent pair to show
    majority = deepcopy(minority)
    majority['weakness']['emissions'][0]['train'][0][1] = [[1, 1, 1, 0, 0]]  # the default misses most of the pair's changes
    assert lines.episode_check(majority, task)['line']['pairs']['status'] == 'fail'
    pairs2 = {'train': [pair(_row(1, 1), _row(2, 2)), pair(_row(1), _row(2, 0, 0, 0, 0, 0, 5))], 'test': [pair(_row(1), _row(2))]}
    two = {'study': {'line': 'SETTLE', 'fence': 'two_pairs'}, 'complete_task': pairs2,
           **_emit(train=[[pairs2['train'][0]['input'], _row(2, 2)], [pairs2['train'][1]['input'], _row(2)]],
                   points=[_pt('p1', [[0, 1, 1]])])}
    assert lines.episode_check(two, pairs2)['line']['status'] == 'pass'  # one consistent pair, one settling pair
    single = deepcopy(two); single['study']['fence'] = 'single_cell'  # the point needs two separate places in the test
    single['complete_task'] = pairs2 = {**pairs2, 'train': pairs2['train'] + [pair(_row(1, 1, 1), _row(2, 2, 2))]}
    single['weakness']['emissions'][0]['train'].append([_row(1, 1, 1), _row(2, 2, 2)])
    assert lines.episode_check(single, pairs2)['line']['points']['status'] == 'fail'  # one place only
    single['weakness']['emissions'][0]['points'] = [_pt('p1', [[0, 1, 1], [0, 5, 1]])]
    check = lines.episode_check(single, pairs2)['line']
    assert check['points']['places'] == {'1': 2} and check['status'] == 'pass'
    single = lines._pairs(task, {lines._key(task['train'][0]['input']): [[2, 2, 2, 0, 1]]}, {'min': 0, 'max': 2})
    assert single['differing_pairs'][0]['open_ground'] is False  # changes within two cells of the refuting cell


def test_repair_breakage_retag_and_the_line_tag(monkeypatch):
    monkeypatch.setattr(gate, 'current_blind', lambda tid, rec, calls: [])
    record = _settle_record(OK_TRAIN, OK_POINT)
    first = deepcopy(record['complete_task'])
    repaired = deepcopy(first); repaired['train'].append({'input': [[5, 0]], 'output': [[5, 5]]})
    record['complete_task'] = repaired
    record['difficulty_versions'] = {'a': {'episode': first}, 'b': {'episode': repaired}}
    record['weakness']['emissions'].append({'source': 'repair', 'call_id': 'p', 'palette': 'final', 'problems': [],
                                            'train': [[[[5, 0]], [[0, 5]]]], **{k: None for k in lines.LEDGERS}, 'points': OK_POINT})
    value = lines.fields('t', record, [])['weakness']
    assert value['first_version']['line']['status'] == 'pass' and value['final_version']['line']['status'] == 'fail'
    assert value['repair_breakage'] == {'line': True} and value['repairs_touched'] == [['train_4']] and value['line_tag'] == 'lost'
    assert value['line'] == 'SETTLE' and value['size_arm'] == 'small' and value['points'][0]['id'] == 'p1'
    record['weakness']['emissions'][-1]['train'] = [[[[5, 0]], [[5, 5]]]]  # the added pair is consistent: the property survives
    value = lines.fields('t', record, [])['weakness']
    assert value['final_version']['line']['status'] == 'pass' and value['repair_breakage'] == {'line': False}
    assert value['line_tag'] == 'kept' and value['stratum'] == 'strict'
    minority = deepcopy(record); minority['study']['fence'] = 'in_pair_minority'
    minority['blind_results'] = [{'response': {'predicted_test_outputs': [[[9, 9]]]}}]
    assert lines.fields('t', minority, [])['weakness']['line_tag'] == 'lost'  # needs Sol's first answer right
    assert lines.fields('t', minority, [])['weakness']['stratum'] == 'in_pair_minority'
    # COINCIDE: a repair that breaks the coincidence in a training pair retags it as a settling-pair task
    task = _task([([[1]], [[2]]), ([[3]], [[4]])], [([[5]], [[6]])])
    broken = deepcopy(task); broken['train'].append({'input': [[7]], 'output': [[8]]})
    coincide = {'study': {'line': 'COINCIDE'}, 'complete_task': broken,
                'difficulty_versions': {'a': {'episode': task}, 'b': {'episode': broken}},
                **_emit(train=[[[[1]], [[2]]], [[[3]], [[4]]], [[[7]], [[0]]]], points=[_pt('p1', [[0, 0, 1]])],
                        killer={'family': 'no_single_answer', 'invariant': None, 'finished_sites': None})}
    value = lines.fields('t', coincide, [])['weakness']
    assert value['retagged'] == 'SETTLE' and value['line_tag'] == 'lost'
    assert lines.fields('t', {'study': {'primary_arm': 'SEE'}}, []) == {}


def test_points_ledgers_and_capture():
    task = _task([([[1, 0, 0]], [[0, 0, 1]]), ([[2, 0, 0]], [[0, 0, 2]])], [([[3, 0, 0]], [[0, 0, 3]])])
    control = {'study': {'line': 'CONTROL'}, 'complete_task': task, **_emit(points=[_pt('p1', [[0, 1, 3], [0, 2, 0]])])}
    assert lines.episode_check(control, task)['line']['status'] == 'pass'
    edge = {'study': {'line': 'FORCED', 'weak_overlays': {'EDGE': 'drawn'}}, 'complete_task': task,
            **_emit(overlay_points=[_pt('e1', [[0, 2, 3]])])}
    assert lines.episode_check(edge, task)['overlay']['points']['status'] == 'fail'  # its alternative changes nothing
    decisions = [_pt(f'd{c}', [[0, c, 7]], kind=k, at=[0, c], taught=('train_1', 'train_2')) for c, k in ((2, 'stop'), (1, 'cross'))]
    precise = {'study': {'line': 'PRECISE', 'rung': None}, 'complete_task': task, **_emit(points=decisions)}
    check = lines.episode_check(precise, task)['line']['points']
    assert check['kinds'] == {'cross': ['train_1', 'train_2'], 'stop': ['train_1', 'train_2']} and check['status'] == 'fail'  # < 10
    trace = {'study': {'line': 'TRACE'}, 'complete_task': task,
             **_emit(points=[_pt(f'j{i}', [[0, i % 3, 5]], kind='junction', at=[0, i % 3]) for i in range(4)])}
    assert lines.episode_check(trace, task)['line']['status'] == 'pass'
    trace['weakness']['emissions'][0]['points'] = trace['weakness']['emissions'][0]['points'][:3]
    assert lines.episode_check(trace, task)['line']['status'] == 'fail'  # at least four junctions
    # point capture (DESIGN §6.1): eligible where the sample errs on the point's cells; captured at >= 80% of them
    truth, point = [[0, 0, 3, 0, 0]], _pt('p1', [[0, i, 3] for i in (0, 1, 3, 4)] + [[0, 2, 0]])
    assert lines.point_capture([point], truth, [[3, 3, 0, 3, 0]]) == [{'id': 'p1', 'eligible': True, 'captured': True, 'share': 0.8}]
    assert lines.point_capture([point], truth, [[3, 3, 0, 0, 0]])[0]['captured'] is False
    assert lines.point_capture([point], truth, truth) == [] and lines.point_capture([point], truth, [[3]]) == []
    near = lines.point_capture([_pt('a', [[0, 0, 3]]), _pt('b', [[0, 4, 3]])], truth, [[3, 0, 3, 0, 0]])
    assert [(r['id'], r['eligible'], r['captured']) for r in near] == [('a', True, True), ('b', False, False)]  # per point
    right = {'response': {'predicted_test_outputs': [[[0, 0, 3]]]}}
    trapped = {'response': {'predicted_test_outputs': [[[0, 3, 0]]], 'primary_inferred_rule': 'Move one step.'}}
    assert lines.on_points(control, trapped['response']) == ['line'] and lines.on_points(control, right['response']) == []
    assert lines.capture_route(control, [trapped, trapped]) == 'line' and lines.capture_route(control, [trapped, right]) is None
    assert lines.capture_route(control, [trapped]) is None
    control['concrete_plan'] = _lplan(rules=['Move one step right.'])
    instruction, challenges, evidence = lines.clarify(control, 'line', [trapped, trapped])
    assert 'Move one step right.' in instruction and evidence['blind_rules'] == ['Move one step.', 'Move one step.']
    from arcgen.astra_evidence_runtime_revision_20260911.structure import check as schema_check
    from arcgen.astra_usage_diversity_20260913 import adjudication
    for c in challenges:
        schema_check(c, adjudication.CHALLENGE)


def test_relation_objects_check():
    obj = lambda g, o, colour, size, shape, out, region='left': {'grid': g, 'object': o, 'colour': colour, 'size': size,
                                                                'shape': shape, 'region': region, 'relation': 'r', 'outcome': out}
    good = [obj('train_1', 'a', 3, 4, 'L', 'moves'), obj('train_1', 'b', 3, 4, 'L', 'stays'), obj('train_2', 'c', 5, 2, 'bar', 'moves'),
            obj('train_2', 'd', 5, 2, 'bar', 'stays', 'right'), obj('test_1', 'e', 3, 4, 'L', 'moves', 'right')]
    check = lines._objects(good)
    assert check['minimal_contrast'] and check['status'] == 'pass', check
    leak = [dict(r, outcome='moves' if r['region'] == 'left' else 'stays') for r in good]
    assert 'region' in lines._objects(leak)['own_feature_predictors'] and lines._objects(leak)['status'] == 'fail'
    assert lines._objects([r for r in good if r['object'] != 'b' and r['object'] != 'd'])['minimal_contrast'] is False


def test_forced_assembly_checker():
    domino = [[0, 0], [0, 1]]
    a = {'constraint': 'cover_region', 'transforms': 'turns', 'target': [[r, c] for r in range(2) for c in range(3)],
         'pieces': [{'id': 'A', 'cells': [[0, 0], [1, 0], [1, 1]], 'colour': 1}, {'id': 'B', 'cells': [[0, 0], [0, 1], [1, 1]], 'colour': 2}],
         'anchor': {'piece': 'A', 'cells': [[0, 0], [1, 0], [1, 1]]}}
    result = lines.solve_assembly(a, 10000)
    assert result['solutions'] == 1 and result['branch_points'] == 0
    turned = deepcopy(a); turned['transforms'] = 'none'
    turned['anchor']['cells'] = [[0, 0], [0, 1], [1, 0]]  # the anchor turned, though the other pieces may not turn
    turned['pieces'][1]['cells'] = [[0, 1], [1, 0], [1, 1]]
    assert lines.solve_assembly(turned, 10000)['solutions'] == 1
    free = {'constraint': 'cover_region', 'transforms': 'turns', 'target': [[r, c] for r in range(2) for c in range(4)],
            'pieces': [{'id': i, 'cells': domino, 'colour': k} for k, i in enumerate('ABCD')], 'anchor': {'piece': 'A', 'cells': [[0, 0], [0, 1]]}}
    assert lines.solve_assembly(free, 10000)['solutions'] == 2  # differently coloured dominoes tile two ways: not unique
    same = deepcopy(free)
    for p in same['pieces']:
        p['colour'] = 1
    assert lines.solve_assembly(same, 10000)['solutions'] == 1  # tilings that draw the same answer count once
    task = _task([([[0]], [[0]])], [([[0, 0, 0], [0, 0, 0]], [[1, 2, 2], [1, 1, 2]])])
    check = lines._assembly(task, a, {'branch_max': 2, 'node_limit': 10000})
    assert check['status'] == 'pass' and check['matches_answer'] is True
    painted = _task([([[0]], [[0]])], [([[0, 0, 0], [0, 0, 0]], [[1, 2, 3], [5, 1, 2]])])  # screen hole 2: pieces carry
    check = lines._assembly(painted, a, {'branch_max': 2, 'node_limit': 10000})  # several colours; the ledger gives one
    assert check['matches_answer'] is None and check['colours_compared'] is False and check['status'] == 'pass'
    wrong = _task([([[0]], [[0]])], [([[0, 0, 0], [0, 0, 0]], [[2, 1, 1], [2, 2, 1]])])  # piece colours, other places
    assert lines._assembly(wrong, a, {'branch_max': 2, 'node_limit': 10000})['status'] == 'fail'
    assert lines._assembly(task, {**a, 'constraint': 'connectors'}, {'branch_max': 2, 'node_limit': 10000})['status'] == 'unchecked'
    assert lines._assembly(task, {**free}, {'branch_max': 2, 'node_limit': 3})['status'] == 'unchecked'  # node limit


def test_the_assembly_solver_is_bounded_and_stored_with_its_emission(tmp_path, monkeypatch):
    """Review B1: an unforced 12x12 partition stops at the node limit within seconds (Algorithm X), the realizer's
    emission stores the result, and the checks read it rather than searching again."""
    import random, time
    rng = random.Random(12)
    cells = {(r, c) for r in range(12) for c in range(12)}
    owner = {cell: i for i, cell in enumerate(rng.sample(sorted(cells), 14))}
    while len(owner) < len(cells):
        r, c = rng.choice(sorted(owner))
        for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if nb in cells and nb not in owner:
                owner[nb] = owner[(r, c)]; break
    pieces = {}
    for cell, i in owner.items():
        pieces.setdefault(i, []).append(list(cell))
    a = {'constraint': 'cover_region', 'transforms': 'turns', 'target': [list(x) for x in sorted(cells)],
         'pieces': [{'id': str(i), 'cells': v, 'colour': i % 9 + 1} for i, v in sorted(pieces.items())],
         'anchor': {'piece': '0', 'cells': pieces[0]}}
    start = time.monotonic()
    result = lines.solve_assembly(a, lines.ASSEMBLY_NODES)
    assert result == lines.solve_assembly(a, lines.ASSEMBLY_NODES)  # deterministic: a resumed emission compares equal
    assert time.monotonic() - start < 60 and lines.LINES['FORCED']['grids']['assembly']['node_limit'] == lines.ASSEMBLY_NODES
    assert result['solutions'] in (None, 2)  # past the limit (unchecked) or not unique: never a long search
    task = _task([([[0]], [[0]])], [([[0, 0, 0], [0, 0, 0]], [[1, 2, 2], [1, 1, 2]])])
    small = {'constraint': 'cover_region', 'transforms': 'turns', 'target': [[r, c] for r in range(2) for c in range(3)],
             'pieces': [{'id': 'A', 'cells': [[0, 0], [1, 0], [1, 1]], 'colour': 1}, {'id': 'B', 'cells': [[0, 0], [0, 1], [1, 1]], 'colour': 2}],
             'anchor': {'piece': 'A', 'cells': [[0, 0], [1, 0], [1, 1]]}}
    runtime = SimpleNamespace(run_dir=tmp_path, state=None)
    base = {'status': 'realized', 'task': {'train': task['train'], 'test': [{'input': task['test'][0]['input']}]},
            'authoring_metadata': {'expected_test_outputs': [task['test'][0]['output']]}}
    entry = lines.realizer_emission(runtime, 't', 'c1', {'line': 'FORCED'}, base,
                                    {'training_outputs': None, 'points': [_pt('f1', [[0, 1, 1]])], 'assembly': small, 'overlay_points': None,
                                     'objects': None, 'killer': None})
    assert entry['assembly_solution']['solutions'] == 1 and entry['episode']['test'][0]['output'] == task['test'][0]['output']
    record = {'study': {'line': 'FORCED'}, 'complete_task': task, 'weakness': {'emissions': [entry]}}
    monkeypatch.setattr(lines, 'solve_assembly', lambda a, limit: pytest.fail('searched again'))
    monkeypatch.setattr(gate, 'current_blind', lambda tid, rec, calls: [])
    value = lines.fields('t', record, [])['weakness']
    assert value['first_version_basis'] == 'realizer' and value['final_version']['line']['assembly']['status'] == 'pass'
    wrong = deepcopy(task); wrong['test'][0]['output'] = [[2, 2, 2], [1, 1, 2]]  # the unique solution is not the answer
    assert lines.episode_check(record, wrong)['line']['assembly']['status'] == 'fail'


def test_traces_count_junctions_only_and_tags_need_an_episode(monkeypatch):
    task = _task([([[1, 0, 0]], [[0, 0, 1]])], [([[3, 0, 0, 0]], [[0, 0, 0, 3]])])
    points = [_pt(f'j{c}', [[0, c, 5]], kind='junction') for c in range(3)] + [_pt('cue', [[0, 3, 0]], kind='local cue')]
    trace = {'study': {'line': 'TRACE'}, 'complete_task': task, **_emit(points=points)}
    check = lines.episode_check(trace, task)['line']['points']
    assert check['junctions'] == 3 and check['status'] == 'fail'  # review N1: the local cue is not a junction
    monkeypatch.setattr(gate, 'current_blind', lambda tid, rec, calls: [])
    plan = _lplan(invariants=['Trace: a route.'], rules=['Shortcut: nearest.'])
    unrealized = {'study': {'line': 'TRACE'}, 'weakness': {'plan_check': lines.plan_check('TRACE', None, plan)}}
    assert lines.fields('t', unrealized, [])['weakness']['line_tag'] == 'no_episode'  # review N2


def test_points_are_counted_per_test():
    """Review T7: with two tests, PRECISE's decision floor holds in each test, not over the task."""
    row = [0] * 12
    task = _task([([row], [row]), ([row], [row])], [([row], [row]), ([row], [row])])
    decisions = lambda test, n: [_pt(f'd{test}-{c}', [[0, c, 7]], kind=('stop', 'cross')[c % 2], test=test, at=[0, c],
                                     taught=('train_1', 'train_2')) for c in range(n)]
    record = lambda points: {'study': {'line': 'PRECISE'}, 'complete_task': task, **_emit(points=points)}
    assert lines.episode_check(record(decisions(1, 10) + decisions(2, 10)), task)['line']['points']['status'] == 'pass'
    check = lines.episode_check(record(decisions(1, 12) + decisions(2, 8)), task)['line']['points']
    assert check['per_test'] == {'1': 12, '2': 8} and check['status'] == 'fail'


def test_commit_holds_judge_decoys_and_pending_consensus():
    from arcgen.batch import commit
    rows = [{'task_id': 'a', 'blind_performance': {'weakness': {'judge_decoy': True, 'line_tag': 'decoy'}}},
            {'task_id': 'b', 'blind_performance': {'weakness': {'line': 'JUDGE', 'line_tag': 'pending_consensus'}}},
            {'task_id': 'c', 'blind_performance': {'weakness': {'line': 'JUDGE', 'line_tag': 'lost'}}},
            {'task_id': 'd', 'blind_performance': {'label': None}}]
    held = commit.Commit.weakness_holds(SimpleNamespace(release_path=None, track=lambda p: None), {'tasks': rows})
    assert set(held) == {'a', 'b'} and 'decoy' in held['a'] and 'consensus' in held['b']
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / 'released.json'
        path.write_text(json.dumps(['a', 'b']))
        held = commit.Commit.weakness_holds(SimpleNamespace(release_path=path, track=lambda p: None), {'tasks': rows})
    assert set(held) == {'a'}  # a decoy is never released
    assert commit.Commit.weakness_holds(SimpleNamespace(release_path=None, track=lambda p: None), {'tasks': rows[3:]}) == {}


def test_spec_rows_are_checked_against_the_registry_and_profile(weak):
    from arcgen.batch.allocations.make_allocation import build
    spec = {'revision': 'r', 'seed': 1, 'windows': {'w': {'account': 'main', 'wave': 'A', 'context': 'c01', 'commit': 'explore'}}}
    ok = [{'skill': 'SETTLE', 'fence': 'relaxed'}, {'skill': 'JUDGE', 'decoy': True}, {'skill': 'TRACE', 'fence': 'strict'},
          {'control': True, 'size': 'R4'}]
    build({**spec, 'rows': [['w', 'GEO', None, 'L', 'sol', 'high', x] for x in ok]}, {'sol': 'm'}, {}, lambda a, i: True)
    for bad in ({'skill': 'SETTLE', 'fence': 'binary'}, {'skill': 'REFRAME', 'fence': 'relaxed'}, {'skill': 'JUDGE', 'fence': 'decoy'},
                {'skill': 'SETTLE', 'decoy': True}, {'fence': 'two_pairs'}):
        with pytest.raises(ValueError):  # review N7: refused before the idea calls are paid
            build({**spec, 'rows': [['w', 'GEO', None, 'L', 'sol', 'high', bad]]}, {'sol': 'm'}, {}, lambda a, i: True)
    weak['lines'].pop('CONTROL')
    with pytest.raises(ValueError, match='control line'):
        build({**spec, 'rows': [['w', 'GEO', None, 'L', 'sol', 'high', {'control': True}]]}, {'sol': 'm'}, {}, lambda a, i: True)


def _plan_roles(*roles, background='ground'):
    plan = _lplan(roles=[{'role': r, 'color': c, 'functional_meaning': 'x'} for r, c in roles])
    plan['public_contract']['episode_overview']['palette_contract'] = {'background_role': background}
    return plan


def test_process_conservation_and_coincide_killers():
    plan = _plan_roles(('ground', 0), ('sand', 4))
    task = _task([([[4, 0], [0, 0]], [[0, 0], [4, 0]]), ([[4, 4], [0, 0]], [[0, 0], [4, 4]])], [([[0, 4], [0, 0]], [[0, 0], [0, 4]])])
    plan['public_contract']['invariants'].append('Conserved: the number of @sand cells is the same in input and output.')
    record = {'study': {'line': 'PROCESS'}, 'complete_task': task, 'concrete_plan': plan, **_emit(points=[_pt('p1', [[0, 1, 4], [1, 1, 0]])])}
    check = lines.episode_check(record, task)['line']
    assert check['conservation']['status'] == 'pass' and check['status'] == 'pass'
    assert lines.conservation_break(record, [[0, 0], [4, 4]]) is True and lines.conservation_break(record, [[0, 4], [0, 0]]) is False
    plan['public_contract']['invariants'][-1] = 'Conserved: none; the pairs show frames over time.'
    assert lines.episode_check(record, task)['line']['conservation']['status'] == 'unchecked'
    leaky = deepcopy(task); leaky['train'][0]['output'] = [[0, 0], [4, 4]]
    plan['public_contract']['invariants'][-1] = 'Conserved: the number of non-background cells is the same in input and output.'
    assert lines.episode_check({**record, 'complete_task': leaky}, leaky)['line']['conservation']['status'] == 'fail'
    # COINCIDE: K1 (two completions disagree, both wrong), K3 (the rival breaks a regularity), K4 (a finished site)
    killer = lambda family, **kw: {'family': family, 'invariant': None, 'finished_sites': None, **kw}
    base = {'study': {'line': 'COINCIDE'}, 'complete_task': task, 'concrete_plan': plan}
    pairs = [[p['input'], p['output']] for p in task['train']]
    k1 = {**base, **_emit(train=pairs, points=[_pt('b1', [[1, 1, 0], [1, 0, 4]]), _pt('b2', [[1, 1, 0], [0, 0, 4]])],
                          killer=killer('no_single_answer'))}
    assert lines.episode_check(k1, task)['line']['status'] == 'pass'
    k1['weakness']['emissions'][0]['points'] = [_pt('b1', [[1, 1, 0], [1, 0, 4]])] * 2
    assert lines.episode_check(k1, task)['line']['killer']['completions_disagree'] is False
    k3 = {**base, **_emit(train=pairs, points=[_pt('b1', [[1, 1, 0]])],
                          killer=killer('broken_regularity', invariant='the number of @sand cells is the same in input and output'))}
    check = lines.episode_check(k3, task)['line']['killer']
    assert check['regularity_broken_by_rival'] and check['status'] == 'pass'
    # screen hole 2: the finished instance is drawn in the test input and kept by both readings; the rival differs elsewhere
    k4 = {**base, **_emit(train=pairs, points=[_pt('b1', [[0, 1, 9], [1, 1, 0]])],
                          killer=killer('finished_instance', finished_sites=[[[0, 0]], [[0, 0]], [[1, 0]]]))}
    check = lines.episode_check(k4, task)['line']['killer']
    assert check['site_kept_by_answer'] and check['rival_differs'] and check['contradiction'] == 'not machine-checked'
    redone = deepcopy(k4); redone['weakness']['emissions'][0]['killer']['finished_sites'][2] = [[0, 1]]  # the answer changes it
    assert lines.episode_check(redone, task)['line']['killer']['status'] == 'fail'
    assert lines.episode_check(k4, task)['line']['killer']['status'] == 'pass'
    not_coincident = {**base, **_emit(train=[[pairs[0][0], [[9, 9], [9, 9]]], pairs[1]], points=[_pt('b1', [[1, 1, 0]])],
                                      killer=killer('no_single_answer'))}
    assert lines.episode_check(not_coincident, task)['line']['pairs']['status'] == 'fail'  # the rival must fit every pair
    assert lines.regularity('the output is mirror-symmetric left to right', plan)([[1]], [[1, 2, 1]])
    assert lines.regularity('an obscure invariant', plan) is None


def test_growth_and_the_r4_size_check(weak):
    task = _task([([[0] * 4] * 4, [[0] * 4] * 4)], [([[0] * 5] * 5, [[0] * 5] * 5)])
    depart = {'study': {'line': 'DEPART'}, 'complete_task': task, **_emit(points=[_pt('p1', [[0, 0, 1]])])}
    growth = lines.episode_check(depart, task)['line']['growth']
    assert growth == {'ratio': 1.562, 'area_max': 1.5, 'parts_ratio': 0, 'parts_max': 2, 'status': 'fail'}
    dots = lambda n, size: [[(1 if r == 0 and c % 2 == 0 and c < 2 * n else 0) for c in range(size)] for r in range(size)]
    grown = _task([(dots(2, 6), dots(2, 6))], [(dots(2, 6), dots(2, 6)), (dots(3, 6), dots(3, 6))])
    assert lines.episode_check({**depart, 'complete_task': grown}, grown)['line']['growth']['status'] == 'pass'  # 1.5x the parts
    many = _task([(dots(1, 6), dots(1, 6))], [(dots(3, 6), dots(3, 6))])  # three parts against one
    assert lines.episode_check({**depart, 'complete_task': many}, many)['line']['growth'] == {
        'ratio': 1.0, 'area_max': 1.5, 'parts_ratio': 3.0, 'parts_max': 2, 'status': 'fail'}
    assert lines._parts([[0, 1, 0], [0, 0, 1], [2, 0, 0]]) == 2  # diagonal neighbours join; the ground is the commonest colour
    stripes = [[(r // 3) % 2 for _ in range(30)] for r in range(30)]  # few runs per row
    noisy = [[(r * 7 + c * 3) % 5 for c in range(30)] for r in range(30)]  # a run per cell
    blank = [[9] * 30 for _ in range(30)]
    small = [[0] * 14 for _ in range(15)]  # 210 cells: at least min(450, 200)
    r4 = lambda out, plan=None, train=small: {'study': {'line': 'PRECISE', 'rung': 'R4'}, 'concrete_plan': plan or _lplan(),
                                              'complete_task': _task([(train, train)], [(blank, out)])}
    ok = lines.size_check(r4(stripes), r4(stripes)['complete_task'])
    assert ok['status'] == 'pass' and ok['row_runs'] <= lines.R4_ROW_RUNS and ok['training_cells_needed'] == 200
    assert lines.size_check(r4(noisy), r4(noisy)['complete_task'])['status'] == 'fail'  # over the compressibility floor
    regular = _lplan(invariants=['Regular field: a cycle of colours by distance from the marker.'])
    assert lines.size_check(r4(noisy, regular), r4(noisy)['complete_task'])['status'] == 'pass'  # a regular field is exempt
    tiny = [[0] * 10 for _ in range(10)]
    assert lines.size_check(r4(stripes, train=tiny), r4(stripes, train=tiny)['complete_task'])['status'] == 'fail'
    assert lines.size_check({'study': {'rung': 'R3'}}, task) is None
    half = [r[:15] for r in stripes]  # 450 cells: under R4's floor of 800
    assert lines.size_check(r4(half), r4(half)['complete_task'])['status'] == 'fail'
    assert lines.size_check(r4(half), r4(half)['complete_task'])['output_cells_min'] == 800
    assert lines.R4_ROW_RUNS == 351  # the 90th percentile of eval test outputs of 600 cells or more (v2 §12)


JUDGE_INV = ('Judgement: in every pair a count reading and a reach reading both fit; the test separates them, and a salient '
             'role settles it.')


def test_judge_plan_fields():
    """DESIGN §6.9: the named prior comes from the approved list (dropped, supporting-only and excluded priors are
    refused at the plan check, with the one re-plan), the rival is declared, and the plan states there is no strict
    killer (a test a strict killer settles is COINCIDE's)."""
    judge = lambda why, rules=('Rival: count the marks.',): _lplan(invariants=[JUDGE_INV], rules=rules, why=why)
    ok = judge('Prior: salient role. The panel varies with the outputs. No strict killer: both readings give one answer.')
    assert lines.plan_issues('JUDGE', None, ok) == [] and lines.declared('JUDGE', ok)['private_value'] == 'salient_role'
    assert lines.form('JUDGE', lines.declared('JUDGE', ok))['label'] == 'salient_role'
    for prior in ('physical plausibility', 'one salient regularity'):
        assert lines.plan_issues('JUDGE', None, judge(f'Prior: {prior}. Why. No strict killer: none.')) == []
    for prior, why in (('near misses', 'dropped'), ('relational over absolute', 'dropped'), ('objects stay whole', 'supporting only'),
                       ('a rule without exceptions', 'supporting only'), ('which layer lies on top', 'excluded'),
                       ('the new element matters', 'excluded'), ('colour order', 'excluded'), ('taste', 'one of')):
        issues = lines.plan_issues('JUDGE', None, judge(f'Prior: {prior}. No strict killer: none.'))
        assert len(issues) == 1 and why in issues[0] and 'physical plausibility, one salient regularity, salient role' in issues[0], issues
    assert lines.plan_issues('JUDGE', None, judge('Prior: salient role. The panel is used.')) == [
        'test_1\'s why_one_output_is_best_supported did not state "No strict killer" (a test settled by a strict killer is '
        'another line\'s, not a judgement)']
    assert lines.plan_issues('JUDGE', None, judge('Prior: salient role. No strict killer: x.', rules=('Shortcut: x.',))) == [
        'no strongest_simple_shortcuts entry began "Rival:"']
    assert lines.plan_issues('JUDGE', None, _lplan(risk='Not carried: no shared preference fits.')) == []
    decoy = lines.plan_issues('JUDGE', None, judge('Prior: near misses. No strict killer: none.'), 'decoy')
    assert len(decoy) == 1  # a decoy plan records the same fields


def test_judge_rival_checks_records_tags_and_routes(monkeypatch):
    monkeypatch.setattr(gate, 'current_blind', lambda tid, rec, calls: [])
    task = _task([([[1, 0]], [[0, 1]]), ([[2, 0]], [[0, 2]])], [([[3, 0]], [[0, 3]])])
    plan = _lplan(invariants=[JUDGE_INV], rules=['Rival: move each thing to the far side.'],
                  why='Prior: physical plausibility. No strict killer: both readings give one answer.')
    pairs = [[p['input'], p['output']] for p in task['train']]
    rival = _pt('r1', [[0, 0, 3], [0, 1, 0]], kind='rival')

    def record(train, points, **study):
        r = {'study': {'line': 'JUDGE', **study}, 'complete_task': task, 'concrete_plan': plan,
             'realization': {'authoring_metadata': {'intended_rule': 'r'}}, **_emit(train=train, points=points)}
        r['weakness']['plan_check'] = lines.plan_check('JUDGE', None, plan, fence=study.get('fence'))
        return r
    good = record(pairs, [rival])
    value = lines.fields('t', good, [])['weakness']
    assert value['final_version']['line']['status'] == 'pass' and value['line_tag'] == 'pending_consensus'  # consensus decides
    assert value['prior'] == 'physical' and value['alternatives'] == [{'id': 'r1', 'test': 1, 'grid': [[3, 0]]}]
    assert value['judge_decoy'] is False and 'retagged' not in value and value['stratum'] == 'strict'
    refuted = record([pairs[0], [pairs[1][0], [[2, 0]]]], [rival])  # a pair refutes the rival: a settling-pair task
    value = lines.fields('t', refuted, [])['weakness']
    assert value['retagged'] == 'SETTLE' and value['line_tag'] == 'lost'
    two = record(pairs, [rival, _pt('r2', [[0, 1, 5]], kind='rival')])  # one judgement point per task
    assert lines.fields('t', two, [])['weakness']['line_tag'] == 'lost'
    decoy = record(pairs, [rival], fence='decoy', judge_decoy=True)
    value = lines.fields('t', decoy, [])['weakness']
    assert value['line_tag'] == 'decoy' and value['stratum'] == 'decoy' and value['judge_decoy'] is True
    assert lines.review(good, []) == (['JUDGE'], None)  # the policy needs the rival verified on every pair
    assert lines.review(refuted, []) == ([], 'check_not_passed')
    on_rival = {'response': {'predicted_test_outputs': [[[3, 0]]]}}
    assert lines.on_points(good, on_rival['response']) == ['line']  # the first answer on the rival gets the second call
    assert lines.capture_route(good, [on_rival, on_rival]) is None  # but two such answers go to adjudication, not a repair
    notes = lines.repair_notes(good)
    assert notes[0] == lines.body('lines/JUDGE.repair_note.md') and 'whose rule begins "Rival:"' in notes[1]
    assert lines.policy_text(['JUDGE']).startswith('Review policy for this episode. Its plan leaves two readings')


def test_judge_decoy_rows_and_draws(weak, monkeypatch):
    from arcgen.batch.allocations.make_allocation import build
    weak['lines']['JUDGE']['relaxed'] = {}
    decoys = admission.configured_routes(None, rows(10, idea_modifiers=['JUDGE'], judge_decoy=True))
    assert {(r['study']['fence'], r['study']['judge_decoy']) for r in decoys} == {('decoy', True)}
    plain = admission.configured_routes(None, rows(300, idea_modifiers=['JUDGE']))
    assert not any(r['study'].get('fence') == 'decoy' for r in plain)
    for bad in (dict(idea_modifiers=['SETTLE'], judge_decoy=True), dict(idea_modifiers=['JUDGE'], fence_call='decoy')):
        with pytest.raises(IntegrityError):
            admission.configured_routes(None, rows(1, **bad))
    spec = {'revision': 'r', 'seed': 1, 'windows': {'w': {'account': 'main', 'wave': 'A', 'context': 'c01', 'commit': 'explore'}},
            'rows': [['w', 'JUDG', None, 'L', 'sol', 'high', {'skill': 'JUDGE', 'size': 'small'}],
                     ['w', 'JUDG', None, 'L', 'sol', 'high', {'skill': 'JUDGE', 'size': 'small', 'decoy': True}]]}
    calls = build(spec, {'sol': 'gpt-6.1-sol'}, {}, lambda a, i: True)['calls']
    assert [c.get('decoy') for c in calls] == [None, True] and calls[0]['private_id'] != calls[1]['private_id']
    for bad in ({'decoy': True}, {'skill': 'JUDGE', 'decoy': 'yes'}, {'skill': 'JUDGE', 'decoy': True, 'fence': 'decoy'}):
        with pytest.raises(ValueError):
            build({**spec, 'rows': [['w', 'JUDG', None, 'L', 'sol', 'high', bad]]}, {'sol': 'm'}, {}, lambda a, i: True)
    profile = _profile(idea_modifiers={'SETTLE': 0.3, 'MARKS': 0.3, 'IGNORE': 0.1, 'JUDGE': 0.1},
                       modifier_slots={'line': ['SETTLE', 'MARKS', 'JUDGE'], 'relevance': ['IGNORE']})
    profile['science']['weakness']['lines']['JUDGE'] = {'rungs': {'R1': 0.3, 'R2': 0.3, 'R3': 0.1}}
    config.validate(profile)
    profile['science']['weakness']['lines']['JUDGE']['relaxed'] = {'decoy': 0.1}  # decoys only by spec row
    with pytest.raises(config.ConfigError):
        config.validate(profile)
    profile['science']['weakness']['lines']['JUDGE'] = {'rungs': {'R4': 0.1}}  # JUDGE keeps the judgement point visible: no R4
    with pytest.raises(config.ConfigError):
        config.validate(profile)


def test_the_section_12_strata():
    """v2 §12 as relaxed strata (DESIGN §6.9: in the pooled relaxed set): FORCED's three-times pieces, PROCESS's
    interaction shown once in a two-pair task, TRACE's chained use and chosen route, PRECISE's periodic phase, RELATION's
    contrast across grids."""
    assert {n: sorted(s['relax']) for n, s in lines.LINES.items() if s.get('relax')} == {
        'DEPART': ['two_situations'], 'SETTLE': ['in_pair_minority', 'ordinal', 'single_cell', 'two_pairs'],
        'MARKS': ['binary', 'inside'], 'INGRID': ['progression', 'variant_pairs'], 'PRECISE': ['periodic_phase'],
        'TRACE': ['chained_once', 'route_choice'], 'PROCESS': ['contact', 'interaction_once'],
        'RELATION': ['across_grids', 'distortion'], 'FORCED': ['many_pieces'], 'JUDGE': ['decoy']}  # 2026-10-22: contact, distortion
    assert 'about twice as many pieces' in (lines.LINE_DIR / 'FORCED.planner.md').read_text()
    assert 'about three times as many pieces' in (lines.LINE_DIR / 'FORCED.relax.many_pieces.md').read_text()
    task = _task([([[1, 0, 0]], [[0, 0, 1]]), ([[2, 0, 0]], [[0, 0, 2]])], [([[3, 0, 0]], [[0, 0, 3]])])
    phase = [_pt(f'd{c}', [[0, c, 7]], kind='phase', at=[0, c], taught=('train_1', 'train_2')) for c in (1, 2)]
    for fence, status in ((None, 'fail'), ('periodic_phase', 'pass')):
        rec = {'study': {'line': 'PRECISE', 'fence': fence}, 'complete_task': task, **_emit(points=phase)}
        assert lines.episode_check(rec, task)['line']['points']['status'] == status, fence
    obj = lambda g, o, colour, out, region: {'grid': g, 'object': o, 'colour': colour, 'size': 4, 'shape': 'L', 'region': region,
                                             'relation': 'r', 'outcome': out}
    across = [obj('train_1', 'a', 3, 'moves', 'left'), obj('train_1', 'b', 5, 'stays', 'left'), obj('train_2', 'c', 3, 'stays', 'right'),
              obj('train_2', 'd', 5, 'moves', 'right'), obj('test_1', 'e', 3, 'moves', 'right')]
    assert lines._objects(across)['status'] == 'fail' and lines._objects(across, {'contrast': 'across'})['status'] == 'pass'
    assert lines.spec('RELATION', 'across_grids')['grids']['objects'] == {'contrast': 'across'}
    process = _lplan(invariants=['Process: things fall until they rest.',
                                 'Conserved: the number of movers is the same in input and output, counting what leaves the grid.'],
                     rules=['Shortcut: each place alone.'])
    assert lines.plan_issues('PROCESS', None, process) == []
    assert lines.plan_issues('PROCESS', None, process, 'interaction_once') == ['it had 3 training pairs, not 2']
    assert lines.regularity(process['public_contract']['invariants'][-1], process) is None  # what the grids cannot show
    rec = {'study': {'line': 'PROCESS'}, 'complete_task': task, 'concrete_plan': process}
    assert lines._conservation(rec, task)['status'] == 'unchecked'
    assert lines.regularity('the number of @sand cells is the same in input and output, counting what leaves the grid',
                            _plan_roles(('ground', 0), ('sand', 4))) is None


# ---------------------------------------------------------------- review policies and repair notes


def test_review_policies_withholding_and_repair_notes(monkeypatch):
    settle = {'study': {'line': 'SETTLE'}, 'complete_task': _task([([[1]], [[2]])], [([[1]], [[2]])]),
              'concrete_plan': _lplan(invariants=[SETTLE_INV], rules=['Shortcut: x.']),
              'realization': {'authoring_metadata': {'intended_rule': 'r'}}, 'technical_results': [{'response': {}}]}
    right = [{'response': {'predicted_test_outputs': [[[2]]]}}]
    wrong = [{'response': {'predicted_test_outputs': [[[3]]]}}]
    assert lines.review(settle, wrong) == (['SETTLE'], None)
    assert lines.review({**settle, 'concrete_plan': _lplan()}, wrong) == ([], 'no_invariant')
    assert lines.review({**settle, 'complete_task': None}, wrong) == ([], 'no_episode')
    assert lines.review({'study': {'line': 'CONTROL'}}, wrong) == (None, None)  # the control has no policy
    dep = 'No training pair shows a key on the left; the test does, and every law its answer uses there acts in the pairs.'
    depart = {**settle, 'study': {'line': 'DEPART'}, 'concrete_plan': _lplan(invariants=[dep])}
    assert lines.review(depart, wrong) == ([], 'no_right_answer') and lines.review(depart, right) == (['DEPART'], None)
    edge_inv = "Edge: things cut by the grid's border move like whole ones; they occur in the training pairs and the test."
    both = {**settle, 'study': {'line': 'SETTLE', 'weak_overlays': {'EDGE': 'drawn'}}, 'concrete_plan': _lplan(invariants=[SETTLE_INV, edge_inv])}
    assert lines.review(both, wrong) == (['SETTLE', 'EDGE'], None)
    cinv = 'Coincidence: in every pair size and count agree; the test separates them, and a tie settles it.'
    task = _task([([[1]], [[2]])], [([[1, 1]], [[2, 2]])])
    coincide = {**settle, 'study': {'line': 'COINCIDE'}, 'complete_task': task, 'concrete_plan': _lplan(invariants=[cinv]),
                **_emit(train=[[[[1]], [[2]]]], points=[_pt('b1', [[0, 0, 5]]), _pt('b2', [[0, 1, 5]])],
                        killer={'family': 'no_single_answer', 'invariant': None, 'finished_sites': None})}
    assert lines.review(coincide, wrong) == (['COINCIDE'], None)  # the killer is verified: the policy applies
    coincide['weakness']['emissions'][0]['points'] = [_pt('b1', [[0, 0, 5]])]
    assert lines.review(coincide, wrong) == ([], 'check_not_passed')  # unverified: the rival stays comparable
    text = lines.policy_text(['SETTLE', 'EDGE'])
    assert text.startswith('Review policy for this episode. Its plan settles one point') and 'Review policy for things cut' in text
    assert lines.policy_text(['DEPART']) == gate.DEPART_POLICY  # DEPART v4 keeps the departure package's policy

    class _Rt:
        def __init__(self, record):
            self.state = {'semantic_records': {'t': record}, 'call_ledger': {}}
    monkeypatch.setattr(gate, 'technical_bound', lambda tid, rec, calls: True)
    monkeypatch.setattr(gate.adjudication, 'concerns', lambda blind: [])
    for record, blind, policy, withheld in ((settle, wrong, lines.policy_text(['SETTLE']), False), (depart, right, gate.DEPART_POLICY, True),
                                            (depart, wrong, '', False), ({**settle, 'study': {}}, wrong, '', False)):
        record = deepcopy(record)
        record['concrete_plan']['private_execution']['test_outputs'][0].update(test_only_semantics_audit='a', why_one_output_is_best_supported='a')
        record['concrete_plan']['private_execution']['cross_pair_concrete_audit']['relevant_and_incidental_variation'] = 'x'
        monkeypatch.setattr(gate, 'current_blind', lambda tid, rec, calls, b=blind: b)
        values = gate._adjudication_values(_Rt(record), 't')
        assert values['REVIEW_POLICY'] == policy
        out = values['AUTHOR_SPECIFICATION']['concrete_plan']['private_execution']['test_outputs'][0]
        assert (out['test_only_semantics_audit'] == gate.WITHHELD) is withheld
    notes = lines.repair_notes(settle)
    assert notes[0] == lines.body('lines/SETTLE.repair_note.md')
    assert '- changed_pairs: every training pair' in notes[1] and '- points:' in notes[1] and '- objects:' not in notes[1]
    assert 'whose rule begins "Shortcut:"' in notes[1] and '{' not in notes[1]
    assert 'Never add a pair that also refutes the simpler reading' in notes[0]
    assert len(lines.repair_notes({**settle, 'concrete_plan': _lplan()})) == 1  # no invariant: only the outputs to re-emit
    assert lines.repair_notes({'study': {'line': 'FORCED', 'weak_overlays': {'ROLES': 'drawn'}}, 'concrete_plan': _lplan()}) == [
        lines.body('lines/ROLES.repair_note.md'), lines.emit_note({'line': 'FORCED'})]
    control = lines.emit_note({'line': 'CONTROL', 'weak_overlays': {'EDGE': 'drawn'}})
    assert 'begins "Edge shortcut:"' in control and '- changed_pairs' not in control and '- overlay_points:' in control
    assert lines.repair_notes({'study': {}}) == []
    assert lines.repair_notes({**depart})[0] == gate.DEPART_REPAIR_NOTE  # DEPART v4: repairs never show the new case


# ---------------------------------------------------------------- end to end


SHORTCUT_RULE = 'Shortcut: move the signal two columns in every pair.'
E2E_INV = ('Settled point: moving the signal two columns gives every training output except in the pair(s) showing a '
           'blocked path, where it differs visibly; nothing else sets those pairs apart.')


def _bump(grid, r=0, c=0):
    out = deepcopy(grid); out[r][c] = (out[r][c] + 1) % 10
    return out


def _patch(truth, r, c):
    return [[r, c, (truth[r][c] + 1) % 10]]


class _Proxy:
    """A monkeypatch that also wraps the fixture's mock transport, so the realizer and repairer return weakness_outputs."""
    def __init__(self, mp, wrap):
        self._mp, self._wrap = mp, wrap

    def setattr(self, target, name, value=None, *args, **kwargs):
        from arcgen.lib.factory_runtime import FactoryModelInvoker
        if target is FactoryModelInvoker and name == '_mock_response':
            value = self._wrap(value)
        return self._mp.setattr(target, name, value, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._mp, name)


def _e2e(tmp_path, monkeypatch, weak, line, plan_edit, realizer_outputs, repair_outputs, case='repair', expected=8,
         usable=True, crash=None, study=None):
    """One line idea through the mock pipeline (pilot_fixture): the plan edited for the line, the realizer's and the
    repairer's responses given the line's weakness_outputs. Returns (record, stage list, prompts by stage pattern)."""
    from support import pilot as pilot_fixture
    from arcgen.batch import repair, runtime
    weak['overlays'] = {}
    weak['lines'][line] = {'rungs': {}}
    original = pilot_fixture.episode

    def episode(tid, **kw):
        plan, realized = original(tid, **{**kw, 'train': 3})
        plan_edit(plan)
        if realized.get('authoring_metadata'):
            realized['authoring_metadata']['plan_fidelity']['preserved_frozen_decisions'] = list(plan['public_contract']['invariants'])
        return plan, realized
    monkeypatch.setattr(pilot_fixture, 'episode', episode)
    monkeypatch.setattr(pilot_fixture, 's', SimpleNamespace(PLAN_STAGE=pilot_fixture.s.PLAN_STAGE, REALIZER_STAGE=lines.REALIZER_STAGE))
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
    empty = {'training_outputs': None, 'points': None, 'overlay_points': None, 'objects': None, 'assembly': None, 'killer': None}

    def wrap(answer):
        def respond(invoker, stage, cid):
            text = answer(invoker, stage, cid)
            if stage not in (lines.REALIZER_STAGE, lines.REPAIR_STAGE):
                return text
            value = json.loads(text)
            tid = json.loads((invoker.calls_dir / stage / cid / 'request.json').read_text())['parent_ids'][0]
            if stage == lines.REALIZER_STAGE:
                value['weakness_outputs'] = {**empty, **realizer_outputs(value)}
            else:
                record = runtimes[-1].state['semantic_records'][tid]
                outputs = {**empty, 'changed_pairs': [], **repair_outputs(value, record)}
                outputs.pop('training_outputs')
                value['weakness_outputs'] = outputs
            return json.dumps(value)
        return respond
    from support import pilot as pf
    first, calls, *_ = pf.run_case(tmp_path, _Proxy(monkeypatch, wrap), case, expected, usable, arm='SEE', planner='sol',
                                   study={'idea_modifiers': [line], **(study or {})}, **({'crash_point': crash} if crash else {}))
    prompts = lambda pattern: [p.read_text() for p in sorted(tmp_path.glob(f'accounts/**/model_calls/{pattern}/*/prompt.md'))]
    return next(iter(first['runtime'].state['semantic_records'].values())), [st for st, _ in calls], prompts


def _shortcut(rule):
    return [dict(rule=rule, fits_every_training_pair=False, comparably_simple=True, predicts_a_different_test_output=True,
                 evidence_that_defeats_it='The second pair.')]


def _settle_plan(plan):
    pub = plan['public_contract']
    pub['invariants'] = list(pub['invariants']) + [E2E_INV]
    for i, pair in enumerate(pub['training_pairs']):
        pair['what_it_helps_communicate'] = ('Settles: ' if i == 1 else 'Consistent: ') + pair['what_it_helps_communicate']
    plan['private_execution']['cross_pair_concrete_audit']['strongest_simple_shortcuts'] = _shortcut(SHORTCUT_RULE)


def _point(pid, truths, at, kind='point', test=1):
    return {'id': pid, 'kind': kind, 'note': 'n', 'test': test, 'at': None, 'taught_by': [], 'cells': _patch(truths[0], *at)}


@pytest.mark.parametrize('variant', ['capture', 'adjudicated', 'breakage', 'capture_twice'])
def test_a_settle_task_runs_end_to_end_with_points(tmp_path, monkeypatch, weak, variant, crash=None):
    """A SETTLE idea through the mock pipeline: the planner gets the line last, the realizer records the default on every
    pair and one point per settled point, a first answer on the point gets a second call and two such answers go to a
    clarifying repair (capture, breakage) instead of the adjudicator; otherwise the adjudicator sees the SETTLE policy
    (adjudicated). The repair carries the line's notes and re-emits the changed pair's output and the points; the checks
    run on both versions. capture_twice: after one clarifying repair, a second capture goes to adjudication (review S2)."""
    at = (0, 0) if variant != 'adjudicated' else (1, 1)  # the fixture's missed answers bump cell (0, 0) of the test
    twice = variant == 'capture_twice'

    def realized(value):
        task = value['task']
        return {'training_outputs': [_bump(p['output'], 3, 0) if i == 1 else p['output'] for i, p in enumerate(task['train'])],
                'points': [_point('p1', value['authoring_metadata']['expected_test_outputs'], at, 'settled point')]}

    def repaired(value, record):
        k = record.get('repair_count', 0) if twice else 0  # the pair the fixture's repair changes
        new = value['repaired_realization']['task']['train'][k]
        out = _bump(new['output'], 3, 0) if k == 1 else _bump(new['output'], 2, 2) if variant == 'breakage' else new['output']
        return {'changed_pairs': [{'pair_number': k + 1, 'shortcut_output': out}],
                'points': [_point('p1', record['realization']['authoring_metadata']['expected_test_outputs'], at, 'settled point')]}
    record, stages, prompts = _e2e(tmp_path, monkeypatch, weak, 'SETTLE', _settle_plan, realized, repaired,
                                   case='repair_twice' if twice else 'repair', expected=13 if twice else 8, crash=crash)
    assert stages.count(lines.REALIZER_STAGE) == 1 and stages.count(lines.REPAIR_STAGE) == (2 if twice else 1)
    plan, = prompts('astra_concrete_plan_v3')
    assert plan.rstrip().endswith(lines.body('lines/SETTLE.planner.md').rstrip())
    realizer, = prompts(lines.REALIZER_STAGE)
    assert 'Measurement records for this episode' in realizer and 'Apply it, exactly as its rule states it, to every training input' in realizer
    repair_prompt = prompts(lines.REPAIR_STAGE)[0]
    assert lines.body('lines/SETTLE.repair_note.md') in repair_prompt and '- changed_pairs: every training pair' in repair_prompt
    adjudications = prompts('*adjudicat*')
    versions = list(record['difficulty_versions'].values())
    weakness = record['blind_performance']['weakness']
    if twice:  # one clarifying repair, then the second capture is adjudicated and repaired as usual
        assert versions[0]['shortcut_capture'] == 'line' and 'shortcut_capture' not in versions[1]
        assert len(adjudications) == 1 and versions[1]['adjudication']['decision'] == 'repairable'
        assert weakness['capture_routes'] == ['line'] and record['pipeline_usable_before_union_audit'] is True
        return
    assert [e['source'] for e in record['weakness']['emissions']] == ['realizer', 'repair']
    if variant == 'adjudicated':
        assert len(adjudications) == 1 and lines.policy_text(['SETTLE']) in adjudications[0]
        assert versions[0]['line_review'] == ['SETTLE'] and 'shortcut_capture' not in versions[0]
        assert 'Both blind solvers answered' not in repair_prompt
    else:
        assert not adjudications and versions[0]['shortcut_capture'] == 'line' and versions[0]['route'] == 'second_then_adjudicate'
        assert 'Both blind solvers answered the test with the simpler reading' in repair_prompt and SHORTCUT_RULE in repair_prompt
        assert record['difficulty_repair']['route'] == 'shortcut_capture'
    assert weakness['line'] == 'SETTLE' and weakness['size_arm'] == 'small' and weakness['plan_check']['final_issues'] == []
    assert weakness['first_version_basis'] == 'realizer' and weakness['changed_before_review'] is False
    assert weakness['first_version']['line']['status'] == 'pass' and weakness['repairs_touched'] == [['train_1']]
    assert weakness['final_version']['line']['status'] == ('fail' if variant == 'breakage' else 'pass')
    assert weakness['line_tag'] == ('lost' if variant == 'breakage' else 'kept')
    assert weakness['repair_breakage'] == {'line': variant == 'breakage'} and weakness['repair_missing'] == {'line': False}
    assert weakness['first_blind']['on_points'] == ([] if variant == 'adjudicated' else ['line'])
    first_points = weakness['first_blind']['points']  # review S11: per point, with the exact-equality flag
    assert [(r['id'], r['eligible'], r['captured'], r['equals_alternative']) for r in first_points] == [
        ('p1', variant != 'adjudicated', variant != 'adjudicated', variant != 'adjudicated')]
    assert weakness['capture_routes'] == ([] if variant == 'adjudicated' else ['line'])
    assert [p['id'] for p in weakness['points']] == ['p1'] and weakness['points'][0]['cells']
    assert read(next(tmp_path.glob('accounts/**/line_check/*.json')))['used'] == 'first'


@pytest.mark.parametrize('crash', ['repair_response_saved', 'expansion_saved', 'repair_stored', 'technical_recheck_saved'])
def test_the_clarifying_repair_resumes_from_every_crash_point(tmp_path, monkeypatch, weak, crash):
    """The capture route, interrupted at each saved step of its repair, resumes to the same record with no new call."""
    test_a_settle_task_runs_end_to_end_with_points(tmp_path, monkeypatch, weak, 'capture', crash=crash)


JUDGE_E2E = ('Judgement: in every pair moving the signal two columns and moving it to a fixed column both fit; the test '
             'separates them, and a salient role settles it.')


def _judge_plan(plan):
    pub, priv = plan['public_contract'], plan['private_execution']
    pub['invariants'] = list(pub['invariants']) + [JUDGE_E2E]
    priv['cross_pair_concrete_audit']['strongest_simple_shortcuts'] = _shortcut('Rival: move the signal to a fixed column.')
    priv['test_outputs'][0]['why_one_output_is_best_supported'] = (
        'Prior: salient role. The anchor varies with the outputs and the rival leaves it idle. No strict killer: both '
        'readings give one answer in the test.')


def _rival_outputs(at):
    def realized(value):
        return {'training_outputs': [p['output'] for p in value['task']['train']],
                'points': [_point('r1', value['authoring_metadata']['expected_test_outputs'], at, 'rival')]}

    def repaired(value, record):
        new = value['repaired_realization']['task']['train'][0]
        return {'changed_pairs': [{'pair_number': 1, 'shortcut_output': new['output']}],
                'points': [_point('r1', record['realization']['authoring_metadata']['expected_test_outputs'], at, 'rival')]}
    return realized, repaired


@pytest.mark.parametrize('variant', ['adjudicated', 'capture', 'decoy'])
def test_judge_runs_end_to_end(tmp_path, monkeypatch, weak, variant, crash=None):
    """JUDGE through the gate (review T5): the rival is checked on every pair, so the JUDGE policy reaches the
    adjudicator and the repair carries the JUDGE note; two answers on the rival go to adjudication, never a clarifying
    repair (no clarifying note). A decoy is adjudicated as an ordinary episode and never repaired (review S3); its rival
    grids are recorded from the first version."""
    at = (0, 0) if variant == 'capture' else (1, 1)
    realized, repaired = _rival_outputs(at)
    decoy = variant == 'decoy'
    record, stages, prompts = _e2e(tmp_path, monkeypatch, weak, 'JUDGE', _judge_plan, realized, repaired,
                                   expected=5 if decoy else 9 if variant == 'capture' else 8, usable=not decoy, crash=crash,
                                   study={'judge_decoy': True} if decoy else None)
    plan, = prompts('astra_concrete_plan_v3')
    adjudication, = prompts('*adjudicat*')
    weakness = record['blind_performance']['weakness']
    versions = list(record['difficulty_versions'].values())
    assert weakness['prior'] == 'salient_role' and weakness['first_alternatives'][0]['id'] == 'r1'
    assert weakness['first_version']['line']['status'] == 'pass'
    if decoy:
        assert plan.rstrip().endswith(lines.body('lines/JUDGE.relax.decoy.md').rstrip())
        assert lines.policy_text(['JUDGE']) not in adjudication and stages.count(lines.REPAIR_STAGE) == 0
        assert versions[0]['outcome']['reason'] == 'decoy_not_repaired' and weakness['line_tag'] == 'decoy'
        assert weakness['stratum'] == 'decoy' and weakness['judge_decoy'] is True
        return
    assert plan.rstrip().endswith(lines.body('lines/JUDGE.planner.md').rstrip())
    assert lines.policy_text(['JUDGE']) in adjudication and versions[0]['line_review'] == ['JUDGE']
    repair_prompt, = prompts(lines.REPAIR_STAGE)
    assert lines.body('lines/JUDGE.repair_note.md') in repair_prompt and 'whose rule begins "Rival:"' in repair_prompt
    assert 'shortcut_capture' not in versions[0] and weakness['capture_routes'] == []
    if variant == 'capture':
        assert versions[0]['route'] == 'second_then_adjudicate' and weakness['first_blind']['on_points'] == ['line']
    assert weakness['final_version']['line']['status'] == 'pass' and weakness['line_tag'] == 'pending_consensus'
    assert weakness['alternatives'][0]['grid'] != record['complete_task']['test'][0]['output']


@pytest.mark.parametrize('crash', ['response_saved', 'repair_response_saved', 'expansion_saved', 'repair_stored',
                                   'technical_recheck_saved'])
def test_judge_resumes_from_every_crash_point(tmp_path, monkeypatch, weak, crash):
    test_judge_runs_end_to_end(tmp_path, monkeypatch, weak, 'adjudicated', crash=crash)


def test_coincide_runs_end_to_end(tmp_path, monkeypatch, weak):
    """COINCIDE through the gate: the killer is verified on the grids, so its policy reaches the adjudicator; the repair
    re-emits the rival's pair, the completions and the killer."""
    inv = 'Coincidence: in every pair the signal\'s column and its step agree; the test separates them, and a tie settles it.'

    def plan_edit(plan):
        pub, priv = plan['public_contract'], plan['private_execution']
        pub['invariants'] = list(pub['invariants']) + [inv]
        priv['cross_pair_concrete_audit']['strongest_simple_shortcuts'] = _shortcut('Shortcut: move the signal to a fixed column.')
        priv['test_outputs'][0]['why_one_output_is_best_supported'] = 'Killer: no single answer, since the rival ties.'
    killer = {'family': 'no_single_answer', 'invariant': None, 'finished_sites': None}

    def completions(truths):
        return [_point('b1', truths, (1, 1), 'completion'), _point('b2', truths, (1, 2), 'completion')]

    def realized(value):
        return {'training_outputs': [p['output'] for p in value['task']['train']], 'killer': killer,
                'points': completions(value['authoring_metadata']['expected_test_outputs'])}

    def repaired(value, record):
        new = value['repaired_realization']['task']['train'][0]
        return {'changed_pairs': [{'pair_number': 1, 'shortcut_output': new['output']}], 'killer': killer,
                'points': completions(record['realization']['authoring_metadata']['expected_test_outputs'])}
    record, stages, prompts = _e2e(tmp_path, monkeypatch, weak, 'COINCIDE', plan_edit, realized, repaired)
    adjudication, = prompts('*adjudicat*')
    weakness = record['blind_performance']['weakness']
    assert lines.policy_text(['COINCIDE']) in adjudication
    assert weakness['final_version']['line']['killer']['status'] == 'pass' and weakness['line_tag'] == 'kept'
    assert lines.body('lines/COINCIDE.repair_note.md') in prompts(lines.REPAIR_STAGE)[0]


@pytest.mark.parametrize('crash', [None, 'repair_response_saved', 'expansion_saved'])
def test_forced_and_relation_ledgers_go_through_repair(tmp_path, monkeypatch, weak, crash):
    """FORCED's pieces record through realizer and repair (review T5, B1): the solver runs once per emission, outside the
    lock, and the bank fields read its stored result; RELATION-style object rows ride along in the same record shape."""
    calls = []
    solve = lines.solve_assembly

    def counted(a, limit):
        calls.append(limit); return solve(a, limit)
    monkeypatch.setattr(lines, 'solve_assembly', counted)

    def plan_edit(plan):
        pub, priv = plan['public_contract'], plan['private_execution']
        pub['invariants'] = list(pub['invariants']) + ['Forced: the anchor fixes the signal; every step follows from what is already placed.']
        priv['cross_pair_concrete_audit']['strongest_simple_shortcuts'] = _shortcut('Shortcut: place each piece where it lies.')

    def assembly(truths):  # the test answer's two marks, the signal as the anchor: the corner mark is forced
        truth = truths[0]
        cells = {x: [[r, c] for r, row in enumerate(truth) for c, v in enumerate(row) if v == x] for x in {v for row in truth for v in row} - {0}}
        (one,), (eight,) = cells[1], cells[8]
        return {'constraint': 'cover_region', 'transforms': 'none', 'target': [one, eight],
                'pieces': [{'id': 'A', 'cells': [[0, 0]], 'colour': 1}, {'id': 'B', 'cells': [[0, 0]], 'colour': 8}],
                'anchor': {'piece': 'A', 'cells': [one]}}

    def realized(value):
        truths = value['authoring_metadata']['expected_test_outputs']
        return {'points': [_point('f1', truths, (1, 1), 'placement')], 'assembly': assembly(truths)}

    def repaired(value, record):
        truths = record['realization']['authoring_metadata']['expected_test_outputs']
        return {'points': [_point('f1', truths, (1, 1), 'placement')], 'assembly': assembly(truths)}
    record, stages, prompts = _e2e(tmp_path, monkeypatch, weak, 'FORCED', plan_edit, realized, repaired, crash=crash)
    emissions = record['weakness']['emissions']
    assert [e['source'] for e in emissions] == ['realizer', 'repair']
    assert all(e['assembly_solution']['solutions'] == 1 and e['assembly_solution']['branch_points'] == 0 for e in emissions)
    weakness = record['blind_performance']['weakness']
    assert weakness['final_version']['line']['assembly']['status'] == 'pass'
    assert weakness['final_version']['line']['assembly']['matches_answer'] is True
    monkeypatch.setattr(lines, 'solve_assembly', lambda a, limit: pytest.fail('searched again under the lock'))
    assert lines.fields('t', record, [])['weakness']['final_version']['line']['assembly']['status'] == 'pass'


@pytest.mark.parametrize('line', ['RELATION', 'TRACE'])
def test_relation_and_trace_ledgers_go_through_repair(tmp_path, monkeypatch, weak, line):
    """RELATION's object table and TRACE's junction points are re-emitted whole by the repair, and the final checks read
    the repair's records (review T5)."""
    inv = {'RELATION': 'Relation: what a thing touches decides whether it moves.',
           'TRACE': 'Trace: the signal is followed through each crossing, going straight at a crossing.'}[line]

    def plan_edit(plan):
        pub, priv = plan['public_contract'], plan['private_execution']
        pub['invariants'] = list(pub['invariants']) + [inv]
        priv['cross_pair_concrete_audit']['strongest_simple_shortcuts'] = _shortcut('Shortcut: by the nearest mark.')

    def outputs(truths, source):
        if line == 'TRACE':
            junctions = [{**_point(f'j{k}', truths, (1, k), 'junction'), 'at': [1, k]} for k in range(4)]
            return {'points': junctions + [_point('cue', truths, (0, 0), 'local cue')]}
        obj = lambda g, o, colour, out, region: {'grid': g, 'object': o, 'colour': colour, 'size': 1, 'shape': 'dot',
                                                 'region': region, 'relation': source, 'outcome': out}
        rows = [obj('train_1', 'a', 1, 'moves', 'left'), obj('train_1', 'b', 1, 'stays', 'left'), obj('train_2', 'c', 8, 'stays', 'right'),
                obj('train_2', 'd', 8, 'moves', 'right'), obj('test_1', 'e', 1, 'moves', 'right')]
        return {'points': [_point('o1', truths, (1, 1), 'own feature')], 'objects': rows}

    def realized(value):
        return outputs(value['authoring_metadata']['expected_test_outputs'], 'realizer')

    def repaired(value, record):
        return outputs(record['realization']['authoring_metadata']['expected_test_outputs'], 'repair')
    record, stages, prompts = _e2e(tmp_path, monkeypatch, weak, line, plan_edit, realized, repaired)
    emissions = record['weakness']['emissions']
    assert [e['source'] for e in emissions] == ['realizer', 'repair']
    repair_prompt, = prompts(lines.REPAIR_STAGE)
    final = record['blind_performance']['weakness']['final_version']['line']
    if line == 'RELATION':
        assert '- objects:' in repair_prompt and emissions[1]['objects'][0]['relation'] == 'repair'
        assert final['objects']['status'] == 'pass' and final['objects']['minimal_contrast'] is True
    else:
        assert '- objects:' not in repair_prompt and final['points']['junctions'] == 4 and final['status'] == 'pass'


def test_a_repair_cannot_drop_the_protected_invariant(weak):
    """The line's invariant is frozen through repair: the weakness repair stage validates the repair exactly as the
    ordinary one does, so a repair that drops or rewrites the invariant is refused; one that keeps it passes, with its
    re-emitted outputs split off."""
    from support.fixtures import episode
    from arcgen.batch import repair, repair_v1
    from arcgen.lib.schemas import SchemaError
    tid = 'dtest0000000000000501'
    plan, realized = episode(tid, train=3)
    plan['public_contract']['invariants'] = list(plan['public_contract']['invariants']) + [E2E_INV]
    realized['authoring_metadata']['plan_fidelity']['preserved_frozen_decisions'] = list(plan['public_contract']['invariants'])
    b = contracts.Bindings()
    values = {'CANDIDATE_ID': tid, 'REPAIR_TYPE': 'ambiguity', 'REPAIR_INSTRUCTION': 'Replace train_1.',
              'AUTHOR_SPECIFICATION': {'intended_rule': realized['authoring_metadata']['intended_rule'], 'concrete_plan': plan,
                                       'realization': realized},
              'EVALUATOR_EVIDENCE': {'route': 'shortcut_capture'}, 'ALLOW_MECHANICAL_TEST_INPUT_RESTORATION': False}
    _, context = b.prepare_values('targeted_repair', values)

    def response(invariants=repair.REUSE_INVARIANTS):
        p, r = deepcopy(plan), deepcopy(realized)
        for name in ('input', 'output'):
            grid = r['task']['train'][0][name]; grid[0], grid[1] = grid[1], grid[0]
        p['public_contract']['training_pairs'][0]['exact_input_construction'] = 'Place the marked cell in row 1.'
        p['public_contract']['test_inputs'] = repair.REUSE_TESTS
        p['public_contract']['invariants'] = invariants
        p['private_execution']['test_outputs'] = repair.REUSE_ANSWERS
        r['task']['test'] = repair.REUSE_TESTS
        r['authoring_metadata']['expected_test_outputs'] = repair.REUSE_ANSWERS
        if invariants != repair.REUSE_INVARIANTS:
            r['authoring_metadata']['plan_fidelity']['preserved_frozen_decisions'] = invariants
        return {'contract_version': repair.VERSION, 'training_support_updates': [], 'candidate_id': tid, 'repair_type': 'ambiguity',
                'status': 'repaired', 'unrepairable_reason': '', 'plan_update': p, 'repaired_realization': r,
                'changes': ['Replaced one pair.'], 'why_rule_and_test_purpose_are_preserved': 'Unchanged rule and tests.',
                'weakness_outputs': {'changed_pairs': [{'pair_number': 1, 'shortcut_output': r['task']['train'][0]['output']}],
                                     'points': None, 'overlay_points': None, 'objects': None, 'assembly': None, 'killer': None}}
    good = lines.parse_repair(response())
    b.validate_response(lines.REPAIR_STAGE, good, context)
    assert repair_v1.expand(lines.split(good)[0], values)['repaired_plan']['public_contract']['invariants'][-1] == E2E_INV
    for changed in (plan['public_contract']['invariants'][:-1],
                    plan['public_contract']['invariants'][:-1] + [E2E_INV.replace('two columns', 'three columns')]):
        with pytest.raises(SchemaError, match='frozen public commitment'):
            b.validate_response(lines.REPAIR_STAGE, lines.parse_repair(response(changed)), context)
    # review B2: a repair's measurement record is lenient too: missing or malformed, the repair itself stands
    assert lines.parse_repair({k: v for k, v in response().items() if k != 'weakness_outputs'})['weakness_outputs'] is None
    bad = response()
    bad['weakness_outputs']['changed_pairs'] = [{'pair_number': 99, 'shortcut_output': None}, bad['weakness_outputs']['changed_pairs'][0]]
    parsed = lines.parse_repair(bad)
    b.validate_response(lines.REPAIR_STAGE, parsed, context)
    assert len(parsed['weakness_outputs']['changed_pairs']) == 1 and parsed['weakness_outputs']['_problems'][0].startswith(
        'changed_pairs: dropped 1 of 2')
