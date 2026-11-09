"""Composite skill lines: the four newer
skill lines (RECURSE, AUTOMATON, RANK, GRAPH) are built as the blind-spot skill lines are (registry, idea text, binding
planner line, skill-scoped novelty) with their ONE exclusions; a weakness run that pins skill lines judges novelty with the
composite selector; a skill line's matched control ({"slot_control": SKILL}) and fixed sizes for arm ideas in a weakness run;
and the parallel gate deletes each job's basetemp as soon as the job ends."""
from copy import deepcopy
import json, os, pathlib, re, subprocess, sys
import pytest
from arcgen.batch import admission, config, contracts, idea_v1, slots, supply
from arcgen.batch.allocation import assign_modifiers
from arcgen.batch.common import IntegrityError
from test_weakness_lines import WEAK, FLOORS, rows, _profile
from test_targeted import _call

NEW = ('RECURSE', 'AUTOMATON', 'RANK', 'GRAPH')
VERSION = {'RECURSE': 1, 'AUTOMATON': 1, 'RANK': 2, 'GRAPH': 1}  # RANK v2: all-rounder batch c2 (2026-10-26, review5)
V1 = idea_v1.PROMPTS.parent
SKETCHES = [{'id': 's01', 'title': 'T', 'idea': 'I.', 'recipe': 'R.'}]
NEAR = [[{'reference_id': 'committed:x', 'similarity': 0.5, 'text': 'A task.'}]]


def _texts(name):
    idea = (idea_v1.PROMPTS / 'modifiers' / f'{name}.md').read_text()
    planner = (contracts.SLOT_DIR / f'{name}.planner.md').read_text()
    return idea, planner


def _body(text):
    return text.split('-->', 1)[1].strip()


@pytest.fixture
def weak(monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'weakness', deepcopy(WEAK))
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    return config.SCIENCE['weakness']


# ---------------------------------------------------------------- registry and texts


def test_the_new_skills_are_skill_lines_with_both_texts():
    assert config.SKILL_MODIFIERS[-4:] == NEW
    for name in NEW:
        assert name in config.MODIFIER_NAMES and name in idea_v1.MODIFIERS and name in config.NOVELTY_LINES
        assert name in contracts.SLOT_BRIDGES and name not in config.LINE_MODIFIERS
        idea, planner = _texts(name)
        assert idea.startswith(f'<!-- prompt-version: composite-c1-idea-modifier-{name}-{VERSION[name]} -->')
        assert planner.startswith(f'<!-- prompt-version: composite-c1-slot-planner-{name}-{VERSION[name]} -->')
        assert _body(idea).startswith('Additional requirement for this call: in each idea,')
        assert _body(planner).startswith("This idea's insight is")
        for text in (idea, planner):
            assert not re.search(r'\b[0-9a-f]{8}\b', text)  # no task IDs in generation prompts
            assert 'ARC-AGI' not in text and 'evaluation' not in text.lower()


@pytest.mark.parametrize('name, phrases', [
    ('RECURSE', ('copies of itself', 'different shape', 'two or three levels', 'every case the test needs',
                 'never from a longer list of positions')),
    ('AUTOMATON', ('applied step after step', 'two to four plain cases', 'never by a hidden count', 'at least two pairs',
                   'step by step in private_execution', 'never through more steps', 'well-known automaton or game')),
    ('RANK', ('an order read by looking', 'no two equal unless the pairs show how a tie goes', 'different numbers of things',
              'every rank the test needs', 'never only report the order', 'each plainly apart and ranked on its own')),
    ('GRAPH', ('a drawn network', 'at a glance what is joined to what', 'about a dozen nodes', 'Vary the layout',
               'every case the test meets', 'never through one longer chain')),
])
def test_each_planner_line_asks_for_plain_instances_variety_and_fair_teaching(name, phrases):
    _, planner = _texts(name)
    for phrase in phrases:
        assert phrase in planner, (name, phrase)


def test_rank_v2_settles_things_groups_values_ties_consequence_and_cues():
    """All-rounder batch c2 (2026-10-26): review5 found RANK's ties and groupings settled ambiguously (which segments count
    as one thing, which thing ranks where) and a test lacking the marks it needed. Both texts now ask for each of the five
    points; v1's weaker wording is gone."""
    idea, planner = (_body(t) for t in _texts('RANK'))
    for text, phrases in ((planner, (
            'one unmistakable object', 'never touching or bridged by another shape', 'read two pieces as one thing or one as two',
            'every group is plainly apart', 'the grouping acts in every training pair', 'nothing in the test joins two groups',
            'Read each value one way only', 'the same in every pair', 'no second measure',
            'no two equal unless the pairs show how a tie goes',
            'what each rank gets is plain from them', 'never only report the order',
            "Every input, the test's too, carries every mark the rule reads")),
                          (idea, (
            'one unmistakable object', 'any group ranked on its own is plainly apart', 'each value reads one way only',
            'never equal unless the examples show how a tie goes', "give every input, the test's too, every mark the rule reads",
            'never a mere report of the order', 'Colour numbers carry no meaning'))):
        for phrase in phrases:
            assert phrase in text, phrase
    assert 'no ties unless' not in planner and "each thing's rank can be read only one way" not in planner
    # still a single-pair-compatible line: nothing asks for two pairs, so ONE's binding line stays the only pair rule
    assert 'two pairs' not in planner and 'two pairs' not in idea and 'RANK' not in slots.ONE_EXCLUDED_MODIFIERS


def test_the_planner_composes_each_skill_line_with_the_slots():
    b = contracts.Bindings()
    for name in NEW:
        texts = '\n'.join(t[0] for t in b.planning_templates('GEO', (), slots=(name,)))
        assert _body(_texts(name)[1]) in texts
        assert contracts.planner_slots({'idea_modifiers': [name, 'LIGHT']}) == tuple(sorted((name, 'LIGHT')))
    both = '\n'.join(t[0] for t in b.planning_templates('SEE', (), slots=('ONE', 'RANK')))  # RANK is allowed with ONE
    assert 'One demonstration for this idea (binding' in both and _body(_texts('RANK')[1]) in both


def test_new_skill_calls_use_the_skill_steps_and_carry_their_idea_text(monkeypatch):
    plain, _ = _call()
    plain = {**plain, 'commit': 'explore'}

    def steps(call):
        req = idea_v1.sketch_request(call)
        return (req['prompt'], idea_v1.explore_request(call, req, SKETCHES, NEAR)['prompt'],
                idea_v1.commit_request(call, req, SKETCHES, NEAR)['prompt'])
    monkeypatch.delitem(config.SCIENCE, 'skill_novelty', raising=False)
    _, explore, commit = steps({**plain, 'idea_modifiers': ['GRAPH']})
    assert 'would be far from the ones shown' in explore and 'sharing the skill is not redundancy' not in commit
    monkeypatch.setitem(config.SCIENCE, 'skill_novelty', True)
    for name in NEW:
        call = {**plain, 'idea_modifiers': [name]}
        sketch, explore, commit = steps(call)
        assert idea_v1.skill_call(call) and idea_v1.line_call(call) is None
        assert sketch.index('## Direction for this call') < sketch.index(_body(_texts(name)[0]))
        assert "still centre on this call's skill" in explore and 'sharing the skill is not redundancy' in commit


# ---------------------------------------------------------------- selectors


def _passage(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i) + len(end)]


def test_the_composite_selector_is_the_targeted_text_with_both_passages(monkeypatch):
    targeted, weakness, skill, composite = ((V1 / f'selector_{n}.md').read_text() for n in ('targeted', 'weakness', 'skill', 'composite'))
    weak = _passage(weakness, '- Many candidates in this run are built to carry a task property',
                    'not a convention the solver must be told.\n')
    sk = _passage(skill, '- Many candidates in this run centre on a common reasoning skill', 'separate things are not.\n')
    anchor = '- A candidate unlike every reference is eligible'
    k = targeted.index(anchor)
    assert composite == (targeted[:k] + weak + sk + targeted[k:]).replace('diversity-v1-selector-targeted-2',
                                                                        'composite-c1-selector-composite-1')
    for words in ('self-similar structure', 'a local rule applied step\n  by step', 'ordering by a visible quantity',
                  'drawn networks', 'Two plain\n  rules joined'):
        assert words in sk, words
    seen = []

    def fake(path):
        seen.append(pathlib.Path(path).name); raise RuntimeError('stop')
    monkeypatch.setattr(supply, 'load_prompt', fake)
    monkeypatch.setitem(config.SCIENCE, 'prompt_set', 'v1')
    monkeypatch.setitem(config.SCIENCE, 'skill_novelty', True)
    mods = dict(config.SCIENCE.get('idea_modifiers') or {})
    for weakness_run, modifiers in ((True, {**mods, 'RANK': 0.1}), (True, {k: v for k, v in mods.items() if k not in config.SKILL_MODIFIERS}),
                                    (False, {**mods, 'RANK': 0.1})):
        if weakness_run:
            monkeypatch.setitem(config.SCIENCE, 'weakness', deepcopy(WEAK))
        else:
            monkeypatch.delitem(config.SCIENCE, 'weakness', raising=False)
        monkeypatch.setitem(config.SCIENCE, 'idea_modifiers', modifiers)
        with pytest.raises(RuntimeError):
            supply._selector_prompt({'id': 'w'}, {'payload': {}, 'packed_hash': 'x'})
    assert seen == ['selector_composite.md', 'selector_weakness.md', 'selector_skill.md']


# ---------------------------------------------------------------- config and ONE


def test_one_excludes_recurse_and_automaton_but_not_rank_or_graph():
    assert {'RECURSE', 'AUTOMATON'} <= set(slots.ONE_EXCLUDED_MODIFIERS)
    assert not {'RANK', 'GRAPH'} & set(slots.ONE_EXCLUDED_MODIFIERS)
    needed = slots.required_exclusions({'ONE': 1, **{n: 1 for n in NEW}})
    assert {'RECURSE', 'AUTOMATON'} <= set(needed) and not {'RANK', 'GRAPH'} & set(needed)
    value = json.loads((config.PROFILES / 'v1.json').read_text()); s = value['science']
    s['idea_modifiers'].update(ONE=0.1, RECURSE=0.1, RANK=0.1)
    s['modifier_slots']['skill'] = ['RECURSE', 'RANK']
    s['skill_novelty'] = True
    with pytest.raises(config.ConfigError, match='RECURSE'):
        config.validate(value)
    s['modifier_exclusions']['ONE'] = sorted(set(s['modifier_exclusions']['ONE']) | {'RECURSE'})
    config.validate(value)  # RANK needs no exclusion from ONE


def test_a_weakness_profile_may_pin_skill_lines_with_scoped_novelty():
    value = _profile()
    s = value['science']
    s['idea_modifiers'].update(RANK=0.1, GRAPH=0.1, MATCH=0.1)
    s['modifier_slots']['skill'] = ['RANK', 'GRAPH', 'MATCH']
    config.validate(value)
    s['skill_novelty'] = False  # the flag stays optional
    config.validate(value)


# ---------------------------------------------------------------- spec rows, placement and admission


def _spec(*targets):
    return {'revision': 'r', 'seed': 1, 'windows': {'w': {'account': 'main', 'wave': 'A', 'context': 'c01'}},
            'rows': [['w', 'GEO', None, 'U', 'sol', 'high', t] for t in targets]}


def test_spec_rows_mark_a_skill_lines_matched_control():
    from arcgen.batch.allocations.make_allocation import build
    made = build(_spec({'skill': 'RANK', 'size': 'R2'}, {'slot_control': 'RANK', 'size': 'R2'},
                       {'slot_control': 'GRAPH', 'skill': 'LIGHT'}, {'skill': 'MARKS', 'size': 'R4'}),
                 config.SCIENCE['models'], {}, lambda a, i: True)
    assert [c.get('slot_control') for c in made['calls']] == [None, 'RANK', 'GRAPH', None]
    for bad, match in (({'slot_control': 'RANK', 'skill': 'RANK'}, 'pins no skill line'),
                       ({'slot_control': 'RANK', 'skill': ['GRAPH', 'LIGHT']}, 'pins no skill line'),
                       ({'slot_control': 'RANK', 'skill': 'ONE'}, 'pins no skill line'),
                       ({'slot_control': 'RANK', 'skill': 'MARKS'}, 'weakness line'),
                       ({'slot_control': 'RANK', 'structure': 'SHAPE'}, 'structure'),
                       ({'slot_control': 'RANK', 'control': True}, 'control'),
                       ({'slot_control': 'RANK', 'fence': 'strict'}, 'fence'),
                       ({'slot_control': 'RANK', 'size': 'R4'}, 'not R4'),
                       ({'slot_control': 'LIGHT'}, 'not LIGHT'),
                       ({'skill': 'RANK', 'size': 'R4'}, 'R4 needs a weakness line'),
                       ({'size': 'R4'}, 'R4 needs a weakness line')):
        with pytest.raises(ValueError, match=match):
            build(_spec(bad), config.SCIENCE['models'], {}, lambda a, i: True)


def test_placement_never_draws_a_skill_onto_its_control():
    calls = [{'opaque_id': f'p{i:04d}', 'arm': 'GEO', **({'slot_control': 'RANK'} if i < 50 else {})} for i in range(100)]
    assert assign_modifiers(calls, {'RANK': 0.5}, slots={'skill': ['RANK']}, cap=2) == {}
    assert sum('RANK' in c['idea_modifiers'] for c in calls) == 50
    assert not any('RANK' in c['idea_modifiers'] for c in calls if c.get('slot_control'))


def test_arm_ideas_in_a_weakness_run_take_their_calls_fixed_size(weak, monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'rung_draw', {'ladder': {'R2': 1.0}, 'excluded': []})
    out = admission.configured_routes(None, rows(4, idea_modifiers=['RANK'], size_arm='small')
                                      + rows(4, 10, idea_modifiers=['GRAPH'], size_arm='large', size_rung='R2')
                                      + rows(4, 20, slot_control='GRAPH', size_arm='large', size_rung='R3')
                                      + rows(4, 30, idea_modifiers=['RECURSE'], size_arm='large')
                                      + rows(4, 40, idea_modifiers=['AUTOMATON']))  # no size: the run's general draw
    assert [r['study']['rung'] for r in out] == ['R1'] * 4 + ['R2'] * 4 + ['R3'] * 8 + ['R2'] * 4
    assert all('line' not in r['study'] and 'structure' not in r['study'] for r in out)  # arm ideas keep the overlay draws
    assert [r['study']['size_arm'] for r in out[:16]] == ['small'] * 4 + ['large'] * 12
    with pytest.raises(IntegrityError, match='outside the levels'):
        admission.configured_routes(None, rows(1, idea_modifiers=['RANK'], size_arm='large', size_rung='R4'))
    monkeypatch.delitem(config.SCIENCE, 'weakness')  # outside a weakness run nothing changes: the general draw decides
    out = admission.configured_routes(None, rows(4, 50, idea_modifiers=['RANK'], size_arm='small'))
    assert [r['study']['rung'] for r in out] == ['R2'] * 4


# ---------------------------------------------------------------- the parallel gate's temp data

GATE = r'''
import importlib.util, json, pathlib, sys, xml.etree.ElementTree as ET
root, tmp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('gate_entry', root / 'arcgen/batch/entry.py')
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
entry.GATE = tmp / 'gate'  # the merged junit files land here, never in the workspace's gate
entry.TEST_SUITES = {'v1': tmp, 'default': tmp}
ids = {'v1': ['a.py::slow', 'a.py::f1', 'a.py::f2'], 'default': ['b.py::f3']}
entry._collect = lambda profile, suite: ids[profile]
entry._last_durations = lambda profile: {'a.py::slow': 100.0}
seen = []
def call(cmd, **kw):
    basetemp = pathlib.Path(next(a for a in cmd if a.startswith('--basetemp=')).split('=', 1)[1])
    xml = pathlib.Path(next(a for a in cmd if a.startswith('--junitxml=')).split('=', 1)[1])
    basetemp.mkdir(parents=True)
    (basetemp / 'run').mkdir(); (basetemp / 'run' / 'bank.json').write_bytes(b'x' * 100000)
    seen.append(str(basetemp))
    names = [a for a in cmd if '::' in a]
    suite = ET.Element('testsuite', tests=str(len(names)), errors='0', failures='0', skipped='0', time='1.0')
    for n in names:
        ET.SubElement(suite, 'testcase', classname='tests.' + n.split('.py')[0], name=n.split('::')[-1], time='1.0')
    top = ET.Element('testsuites'); top.append(suite); ET.ElementTree(top).write(xml)
    kw['stdout'].write('log of ' + basetemp.name + '\n')
    return 1 if 'b.py::f3' in names else 0  # a failing job loses its temp data too
entry.subprocess.call = call
results = entry.run_tests_parallel(2)
scratch = pathlib.Path(seen[0]).parent
print(json.dumps({'results': results, 'seen': seen, 'left': sorted(p.name for p in scratch.iterdir())}))
'''


def test_the_parallel_gate_deletes_each_jobs_basetemp_and_keeps_its_log_and_xml(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    env = dict(os.environ, TMPDIR=str(tmp_path))
    out = subprocess.run([sys.executable, '-B', '-c', GATE, str(root), str(tmp_path)], env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-3000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report['results'] == {'v1': 'passed', 'default': 'failed'}
    assert len(report['seen']) == 3 and not any(pathlib.Path(p).exists() for p in report['seen'])
    left = report['left']
    assert not [n for n in left if n.startswith('tmp-')]
    assert sorted(n for n in left if n.endswith('.log')) == ['default-002.log', 'v1-000.log', 'v1-001.log']
    assert sorted(n for n in left if n.endswith('.xml')) == ['default-002.xml', 'v1-000.xml', 'v1-001.xml']
    assert (tmp_path / 'gate/focused_tests_v1.xml').exists()
