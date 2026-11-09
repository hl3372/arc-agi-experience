"""Targeted families and scale rungs (2026-10-12): call attributes
that leave untargeted call IDs unchanged, family idea texts, the family pivot and selector sentence, planner family and
rung lines, the plan-stage floor check with one resumable re-plan, the general run's R1 draw, and the grid measures."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest
from arcgen.batch import admission, config, contracts, idea_v1, properties, rungs, supply
from arcgen.batch.allocation import assign_modifiers, design
from arcgen.batch.common import IntegrityError, digest, read
from arcgen.lib.util import render_prompt

FLOORS = {'R1': {'output_cells': 150, 'computed_cells': 30, 'escalation': 0},
          'R2': {'output_cells': 300, 'computed_cells': 80, 'escalation': 1.3},
          'R3': {'output_cells': 450, 'computed_cells': 150, 'escalation': 1.3}}  # R3 600 -> 450, 2026-10-13
V1 = idea_v1.PROMPTS.parent


def _call(**extra):
    c = next(c for w in design()['windows'] if w['kind'] == 'fresh_supply' for c in w['calls'])
    return supply.make_call(arm=c['arm'], bank_setting='loop', effort=c['effort'], idea_model=c['idea_model'], context=c['context'],
                            seed=c['seed'], account=c['account'], window=c['window'], replicate=c['replicate'], **extra), c


def test_untargeted_call_ids_are_unchanged_and_targeted_ones_differ():
    plain, c = _call()
    recipe = {'revision': supply.REVISION, 'arm': c['arm'], 'invitation': None, 'bank_setting': 'loop', 'effort': c['effort'],
              'idea_model': c['idea_model']}
    key = {**recipe, 'context': c['context'], 'seed': c['seed'], 'account': c['account'], 'window': c['window'],
           'replicate': c['replicate']}
    assert plain['opaque_id'] == 'p' + digest(key)[:18] and 'family' not in plain  # the formula every earlier batch used
    targeted, _ = _call(family='ENCL', rung='R2')
    assert targeted['opaque_id'] != plain['opaque_id'] and (targeted['family'], targeted['rung']) == ('ENCL', 'R2')
    for bad in ({'family': 'ENCL'}, {'rung': 'R2'}, {'family': 'NOPE', 'rung': 'R2'}, {'family': 'SIM', 'rung': 'R9'}):
        with pytest.raises(IntegrityError):
            _call(**bad)


def test_allocation_rows_carry_family_and_rung_only_when_given():
    from arcgen.batch.allocations.make_allocation import build
    spec = {'revision': 'r', 'seed': 1, 'windows': {'w': {'account': 'main', 'wave': 'A', 'context': 'c01', 'commit': 'explore'}},
            'rows': [['w', 'SEE', None, 'L', 'sol', 'high'], ['w', 'SEE', None, 'L', 'sol', 'high', {'family': 'FIT', 'rung': 'R1'}]]}
    value = build(spec, {'sol': 'm'}, {}, lambda a, i: True)
    plain, targeted = value['calls']
    assert 'family' not in plain and (targeted['family'], targeted['rung']) == ('FIT', 'R1')
    assert value['coverage']['family_rung_calls'] == {'FIT/R1': 1}
    old = build({**spec, 'rows': spec['rows'][:1]}, {'sol': 'm'}, {}, lambda a, i: True)['calls'][0]
    assert old['private_id'] == plain['private_id']
    with pytest.raises(ValueError):
        build({**spec, 'rows': [['w', 'SEE', None, 'L', 'sol', 'high', {'family': 'FIT'}]]}, {'sol': 'm'}, {}, lambda a, i: True)


def test_targeted_idea_prompts_add_the_family_and_rung_texts_and_use_the_family_steps():
    targeted, _ = _call(family='SIM', rung='R3')
    targeted.update(commit='explore', idea_modifiers=['IGNORE'])
    plain = {**targeted, 'family': None, 'rung': None}
    t, p = idea_v1.sketch_request(targeted)['prompt'], idea_v1.sketch_request(plain)['prompt']
    family = (idea_v1.PROMPTS / 'families/SIM.md').read_text().split('-->\n', 1)[1].strip()
    rung = (idea_v1.PROMPTS / 'rungs/R3.md').read_text().split('-->\n', 1)[1].strip()
    assert family in t and rung in t and 'Kind of task for this call' not in p
    assert t.index('## Direction for this call') < t.index(family) < t.index(rung) < t.index('Additional requirement')
    assert 'its test will be large, with the laws deciding many places' in t and 'a simple idea can be small and clean' in p
    sketches = [{'id': 's01', 'title': 'T', 'idea': 'I.', 'recipe': 'R.'}]
    near = [[{'reference_id': 'committed:x', 'similarity': 0.5, 'text': 'A task.'}]]
    req = idea_v1.sketch_request(targeted)
    explore = idea_v1.explore_request(targeted, req, sketches, near)['prompt']
    commit = idea_v1.commit_request(targeted, req, sketches, near)['prompt']
    assert 'a different kind of encounter between the laws' in explore and 'at most 25 words' in explore
    assert 'a listed task\'s rule run on a bigger or busier scene does make it redundant' in commit
    plain_explore = idea_v1.explore_request(plain, idea_v1.sketch_request(plain), sketches, near)['prompt']
    assert 'would be far from the ones shown' in plain_explore
    with pytest.raises(IntegrityError):  # targeted calls use the explore loop
        idea_v1.sketch_request({**targeted, 'commit': 'kept'})


def test_every_family_and_rung_has_its_texts_and_no_eval_identifiers():
    import re
    for f in config.FAMILIES:
        assert (idea_v1.PROMPTS / 'families' / f'{f}.md').exists() and (contracts.FAMILY_DIR / f'{f}.planner.md').exists()
        for r in config.RUNGS:
            assert (contracts.RUNG_DIR / f'{f}.{r}.planner.md').exists()
    assert (contracts.RUNG_DIR / 'ANY.R1.planner.md').exists()
    texts = [p.read_text() for d in (idea_v1.PROMPTS / 'families', idea_v1.PROMPTS / 'rungs', contracts.FAMILY_DIR, contracts.RUNG_DIR)
             for p in d.glob('*.md')]
    assert texts and not any(re.search(r'\b[0-9a-f]{8}\b', t.split('-->', 1)[1]) for t in texts)  # no task IDs


def test_targeted_selector_adds_exactly_its_two_passages():
    """The fresh-insight sentence (2026-10-12) and, after the screen (2026-10-13), conventions as defects, not novelty."""
    plain, targeted = (V1 / 'selector.md').read_text(), (V1 / 'selector_targeted.md').read_text()
    sentence = ("\n- Familiar laws are not redundancy by themselves: when the answers depend on how those\n  laws meet in a way no "
                "reference's examples teach, that is a fresh insight; when a\n  reference's rule run on a larger or busier "
                "scene already produces the answers, it is not.")
    conventions = ("When a candidate's fresh insight is a convention the solver must be told rather than\nsomething the grids show, "
                   "such as a colour code, a certificate, a processing order, a\ntie-break or a framing rule, treat it as a quality "
                   "defect, not as novelty. In `develop`,\nwhere the kernel leaves a choice open, ask for scenes in which the picture "
                   "settles the\nchoice or it never arises, never for a convention to establish or teach.\n")
    assert targeted.count(sentence) == targeted.count(conventions) == 1
    assert targeted.replace(sentence, '', 1).replace(conventions, '', 1).replace('diversity-v1-selector-targeted-2', 'diversity-v1-selector-2') == plain


def test_quality_scale_changes_only_the_two_sentences():
    q, s = (V1 / 'quality.md').read_text(), (V1 / 'quality_scale.md').read_text()
    assert 'small clean\nepisode is the right answer' in q and 'small clean' not in s
    assert 'long step-by-step bookkeeping' in q and 'long step-by-step bookkeeping' not in s
    assert 'is not bookkeeping' in s
    assert 'settle it with the\npicture' in s and 'No test answer may repeat a training\nanswer' in s  # 2026-10-13
    same = lambda t: [l for l in t.splitlines()[12:] if l]  # everything after the two changed paragraphs
    assert same(q)[-20:] == same(s)[-20:]


def test_planner_family_and_rung_lines_and_unchanged_untargeted_composition():
    b = contracts.Bindings()
    before = b.planning_templates('SEE', (), colour='vary', pose=True, slots=('IGNORE',))
    assert before == b.planning_templates('SEE', (), colour='vary', pose=True, slots=('IGNORE',), family=None, rung=None)
    rows = b.planning_templates('SEE', (), colour='vary', pose=True, slots=('IGNORE',), family='ENCL', rung='R2')
    texts = [r[0] for r in rows]
    assert 'is not bookkeeping' in texts[1] and 'small clean' not in texts[1]
    assert 'Kind of task for this idea: enclosure' in texts[-2] and 'at least 300 cells' in texts[-1]
    assert [r[2] for r in rows[2:-2]] == [r[2] for r in before[2:]]
    general = b.planning_templates('SEE', (), rung='R1')
    assert 'plan it at the largest scale it supports' in general[-1][0]
    with pytest.raises(Exception):
        b.planning_templates('SEE', (), family='ENCL')  # a family needs its rung
    joint = b.planning_templates('J', ('v43_episode_joint_plan',), family='FIT', rung='R3')
    assert 'at least 450 cells' in joint[-2][0]  # the route addendum stays last
    # 2026-10-13: R3 states the longest dependent chain per pair, with one comparative invariant, and the per-event third
    assert 'longest chain of placements' in joint[-2][0] and 'more than a third of the computed cells' in joint[-2][0]
    sim = b.planning_templates('SEE', (), family='SIM', rung='R3')[-1][0]
    assert "The test's longest chain of dependent encounters is longer than that of any training pair." in sim
    assert 'makes up most' not in sim
    r2 = b.planning_templates('SEE', (), family='SIM', rung='R2')[-1][0]
    assert 'a plausible rival reading would place it differently' in r2 and 'makes up most of the computed cells' in r2
    fit = b.planning_templates('SEE', (), family='FIT', rung='R2')[-1][0]
    assert 'what exactly makes two edges fit' in fit and 'near miss' in fit


def _plan(tid, train, test_in, test_out, status='ready'):
    dims = lambda a: {'height': a[0], 'width': a[1]}
    return {'candidate_id': tid, 'status': status,
            'public_contract': {'training_pairs': [{'input_dimensions': dims(a), 'output_dimensions': dims(a)} for a in train],
                                'test_inputs': [{'input_dimensions': dims(test_in)}]},
            'private_execution': {'test_outputs': [{'output_dimensions': dims(test_out)}]}} if status == 'ready' else \
        {'candidate_id': tid, 'status': 'abort', 'abort_reason': 'cannot', 'public_contract': None, 'private_execution': None}


def test_plan_measures_floors_and_feedback(monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    m = rungs.measures(_plan('t', [(10, 10), (12, 12)], (14, 14), (14, 14)))
    assert m['test_output_cells'] == 196 and m['escalation'] == round(196 / 122, 3)
    assert rungs.reached(m) == 'R1' and rungs.shortfalls('R1', m) == []
    short = rungs.shortfalls('R2', m)
    assert len(short) == 1 and '14×14 = 196 cells' in short[0]
    m2 = rungs.measures(_plan('t', [(20, 20)], (20, 20), (20, 20)))
    assert rungs.reached(m2) == 'R1' and len(rungs.shortfalls('R2', m2)) == 1 and 'times the size' in rungs.shortfalls('R2', m2)[0]
    assert 'abort; the earlier plan will then be used' in rungs.feedback('R2', m)
    assert rungs.plan_of({'concrete_plan': _plan('t', [(3, 3)], (3, 3), (3, 3))})['candidate_id'] == 't'
    assert rungs.plan_of(_plan('t', [], None, None, status='abort')) is None
    assert rungs.task_of_plan_call('d0123-replan-concrete-plan') == rungs.task_of_plan_call('d0123-concrete-plan') == 'd0123'
    from arcgen.batch.routing import resolve
    assert resolve('planning', 'd0123-replan-concrete-plan', planner='sol')


class _Fake:
    """Just enough of the batch runtime for _plan_with_colour: bindings, a run directory and a scripted planner."""
    def __init__(self, tmp_path, answers):
        self.q2, self.run_dir, self.answers, self.calls = contracts.Bindings(), tmp_path, answers, []

    def invoke_rendered(self, **kw):
        self.q2.validate_request(kw['stage'], kw['prompt'], kw['request_context'])
        self.calls.append(kw)
        return SimpleNamespace(value=deepcopy(self.answers[kw['call_id']]))


def _planner_kw(tid, record):
    values = {'CANDIDATE': {'candidate_id': tid, 'central_relation': 'A relation.'},
              'EDITORIAL_BRIEF': {'candidate_id': tid, 'development_priority': 1, 'route': 'wildcard', 'preserve': 'P',
                                  'develop': 'Use a small grid.', 'avoid': 'R.', 'closest_similarity_concern': 'x',
                                  'selection_reason': 'y', 'optional_development_suggestion': None},
              'ROUTE_GUIDANCE': 'G.', 'CANDIDATE_ID': tid}
    return {'composed_with': ('v44_quality_plan',), 'values': contracts.v1_planner_values(record, values),
            'call_id': f'{tid}-concrete-plan', 'parent_ids': [tid], 'arm_id': 'anonymous-authoring', 'role': 'concrete_planner',
            'expected': {'candidate_id': tid}}


def test_a_plan_below_its_floor_is_planned_once_more_and_resumes_exactly(monkeypatch, tmp_path):
    from arcgen.batch.runtime import Runtime
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    tid = 'dtest0000000000000002'
    record = {'private_route': 'ordinary_current', 'selection_note': {'useful_contribution': 'Distinct.'},
              'study': {'primary_arm': 'SEE', 'family': 'ENCL', 'rung': 'R2', 'colour_roles': 'v1', 'pose': False}}
    small, large = _plan(tid, [(10, 10)], (12, 12), (12, 12)), _plan(tid, [(12, 12), (14, 14)], (20, 20), (20, 20))
    fake = _Fake(tmp_path, {f'{tid}-concrete-plan': small, f'{tid}-replan-concrete-plan': large})
    value = Runtime._plan_with_colour(fake, _planner_kw(tid, record), record, 'v1', False)
    assert value == large and [c['call_id'] for c in fake.calls] == [f'{tid}-concrete-plan', f'{tid}-replan-concrete-plan']
    first, second = fake.calls
    assert second['prompt'].startswith(first['prompt'] + '\n\n') and '12×12 = 144 cells' in second['prompt']
    assert 'at least 300 cells' in first['prompt'] and 'Kind of task for this idea: enclosure' in first['prompt']
    receipt = read(tmp_path / 'rung_check' / f'{tid}.json')
    assert receipt['used'] == 'second' and receipt['first_reached'] is None and receipt['reached'] == 'R2'
    again = _Fake(tmp_path, fake.answers)  # resume: the same calls and the same receipt
    assert Runtime._plan_with_colour(again, _planner_kw(tid, record), record, 'v1', False) == large
    assert [c['prompt'] for c in again.calls] == [c['prompt'] for c in fake.calls]
    # A re-plan that aborts keeps the first plan, labelled with the rung it reached.
    tid2 = 'dtest0000000000000003'
    fake2 = _Fake(tmp_path, {f'{tid2}-concrete-plan': _plan(tid2, [(10, 10)], (13, 13), (13, 13)),
                             f'{tid2}-replan-concrete-plan': _plan(tid2, [], None, None, status='abort')})
    value = Runtime._plan_with_colour(fake2, _planner_kw(tid2, record), record, 'v1', False)
    assert value['public_contract'] and read(tmp_path / 'rung_check' / f'{tid2}.json')['used'] == 'first'
    assert read(tmp_path / 'rung_check' / f'{tid2}.json')['reached'] == 'R1'
    # A plan at its floor is not re-planned; an idea without a rung is never checked.
    tid3 = 'dtest0000000000000004'
    fake3 = _Fake(tmp_path, {f'{tid3}-concrete-plan': _plan(tid3, [(12, 12)], (18, 18), (18, 18))})
    Runtime._plan_with_colour(fake3, _planner_kw(tid3, record), record, 'v1', False)
    assert len(fake3.calls) == 1 and read(tmp_path / 'rung_check' / f'{tid3}.json')['replanned'] is False
    plain = {**record, 'study': {'primary_arm': 'SEE', 'colour_roles': 'v1', 'pose': False}}
    tid4 = 'dtest0000000000000005'
    fake4 = _Fake(tmp_path, {f'{tid4}-concrete-plan': _plan(tid4, [(3, 3)], (3, 3), (3, 3))})
    Runtime._plan_with_colour(fake4, _planner_kw(tid4, plain), plain, 'v1', False)
    assert len(fake4.calls) == 1 and not (tmp_path / 'rung_check' / f'{tid4}.json').exists()


def test_a_replan_prompt_must_carry_its_rung(tmp_path):
    tid = 'dtest0000000000000006'
    record = {'private_route': 'ordinary_current', 'selection_note': {'useful_contribution': 'Distinct.'},
              'study': {'primary_arm': 'SEE', 'colour_roles': 'v1', 'pose': False}}
    from arcgen.batch.runtime import Runtime
    fake = _Fake(tmp_path, {f'{tid}-concrete-plan': _plan(tid, [(3, 3)], (3, 3), (3, 3))})
    Runtime._plan_with_colour(fake, _planner_kw(tid, record), record, 'v1', False)
    call, = fake.calls
    b = contracts.Bindings()
    with pytest.raises(Exception, match='re-plan note without a rung'):
        b.validate_request(call['stage'], call['prompt'] + '\n\nnote', {**call['request_context'], 'q2_replan': 'note'})
    with pytest.raises(Exception, match='exact local components'):  # and the note must be the one appended
        b.validate_request(call['stage'], call['prompt'] + '\n\nother', {**call['request_context'], 'q2_rung': 'R1',
                                                                          'q2_replan': 'note'})


def test_exclusions_can_name_a_family_or_rung():
    calls = [{'opaque_id': f'p{i:04d}', 'arm': 'SEE', 'family': ('SIM', 'LEGEND')[i % 2], 'rung': ('R1', 'R2', 'R3')[i % 3]}
             for i in range(120)]
    short = assign_modifiers(calls, {'EASIER': 0.2, 'BUSY': 0.1}, slots={'demand': ['EASIER'], 'scene': ['BUSY']}, cap=1,
                             exclusions={'EASIER': ['R2', 'R3'], 'BUSY': ['SIM']})
    assert short == {}
    assert all(c['rung'] == 'R1' for c in calls if 'EASIER' in c['idea_modifiers'])
    assert all(c['family'] == 'LEGEND' for c in calls if 'BUSY' in c['idea_modifiers'])


def test_general_run_draws_r1_per_idea_except_excluded_ideas(monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    monkeypatch.setitem(config.SCIENCE, 'rung_draw', {'rung': 'R1', 'rate': 0.3, 'excluded': ['DIGEST', 'SCENE', 'MANY']})
    monkeypatch.setitem(config.SCIENCE, 'pose_off_families', ['SIM'])
    rows = [{'source_candidate_id': f'k{i:04d}', 'private_source': 'ordinary_unconditioned',
             'study': {'primary_arm': 'SEE', 'idea_modifiers': ['SCENE'] if i % 5 == 0 else []}} for i in range(1000)]
    routed = admission.configured_routes({}, deepcopy(rows))
    drawn = [r for r in routed if r['study'].get('rung')]
    assert all('SCENE' not in r['study']['idea_modifiers'] and r['study'].get('overlay') != 'MANY' for r in drawn)
    eligible = [r for r in routed if 'SCENE' not in r['study']['idea_modifiers'] and r['study'].get('overlay') != 'MANY']
    assert 0.25 < len(drawn) / len(eligible) < 0.35 and {r['study']['rung'] for r in drawn} == {'R1'}
    assert routed == admission.configured_routes({}, deepcopy(rows))  # seeded: the same on resume
    sim = [{'source_candidate_id': f'q{i:04d}', 'private_source': 'ordinary_unconditioned',
            'study': {'primary_arm': 'WORLD', 'family': 'SIM', 'rung': 'R2'}} for i in range(200)]
    out = admission.configured_routes({}, sim)
    assert not any(r['study']['pose'] for r in out) and all(r['study']['rung'] == 'R2' for r in out)


def test_config_validates_rungs_and_draws():
    base = json.loads((config.PROFILES / 'v1.json').read_text())
    def ok(**science):
        v = deepcopy(base); v['science'].update(science); return config.validate(v)
    ok(rungs=FLOORS, rung_draw={'rung': 'R1', 'rate': 0.3, 'excluded': ['DIGEST', 'MANY']}, targeted=True,
       pose_off_families=['SIM'])
    for bad in ({'rungs': {'R4': FLOORS['R1']}}, {'rungs': {'R1': {'output_cells': 1}}},
                {'rung_draw': {'rung': 'R1', 'rate': 0.3, 'excluded': []}}, {'targeted': True},
                {'rungs': FLOORS, 'rung_draw': {'rung': 'R1', 'rate': 2, 'excluded': []}},
                {'rungs': FLOORS, 'rung_draw': {'rung': 'R1', 'rate': 0.3, 'excluded': ['NOPE']}},
                {'pose_off_families': ['NOPE']}):
        with pytest.raises(config.ConfigError):
            ok(**bad)


def _task(train, test):
    return {'train': [{'input': a, 'output': b} for a, b in train], 'test': [{'input': a, 'output': b} for a, b in test]}


def test_scene_check_and_scale_measures(monkeypatch):
    blank = lambda h, w, c=0: [[c] * w for _ in range(h)]
    def dots(h, w, cells, c=3):
        g = blank(h, w)
        for r, k in cells:
            g[r][k] = c
        return g
    # an edit: same size, most cells kept
    edit = _task([(dots(6, 6, [(1, 1)]), dots(6, 6, [(1, 1), (2, 2)]))] * 2, [(dots(7, 7, [(1, 1)]), dots(7, 7, [(1, 1), (5, 5)]))])
    assert properties.checks(edit)['SCENE'] is False
    # a crop: the output is a sub-grid of the input
    big = [[(r * 7 + c) % 5 for c in range(12)] for r in range(12)]
    crop = _task([(big, [row[2:9] for row in big[1:8]]), (big, [row[0:7] for row in big[0:7]])], [(big, [row[3:10] for row in big[2:9]])])
    assert properties.checks(crop)['SCENE'] is False
    # a new scene: outputs unlike their inputs, different sizes, more than 30 cells
    scene = _task([(dots(10, 10, [(1, 1), (5, 5)]), blank(6, 6, 4)), (dots(10, 10, [(2, 2)]), blank(7, 5, 2))],
                  [(dots(10, 10, [(3, 3)]), [[(r + c) % 3 + 1 for c in range(8)] for r in range(5)])])
    assert properties.checks(scene)['SCENE'] is True
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    test_in = dots(14, 14, [(2, 2), (2, 10), (10, 2), (10, 10)], 5)
    test_out = deepcopy(test_in)
    for r0, c0 in ((2, 2), (2, 10), (10, 2), (10, 10)):
        for r in range(r0 - 1, r0 + 2):
            for c in range(c0 - 1, c0 + 2):
                if (r, c) != (r0, c0):
                    test_out[r][c] = 4
    s = properties.scale(_task([(dots(8, 8, [(2, 2)]), dots(8, 8, [(2, 2)]))], [(test_in, test_out)]))
    assert (s['computed_cells'], s['regions'], s['cells_per_region'], s['test_output_cells']) == (32, 4, 8.0, 196)
    assert s['amplification'] == 8.0 and s['escalation'] == round(196 / 64, 2) and s['rung'] == 'R1'
    value = properties.report([{'complete_task': _task([(dots(8, 8, [(2, 2)]), dots(8, 8, [(2, 2)]))], [(test_in, test_out)]),
                                'study': {'family': 'SIM', 'rung': 'R1', 'primary_arm': 'WORLD'}}])
    assert value['by_rung'] == {'SIM R1': {'verified': 1, 'reached': 1, 'test_output_cells': 196, 'computed_cells': 32,
                                           'regions': 4, 'cells_per_region': 8.0}}
    assert any('Scale rungs' in line for line in properties.lines(value))
    monkeypatch.delitem(config.SCIENCE, 'rungs')  # a summary loaded under a profile without rungs uses the run's floors
    again = properties.report([{'complete_task': _task([(dots(8, 8, [(2, 2)]), dots(8, 8, [(2, 2)]))], [(test_in, test_out)]),
                                'study': {'family': 'SIM', 'rung': 'R1', 'primary_arm': 'WORLD'}}], FLOORS)
    assert again['by_rung']['SIM R1']['reached'] == 1


def test_targeted_full_mock_pipeline_replans_once_and_resumes(tmp_path, monkeypatch):
    """A full mock development of an ENCL R1 idea: the fixture's small plan misses the floor, is planned once more (the
    fifth primary call of its group), then realized, verified and solved; two resumes make no new call."""
    import sys, pathlib
    from support.pilot import run_case
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    run_case(tmp_path, monkeypatch, 'clear', 5, True, arm='SEE', planner='sol', study={'family': 'ENCL', 'rung': 'R1'}, primary=5)
    plans = sorted(tmp_path.glob('accounts/**/model_calls/astra_concrete_plan_v3/*/prompt.md'))
    assert sorted(p.parent.name.split('-', 1)[1] for p in plans) == ['concrete-plan', 'replan-concrete-plan']
    first, second = sorted((p.read_text() for p in plans), key=len)
    assert second.startswith(first + '\n\nScale check on an earlier plan') and 'Kind of task for this idea: enclosure' in first
    receipt = read(next(tmp_path.glob('accounts/**/rung_check/*.json')))
    assert receipt['replanned'] and receipt['used'] == 'second' and receipt['reached'] is None


def test_summary_reports_family_rungs_plan_checks_and_rule_length(tmp_path, monkeypatch):
    from arcgen.batch import cli
    monkeypatch.setattr(cli, 'run_root', lambda name: tmp_path / name)
    monkeypatch.setattr(cli, 'ops', lambda name: (tmp_path / (name + '-ops')).mkdir(exist_ok=True) or tmp_path / (name + '-ops'))
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    root = tmp_path / 'b'; down = root / 'accounts/main/w-develop-01/downstream'
    (down / 'rung_check').mkdir(parents=True); (root / 'analysis').mkdir()
    task = _task([([[0] * 4] * 4, [[1] * 4] * 4)], [([[0] * 4] * 4, [[1] * 4] * 4)])
    plan = {'public_contract': {'semantic_core': {'central_rule': 'Fill every enclosed hole with the frame colour.'}}}
    records = {'t1': {'complete_task': task, 'concrete_plan': plan, 'study': {'family': 'ENCL', 'rung': 'R1', 'primary_arm': 'SEE'}},
               't2': {'complete_task': task, 'concrete_plan': plan, 'study': {'primary_arm': 'SEE'}}}
    (down / 'state.json').write_text(json.dumps({'semantic_records': records}))
    rows = [{'task_id': 't1', 'usable_after_exact_palette_union': True, 'arm': 'SEE', 'generation_model': 'm', 'effort': 'high',
             'family': 'ENCL', 'rung': 'R1', 'status': 'x'},
            {'task_id': 't2', 'usable_after_exact_palette_union': True, 'arm': 'SEE', 'generation_model': 'm', 'effort': 'high', 'status': 'x'}]
    (root / 'analysis/terminal_report.private.json').write_text(json.dumps({'tasks': rows}))
    (down / 'rung_check/t1.json').write_text(json.dumps({'rung': 'R1', 'replanned': True, 'used': 'second', 'reached': 'R1'}))
    value = cli.summary('b', write_files=True)
    assert value['by_family_rung'] == {'ENCL R1': 1} and value['central_rule_words_median'] == {'ENCL R1': 8}
    assert value['rung_plan_checks'] == {'plans_checked': 1, 'replanned': 1, 'replan_kept': 1, 'reached_planned_rung': 1}
    text = (tmp_path / 'b-ops/SUMMARY.md').read_text()
    assert 'Plan-stage scale check: 1 plans checked' in text and 'Central rule, median words' in text
