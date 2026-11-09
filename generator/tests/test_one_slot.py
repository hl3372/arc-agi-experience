"""ONE, the single-demonstration-pair slot (2026-10-23): idea and binding planner texts, exclusions enforced in config, spec rows and admission, the planned pair
count with the shared re-plan and its receipt, the adjudicator policy and repair note, the final pair count, and a one-pair
task through the mock pipeline (no validator, verifier or schema refuses it). Also the RECOMBINE fix: the loosened text,
the form label and the order check that follows it."""
from copy import deepcopy
import json, pathlib, re, sys
import pytest
from arcgen.batch import admission, config, contracts, gate, idea_v1, lines, properties, slots, structures, supply
from arcgen.batch.allocation import assign_modifiers
from arcgen.batch.common import IntegrityError, read
from arcgen.lib.astra_authoring_contract_v3 import pair_specific_test_narrative
from arcgen.lib.util import render_prompt
from test_weakness_lines import WEAK, FLOORS, rows, _lplan, _record, _task, _row, _emit
from test_targeted import _Fake, _planner_kw, _plan as _tplan
from support.pilot import run_case

V1 = pathlib.Path(contracts.HERE) / 'prompts/v1'
ONE_TEXT = (V1 / 'slots/ONE.planner.md').read_text().split('-->', 1)[1].strip()
POLICY = slots.body('ONE.adjudicator.md')
NOTE = slots.body('ONE.repair_note.md')


@pytest.fixture
def weak(monkeypatch):
    w = deepcopy(WEAK)
    w['structures'] = {'RECOMBINE': {}, 'SHAPE': {}}
    monkeypatch.setitem(config.SCIENCE, 'weakness', w)
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    return config.SCIENCE['weakness']


# ---------------------------------------------------------------- texts


def test_the_texts_say_one_pair_and_override_pair_counts_for_this_idea_only():
    idea = (V1 / 'ideas/modifiers/ONE.md').read_text()
    assert '<!-- prompt-version: weakness-v1-idea-modifier-ONE-' in idea and idea.split('-->', 1)[1].strip().startswith(
        'Additional requirement for this call: each idea is taught by a single demonstration, one example pair in place of the '
        'usual few.')
    for phrase in ('several instances inside the one grid', 'a new arrangement of the same kinds of case', 'not a copy'):
        assert phrase in idea, phrase
    for phrase in ('(binding; for this idea only, it holds over any line or note above or below that asks for two or more '
                   'training pairs', 'Use exactly one training pair.', 'Laws: one or two', 'acts at least twice, on separate things',
                   'plain at a glance', 'Each test meets no kind of case the pair does not show',
                   'a new configuration of those cases (different counts, positions or arrangement)',
                   'not the pair\'s input recoloured, shifted, turned, mirrored or cropped'):
        assert phrase in ONE_TEXT, phrase
    assert 'ONE' in contracts.SLOT_BRIDGES and 'ONE' in config.MODIFIER_NAMES and 'ONE' in idea_v1.MODIFIERS
    for text in (ONE_TEXT, POLICY, NOTE, idea):
        assert not re.search(r'\b[0-9a-f]{8}\b', text)
    assert POLICY.startswith('Review policy for this episode. It teaches its rule from exactly one training pair by design.')
    assert 'judge sufficiency by the instances inside the pair' in POLICY and 'rather than adding a second' in POLICY
    assert 'Keep exactly one training pair' in NOTE and slots.POLICY_VERSION == 'weakness-v1-slot-adjudicator-ONE-1'
    # the base adjudicator already tells a one-demo task to replace its demonstration; the policy agrees with it
    assert 'A one-demo task\nmust replace its demonstration.' in (V1 / 'difficulty_adjudicator.md').read_text()


def test_the_planner_composes_the_line_with_the_slots_and_a_one_idea_request_carries_the_idea_text():
    b = contracts.Bindings()
    texts = [r[0] for r in b.planning_templates('SEE', (), slots=('EASIER', 'ONE'))]
    assert any(ONE_TEXT in t for t in texts) and texts.index(next(t for t in texts if ONE_TEXT in t)) == len(texts) - 1
    rung = [r[0] for r in b.planning_templates('SEE', (), slots=('ONE',), rung='R2', canvas='square')]
    assert ONE_TEXT in rung[-3] and 'Scale for this idea' in rung[-2] and 'Canvas for this idea' in rung[-1]
    call = supply.make_call(arm='SEE', bank_setting='loop', context='c01', seed=1, account='main', window='w', effort='high',
                            idea_model=config.SCIENCE['models']['sol'])
    prompt = idea_v1.sketch_request({**call, 'idea_modifiers': ['ONE']})['prompt']
    assert 'each idea is taught by a single demonstration' in prompt and 'unrelated addition' in prompt


# ---------------------------------------------------------------- exclusions: config, spec rows, placement, admission


def _v1():
    return json.loads((config.PROFILES / 'v1.json').read_text())


def test_the_reference_profile_keeps_one_in_the_episode_slot_with_its_exclusions():
    s = config.SCIENCE
    assert s['idea_modifiers']['ONE'] == 0.0 and 'ONE' in s['modifier_slots']['episode']
    assert set(slots.ONE_EXCLUDED_ARMS) | set(slots.ONE_EXCLUDED_FAMILIES) <= set(s['modifier_exclusions']['ONE'])
    assert slots.exclusion_problems(s, config.LINE_MODIFIERS) == []
    assert {'KEY', 'CHAINED', 'MINIMAL', 'DISTRIBUTED', 'INSTRUCT'} <= set(s['modifier_slots']['episode'])
    for kept in ('CONSTRUCT', 'EASIER', 'COLOURFUL', 'BUSY', 'PACKED', 'VACANCY', 'MONO', 'LIGHT', 'MEASURE'):
        assert kept not in slots.ONE_EXCLUDED_MODIFIERS  # compatible: nothing in their text needs a second pair
    for arm in ('SEE', 'GEO', 'WORLD', 'PER', 'RESTORE', 'GEN', 'TEMPLATE', 'DIRECT', 'BOARD', 'RHYTHM', 'SEM', 'SIGNAL'):
        assert arm not in slots.ONE_EXCLUDED_ARMS and arm in idea_v1.ARMS
    assert set(slots.ONE_EXCLUDED_ARMS) <= set(idea_v1.ARMS) and set(slots.ONE_EXCLUDED_FAMILIES) <= set(config.FAMILIES)
    assert set(slots.ONE_EXCLUDED_MODIFIERS) <= set(config.MODIFIER_NAMES)
    # every skill line needs evidence across pairs except RANK and GRAPH (composite batch c1, 2026-10-25), whose cases
    # repeat inside one scene
    assert set(config.SKILL_MODIFIERS) & set(slots.ONE_EXCLUDED_MODIFIERS) == set(config.SKILL_MODIFIERS) - {'RANK', 'GRAPH'}


def test_config_refuses_a_profile_that_lets_one_meet_an_excluded_name():
    value = _v1(); value['science']['idea_modifiers']['ONE'] = 0.2
    config.validate(value)
    for name in ('SPREAD', 'J', 'LEGEND', 'EXTENT', 'DIGEST'):
        bad = deepcopy(value); bad['science']['modifier_exclusions']['ONE'].remove(name)
        with pytest.raises(config.ConfigError, match=name):
            config.validate(bad)
    # the other side's list or a shared slot also keeps them apart
    other = deepcopy(value); other['science']['modifier_exclusions']['ONE'].remove('EXTENT')
    other['science']['modifier_exclusions']['EXTENT'] = ['ONE']
    config.validate(other)
    moved = deepcopy(value); moved['science']['modifier_exclusions']['ONE'].remove('KEY')
    config.validate(moved)  # KEY shares the episode slot
    moved['science']['modifier_slots']['episode'].remove('KEY'); moved['science']['modifier_slots']['key'] = ['KEY']
    with pytest.raises(config.ConfigError, match='KEY'):
        config.validate(moved)
    bare = deepcopy(value); bare['science'].pop('modifier_exclusions')
    with pytest.raises(config.ConfigError):
        config.validate(bare)


def test_config_refuses_one_beside_a_weakness_line_it_does_not_exclude():
    from test_weakness_lines import _profile
    value = _profile()
    s = value['science']
    s['idea_modifiers']['ONE'] = 0.2; s['modifier_slots']['pairs'] = ['ONE']
    s['modifier_exclusions']['ONE'] = slots.required_exclusions({k: 1 for k in s['idea_modifiers'] if k != 'MARKS'})
    with pytest.raises(config.ConfigError, match='MARKS'):
        config.validate(value)
    s['modifier_exclusions']['ONE'] = slots.required_exclusions(s['idea_modifiers'], config.LINE_MODIFIERS)
    assert {'SETTLE', 'MARKS'} <= set(s['modifier_exclusions']['ONE'])
    config.validate(value)


def test_spec_rows_pin_one_on_arms_and_mark_its_control():
    from arcgen.batch.allocations.make_allocation import build
    def spec(*targets):
        return {'revision': 'r', 'seed': 1, 'windows': {'w': {'account': 'main', 'wave': 'A', 'context': 'c01'}},
                'rows': [['w', 'SEE', None, 'U', 'sol', 'high', t] for t in targets]}
    made = build(spec({'skill': 'ONE', 'size': 'small'}, {'skill': ['ONE', 'EASIER'], 'size': 'R3'},
                      {'slot_control': 'ONE', 'size': 'R2'}), config.SCIENCE['models'], {}, lambda a, i: True)
    assert [c.get('skill') for c in made['calls']] == ['ONE', ['ONE', 'EASIER'], None]
    assert [c.get('slot_control') for c in made['calls']] == [None, None, 'ONE']
    for bad, match in (({'skill': ['ONE', 'MARKS']}, 'no weakness line'), ({'skill': 'ONE', 'structure': 'RECOMBINE'}, 'structure'),
                       ({'skill': 'ONE', 'control': True}, 'control'), ({'skill': 'ONE', 'fence': 'strict'}, 'fence'),
                       ({'skill': 'ONE', 'size': 'R4'}, 'R1 to R3'), ({'slot_control': 'KEY'}, 'not KEY'),
                       ({'skill': 'ONE', 'slot_control': 'ONE'}, 'its own matched control'),
                       ({'slot_control': 'ONE', 'structure_control': True}, 'structure_control'),
                       ({'slot_control': 'ONE', 'skill': 'MARKS'}, 'no weakness line')):
        with pytest.raises(ValueError, match=match):
            build(spec(bad), config.SCIENCE['models'], {}, lambda a, i: True)


def test_placement_keeps_one_off_excluded_arms_and_off_its_controls():
    excl = slots.required_exclusions({'ONE': 1, 'KEY': 1})
    calls = [{'opaque_id': f'p{i:04d}', 'arm': ('SEE', 'SPREAD', 'GEO', 'J')[i % 4]} for i in range(200)]
    for c in calls[:20]:
        c['slot_control'] = 'ONE'
    short = assign_modifiers(calls, {'ONE': 0.2, 'KEY': 0.2}, slots={'episode': ['ONE', 'KEY']}, cap=2, exclusions={'ONE': excl})
    assert short == {}
    placed = [c for c in calls if 'ONE' in c['idea_modifiers']]
    assert len(placed) == 40 and {c['arm'] for c in placed} == {'SEE', 'GEO'}
    assert not any('KEY' in c['idea_modifiers'] for c in placed) and not any(c.get('slot_control') for c in placed)
    with pytest.raises(IntegrityError, match='does not fit'):
        assign_modifiers([{'opaque_id': 'x', 'arm': 'STAGES', 'pinned_modifiers': ['ONE']}], {'ONE': 1.0},
                         exclusions={'ONE': slots.required_exclusions({'ONE': 1})})


def test_admission_gives_one_and_its_control_no_teaching_departure_or_many_and_fixed_presentation(monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'teach_draw', {'plain': 1.0, 'excluded': []})
    monkeypatch.setitem(config.SCIENCE, 'depart_draw', {'test': 1.0, 'excluded': []})
    monkeypatch.setitem(config.SCIENCE, 'pose_rate', 1.0)
    monkeypatch.setitem(config.SCIENCE, 'colour_vary_rate', 1.0)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 1.0)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'kinds', ['MANY', 'XFER'])
    one = admission.configured_routes(None, rows(60, idea_modifiers=['ONE']) + rows(60, 60, slot_control='ONE'))
    for r in one:
        st = r['study']
        assert 'teach' not in st and 'depart' not in st and st.get('overlay') == 'XFER'
        assert st['colour_roles'] == 'v1' and st['pose'] is False
    other = admission.configured_routes(None, rows(60, 200))
    assert all(r['study']['teach'] == 'plain' and r['study']['depart'] == 'test' and r['study']['colour_roles'] == 'vary'
               and r['study']['pose'] for r in other)
    assert {r['study']['overlay'] for r in other} == {'MANY', 'XFER'}


def test_admission_in_a_weakness_run_refuses_lines_and_structures_and_fixes_sizes(weak):
    out = admission.configured_routes(None, rows(6, idea_modifiers=['ONE'], size_arm='small')
                                      + rows(6, 10, idea_modifiers=['ONE'], size_arm='large', size_rung='R2')
                                      + rows(6, 20, slot_control='ONE', size_arm='large', size_rung='R2')
                                      + rows(6, 30, idea_modifiers=['ONE'], size_arm='large'))
    assert [r['study']['rung'] for r in out] == ['R1'] * 6 + ['R2'] * 12 + ['R3'] * 6
    assert not any('weak_overlays' in r['study'] or 'line' in r['study'] or 'structure' in r['study'] for r in out)
    weak['overlays'] = {'EDGE': {'share': 1.0}}
    assert all('weak_overlays' not in r['study'] for r in admission.configured_routes(None, rows(10, idea_modifiers=['ONE'])))
    assert all(r['study']['weak_overlays'] == {'EDGE': 'drawn'} for r in admission.configured_routes(None, rows(10, 50)))
    for bad in ({'idea_modifiers': ['MARKS', 'ONE']}, {'idea_modifiers': ['ONE'], 'structure_call': 'RECOMBINE'},
                {'slot_control': 'ONE', 'structure_control': True}, {'idea_modifiers': ['ONE'], 'line_control': True},
                {'slot_control': 'ONE', 'idea_modifiers': ['MARKS']}):
        with pytest.raises(IntegrityError, match='ONE'):
            admission.configured_routes(None, rows(1, **bad))
    with pytest.raises(IntegrityError, match='outside the levels'):
        admission.configured_routes(None, rows(1, idea_modifiers=['ONE'], size_arm='large', size_rung='R4'))


# ---------------------------------------------------------------- the planned pair count and the shared re-plan


def _one_plan(tid, n):
    return _tplan(tid, [(10, 10)] * n, (12, 12), (12, 12))


def test_a_plan_with_more_than_one_pair_is_planned_once_more_and_resumes(tmp_path):
    from arcgen.batch.runtime import Runtime
    tid = 'dtest0000000000000301'
    record = _record({'idea_modifiers': ['ONE', 'EASIER']})
    three, one = _one_plan(tid, 3), _one_plan(tid, 1)
    fake = _Fake(tmp_path, {f'{tid}-concrete-plan': three, f'{tid}-replan-concrete-plan': one})
    assert Runtime._plan_with_colour(fake, _planner_kw(tid, record), record, 'v1', False) == one
    first, second = fake.calls
    assert ONE_TEXT in first['prompt'] and second['prompt'].startswith(first['prompt'] + '\n\nPair check on an earlier plan')
    assert 'it planned 3 training pairs, where this idea is taught by exactly one' in second['prompt']
    assert second['prompt_template_version'].endswith('+slot-replan-1') and second['request_context']['q2_slots'] == ['EASIER', 'ONE']
    receipt = read(tmp_path / 'slot_check' / f'{tid}.json')
    assert receipt == {'task_id': tid, 'slots': ['ONE'], 'first_pairs': 3, 'first_issues': [
        'it planned 3 training pairs, where this idea is taught by exactly one'], 'replanned': True, 'used': 'second',
        'second_pairs': 1, 'final_pairs': 1, 'final_issues': []}
    assert not (tmp_path / 'rung_check' / f'{tid}.json').exists() and not (tmp_path / 'line_check' / f'{tid}.json').exists()
    again = _Fake(tmp_path, fake.answers)  # resume: the same calls and the same receipt
    assert Runtime._plan_with_colour(again, _planner_kw(tid, record), record, 'v1', False) == one
    assert [c['prompt'] for c in again.calls] == [c['prompt'] for c in fake.calls]
    # one pair: no re-plan; a re-plan that still plans two is used and recorded; an abort keeps the first plan
    tid2, tid3, tid4 = 'dtest0000000000000302', 'dtest0000000000000303', 'dtest0000000000000304'
    fine = _Fake(tmp_path, {f'{tid2}-concrete-plan': _one_plan(tid2, 1)})
    Runtime._plan_with_colour(fine, _planner_kw(tid2, record), record, 'v1', False)
    assert len(fine.calls) == 1 and read(tmp_path / 'slot_check' / f'{tid2}.json')['replanned'] is False
    still = _Fake(tmp_path, {f'{tid3}-concrete-plan': _one_plan(tid3, 2), f'{tid3}-replan-concrete-plan': _one_plan(tid3, 2)})
    Runtime._plan_with_colour(still, _planner_kw(tid3, record), record, 'v1', False)
    assert read(tmp_path / 'slot_check' / f'{tid3}.json')['final_issues'] and len(still.calls) == 2
    aborted = _Fake(tmp_path, {f'{tid4}-concrete-plan': _one_plan(tid4, 2),
                               f'{tid4}-replan-concrete-plan': _tplan(tid4, [], None, None, status='abort')})
    assert Runtime._plan_with_colour(aborted, _planner_kw(tid4, record), record, 'v1', False)['public_contract']
    assert read(tmp_path / 'slot_check' / f'{tid4}.json')['used'] == 'first'


def test_a_rung_and_the_pair_count_share_the_one_replan(tmp_path, monkeypatch):
    from arcgen.batch.runtime import Runtime
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    tid = 'dtest0000000000000305'
    record = _record({'idea_modifiers': ['ONE'], 'rung': 'R2'})
    small, good = _tplan(tid, [(10, 10)] * 2, (12, 12), (12, 12)), _tplan(tid, [(12, 12)], (20, 20), (20, 20))
    fake = _Fake(tmp_path, {f'{tid}-concrete-plan': small, f'{tid}-replan-concrete-plan': good})
    Runtime._plan_with_colour(fake, _planner_kw(tid, record), record, 'v1', False)
    assert len(fake.calls) == 2 and fake.calls[1]['prompt_template_version'].endswith('+rung-replan-1+slot-replan-1')
    assert read(tmp_path / 'rung_check' / f'{tid}.json')['used'] == 'second' and read(tmp_path / 'slot_check' / f'{tid}.json')['final_pairs'] == 1


def test_a_replan_note_needs_a_rung_line_or_checked_slot(tmp_path):
    from arcgen.batch.runtime import Runtime
    tid = 'dtest0000000000000306'
    record = _record({'idea_modifiers': ['EASIER']})
    fake = _Fake(tmp_path, {f'{tid}-concrete-plan': _one_plan(tid, 2)})
    Runtime._plan_with_colour(fake, _planner_kw(tid, record), record, 'v1', False)
    call, = fake.calls  # no checked slot: never re-planned, no receipt
    assert not (tmp_path / 'slot_check' / f'{tid}.json').exists()
    with pytest.raises(Exception, match='re-plan note without a rung'):
        contracts.Bindings().validate_request(call['stage'], call['prompt'] + '\n\nnote', {**call['request_context'], 'q2_replan': 'note'})


# ---------------------------------------------------------------- review policy, repair note and the recorded count


def _rec(study, train=1, **extra):
    task = _task([(_row(1), _row(0, 1))] * train, [(_row(4), _row(0, 4))])
    return {'study': study, 'complete_task': task, **extra}


def test_the_policy_applies_to_one_pair_episodes_only():
    assert slots.review(_rec({'idea_modifiers': ['ONE']})) == (['ONE'], None)
    assert slots.review(_rec({'idea_modifiers': ['ONE']}, train=2)) == ([], 'pairs:2')
    assert slots.review({'study': {'idea_modifiers': ['ONE']}}) == ([], 'no_episode')
    assert slots.review(_rec({'idea_modifiers': ['EASIER']})) == (None, None)
    assert slots.review(_rec({'slot_control': 'ONE'})) == (None, None)  # the control is reviewed as an ordinary task
    assert slots.policy_text(['ONE']) == POLICY and slots.policy_text([]) == ''
    assert slots.repair_notes(_rec({'idea_modifiers': ['ONE']})) == [NOTE] and slots.repair_notes(_rec({'slot_control': 'ONE'})) == []


def test_the_final_pair_count_is_recorded_for_one_and_its_control():
    rec = _rec({'idea_modifiers': ['ONE']}, slot_check={'final_pairs': 1, 'first_pairs': 3, 'replanned': True}, repair_count=1)
    assert slots.fields(rec) == {'one_pair': {'slot': 'ONE', 'control': False, 'train_pairs': 1, 'exactly_one': True,
                                              'planned_pairs': 1, 'first_planned_pairs': 3, 'replanned': True, 'repairs': 1}}
    two = slots.fields(_rec({'idea_modifiers': ['ONE']}, train=2))['one_pair']
    assert two['train_pairs'] == 2 and two['exactly_one'] is False and two['planned_pairs'] is None
    control = slots.fields(_rec({'slot_control': 'ONE'}, train=3))['one_pair']
    assert control['control'] and control['slot'] is None and control['train_pairs'] == 3
    assert slots.fields(_rec({'idea_modifiers': ['EASIER']})) == {} and slots.fields({'study': {'idea_modifiers': ['ONE']}})['one_pair']['train_pairs'] is None
    assert properties.checks(_rec({})['complete_task'])['ONE'] and not properties.checks(_rec({}, train=2)['complete_task'])['ONE']
    rep = properties.report([{'complete_task': _rec({})['complete_task'], 'study': {'primary_arm': 'SEE', 'idea_modifiers': ['ONE']}}])
    assert rep['by_property']['ONE'] == {'verified': 1, 'pass': 1} and rep['format_checks_all_verified']['ONE'] == 1


def test_no_validator_schema_or_diagnostic_refuses_a_one_pair_task():
    from support.fixtures import episode
    from arcgen.lib.validation import validate_arc_task
    from arcgen.lib.astra_authoring_contract_v3 import validate_plan, validate_realization, public_diagnostics, parse_stage
    from arcgen.lib import astra_authoring_schemas_v3 as schemas
    plan, real = episode('dtest0000000000000310', train=1)
    validate_plan(plan); parse_stage(schemas.PLAN_STAGE, plan); validate_realization(plan, real)
    task = {'train': real['task']['train'], 'test': [{'input': p['input'], 'output': o} for p, o in
                                                     zip(real['task']['test'], real['authoring_metadata']['expected_test_outputs'])]}
    assert len(task['train']) == 1 and validate_arc_task(task, require_test_outputs=True).valid
    diagnostics = public_diagnostics(real['task'])
    assert diagnostics['hard_valid'] and not diagnostics['hard_errors']


@pytest.fixture
def one_pair(monkeypatch):
    from support import pilot as pilot_fixture
    original = pilot_fixture.episode
    monkeypatch.setattr(pilot_fixture, 'episode', lambda tid, **kw: original(tid, **{**kw, 'train': 1}))
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)


def _prompts(tmp_path, pattern):
    return [p.read_text() for p in sorted(tmp_path.glob(f'accounts/**/model_calls/{pattern}/*/prompt.md'))]


@pytest.mark.parametrize('case,calls_n,usable', [('clear', 4, True), ('flagged_right', 6, True)])
def test_a_one_pair_task_runs_end_to_end_with_the_policy(tmp_path, monkeypatch, one_pair, case, calls_n, usable):
    first, calls, *_ = run_case(tmp_path, monkeypatch, case, calls_n, usable, arm='SEE', planner='sol', study={'idea_modifiers': ['ONE']})
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert len(record['complete_task']['train']) == 1 and record['slot_check']['final_pairs'] == 1
    assert record['blind_performance']['one_pair']['exactly_one'] is True
    plans = _prompts(tmp_path, 'astra_concrete_plan_v3')
    assert plans and all(ONE_TEXT in p for p in plans)
    adjudications = _prompts(tmp_path, '*adjudicat*')
    if case == 'flagged_right':
        assert adjudications and all(POLICY in t for t in adjudications)
        assert [v.get('slot_review') for v in record['difficulty_versions'].values()] == [['ONE']]
        assert [v.get('slot_policy_version') for v in record['difficulty_versions'].values()] == [slots.POLICY_VERSION]
    else:
        assert not adjudications


def test_a_repair_of_a_one_pair_task_carries_the_note(tmp_path, monkeypatch, one_pair):
    import types
    from support import pilot as pilot_fixture
    from arcgen.batch import repair, repair_v1
    original = pilot_fixture.make_adjudication
    def make(record, tid, calls, case):
        return original(record, tid, calls, 'repair' if case == 'flagged_right' and record['repair_count'] == 0 else case)
    monkeypatch.setattr(pilot_fixture, 'make_adjudication', make)
    monkeypatch.setattr(pilot_fixture, 'repair', types.SimpleNamespace(STAGE=repair_v1.STAGE, REUSE_TESTS=repair.REUSE_TESTS,
                                                                      REUSE_ANSWERS=repair.REUSE_ANSWERS, VERSION=repair.VERSION))
    base = pilot_fixture.limits
    def roomy(account='main', primary=4):
        value = base(account)
        for cats in [value['phases'][0]['categories'], value['accounts'][account]['categories']]:
            cats['conditional_evidence_repair'] = 12
        value['logical_cap'] = value['phases'][0]['logical_cap'] = value['phases'][0]['astra_logical_cap'] = \
            value['accounts'][account]['logical_cap'] = 30
        return value
    monkeypatch.setattr(pilot_fixture, 'limits', roomy)
    first, *_ = run_case(tmp_path, monkeypatch, 'flagged_right', 11, True, arm='SEE', planner='sol', study={'idea_modifiers': ['ONE']})
    repairs = _prompts(tmp_path, '*repair*')
    assert repairs and all(NOTE in t for t in repairs)
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert len(record['complete_task']['train']) == 1 and record['blind_performance']['one_pair']['repairs'] == 1


# ---------------------------------------------------------------- RECOMBINE after the screen's opt-outs


def test_the_loosened_recombine_text_keeps_the_core_and_the_opt_out():
    text = (lines.LINE_DIR / 'RECOMBINE.planner.md').read_text().split('-->', 1)[1]
    for phrase in ('Laws: two, three at most.', 'They may be the kernel\'s own parts', 'add no unrelated law',
                   'each law acts in at least two training pairs', 'at least one pair shows each law at work without the other',
                   'other pairs may show both at work on separate things that never meet',
                   'the test needs them together in a combination no pair shows', '(order-free)', '(in turn)',
                   'Never let the answer depend on a priority between the laws that no pair shows',
                   'at most one and a half times the area', '"Structure: none — <why>"', 'where the two conflict, the line above holds'):
        assert phrase in text, phrase
    assert 'The combination is settled because the laws commute' not in text  # the binding cause of 19 of 21 opt-outs
    for template in ('Combination: recolour and shift each act apart in the pairs; the test combines them, order-free.',
                     'Combination: falling and tipping each act apart in the pairs; the test combines them in turn: along the fall.'):
        assert not pair_specific_test_narrative(template) and 'first test' not in template, template
    realizer = (lines.LINE_DIR / 'RECOMBINE.realizer.md').read_text()
    assert 'when the plan\'s "Combination:" invariant says order-free' in realizer and 'leave them out when the laws act in turn' in realizer
    policy = lines.body('lines/RECOMBINE.adjudicator.md')
    assert 'Where the laws act in turn' in policy and 'where the plan calls the combination order-free' in policy
    assert lines.POLICY_VERSION == 'weakness-v1-policies-4'


def test_the_recombine_form_is_declared_and_steers_the_order_check():
    form = lambda inv: lines.form('RECOMBINE', lines.declared('RECOMBINE', _lplan(invariants=[inv])))['label']
    assert form('Combination: a and b each act apart in the pairs; the test combines them in turn: along the course.') == 'in turn'
    assert form('Combination: a and b each act apart in the pairs; the test combines them, order-free.') == 'order-free'
    assert form('Combination: a and b each act apart in the pairs and commute; the test combines them.') == 'order-free'  # screen form
    assert form('Combination: a and b.') is None
    check = lines.plan_check(None, None, _lplan(invariants=['Combination: a and b each act apart in the pairs; the test combines them '
                                                            'in turn: outside in.'], rules=['Single law: a.', 'Single law: b.',
                                                                                             'Replay: c.']), structure='RECOMBINE')
    assert check['structure_form']['label'] == 'in turn' and check['final_issues'] == []


def test_laws_acting_in_turn_need_no_order_records():
    from test_structure_lines import T, OUT, TRUTH, _two, _alt, _srec
    rec = lambda alts, inv: _srec(T, {'structure': 'RECOMBINE'}, alts,
                                  plan=_lplan(invariants=[inv], rules=['Single law: a.', 'Single law: b.', 'Replay: c.']))
    check = lambda alts, inv: lines.episode_check(rec(alts, inv), T)['structure']
    law1 = _alt('l1', 'single law', [OUT[0], _two(OUT[1]), _two(OUT[2])], [_two(TRUTH)], line='RECOMBINE')
    law2 = _alt('l2', 'single law', [_two(OUT[0]), OUT[1], OUT[2]], [[[5] * 7]], line='RECOMBINE')
    replay = _alt('p1', 'replay', [], [[[6] * 7]], line='RECOMBINE')
    orders = [_alt(o, 'order', [], [TRUTH], line='RECOMBINE') for o in ('o1', 'o2')]
    in_turn = 'Combination: a and b each act apart in the pairs; the test combines them in turn: along the course.'
    order_free = 'Combination: a and b each act apart in the pairs; the test combines them, order-free.'
    turn = check([law1, law2, replay], in_turn)
    assert turn['status'] == 'pass' and turn['alternatives']['form'] == 'in turn' and turn['alternatives']['orders']['required'] is False
    free = check([law1, law2, replay], order_free)
    assert free['status'] == 'incomplete' and free['alternatives']['orders']['required'] is True  # order-free: both orders owed
    assert check([*orders, law1, law2, replay], order_free)['status'] == 'pass'
    wrong = [orders[0], _alt('o2', 'order', [], [_two(TRUTH)], line='RECOMBINE')]
    assert check([*wrong, law1, law2, replay], order_free)['status'] == 'fail'
    assert check([*wrong, law1, law2, replay], in_turn)['status'] == 'pass'  # recorded, not judged, when the laws act in turn
    # the screen's form ("and commute") is order-free, as before
    assert check([law1, law2, replay], 'Combination: a and b each act apart in the pairs and commute; the test combines them.')['status'] == 'incomplete'
    # a mixed pair (both laws at work apart) is allowed and recorded; each law still reproduces its own pair
    mixed = _alt('l2', 'single law', [_two(OUT[0]), OUT[1], _two(OUT[2])], [[[5] * 7]], line='RECOMBINE')
    value = check([law1, mixed, replay], in_turn)
    assert value['status'] == 'pass' and value['alternatives']['laws']['pairs_no_single_law'] == [3]
    alone = _alt('l2', 'single law', [_two(OUT[0]), _two(OUT[1]), _two(OUT[2])], [[[5] * 7]], line='RECOMBINE')
    assert check([law1, alone, replay], in_turn)['status'] == 'fail'  # a law never seen alone
