"""Departure package (2026-10-16): for a seeded share of ideas the first test meets one visible situation no training pair shows."""
import copy, json, pathlib, sys
import pytest
from arcgen.batch import admission, config, contracts, gate, idea_v1
from support.pilot import run_case

V1 = pathlib.Path(__file__).resolve().parents[1] / 'arcgen/batch/prompts/v1'
EXCLUDED = ['R2', 'J', 'T', 'STAGES', 'CHAINED', 'ACROSS', 'DERIVED', 'logical_leap', 'episode_joint_inference']


def rows(n, **study):
    return [{'source_candidate_id': f'k{i:04d}', 'private_source': 'ordinary_unconditioned',
             'study': {'primary_arm': 'SEE', **study}} for i in range(n)]


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)


def test_depart_draw_is_seeded_and_skips_excluded_ideas(monkeypatch, plain):
    monkeypatch.setitem(config.SCIENCE, 'depart_draw', {'test': 0.5, 'excluded': EXCLUDED})
    out = admission.configured_routes(None, rows(3000, family='SIM', rung='R3'))
    share = sum(r['study']['depart'] == 'test' for r in out) / len(out)
    assert abs(share - 0.5) < 0.03 and {r['study']['depart'] for r in out} == {'test', 'control'}
    assert admission.configured_routes(None, rows(3000, family='SIM', rung='R3')) == out
    for excluded in (rows(50, family='SIM', rung='R2'), rows(50, primary_arm='ACROSS'), rows(50, idea_modifiers=['CHAINED'])):
        assert not any('depart' in r['study'] for r in admission.configured_routes(None, excluded))


def test_depart_draw_validation():
    base = json.loads((config.PROFILES / 'v1.json').read_text())
    for good in ({'test': 0.5}, {'test': 0.5, 'excluded': EXCLUDED}):
        value = copy.deepcopy(base); value['science']['depart_draw'] = good
        config.validate(value)
    for bad in ({'test': 1.5}, {'test': True}, {'depart': 0.5}, {}, {'test': 0.5, 'excluded': 'R2'}):
        value = copy.deepcopy(base); value['science']['depart_draw'] = bad
        with pytest.raises(config.ConfigError):
            config.validate(value)


def test_departure_line_is_the_last_binding_line():
    b = contracts.Bindings()
    texts = [r[0] for r in b.planning_templates('SEE', (), family='SIM', rung='R3', canvas='square', teach='plain', depart='test')]
    assert 'Departure for this idea (binding' in texts[-1]
    assert 'Teaching for this idea' in texts[-2] and 'Canvas for this idea' in texts[-3]
    assert not any('Departure for this idea' in r[0] for r in b.planning_templates('SEE', (), family='SIM', rung='R3', depart='control'))
    with pytest.raises(Exception):
        b.planning_templates('SEE', (), family='SIM', rung='R3', depart='always')


def test_departure_texts_carry_the_critique():
    planner = (V1 / 'depart.planner.md').read_text()
    for phrase in ('a turned frame', 'replaying the closest training pair as it stands', 'a lower-level description',
                   'what counts as one thing becomes unclear', 'Keep every training pair clear of it and of its kind',
                   'as a requirement on the laws', 'No training pair shows', '"Departure:"', '"No departure:"',
                   'any invitation below'):
        assert phrase in planner, phrase
    policy = (V1 / 'depart.adjudicator.md').read_text()
    for phrase in ('reproduces every training output cell for cell', 'this policy does not apply', 'less_supported',
                   'comparable_rival', 'or when you are unsure', 'not a judgement of familiarity', 'decide serious_ambiguity'):
        assert phrase in policy, phrase
    assert 'keep that invariant true' in (V1 / 'depart.repair_note.md').read_text()
    assert (V1 / 'difficulty_adjudicator.md').read_text().rstrip().endswith('{{REVIEW_POLICY}}')


def _record(depart='test', invariant=True, episode=True):
    plan = {'public_contract': {'invariants': ['Coordinates are zero-indexed.'] + (
                ['No training pair shows a key on the right edge; the first test does, and every law its answer uses there '
                 'acts in the pairs.'] if invariant else [])},
            'private_execution': {'test_outputs': [{'test_only_semantics_audit': 'author argument',
                                                    'why_one_output_is_best_supported': 'author argument'}],
                                  'cross_pair_concrete_audit': {'relevant_and_incidental_variation': 'x',
                                                                'remaining_ambiguity_risk': 'author argument',
                                                                'strongest_simple_shortcuts': [{'rule': 'r', 'evidence_that_defeats_it': 'author argument'}]}}}
    record = {'study': {'depart': depart} if depart else {}, 'concrete_plan': plan,
              'realization': {'authoring_metadata': {'intended_rule': 'rule'}},
              'technical_results': [{'response': {'status': 'pass'}}]}
    if episode:
        record['complete_task'] = {'train': [{'input': [[0]], 'output': [[1]]}], 'test': [{'input': [[0]], 'output': [[1]]}]}
    return record


class _Runtime:
    def __init__(self, record):
        self.state = {'semantic_records': {'t': record}, 'call_ledger': {}}


@pytest.mark.parametrize('kwargs,answers,expect', [
    ({'depart': None}, [[[1]]], None),
    ({'invariant': False}, [[[1]]], 'no_invariant'),
    ({'episode': False}, [[[1]]], 'no_episode'),
    ({}, [[[2]]], 'no_right_answer'),
    ({}, [[[2]], [[1]]], 'policy'),  # a split: one call right is enough for the policy
])
def test_departure_review_conditions(monkeypatch, kwargs, answers, expect):
    record = _record(**kwargs)
    monkeypatch.setattr(gate, 'current_blind', lambda tid, rec, calls: [{'response': {'predicted_test_outputs': [a]}} for a in answers])
    assert gate.departure_review('t', record, []) == expect


@pytest.mark.parametrize('kwargs,answers,policy,withheld', [  # arguments are withheld only where the policy applies
    ({}, [[[1]]], True, True), ({}, [[[2]]], False, False), ({'invariant': False}, [[[1]]], False, False),
    ({'depart': 'control'}, [[[1]]], False, False)])
def test_adjudication_values_carry_the_policy_and_withhold_arguments(monkeypatch, kwargs, answers, policy, withheld):
    record = _record(**kwargs)
    monkeypatch.setattr(gate, 'current_blind', lambda tid, rec, calls: [{'response': {'predicted_test_outputs': [a]}} for a in answers])
    monkeypatch.setattr(gate, 'technical_bound', lambda tid, rec, calls: True)
    monkeypatch.setattr(gate.adjudication, 'concerns', lambda blind: [])
    values = gate._adjudication_values(_Runtime(record), 't')
    assert (values['REVIEW_POLICY'] == gate.DEPART_POLICY) is policy and (values['REVIEW_POLICY'] == '') is (not policy)
    plan = values['AUTHOR_SPECIFICATION']['concrete_plan']['private_execution']
    texts = [plan['test_outputs'][0]['test_only_semantics_audit'], plan['cross_pair_concrete_audit']['remaining_ambiguity_risk'],
             plan['cross_pair_concrete_audit']['strongest_simple_shortcuts'][0]['evidence_that_defeats_it']]
    assert all((t == gate.WITHHELD) is withheld for t in texts)
    assert record['concrete_plan']['private_execution']['test_outputs'][0]['test_only_semantics_audit'] == 'author argument'


def test_family_idea_requirements_name_a_departure_only_in_departure_runs(monkeypatch):
    plain_text = idea_v1._base('SEE', (), 'SIM', 'R1')[2]
    assert 'name one visible situation it could meet that no training pair shows' not in plain_text
    monkeypatch.setitem(config.SCIENCE, 'depart_draw', {'test': 0.5})
    depart_text = idea_v1._base('SEE', (), 'SIM', 'R1')[2]
    assert 'name one visible situation it could meet that no training pair shows' in depart_text
    assert 'name one visible situation' not in idea_v1._base('SEE', ())[2]  # untargeted calls render as before


@pytest.mark.parametrize('case,calls_n,usable', [('flagged_right', 6, True), ('flagged_rival', 6, False), ('clear', 4, True)])
def test_a_drawn_idea_runs_end_to_end(tmp_path, monkeypatch, plain, case, calls_n, usable):
    # the fixture's plans carry no departure invariant, so the policy never applies; the planner prompt still carries the line
    first, calls, *_ = run_case(tmp_path, monkeypatch, case, calls_n, usable, arm='SEE', planner='sol', study={'depart': 'test'})
    plans = sorted(tmp_path.glob('accounts/**/model_calls/astra_concrete_plan_v3/*/prompt.md'))
    assert plans and all('Departure for this idea (binding' in p.read_text() for p in plans)
    adjudications = sorted(tmp_path.glob('accounts/**/model_calls/*adjudicat*/*/prompt.md'))
    assert not any('Review policy for this episode' in p.read_text() for p in adjudications)
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert all(v.get('departure_review') in (None, 'no_invariant') for v in record.get('difficulty_versions', {}).values())


def test_the_dictated_invariant_passes_the_plan_validator():
    """Code review 2026-10-16: "the first test does" made validate_plan reject every drawn plan that followed the line."""
    import re
    from arcgen.lib.astra_authoring_contract_v3 import pair_specific_test_narrative
    template = re.search(r'"(No training pair shows <[^>]+>; [^"]+)"', (V1 / 'depart.planner.md').read_text())[1]
    filled = re.sub(r'<[^>]+>', 'a key on the right edge', template)
    assert not pair_specific_test_narrative(filled) and 'first test' not in filled
    for opening in ('Departure: the key sits on the right edge, a side no training pair shows; the closest pair is train_2.',
                    'No departure: no situation of these kinds could be settled by the pairs.'):
        assert not pair_specific_test_narrative(opening)


def test_departure_review_on_aborted_and_partial_plans(monkeypatch):
    monkeypatch.setattr(gate, 'current_blind', lambda tid, rec, calls: [])
    record = _record()
    record['concrete_plan'] = {'status': 'abort', 'public_contract': None, 'private_execution': None}
    assert gate.departure_review('t', record, []) == 'no_invariant' and gate.departure_invariant(record) is False
    record['concrete_plan'] = None
    assert gate.departure_review('t', record, []) == 'no_invariant'
    record = _record(episode=False); record['concrete_plan'] = None
    assert gate.departure_review('t', record, []) == 'no_episode'


INV = 'No training pair shows a key on the right edge; the test does, and every law its answer uses there acts in the pairs.'


@pytest.fixture
def depart_plans(monkeypatch, plain):
    from support import pilot as pilot_fixture
    original = pilot_fixture.episode
    def episode(tid, **kw):
        plan, realized = original(tid, **kw)
        plan['public_contract']['invariants'] = list(plan['public_contract']['invariants']) + [INV]
        if realized.get('authoring_metadata'):
            realized['authoring_metadata']['plan_fidelity']['preserved_frozen_decisions'] = list(plan['public_contract']['invariants'])
        return plan, realized
    monkeypatch.setattr(pilot_fixture, 'episode', episode)


def _prompts(tmp_path, pattern):
    return [p.read_text() for p in sorted(tmp_path.glob(f'accounts/**/model_calls/{pattern}/*/prompt.md'))]


@pytest.mark.parametrize('crash', [None, 'response_saved', 'adjudication_saved', 'outcome_saved'])
def test_policy_applies_end_to_end_and_resumes(tmp_path, monkeypatch, depart_plans, crash):
    # code review 2026-10-16: a flagged, right answer on a drawn plan with the invariant reaches the adjudicator with the
    # policy (the gate draw is off here), and every crash point resumes to the same record
    first, calls, *_ = run_case(tmp_path, monkeypatch, 'flagged_right', 6, True, crash_point=crash, arm='SEE', planner='sol',
                                study={'depart': 'test'})
    adjudications = _prompts(tmp_path, '*adjudicat*')
    assert adjudications and all('Review policy for this episode' in t for t in adjudications)
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert [v.get('departure_review') for v in record['difficulty_versions'].values()] == ['policy']
    assert [v.get('departure_policy_version') for v in record['difficulty_versions'].values()] == [gate.DEPART_POLICY_VERSION]


@pytest.mark.parametrize('crash', [None, 'repair_response_saved', 'expansion_saved', 'repair_stored', 'technical_recheck_saved'])
def test_a_repair_of_a_drawn_plan_carries_the_note_and_resumes(tmp_path, monkeypatch, depart_plans, crash):
    import types
    from support import pilot as pilot_fixture
    from arcgen.batch import repair, repair_v1
    original = pilot_fixture.make_adjudication
    def make(record, tid, calls, case):
        if case == 'flagged_right' and record['repair_count'] == 0:
            return original(record, tid, calls, 'repair')
        return original(record, tid, calls, case)
    monkeypatch.setattr(pilot_fixture, 'make_adjudication', make)
    shim = types.SimpleNamespace(STAGE=repair_v1.STAGE, REUSE_TESTS=repair.REUSE_TESTS, REUSE_ANSWERS=repair.REUSE_ANSWERS,
                                 VERSION=repair.VERSION)
    monkeypatch.setattr(pilot_fixture, 'repair', shim)
    base = pilot_fixture.limits
    def roomy(account='main', primary=4):
        value = base(account)
        for cats in [value['phases'][0]['categories'], value['accounts'][account]['categories']]:
            cats['conditional_evidence_repair'] = 12
        value['logical_cap'] = value['phases'][0]['logical_cap'] = value['phases'][0]['astra_logical_cap'] = \
            value['accounts'][account]['logical_cap'] = 30
        return value
    monkeypatch.setattr(pilot_fixture, 'limits', roomy)
    run_case(tmp_path, monkeypatch, 'flagged_right', 11, True, crash_point=crash, arm='SEE', planner='sol', study={'depart': 'test'})
    repairs = _prompts(tmp_path, '*repair*')
    assert repairs and all(gate.DEPART_REPAIR_NOTE in t for t in repairs)
