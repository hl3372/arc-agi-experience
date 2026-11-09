"""Gate study (2026-10-15): ideas drawn into the right-twice gate
clear a flagged episode without adjudication or repair when both blind calls give the intended answer."""
import copy, json, pathlib, sys
import pytest
from arcgen.batch import admission, config, gate
from support.pilot import run_case


def rows(n, **study):
    return [{'source_candidate_id': f'k{i:04d}', 'private_source': 'ordinary_unconditioned',
             'study': {'primary_arm': 'SEE', **study}} for i in range(n)]


def test_gate_draw_is_seeded_per_idea(monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'gate_draw', {'right_twice': 0.7})
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    out = admission.configured_routes(None, rows(3000))
    share = sum(r['study'].get('gate') == 'right_twice' for r in out) / len(out)
    assert abs(share - 0.7) < 0.03
    assert admission.configured_routes(None, rows(3000)) == out
    assert {r['study'].get('gate') for r in out} == {'right_twice', 'control'}  # the control keeps the old gate


def test_gate_draw_validation():
    base = json.loads((config.PROFILES / 'v1.json').read_text())
    for good in ({'right_twice': 0.7}, {'right_twice': 1}, {'right_twice': 0.0}):
        value = copy.deepcopy(base); value['science']['gate_draw'] = good
        config.validate(value)
    for bad in ({'right_twice': 1.2}, {'right_twice': '0.7'}, {'paired': 0.7}, {}, [0.7]):
        value = copy.deepcopy(base); value['science']['gate_draw'] = bad
        with pytest.raises(config.ConfigError):
            config.validate(value)


@pytest.fixture
def plain_plan(monkeypatch):
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)  # the fixture plans without overlays


def test_flagged_but_right_twice_clears_without_adjudication(tmp_path, monkeypatch, plain_plan):
    # plan, realization, technical, first blind, second blind: no adjudication and no repair
    first, calls, *_ = run_case(tmp_path, monkeypatch, 'flagged_right', 5, True, arm='SEE', planner='sol',
                                study={'gate': 'right_twice'})
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert record['answer_evidence']['difficulty_outcome']['reason'] == 'right_twice'
    assert record['answer_evidence']['status'] == 'evidence_clear_after_review'
    assert record.get('repair_count', 0) == 0 and not record.get('ambiguity_adjudication')
    assert all(r['response']['serious_different_output_alternative'] for r in record['blind_results'][:2])
    perf = record['blind_performance']
    assert perf['gate_arm'] == 'right_twice' and perf['gate_resolution'] == 'right_twice_clear'
    assert perf['right_twice_eligible'] and perf['contested']  # the second call flagged the task too


def test_the_old_gate_still_adjudicates_a_flagged_right_answer(tmp_path, monkeypatch, plain_plan):
    # the control: the same flagged, right answers go to the adjudicator, as before
    first, calls, *_ = run_case(tmp_path, monkeypatch, 'flagged_right', 6, True, arm='SEE', planner='sol')
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert record['answer_evidence']['difficulty_outcome']['reason'] != 'right_twice'
    assert record.get('ambiguity_adjudication')
    assert 'gate_arm' not in record['blind_performance']  # runs without a gate draw keep their old fields


def test_the_control_arm_records_what_right_twice_would_have_done(tmp_path, monkeypatch, plain_plan):
    first, calls, *_ = run_case(tmp_path, monkeypatch, 'flagged_right', 6, True, arm='SEE', planner='sol',
                                study={'gate': 'control'})
    perf = next(iter(first['runtime'].state['semantic_records'].values()))['blind_performance']
    assert perf['gate_arm'] == 'control' and perf['right_twice_eligible'] and not perf['contested']
    assert perf['gate_resolution'] != 'right_twice_clear' and perf['rival_test_outputs'] == []


def test_a_flagged_miss_is_never_cleared_by_the_draw(tmp_path, monkeypatch, plain_plan):
    # first blind answer wrong and flagged: the right-twice rule does not apply, the adjudicator decides
    first, calls, *_ = run_case(tmp_path, monkeypatch, 'flagged_rival', 6, False, arm='SEE', planner='sol',
                                study={'gate': 'right_twice'})
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert record['answer_evidence']['difficulty_outcome']['reason'] != 'right_twice'


def _fake(monkeypatch, responses, test_input=((0, 0), (0, 5)), answer=((0, 0), (3, 5))):
    record = {'study': {'gate': 'right_twice'},
              'complete_task': {'test': [{'input': [list(r) for r in test_input], 'output': [list(r) for r in answer]}]}}
    monkeypatch.setattr(gate, 'current_blind', lambda tid, rec, calls: [{'response': r} for r in responses])
    return record


def _blind(outputs, serious=True, guessing=True, rivals=()):
    return {'predicted_test_outputs': outputs, 'serious_different_output_alternative': serious,
            'test_requires_unsupported_guessing': guessing,
            'strongest_plausible_alternatives': [{'predicted_test_outputs': r} for r in rivals]}


def test_right_twice_exclusions(monkeypatch):
    right = [[[0, 0], [3, 5]]]
    elsewhere = [[[3, 0], [0, 5]]]  # a rival that changes another cell: a different rule
    repainted = [[[0, 0], [4, 5]]]  # the same cell in another colour: a colour choice
    assert gate.right_twice('t', _fake(monkeypatch, [_blind(right, rivals=[elsewhere]), _blind(right)]), [])
    assert not gate.right_twice('t', _fake(monkeypatch, [_blind(right, rivals=[repainted]), _blind(right)]), [])
    assert not gate.right_twice('t', _fake(monkeypatch, [_blind(right), _blind(right, rivals=[repainted])]), [])
    assert not gate.right_twice('t', _fake(monkeypatch, [_blind(right, serious=False), _blind(right)]), [])  # guessing only
    assert not gate.right_twice('t', _fake(monkeypatch, [_blind(right), _blind(elsewhere)]), [])  # a split
    assert not gate.right_twice('t', _fake(monkeypatch, [_blind(elsewhere), _blind(right)]), [])  # a flagged miss
    assert not gate.right_twice('t', _fake(monkeypatch, [_blind(right)]), [])  # one call only
    record = _fake(monkeypatch, [_blind(right), _blind(right)]); record['study']['gate'] = 'control'
    assert not gate.right_twice('t', record, [])


def test_gate_fields_on_a_development_without_an_episode():
    """v1-b17-t1 stopped on 2026-10-16 03:57: the final-evidence step read the episode of a development that had none."""
    record = {'study': {'gate': 'right_twice'}, 'answer_evidence': {}}
    assert gate.right_twice_evidence('t', record, []) is None and gate.right_twice('t', record, []) is False
    fields = gate._gate_study_fields('t', record, [], {})
    assert fields['gate_arm'] == 'right_twice' and fields['right_twice_eligible'] is False and fields['rival_test_outputs'] == []
    from arcgen.lib import factory_evidence
    assert factory_evidence._right_twice_clear(record, {'resolution': 'right_twice_clear'}, []) is False


@pytest.mark.parametrize('arm_value', ['right_twice', 'control'])
def test_a_refused_realization_in_a_gate_run_does_not_stop_the_window(tmp_path, monkeypatch, plain_plan, arm_value):
    # the realizer refuses, so the record never gets an episode; the window must still finish
    first, calls, *_ = run_case(tmp_path, monkeypatch, 'refusal', 2, False, arm='SEE', planner='sol', study={'gate': arm_value})
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert not record.get('complete_task')
