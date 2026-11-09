"""The teaching line (b17 review, 2026-10-15): blinded raters scored T-type tasks 2.84
(eval 3.41), flagging stacked conventions taught by two pairs, bookkeeping-heavy large tests and undetermined details.
A seeded share of ideas gets prompts/v1/teach.plain.planner.md after the scale and canvas lines; the rest are the control."""
import copy, json, pathlib, sys
import pytest
from arcgen.batch import admission, config, contracts
from support.pilot import run_case

FLOORS = {'R1': {'output_cells': 150, 'computed_cells': 30, 'escalation': 0},
          'R2': {'output_cells': 300, 'computed_cells': 80, 'escalation': 1.3},
          'R3': {'output_cells': 450, 'computed_cells': 150, 'escalation': 1.3}}
TEXT = (pathlib.Path(__file__).resolve().parents[1] / 'arcgen/batch/prompts/v1/teach.plain.planner.md').read_text()


def rows(n, **study):
    return [{'source_candidate_id': f'k{i:04d}', 'private_source': 'ordinary_unconditioned',
             'study': {'primary_arm': 'SEE', **study}} for i in range(n)]


def test_teach_draw_is_seeded_per_idea(monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'teach_draw', {'plain': 0.5})
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    out = admission.configured_routes(None, rows(3000, family='SIM', rung='R2'))
    share = sum(r['study']['teach'] == 'plain' for r in out) / len(out)
    assert abs(share - 0.5) < 0.03 and {r['study']['teach'] for r in out} == {'plain', 'control'}
    assert admission.configured_routes(None, rows(3000, family='SIM', rung='R2')) == out
    # arms and slots that fix their own pair structure take no draw
    monkeypatch.setitem(config.SCIENCE, 'teach_draw', {'plain': 1.0, 'excluded': ['J', 'STAGES', 'CHAINED']})
    excluded = (admission.configured_routes(None, rows(50, family='SIM', rung='R2', primary_arm='STAGES'))
                + admission.configured_routes(None, rows(50, family='SIM', rung='R2', idea_modifiers=['CHAINED'])))
    assert not any('teach' in r['study'] for r in excluded)


def test_teach_draw_validation():
    base = json.loads((config.PROFILES / 'v1.json').read_text())
    for good in ({'plain': 0.5}, {'plain': 1}, {'plain': 0.0}, {'plain': 0.5, 'excluded': ['J', 'CHAINED']}):
        value = copy.deepcopy(base); value['science']['teach_draw'] = good
        config.validate(value)
    for bad in ({'plain': 1.5}, {'plain': '0.5'}, {'plain': True}, {'clear': 0.5}, {}, [0.5], {'plain': 0.5, 'excluded': 'J'}):
        value = copy.deepcopy(base); value['science']['teach_draw'] = bad
        with pytest.raises(config.ConfigError):
            config.validate(value)


def test_teach_line_is_last_and_names_its_four_parts():
    b = contracts.Bindings()
    texts = [r[0] for r in b.planning_templates('SEE', (), family='FIT', rung='R3', canvas='square', teach='plain')]
    assert 'Teaching for this idea (binding' in texts[-1] and 'Canvas for this idea (binding' in texts[-2]
    assert any('Scale for this idea' in t for t in texts[:-2])
    assert not any('Teaching for this idea' in r[0] for r in b.planning_templates('SEE', (), family='FIT', rung='R3', teach='control'))
    with pytest.raises(Exception):
        b.planning_templates('SEE', (), family='FIT', rung='R3', teach='strict')
    for phrase in ('keeps their floors, limits and the one boundary case or chain the scale line requires',
                   'at most one convention the picture gives no reason for', 'acts in at least two training pairs',
                   'Use the fewest pairs that achieve this: two when each pair shows every law',
                   'never a lone cell among clutter or noise', 'counted over all tests, under twice its floor',
                   'one or two links longer than the longest training chain', 'never by trying many placements'):
        assert phrase in TEXT, phrase


@pytest.mark.parametrize('teach,expect', [('plain', True), ('control', False)])
def test_teach_line_reaches_the_planner_prompt(tmp_path, monkeypatch, teach, expect):
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    run_case(tmp_path, monkeypatch, 'clear', 5, True, arm='SEE', planner='sol',
             study={'family': 'ENCL', 'rung': 'R1', 'teach': teach}, primary=5)
    plans = sorted(tmp_path.glob('accounts/**/model_calls/astra_concrete_plan_v3/*/prompt.md'))
    assert plans and all(('Teaching for this idea (binding' in p.read_text()) is expect for p in plans)


def test_three_pairs_choice_gives_round_ones_sentence(monkeypatch):
    """Round 3 (2026-10-20): teach_draw.pairs 'three' gives round 1's pair count; the rest of the line is unchanged."""
    three = (pathlib.Path(__file__).resolve().parents[1] / 'arcgen/batch/prompts/v1/teach.three.planner.md').read_text()
    a = ('Use the fewest pairs that achieve this: two when each pair shows every law, the convention and each kind of case; '
         'three or four only when two cannot. Add no pair just to reach a count.')
    b = 'Use three pairs; two only when each shows all of these, four only when three cannot.'
    assert three == TEXT.replace(a, b).replace('b17-teach-plain-3', 'b17-teach-plain-2')
    assert contracts.TEACH_LINE == 'teach.plain.planner.md'  # the default stays round 2's line
    monkeypatch.setattr(contracts, 'TEACH_LINE', 'teach.three.planner.md')
    texts = [r[0] for r in contracts.Bindings().planning_templates('SEE', (), family='FIT', rung='R1', teach='plain')]
    assert b in texts[-1] and a not in texts[-1]
    base = json.loads((config.PROFILES / 'v1.json').read_text())
    for good in ({'plain': 0.5, 'pairs': 'three'}, {'plain': 0.5, 'pairs': 'fewest', 'excluded': ['J']}):
        value = copy.deepcopy(base); value['science']['teach_draw'] = good
        config.validate(value)
    for bad in ({'plain': 0.5, 'pairs': 'two'}, {'pairs': 'three'}, {'plain': 0.5, 'pairs': 3}):
        value = copy.deepcopy(base); value['science']['teach_draw'] = bad
        with pytest.raises(config.ConfigError):
            config.validate(value)
