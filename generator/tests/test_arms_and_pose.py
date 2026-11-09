"""Reference design: every arm has an idea card and a planner bridge, drawn slot values reach the idea prompt, and the
per-idea planner pose instruction follows its draw."""
import pytest
from arcgen.batch import config, idea_v1, supply, contracts
from arcgen.batch.allocation import design
from support.pilot import run_case


def test_reference_design_arms_have_cards_bridges_and_slot_texts():
    calls = [c for w in design()['windows'] for c in w['calls']]
    arms = {c['arm'] for c in calls}
    assert len(calls) == 56 and arms <= set(idea_v1.ARMS)
    b = contracts.Bindings()
    for arm in sorted(arms):
        assert contracts.known_primary(arm)
        assert b.planning_templates(arm, ())
        call = next(c for c in calls if c['arm'] == arm)
        prompt = supply.idea_request(call, None, None)['prompt']
        assert 'Direction for this call' in prompt  # every arm has its card
    mods = [m for c in calls for m in c['idea_modifiers']]
    rates = config.SCIENCE['idea_modifiers']
    assert all(mods.count(m) == round(r * 56) for m, r in rates.items())  # exact shares
    with_mod = next(c for c in calls if c['idea_modifiers'])
    without = next(c for c in calls if not c['idea_modifiers'])
    assert 'Additional requirement for this call' in supply.idea_request(with_mod, None, None)['prompt']
    assert 'Additional requirement for this call' not in supply.idea_request(without, None, None)['prompt']


@pytest.mark.parametrize('rate,expect', [(1.0, True), (0.0, False)])
def test_pose_instruction_follows_the_per_idea_draw(tmp_path, monkeypatch, rate, expect):
    monkeypatch.setitem(config.SCIENCE, 'pose_rate', rate)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    first, *_ = run_case(tmp_path, monkeypatch, 'clear', 4, True, arm='H_DISC', planner='sol')
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert record['study']['pose'] is expect
    prompt, = list(tmp_path.glob('accounts/**/model_calls/astra_concrete_plan_v3/*/prompt.md'))
    assert ('Pose instruction for this idea' in prompt.read_text()) is expect
