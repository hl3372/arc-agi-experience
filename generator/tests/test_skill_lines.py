"""Skill lines and skill-scoped novelty (2026-10-18): every skill has
an idea text and a planner line, a skill call in a skill run uses the skill variants of explore and step 3, untargeted and
family calls are unchanged, and the skill selector is the targeted text plus exactly one passage."""
import json
import re
import pytest
from arcgen.batch import config, contracts, idea_v1, supply
from arcgen.batch.common import read
from test_targeted import _call

V1 = idea_v1.PROMPTS.parent
SKETCHES = [{'id': 's01', 'title': 'T', 'idea': 'I.', 'recipe': 'R.'}]
NEAR = [[{'reference_id': 'committed:x', 'similarity': 0.5, 'text': 'A task.'}]]
SKILLS_1018 = ('QUANTITY', 'SYMMETRY', 'MATCH', 'CUED', 'EMIT', 'GROWTH', 'NOISY')


def test_every_skill_has_an_idea_text_and_a_planner_line_without_task_ids():
    assert set(config.SKILL_MODIFIERS) <= set(config.MODIFIER_NAMES) <= set(idea_v1.MODIFIERS)
    for m in config.SKILL_MODIFIERS:
        idea, planner = idea_v1.PROMPTS / 'modifiers' / f'{m}.md', contracts.SLOT_DIR / f'{m}.planner.md'
        assert idea.exists() and planner.exists() and m in contracts.SLOT_BRIDGES
        for p in (idea, planner):
            head, body = p.read_text().split('-->', 1)
            # the blind-spot lines (2026-10-18) and the composite batch's new territory (2026-10-25)
            assert ('blindspot-v1' in head if m in SKILLS_1018 else 'composite-c1' in head) and body.strip()
            assert not re.search(r'\b[0-9a-f]{8}\b', body)  # no task IDs in generation prompts
        assert idea.read_text().split('-->', 1)[1].strip().startswith('Additional requirement for this call:')
    for name in ('step2_explore_skill.md', 'step3_kept_skill.md'):
        assert not re.search(r'\b[0-9a-f]{8}\b', (idea_v1.PROMPTS / name).read_text())


def test_skill_calls_use_the_skill_steps_only_in_a_skill_run(monkeypatch):
    plain, _ = _call()
    plain = {**plain, 'commit': 'explore'}  # general runs use the explore loop
    skilled = {**plain, 'idea_modifiers': ['QUANTITY']}
    other = {**plain, 'idea_modifiers': ['LIGHT']}

    def steps(call):
        req = idea_v1.sketch_request(call)
        return (idea_v1.explore_request(call, req, SKETCHES, NEAR)['prompt'],
                idea_v1.commit_request(call, req, SKETCHES, NEAR)['prompt'])

    # without the run flag, a skill call renders exactly as any general call (control runs, earlier batches)
    monkeypatch.delitem(config.SCIENCE, 'skill_novelty', raising=False)
    explore, commit = steps(skilled)
    assert 'would be far from the ones shown' in explore and 'sharing the skill is not redundancy' not in commit
    monkeypatch.setitem(config.SCIENCE, 'skill_novelty', True)
    explore, commit = steps(skilled)
    assert 'still centre on this call\'s skill' in explore and 'would be far from the ones shown' not in explore
    assert 'sharing the skill is not redundancy' in commit
    assert "a listed task's rule, carried over to the sketch's pictures, would give its answers" in commit
    assert 'more than one convention a person would find arbitrary' in commit
    # a call without a skill line keeps the standard steps in a skill run
    explore, commit = steps(other)
    assert 'would be far from the ones shown' in explore and 'sharing the skill is not redundancy' not in commit
    # the skill text itself is in the call's prompt, after the arm card
    sketch = idea_v1.sketch_request(skilled)['prompt']
    assert sketch.index('## Direction for this call') < sketch.index('a quantity the human solver reads by looking')


def test_skill_selector_is_the_targeted_text_plus_one_passage(monkeypatch):
    from pathlib import Path
    targeted, skill = (V1 / 'selector_targeted.md').read_text(), (V1 / 'selector_skill.md').read_text()
    start = skill.index('- Many candidates in this run centre on a common reasoning skill')
    end = skill.index('separate things are not.\n', start) + len('separate things are not.\n')
    assert skill[:start].replace('blindspot-v1-selector-skill-3', 'diversity-v1-selector-targeted-2') + skill[end:] == targeted
    assert 'however much more plainly it teaches' in skill[start:end]  # a better-taught copy of the same rule stays redundant
    seen = []

    def fake(path):
        seen.append(Path(path).name); raise RuntimeError('stop')
    monkeypatch.setattr(supply, 'load_prompt', fake)
    monkeypatch.setitem(config.SCIENCE, 'prompt_set', 'v1')
    for skill_novelty, targeted in ((True, False), (False, True), (False, False)):
        monkeypatch.setitem(config.SCIENCE, 'skill_novelty', skill_novelty)
        monkeypatch.setitem(config.SCIENCE, 'targeted', targeted)
        with pytest.raises(RuntimeError):
            supply._selector_prompt({'id': 'w'}, {'payload': {}, 'packed_hash': 'x'})
    assert seen == ['selector_skill.md', 'selector_targeted.md', 'selector.md']


def test_skill_novelty_config_is_validated():
    import os
    base = read(os.environ.get('ARCGEN_CONFIG') or config.PROFILES / 'default.json')
    def check(**science):
        v = json.loads(json.dumps(base)); v['science'].update(science)
        return config.validate(v)
    mods = dict(base['science'].get('idea_modifiers') or {})
    if base['science'].get('idea_stage') != 'v1_loop':
        pytest.skip('the default profile is not a v1-loop profile')
    with pytest.raises(config.ConfigError):
        check(skill_novelty='yes')
    with pytest.raises(config.ConfigError):  # needs a skill line among the modifiers
        check(skill_novelty=True, idea_modifiers={k: v for k, v in mods.items() if k not in config.SKILL_MODIFIERS})


def test_pinned_skills_are_placed_first_and_checked():
    from arcgen.batch.allocation import assign_modifiers
    from arcgen.batch.common import IntegrityError
    calls = [{'opaque_id': f'c{k}', 'arm': arm, **({'pinned_modifiers': [skill]} if skill else {})}
             for k, (arm, skill) in enumerate([('JUDG', 'QUANTITY'), ('RESTORE', 'SYMMETRY'), ('GEO', 'QUANTITY'), ('SEE', 'SYMMETRY')])]
    rates = {'QUANTITY': 0.5, 'SYMMETRY': 0.5, 'LIGHT': 0.5}
    slots = {'skill': ['QUANTITY', 'SYMMETRY'], 'demand': ['LIGHT']}
    excl = {'QUANTITY': ['RESTORE'], 'SYMMETRY': ['JUDG']}
    assert assign_modifiers(calls, rates, slots, 2, excl) == {}
    assert [c['idea_modifiers'] for c in calls][:4] and all(
        set(c['idea_modifiers']) & {'QUANTITY', 'SYMMETRY'} == set(c['pinned_modifiers']) for c in calls)
    assert sum('LIGHT' in c['idea_modifiers'] for c in calls) == 2
    bad = [{'opaque_id': 'x', 'arm': 'RESTORE', 'pinned_modifiers': ['QUANTITY']}, {'opaque_id': 'y', 'arm': 'GEO'}]
    with pytest.raises(IntegrityError, match='does not fit'):
        assign_modifiers(bad, {'QUANTITY': 0.5}, {'skill': ['QUANTITY']}, 2, {'QUANTITY': ['RESTORE']})
    over = [{'opaque_id': f'z{k}', 'arm': 'GEO', 'pinned_modifiers': ['QUANTITY']} for k in range(3)]
    with pytest.raises(IntegrityError, match='more calls pin'):
        assign_modifiers(over, {'QUANTITY': 0.34}, {'skill': ['QUANTITY']}, 2)


def test_allocation_rows_may_pin_a_skill():
    from arcgen.batch.allocations.make_allocation import build
    spec = {'revision': 'r', 'seed': 1, 'windows': {'w': {'account': 'main', 'wave': 'A', 'context': 'c01', 'commit': 'explore'}},
            'rows': [['w', 'GEO', None, 'L', 'sol', 'high', {'skill': 'QUANTITY'}], ['w', 'SEE', None, 'L', 'sol', 'high']]}
    calls = build(spec, {'sol': 'gpt-6.1-sol'}, {}, lambda a, i: True)['calls']
    assert calls[0]['skill'] == 'QUANTITY' and 'skill' not in calls[1]
    for bad in ({'skill': 3}, {'family': 'FIT'}, {'skill': 'QUANTITY', 'other': 1}):
        with pytest.raises(ValueError):
            build({**spec, 'rows': [['w', 'GEO', None, 'L', 'sol', 'high', bad]]}, {'sol': 'm'}, {}, lambda a, i: True)
