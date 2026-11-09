"""Property slots and property arms (2026-10-07): exact slot shares, the cap and exclusions, the coherence preamble,
planner slot lines, chained STAGES ideas, the network retry pause and the grid checks of achieved properties."""
from copy import deepcopy
import pytest
from arcgen.batch import admission, config, contracts, idea_v1, properties, supply
from arcgen.batch.allocation import assign_modifiers, design
from arcgen.lib.util import render_prompt

NEW_ARMS = ('LEVELS', 'INVERSE', 'AIM', 'ACROSS', 'STAGES', 'CONTRAST', 'VIEWS', 'SPREAD')
NEW_SLOTS = ('DIGEST', 'EXTENT', 'PACKED', 'VACANCY', 'IGNORE', 'PAYLOAD', 'MONO', 'LIGHT', 'BUSY', 'COLOURFUL',
             'KEY', 'CHAINED', 'MINIMAL', 'DISTRIBUTED', 'DERIVED', 'MEASURE', 'EASIER', 'SCENE', 'JOINED', 'INSTRUCT', *config.SKILL_MODIFIERS,  # skill lines (2026-10-18) have planner lines too
             'ONE')  # 2026-10-23: a single demonstration pair (slots.py)


def _calls(n, arms=('SEE', 'FORM', 'SIGNAL', 'WORLD')):
    return [{'opaque_id': f'p{i:05d}', 'arm': arms[i % len(arms)]} for i in range(n)]


def test_every_arm_and_modifier_has_its_texts():
    p = idea_v1.PROMPTS
    for arm in NEW_ARMS:
        assert arm in idea_v1.ARMS and contracts.known_primary(arm) and arm in contracts.V1_BRIDGES
        assert (p / 'cards' / f'{arm}.md').read_text().count('Seek a task') == 1
    for m in idea_v1.MODIFIERS:
        assert (p / 'modifiers' / f'{m}.md').exists()
    assert set(NEW_SLOTS) == contracts.SLOT_BRIDGES  # every new slot value has a planner line, and only those


def test_reference_profile_slots_partition_the_modifiers():
    s = config.SCIENCE
    assert s['modifier_assignment'] == 'exact' and s['modifier_cap'] == 3
    assert sorted(m for v in s['modifier_slots'].values() for m in v) == sorted(s['idea_modifiers'])
    # skills: skill runs only; weakness lines (2026-10-20): weakness runs only
    assert set(s['idea_modifiers']) == set(config.MODIFIER_NAMES) - set(config.SKILL_MODIFIERS) - set(config.LINE_MODIFIERS)


def test_reference_design_gets_exact_shares_within_slots_cap_and_exclusions():
    s = config.SCIENCE
    calls = [c for w in design()['windows'] if w['kind'] == 'fresh_supply' for c in w['calls']]
    slot_of = {m: k for k, v in s['modifier_slots'].items() for m in v}
    for m, rate in s['idea_modifiers'].items():
        assert sum(m in c['idea_modifiers'] for c in calls) == round(rate * len(calls)), m
    for c in calls:
        mods = c['idea_modifiers']
        assert len(mods) <= s['modifier_cap'] and len({slot_of[m] for m in mods}) == len(mods)
        for m in mods:
            assert c['arm'] not in s['modifier_exclusions'].get(m, ())
            assert not set(mods) & set(s['modifier_exclusions'].get(m, ()))


def test_assign_modifiers_without_slots_is_the_plain_exact_draw():
    """No slots, cap or exclusions: every modifier independently gets its exact share (the v1-b11 behaviour)."""
    calls = _calls(200)
    rates = {'CONSTRUCT': 0.75, 'EASIER': 0.2, 'COLOURFUL': 0.3, 'BUSY': 0.25}
    assert assign_modifiers(calls, rates) == {}
    for m, r in rates.items():
        assert sum(m in c['idea_modifiers'] for c in calls) == round(r * 200)
    again = _calls(200); assign_modifiers(again, dict(reversed(list(rates.items()))))
    assert [c['idea_modifiers'] for c in again] == [c['idea_modifiers'] for c in calls]  # order-free


def test_assign_modifiers_honours_slots_cap_and_both_directions_of_exclusion():
    calls = _calls(400)
    rates = {'A': 0.5, 'B': 0.5, 'C': 0.3, 'D': 0.3, 'E': 0.3}
    short = assign_modifiers(calls, rates, slots={'s1': ['A', 'B'], 's2': ['C'], 's3': ['D'], 's4': ['E']}, cap=2,
                             exclusions={'C': ['SIGNAL', 'D']})
    for c in calls:
        mods = c['idea_modifiers']
        assert not {'A', 'B'} <= set(mods) and len(mods) <= 2
        assert not ('C' in mods and (c['arm'] == 'SIGNAL' or 'D' in mods))
    placed = {m: sum(m in c['idea_modifiers'] for c in calls) for m in rates}
    assert all(placed[m] + short.get(m, 0) == round(rates[m] * 400) for m in rates)
    assert placed['C'] == 120  # the rarest values go first, so the cap does not starve them


def test_unknown_exclusion_targets_are_refused():
    from arcgen.batch.common import IntegrityError
    with pytest.raises(IntegrityError):
        assign_modifiers(_calls(10), {'A': 0.5}, exclusions={'A': ['NOT_AN_ARM']})


def test_preamble_only_with_modifiers():
    call = next(c for w in design()['windows'] for c in w['calls'] if c['idea_modifiers'])
    req = supply.idea_request(call, None, None)
    assert 'why_coherent' in req['prompt'] and 'Additional requirement' in req['prompt'] or 'Additional aim' in req['prompt']
    bare = dict(call, idea_modifiers=[])
    assert 'unrelated addition' not in supply.idea_request(bare, None, None)['prompt']
    assert 'unrelated addition' in req['prompt']


def test_planner_gets_slot_lines_after_the_bridge_and_validates():
    b = contracts.Bindings()
    rows = b.planning_templates('SEE', (), colour='vary', pose=True, slots=('IGNORE', 'PACKED'))
    texts = [r[0] for r in rows]
    assert any('irrelevant' in t for t in texts[-2:]) and any('Keep the inputs dense' in t for t in texts[-2:])
    with pytest.raises(Exception):
        b.planning_templates('SEE', (), slots=('CONSTRUCT',))  # no planner line for the original modifiers
    with pytest.raises(Exception):
        b.planning_templates('SEE', (), slots=('PACKED', 'IGNORE'))  # fixed order only
    tid = 'dtest0000000000000001'
    values = {'CANDIDATE': {'candidate_id': tid, 'central_relation': 'A relation.'},
              'EDITORIAL_BRIEF': {'candidate_id': tid, 'development_priority': 1, 'route': 'wildcard', 'preserve': 'P',
                                  'develop': 'D.', 'avoid': 'R.', 'closest_similarity_concern': 'x', 'selection_reason': 'y',
                                  'optional_development_suggestion': None}, 'ROUTE_GUIDANCE': 'G.', 'CANDIDATE_ID': tid}
    record = {'private_route': 'ordinary_current', 'study': {}, 'selection_note': {'useful_contribution': 'Distinct.'}}
    prepared, context = b.prepare_values('concrete_planner', contracts.v1_planner_values(record, values))
    context.update(q2_primary='SEE', q2_template_hashes=[r[2] for r in rows], q2_colour_roles='vary', q2_pose=True,
                   q2_slots=['IGNORE', 'PACKED'])
    prompt = render_prompt('\n\n'.join(texts), prepared)
    b.validate_request(b.prompt_overrides['concrete_planner'][1], prompt, context)
    context['q2_slots'] = ['PACKED']
    with pytest.raises(Exception):
        b.validate_request(b.prompt_overrides['concrete_planner'][1], prompt, context)


def test_planner_slots_come_from_the_idea_call():
    assert contracts.planner_slots({'idea_modifiers': ['PACKED', 'CONSTRUCT', 'DIGEST']}) == ('DIGEST', 'PACKED')
    assert contracts.planner_slots({}) == ()


def test_stages_and_contrast_ideas_keep_one_colouring_and_no_pose():
    rows = [{'source_candidate_id': f'k{i:04d}', 'private_source': 'ordinary_unconditioned',
             'study': {'primary_arm': ('STAGES', 'CONTRAST')[i % 4 // 2] if i % 2 else 'SEE'}} for i in range(400)]
    routed = admission.configured_routes({}, deepcopy(rows))
    stages = [r for r in routed if r['study']['primary_arm'] in ('STAGES', 'CONTRAST')]
    assert {r['study']['primary_arm'] for r in stages} == {'STAGES', 'CONTRAST'}
    others = [r for r in routed if r['study']['primary_arm'] == 'SEE']
    assert all(r['study']['colour_roles'] == 'v1' and r['study']['pose'] is False for r in stages)
    assert {r['study']['colour_roles'] for r in others} == {'vary', 'v1'} and any(r['study']['pose'] for r in others)


def test_network_failure_waits_before_the_retry(monkeypatch, tmp_path):
    from arcgen.lib import e18_calls
    assert e18_calls.network_failure('ERROR: Reconnecting... 5/5\nERROR: workspace routing discovery failed')
    assert not e18_calls.network_failure('$.kernels: expected 4 items')
    assert e18_calls.network_failure('ERROR: Reconnecting... 2/5\nERROR: Selected model is at capacity. Please try a different model.')
    assert e18_calls.network_failure('Codex exited with status 1: ERROR: exceeded retry limit, last status: 429 Too Many Requests')
    slept = []
    monkeypatch.setattr(e18_calls.time, 'sleep', lambda s: slept.append(s))
    from arcgen.lib.model import ModelInvocationError
    from dataclasses import dataclass
    from pathlib import Path

    @dataclass
    class Result:
        cache_source: str = 'live'
        log_dir: Path = tmp_path
        attempts: int = 1

    class Inner:
        output_schemas = {'stage': {}}
        log_root = tmp_path
        calls = 0

        def invoke(self, **kw):
            Inner.calls += 1
            if Inner.calls == 1:
                raise ModelInvocationError('Codex exited with status 1: ERROR: Reconnecting... 5/5')
            assert 'failed the validation' not in kw['prompt']
            return Result()

    inv = e18_calls.E18AttemptInvoker(Inner(), tmp_path / 'journal.json', maximum_attempts=2)
    monkeypatch.setattr(e18_calls, '_call_dir', lambda inner, stage, call_id: tmp_path / 'calls' / call_id)
    monkeypatch.setattr(e18_calls, 'SEMANTIC_FEEDBACK_STAGES', {'stage'})
    monkeypatch.setattr(e18_calls, 'retry_feedback', lambda d, s: None)
    monkeypatch.setattr(inv, '_archive_failed_attempt', lambda d, used: None)
    inv.invoke(stage='stage', call_id='c1', prompt='P')
    b = e18_calls.NETWORK_BACKOFF_SECONDS
    assert len(slept) == 1 and b <= slept[0] <= b + e18_calls.NETWORK_JITTER_SECONDS and Inner.calls == 2


def test_grid_checks_of_achieved_properties():
    g = lambda rows: [list(r) for r in rows]
    chain = {'train': [{'input': g(['10', '00']), 'output': g(['11', '00'])}, {'input': g(['11', '00']), 'output': g(['11', '10'])}],
             'test': [{'input': g(['11', '10']), 'output': g(['11', '11'])}]}
    c = properties.checks(chain)
    assert c['STAGES'] and c['MONO'] and not c['PACKED'] and c['same_size']
    digest_task = {'train': [{'input': g(['1234'] * 4), 'output': g(['1'])}, {'input': g(['4321'] * 4), 'output': g(['42'])}],
                   'test': [{'input': g(['1324'] * 4), 'output': g(['3'])}]}
    c = properties.checks(digest_task)
    assert c['DIGEST'] and c['EXTENT'] and c['PACKED'] and not c['STAGES']
    rep = properties.report([{'complete_task': chain, 'study': {'primary_arm': 'STAGES', 'idea_modifiers': ['MONO', 'LIGHT']}}])
    assert rep['by_property']['STAGES'] == {'verified': 1, 'pass': 1} and rep['by_property']['LIGHT'] == {'verified': 1}
    assert properties.lines(rep)
    # each grid against its own ground (v1-b14 audit): ground and figure colours that swap across pairs pass MONO, and a
    # 60% ground fails PACKED although no one colour is the ground of every input; DIGEST takes a 40-cell answer
    swap = {'train': [{'input': g(['0001', '0011', '0000']), 'output': g(['1'])}, {'input': g(['5552', '5522', '5555']), 'output': g(['2'])}],
            'test': [{'input': g(['2223', '2222', '2232']), 'output': g(['3'])}]}
    c = properties.checks(swap)
    assert c['MONO'] and not c['PACKED']
    big = [list(range(10)) * 2] * 10
    c = properties.checks({'train': [{'input': big, 'output': [[1] * 8] * 5}], 'test': [{'input': big, 'output': [[2] * 10] * 5}]})
    assert c['DIGEST'] and c['PACKED']
    assert not properties.checks({'train': [{'input': big, 'output': [[1] * 17] * 3}], 'test': [{'input': big, 'output': [[2]]}]})['DIGEST']


def test_unmeetable_shares_stop_the_design(monkeypatch):
    from arcgen.batch.common import IntegrityError
    monkeypatch.setitem(config.SCIENCE, 'modifier_exclusions', {**config.SCIENCE['modifier_exclusions'],
                                                                'MONO': sorted(idea_v1.ARMS)})
    with pytest.raises(IntegrityError, match='cannot be met'):
        design()


def test_summary_survives_a_failing_property_report(monkeypatch):
    from arcgen.batch import cli
    monkeypatch.setattr(properties, 'report', lambda records: 1 / 0)
    value = cli._properties([{'complete_task': None, 'study': {}}])
    assert 'error' in value
    monkeypatch.undo()
    task = {'train': [{'input': [[1, 0]], 'output': [[1, 1]]}], 'test': [{'input': [[0, 1]], 'output': [[1, 1]]}]}
    rep = properties.report([{'complete_task': task, 'study': {'primary_arm': None, 'idea_modifiers': ['MONO']}},
                             {'complete_task': task, 'study': {'primary_arm': 'SEE'}}])  # a missing arm sorts too
    assert properties.lines(rep)


def test_bare_reservations_and_arm_caps():
    calls = _calls(300, arms=('LEVELS', 'SEE', 'FORM'))
    rates = {'A': 0.6, 'B': 0.6, 'C': 0.6}
    assign_modifiers(calls, rates, slots={'a': ['A'], 'b': ['B'], 'c': ['C']}, cap=3, arm_caps={'LEVELS': 2},
                     bare={'LEVELS': 5, '*': 7, 'STAGES': 3})  # STAGES has no calls here: no reservation needed
    assert sum(not c['idea_modifiers'] for c in calls if c['arm'] == 'LEVELS') >= 5
    assert sum(not c['idea_modifiers'] for c in calls) >= 12
    assert max(len(c['idea_modifiers']) for c in calls if c['arm'] == 'LEVELS') <= 2
    from arcgen.batch.common import IntegrityError
    with pytest.raises(IntegrityError):
        assign_modifiers(_calls(6, arms=('LEVELS', 'SEE')), {'A': 0.5}, bare={'LEVELS': 5})
