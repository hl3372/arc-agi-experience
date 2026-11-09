"""v1 downstream: prompt set, planner notes, arm bridges and presentation overlays, under the v1 profile."""
from copy import deepcopy
import pytest
from arcgen.lib.util import content_hash, render_prompt
from arcgen.lib import astra_authoring_schemas_v3 as schemas
from arcgen.batch import config, admission, contracts
from arcgen.batch.contracts import Bindings, v1_planner_values, OVERLAY_INVITATIONS

from support.pilot import run_case


def test_profile_selects_v1_prompts_and_overlays():
    assert config.SCIENCE['prompt_set'] == 'v1'
    b = Bindings()
    assert b.prompt_overrides['concrete_planner'][0] == 'v1/planner.md'
    assert b.prompt_overrides['evidence_scale_addendum'][0] == 'v1/overlay_addendum.md'
    # b2: the planner-declined scale and RICH invitations are retired; XFER is listed twice (double weight)
    assert config.SCIENCE['routes']['scale_overlay']['kinds'] == ['MANY', 'XFER', 'XFER']  # COMPACT dropped 2026-10-05


def _row(sid, source='ordinary_unconditioned'):
    return {'source_candidate_id': sid, 'private_source': source, 'study': {'dispatch_ordinal': 1}}


def test_overlay_draw_rate_kinds_and_specialists():
    rows = [_row(f'k{i:04d}') for i in range(2000)]
    routed = admission.configured_routes({}, rows)
    used = [r for r in routed if r['private_route'] == 'evidence_preserving_scale']
    policy = config.SCIENCE['routes']['scale_overlay']
    assert abs(len(used) / len(routed) - policy['rate']) < 0.04
    kinds = [r['study']['overlay'] for r in used]
    assert set(kinds) == set(policy['kinds'])
    assert abs(kinds.count('XFER') / len(kinds) - 2 / 3) < 0.05  # listed twice of three
    assert all('overlay' not in r['study'] for r in routed if r['private_route'] == 'ordinary_current')
    special = admission.configured_routes({}, [_row('kJ', 'episode_joint_inference'), _row('kT', 'logical_leap')])
    assert [r['private_route'] for r in special] == ['episode_joint_inference', 'logical_leap']
    again = admission.configured_routes({}, deepcopy(rows[:50]))
    assert [r['study'].get('overlay') for r in again] == [r['study'].get('overlay') for r in routed[:50]]


def _values(tid):
    return {'CANDIDATE': {'candidate_id': tid, 'central_relation': 'A relation.'},
            'EDITORIAL_BRIEF': {'candidate_id': tid, 'development_priority': 1, 'route': 'wildcard', 'preserve': 'P',
                                'develop': 'Develop note.', 'avoid': 'Risk note.', 'closest_similarity_concern': 'x',
                                'selection_reason': 'y', 'optional_development_suggestion': None},
            'ROUTE_GUIDANCE': 'Test the high-upside idea honestly; abort if exact, fair, economical grids cannot preserve both editorial floors.',
            'CANDIDATE_ID': tid}


@pytest.mark.parametrize('primary,kind', [('RICH', 'RICH'), ('T', None), ('SEM', 'MANY'), ('ABS', 'COMPACT'), ('FORM', None)])
def test_v1_planner_prompt_renders_and_validates(primary, kind):
    b = Bindings(); tid = 'dtest0000000000000001'
    record = {'private_route': 'evidence_preserving_scale' if kind else 'ordinary_current',
              'study': {'overlay': kind} if kind else {}, 'selection_note': {'useful_contribution': 'The distinct part.'}}
    values = v1_planner_values(record, _values(tid))
    routes = ('evidence_scale_addendum',) if kind else ()
    if primary == 'T' and not kind:
        routes = ('logical_leap_planner_addendum',)
    rows = b.planning_templates(primary, routes)
    prepared, context = b.prepare_values('concrete_planner', values)
    context.update(q2_primary=primary, q2_template_hashes=[r[2] for r in rows])
    prompt = render_prompt('\n\n'.join(r[0] for r in rows), prepared)
    stage = b.prompt_overrides[routes[-1] if routes else 'concrete_planner'][1]
    b.validate_request(stage, prompt, context)
    assert 'The distinct part.' in prompt and 'economical' not in prompt and 'editorial floors' not in prompt
    assert 'A role may\ninstead take a different colour in different pairs' in prompt
    assert 'simplest legible embodiment' not in prompt and 'literal_numeric' not in prompt
    assert 'Keep the kernel\'s reasoning mode.' in prompt
    if kind:
        assert OVERLAY_INVITATIONS[kind] in prompt
    if primary == 'RICH':
        assert 'Seek a coherent task whose actual grids' not in prompt  # embodiment bridges are replaced by overlays
    if primary in ('SEM', 'FORM'):
        assert f'diversity-v1-bridge-{primary}-' in prompt  # any version of this arm's bridge


@pytest.mark.parametrize('case,calls,checkpoint', [('clear', 4, None)])
def test_v1_full_mock_pipeline(tmp_path, monkeypatch, case, calls, checkpoint):
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    run_case(tmp_path, monkeypatch, case, calls, True, crash_point=checkpoint, arm='H_DISC', planner='sol')
    import json
    plans = list(tmp_path.glob('accounts/**/model_calls/astra_concrete_plan_v3/*/prompt.md'))
    assert plans and all('diversity-v1-planner-' in p.read_text() for p in plans)  # plain or colour-vary (the mix)
    text = plans[0].read_text()
    assert 'Development notes from selection' in text and 'what_makes_it_distinct' in text
    real = list(tmp_path.glob('accounts/**/model_calls/astra_task_realizer_v3/*/prompt.md'))
    assert real and 'follow\neach pair\'s stated aliases exactly' in real[0].read_text()


def test_v1_repair_tokens_expand_to_the_frozen_fields():
    from support.fixtures import episode
    from arcgen.batch import repair, repair_v1
    tid = 'dtest0000000000000002'
    plan, real = episode(tid)
    p = deepcopy(plan); r = deepcopy(real)
    for name in ('input', 'output'):
        grid = r['task']['train'][0][name]; grid[0], grid[1] = grid[1], grid[0]
    changed = deepcopy(p['public_contract']['training_pairs'][0])
    changed['exact_input_construction'] = 'Place the marked cell in row 1 and preserve the anchor.'
    pub = p['public_contract']
    pub['training_pairs'] = [changed] + [f'reuse_frozen_training_pair_{k}' for k in range(2, len(plan['public_contract']['training_pairs']) + 1)]
    pub['semantic_core'] = repair_v1.REUSE_CORE; pub['allowed_realizer_freedom'] = repair_v1.REUSE_FREEDOM
    pub['episode_overview'] = repair_v1.REUSE_OVERVIEW; pub['invariants'] = repair.REUSE_INVARIANTS
    pub['test_inputs'] = repair.REUSE_TESTS; p['private_execution']['test_outputs'] = repair.REUSE_ANSWERS
    r['task']['test'] = repair.REUSE_TESTS; r['authoring_metadata']['expected_test_outputs'] = repair.REUSE_ANSWERS
    value = dict(contract_version=repair.VERSION, training_support_updates=[], candidate_id=tid, repair_type='ambiguity',
                 status='repaired', unrepairable_reason='', plan_update=p, repaired_realization=r,
                 changes=['Replaced one public demonstration.'], why_rule_and_test_purpose_are_preserved='Unchanged rule and tests.')
    values = {'CANDIDATE_ID': tid, 'REPAIR_TYPE': 'ambiguity', 'AUTHOR_SPECIFICATION': {'concrete_plan': plan, 'realization': real},
              'ALLOW_MECHANICAL_TEST_INPUT_RESTORATION': False}
    out = repair_v1.expand(value, values)
    after = out['repaired_plan']['public_contract']; before = plan['public_contract']
    for key in ('semantic_core', 'allowed_realizer_freedom', 'invariants', 'test_inputs'):
        assert after[key] == before[key]
    assert after['episode_overview'] == before['episode_overview']
    assert after['training_pairs'][1:] == before['training_pairs'][1:] and after['training_pairs'][0] != before['training_pairs'][0]
    bad = deepcopy(value); bad['plan_update']['public_contract']['training_pairs'][1] = 'reuse_frozen_training_pair_9'
    with pytest.raises(Exception):
        repair_v1.expand(bad, values)


def test_v1_validation_failures_get_their_own_label():
    from arcgen.batch import gate
    from arcgen.lib.schemas import SchemaError
    assert gate.MAX_REPAIRS == 3 and gate.MAX_ADDED_PAIRS == 2
    try:
        try:
            raise SchemaError('$.plan_update', 'ambiguity repair changed a frozen public commitment')
        except SchemaError as inner:
            raise RuntimeError('attempts exhausted') from inner
    except RuntimeError as exc:
        assert gate._error_kind(exc) == 'validation'
    assert gate._error_kind(RuntimeError('network')) == 'operational'


def test_v1_full_mock_pipeline_repair_uses_v1_stage(tmp_path, monkeypatch):
    import types
    from support import pilot as pilot_fixture
    from arcgen.batch import repair, repair_v1
    shim = types.SimpleNamespace(STAGE=repair_v1.STAGE, REUSE_TESTS=repair.REUSE_TESTS, REUSE_ANSWERS=repair.REUSE_ANSWERS, VERSION=repair.VERSION)
    monkeypatch.setattr(pilot_fixture, 'repair', shim)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    run_case(tmp_path, monkeypatch, 'repair_clear', 8, True, arm='H_DISC', planner='sol')
    assert list(tmp_path.glob('accounts/**/model_calls/diversity_v1_bounded_repair/*/prompt.md'))
    adj = list(tmp_path.glob('accounts/**/model_calls/difficulty_adjudication_v2/*/prompt.md'))
    assert adj and 'identify its whole family' in adj[0].read_text()


def test_v1_second_repair_is_a_new_call(tmp_path, monkeypatch):
    """A task still ambiguous after one repair gets a second repair under its own call ID and can then verify."""
    import types
    from support import pilot as pilot_fixture
    from arcgen.batch import repair, repair_v1
    shim = types.SimpleNamespace(STAGE=repair_v1.STAGE, REUSE_TESTS=repair.REUSE_TESTS, REUSE_ANSWERS=repair.REUSE_ANSWERS, VERSION=repair.VERSION)
    monkeypatch.setattr(pilot_fixture, 'repair', shim)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    base = pilot_fixture.limits

    def roomy(account='main'):  # two repair rounds need more conditional calls than the one-repair fixture funds
        value = base(account)
        for cats in [value['phases'][0]['categories'], value['accounts'][account]['categories']]:
            cats['conditional_evidence_repair'] = 12
        value['logical_cap'] = value['phases'][0]['logical_cap'] = value['phases'][0]['astra_logical_cap'] = value['accounts'][account]['logical_cap'] = 30
        return value
    monkeypatch.setattr(pilot_fixture, 'limits', roomy)
    first, calls, *_ = run_case(tmp_path, monkeypatch, 'repair_twice', 12, True, arm='H_DISC', planner='sol')
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    repairs = [cid for stage, cid in calls if stage == repair_v1.STAGE]
    assert record['repair_count'] == 2 and len(record['difficulty_versions']) == 3
    assert len(repairs) == 2 and repairs[0].endswith('-ambiguity-repair') and repairs[1] != repairs[0]
    assert repairs[1].endswith('-ambiguity-repair') and '-v' in repairs[1]


def test_v1_repairs_may_replace_pairs_but_add_at_most_two(monkeypatch):
    """The planner's pair count is never capped; repairs may add at most two pairs beyond it."""
    import threading, types
    from arcgen.batch import gate
    seen = {}
    monkeypatch.setattr(gate.adjudication, 'repair_challenges', lambda adj: [])
    monkeypatch.setattr(gate, '_finish', lambda rt, tid, kind, reason, **kw: seen.update(reason=reason))
    monkeypatch.setattr(gate, '_store_repaired_realization', lambda *a, **kw: (False, 'stopped here'))

    def attempt(planned, current, returned):
        seen.clear()
        record = {'repair_count': current - planned, 'complete_task': {'train': [{}] * current},
                  'planning_result': {'concrete_plan': {'public_contract': {'episode_overview': {'training_pair_count': planned}}}}}
        runtime = types.SimpleNamespace(lock=threading.Lock(), state={'semantic_records': {'t': record}}, save=lambda: None)

        def repair_call(rt, tid, *, repair_type, instruction, evidence, call_id):
            seen['instruction'] = instruction
            return {'status': 'repaired', 'repaired_plan': {}, 'repaired_realization': {'task': {'train': [{}] * returned}}}
        monkeypatch.setattr(gate, '_repair_call', repair_call)
        budget = types.SimpleNamespace(reserve_once=lambda key, n: True)
        gate._repair(runtime, 't', budget, 'a' * 64, {'repair_instruction': 'Fix it.'})
        return seen.get('instruction'), seen.get('reason')

    # A task planned with 6 pairs still gets its repair (the old absolute cap refused it).
    assert attempt(6, 6, 6) == ('Fix it.', 'repair_fidelity_failure: stopped here')
    assert attempt(6, 6, 7)[1] == 'repair_fidelity_failure: stopped here'
    # Two pairs already added: the repairer is told to replace, and a third addition is refused.
    instruction, reason = attempt(3, 5, 6)
    assert instruction.startswith('Fix it.') and 'Replace an existing training pair' in instruction
    assert reason == 'repair_added_too_many_pairs'
    assert attempt(3, 5, 5)[1] == 'repair_fidelity_failure: stopped here'


def test_v1_gallery_lists_every_realized_task(tmp_path):
    """The end-of-run gallery shows each realized task with its rule, arm, idea, overlay, checks and status."""
    import json
    from arcgen.batch import gallery
    task = {'train': [{'input': [[0, 1]], 'output': [[1, 0]]}], 'test': [{'input': [[1, 1]], 'output': [[0, 0]]}]}
    plan = {'public_contract': {'semantic_core': {'central_rule': 'Swap the two cells.'}}}
    base = {'source_candidate_id': 'k1', 'concrete_plan': plan, 'complete_task': task, 'answer_evidence': {}}
    records = {'t1': {**base, 'study': {'primary_arm': 'SEM', 'overlay': 'MANY'}, 'private_route': 'evidence_preserving_scale',
                      'planning_result': {'scale_decision': 'expanded'}, 'repair_count': 1},
               't2': {**base, 'source_candidate_id': 'k2', 'study': {'primary_arm': 'T'}, 'private_route': 'ordinary_current',
                      'planning_result': {'decision': 'ordinary_current'}, 'status': 'ambiguous',
                      'answer_evidence': {'difficulty_outcome': {'reason': 'repair_declined'}}}}
    state = tmp_path / 'accounts/main/w-develop-01/downstream/state.json'
    state.parent.mkdir(parents=True); state.write_text(json.dumps({'semantic_records': records}))
    queue = tmp_path / 'accounts/main/w/eligible_queue.private.json'
    queue.parent.mkdir(parents=True)
    queue.write_text(json.dumps({'rows': [{'source_candidate_id': 'k1', 'selector_assessment': {'useful_contribution': 'Useful.'},
                                           'kernel': {'central_relation': 'A swap.', 'what_is_new': 'New.', 'loop_source': 'replacement'}}]}))
    row = {'realized': True, 'state': {'path': str(state)}, 'generation_model': 'gpt-6.1-sol', 'effort': 'high', 'rule': 'Swap.',
           'task_file': None, 'status': 'blind_pass', 'blind_performance': {'any_all_test_success': True, 'label': None}}
    tasks = [{**row, 'task_id': 't1', 'arm': 'SEM', 'usable_after_exact_palette_union': True, 'verified_before_union': True},
             {**row, 'task_id': 't2', 'arm': 'T', 'usable_after_exact_palette_union': False, 'verified_before_union': False},
             {'task_id': 't3', 'realized': False}]
    out = gallery.render(tmp_path, tasks)  # the v1 profile routes the end-of-run call to the v1 gallery
    meta = {r['task_id']: r for r in json.loads((tmp_path / 'gallery/metadata.json').read_text())['tasks']}
    assert out['tasks'] == 2 and set(meta) == {'t1', 't2'}
    assert meta['t1']['status'] == 'verified' and meta['t1']['arm_name'] == 'Meaning learned in the task'
    assert meta['t1']['overlay'] == {'kind': 'MANY', 'decision': 'expanded'} and meta['t1']['repairs'] == 1
    assert meta['t1']['idea'] == 'A swap.' and meta['t1']['idea_origin'] == 'replacement after the bank check'
    assert meta['t2']['status'] == 'held' and meta['t2']['status_text'] == 'Held: repair declined' and meta['t2']['idea'] is None
    page = (tmp_path / 'gallery/index.html').read_text()
    assert '2 realized tasks: 1 verified, 0 bank duplicates, 1 held.' in page and '"task_id":"t1"' in page


def test_colour_roles_choice_selects_the_neutral_colour_templates(monkeypatch):
    """colour_roles 'choice' swaps in planner and repair templates that ask for a neutral per-role colour decision."""
    import copy, pytest
    from arcgen.batch import contracts
    from arcgen.batch.config import validate, ConfigError, CONFIG
    assert contracts.Bindings().prompt_overrides['concrete_planner'][0] == 'v1/planner.md'
    monkeypatch.setattr(contracts, 'COLOUR_ROLES', 'choice')
    b = contracts.Bindings()
    assert b.prompt_overrides['concrete_planner'][0] == 'v1/planner_colour_choice.md'
    assert b.prompt_overrides['targeted_repair'][0] == 'v1/bounded_repair_colour_choice.md'
    for name in ('concrete_planner', 'targeted_repair'):
        text = b.template(name)[0]
        assert 'neither is preferred' in text and 'that is fine' not in text
    monkeypatch.setattr(contracts, 'COLOUR_ROLES', 'vary')
    b = contracts.Bindings()
    assert b.prompt_overrides['concrete_planner'][0] == 'v1/planner_colour_vary.md'
    assert b.prompt_overrides['targeted_repair'][0] == 'v1/bounded_repair_colour_choice.md'
    assert 'without\ncompromising the rule' in b.template('concrete_planner')[0]
    bad = copy.deepcopy({k: v for k, v in CONFIG.items() if not k.startswith('_')})
    bad['science']['colour_roles'] = 'choice'; bad['science']['prompt_set'] = 'r23'; bad['science'].pop('idea_stage', None)
    with pytest.raises(ConfigError):
        validate(bad)


def test_colour_mix_draw_is_per_idea_and_near_the_rate():
    """Each idea is drawn 'vary' or 'v1' reproducibly; about colour_vary_rate of ideas get the vary instruction."""
    from arcgen.batch import admission
    rows = [{'source_candidate_id': f'k{i:04d}', 'private_source': 'ordinary_unconditioned', 'study': {}} for i in range(2000)]
    a = admission.configured_routes({}, rows); b = admission.configured_routes({}, rows)
    draws = [r['study']['colour_roles'] for r in a]
    assert draws == [r['study']['colour_roles'] for r in b] and set(draws) == {'vary', 'v1'}
    assert abs(draws.count('vary') / len(draws) - config.SCIENCE['colour_vary_rate']) < 0.04


@pytest.mark.parametrize('rate,expect_vary', [(1.0, True), (0.0, False)])
def test_colour_mix_planner_prompt_follows_the_draw(tmp_path, monkeypatch, rate, expect_vary):
    """A full mock development: the planner prompt carries the vary paragraph exactly for ideas drawn 'vary'."""
    monkeypatch.setitem(config.SCIENCE, 'colour_vary_rate', rate)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    first, *_ = run_case(tmp_path, monkeypatch, 'clear', 4, True, arm='H_DISC', planner='sol')
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert record['study']['colour_roles'] == ('vary' if expect_vary else 'v1')
    prompt, = list(tmp_path.glob('accounts/**/model_calls/astra_concrete_plan_v3/*/prompt.md'))
    text = prompt.read_text()
    assert ('without\ncompromising the rule' in text) is expect_vary and ('that is fine' in text) is not expect_vary


def test_geo_is_a_v1_arm_with_a_card_and_planner_bridge():
    from arcgen.batch import idea_v1, supply, contracts
    from arcgen.batch.allocation import design
    assert 'GEO' in idea_v1.ARMS and contracts.known_primary('GEO')
    call = next(c for w in design()['windows'] for c in w['calls'] if c['arm'] == 'GEO')
    req = supply.idea_request(call, None, None)
    assert 'spatial geometry of the picture' in req['prompt']
    rows = contracts.Bindings().planning_templates('GEO', ())
    assert any('Preserve the spatial relation' in r[0] for r in rows)
