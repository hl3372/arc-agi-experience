"""Diversity-v1 idea stage: draft -> look up nearest bank tasks -> replace collisions -> commit four kernels.

One idea call keeps the batch's shape (four opaque kernel IDs, source rows, selector), but runs two model turns:
  1. `v1_idea_sketch`: the author writes 12-16 sketches, each with a theme-free recipe sentence;
  2. `v1_idea_commit`: the author sees each sketch's nearest existing bank tasks (retrieved from the run's frozen
     reference index), drops sketches that need no fresh insight ("replace, never extend"), and commits four kernels.
The committed kernels become ordinary source rows. The prompt texts are the ones the Step 3 screen's group D used
(prompts/v1/ideas/): the bar, the formal line and the v1 concept arms; no calibration, bank list, pins or nomination.
"""
from copy import deepcopy
from arcgen.lib.factory_runtime import StageBinding
from arcgen.lib.factory_supply import IDEA_PURPOSES
from arcgen.lib.factory_context import FactoryContextGuard
from arcgen.lib.schemas import SchemaError, parse_json_object
from arcgen.lib.util import render_prompt
from arcgen.lib.v4_schemas import _transport_schema
from arcgen.astra_evidence_runtime_revision_20260911.structure import check
from .common import HERE, IntegrityError, component, digest, ref, frozen
from .config import MODIFIER_NAMES, FAMILIES, RUNGS, SCIENCE, SKILL_MODIFIERS, LINE_MODIFIERS, NOVELTY_LINES, LINE_REGISTRY

STAGE_SKETCH = 'v1_idea_sketch'
STAGE_COMMIT = 'v1_idea_commit'
STAGE_COMMIT_KEPT = 'v1_idea_commit_kept'  # commit every bank-checked kept draft (up to six); no new ideas
STAGE_EXPLORE = 'v1_idea_explore'  # explore round: new sketches aimed beyond where the first ones landed, then checked too
EXPLORE_COUNT = 6
SEARCH_MODE = 'v1_loop'
BANK_SETTING = 'loop'
ARMS = ('T', 'J', 'ABS', 'DISC', 'H_DISC', 'H_REASON', 'H_COMPOSE', 'JUDG', 'D1', 'ACC', 'EXEC', 'FORM', 'SEM', 'PER', 'GEO',
        'WORLD', 'RESTORE', 'GEN', 'EXEMPLAR', 'INTERACT', 'SEE', 'TEMPLATE', 'OPEN', 'DIRECT', 'BOARD', 'RHYTHM', 'SIGNAL',
        # property arms (2026-10-07): each names a property of a task that admits many rules, not a family of rules
        'LEVELS', 'INVERSE', 'AIM', 'ACROSS', 'STAGES', 'CONTRAST', 'VIEWS',
        # 2026-10-09: many pairs, each revealing part of one system
        'SPREAD')
MODIFIERS = MODIFIER_NAMES  # optional idea-call additions and property slots (prompts/v1/ideas/modifiers/)
SOURCE_CLASS = {'J': 'episode_joint_inference', 'T': 'logical_leap'}
SKETCHES, NEIGHBOURS = 14, 6
BATCH_NEIGHBOURS = 2  # in-batch references shown besides the bank neighbours (waves.py)
PROMPTS = HERE / 'prompts/v1/ideas'

STR = {'type': 'string', 'minLength': 1, 'maxLength': 4000}


def _obj(props):
    return {'type': 'object', 'additionalProperties': False, 'required': list(props), 'properties': props}


SKETCH_SCHEMA = _obj({'sketches': {'type': 'array', 'minItems': 12, 'maxItems': 16,
                                   'items': _obj({'id': STR, 'title': STR, 'idea': STR, 'recipe': STR})}})
KERNEL_FIELDS = ('source', 'central_relation', 'recipe', 'what_is_new', 'why_coherent',
                 'possible_test_generalization', 'main_authoring_risk')
COMMIT_SCHEMA = _obj({'sketch_verdicts': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': _obj(
                          {'id': STR, 'verdict': {'type': 'string', 'enum': ['keep', 'drop']}, 'closest_existing': STR})},
                      'kernels': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': _obj({k: STR for k in KERNEL_FIELDS})}})
KEPT_MAX = 6
COMMIT_KEPT_SCHEMA = _obj({'sketch_verdicts': {**COMMIT_SCHEMA['properties']['sketch_verdicts'], 'maxItems': 24},
                           'kernels': {'type': 'array', 'minItems': 0, 'maxItems': KEPT_MAX,
                                       'items': _obj({k: STR for k in KERNEL_FIELDS})}})


def parse_sketch(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    check(value, SKETCH_SCHEMA)
    ids = [s['id'] for s in value['sketches']]
    if len(ids) != len(set(ids)):
        raise SchemaError('$.sketches', 'sketch ids must be unique')
    return value


EXPLORE_SCHEMA = _obj({'sketches': {'type': 'array', 'minItems': 4, 'maxItems': 8,
                                     'items': _obj({'id': STR, 'title': STR, 'idea': STR, 'recipe': STR})}})


def parse_explore(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    check(value, EXPLORE_SCHEMA)
    return value


def explore_mode(call):
    return call.get('commit') == 'explore'


def parse_commit_kept(data):
    """Kept-draft commit: every kernel comes from a distinct sketch the author kept; nothing new is invented."""
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    check(value, COMMIT_KEPT_SCHEMA)
    kept = {v['id'] for v in value['sketch_verdicts'] if v['verdict'] == 'keep'}
    sources = [k['source'] for k in value['kernels']]
    if len(sources) != len(set(sources)) or set(sources) - kept:
        raise SchemaError('$.kernels', 'each kernel must come from a distinct kept sketch; new ideas are not allowed here')
    return value


def kept_mode(call):
    """Kept-draft commit (also used after an explore round): commit every kept, bank-checked draft; invent nothing."""
    return call.get('commit') in ('kept', 'explore')


def parse_commit(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    check(value, COMMIT_SCHEMA)
    return value


def stage_bindings():
    purposes = tuple(sorted(set(IDEA_PURPOSES.values())))
    return {STAGE_SKETCH: StageBinding(_transport_schema(SKETCH_SCHEMA), parse_sketch, purposes),
            STAGE_COMMIT: StageBinding(_transport_schema(COMMIT_SCHEMA), parse_commit, purposes),
            STAGE_COMMIT_KEPT: StageBinding(_transport_schema(COMMIT_KEPT_SCHEMA), parse_commit_kept, purposes),
            STAGE_EXPLORE: StageBinding(_transport_schema(EXPLORE_SCHEMA), parse_explore, purposes)}


def skill_call(call):
    """A general call that drew a skill line in a run judging novelty by skill (2026-10-18): its explore and step 3 use the
    skill variants, where sharing a common skill, or using it plainly, is not redundancy (prompts/v1/ideas/*_skill.md).
    A weakness line (2026-10-20) is judged the same way, with property-worded variants (*_line.md) or its own
    (weakness_lines.json: SETTLE's step 3 allows a familiar reading together with the visible condition that breaks it)."""
    return bool(SCIENCE.get('skill_novelty') and not call.get('family') and set(call.get('idea_modifiers') or ()) & set(NOVELTY_LINES))


def line_call(call):
    """The weakness line of a skill-scoped call, or None."""
    found = sorted(set(call.get('idea_modifiers') or ()) & set(LINE_MODIFIERS)) if skill_call(call) else []
    return found[0] if found else None


def source_class(arm):
    return SOURCE_CLASS.get(arm, 'ordinary_unconditioned')


def _structure_idea(structure):
    """A structure line's idea-time text (2026-10-22, STRUCTURE_SPEC S5: SHAPE is drawn at idea time), or None."""
    if not structure:
        return None
    entry = (LINE_REGISTRY.get('structures') or {}).get(structure)
    if entry is None:
        raise IntegrityError('unknown structure line: ' + str(structure))
    return HERE / 'prompts/v1' / entry['idea'] if entry.get('idea') else None


def _base(arm, modifiers=(), family=None, rung=None, structure=None):
    if arm not in ARMS or set(modifiers) - set(MODIFIERS):
        raise IntegrityError('arm or modifier is not part of the v1 idea stage: ' + arm + ' ' + str(modifiers))
    if (family is None) != (rung is None) or (family is not None and (family not in FAMILIES or rung not in RUNGS)):
        raise IntegrityError('a targeted idea call names both a known family and a known rung')
    # With any modifier, a short preamble asks that each be met inside the idea's one relation and explained in
    # why_coherent, so a drawn property shapes the idea instead of being bolted on as an extra step.
    # A targeted call (2026-10-12) adds its family's text and its rung's idea line after the arm card, and its own
    # requirements (prompts/v1/ideas/families/, rungs/); untargeted calls render exactly as before. In a run with a
    # departure draw (2026-10-16), the family requirements also ask the kernel to name a visible unshown test situation.
    idea = _structure_idea(structure)  # a structure line's idea text joins the modifiers (absent: unchanged)
    paths = ([PROMPTS / 'intro.md', PROMPTS / 'cards' / (arm + '.md')]
             + ([PROMPTS / 'families' / (family + '.md'), PROMPTS / 'rungs' / (rung + '.md')] if family else [])
             + ([PROMPTS / 'modifier_preamble.md'] if modifiers or idea else [])
             + [PROMPTS / 'modifiers' / (m + '.md') for m in modifiers]
             + ([idea] if idea else [])
             + [PROMPTS / (('requirements_family_depart.md' if SCIENCE.get('depart_draw') else 'requirements_family.md')
                          if family else 'requirements.md')])
    parts = [component(p) for p in paths]
    return paths, parts, '\n\n'.join(p[0] for p in parts)


def _call_base(call):
    return _base(call['arm'], call.get('idea_modifiers', ()), call.get('family'), call.get('rung'), call.get('structure'))


def sketch_request(call):
    """Turn 1, fully determined by the frozen call; frozen with the window like any idea request."""
    if call.get('family') and not explore_mode(call):
        raise IntegrityError('targeted idea calls use the explore loop')
    paths, parts, base = _call_base(call)
    step = component(PROMPTS / ('step1_explore.md' if explore_mode(call) else 'step1.md'))
    prompt = render_prompt(base + '\n\n' + step[0], {'SKETCH_COUNT': str(SKETCHES)})
    size = FactoryContextGuard(model=call['idea_model']).inspect(prompt, 'idea_search')
    cls = source_class(call['arm'])
    return {'purpose': IDEA_PURPOSES[cls], 'stage': STAGE_SKETCH, 'call_id': call['opaque_id'] + '-sketch',
            'prompt': prompt, 'prompt_template_version': '+'.join(p[1] for p in parts + [step]),
            'prompt_template_hash': digest({'components': [p[2] for p in parts + [step]]}), 'parent_ids': [],
            'private': {'source_class': cls, 'base_search': SEARCH_MODE, 'conditioned': False, 'bank_setting': BANK_SETTING,
                        'bank_pack_hash': None, 'search_mode': SEARCH_MODE, 'calibration_descriptions': None,
                        'calibration_source_call': None, 'effort': call['effort'], 'account': call['account'],
                        'recipe_id': call['recipe_id'], 'idea_model': call['idea_model'],
                        'comparison_block': call['comparison_block'], 'pinning': 'none', 'author_pack_base_hash': None,
                        'pin_manifest_hash': None, 'component_sources': [ref(p) for p in paths + [PROMPTS / 'step1.md']],
                        **({'idea_modifiers': list(call['idea_modifiers'])} if 'idea_modifiers' in call else {}),
                        'values_hash': digest({'SKETCH_COUNT': SKETCHES}), 'context_check': size}}


def neighbours(index, sketches):
    """Each sketch's nearest accepted bank tasks from the run's frozen reference index (identical texts once)."""
    texts = [f"{s['title']}. {s['idea']} Recipe: {s['recipe']}" for s in sketches]
    out = []
    for hits in index.rankings(texts):
        seen, rows, bank = set(), [], 0
        for rid, score in hits:
            record = index.records[rid]
            if record['source'] != 'accepted_generated_task':
                continue
            batch = rid.startswith('batch:')
            if batch:  # selected earlier in this batch (waves.py): up to two, on top of the bank's own slots
                if sum(r['reference_id'].startswith('batch:') for r in rows) == BATCH_NEIGHBOURS:
                    continue
                text = '(selected earlier in this batch) ' + record['central_relation'] + ' ' + record['summary']
            else:
                text = record.get('summary') or record['central_relation']
            if text in seen:
                continue
            seen.add(text); bank += not batch
            rows.append({'reference_id': rid, 'similarity': round(score, 4), 'text': text[:400]})
            if bank == NEIGHBOURS:
                break
        out.append(rows)
    return out


def _blocks(sketches, near):
    blocks = []
    for s, rows in zip(sketches, near):
        lines = [f"### {s['id']}: {s['title']}", f"Idea: {s['idea']}", f"Recipe: {s['recipe']}", 'Most similar existing tasks:']
        lines += [f"- [{r['reference_id'].removeprefix('committed:')}] {r['text']}" for r in rows]
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks)


def explore_request(call, sketch_req, sketches, near):
    """Turn 2 of an explore call: see where the first sketches landed, then sketch beyond them."""
    paths, parts, base = _call_base(call)
    # A targeted call's explore turn stays inside its family: new ground comes from what the solver must notice there.
    line = line_call(call)  # a weakness line's own variant (weakness_lines.json), else the property-worded one
    step = component(PROMPTS / ('step2_explore_family.md' if call.get('family') else
                                LINE_REGISTRY['lines'][line].get('explore', 'step2_explore_line.md') if line
                                else 'step2_explore_skill.md' if skill_call(call) else 'step2_explore.md'))
    prompt = render_prompt(base + '\n\n' + step[0], {'SKETCH_BLOCKS': _blocks(sketches, near), 'EXPLORE_COUNT': str(EXPLORE_COUNT)})
    req = deepcopy(sketch_req)
    req.update(stage=STAGE_EXPLORE, call_id=call['opaque_id'] + '-explore', prompt=prompt, parent_ids=[sketch_req['call_id']],
               prompt_template_version='+'.join(p[1] for p in parts + [step]),
               prompt_template_hash=digest({'components': [p[2] for p in parts + [step]]}))
    req['private']['loop'] = {'sketch_hash': digest(sketches), 'neighbour_hash': digest(near)}
    return req


def commit_request(call, sketch_req, sketches, near):
    paths, parts, base = _call_base(call)
    line = line_call(call)
    step = component(PROMPTS / ('step3_kept_family.md' if call.get('family') else
                                (LINE_REGISTRY['lines'][line].get('step3', 'step3_kept_line.md') if line
                                 else 'step3_kept_skill.md' if skill_call(call) else 'step3_kept.md') if explore_mode(call)
                                else 'step2_kept.md' if kept_mode(call) else 'step2.md'))
    prompt = render_prompt(base + '\n\n' + step[0], {'SKETCH_BLOCKS': _blocks(sketches, near)})
    req = deepcopy(sketch_req)
    req.update(stage=STAGE_COMMIT_KEPT if kept_mode(call) else STAGE_COMMIT, call_id=call['opaque_id'] + '-commit', prompt=prompt, parent_ids=[sketch_req['call_id']],
               prompt_template_version='+'.join(p[1] for p in parts + [step]),
               prompt_template_hash=digest({'components': [p[2] for p in parts + [step]]}))
    req['private']['loop'] = {'sketch_hash': digest(sketches), 'neighbour_hash': digest(near)}
    return req


def as_ideas(call, commit):
    """The committed kernels in the batch's idea-response shape, under the call's frozen kernel IDs."""
    kernels = []
    for key, k in zip(call['opaque_kernel_ids'], commit['kernels']):
        kernels.append({'candidate_id': key, 'search_mode': SEARCH_MODE, **{f: k[f] for f in KERNEL_FIELDS if f != 'source'},
                        'loop_source': k['source']})
    return {'search_call_id': call['opaque_id'], 'search_mode': SEARCH_MODE, 'ideas': kernels}


def validate_ideas(call, value):
    if (value['search_call_id'] != call['opaque_id'] or value['search_mode'] != SEARCH_MODE
            or [k['candidate_id'] for k in value['ideas']] != call['opaque_kernel_ids'][:len(value['ideas']) if kept_mode(call) else None]
            or any(not k['central_relation'].strip() for k in value['ideas'])):
        raise IntegrityError('v1 idea response changed identity or lacks a relation')


def run_loop(rt, call, request, index, invoke_call, directory):
    """Both turns for one idea call; each turn is an ordinary, resumable, budgeted model call."""
    first = invoke_call(rt, {k: v for k, v in request.items() if k != 'private'})
    if first is None:  # a terminal logical failure is data: this call contributes no ideas
        return None
    sketches = parse_sketch(first)['sketches']
    near = neighbours(index, sketches)
    if explore_mode(call):
        # Explore round: the author sees where its sketches landed and sketches beyond them; the new sketches get
        # their own nearest bank tasks before anything is committed, so nothing unchecked can be committed.
        explore_req = explore_request(call, request, sketches, near)
        frozen(directory / 'idea_loop' / (call['opaque_id'] + '.private.json'),
               {'call': call['opaque_id'], 'sketches': sketches, 'neighbours': near,
                'explore_request_hash': digest({k: v for k, v in explore_req.items() if k != 'private'})})
        explored = invoke_call(rt, {k: v for k, v in explore_req.items() if k != 'private'})
        if explored is None:
            return None
        extra = [{**s, 'id': f'x{i:02d}'} for i, s in enumerate(parse_explore(explored)['sketches'], 1)]
        extra_near = neighbours(index, extra)
        second_req = commit_request(call, request, sketches + extra, near + extra_near)
        frozen(directory / 'idea_loop' / (call['opaque_id'] + '.explore.private.json'),
               {'call': call['opaque_id'], 'sketches': extra, 'neighbours': extra_near,
                'commit_request_hash': digest({k: v for k, v in second_req.items() if k != 'private'})})
        return _commit(rt, call, second_req, invoke_call, directory)
    second_req = commit_request(call, request, sketches, near)
    frozen(directory / 'idea_loop' / (call['opaque_id'] + '.private.json'),
           {'call': call['opaque_id'], 'sketches': sketches, 'neighbours': near,
            'commit_request_hash': digest({k: v for k, v in second_req.items() if k != 'private'})})
    return _commit(rt, call, second_req, invoke_call, directory)


def _commit(rt, call, second_req, invoke_call, directory):
    second = invoke_call(rt, {k: v for k, v in second_req.items() if k != 'private'})
    if second is None:
        return None
    second = parse_commit_kept(second) if kept_mode(call) else parse_commit(second)
    value = as_ideas(call, second)
    frozen(directory / 'idea_loop' / (call['opaque_id'] + '.committed.private.json'),
           {'verdicts': second['sketch_verdicts'], 'ideas': value})
    return value
