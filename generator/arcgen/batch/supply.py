"""Orthogonal recipes and source-hidden selector requests, with complete resume pins."""
from copy import deepcopy
from .routing import IDEA_EFFORTS, ASTRA
from pathlib import Path
from random import Random
from arcgen.lib.factory_supply import IDEA_PURPOSES
from arcgen.lib.e16_bank_context_tuning import parse_e16_stage
from arcgen.lib.factory_runtime import PLACEHOLDER_RE
from arcgen.lib.factory_context import FactoryContextGuard
from arcgen.lib.util import render_prompt, load_prompt
from arcgen.lib.v4_reservoir import concept_hash
from .common import (CARDS, HERE, ROOT, SOURCE, SETTINGS, REVISION, IntegrityError,
                     read, ref, digest, component, HOMES)
from arcgen.astra_usage_diversity_20260913 import selection
from .retrieval import check_size, over_budget, public_reference

ARMS = {a['id']: a for a in read(CARDS / 'catalog.proposed.json')['arms']}
# Invitation -> host arm (fixed by the arm/variant review of 2026-09-10; inlined, the review file is not shipped).
HOSTS = {k: v for k, v in {'PER': 'ACC', 'GEO': 'REF_STYLE', 'REL': 'ACC', 'FIELD': 'EXP', 'SEM': 'DISC', 'CAUSE': 'MANY', 'ANA': 'XFER', 'INV': 'ABS', 'NUM': 'COMPACT', 'FORM': 'EXEC', 'MAKE': 'REF_STYLE', 'PAT': 'D1'}.items() if k != 'FORM'}
INVITATION_HOSTS = {key: {host} for key, host in HOSTS.items()}
for arm in ARMS.values():
    for invitation in arm.get('additional_invitations', []):
        INVITATION_HOSTS[invitation].add(arm['id'])
from .config import SCIENCE, FAMILIES, RUNGS, ALL_RUNGS, SKILL_MODIFIERS
from . import idea_v1
V1_IDEAS = SCIENCE.get('idea_stage', 'r23') == 'v1_loop'
BANK_SETTINGS = ('unconditioned', 'bank_aware', 'bank_inspired') + (('loop',) if V1_IDEAS else ())


def valid_invitation(arm, invitation):
    return invitation is None or arm in INVITATION_HOSTS.get(invitation, set())


def make_call(*, arm, invitation=None, bank_setting, context, seed, account, window,
              replicate=1, effort=None, idea_model=ASTRA, comparison_block=None, comparison_role=None, family=None, rung=None):
    if (family is None) != (rung is None) or (family is not None and (not V1_IDEAS or family not in FAMILIES or rung not in RUNGS)):
        raise IntegrityError('a targeted call names both a known family and a known rung, with the v1 idea loop')
    if V1_IDEAS:
        if arm not in idea_v1.ARMS or bank_setting != 'loop' or invitation is not None:
            raise IntegrityError('v1 idea calls use a concept arm, the loop and no invitation')
    elif arm not in ARMS or bank_setting not in BANK_SETTINGS:
        raise IntegrityError('unknown arm or bank setting')
    if not V1_IDEAS and not valid_invitation(arm, invitation):
        raise IntegrityError('invitation is outside its explicit local host mapping')
    if account not in HOMES:
        raise IntegrityError('unbound account')
    if idea_model not in IDEA_EFFORTS:
        raise IntegrityError('unbound idea model')
    effort = effort or 'xhigh'
    if effort not in IDEA_EFFORTS.get(idea_model,()):
        raise IntegrityError('unsupported idea effort')
    recipe = {'revision': REVISION, 'arm': arm, 'invitation': invitation,
              'bank_setting': bank_setting, 'effort': effort, 'idea_model': idea_model,
              **({'family': family, 'rung': rung} if family else {})}  # absent for untargeted calls: their IDs are unchanged
    key = {**recipe, 'context': context, 'seed': seed, 'account': account,
           'window': window, 'replicate': replicate}
    opaque = 'p' + digest(key)[:18]
    extra = {}
    if V1_IDEAS and SCIENCE.get('idea_modifiers'):  # per-call draws, independent of each other and of the call identity
        extra['idea_modifiers'] = sorted(m for m, rate in SCIENCE['idea_modifiers'].items()
                                         if int(digest({'policy': 'idea-modifier-v1', 'modifier': m, 'call': opaque})[:12], 16) / 16 ** 12 < rate)
    return {**key, **extra, 'private_id': 'call-' + digest(key)[:20], 'opaque_id': opaque,
            'opaque_kernel_ids': ['k' + digest({'call': key, 'kernel': i})[:18] for i in range(4)],
            'recipe_id': 'recipe-' + digest(recipe)[:20],
            'comparison_block': comparison_block, 'comparison_role': comparison_role}


def idea_request(call, context, inputs, *, bank_pack=None):
    """The frozen first-turn request of one idea call. Only the v1 idea loop (bank_setting "loop") is supported: its
    later turns are built during the run from the sketches and their nearest bank tasks (idea_v1.run_loop)."""
    if call['bank_setting'] != 'loop':
        raise IntegrityError('only the v1 idea loop (bank_setting "loop") is supported: ' + str(call['bank_setting']))
    if bank_pack is not None:
        raise IntegrityError('the v1 loop takes no bank pack')
    return idea_v1.sketch_request(call)


def idea_validator(call, request):
    if request['stage'] == idea_v1.STAGE_SKETCH:
        return lambda value: idea_v1.validate_ideas(call, value)
    def validate(value):
        parse_e16_stage('e16_idea_search', value)
        if value['search_call_id'] != call['opaque_id'] or value['search_mode'] != request['private']['search_mode'] or [k['candidate_id'] for k in value['ideas']] != call['opaque_kernel_ids']:
            raise ValueError('idea response changed identity, four kernels or mode')
        if any(k['search_mode'] != value['search_mode'] for k in value['ideas']):
            raise ValueError('kernel search mode changed')
    return validate


def source_rows(window, requests, responses):
    rows = {}
    for call in window['calls']:
        value = responses.get(call['opaque_id'])
        if value is None: continue
        req = requests[call['opaque_id']]; idea_validator(call, req)(value)
        for kernel in value['ideas']:
            key = kernel['candidate_id']
            if key in rows: raise IntegrityError('duplicate source candidate ID')
            rows[key] = {'candidate_id': key, 'kernel': deepcopy(kernel),
                'private_source': req['private']['source_class'], 'private_conditioned': req['private']['conditioned'],
                'private_source_call_id': call['opaque_id'], 'private_effort': call['effort'],
                'study': {'primary_arm': call['arm'], 'invitation': call.get('invitation'), 'recipe_id': call['recipe_id'],
                    'bank_setting': call['bank_setting'], 'bank_pack_hash': req['private']['bank_pack_hash'],
                    'base_search': req['private']['base_search'], 'context': window['context'], 'window': window['id'],
                    'account': call['account'], 'idea_call_private_id': call['private_id'], 'replicate': call['replicate'],
                    'target_profile': ARMS.get(call['arm'], {}).get('target_profile'),
                    'comparison_parent': ARMS.get(call['arm'], {}).get('comparison_parent'),
                    'bank_parent_arms': ARMS.get(call['arm'], {}).get('bank_parent_arms', []),
                    'cohort': window.get('cohort', 'usage_diversity'), 'wave': window.get('wave'), 'generation_model': call['idea_model'],
                    **{k:deepcopy(req['private'][k]) for k in ('pinning','author_pack_base_hash','pin_manifest_hash')},
                    'comparison_block': call['comparison_block'], 'comparison_role': call['comparison_role'],
                    **({'idea_modifiers': list(call['idea_modifiers'])} if 'idea_modifiers' in call else {}),
                    **({'family': call['family'], 'rung': call['rung']} if call.get('family') else {}),
                    # weakness run (2026-10-20): a matched-control call, the call's size, relaxed fence and decoy flag;
                    # absent elsewhere
                    **({'line_control': True} if call.get('line_control') else {}),
                    **(size_study(call['size_arm']) if call.get('size_arm') else {}),
                    **({'fence_call': call['fence']} if call.get('fence') else {}),
                    **({'judge_decoy': True} if call.get('judge_decoy') else {}),
                    # structure lines (2026-10-22): the call's structure line, applied at admission (host matrix)
                    **({'structure_call': call['structure']} if call.get('structure') else {}),
                    **({'structure_control': True} if call.get('structure_control') else {}),
                    # ONE (2026-10-23): a slot's matched control, with the slot's exclusions and presentation (slots.py)
                    **({'slot_control': call['slot_control']} if call.get('slot_control') else {})}}
    return rows


def size_study(size):
    """A weakness call's fixed size (spec row {"size": ...}): its arm, and its rung when the row names one."""
    if size in ('small', 'large'):
        return {'size_arm': size}
    if size not in ALL_RUNGS:
        raise IntegrityError('unknown call size: ' + str(size))
    return {'size_arm': 'small' if size == ALL_RUNGS[0] else 'large', 'size_rung': size}


def window_kernel_ids(window):
    """The candidate IDs a selector window may hold: its idea calls' kernels, or a saved-ideas window's frozen list."""
    if window.get('kind') == 'saved_ideas':
        if window['calls']: raise IntegrityError('saved ideas windows have no idea calls')
        return list(window['saved_kernel_ids'])
    return [k for c in window['calls'] for k in c['opaque_kernel_ids']]


def exact_partition(window, source, base, pinned):
    memory = base['selector_memory']
    exact = set(base['accepted_concept_hashes'])
    exact.update(r.get('concept_hash') or concept_hash(r['mechanism_summary']) for r in memory['active_reservations'])
    exact.update(concept_hash(r['mechanism_summary']) for r in pinned['in_flight_reservations'])
    exact.update(r.get('concept_hash') or concept_hash(r['mechanism_summary']) for r in
                 [*memory['tiered_rejection_memory'], *memory['quarantine_failure_mode_warnings']]
                 if r.get('active', True) and r['tier'] == 'hard_duplicate')
    ids = sorted(window_kernel_ids(window))
    if len(ids) != len(set(ids)) or len(ids) > SETTINGS['selector']['maximum_candidates'] or set(source) - set(ids):
        raise IntegrityError('window membership outside the fixed source IDs')
    Random(window['seed'] + 180169).shuffle(ids)
    ids = [k for k in ids if k in source]
    conflicts = {k: 'exact normalized relation already accepted, reserved or hard duplicate'
                 for k in ids if concept_hash(source[k]['kernel']['central_relation']) in exact}
    return ids, conflicts


def _identity(window, source, base, pinned, extra):
    value = {'window': window, 'source': source, 'base': base, 'memory': pinned, 'settings': SETTINGS, 'revision': REVISION}
    if extra:  # in-batch references of a later wave (waves.py); absent, the identity is unchanged
        value['batch_references'] = digest(list(extra))
    return digest(value)


def selector_inputs(window, source, inputs, index):
    base = read(inputs['selector_context'])
    pinned = read(inputs['contexts'][window['context']]['selector_memory'])['payload']
    extra = list(getattr(index, 'batch_references', ()))
    expected_index = digest(sorted(base['references'] + extra, key=lambda r: r['reference_id']))
    if index.snapshot_hash != expected_index:
        raise IntegrityError('retrieval index does not cover the frozen full reference corpus')
    ids, conflicts = exact_partition(window, source, base, pinned)
    eligible = [k for k in ids if k not in conflicts]
    enough = len(eligible) >= window['minimum_selector_eligible_kernels']
    kernels = {k: source[k]['kernel'] for k in eligible}
    top, floor = SETTINGS['selector']['accepted_neighbors_per_candidate'], SETTINGS['selector']['minimum_accepted_neighbors']

    def pack(retrieved, depth=top):
        packages = {k: {'kernel': deepcopy(source[k]['kernel']), **retrieved['candidate_references'][k],
                        'staleness': {'status': 'current', 'reasons': []}} for k in eligible}
        payload = {'proposal_ids': eligible, 'packages': packages, 'shared_references': retrieved['references'],
                   'in_flight_reservations': deepcopy(pinned['in_flight_reservations']),
                   'rejected_memory_digest': deepcopy(pinned['rejected_memory_digest'])}
        packed = {'raw_proposal_ids': ids, 'exact_conflicts': conflicts, 'selection_eligible': enough,
                  'source_hidden': True, 'payload': payload, 'private_retrieval': retrieved['private_retrieval'],
                  'input_identity': _identity(window, source, base, pinned, extra)}
        if depth < top:  # recorded before hashing, so the prompt measured below is the one the selector is sent
            packed['context_reduction'] = {'accepted_k': depth, 'from': top, 'reason': 'complete selector prompt over the frozen budget at higher k'}
        packed['packed_hash'] = digest(packed)
        return packed
    if not enough:
        return pack({'references': [], 'candidate_references': {k: {'accepted_reference_ids': [], 'real_reference_ids': []} for k in eligible},
                     'private_retrieval': {'snapshot_hash': index.snapshot_hash, 'provider': index.provider_identity, 'not_dispatched': True}})
    # A window whose complete prompt would exceed the budget gets fewer accepted neighbours per candidate, one at a
    # time down to the floor, never a truncated text. At the floor the selector request still refuses (check_size).
    rankings = index.rankings([kernels[k]['central_relation'] for k in eligible])
    for depth in range(top, floor - 1, -1):
        packed = pack(index.pack(kernels, accepted_k=depth, rankings=rankings), depth)
        if depth == floor or not over_budget(*_selector_prompt(window, packed)[:3]):
            return packed


def verify_selector_inputs(window, source, inputs, packed, extra=(), index=None):
    if packed['packed_hash'] != digest({k: v for k, v in packed.items() if k != 'packed_hash'}):
        raise IntegrityError('saved selector input changed')
    base = read(inputs['selector_context']); pinned = read(inputs['contexts'][window['context']]['selector_memory'])['payload']
    wanted = _identity(window, source, base, pinned, list(extra))
    if packed['input_identity'] != wanted:
        raise IntegrityError('selector input belongs to a different frozen window/corpus/configuration')
    ids, conflicts = exact_partition(window, source, base, pinned)
    p = packed['payload']; eligible = [k for k in ids if k not in conflicts]
    if packed['raw_proposal_ids'] != ids or packed['exact_conflicts'] != conflicts or p['proposal_ids'] != eligible:
        raise IntegrityError('exact-conflict partition or order changed')
    if set(p['packages']) != set(eligible) or any(p['packages'][k]['kernel'] != source[k]['kernel'] for k in eligible):
        raise IntegrityError('selector kernel membership/content changed')
    if any(p[k] != pinned[k] for k in ('in_flight_reservations', 'rejected_memory_digest')):
        raise IntegrityError('frozen memory changed')
    if packed['selection_eligible'] != (len(eligible) >= window['minimum_selector_eligible_kernels']):
        raise IntegrityError('eligibility threshold changed')
    top = SETTINGS['selector']['accepted_neighbors_per_candidate']
    accepted_k = packed['context_reduction']['accepted_k'] if 'context_reduction' in packed else top
    if (type(accepted_k) is not int or not SETTINGS['selector']['minimum_accepted_neighbors'] <= accepted_k <= top
            or ('context_reduction' in packed and (accepted_k == top or packed['context_reduction']['from'] != top
                                                  or packed['private_retrieval'].get('accepted_k') != accepted_k))):
        raise IntegrityError('selector neighbour reduction outside its frozen floor')
    if 'context_reduction' in packed:  # a reduced window must be exactly what the reduction yields (cached embeddings)
        if index is None or selector_inputs(window, source, inputs, index)['packed_hash'] != packed['packed_hash']:
            raise IntegrityError('selector neighbour reduction does not reproduce')
    refs = {r['reference_id']: r for r in base['references'] + list(extra)}
    shared = {r['reference_id']: r for r in p['shared_references']}
    if len(shared) != len(p['shared_references']): raise IntegrityError('duplicate shared record')
    for rid, row in shared.items():
        if rid not in refs or row != public_reference(refs[rid]):
            raise IntegrityError('shared comparator text changed')
    linked = set()
    for pkg in p['packages'].values():
        for kind, field, cap in [('accepted_generated_task', 'accepted_reference_ids', accepted_k),
                                 ('real_task', 'real_reference_ids', SETTINGS['selector']['real_neighbors_per_candidate'])]:
            values = pkg[field]
            expected = min(cap, sum(r['source'] == kind for r in refs.values())) if packed['selection_eligible'] else 0
            if len(values) != expected or len(set(values)) != len(values) or any(k not in shared or shared[k]['source'] != kind for k in values):
                raise IntegrityError('candidate-specific reference coverage/identity changed')
            linked.update(values)
    if linked != set(shared): raise IntegrityError('unlinked comparator in shared context')
    return packed


def _selector_prompt(window, packed):
    p = packed['payload']; sid = 's' + digest({'revision': REVISION, 'window': window['id'], 'input': packed['packed_hash']})[:18]
    # A targeted run's selector adds one sentence: familiar laws meeting in a new way are a fresh insight, a reference's
    # rule run on a larger or busier scene is not (prompts/v1/selector_targeted.md; tests_v1 checks it is the only change).
    # A skill run's selector (2026-10-18) is the targeted text plus one passage: sharing a common skill is not redundancy,
    # novelty is judged on what the solver reads and decides in the scene, and a better-taught copy stays redundant.
    # A weakness run's selector (2026-10-20) is the targeted text plus one passage on the batch's properties instead.
    # A weakness run that also pins skill lines (2026-10-25, the composite batch c1) carries both passages, the weakness
    # passage first (prompts/v1/selector_composite.md; tests_v1/test_c1_skills.py checks it is exactly that).
    composite = SCIENCE.get('weakness') and set(SKILL_MODIFIERS) & set(SCIENCE.get('idea_modifiers') or {})
    name = (('prompts/v1/selector_composite.md' if composite else 'prompts/v1/selector_weakness.md' if SCIENCE.get('weakness')
             else 'prompts/v1/selector_skill.md')
            if SCIENCE.get('skill_novelty') else 'prompts/v1/selector_targeted.md'
            if SCIENCE.get('targeted') else 'prompts/v1/selector.md')
    template, version, h = load_prompt(HERE / name)
    prompt = render_prompt(template, {'CANDIDATES_WITH_REFERENCES': [p['packages'][k] for k in p['proposal_ids']],
        'SHARED_REFERENCES': p['shared_references'], 'IN_FLIGHT_RESERVATIONS': p['in_flight_reservations'],
        'REJECTED_MEMORY_DIGEST': p['rejected_memory_digest'], 'SELECTOR_ID': sid})
    return prompt, p['shared_references'], selection.stage_bindings(packed)[selection.STAGE].schema, sid, version, h


def selector_request(window, packed):
    p = packed['payload']
    prompt, _, schema, sid, version, h = _selector_prompt(window, packed)
    check_size(prompt, p['shared_references'], schema)
    def validate(value):
        selection.parse(value, candidate_ids=p['proposal_ids'], reference_ids=[r['reference_id'] for r in p['shared_references']])
        if value['selector_id'] != sid: raise ValueError('selector changed its opaque ID')
    return {'purpose': 'full_window_selection', 'stage': selection.STAGE, 'call_id': sid, 'prompt': prompt,
            'prompt_template_version': version, 'prompt_template_hash': h, 'parent_ids': p['proposal_ids'], 'response_validator': validate}
