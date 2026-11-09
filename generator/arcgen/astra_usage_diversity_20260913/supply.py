"""Orthogonal recipes and source-hidden selector requests, with complete resume pins."""
from copy import deepcopy
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
from . import selection
from .retrieval import check_size, public_reference

ARMS = {a['id']: a for a in read(CARDS / 'catalog.proposed.json')['arms']}
# Invitation -> host arm (fixed by the arm/variant review of 2026-09-10; inlined, the review file is not shipped).
HOSTS = {k: v for k, v in {'PER': 'ACC', 'GEO': 'REF_STYLE', 'REL': 'ACC', 'FIELD': 'EXP', 'SEM': 'DISC', 'CAUSE': 'MANY', 'ANA': 'XFER', 'INV': 'ABS', 'NUM': 'COMPACT', 'FORM': 'EXEC', 'MAKE': 'REF_STYLE', 'PAT': 'D1'}.items() if k != 'FORM'}
INVITATION_HOSTS = {key: {host} for key, host in HOSTS.items()}
for arm in ARMS.values():
    for invitation in arm.get('additional_invitations', []):
        INVITATION_HOSTS[invitation].add(arm['id'])
BANK_SETTINGS = ('unconditioned', 'bank_aware', 'bank_inspired')


def valid_invitation(arm, invitation):
    return invitation is None or arm in INVITATION_HOSTS.get(invitation, set())


def make_call(*, arm, invitation=None, bank_setting, context, seed, account, window,
              replicate=1, effort=None, idea_model='gpt-6-astra', comparison_block=None, comparison_role=None):
    if arm not in ARMS or bank_setting not in BANK_SETTINGS:
        raise IntegrityError('unknown arm or bank setting')
    if not valid_invitation(arm, invitation):
        raise IntegrityError('invitation is outside its explicit local host mapping')
    if account not in HOMES:
        raise IntegrityError('unbound account')
    if idea_model not in ('gpt-6-astra', 'gpt-5.6-sol'):
        raise IntegrityError('unbound idea model')
    effort = effort or 'xhigh'
    if effort not in ('high', 'xhigh', 'max'):
        raise IntegrityError('unsupported idea effort')
    recipe = {'revision': REVISION, 'arm': arm, 'invitation': invitation,
              'bank_setting': bank_setting, 'effort': effort, 'idea_model': idea_model}
    key = {**recipe, 'context': context, 'seed': seed, 'account': account,
           'window': window, 'replicate': replicate}
    opaque = 'p' + digest(key)[:18]
    return {**key, 'private_id': 'call-' + digest(key)[:20], 'opaque_id': opaque,
            'opaque_kernel_ids': ['k' + digest({'call': key, 'kernel': i})[:18] for i in range(4)],
            'recipe_id': 'recipe-' + digest(recipe)[:20],
            'comparison_block': comparison_block, 'comparison_role': comparison_role}


def idea_request(call, context, inputs, *, bank_pack=None):
    arm, setting = call['arm'], call['bank_setting']
    if arm not in ARMS or setting not in BANK_SETTINGS or not valid_invitation(arm, call.get('invitation')):
        raise IntegrityError('unbound recipe composition')
    b = context['bank_memory_and_specialists']
    if arm in ('J', 'T'):
        spec = b['specialists'][arm]
        mode, calibration, source_call = spec['search_mode'], spec['calibration_descriptions'], spec['call_id']
    else:
        mode, calibration = context['search_mode'], context['calibration_descriptions']
        source_call = b['matched_ordinary_context']['conditioned_payload_source_call_id']
    conditioned = setting != 'unconditioned'
    source_class = {'J': 'episode_joint_inference', 'T': 'logical_leap'}.get(
        arm, 'ordinary_bank_conditioned' if conditioned else 'ordinary_unconditioned')
    policy = 'idea_' + {'open': 'open', 'concept_led': 'concept', 'composition_led': 'composition'}[mode] + '.proposed.md'
    if mode == 'composition_led' and setting == 'bank_inspired': policy = 'idea_composition_inspired.md'
    paths = [CARDS / 'prompts/policy' / policy]
    if arm in ('J', 'T'):
        paths.append(SOURCE / 'prompts' / ('v4_3_episode_joint_idea_addendum.md' if arm == 'J' else 'e17_logical_leap_idea_addendum.md'))
    paths.append(SOURCE / 'prompts/e16_idea_nomination_addendum.md')
    if conditioned:
        paths.append(SOURCE / 'prompts/e16_bank_diversity_addendum.md' if setting == 'bank_aware' else CARDS / 'prompts/bank_inspired.md')
    # Base search is independent of bank visibility and transport purpose.
    base_search = 'specialist' if arm in ('J', 'T') else 'ordinary' if arm == 'C' else 'i1_diverge_discard_commit'
    if base_search == 'i1_diverge_discard_commit': paths.append(SOURCE / 'prompts/e20_u_i1_diverge_discard_commit.md')
    if ARMS[arm].get('idea_card'): paths.append(CARDS / ARMS[arm]['idea_card'])
    if call.get('invitation'): paths.append(CARDS / 'prompts/families' / (call['invitation'] + '.proposed.md'))
    paths.append(CARDS / 'prompts/coverage_guidance.md')
    if arm != 'FORM': paths.append(CARDS / 'prompts/ordinary_territory.idea.md')
    parts = [component(p) for p in paths]
    template = '\n\n'.join(p[0] for p in parts)
    values = {'CALIBRATION_DESCRIPTIONS': calibration, 'SEARCH_CALL_ID': call['opaque_id'],
              'CANDIDATE_IDS': call['opaque_kernel_ids']}
    if conditioned:
        if (bank_pack is None or len(bank_pack['rows']) + len(bank_pack['provisional_rows']) != SETTINGS['author_bank']['entries']
                or bank_pack.get('private', {}).get('pinning') != 'pinned'):
            raise IntegrityError('conditioned recipe requires its frozen 400-reference pack')
        expected = {'arm': arm, 'invitation': call.get('invitation'), 'context': call['context']}
        if any(bank_pack['identity'][k] != v for k, v in expected.items()) or bank_pack['pack_hash'] != digest({'accepted': bank_pack['rows'], 'provisional': bank_pack['provisional_rows']}):
            raise IntegrityError('bank pack identity or content changed')
        values['ACCEPTED_CONCEPT_BANK'] = bank_pack['rows']
    elif bank_pack is not None:
        raise IntegrityError('unconditioned recipe must not receive a bank pack')
    if set(PLACEHOLDER_RE.findall(template)) != set(values):
        raise IntegrityError('idea prompt information boundary changed')
    prompt = render_prompt(template, values)
    if conditioned:
        pin_template = (HERE/'author_pins.md').read_text()
        prompt += '\n\n' + render_prompt(pin_template, {'PINNED_COVERAGE_REFERENCES': bank_pack['private']['visible_pins']})
        provisional_template = (CARDS/'prompts/provisional_coverage.md').read_text()
        prompt += '\n\n' + render_prompt(provisional_template, {'PROVISIONAL_SELECTED_CONCEPTS': bank_pack['provisional_rows']})
    size = FactoryContextGuard(model=call['idea_model']).inspect(prompt, 'idea_search')
    if size['estimated_prompt_tokens'] > SETTINGS['author_bank']['prompt_token_budget']:
        raise IntegrityError('author prompt exceeds its bounded context; no truncation')
    return {'purpose': IDEA_PURPOSES[source_class], 'stage': 'e16_idea_search', 'call_id': call['opaque_id'],
            'prompt': prompt, 'prompt_template_version': '+'.join(p[1] for p in parts) + ('+pinned-coverage-v1' if conditioned else ''),
            'prompt_template_hash': digest({'components': [p[2] for p in parts], 'template': template, 'pin_template': pin_template if conditioned else None, 'provisional_template': provisional_template if conditioned else None}),
            'parent_ids': [], 'private': {'source_class': source_class, 'base_search': base_search,
                'conditioned': conditioned, 'bank_setting': setting, 'bank_pack_hash': bank_pack['pack_hash'] if conditioned else None,
                'search_mode': mode, 'calibration_descriptions': calibration, 'calibration_source_call': source_call,
                'effort': call['effort'], 'account': call['account'], 'recipe_id': call['recipe_id'],
                'idea_model': call['idea_model'], 'comparison_block': call['comparison_block'],
                'pinning': 'pinned' if conditioned else 'unconditioned',
                'author_pack_base_hash': bank_pack['private']['base_pack_hash'] if conditioned else None,
                'pin_manifest_hash': bank_pack['private']['pin_manifest_hash'] if conditioned else None,
                'component_sources': [ref(p) for p in paths] + ([ref(HERE/'author_pins.md'), ref(CARDS/'prompts/provisional_coverage.md')] if conditioned else []), 'values_hash': digest(values), 'context_check': size}}


def idea_validator(call, request):
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
                    'target_profile': ARMS[call['arm']].get('target_profile'),
                    'comparison_parent': ARMS[call['arm']].get('comparison_parent'),
                    'bank_parent_arms': ARMS[call['arm']].get('bank_parent_arms', []),
                    'cohort': window.get('cohort', 'usage_diversity'), 'wave': window.get('wave'), 'generation_model': call['idea_model'],
                    **{k:deepcopy(req['private'][k]) for k in ('pinning','author_pack_base_hash','pin_manifest_hash')},
                    'comparison_block': call['comparison_block'], 'comparison_role': call['comparison_role']}}
    return rows


def exact_partition(window, source, base, pinned):
    memory = base['selector_memory']
    exact = set(base['accepted_concept_hashes'])
    exact.update(r.get('concept_hash') or concept_hash(r['mechanism_summary']) for r in memory['active_reservations'])
    exact.update(concept_hash(r['mechanism_summary']) for r in pinned['in_flight_reservations'])
    exact.update(r.get('concept_hash') or concept_hash(r['mechanism_summary']) for r in
                 [*memory['tiered_rejection_memory'], *memory['quarantine_failure_mode_warnings']]
                 if r.get('active', True) and r['tier'] == 'hard_duplicate')
    ids = sorted(k for c in window['calls'] for k in c['opaque_kernel_ids'])
    if len(ids) != len(set(ids)) or len(ids) > SETTINGS['selector']['maximum_candidates'] or set(source) - set(ids):
        raise IntegrityError('window membership outside the fixed source IDs')
    Random(window['seed'] + 180169).shuffle(ids)
    ids = [k for k in ids if k in source]
    conflicts = {k: 'exact normalized relation already accepted, reserved or hard duplicate'
                 for k in ids if concept_hash(source[k]['kernel']['central_relation']) in exact}
    return ids, conflicts


def selector_inputs(window, source, inputs, index):
    base = read(inputs['selector_context'])
    pinned = read(inputs['contexts'][window['context']]['selector_memory'])['payload']
    expected_index = digest(sorted(base['references'], key=lambda r: r['reference_id']))
    if index.snapshot_hash != expected_index:
        raise IntegrityError('retrieval index does not cover the frozen full reference corpus')
    ids, conflicts = exact_partition(window, source, base, pinned)
    eligible = [k for k in ids if k not in conflicts]
    enough = len(eligible) >= window['minimum_selector_eligible_kernels']
    retrieved = index.pack({k: source[k]['kernel'] for k in eligible}) if enough else {
        'references': [], 'candidate_references': {k: {'accepted_reference_ids': [], 'real_reference_ids': []} for k in eligible},
        'private_retrieval': {'snapshot_hash': index.snapshot_hash, 'provider': index.provider_identity, 'not_dispatched': True}}
    packages = {k: {'kernel': deepcopy(source[k]['kernel']), **retrieved['candidate_references'][k],
                    'staleness': {'status': 'current', 'reasons': []}} for k in eligible}
    payload = {'proposal_ids': eligible, 'packages': packages, 'shared_references': retrieved['references'],
               'in_flight_reservations': deepcopy(pinned['in_flight_reservations']),
               'rejected_memory_digest': deepcopy(pinned['rejected_memory_digest'])}
    packed = {'raw_proposal_ids': ids, 'exact_conflicts': conflicts, 'selection_eligible': enough,
              'source_hidden': True, 'payload': payload, 'private_retrieval': retrieved['private_retrieval'],
              'input_identity': digest({'window': window, 'source': source, 'base': base, 'memory': pinned,
                                        'settings': SETTINGS, 'revision': REVISION})}
    packed['packed_hash'] = digest(packed)
    return packed


def verify_selector_inputs(window, source, inputs, packed):
    if packed['packed_hash'] != digest({k: v for k, v in packed.items() if k != 'packed_hash'}):
        raise IntegrityError('saved selector input changed')
    base = read(inputs['selector_context']); pinned = read(inputs['contexts'][window['context']]['selector_memory'])['payload']
    wanted = digest({'window': window, 'source': source, 'base': base, 'memory': pinned, 'settings': SETTINGS, 'revision': REVISION})
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
    refs = {r['reference_id']: r for r in base['references']}
    shared = {r['reference_id']: r for r in p['shared_references']}
    if len(shared) != len(p['shared_references']): raise IntegrityError('duplicate shared record')
    for rid, row in shared.items():
        if rid not in refs or row != public_reference(refs[rid]):
            raise IntegrityError('shared comparator text changed')
    linked = set()
    for pkg in p['packages'].values():
        for kind, field, cap in [('accepted_generated_task', 'accepted_reference_ids', SETTINGS['selector']['accepted_neighbors_per_candidate']),
                                 ('real_task', 'real_reference_ids', SETTINGS['selector']['real_neighbors_per_candidate'])]:
            values = pkg[field]
            expected = min(cap, sum(r['source'] == kind for r in refs.values())) if packed['selection_eligible'] else 0
            if len(values) != expected or len(set(values)) != len(values) or any(k not in shared or shared[k]['source'] != kind for k in values):
                raise IntegrityError('candidate-specific reference coverage/identity changed')
            linked.update(values)
    if linked != set(shared): raise IntegrityError('unlinked comparator in shared context')
    return packed


def selector_request(window, packed):
    p = packed['payload']; sid = 's' + digest({'revision': REVISION, 'window': window['id'], 'input': packed['packed_hash']})[:18]
    template, version, h = load_prompt(HERE / 'selector.md')
    prompt = render_prompt(template, {'CANDIDATES_WITH_REFERENCES': [p['packages'][k] for k in p['proposal_ids']],
        'SHARED_REFERENCES': p['shared_references'], 'IN_FLIGHT_RESERVATIONS': p['in_flight_reservations'],
        'REJECTED_MEMORY_DIGEST': p['rejected_memory_digest'], 'SELECTOR_ID': sid})
    bound = selection.stage_bindings(packed)[selection.STAGE]
    check_size(prompt, p['shared_references'], bound.schema)
    def validate(value):
        selection.parse(value, candidate_ids=p['proposal_ids'], reference_ids=[r['reference_id'] for r in p['shared_references']])
        if value['selector_id'] != sid: raise ValueError('selector changed its opaque ID')
    return {'purpose': 'full_window_selection', 'stage': selection.STAGE, 'call_id': sid, 'prompt': prompt,
            'prompt_template_version': version, 'prompt_template_hash': h, 'parent_ids': p['proposal_ids'], 'response_validator': validate}
