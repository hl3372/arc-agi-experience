"""Source-hidden contribution judgments and deterministic feasible representatives."""
from copy import deepcopy
import hashlib
from jsonschema import Draft202012Validator
from arcgen.lib.factory_runtime import StageBinding
from arcgen.lib.schemas import parse_json_object
from arcgen.lib.v4_schemas import _transport_schema
from .common import HERE, SETTINGS, IntegrityError, digest, read

STAGE = 'arm_bank_selector_v1'
SCHEMA = read(HERE / 'selector_schema.json')


def parse(data, *, candidate_ids=None, reference_ids=None):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    Draft202012Validator(SCHEMA).validate(value)
    rows = value['candidate_assessments']
    ids = [r['candidate_id'] for r in rows]
    ranked = value['ranked_plausible']
    plausible = {r['candidate_id'] for r in rows if r['eligibility'] == 'plausible'}
    if len(ids) != len(set(ids)) or candidate_ids is not None and set(ids) != set(candidate_ids):
        raise ValueError('selector candidate membership changed')
    rank_ids = [r['candidate_id'] for r in ranked]
    if len(rank_ids) != len(set(rank_ids)) or set(rank_ids) != plausible:
        raise ValueError('rank all and only intrinsically plausible candidates')
    if sorted(r['development_priority'] for r in ranked) != list(range(1, len(ranked) + 1)):
        raise ValueError('selector ranks must be genuine and contiguous')
    if sum(r['optional_development_suggestion'] is not None for r in ranked) > 2:
        raise ValueError('at most two optional suggestions')
    for row in rows:
        for field in ('redundant_with', 'same_procedure_family_with'):
            if row['candidate_id'] in row[field] or not set(row[field]) <= set(ids):
                raise ValueError('within-window links must name other candidates, never bank IDs')
        refs = [r['reference_id'] for r in row['bank_comparisons']]
        if len(refs) != len(set(refs)) or set(refs) & set(ids):
            raise ValueError('bank reference and candidate namespaces are distinct')
        if reference_ids is not None and not set(refs) <= set(reference_ids):
            raise ValueError('bank comparison refers outside the supplied shared references')
        if row['eligibility'] == 'plausible' and not (row['useful_contribution'] or '').strip():
            raise ValueError('plausibility requires a concrete useful contribution')
        if any(r['relationship'] == 'redundant' for r in row['bank_comparisons']) and row['eligibility'] != 'ineligible':
            raise ValueError('redundancy with accepted/reference work is not a useful-relative claim')
    forbidden = set(reference_ids or ()) | {x['reference_id'] for row in rows for x in row['bank_comparisons']}
    for row in ranked:
        text = row['development_brief'] + '\n' + (row['optional_development_suggestion'] or '')
        if any(r in text for r in forbidden) or 'committed:' in text or 'real:' in text:
            raise ValueError('reference identity leaked into an authoring brief')
    return value


def stage_bindings(packed=None):
    schema = deepcopy(SCHEMA)
    if packed is None:
        validator = parse
    else:
        ids = packed['payload']['proposal_ids']
        refs = [r['reference_id'] for r in packed['payload']['shared_references']]
        a = schema['properties']['candidate_assessments']['items']['properties']
        a['candidate_id']['enum'] = ids
        for field in ('redundant_with', 'same_procedure_family_with'):
            a[field]['items']['enum'] = ids
        schema['properties']['ranked_plausible']['items']['properties']['candidate_id']['enum'] = ids
        if refs:
            a['bank_comparisons']['items']['properties']['reference_id']['enum'] = refs
        else:
            a['bank_comparisons']['maxItems'] = 0
        validator = lambda data: parse(data, candidate_ids=ids, reference_ids=refs)
    return {STAGE: StageBinding(_transport_schema(schema), validator, ('full_window_selection',))}


def components(assessments):
    ids = list(assessments)
    def groups(fields):
        parent = {k: k for k in ids}
        def root(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]; k = parent[k]
            return k
        for key, row in assessments.items():
            for field in fields:
                for peer in row[field]:
                    a, b = sorted((root(key), root(peer)))
                    parent[b] = a
        return {k: root(k) for k in ids}
    return groups(('redundant_with',)), groups(('redundant_with', 'same_procedure_family_with'))


def allocate(window, source, response, selection_policy, *, protected=True):
    """No new slots. Source/recipe attribution and real model ranks remain intact."""
    response = parse(response)
    ass = {r['candidate_id']: r for r in response['candidate_assessments']}
    if set(ass) - set(source):
        raise IntegrityError('selector assessed foreign source candidates')
    ranked = sorted(response['ranked_plausible'], key=lambda r: r['development_priority'])
    brief = {r['candidate_id']: r for r in ranked}
    ids = [r['candidate_id'] for r in ranked]
    rank = {k: brief[k]['development_priority'] for k in ids}
    red, fam = components(ass)
    cap = SETTINGS['selector']['maximum_family_developments']
    calls = {c['private_id']: c for c in window['calls']}
    total = window['maximum_developments']
    if type(total) is not int or not 0 <= total <= 12:
        raise IntegrityError('window development ceiling must remain within 12')

    def contribution(k):
        return ' '.join((ass[k]['useful_contribution'] or '').casefold().split())

    def can_add(key, picked):
        if key in picked:
            return True
        if any(red[key] == red[k] for k in picked):
            return False
        relatives = [k for k in picked if fam[k] == fam[key]]
        return len(relatives) < cap and (not relatives or
               bool(contribution(key)) and all(contribution(key) != contribution(k) for k in relatives))

    def bounded(order, reserved=()):
        chosen = list(reserved)
        eligible = []
        for key in order:
            if key in chosen or can_add(key, chosen):
                if key not in chosen: chosen.append(key)
                eligible.append(key)
        return eligible

    def row(key):
        raw = source[key]; real_rank = rank.get(key)
        return {'source_candidate_id': key,
                'lineage_id': digest({'revision': 'post-boardless-r2', 'source': key}),
                'kernel': deepcopy(raw['kernel']), 'search_mode': raw['kernel']['search_mode'],
                **{k: deepcopy(raw[k]) for k in ('private_source', 'private_conditioned', 'private_source_call_id', 'private_effort')},
                'source_development_priority': real_rank or 49,
                'editorial_brief': {'candidate_id': key, 'development_priority': real_rank, 'route': 'wildcard',
                    'preserve': raw['kernel']['central_relation'],
                    'develop': brief.get(key, {}).get('development_brief', 'Realize the relation with fair public evidence.'),
                    'avoid': raw['kernel']['main_authoring_risk'],
                    'closest_similarity_concern': 'Preserve a useful, coherent task and fair public evidence.',
                    'selection_reason': 'A credible worthwhile episode under the common contribution policy.',
                    'optional_development_suggestion': brief.get(key, {}).get('optional_development_suggestion')},
                'novelty': {'selector_archive_checked': True, 'archive_scope': 'full_search_bounded_prompt',
                    'confirmed_same_procedure_duplicate': False,
                    'within_window_sibling_ids': deepcopy(ass[key]['redundant_with']),
                    'selected_as_sibling_representative': bool(ass[key]['redundant_with'])},
                'selector_assessment': deepcopy(ass[key]),
                'study': {**deepcopy(raw['study']), 'real_selector_rank': real_rank,
                          'procedure_family_group': fam[key], 'redundancy_group': red[key]}}

    # J/T are primary modes. Experimental coverage is declared by protected
    # targets below, not by treating their native planner routes as invitations.
    natural = bounded(ids)[:SETTINGS['selector']['natural_frontier']][:total]
    natural_drops, final_drops = [], []

    targets = []
    diagnostics = []
    for spec in window.get('protected_targets', []) if protected else []:
        if spec['call_id'] not in calls:
            raise IntegrityError('protected target refers outside this window')
        pool = set(calls[spec['call_id']]['opaque_kernel_ids'])
        choices = [k for k in ids if k in pool]
        targets.append({**deepcopy(spec), 'pool': sorted(pool), 'choices': choices,
                        'nominee': None, 'kind': 'plausible' if choices else 'empty'})
        if not choices:
            diagnostics += [(len(targets)-1, k) for k in sorted(pool) if k in ass and
                            ass[k]['eligibility'] == 'uncertain' and ass[k]['fair_realization_potential'] != 'infeasible'
                            and not ass[k]['redundant_with'] and not ass[k]['same_procedure_family_with']]
    pools = [set(t['pool']) for t in targets]
    if any(a & b for i, a in enumerate(pools) for b in pools[i+1:]):
        raise IntegrityError('protected source pools must remain disjoint')
    diagnostics.sort(key=lambda x: hashlib.sha256(f"{window['seed']}|diagnostic|{window['context']}|{x[1]}".encode()).hexdigest())
    if diagnostics:
        i, key = diagnostics[0]
        targets[i]['choices'] = [key]; targets[i]['kind'] = 'diagnostic'

    nominees = []
    order = sorted(range(len(targets)), key=lambda i: min(
        (rank.get(k, 49) for k in targets[i]['choices']), default=50))
    for i in order:
        target = targets[i]
        target['nominee'] = next((k for k in target['choices'] if can_add(k, nominees)), None)
        if target['nominee']:
            nominees.append(target['nominee'])
    if len(nominees) > total:
        raise IntegrityError('protected contributions exceed the declared development cap')
    frontier = bounded(ids, nominees)[:SETTINGS['selector']['natural_frontier']]
    retained = list(dict.fromkeys(frontier + nominees))
    if len(retained) > window.get('maximum_retained', 24):
        raise IntegrityError('retained frontier cap')
    if protected:
        count = window['merit_positions']
        if type(count) is not int or not 0 <= count <= total:
            raise IntegrityError('invalid merit allocation')
        while True:
            merit = frontier[:count]
            following = window['merit_positions'] + len(set(merit) & set(nominees))
            if following == count:
                break
            count = following
        intended = list(dict.fromkeys(merit + nominees))
    else:
        merit = frontier[:total]
        intended = list(merit)
    if len(intended) > total:
        raise IntegrityError('protected allocation exceeds the declared development cap')
    backfill = []
    for key in frontier:
        if len(intended) >= total:
            break
        if key not in intended and can_add(key, intended):
            intended.append(key)
            merit.append(key)
            backfill.append(key)
    kept = sorted(intended, key=lambda k: (rank.get(k, 49), k))
    dropped, representation_retries = [], []
    rows = []
    for ordinal, key in enumerate(kept, 1):
        record = row(key)
        target = next((t for t in targets if t['nominee'] == key), None)
        record['source_development_priority'] = ordinal
        record['study'].update(dispatch_ordinal=ordinal,
            admission='diagnostic' if target and target['kind'] == 'diagnostic' else 'natural' if key in merit else 'protected',
            natural_eligible=key in natural, focal_satisfied=target is not None,
            focal_call_id=target['call_id'] if target else None, merit_admitted=key in merit)
        rows.append(record)
    for target in targets:
        target['outcome'] = ('developed' if target['nominee'] in kept else 'capacity_deferred' if target['nominee']
                             else 'redundancy_or_family_deferred' if target['choices'] else 'empty')
    return {'rows': rows, 'targets': targets, 'retained_ids': retained, 'natural_ids': natural,
            'natural_ceiling_drops': natural_drops + final_drops, 'ceiling_drops': dropped,
            'representative_source_deferrals': representation_retries,
            'diagnostic_hash_pool': [k for _, k in diagnostics],
            'no_replacement_generation': True, 'real_ranks': rank,
            'source_ceilings_applied': False, 'backfill_ids': backfill,
            'procedure_family_cap': cap, 'maximum_developments': total}
