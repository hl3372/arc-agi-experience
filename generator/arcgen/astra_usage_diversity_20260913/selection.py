"""All useful eligible contributions enter the queue; only true repeats collapse."""
from copy import deepcopy
from arcgen.astra_targeted_followup_20260911.selection import (
    parse, stage_bindings, components, STAGE, SCHEMA,
)
from .common import IntegrityError, digest


def allocate(window, source, response, selection_policy=None, *, protected=True):
    response = parse(response)
    assessments = {r['candidate_id']: r for r in response['candidate_assessments']}
    if set(assessments) - set(source):
        raise IntegrityError('selector assessed foreign source candidates')
    ranked = sorted(response['ranked_plausible'], key=lambda r: r['development_priority'])
    briefs = {r['candidate_id']: r for r in ranked}
    ranks = {k: r['development_priority'] for k, r in briefs.items()}
    redundancy, families = components(assessments)
    chosen, seen, deferred = [], {}, []
    for brief in ranked:
        key = brief['candidate_id']; group = redundancy[key]
        if group in seen:
            deferred.append({'candidate_id': key, 'reason': 'interchangeable_within_window',
                             'representative_id': seen[group]})
        else:
            seen[group] = key; chosen.append(key)
    rows = rows_for_ids(window, source, response, chosen)
    return {'rows': rows, 'retained_ids': chosen, 'natural_ids': chosen,
        'targets': [], 'real_ranks': ranks, 'redundancy_deferrals': deferred,
        'uncertain_ids': [k for k, v in assessments.items() if v['eligibility'] == 'uncertain'],
        'ineligible_ids': [k for k, v in assessments.items() if v['eligibility'] == 'ineligible'],
        'capacity_deferrals': [], 'family_quota': None, 'rank_cutoff': None,
        'maximum_developments': len(rows), 'no_replacement_generation': True,
        'policy': 'useful_contribution_no_quota'}


def group_allocation(queue, ordinal, *, group_size=6):
    if type(ordinal) is not int or ordinal < 0 or type(group_size) is not int or not 1 <= group_size <= 6:
        raise IntegrityError('invalid development group')
    selected = deepcopy(queue['rows'][ordinal*group_size:(ordinal+1)*group_size])
    for index, row in enumerate(selected, 1):
        row['study']['queue_ordinal'] = row['study']['dispatch_ordinal']
        row['study']['dispatch_ordinal'] = index
        row['source_development_priority'] = index
    return {**deepcopy(queue), 'rows': selected,
        'retained_ids': [r['source_candidate_id'] for r in selected],
        'maximum_developments': len(selected), 'queue_hash': digest(queue), 'group_ordinal': ordinal}


def rows_for_ids(window, source, response, chosen):
    response = parse(response)
    assessments = {r['candidate_id']: r for r in response['candidate_assessments']}
    briefs = {r['candidate_id']: r for r in response['ranked_plausible']}
    ranks = {k: r['development_priority'] for k, r in briefs.items()}
    redundancy, families = components(assessments)
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(briefs) & set(source):
        raise IntegrityError('materialization requires exact distinct plausible source IDs')
    rows = []
    for ordinal, key in enumerate(chosen, 1):
        raw = source[key]; assessment = assessments[key]; brief = briefs[key]
        rows.append({'source_candidate_id': key,
            'lineage_id': digest({'revision': 'post-boardless-r2', 'source': key}),
            'kernel': deepcopy(raw['kernel']), 'search_mode': raw['kernel']['search_mode'],
            **{k: deepcopy(raw[k]) for k in ('private_source', 'private_conditioned', 'private_source_call_id', 'private_effort')},
            'source_development_priority': ordinal,
            'editorial_brief': {'candidate_id': key, 'development_priority': ranks[key], 'route': 'wildcard',
                'preserve': raw['kernel']['central_relation'], 'develop': brief['development_brief'],
                'avoid': raw['kernel']['main_authoring_risk'],
                'closest_similarity_concern': 'Preserve a useful, coherent task and fair public evidence.',
                'selection_reason': 'A credible worthwhile episode under the common contribution policy.',
                'optional_development_suggestion': brief['optional_development_suggestion']},
            'novelty': {'selector_archive_checked': True, 'archive_scope': 'full_search_bounded_prompt',
                'confirmed_same_procedure_duplicate': False,
                'within_window_sibling_ids': deepcopy(assessment['redundant_with']),
                'selected_as_sibling_representative': bool(assessment['redundant_with'])},
            'selector_assessment': deepcopy(assessment),
            'study': {**deepcopy(raw['study']), 'real_selector_rank': ranks[key],
                'procedure_family_group': families[key], 'redundancy_group': redundancy[key],
                'dispatch_ordinal': ordinal, 'admission': 'quality_eligible',
                'natural_eligible': True, 'merit_admitted': True, 'focal_satisfied': False,
                'focal_call_id': None, 'original_selector_window': window['id']}})
    return rows
