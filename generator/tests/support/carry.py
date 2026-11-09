"""Synthetic carry-over files (a saved queue and saved ideas), built the way a stopped run's carry produces them."""
from hashlib import sha256
import json
from pathlib import Path


def kernels_commit(n=4):
    from test_v1_ideas import _commit
    return _commit()


def source_rows(inputs, wid, seeds=(901, 902)):
    """Source rows of one idea window, from mock idea-loop commits through supply.source_rows."""
    from arcgen.batch import idea_v1, supply
    calls = [supply.make_call(arm='SEM', bank_setting='loop', context='c01', seed=s, account='main', window=wid,
                              idea_model='gpt-6.1-sol', effort='high') for s in seeds]
    w = {'id': wid, 'context': 'c01', 'calls': calls, 'wave': 'A'}
    requests = {c['opaque_id']: supply.idea_request(c, None, inputs) for c in calls}
    responses = {}
    for n, c in enumerate(calls):
        value = idea_v1.as_ideas(c, idea_v1.parse_commit(kernels_commit()))
        for i, k in enumerate(value['ideas']):
            k['central_relation'] += f' (variant {n}.{i})'
        responses[c['opaque_id']] = value
    return supply.source_rows(w, requests, responses)


def selector_response(ids, cid='fixture-selector'):
    return {'selector_id': cid, 'window_observations': None,
            'candidate_assessments': [{'candidate_id': k, 'eligibility': 'plausible', 'conceptual_potential': 'solid',
                'coherence_potential': 'solid', 'fair_realization_potential': 'plausible', 'redundant_with': [],
                'same_procedure_family_with': [], 'bank_comparisons': [], 'useful_contribution': 'Fresh insight ' + k,
                'reason': 'Coherent synthetic relation.'} for k in ids],
            'ranked_plausible': [{'candidate_id': k, 'development_priority': i + 1, 'development_brief': 'Realize fairly.',
                'optional_development_suggestion': None} for i, k in enumerate(ids)]}


def saved_queue(path, inputs, wid='old-supply-01'):
    """A saved queue file: selected rows of one source window (selection.allocate) with their source rows."""
    from arcgen.astra_usage_diversity_20260913 import selection
    from arcgen.batch.common import config_for
    source = source_rows(inputs, wid)
    window = {'id': wid, 'phase': 'screen', 'account': 'main', 'context': 'c01', 'seed': 777, 'calls': [],
              'maximum_developments': 12, 'minimum_selector_eligible_kernels': 1, 'editorial_mode': 'bypass',
              'protected': False, 'protected_targets': [], 'maximum_retained': len(source)}
    policy = config_for(Path(path).parent / 'selection-policy', mock=True)['selection']
    allocation = selection.allocate(window, source, selector_response(sorted(source)), policy, protected=False)
    rows = allocation['rows']
    value = {'provenance': {'id': 'synthetic-queue', 'source_run': 'runs/old', 'note': 'synthetic saved queue for tests'},
             'rows': rows, 'source_rows': {r['source_candidate_id']: source[r['source_candidate_id']] for r in rows}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))
    return value


def saved_ideas(path, inputs, wid='old-supply-02'):
    rows = source_rows(inputs, wid, (903, 904))
    value = {'provenance': {'id': 'synthetic-ideas', 'source_run': 'runs/old', 'source_window': wid}, 'source_rows': rows}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))
    return value


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()
