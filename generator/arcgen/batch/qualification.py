"""Offline qualification of a prepared run: no model or embedding calls.

It checks what the run can check on its own data:
  * the retrieval index covers the frozen reference corpus (its snapshot hash is what the launch binds);
  * every prepared idea request belongs to this design and fits its model's context;
  * a full selector window (24 synthetic candidates with distinct bank neighbours) fits the selector's prompt budget
    at the floor `science.selector.minimum_accepted_neighbors` (live windows over budget at
    the configured depth step down to that floor, supply.selector_inputs).
"""
from pathlib import Path
from arcgen.astra_usage_diversity_20260913 import selection
from .common import HOMES, REVISION, SETTINGS, read, ref, digest, frozen, verify_refs, code_identity, IntegrityError
from .retrieval import check_size
from . import supply

WINDOW = 24  # candidates in a full idea window (6 ideas x 4 explore calls)


def _synthetic_window(index, depth):
    """A selector input of WINDOW candidates, each with `depth` distinct bank neighbours taken in corpus order."""
    from copy import deepcopy
    from .retrieval import public_reference
    accepted = [index.records[k] for k in index.ids if index.records[k]['source'] == 'accepted_generated_task']
    packages, shared = {}, {}
    for i in range(WINDOW):
        kid = f'kqualification{i:04d}'
        chosen = accepted[(i * depth) % max(1, len(accepted)):][:depth] if accepted else []
        if len(chosen) < min(depth, len(accepted)):
            chosen = (chosen + accepted)[:min(depth, len(accepted))]
        for r in chosen:
            shared[r['reference_id']] = public_reference(r)
        packages[kid] = {'kernel': {'candidate_id': kid, 'search_mode': 'v1_loop',
                                    'central_relation': 'A qualification candidate relation of typical length. ' * 6,
                                    'recipe': 'A recipe sentence. ' * 4, 'what_is_new': 'What is new. ' * 4,
                                    'why_coherent': 'Why coherent. ' * 4, 'possible_test_generalization': 'Test. ' * 4,
                                    'main_authoring_risk': 'Risk. ' * 4},
                         'accepted_reference_ids': [r['reference_id'] for r in chosen], 'real_reference_ids': [],
                         'staleness': {'status': 'current', 'reasons': []}}
    payload = {'proposal_ids': sorted(packages), 'packages': packages, 'shared_references': [shared[k] for k in sorted(shared)],
               'in_flight_reservations': [], 'rejected_memory_digest': {'hard_duplicate': [], 'soft_mechanism': [],
                                                                         'failure_mode': [], 'recoverable_work': []}}
    return {'payload': deepcopy(payload), 'packed_hash': digest(payload)}


def qualify(root, *, inputs, index):
    root = Path(root); path = root / 'selector_qualification.private.json'
    if path.exists():
        value = read(path); verify_refs(value)
        if value['code_hash'] != code_identity(): raise IntegrityError('qualification code changed')
        return value
    corpus = sorted(read(inputs['selector_context'])['references'], key=lambda r: r['reference_id'])
    if index.snapshot_hash != digest(corpus):
        raise IntegrityError('retrieval index does not cover the frozen reference corpus')
    from .allocation import design
    plan = design()
    requests = [request for p in sorted((root / 'design/windows').glob('*/ideas.private.json')) for request in read(p).values()]
    expected = sum(len(w['calls']) for w in plan['windows'] if w.get('wave') == 'A')
    if len(requests) != expected or (expected and {r['private']['account'] for r in requests} != set(HOMES)):
        raise IntegrityError('prepared author request membership changed')
    floor = SETTINGS['selector']['minimum_accepted_neighbors']
    packed = _synthetic_window(index, floor)
    request = supply.selector_request({'id': 'qualification-window'}, packed)
    schema = selection.stage_bindings(packed)[selection.STAGE].schema
    size = check_size(request['prompt'], packed['payload']['shared_references'], schema)
    value = {'status': 'passed', 'revision': REVISION, 'code_hash': code_identity(), 'reference_snapshot_hash': index.snapshot_hash,
             'reference_count': len(corpus), 'settings': SETTINGS, 'composed_recipe_requests': len(requests),
             'maximum_author_prompt_tokens': max((r['private']['context_check']['estimated_prompt_tokens'] for r in requests), default=0),
             'selector_window_at_floor': {'candidates': WINDOW, 'accepted_neighbours': floor, 'context': size},
             'new_text_calls': 0, 'new_embedding_requests': 0,
             'limitation': 'Cached retrieval and context checks only; no semantic recall or model quality guarantee.'}
    return frozen(path, value)
