"""Stub model for domain A dry runs: one latch world; scripted pilots. Requests whose id contains 'bad' fail."""
import json

LATCH = {'state': {'x': {'range': [0, 15], 'settable': True}, 'h': {'range': [0, 4]}},
         'actions': {'P': {'rules': [{'set': {'h': 'h + 1'}}]},
                     'R': {'rules': [{'if': 'h >= 3', 'set': {'x': 'x + 4', 'h': '0'}}, {'set': {'h': '0'}}]},
                     'D': {'rules': [{'set': {'x': 'x - 1'}}]}},
         'readouts': {'r1': 'x', 'lamp': 'h > 0'}, 'max_len': 6}
RIVAL = {**LATCH, 'actions': {**LATCH['actions'], 'R': {'rules': [{'set': {'x': 'x + 4', 'h': '0'}}]}}}
PLAN = [{'start': {'x': x}, 'inputs': [['P']] * k + [['R']]} for k in range(1, 5) for x in (2, 9)]


def call(prompt, schema, record_dir, name, effort='high'):
    if 'bad' in str(record_dir):
        raise RuntimeError('stubbed model failure')
    cand = {'title': 'latch', 'story': 'a latch', 'world_json': json.dumps(LATCH), 'public_description': 'A device.',
            'rivals': [{'label': 'always fires', 'world_json': json.dumps(RIVAL)}], 'plan_json': json.dumps(PLAN),
            'demand_rationale': ''}
    if name == 'ideas':
        return {'candidates': [cand]}
    if name == 'select':
        return {'chosen_index': 0, 'reason': 'only one'}
    if name == 'plan':
        return {k: cand[k] for k in ('world_json', 'public_description', 'rivals', 'plan_json')} | {'notes': ''}
    if name.endswith('_queries'):
        return {'answers_json': json.dumps({f'Q{i}': [0, 0] for i in range(1, 40)}), 'law': 'unknown'}
    return {'experiments_json': json.dumps([{'start': {'s1': 3}, 'actions': ['A', 'A', 'A', 'B']}]), 'belief': '',
            'done': True}
