"""Stub model for dry runs (RE_LLM_STUB): canned responses for each call name. Requests whose id contains 'bad' fail."""
import json

SPEC = {'object': 'graph', 'kind': 'optimize', 'params': {'n': 8},
        'constraints': [{'name': 'max_deg_3', 'expr': 'all(deg(v) <= 3 for v in V)'},
                        {'name': 'no_triangle', 'expr': 'triangles() == 0'}],
        'objective': {'expr': 'len(E)', 'direction': 'max'}}
WITNESS = ('import json\nprint(json.dumps({"n": 8, "edges": [[i, i ^ (1 << b)] for i in range(8) for b in range(3) '
           'if i < i ^ (1 << b)]}))')
PROGRAM = 'import json\nprint(json.dumps({"n": 8, "edges": [[i, (i + 1) % 8] for i in range(8)]}))'


def call(prompt, schema, record_dir, name, effort='high'):
    if 'bad' in str(record_dir):
        raise RuntimeError('stubbed model failure')
    cand = {'title': 'cubic triangle-free', 'statement': 'max edges, degree <= 3, no triangle', 'spec_json': json.dumps(SPEC),
            'sibling_params_json': '{"n": 5}', 'search_hint_json': '{}', 'demand_rationale': '', 'construction_idea': '',
            'novelty_note': ''}
    fixed = {'ideas': {'candidates': [cand]},
             'select': {'chosen_index': 0, 'reason': 'only one'},
             'plan': {'spec_json': json.dumps(SPEC), 'statement': cand['statement'], 'witness_program': WITNESS,
                      'witness_is_construction': True, 'degenerate_exprs': ['len(E) == 0'], 'sibling_params_json': '{"n": 5}',
                      'search_hint_json': '{}', 'notes': ''},
             'restate': {'spec_json': json.dumps(SPEC), 'ambiguities': ''},
             'naturalness': {'score': 4, 'about': 'x', 'arbitrary_parts': [], 'reasons': ''}}
    base = name[:-2] if name.endswith('_2') else name     # retry calls (ideas_2, select_2) answer like the first
    if base in fixed:
        return fixed[base]
    if name.startswith('pilot_object'):
        return {'object_json': json.dumps({'n': 8, 'edges': [[0, 1]]}), 'note': ''}
    return {'program': PROGRAM, 'note': ''}
