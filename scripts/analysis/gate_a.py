"""Phase 1 gate for domain A: two hand-built worlds through the full audit (no model calls)."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.discover import audit

RESERVOIR = {  # A2 hidden variable: P charges a hidden reservoir h; R releases it into the visible level; D drains
    'state': {'x': {'range': [0, 15], 'settable': True}, 'h': {'range': [0, 5]}},
    'actions': {'P': {'rules': [{'set': {'h': 'h + 1'}}]},
                'R': {'rules': [{'set': {'x': 'x + h', 'h': '0'}}]},
                'D': {'rules': [{'set': {'x': 'x - 2'}}]}},
    'readouts': {'r1': 'x'}, 'max_len': 6}
RESERVOIR_PLAN = [{'start': {'x': x}, 'inputs': [['P']] * k + [['R']]} for k in range(1, 6) for x in (0, 7)] + \
                 [{'start': {'x': 3}, 'inputs': [['P'], ['P'], ['D'], ['R'], ['P'], ['R']]},
                  {'start': {'x': 12}, 'inputs': [['P']] * 5 + [['R']]}] + \
                 [{'start': {'x': x}, 'inputs': [['D'], ['D'], ['R']]} for x in (1, 6, 11)]

LATCH = {  # A2 with a clue: P charges a hidden latch (the lamp shows it is charged at all); R fires only at charge >= 3
    'state': {'x': {'range': [0, 15], 'settable': True}, 'h': {'range': [0, 4]}},
    'actions': {'P': {'rules': [{'set': {'h': 'h + 1'}}]},
                'R': {'rules': [{'if': 'h >= 3', 'set': {'x': 'x + 4', 'h': '0'}}, {'set': {'h': '0'}}]},
                'D': {'rules': [{'set': {'x': 'x - 1'}}]}},
    'readouts': {'r1': 'x', 'lamp': 'h > 0'}, 'max_len': 6}
LATCH_PLAN = [{'start': {'x': x}, 'inputs': [['P']] * k + [['R']]} for k in range(1, 5) for x in (2, 9)] + \
             [{'start': {'x': x}, 'inputs': [['D'], ['P'], ['D'], ['R']]} for x in (0, 5, 14)] + \
             [{'start': {'x': 6}, 'inputs': [['P'], ['P'], ['P'], ['P'], ['P'], ['R']]}]

LATCH5 = {  # A2, stronger: R fires only at charge 5; the lamp shows the charge coarsely (0, 1-2, 3-4, 5)
    'state': {'x': {'range': [0, 15], 'settable': True}, 'h': {'range': [0, 5]}},
    'actions': {'P': {'rules': [{'set': {'h': 'h + 1'}}]},
                'R': {'rules': [{'if': 'h >= 5', 'set': {'x': 'x + 4', 'h': '0'}}, {'set': {'h': '0'}}]},
                'D': {'rules': [{'set': {'x': 'x - 1'}}]}},
    'readouts': {'r1': 'x', 'lamp': '(h + 1) // 2'}, 'max_len': 6}
LATCH5_PLAN = [{'start': {'x': x}, 'inputs': [['P']] * k + [['R']]} for k in range(1, 6) for x in (2, 9)] + \
              [{'start': {'x': x}, 'inputs': [['D'], ['P'], ['D'], ['R']]} for x in (0, 5, 14)]
LATCH_NAMED = [{'label': 'R always fires', 'world': {**LATCH, 'actions': {**LATCH['actions'], 'R': {'rules': [{'set': {'x': 'x + 4', 'h': '0'}}]}}}},
               {'label': 'R never fires', 'world': {**LATCH, 'actions': {**LATCH['actions'], 'R': {'rules': [{'set': {'h': '0'}}]}}}}]
LATCH5_NAMED = [{'label': 'R always fires', 'world': {**LATCH5, 'actions': {**LATCH5['actions'], 'R': {'rules': [{'set': {'x': 'x + 4', 'h': '0'}}]}}}},
                {'label': 'R never fires', 'world': {**LATCH5, 'actions': {**LATCH5['actions'], 'R': {'rules': [{'set': {'h': '0'}}]}}}},
                {'label': 'fires at 3', 'world': {**LATCH5, 'actions': {**LATCH5['actions'], 'R': {'rules': [{'if': 'h >= 3', 'set': {'x': 'x + 4', 'h': '0'}}, {'set': {'h': '0'}}]}}}}]

DIRECT = {  # calibration: all state is visible
    'state': {'x': {'range': [0, 7], 'settable': True}, 'y': {'range': [0, 7], 'settable': True}},
    'actions': {'I': {'rules': [{'set': {'x': 'x + 1'}}]}, 'J': {'rules': [{'set': {'y': 'y + 2'}}]},
                'S': {'rules': [{'set': {'x': 'y', 'y': 'x'}}]}},
    'readouts': {'r1': 'x', 'r2': 'y'}, 'max_len': 6}
DIRECT_PLAN = [{'start': {'x': 1, 'y': 2}, 'inputs': [['I'], ['J'], ['S']]}]


def brief(a):
    b = {k: (v or {}).get('public', {}).get('critical') for k, v in a['baselines'].items()}
    o = {k: ((v or {}).get('oracle', {}).get('learner'), (v or {}).get('oracle', {}).get('critical')) for k, v in a['baselines'].items()}
    return {'states': a['states'], 'rivals': a['rivals'], 'equiv': len(a['interface_equivalent']), 'needles': a['needles'],
            'queries': a['queries'], 'identifiability': {k: v if not isinstance(v, list) else len(v) for k, v in a['identifiability'].items()},
            'coverage': a['coverage'], 'public_critical': b, 'oracle_critical': o, 'audit': a['audit']}


for name, spec, demand, plan, named in (('latch', LATCH, 'A2', LATCH_PLAN, LATCH_NAMED),
                                        ('latch5', LATCH5, 'A2', LATCH5_PLAN, LATCH5_NAMED),
                                        ('direct', DIRECT, 'A2', DIRECT_PLAN, ())):
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    t0 = time.time()
    truth, R, q, extra, a = audit.build(spec, demand, plan, named=named, budget=20)
    print(f'== {name} ({time.time() - t0:.1f}s)')
    print(json.dumps(brief(a), indent=1, default=str))
