"""Phase 1 gate (docs/PHASE1_GATE.md): two hand-built calibration instances through every mechanical check, no model
calls. Writes runs/gate/phase1_gate.json."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.construct.evaluate import Spec
from research_envs.construct import search, sibling, probes, novelty
from research_envs.construct.sandbox import run_program

TASKS = {
 'ramsey44_17': {
   'spec': {"object": "graph", "kind": "satisfy", "params": {"n": 17},
            "constraints": [{"name": "no_k4", "expr": "clique_number() <= 3"},
                            {"name": "no_independent_4", "expr": "independence_number() <= 3"}]},
   'hint': None, 'sibling': {'n': 7}, 'demand': 'D2',
   'shortcut': ['anneal', 'kick', 'hill_climb', 'greedy_add'], 'method': ['symmetric'],
   'witness': '''
import json
n = 17
qr = {(x * x) % n for x in range(1, n)}
print(json.dumps({"n": n, "edges": [[a, b] for a in range(n) for b in range(a + 1, n) if (b - a) % n in qr]}))
'''},
 'pg23_packing': {
   'spec': {"object": "set_system", "kind": "optimize", "params": {"m": 13},
            "constraints": [{"name": "blocks_of_4", "expr": "all(len(B) == 4 for B in F)"},
                            {"name": "meet_exactly_once", "expr": "all(len(A & B) == 1 for A, B in combinations(F, 2))"}],
            "objective": {"expr": "len(F)", "direction": "max"}},
   'hint': {'set_sizes': [4]}, 'sibling': {'m': 6}, 'demand': 'D2',
   'shortcut': ['anneal', 'kick', 'hill_climb', 'greedy_add'], 'method': ['symmetric'],
   'witness': '''
import json
print(json.dumps({"m": 13, "sets": [sorted((i + d) % 13 for d in (0, 1, 3, 9)) for i in range(13)]}))
'''},
}

report = {}
for name, t in TASKS.items():
    t0 = time.time(); r = {}
    spec = Spec(t['spec'])
    w = run_program(t['witness'], t['spec'], timeout=60)
    wv = spec.evaluate(w['object']) if w['ok'] else None
    r['witness'] = {'ran': w['ok'], 'valid': wv and wv['valid'], 'objective': wv and wv['objective']}
    sib_raw = sibling.at_size(t['spec'], **t['sibling'])
    sib_hint = t['hint']
    sib = sibling.enumerate_all(sib_raw, sib_hint)
    r['sibling'] = {k: v for k, v in sib.items() if k != 'optimal_examples'}
    for budget in (20000, 200000):
        bat = search.battery(t['spec'], t['hint'], methods=t['shortcut'] + t['method'], budget=budget, seeds=(0, 1))
        r[f'battery_{budget}'] = {m: {'valid': b.get('valid'), 'objective': b.get('objective'), 'evals': b.get('evals')}
                                   for m, b in bat['best'].items()}
    objs = [w['object']] + [b['object'] for b in bat['best'].values() if b.get('object')]
    ps = probes.probe_set(t['spec'], objs, t['hint'])
    ad = probes.adequacy(t['spec'], ps)
    r['probes'] = {'count': len(ps), 'adequacy': round(ad['score'], 3), 'survivors': ad['survivors']}
    fp = novelty.fingerprint({**t['spec'], 'params': {}}, t['hint'])
    arch = novelty.load_archive()
    r['novelty'] = {'fingerprint': fp['values'],
                    'textbook_matches': [a['id'] for a in arch if a['object'] == t['spec']['object'] and novelty.same_family(fp, a['fingerprint'])],
                    'zoo_match': novelty.zoo_match(t['spec']['object'], w['object']) if w['ok'] else None}
    # two-sided audit at each budget
    for budget in (20000, 200000):
        b = r[f'battery_{budget}']
        def ok(m):
            x = b[m]
            if not x['valid']:
                return False
            return spec.kind != 'optimize' or x['objective'] == r['witness']['objective']
        r[f'audit_{budget}'] = {'shortcut_fails': not any(ok(m) for m in t['shortcut']),
                                'method_succeeds': any(ok(m) for m in t['method'])}
    r['seconds'] = round(time.time() - t0)
    report[name] = r
    print(name, json.dumps(r, indent=1, default=str)[:3000], flush=True)
from research_envs.common import ROOT
(ROOT / 'runs' / 'gate').mkdir(parents=True, exist_ok=True)
json.dump(report, open(ROOT / 'runs' / 'gate' / 'phase1_gate.json', 'w'), indent=1, default=str)
