"""Aggregate the model review panel: per family and domain, verdicts, fairness, scores, agreement, flagged tasks.

usage: review_summary.py OUT
  Reads runs/OUT/index.json and runs/OUT/verdicts/{gpt,opus}/*.json, and each task's class from its run's dossier.
"""
import collections
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.common import ROOT
if len(sys.argv) < 2:
    raise SystemExit(__doc__)
R = ROOT / 'runs' / sys.argv[1]
index = {e['id']: e for e in json.loads((R / 'index.json').read_text())}
cls = {}
for e in index.values():
    d = ROOT / 'runs' / str(e.get('round')) / 'tasks' / e['id'] / 'dossier.json'
    cls[e['id']] = json.loads(d.read_text()).get('class', 'accepted') if e.get('round') and d.exists() else 'accepted'
V = {}
for fam in ('gpt', 'opus'):
    V[fam] = {}
    for f in (R / 'verdicts' / fam).glob('*.json'):
        try:
            v = json.loads(f.read_text())
            if f.stem in index:
                V[fam][f.stem] = v
        except Exception:
            pass
for fam, vs in V.items():
    print(f'== {fam}: {len(vs)} reviews')
    for dom, group in (('A frontier', [i for i in vs if index[i]['domain'] == 'A' and cls[i] == 'frontier']),
                       ('A solved', [i for i in vs if index[i]['domain'] == 'A' and cls[i] == 'solved']),
                       ('B', [i for i in vs if index[i]['domain'] == 'B'])):
        if not group:
            continue
        g = [vs[i] for i in group]
        print(f"  {dom:11s} n={len(g)} verdicts {dict(collections.Counter(x.get('verdict') for x in g))} "
              f"fairness {dict(collections.Counter(x.get('fairness') for x in g))} "
              f"coherence {statistics.mean(x.get('coherence', 0) for x in g):.1f} interest {statistics.mean(x.get('interest', 0) for x in g):.1f} "
              f"known {sum(1 for x in g if (x.get('known_as') or '').strip())} desc_ok {sum(1 for x in g if x.get('description_ok'))}"
              + (f" solver_failure {dict(collections.Counter(x.get('solver_failure') for x in g))}" if dom.startswith('A') else ''))
both = sorted(set(V['gpt']) & set(V['opus']))
if both:
    agree = sum(V['gpt'][i]['verdict'] == V['opus'][i]['verdict'] for i in both)
    fair_agree = sum(V['gpt'][i]['fairness'] == V['opus'][i]['fairness'] for i in both)
    print(f'\nagreement on {len(both)} tasks: verdict {agree}/{len(both)}, fairness {fair_agree}/{len(both)}')
print('\nflagged (reject or unfair by either family):')
for i in sorted(index):
    flags = [(fam, V[fam][i]) for fam in V if i in V[fam] and (V[fam][i].get('verdict') == 'reject' or V[fam][i].get('fairness') == 'unfair')]
    for fam, v in flags:
        print(f"  {i:12s} [{cls[i]}] {fam}: {v.get('verdict')}/{v.get('fairness')} | {(v.get('main_issue') or v.get('fairness_reason') or '')[:170]}")
