"""Per-world critical accuracy across solvers: the round's own pilot (gpt-6.1-sol high) and every validation solver."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.construct.pipeline import ROOT

rd = ROOT / 'runs' / sys.argv[1]
cols = {'high (round pilot)': {}}
for f in sorted((rd / 'tasks').glob('*/dossier.json')):
    t = json.loads(f.read_text())
    if t.get('final') and t.get('class') in ('solved', 'frontier'):
        full = (t.get('pilots') or {}).get('full') or {}
        cols['high (round pilot)'][t['id']] = (full.get('critical_accuracy'), full.get('experiments_used'), t.get('demand'), t.get('title'))
for d in sorted((rd / 'validation').glob('*')):
    if d.is_dir():
        cols[d.name] = {}
        for f in d.glob('*.json'):
            r = json.loads(f.read_text())
            cols[d.name][r['task']] = (r.get('critical_accuracy'), r.get('experiments_used'), r.get('demand'), r.get('title'))
names = list(cols)
ids = sorted(cols['high (round pilot)'])
print(f"{'task':10s} {'arm':5s} " + ' '.join(f'{n[:24]:>24s}' for n in names) + '  title')
for i in ids:
    arm = str(cols['high (round pilot)'][i][2])
    cells = []
    for n in names:
        v = cols[n].get(i)
        cells.append(f"{(f'{v[0]:.2f}' if v and v[0] is not None else '-'):>17s} ({str(v[1]) if v else '-':>3s})")
    print(f"{i:10s} {arm:5s} " + ' '.join(c.rjust(24) for c in cells) + '  ' + str(cols['high (round pilot)'][i][3])[:40])
for n in names:
    vals = [v[0] for v in cols[n].values() if v[0] is not None]
    if vals:
        print(f"{n}: mean {sum(vals) / len(vals):.2f}, perfect {sum(x == 1.0 for x in vals)}/{len(vals)}, below 0.9: {sum(x < 0.9 for x in vals)}")
