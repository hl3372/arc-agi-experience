"""Hard or unfair? For each frontier world: does the pilot's final predictor fit its own observations, and how often
does a public random experiment separate the pilot's law from the true law?"""
import json
import random
import os
import sys
os.environ.setdefault('RE_A_V2', '1')   # the published runs' interface; RE_A_V2=0 for round-1 style runs
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.construct.pipeline import ROOT
from research_envs.construct.sandbox import run_program
from research_envs.discover.world import World
from research_envs.discover.harness import PublicMap
from research_envs.discover.fairness import sample_experiment

rd = ROOT / 'runs' / sys.argv[1]
for f in sorted((rd / 'tasks').glob('*/dossier.json')):
    t = json.loads(f.read_text())
    if t.get('class') != 'frontier':
        continue
    w = World(t['world']); pm = PublicMap(w)
    resp = f.parent / 'calls' / 'pilot_full_queries' / 'response.json'
    prog = json.loads(resp.read_text()).get('predictor_program') if resp.exists() else None
    full = t['pilots']['full']
    a = t['audit']
    line = (f"{t['id']:13s} {str(t.get('lever') or t.get('demand') or 'ctl'):7s} crit {full['critical_accuracy']:.2f} "
            f"| rivals left {len(a['identifiability'].get('rivals_left', []))}, needles {len(a.get('needles', []))}, dropped {t.get('queries_dropped')}")
    if not prog:
        print(line, '| no predictor program'); continue
    # 1. fit to its own observations: every prefix of every experiment it ran
    obs = []
    for h in full['history']:
        if isinstance(h.get('trace'), list):
            for k in range(len(h['actions']) + 1):
                obs.append(({'start': h['start'] or {}, 'actions': h['actions'][:k]}, h['trace'][k]))
    # 2. separation from the truth on public random experiments
    rng = random.Random(0)
    rand = []
    for _ in range(1500):
        st, seq = sample_experiment(w, rng)
        q = {'start': st, 'inputs': seq}
        rand.append((pm.public_query(q), list(w.run_inputs(w.start_state(st), seq)[-1])))
    qs = [o[0] for o in obs] + [r[0] for r in rand]
    run = run_program(prog, {'queries': qs}, timeout=120)
    if not run['ok'] or not isinstance(run['object'], dict):
        print(line, '| predictor did not run on the batch'); continue
    pred = run['object']
    def norm(v):
        if isinstance(v, list) and v and isinstance(v[0], list):
            v = v[-1]
        return [int(x) for x in v] if isinstance(v, list) else v
    fit = sum(norm(pred.get(f'Q{i + 1}')) == list(o[1]) for i, o in enumerate(obs))
    sep = sum(norm(pred.get(f'Q{len(obs) + i + 1}')) != r[1] for i, r in enumerate(rand))
    print(line, f"| fits own observations {fit}/{len(obs)} | separated from truth on {sep / len(rand):.0%} of random experiments")
