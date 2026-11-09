"""Rescore domain A validation results with the current answer normalization, from the recorded query calls (answers
and predictor program; the program is rerun in the sandbox). Keeps the original scores as *_orig. Idempotent.

usage: rescore_a.py <round>
"""
import json
import os
import sys
os.environ.setdefault('RE_A_V2', '1')   # the published runs' interface; RE_A_V2=0 for round-1 style runs
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.construct.pipeline import ROOT, _write
from research_envs.construct.sandbox import run_program
from research_envs.discover import harness as H
from research_envs.discover.fairness import answer
from research_envs.discover.world import World

rd = ROOT / 'runs' / sys.argv[1]
tasks = {json.loads(f.read_text())['id']: json.loads(f.read_text()) for f in (rd / 'tasks').glob('*/dossier.json')}
changed = 0
for sd in sorted((rd / 'validation').glob('*')):
    if not sd.is_dir():
        continue
    for f in sorted(sd.glob('*.json')):
        r = json.loads(f.read_text())
        t = tasks.get(r.get('task'))
        qcall = sd / r['task'] / 'calls' / 'val_queries' / 'response.json'
        if not qcall.exists():
            qcall = sd / r['task'] / 'calls' / 'val_queries_program' / 'response.json'   # the program-only fallback
        if not t or r.get('error') or not qcall.exists():
            continue
        resp = json.loads(qcall.read_text())
        w = World(t['world']); pm = H.PublicMap(w)
        qs = t['queries']; pq = [pm.public_query(q) for q in qs]
        truths = [list(answer(w, q)) for q in qs]
        try:
            direct = json.loads(resp.get('answers_json') or '{}')
        except json.JSONDecodeError:
            direct = {}
        dh = H._score(direct, truths, w.readout_names)
        ph = None
        if (resp.get('predictor_program') or '').strip():
            run = run_program(resp['predictor_program'], {'queries': pq}, timeout=60)
            if run['ok'] and isinstance(run['object'], dict) and sum(f'Q{i + 1}' in run['object'] for i in range(len(qs))) >= 0.9 * len(qs):
                ph = H._score(run['object'], truths, w.readout_names)
        hits = ph if ph is not None else dh
        crit = [h for h, q in zip(hits, qs) if q.get('disagree')]
        new = {'accuracy': sum(hits) / len(hits), 'critical_accuracy': sum(crit) / len(crit) if crit else None,
               'scored_from': 'program' if ph is not None else 'answers'}
        if 'critical_accuracy_orig' not in r:
            r['critical_accuracy_orig'], r['accuracy_orig'] = r.get('critical_accuracy'), r.get('accuracy')
        if new['critical_accuracy'] != r.get('critical_accuracy'):
            changed += 1
            print(f"{sd.name} {r['task']}: critical {r.get('critical_accuracy_orig')} -> {new['critical_accuracy']}")
        r.update(new)
        _write(f, r)
print(f'{changed} results changed')
