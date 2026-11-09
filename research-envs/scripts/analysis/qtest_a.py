"""Offline test of the discriminating query chooser: rebuild queries for chosen worlds (no model calls) into
runs/qtest_a/tasks/<id>/dossier.json (world, rivals and old pilot attempt kept; only the queries change)."""
import json
import os
import sys
for k, v in (('RE_A_V2', '1'), ('RE_A_TRIVIAL_CAP', '0.3'), ('RE_A_ANSWER_CAP', '0.4'), ('RE_A_QUERY_MODE', 'discriminating')):
    os.environ.setdefault(k, v)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.common import ROOT
RUNS = ROOT / 'runs'


def one(arg):
    rnd, tid = arg
    from research_envs.discover import audit as Au, pipeline as Pp
    from research_envs.discover.world import World
    t = json.loads((RUNS / rnd / 'tasks' / tid / 'dossier.json').read_text())
    w = World(t['world'])
    _, _, queries, extra, a = Au.build(t['world'], t['demand'], Pp.parse_plan(w, t['plan_experiments']),
                                       named=t['rivals'], budget=40)
    ident = a.get('identifiability', {})
    left = set(ident.get('rivals_left_idx', [])) | set(ident.get('needle_idx', []))
    t['queries_old'] = t['queries']
    t['queries'] = [q for q in queries if not (set(q.get('any_rival', [])) & left)]
    t['queries_dropped'] = len(queries) - len(t['queries'])
    t['audit_qtest'] = a
    out = RUNS / 'qtest_a' / 'tasks' / tid
    out.mkdir(parents=True, exist_ok=True)
    (out / 'dossier.json').write_text(json.dumps(t, default=str))
    return tid, len(t['queries']), t['queries_dropped'], a['queries']


if __name__ == '__main__':
    import multiprocessing as mp
    pairs = [tuple(x.split(':')) for x in sys.argv[1].split(',')]
    with mp.get_context('spawn').Pool(len(pairs)) as p:
        for r in p.imap_unordered(one, pairs):
            print(r, flush=True)
