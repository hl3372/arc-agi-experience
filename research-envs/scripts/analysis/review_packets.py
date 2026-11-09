"""Blind review packets for the model review panel: the task as the solver sees it, the hidden law, the queries with
true answers, and (domain A) the strongest solver's experiment log and stated law. No pipeline verdicts.

usage: review_packets.py OUT RUN [RUN ...] [--v1]
  Writes runs/OUT/packets/<id>.md and runs/OUT/index.json for every usable task in the given runs: domain A worlds
  classed solved or frontier, domain B problems classed accepted, closed or calibration with a witness. --v1 for
  domain A runs made without RE_A_V2 (round-1 style interface names).
Review them with review_gpt.py OUT (the GPT half of the panel) and summarize with review_summary.py OUT. The paper's
other half (Opus 5.5) was run by hand from the same packets and docs/REVIEW_RUBRIC.md; its verdicts go in
runs/OUT/verdicts/opus/<id>.json with the same fields.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.common import ROOT

PKG = Path(__file__).resolve().parents[2]


def a_packet(rnd, t, v2):
    code = f"""
import json, sys
sys.path.insert(0, {str(PKG)!r})
from research_envs.discover.world import World
from research_envs.discover import prompts as P
from research_envs.discover.harness import PublicMap
from research_envs.discover.fairness import answer
t = json.loads(open({str(ROOT / 'runs')!r} + '/{rnd}/tasks/{t}/dossier.json').read())
w = World(t['world']); pm = PublicMap(w); pub = P.public_interface(w, t.get('description') or '', 40)
full = t['pilots']['full']
qs = t['queries'][:]
import random; random.Random(1).shuffle(qs)
lines = []
lines.append('# Task ' + t['id'] + ' (domain: experimental discovery in a synthetic device)')
lines.append('\\n## What the solver sees\\n\\n' + P.interface_text(pub))
lines.append('\\n## The hidden law (private)\\n\\nGenerator\\'s one-line story: ' + (t.get('story') or ''))
lines.append('\\nWorld definition (a deterministic machine; each action applies its first rule whose condition holds, assignments are simultaneous, then values are clamped or wrapped to their ranges; readouts are shown after the start and after every action; settable variables are chosen at the start, the others start at their init value):\\n')
lines.append('```json\\n' + json.dumps(t['world'], indent=1) + '\\n```')
lines.append('\\n## The withheld queries and their true answers (all of them)\\n')
for q in qs:
    pq = pm.public_query(q)
    lines.append(f"- start {{pq['start']}}, actions {{pq['actions']}} -> readouts after the last action: {{list(answer(w, q))}}")
lines.append('\\n## A strong AI solver\\'s attempt (40-experiment budget)\\n')
lines.append(f"It ran {{full.get('experiments_used')}} experiments and predicted {{round(100 * (full.get('critical_accuracy') or 0))}}% of the decisive queries correctly. Its stated law: " + (full.get('law') or '(none)'))
lines.append('\\nIts experiments (start, actions, readouts after the start and after each action):\\n')
for h in full['history']:
    lines.append(f"- {{h.get('start')}} {{h.get('actions')}} -> {{h.get('trace')}}")
print('\\n'.join(lines))
"""
    env = {**os.environ, 'RE_A_V2': '1' if v2 else '0'}
    return subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True).stdout


def b_packet(t):
    spec = t['spec']
    bk = (t.get('best_known') or {}).get('objective')
    lines = [f"# Task {t['id']} (domain: mathematical construction, object: {t['object'].replace('_', ' ')})",
             '\n## The problem as the solver sees it\n', t.get('statement') or '',
             '\nFormal spec (authoritative; expressions in a small Python subset over the object):\n',
             '```json\n' + json.dumps({k: v for k, v in spec.items() if k in ('object', 'kind', 'params', 'constraints', 'objective', 'claim')}, indent=1) + '\n```',
             '\n## Known results (private)\n',
             f"The generator's own reference construction scores {t.get('reference')}" + (f"; the best value found by any solver so far is {bk}." if bk is not None else '.') if spec.get('kind') == 'optimize'
             else 'The generator supplied a valid object (a counterexample or witness).',
             f"Strong AI solvers' outcomes: {[o['class'].replace('_', ' ') for o in t.get('outcomes', [])]}."]
    return '\n'.join(lines)


def main():
    args = [a for a in sys.argv[1:] if a != '--v1']
    v2 = '--v1' not in sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    out = ROOT / 'runs' / args[0] / 'packets'
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for rnd in args[1:]:
        for f in sorted((ROOT / 'runs' / rnd / 'tasks').glob('*/dossier.json')):
            t = json.loads(f.read_text())
            if 'world' in t and t.get('class') in ('solved', 'frontier'):
                (out / f"{t['id']}.md").write_text(a_packet(rnd, t['id'], v2))
                index.append({'id': t['id'], 'domain': 'A', 'round': rnd})
            elif 'spec' in t and t.get('class') in ('accepted', 'closed', 'calibration') and t.get('witness'):
                (out / f"{t['id']}.md").write_text(b_packet(t))
                index.append({'id': t['id'], 'domain': 'B', 'round': rnd})
    (out.parent / 'index.json').write_text(json.dumps(index, indent=1))
    print(len(index), 'packets:', sum(1 for i in index if i['domain'] == 'A'), 'A,', sum(1 for i in index if i['domain'] == 'B'), 'B')


if __name__ == '__main__':
    main()
