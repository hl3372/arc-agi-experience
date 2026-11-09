"""Round report: classes by arm, audit modes, naturalness, pilots, and duplicate families inside the round."""
import collections
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed

rd = Path(sys.argv[1])
reqs = json.loads((rd / 'requests.json').read_text())
rows = []
for r in reqs:
    f = rd / 'tasks' / r['id'] / 'dossier.json'
    t = json.loads(f.read_text()) if f.exists() else {'id': r['id'], 'demand': r['demand'], 'object': r['object']}
    rows.append(t)
arm = lambda t: f"{t.get('demand') or 'ctl'}/{'g' if t.get('object') == 'graph' else 's'}"
print(f"{len(rows)} requests; final {sum(bool(t.get('final')) for t in rows)}")
by = collections.defaultdict(collections.Counter)
for t in rows:
    by[arm(t)][t.get('class') if t.get('final') else 'pending'] += 1
for a in sorted(by):
    print(f'  {a:8s} {dict(by[a])}')
print('classes', dict(collections.Counter(t.get('class') if t.get('final') else 'pending' for t in rows)))
print('audit modes', dict(collections.Counter(t.get('audit_mode') for t in rows if t.get('demand') and t.get('final'))))
nat = [t['naturalness']['rater'].get('score') for t in rows if (t.get('naturalness') or {}).get('rater', {}).get('score')]
if nat:
    print(f'naturalness median {statistics.median(nat)} (n={len(nat)}), distribution {dict(sorted(collections.Counter(nat).items()))}')
cons = [t['naturalness']['mechanical']['constraints'] for t in rows if (t.get('naturalness') or {}).get('mechanical')]
if cons:
    print(f'constraints per task: median {statistics.median(cons)}, max {max(cons)}')
pil = collections.Counter(o['class'] for t in rows for o in t.get('outcomes', []))
print('pilot outcomes', dict(pil))
fam = collections.defaultdict(list)
for t in rows:
    fp = ((t.get('audit') or {}).get('novelty') or {}).get('fingerprint')
    if fp and sum(v is not None for v in fp) >= 3:
        fam[(t['object'], json.dumps(fp))].append(t['id'])
dups = [ids for ids in fam.values() if len(ids) > 1]
print('same exhaustive fingerprint inside the round:', dups or 'none')
print()
for t in rows:
    if not t.get('final'):
        continue
    nat = (t.get('naturalness') or {}).get('rater', {}).get('score')
    print(f"{t['id']:12s} {t.get('class'):12s} {str(t.get('audit_mode')):18s} nat {nat} ref {t.get('reference')} "
          f"pilots {[o['class'] for o in t.get('outcomes', [])]} | {t.get('title')}")


# ------------------------------------------------------------------ pre-registered bars and falsifiers (DESIGN_B 8)
def best_gap(t):
    """Does the best battery result at 10B fall short of the witness by at least max(2, 5%)? None if not measurable."""
    a = t.get('audit') or {}
    ref = t.get('reference')
    bat = a.get('battery_200000') or {}
    spec = t.get('spec') or {}
    if spec.get('kind') != 'optimize':
        if spec.get('kind') in ('satisfy', 'counterexample'):
            return not any(r.get('valid') for r in bat.values())
        return None
    if ref is None:
        return None
    vals = [r['objective'] for r in bat.values() if r.get('valid') and r.get('objective') is not None]
    if not vals:
        return True
    mx = spec['objective']['direction'] == 'max'
    best = max(vals) if mx else min(vals)
    gap = (ref - best) if mx else (best - ref)
    return gap >= max(2, 0.05 * abs(ref))


fin = [t for t in rows if t.get('final')]
pinned = [t for t in fin if t.get('demand')]
ctl = [t for t in fin if not t.get('demand')]
acc = [t for t in fin if t.get('class') == 'accepted']
print('\n== bars (pre-registered) ==')
print(f"two-sided audit passes on pinned: {sum(bool((t.get('audit') or {}).get('audit', {}).get('passes')) for t in pinned)}/{len(pinned)} (bar >= 9 of 16)")
def rate(ts):
    g = [best_gap(t) for t in ts if t.get('witness')]
    g = [x for x in g if x is not None]
    return (sum(g) / len(g) if g else float('nan')), len(g)
rp, np_ = rate(pinned); rc, nc = rate(ctl)
print(f"persistent gap to best battery at 10B: pinned {rp:.0%} (n={np_}), controls {rc:.0%} (n={nc}); bar: pinned - controls >= 25 pts")
codeok = [t for t in acc if any(o['arm'] == 'code' and o['class'] in ('matches_reference', 'beats_reference') for o in t.get('outcomes', []))]
anyok = [t for t in fin if t.get('witness') and any(o['class'] in ('matches_reference', 'beats_reference') for o in t.get('outcomes', []))]
beat = [t for t in fin if any(o['class'] == 'beats_reference' for o in t.get('outcomes', []))]
print(f"falsifier TRIVIAL: code agent reaches the witness on {len(codeok)}/{len(acc)} accepted (trips above 60%)")
print(f"  any pilot reaches or beats the reference: {len(anyok)}/{sum(1 for t in fin if t.get('witness'))} tasks with a witness; beats: {len(beat)}")
dis = [t for t in fin if t.get('witness') and ((t.get('restate') or {}).get('disagree_sibling') or (t.get('restate') or {}).get('disagree_probes'))]
print(f"falsifier AMBIGUOUS: blind-spec disagreement on {len(dis)}/{sum(1 for t in fin if t.get('witness'))} (trips above 30%)")


# ------------------------------------------------------------------ demand audits computed on controls too
from research_envs.construct.pipeline import falls_short, reaches
from research_envs.construct.evaluate import Spec

AUD = {'D2': (['anneal', 'kick'], ['symmetric']), 'D3': (['hill_climb'], ['kick'])}


def demand_pass(t, d):
    a = t.get('audit') or {}
    if not t.get('witness') or not a.get('battery_20000'):
        return None
    spec = Spec(t['spec'])
    ref = t.get('reference')
    sc, me = AUD[d]
    ok = True
    for b in ('battery_20000', 'battery_200000'):
        bat = a.get(b) or {}
        if any(m not in bat for m in sc + me):
            return None
        ok &= all(falls_short(spec, bat[m], ref) for m in sc) and any(reaches(spec, bat[m], ref) for m in me)
    return ok


print('\n== demand audits on pinned tasks vs the same audit applied to controls ==')
for d in AUD:
    pin = [demand_pass(t, d) for t in fin if t.get('demand') == d]
    con = [demand_pass(t, d) for t in ctl]
    pin, con = [x for x in pin if x is not None], [x for x in con if x is not None]
    print(f"{d}: pinned {sum(pin)}/{len(pin)}, controls {sum(con)}/{len(con)}")
