"""The small sibling instance: the same spec at a size small enough to enumerate every object.

Exhaustive enumeration gives, at that size:
- the certified optimum (or proof that no valid object exists) and the number of valid objects;
- every optimal object, one per isomorphism class (true uniqueness, not just "the one we found");
- redundancy: for each constraint, whether some object violates that constraint alone (if none does, the
  constraint is implied by the others at this size);
- an exhaustive comparison of two specs (ours and a blind restatement) on validity and objective value.
"""
import json
from multiprocessing import Pool
from .evaluate import Spec
from .search import Problem, distinct
from .domain import to_json

MAX_OBJECTS = 1 << 22


def at_size(raw, **params):
    """The spec with its size params replaced (e.g. n=6)."""
    return {**raw, 'params': {**raw.get('params', {}), **params}}


def _chunk(job):
    raw, hint, lo, hi, other = job
    spec = Spec(raw)
    alt = Spec(other) if other else None
    p = Problem(spec, hint, max_evals=10 ** 12)
    names = [n for n, _, _ in spec.constraints]
    out = {'valid': 0, 'best': None, 'best_masks': [], 'alone': {n: 0 for n in names}, 'disagree': 0,
           'disagree_examples': []}
    for mask in range(lo, hi):
        chosen = frozenset(i for i in range(len(p.universe)) if mask >> i & 1)
        parsed = p.parsed(chosen)
        r = spec.evaluate_parsed(parsed)
        failing = set(r['violations']) | set(k for k in r['errors'] if k in out['alone'])
        if len(failing) == 1:
            out['alone'][next(iter(failing))] += 1
        if r['valid']:
            out['valid'] += 1
            key = r['objective'] if spec.kind == 'optimize' else 0
            if out['best'] is None or (spec.kind == 'optimize' and spec.better(key, out['best'])):
                out['best'], out['best_masks'] = key, [mask]
            elif key == out['best'] and len(out['best_masks']) < 2000:
                out['best_masks'].append(mask)
        if alt is not None:
            ra = alt.evaluate_parsed(parsed)
            # objectives are compared only on valid objects (two specs may write the objective differently and agree
            # wherever it matters); an object on which the other spec cannot be evaluated is counted apart
            if ra['errors'] and not r['errors']:
                out['other_errors'] = out.get('other_errors', 0) + 1
            elif (r['valid'], r['objective'] if r['valid'] else None) != (ra['valid'], ra['objective'] if ra['valid'] else None):
                out['disagree'] += 1
                if len(out['disagree_examples']) < 5:
                    out['disagree_examples'].append({'object': to_json(spec.object, parsed),
                                                     'ours': [r['valid'], r['objective']],
                                                     'theirs': [ra['valid'], ra['objective']]})
    return out


def enumerate_all(raw, hint=None, other=None, processes=18):
    """Exhaust the sibling instance `raw` (and compare with spec `other` if given)."""
    spec = Spec(raw)
    p = Problem(spec, hint, max_evals=1)
    total = 2 ** len(p.universe)
    if total > MAX_OBJECTS:
        return {'skipped': f'{total} objects exceed the {MAX_OBJECTS} cap'}
    step = max(1, total // (processes * 4))
    jobs = [(raw, hint, lo, min(total, lo + step), other) for lo in range(0, total, step)]
    if processes <= 1:            # no pool (callable from inside a pool worker)
        parts = [_chunk(j) for j in jobs]
    else:
        with Pool(min(processes, len(jobs))) as pool:
            parts = pool.map(_chunk, jobs)
    best, masks = None, []
    for part in parts:
        if part['best'] is None:
            continue
        if best is None or (spec.kind == 'optimize' and spec.better(part['best'], best)):
            best, masks = part['best'], list(part['best_masks'])
        elif part['best'] == best:
            masks += part['best_masks']
    optimal = [to_json(spec.object, p.parsed(frozenset(i for i in range(len(p.universe)) if m >> i & 1)))
               for m in masks[:2000]]
    alone = {n: sum(part['alone'][n] for part in parts) for n in parts[0]['alone']}
    out = {'objects': total, 'valid': sum(part['valid'] for part in parts),
           'optimum': best if spec.kind == 'optimize' else None,
           'optimal_classes': len(distinct(spec.object, optimal)) if optimal else 0,
           'optimal_examples': distinct(spec.object, optimal)[:3],
           'redundant_constraints': sorted(n for n, c in alone.items() if c == 0),
           'violated_alone_counts': alone}
    if other is not None:
        out['disagree'] = sum(part['disagree'] for part in parts)
        out['other_errors'] = sum(part.get('other_errors', 0) for part in parts)
        out['disagree_examples'] = [e for part in parts for e in part['disagree_examples']][:5]
    return out
