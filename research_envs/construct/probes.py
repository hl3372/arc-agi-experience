"""Probe sets and spec mutants: is a task's probe set good enough to tell two readings of the spec apart?

Probe set (per task, at the target size): the witness, the battery's best objects, single-move mutants of each
(toggle one edge or set), and random objects at several densities. Random objects alone are not enough: they
violate tight constraints, so two different specs agree on them trivially.

Spec mutants: single syntactic changes to the constraints and objective (flip a comparison, +-1 on an integer
constant, drop a constraint, drop a conjunct, all <-> any). A probe set is adequate if it distinguishes at least 90%
of the mutants from the true spec on (validity, objective). Undetected mutants are listed: some are genuinely
equivalent at this size, and those are worth a look.
"""
import ast
import copy
import itertools
import random
from .evaluate import Spec
from .lang import SpecError
from .search import Problem
from .domain import to_json


def probe_set(raw, objects, hint=None, seed=0, random_count=200, cap=3000):
    spec = Spec(raw)
    p = Problem(spec, hint, max_evals=10 ** 12)
    rng = random.Random(seed)
    out, seen = [], set()

    def add(chosen):
        key = frozenset(chosen)
        if key not in seen and len(out) < cap:
            seen.add(key)
            out.append(to_json(spec.object, p.parsed(key)))

    bases = []
    for obj in objects:
        try:
            bases.append(p.chosen_from(obj))
        except Exception:
            continue
    for b in bases:
        add(b)
    for b in bases:
        for i in range(len(p.universe)):
            add(b ^ {i})
    for b in bases:  # a few two-move mutants as well
        for _ in range(min(200, len(p.universe))):
            i, j = rng.sample(range(len(p.universe)), 2)
            add(b ^ {i, j})
    from .search import neighbour
    for b in bases:  # structure-preserving chains (degree-preserving swaps, element replacements): they keep
        cur = b      # regularity-type assumptions, so claims and objectives are compared on valid-looking objects
        for k in range(400):
            cur = neighbour(p, b if k % 4 == 0 else cur, rng)
            add(cur)
    for k in range(random_count):
        dens = (k % 10 + 0.5) / 10
        add(frozenset(i for i in range(len(p.universe)) if rng.random() < dens))
    add(frozenset()); add(frozenset(range(len(p.universe))))
    if spec.object == 'set_system':  # change set sizes too: the public task allows any sets, whatever the hint says
        subs = []
        for obj in objects:  # small sub-families (one or two sets) isolate per-set constraints from pairwise ones
            sets = [list(x) for x in obj.get('sets', [])]
            subs += [{'m': obj['m'], 'sets': [st]} for st in sets[:6]]
            subs += [{'m': obj['m'], 'sets': [a, b]} for a, b in itertools.combinations(sets[:6], 2)]
        for obj in list(objects) + subs:
            sets = [list(x) for x in obj.get('sets', [])]
            for i, st in enumerate(sets):
                for x in range(obj['m']):
                    new = sorted(set(st) ^ {x})
                    fam = sets[:i] + ([new] if new else []) + sets[i + 1:]
                    if len({tuple(f) for f in fam}) == len(fam) and len(out) < cap:
                        out.append({'m': obj['m'], 'sets': fam})
    return out


def violated_alone(raw, probes):
    """Per constraint: how many probes violate that constraint and no other (0 suggests it is implied)."""
    spec = Spec(raw)
    counts = {n: 0 for n, _, _ in spec.constraints}
    for obj in probes:
        r = spec.evaluate(obj)
        failing = set(r['violations']) | (set(r['errors']) & set(counts))
        if not r['format_error'] and len(failing) == 1:
            counts[next(iter(failing))] += 1
    return counts


def verdicts(raw, probes):
    spec = Spec(raw)
    out = []
    for obj in probes:
        r = spec.evaluate(obj)
        out.append((r['valid'], r['objective'] if r['valid'] else None))
    return out


class _Mutator(ast.NodeTransformer):
    """Applies the k-th applicable mutation and records its description."""

    FLIPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}

    def __init__(self, target):
        self.target, self.count, self.desc = target, 0, None

    def _hit(self, desc):
        hit = self.count == self.target
        self.count += 1
        if hit:
            self.desc = desc
        return hit

    def visit_Compare(self, node):
        self.generic_visit(node)
        if type(node.ops[0]) in self.FLIPS and self._hit(f'flip {type(node.ops[0]).__name__}'):
            node.ops[0] = self.FLIPS[type(node.ops[0])]()
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            if self._hit(f'{node.value} -> {node.value + 1}'):
                return ast.copy_location(ast.Constant(node.value + 1), node)
            if self._hit(f'{node.value} -> {node.value - 1}'):
                return ast.copy_location(ast.Constant(node.value - 1), node)
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.And) and len(node.values) > 1:
            for i in range(len(node.values)):
                if self._hit(f'drop conjunct {i}'):
                    vals = node.values[:i] + node.values[i + 1:]
                    return vals[0] if len(vals) == 1 else ast.BoolOp(op=ast.And(), values=vals)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ('all', 'any') and self._hit(f'{node.func.id} -> other'):
            node.func = ast.Name(id='any' if node.func.id == 'all' else 'all', ctx=ast.Load())
        return node


def mutants(raw):
    """Every single-change mutant of the spec's constraints and objective, as (description, raw spec)."""
    out = []
    cons = raw.get('constraints', [])
    for i, c in enumerate(cons):
        out.append((f'drop constraint {c["name"]}', {**raw, 'constraints': cons[:i] + cons[i + 1:]}))
    fields = [('constraint', i, c['expr']) for i, c in enumerate(cons)]
    if raw.get('objective'):
        fields.append(('objective', None, raw['objective']['expr']))
    if raw.get('claim'):
        fields.append(('claim', None, raw['claim']['expr']))
    for where, i, src in fields:
        tree = ast.parse(src, mode='eval')
        k = 0
        while True:
            m = _Mutator(k)
            new = m.visit(copy.deepcopy(tree))
            if m.desc is None:
                break
            k += 1
            mutated = ast.unparse(ast.fix_missing_locations(new))
            if where == 'constraint':
                spec = {**raw, 'constraints': [dict(c, expr=mutated) if j == i else c for j, c in enumerate(cons)]}
                label = f'{cons[i]["name"]}: {m.desc}'
            else:
                spec = {**raw, where: {**raw[where], 'expr': mutated}}
                label = f'{where}: {m.desc}'
            try:
                Spec(spec)
            except SpecError:
                continue
            out.append((label, spec))
    return out


def _mutant_killed(job):
    spec, probes, base = job
    try:
        return verdicts(spec, probes) != base
    except Exception:
        return True            # a mutant that breaks evaluation is detected


def adequacy(raw, probes, processes=18):
    """Fraction of spec mutants whose verdicts on the probe set differ from the true spec's."""
    base = verdicts(raw, probes)
    muts = mutants(raw)
    jobs = [(spec, probes, base) for _, spec in muts]
    if len(jobs) > 1 and processes > 1:
        from multiprocessing import Pool
        with Pool(min(processes, len(jobs))) as pool:
            killed_flags = pool.map(_mutant_killed, jobs)
    else:
        killed_flags = [_mutant_killed(j) for j in jobs]
    survivors = [label for (label, _), k in zip(muts, killed_flags) if not k]
    killed = sum(killed_flags)
    return {'mutants': len(muts), 'killed': killed, 'score': killed / len(muts) if muts else 1.0,
            'survivors': survivors}


def compare(raw_a, raw_b, probes):
    """Disagreements between two specs (e.g. ours and a blind restatement) on the probe set, on (validity, objective
    when valid). Probes on which spec b cannot be evaluated (and spec a can) are counted apart, as b_errors."""
    sa, sb = Spec(raw_a), Spec(raw_b)
    diffs, b_errors = [], 0
    for i, obj in enumerate(probes):
        ra, rb = sa.evaluate(obj), sb.evaluate(obj)
        if rb['errors'] and not ra['errors']:
            b_errors += 1
            continue
        a = (ra['valid'], ra['objective'] if ra['valid'] else None)
        b = (rb['valid'], rb['objective'] if rb['valid'] else None)
        if a != b:
            diffs.append((i, a, b))
    return {'probes': len(probes), 'disagree': len(diffs), 'b_errors': b_errors,
            'examples': [{'object': probes[i], 'ours': a, 'theirs': b} for i, a, b in diffs[:5]]}


# ------------------------------------------------------------------ targeted probes for surviving mutants

def _differ_spec(raw, mutant):
    """A satisfy spec whose valid objects are exactly where the mutant and the true spec can disagree: the shared
    constraints hold, and the changed part (a dropped or altered constraint, the objective, or the claim) differs."""
    base = {k: v for k, v in raw.items() if k in ('object', 'params', 'given')}
    rc = {c['name']: c['expr'] for c in raw.get('constraints', [])}
    mc = {c['name']: c['expr'] for c in mutant.get('constraints', [])}
    shared = [{'name': n, 'expr': e} for n, e in rc.items() if mc.get(n) == e]
    changed = [n for n in rc if mc.get(n) != rc[n]]
    if changed:
        n = changed[0]
        differ = f'not ({rc[n]})' if n not in mc else f'({rc[n]}) != ({mc[n]})'
    elif raw.get('objective') and mutant.get('objective', {}).get('expr') != raw['objective']['expr']:
        differ = f"({raw['objective']['expr']}) != ({mutant['objective']['expr']})"
    elif raw.get('claim') and mutant.get('claim', {}).get('expr') != raw['claim']['expr']:
        differ = f"({raw['claim']['expr']}) != ({mutant['claim']['expr']})"
    else:
        return None
    return {**base, 'kind': 'satisfy', 'constraints': shared + [{'name': 'differ', 'expr': differ}]}


def _widen(hint, m):
    sizes = (hint or {}).get('set_sizes')
    if not sizes:
        return hint
    wide = sorted({k for s in sizes for k in (s - 1, s, s + 1) if 0 <= k <= m})
    return {**hint, 'set_sizes': wide}


EQUIV_MAX_BITS = 16      # the equivalence-at-sibling check enumerates at most 2**16 objects per mutant


def _target_one(job):
    """One surviving mutant: ('killed', object) if a short search finds an object the two specs judge differently,
    ('equivalent', None) if they agree on every object at the sibling size, else ('survives', None)."""
    raw, mut, hint, sibling_params, evals, seconds = job
    from . import search, sibling
    spec, ms = Spec(raw), Spec(mut)
    d = _differ_spec(raw, mut)
    if d is not None:
        m = raw['params'].get('m', 0)
        h = _widen(hint, m) if raw['object'] == 'set_system' else hint
        for method in ('anneal', 'kick'):
            for seed in (0, 1):
                try:
                    r = search.run_method((method, d, h, seed, evals, seconds))
                except Exception:
                    continue
                o = r.get('object')
                if r.get('valid') and o:
                    a, b = spec.evaluate(o), ms.evaluate(o)
                    if (a['valid'], a['objective'] if a['valid'] else None) != (b['valid'], b['objective'] if b['valid'] else None):
                        return 'killed', o
    if sibling_params:
        try:
            sib = sibling.at_size(raw, **sibling_params)
            n_univ = len(search.Problem(Spec(sib), hint, max_evals=1).universe)
            if n_univ > EQUIV_MAX_BITS:   # an exhaustive check would take too long in one process: stays a survivor
                return 'survives', None
            r = sibling.enumerate_all(sib, hint, other=sibling.at_size(mut, **sibling_params), processes=1)
            if not r.get('skipped') and r.get('disagree') == 0:
                return 'equivalent', None
        except Exception:
            pass
    return 'survives', None


def targeted_adequacy(raw, probes, hint=None, sibling_params=None, evals=4000, seconds=20, processes=7):
    """Adequacy after a targeted search per surviving mutant (in parallel): survivors that a short search can tell
    apart become killed; survivors that agree with the true spec on every object at the sibling size are counted as
    equivalent and leave the denominator; the rest still survive."""
    from multiprocessing import Pool
    first = adequacy(raw, probes, processes=processes)
    muts = dict(mutants(raw))
    labels = [l for l in first['survivors'] if l in muts]
    jobs = [(raw, muts[l], hint, sibling_params, evals, seconds) for l in labels]
    if jobs:
        with Pool(min(processes, len(jobs))) as pool:
            res = pool.map(_target_one, jobs)
    else:
        res = []
    killed = first['killed'] + sum(r[0] == 'killed' for r in res)
    equivalent = [l for l, r in zip(labels, res) if r[0] == 'equivalent']
    survive = [l for l, r in zip(labels, res) if r[0] == 'survives']
    denom = first['mutants'] - len(equivalent)
    return {'mutants': first['mutants'], 'killed': killed, 'equivalent_at_sibling': equivalent,
            'survivors': survive, 'score': killed / denom if denom else 1.0,
            'targeted_objects': sum(r[0] == 'killed' for r in res), 'plain_score': first['score']}
