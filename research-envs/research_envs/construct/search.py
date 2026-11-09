"""The baseline battery for domain B: executable shortcuts and demand-suited methods, under fixed evaluation budgets.

Every method sees only the spec through the trusted evaluator, never the witness. It works on a fixed element universe:
- graph with fixed n: the n(n-1)/2 possible edges;
- set system with fixed m: candidate subsets (all, or the sizes in a private search hint).

Search scores are lexicographic, (graded violation, badness): graded violation counts failing instances inside each
`all(...)` and distances from comparison bounds; badness is the objective oriented so lower is better (counterexample
specs use the graded margin of falsifying the claim). Validity and the reported objective always come from the
evaluator's exact verdict.

Methods (name -> what it is for):
  random, greedy_add, greedy_delete       naive shortcuts
  hill_climb                               D3 shortcut: 1- and 2-move local search
  kick                                     D3 method: iterated local search with random kicks
  anneal                                   unrestricted search (D2 shortcut, general yardstick)
  symmetric                                D2 method: search over circulants / cyclic orbit unions only
  relax_repair                             D1 shortcut: optimize with one constraint dropped, then repair
  margin                                   D4 shortcut: anneal on the margin of falsifying the claim
  exhaustive                               certified optimum on tiny universes (sibling sizes)
"""
import ast
import itertools
import math
import random
import time
from multiprocessing import Pool
import networkx as nx
from .domain import to_json, LIMITS
from .evaluate import Spec
from .lang import _grade, BUILTINS, _Validator

BIG = 10 ** 6


class Budget(Exception):
    pass


class Problem:
    def __init__(self, spec, hint=None, max_evals=10 ** 5, max_seconds=None):
        self.spec, self.hint, self.max_evals = spec, hint or {}, max_evals
        self.deadline = time.time() + max_seconds if max_seconds else None
        self.time_capped = False
        if spec.object == 'graph':
            self.size = spec.params.get('n')
            if self.size is None:
                raise ValueError('search needs a fixed n')
            self.universe = list(itertools.combinations(range(self.size), 2))
            self.index = {e: i for i, e in enumerate(self.universe)}
        else:
            self.size = spec.params.get('m')
            if self.size is None:
                raise ValueError('search needs a fixed m')
            sizes = list(self.hint.get('set_sizes') or range(0, self.size + 1))
            total = sum(math.comb(self.size, k) for k in sizes)
            if total <= 500000:
                self.universe = [frozenset(c) for k in sizes for c in itertools.combinations(range(self.size), k)]
            elif self.hint.get('sample'):
                # too many candidate sets to list: a fixed random sample of them, plus the sets named in 'include'
                rng = random.Random(0)
                pool = {frozenset(x) for x in self.hint.get('include', []) if all(0 <= e < self.size for e in x)}
                while len(pool) < self.hint['sample']:
                    k = sizes[rng.randrange(len(sizes))]
                    pool.add(frozenset(rng.sample(range(self.size), k)))
                self.universe = sorted(pool, key=lambda f: (len(f), sorted(f)))
            else:
                raise ValueError('set universe too large: give set_sizes in the search hint')
        self.index = {e: i for i, e in enumerate(self.universe)}
        self.evals = 0
        self.best = None          # (score, chosen, verdict) of the best valid object seen
        self._margin = None

    def parsed(self, chosen):
        if self.spec.object == 'graph':
            return (self.size, frozenset(self.universe[i] for i in chosen))
        return (self.size, tuple(sorted((self.universe[i] for i in chosen), key=lambda f: (len(f), sorted(f)))))

    def score(self, chosen):
        if self.evals >= self.max_evals:
            raise Budget()
        if self.deadline and self.evals % 32 == 0 and time.time() > self.deadline:
            self.time_capped = True
            raise Budget()
        self.evals += 1
        if self.spec.object == 'set_system' and len(chosen) > LIMITS['set_family']:
            # outside the answer format: invalid, and the further outside the worse
            return (BIG + len(chosen), BIG), {'valid': False, 'violations': [], 'errors': {'format': 'too many sets'},
                                               'objective': None, 'claim_holds': None}
        parsed = self.parsed(chosen)
        r = self.spec.evaluate_parsed(parsed)
        if self.spec.kind == 'counterexample':
            viol = 0 if not (r['violations'] or r['errors']) else self.spec.violation_degree(parsed)
            bad = 0 if r['claim_holds'] is False else 1 + self.margin(parsed)
        else:
            viol = 0 if not (r['violations'] or r['errors']) else self.spec.violation_degree(parsed)
            if self.spec.kind == 'optimize':
                obj = r['objective']
                bad = BIG if obj is None else (-obj if self.spec.objective[1] == 'max' else obj)
            else:
                bad = 0
        sc = (viol, bad)
        if r['valid'] and (self.best is None or sc < self.best[0]):
            self.best = (sc, frozenset(chosen), r)
        return sc, r

    def margin(self, parsed):
        """Graded distance to falsifying the claim (0 when it fails)."""
        if self._margin is None:
            self._margin = negated_graded(self.spec.raw['claim']['expr'], self.spec.names)
        from .lang import run, timed
        from .domain import OBJECTS
        env = OBJECTS[self.spec.object]['env'](parsed); env.update(self.spec.given)
        try:
            with timed(self.spec.timeout):
                return max(0, int(run(self._margin, env)))
        except Exception:
            return BIG

    def result(self, name, t0, extra=None):
        b = self.best
        return {'method': name, 'valid': b is not None,
                'objective': b[2]['objective'] if b else None,
                'object': to_json(self.spec.object, self.parsed(b[1])) if b else None,
                'evals': self.evals, 'seconds': round(time.time() - t0, 2), **(extra or {})}

    def chosen_from(self, obj):
        items = self.spec.parse(obj)[1]
        if any(x not in self.index for x in items):
            raise ValueError('object uses elements outside the search universe')
        return frozenset(self.index[x] for x in items)


def negated_graded(src, names):
    """A graded measure of how far an object is from making the boolean expression `src` False."""
    tree = ast.parse(src.strip(), mode='eval')
    _Validator(set(names) | set(BUILTINS)).visit(tree)
    node = tree.body
    flip = {ast.LtE: ast.Gt, ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.Gt: ast.LtE, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in flip:
        neg = ast.Compare(left=node.left, ops=[flip[type(node.ops[0])]()], comparators=node.comparators)
    else:
        neg = ast.UnaryOp(op=ast.Not(), operand=node)
    expr = ast.Expression(body=_grade(neg))
    ast.fix_missing_locations(expr)
    _Validator(set(names) | set(BUILTINS), private_ok={'_numeric'}).visit(expr)
    return compile(expr, '<margin>', 'eval')


# ------------------------------------------------------------------ methods (each runs until done or out of budget)

def m_random(p, rng):
    while True:
        p.score(_random_start(p, rng))


def neighbour(p, chosen, rng):
    """A random move: toggle one element, or a structure-preserving move. For graphs, a double-edge swap
    (ab, cd -> ac, bd) that keeps every degree; for set systems, replacing one element of one set (keeps set sizes)."""
    if not chosen or rng.random() < 0.5:
        return chosen ^ {rng.randrange(len(p.universe))}
    items = list(chosen)
    if p.spec.object == 'graph':
        for _ in range(10):
            i, j = rng.sample(items, 2) if len(items) > 1 else (items[0], items[0])
            (a, b), (c, d) = p.universe[i], p.universe[j]
            if len({a, b, c, d}) < 4:
                continue
            if rng.random() < 0.5:
                c, d = d, c
            e1, e2 = (min(a, c), max(a, c)), (min(b, d), max(b, d))
            k1, k2 = p.index[e1], p.index[e2]
            if k1 in chosen or k2 in chosen:
                continue
            return (chosen - {i, j}) | {k1, k2}
        return chosen ^ {rng.randrange(len(p.universe))}
    for _ in range(10):
        i = rng.choice(items)
        st = p.universe[i]
        if not st:
            continue
        x = rng.choice(sorted(st))
        y = rng.randrange(p.size)
        if y in st:
            continue
        k = p.index.get((st - {x}) | {y})
        if k is not None and k not in chosen:
            return (chosen - {i}) | {k}
    return chosen ^ {rng.randrange(len(p.universe))}


def _greedy(p, rng, mode):
    while True:
        chosen = frozenset() if mode == 'add' else (frozenset(range(len(p.universe))) if len(p.universe) <= 3000
                                                    else _random_start(p, rng))
        cur, _ = p.score(chosen)
        while True:
            moves = [i for i in range(len(p.universe)) if (i not in chosen) == (mode == 'add')] \
                if len(p.universe) <= 3000 else [rng.randrange(len(p.universe)) for _ in range(2000)]
            moves = [i for i in moves if (i not in chosen) == (mode == 'add')]
            rng.shuffle(moves)
            step = None
            for i in moves:
                cand = chosen | {i} if mode == 'add' else chosen - {i}
                sc, _ = p.score(cand)
                if sc < cur and (step is None or sc < step[0]):
                    step = (sc, cand)
            if step is None:
                break
            cur, chosen = step


def m_greedy_add(p, rng):
    _greedy(p, rng, 'add')


def m_greedy_delete(p, rng):
    _greedy(p, rng, 'delete')


def _local_opt(p, rng, chosen, cur, patience=None):
    """First-improvement local search over sampled moves (toggles, swaps, structure-preserving moves) until
    `patience` consecutive moves fail to improve."""
    patience = patience or min(4 * len(p.universe), 1500)
    fails = 0
    while fails < patience:
        cand = neighbour(p, chosen, rng)
        if rng.random() < 0.25 and chosen:  # a plain swap: remove one element, add another
            cand = (chosen - {rng.choice(list(chosen))}) | {rng.randrange(len(p.universe))}
        sc, _ = p.score(cand)
        if sc < cur:
            chosen, cur, fails = cand, sc, 0
        else:
            fails += 1
    return chosen, cur


def _sparse_start(p, rng):
    """Empty, or a sparse random object (most constraints of the max-degree kind hold for sparse objects)."""
    if rng.random() < 0.5:
        return frozenset()
    dens = rng.random() * min(0.2, 20 / max(1, len(p.universe)) * 5)
    return frozenset(i for i in range(len(p.universe)) if rng.random() < dens)


def _random_start(p, rng):
    if p.spec.object == 'set_system':  # families of a few times m sets, not half of a huge universe
        k = rng.randint(0, min(len(p.universe) // 2, 6 * p.size))
        return frozenset(rng.sample(range(len(p.universe)), k))
    dens = rng.random() * 0.5
    return frozenset(i for i in range(len(p.universe)) if rng.random() < dens)


def _start(p, rng):
    """Start style per run, so the seeds of one method form a small portfolio: even seeds start from random objects of
    any density, odd seeds from empty or sparse ones (dense-favouring and sparse-favouring specs both get a fair try)."""
    style = getattr(p, 'start_style', 'mixed')
    if style == 'mixed':
        style = 'random' if rng.random() < 0.5 else 'sparse'
    return _random_start(p, rng) if style == 'random' else _sparse_start(p, rng)


def m_hill_climb(p, rng):
    """Restarted local search (1- and 2-move, sampled) from empty or sparse starts."""
    while True:
        start = _start(p, rng)
        _local_opt(p, rng, start, p.score(start)[0])


def m_kick(p, rng):
    """Iterated local search: a local optimum, then random kicks of several moves, keeping the result if not worse."""
    chosen = _start(p, rng)
    chosen, cur = _local_opt(p, rng, chosen, p.score(chosen)[0])
    while True:
        kicked = chosen
        for _ in range(rng.randint(3, max(4, len(chosen) // 5 + 3))):
            kicked = neighbour(p, kicked, rng)
        cand, sc = _local_opt(p, rng, kicked, p.score(kicked)[0])
        if sc <= cur:
            chosen, cur = cand, sc


def _weighted(p, sc):
    v, b = sc
    return p.penalty * v + (b if b != BIG else p.penalty)


def m_anneal(p, rng, iters=20000):
    p.penalty = max(2, len(p.universe))
    while True:
        chosen = _start(p, rng)
        cur = _weighted(p, p.score(chosen)[0])
        temp0 = max(1.0, p.penalty / 2)
        for it in range(iters):
            temp = temp0 * (1 - it / iters) + 1e-3
            cand = neighbour(p, chosen, rng)
            nxt = _weighted(p, p.score(cand)[0])
            if nxt <= cur or rng.random() < math.exp((cur - nxt) / temp):
                chosen, cur = cand, nxt


def orbits(p):
    """Orbits of the universe under the cyclic group Z_size (vertex or ground-element relabelling x -> x+1)."""
    index = {e: i for i, e in enumerate(p.universe)}
    seen, out = set(), []
    for i, e in enumerate(p.universe):
        if i in seen:
            continue
        orb, cur, whole = set(), e, True
        for _ in range(p.size):
            if p.spec.object == 'graph':
                cur = tuple(sorted(((cur[0] + 1) % p.size, (cur[1] + 1) % p.size)))
            else:
                cur = frozenset((x + 1) % p.size for x in cur)
            if cur not in index:          # a sampled universe may hold only part of an orbit: skip it
                whole = False
                break
            orb.add(index[cur])
        seen.add(i)
        if whole:
            seen |= orb
            out.append(frozenset(orb))
    return out


def m_symmetric(p, rng, iters=20000):
    """Search only over objects invariant under Z_size: unions of cyclic orbits (circulant graphs, cyclically
    developed set systems). Exhaustive over orbit subsets when there are at most 16 orbits, annealing otherwise."""
    orbs = orbits(p)
    def obj(sel):
        return frozenset().union(*[orbs[j] for j in sel]) if sel else frozenset()
    if len(orbs) <= 16:
        for mask in range(2 ** len(orbs)):
            p.score(obj([j for j in range(len(orbs)) if mask >> j & 1]))
        raise Budget()
    p.penalty = max(2, len(p.universe))
    while True:
        if p.spec.object == 'set_system':  # a handful of orbits (each about m sets), not 30% of a huge universe
            sel = frozenset(rng.sample(range(len(orbs)), rng.randint(0, min(len(orbs), 8))))
        else:
            sel = frozenset(j for j in range(len(orbs)) if rng.random() < 0.3)
        cur = _weighted(p, p.score(obj(sel))[0])
        for it in range(iters):
            temp = max(1.0, p.penalty / 2) * (1 - it / iters) + 1e-3
            cand = sel ^ {rng.randrange(len(orbs))}
            nxt = _weighted(p, p.score(obj(cand))[0])
            if nxt <= cur or rng.random() < math.exp((cur - nxt) / temp):
                sel, cur = cand, nxt


def m_relax_repair(p, rng):
    """For each constraint: optimize with it dropped (annealing on the relaxed spec), then repair against the full
    spec by local search. Shortcut for D1: treating the constraints one at a time."""
    raw = p.spec.raw
    names = [c['name'] for c in raw.get('constraints', [])]
    share = max(1000, p.max_evals // (2 * max(1, len(names))))
    while True:
        for drop in names:
            relaxed = Spec({**raw, 'constraints': [c for c in raw['constraints'] if c['name'] != drop]}, p.spec.timeout)
            left = (p.deadline - time.time()) if p.deadline else None
            if left is not None and left <= 0:
                p.time_capped = True
                raise Budget()
            q = Problem(relaxed, p.hint, max_evals=share, max_seconds=left)
            try:
                m_anneal(q, rng, iters=max(500, share // 4))
            except Budget:
                pass
            start = q.best[1] if q.best else _random_start(p, rng)
            p.evals += q.evals
            _local_opt(p, rng, start, p.score(start)[0])


def m_margin(p, rng):
    """Annealing on (violation of the assumptions, margin of falsifying the claim): the D4 shortcut."""
    m_anneal(p, rng)


def m_exhaustive(p, rng):
    for mask in range(2 ** len(p.universe)):
        p.score(frozenset(i for i in range(len(p.universe)) if mask >> i & 1))
    raise Budget()


METHODS = {'random': m_random, 'greedy_add': m_greedy_add, 'greedy_delete': m_greedy_delete,
           'hill_climb': m_hill_climb, 'kick': m_kick, 'anneal': m_anneal, 'symmetric': m_symmetric,
           'relax_repair': m_relax_repair, 'margin': m_margin, 'exhaustive': m_exhaustive}


def run_method(job):
    """One (method, spec, hint, seed, budget) job; used directly or in a process pool."""
    name, raw, hint, seed, budget = job[:5]
    max_seconds = job[5] if len(job) > 5 else None
    spec = Spec(raw)
    p, t0 = Problem(spec, hint, max_evals=budget, max_seconds=max_seconds), time.time()
    p.start_style = 'random' if seed % 2 == 0 else 'sparse'
    if name == 'exhaustive' and 2 ** len(p.universe) > budget:
        return {'method': name, 'skipped': f'universe of {len(p.universe)} elements exceeds the budget'}
    try:
        METHODS[name](p, random.Random(seed))
    except Budget:
        pass
    out = p.result(name, t0, {'seed': seed, 'budget': budget, 'time_capped': p.time_capped})
    if name == 'exhaustive':
        out['certified'] = p.evals == 2 ** len(p.universe)
    return out


def battery(raw, hint=None, methods=None, budget=10 ** 5, seeds=(0, 1, 2, 3), processes=None, max_seconds=None):
    """Run each method with each seed in parallel; returns the best valid result per method (all runs kept).
    max_seconds caps each run's wall clock (a run that hits it reports time_capped)."""
    methods = methods or [m for m in METHODS if m != 'exhaustive']
    jobs = [(m, raw, hint, s, budget, max_seconds) for m in methods for s in (seeds if m not in ('exhaustive',) else (0,))]
    with Pool(processes or min(len(jobs), 18)) as pool:
        runs = pool.map(run_method, jobs)
    spec = Spec(raw)
    best = {}
    for r in runs:
        if r.get('skipped'):
            best.setdefault(r['method'], r)
            continue
        cur = best.get(r['method'])
        if cur is None or cur.get('skipped') or (r['valid'] and (not cur['valid'] or _beats(spec, r, cur))):
            best[r['method']] = r
    return {'best': best, 'runs': runs}


def _beats(spec, a, b):
    if spec.kind != 'optimize':
        return False
    return spec.better(a['objective'], b['objective'])


# ------------------------------------------------------------------ distinct solutions up to isomorphism

def as_nx(kind, obj):
    if kind == 'graph':
        g = nx.Graph(); g.add_nodes_from(range(obj['n'])); g.add_edges_from(map(tuple, obj['edges']))
        return g
    g = nx.Graph()  # incidence graph: elements on one side, sets on the other
    g.add_nodes_from((('e', x) for x in range(obj['m'])), side=0)
    for i, s in enumerate(obj['sets']):
        g.add_node(('s', i), side=1)
        g.add_edges_from((('e', x), ('s', i)) for x in s)
    return g


def distinct(kind, objects):
    """One representative per isomorphism class. The WL hash only buckets; an exact check decides."""
    match = nx.algorithms.isomorphism.categorical_node_match('side', None)
    reps = []
    for obj in objects:
        g = as_nx(kind, obj)
        h = nx.weisfeiler_lehman_graph_hash(g, node_attr='side' if kind != 'graph' else None)
        if not any(rh == h and nx.is_isomorphic(g, rg, node_match=match if kind != 'graph' else None)
                   for rh, rg, _ in reps):
            reps.append((h, g, obj))
    return [o for _, _, o in reps]
