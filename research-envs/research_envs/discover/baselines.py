"""Public-information baselines for domain A: experimenters that never see the law, and learners that predict
withheld queries from the traces they collected.

Experimenters (each returns a list of experiments (start settings, input list) within a budget):
- sweep: one factor at a time from a few space-filling starts, all action pairs, single actions repeated to the
  length limit;
- random: the public random sampler;
- given: a fixed list (for example the planner's demand-suited plan), truncated to the budget.

Learners (fit on traces, then predict the final readouts of a query):
- persistence: the start's readout, unchanged (the chance-level reference);
- window(w_obs, w_act): next readout from the last w_obs readouts and the last w_act actions; unseen contexts back
  off to shorter windows, then persistence;
- merge: a passive Moore-machine learner (red-blue state merging over the prefix tree of observed traces);
  unseen transitions keep the state.
Unseen starts map to the nearest seen start by settings distance.
"""
import itertools
import random


# ------------------------------------------------------------------ experimenters

def sweep(w, budget, rng=None):
    rng = rng or random.Random(0)
    starts = w.starts()
    picks = [starts[0], starts[-1], starts[len(starts) // 2]] + rng.sample(starts, min(len(starts), 5))
    seen, out = set(), []

    def add(st, seq):
        key = (tuple(sorted(st.items())), tuple(seq))
        if key not in seen and len(out) < budget:
            seen.add(key)
            out.append((st, list(seq)))
    n = len(w.inputs)
    for st in picks:                                   # each action repeated to the limit, from each start
        for i in range(n):
            add(st, [i] * w.max_len)
    for st in picks[:2]:                               # all ordered pairs
        for i, j in itertools.product(range(n), repeat=2):
            add(st, [i, j])
    for st in starts:                                  # space-filling starts with a mixed sequence
        add(st, [(t % n) for t in range(w.max_len)])
    tries = 0
    while len(out) < budget and tries < 50 * budget:   # fill the budget: long mixed sequences from spread starts
        tries += 1
        add(starts[rng.randrange(len(starts))], [rng.randrange(n) for _ in range(w.max_len)])
    return out


def random_experiments(w, budget, rng):
    from .fairness import sample_experiment
    return [sample_experiment(w, rng) for _ in range(budget)]


def collect(w, experiments):
    return [(st, seq, w.run_inputs(w.start_state(st), seq)) for st, seq in experiments]


# ------------------------------------------------------------------ learners

def _nearest_start(st, seen_starts):
    def dist(a):
        return sum(abs(a.get(k, 0) - v) for k, v in st.items())
    return min(seen_starts, key=dist)


def _key(st):
    return tuple(sorted(st.items()))


class Persistence:
    def __init__(self, data):
        self.start_out = {_key(st): tr[0] for st, _, tr in data}
        self.starts = [dict(k) for k in self.start_out]

    def start_readout(self, st):
        k = _key(st)
        if k in self.start_out:
            return self.start_out[k]
        return self.start_out[_key(_nearest_start(st, self.starts))] if self.starts else None

    def predict(self, st, seq):
        return self.start_readout(st)


class Window(Persistence):
    def __init__(self, data, w_obs=1, w_act=1):
        super().__init__(data)
        self.w_obs, self.w_act = w_obs, w_act
        self.table = {}
        for st, seq, tr in data:
            for t, a in enumerate(seq):
                for wo in range(w_obs, 0, -1):
                    for wa in range(w_act, 0, -1):
                        ctx = (tuple(tr[max(0, t + 1 - wo): t + 1]), tuple(seq[max(0, t + 1 - wa): t + 1]))
                        self.table.setdefault(ctx, {}).setdefault(tr[t + 1], 0)
                        self.table[ctx][tr[t + 1]] += 1

    def predict(self, st, seq):
        obs = [self.start_readout(st)]
        for t in range(len(seq)):
            nxt = None
            for wo in range(self.w_obs, 0, -1):
                for wa in range(self.w_act, 0, -1):
                    ctx = (tuple(obs[max(0, t + 1 - wo): t + 1]), tuple(seq[max(0, t + 1 - wa): t + 1]))
                    if ctx in self.table:
                        c = self.table[ctx]
                        nxt = max(c, key=c.get)
                        break
                if nxt is not None:
                    break
            obs.append(obs[-1] if nxt is None else nxt)
        return obs[-1]


class Merge(Persistence):
    """Red-blue state merging (EDSM-style) for Moore machines over the prefix tree of observed traces. A trial merge
    folds one subtree into another on copy-on-write overlays, so it touches only the nodes it folds."""

    def __init__(self, data, max_nodes=5000, min_score=0):
        super().__init__(data)
        self.min_score = min_score               # merges need at least this much agreeing evidence
        self.out, self.edges = [None], [{}]      # node 0: the root, before a start is chosen
        for st, seq, tr in data:
            n = self._child(0, ('s', _key(st)), tr[0])
            for a, o in zip(seq, tr[1:]):
                n = self._child(n, ('a', a), o)
            if len(self.out) > max_nodes:
                break
        self.rep = list(range(len(self.out)))
        self._learn()

    def _child(self, n, lab, o):
        if lab not in self.edges[n]:
            self.out.append(o); self.edges.append({})
            self.edges[n][lab] = len(self.out) - 1
        return self.edges[n][lab]

    def _f(self, x, rp=None):
        while True:
            y = rp.get(x, self.rep[x]) if rp is not None else self.rep[x]
            if y == x:
                return x
            x = y

    def _fold(self, r, b):
        """Fold b into r: (score, overlays) or None if some pair of merged states has different readouts."""
        rp, o, ed = {}, {}, {}
        get_o = lambda x: o.get(x, self.out[x])
        get_e = lambda x: ed.get(x, self.edges[x])
        score, stack = 0, [(r, b)]
        while stack:
            x, y = stack.pop()
            x, y = self._f(x, rp), self._f(y, rp)
            if x == y:
                continue
            ox, oy = get_o(x), get_o(y)
            if ox is not None and oy is not None:
                if ox != oy:
                    return None
                score += 1
            elif ox is None:
                o[x] = oy
            rp[y] = x
            ex = get_e(x)
            new = None
            for lab, c in get_e(y).items():
                if lab in ex:
                    stack.append((ex[lab], c))
                else:
                    new = new or dict(ex)
                    new[lab] = c
            if new is not None:
                ed[x] = new
        return score, (rp, o, ed)

    def _commit(self, ov):
        rp, o, ed = ov
        for k, v in rp.items():
            self.rep[k] = v
        for k, v in o.items():
            self.out[k] = v
        for k, v in ed.items():
            self.edges[k] = v

    def _learn(self):
        red = [0]
        while True:
            reds = {self._f(r) for r in red}
            blues = sorted({self._f(c) for r in reds for c in self.edges[r].values()} - reds)
            if not blues:
                return
            best, promote = None, None
            for b in blues:
                opts = [res for res in (self._fold(r, b) for r in reds if r != 0)
                        if res is not None and res[0] >= self.min_score]
                if not opts:
                    promote = b           # compatible with no red state: it is a new state
                    break
                top = max(opts, key=lambda t: t[0])
                if best is None or top[0] > best[0]:
                    best = top
            if promote is not None:
                red.append(promote)
            else:
                self._commit(best[1])

    def predict(self, st, seq):
        root = self._f(0)
        k = ('s', _key(st))
        if k not in self.edges[root]:
            k = ('s', _key(_nearest_start(st, self.starts)))
        n = self._f(self.edges[root][k])
        for a in seq:
            nxt = self.edges[n].get(('a', a))
            if nxt is not None:
                n = self._f(nxt)
        return self.out[n]

    def n_states(self):
        return len({self._f(x) for x in range(len(self.out))}) - 1


LEARNERS = {
    'persistence': lambda d: Persistence(d),
    'window_1_1': lambda d: Window(d, 1, 1),
    'window_2_2': lambda d: Window(d, 2, 2),
    'window_3_3': lambda d: Window(d, 3, 3),
    'merge_0': lambda d: Merge(d, min_score=0),
    'merge_2': lambda d: Merge(d, min_score=2),
}


def cv_select(data, names=None, folds=3, rng=None):
    """Choose a learner the way a careful experimenter would, from public data only: k-fold cross-validation on the
    collected experiments (predict each held-out experiment's readouts after every prefix). Returns (name, model
    refitted on all data, cv scores)."""
    names = names or list(LEARNERS)
    rng = rng or random.Random(0)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    scores = {}
    for name in names:
        hit = tot = 0
        for f in range(folds):
            test = set(idx[f::folds])
            train = [d for i, d in enumerate(data) if i not in test]
            if not train:
                continue
            m = LEARNERS[name](train)
            for i in test:
                st, seq, tr = data[i]
                for t in range(1, len(seq) + 1):
                    hit += m.predict(st, seq[:t]) == tr[t]
                    tot += 1
        scores[name] = hit / tot if tot else 0.0
    best = max(names, key=lambda n: (scores[n], -names.index(n)))
    return best, LEARNERS[best](data), scores


def accuracy(model, w, queries):
    from .fairness import answer
    hits = [model.predict(q['start'], q['inputs']) == answer(w, q) for q in queries]
    return sum(hits) / len(hits) if hits else float('nan'), hits


# ------------------------------------------------------------------ symbolic law search (generalizes across values)

SPLIT_COST = 4


class Symbolic(Persistence):
    """A public law search. For each action input and each readout, the cheapest expression over the features (the
    current readouts, plus a hypothesized hidden counter in SymbolicHidden) that fits every observed transition,
    from a small grammar with costs: constants; f_a + c; f_a + f_b + c; f_a - f_b + c; each clamped or wrapped to
    the readout's observed range; comparisons (for 0/1 readouts); and one threshold split (if f_a >= t then e1 else
    e2, costing SPLIT_COST plus both sides). Start readouts are fitted as laws of the start settings. Prediction
    chains the laws. Memoryless by construction: a world whose behaviour depends on hidden state defeats it."""

    def __init__(self, data, max_const=6):
        import numpy as np
        super().__init__(data)
        self.np = np
        self.max_const = max_const
        self.m = len(data[0][2][0]) if data else 0
        allv = np.array([list(o) for _, _, tr in data for o in tr]) if data else np.zeros((0, 0))
        self.lo = allv.min(0) if len(allv) else np.zeros(self.m, dtype=int)
        self.hi = allv.max(0) if len(allv) else np.zeros(self.m, dtype=int)
        self.rows = {}
        for st, seq, tr in data:
            for t, a in enumerate(seq):
                self.rows.setdefault(a, []).append((list(tr[t]), tr[t + 1]))
        self.laws = {a: self._fit_action(prs) for a, prs in self.rows.items()}
        self.cost = sum(l['cx'] for L in self.laws.values() for l in L)
        self.skeys = sorted({k for st, _, _ in data for k in st})
        self.start_laws = None
        if data and self.skeys:
            S = np.array([[st.get(k, 0) for k in self.skeys] for st, _, _ in data])
            Y0 = np.array([tr[0] for _, _, tr in data])
            self.start_laws = [self._fit(S, Y0[:, j], self.lo[j], self.hi[j]) for j in range(self.m)]

    def _fit_action(self, prs, extra=None, split=True, budget=float('inf'), split_features=None):
        np = self.np
        X = np.array([p[0] + (extra[i] if extra else []) for i, p in enumerate(prs)])
        Y = np.array([p[1] for p in prs])
        out, spent = [], 0
        for j in range(self.m):
            law = self._fit(X, Y[:, j], self.lo[j], self.hi[j], split=split, split_features=split_features)
            out.append(law)
            spent += law['cx']
            if budget != float('inf') and spent >= budget:
                return None
        return out

    def _cands(self, nf, lo, hi):
        key = (nf, int(lo), int(hi))
        cache = self.__dict__.setdefault('_cand_cache', {})
        if key not in cache:
            cache[key] = self._make_cands(nf, lo, hi)
        return cache[key]

    def _make_cands(self, nf, lo, hi):
        np, C = self.np, self.max_const
        lo, hi = int(lo), int(hi)
        out = [(1, f'{c}', lambda Z, c=c: np.full(len(Z), c)) for c in range(lo, hi + 1)]
        lin = []
        for a in range(nf):
            for c in range(-C, C + 1):
                lin.append((2 + (c != 0), f'f{a}{c:+d}', lambda Z, a=a, c=c: Z[:, a] + c))
        for a in range(nf):
            for b in range(nf):
                for c in (-1, 0, 1):
                    if a <= b:
                        lin.append((4 + (c != 0), f'f{a}+f{b}{c:+d}', lambda Z, a=a, b=b, c=c: Z[:, a] + Z[:, b] + c))
                    if a != b:
                        lin.append((4 + (c != 0), f'f{a}-f{b}{c:+d}', lambda Z, a=a, b=b, c=c: Z[:, a] - Z[:, b] + c))
        out += lin
        for cx, d, f in lin:
            out.append((cx + 1, f'clamp({d})', lambda Z, f=f: np.clip(f(Z), lo, hi)))
            if hi > lo:
                out.append((cx + 1, f'wrap({d})', lambda Z, f=f: lo + (f(Z) - lo) % (hi - lo + 1)))
        if lo == 0 and hi <= 1:
            for a in range(nf):
                for b in range(nf):
                    if a != b:
                        out.append((3, f'f{a}>f{b}', lambda Z, a=a, b=b: (Z[:, a] > Z[:, b]).astype(int)))
                for c in range(0, 16):
                    out.append((3, f'f{a}>{c}', lambda Z, a=a, c=c: (Z[:, a] > c).astype(int)))
        out.sort(key=lambda t: t[0])
        return out

    def _fit(self, X, y, lo, hi, split=True, split_features=None):
        """The cheapest exact law {'d','f','split','cx','exact'}; if none is exact, the most accurate (cx = inf)."""
        best = None
        for cx, d, f in self._cands(X.shape[1], lo, hi):
            v = f(X)
            acc = float((v == y).mean()) if len(y) else 1.0
            if acc == 1.0:
                exact = {'d': d, 'f': f, 'split': None, 'cx': cx, 'exact': True}
                break
            if best is None or acc > best[0]:
                best = (acc, d, f)
        else:
            exact = None
        if split and len(y) >= 4:
            for a in (range(X.shape[1]) if split_features is None else split_features):
                for t in sorted(set(X[:, a].tolist()))[1:]:
                    m1 = X[:, a] >= t
                    l1 = self._fit(X[m1], y[m1], lo, hi, split=False)
                    l0 = self._fit(X[~m1], y[~m1], lo, hi, split=False)
                    if l1['exact'] and l0['exact']:
                        cx = SPLIT_COST + l1['cx'] + l0['cx']
                        if exact is None or cx < exact['cx']:
                            exact = {'d': f"if f{a}>={t}: {l1['d']} else {l0['d']}", 'f': None,
                                     'split': (a, t, l1, l0), 'cx': cx, 'exact': True}
        if exact is not None:
            return exact
        return {'d': best[1], 'f': best[2], 'split': None, 'cx': float('inf'), 'exact': False}

    def _apply(self, law, Z):
        if law['split'] is None:
            return law['f'](Z)
        a, t, l1, l0 = law['split']
        return self.np.where(Z[:, a] >= t, self._apply(l1, Z), self._apply(l0, Z))

    def start_readout(self, st):
        k = _key(st)
        if k in self.start_out or not self.start_laws:
            return super().start_readout(st)
        z = self.np.array([[st.get(x, 0) for x in self.skeys]])
        return tuple(int(self._apply(law, z)[0]) for law in self.start_laws)

    def _chain(self, z, seq, laws):
        for a in seq:
            if a in laws:
                Z = self.np.array([z])
                z = [int(self._apply(law, Z)[0]) for law in laws[a]]
        return z

    def predict(self, st, seq):
        o = self.start_readout(st)
        return None if o is None else tuple(self._chain(list(o), seq, self.laws))

    def describe(self):
        return {a: [law['d'] for law in L] for a, L in self.laws.items()}


class SymbolicHidden(Symbolic):
    """The law search plus one hypothesized hidden counter h in 0..H (clamped), starting at h0, updated per action
    (keep, +1, -1, reset to 0); the counter is the last feature, and laws may split on it (thresholds on the
    counter). The cheapest complete explanation wins: the memoryless laws, or a hidden hypothesis costing 2 + one per
    action that touches h, plus its laws. The near-cheapest explanations are kept as a committee, for choosing
    experiments. Public information only."""

    UPDATES = ('keep', 'inc', 'dec', 'reset')

    def __init__(self, data, max_h=8, max_hyp=3000, slack=4, committee_size=6, max_seconds=10):
        import itertools as it
        import time as _t
        super().__init__(data)
        self.hidden = None
        self.committee = []          # (cost, hidden or None, laws)
        self.timed_out = False
        if not data:
            return
        if self.cost < float('inf'):
            self.committee.append((self.cost, None, self.laws))
        acts = sorted(self.rows)
        inexact = {a for a, L in self.laws.items() if any(not l['exact'] for l in L)}
        deadline = _t.time() + max_seconds

        def hyps():
            # lazily, simplest first: k actions touch the counter (k = 1, 2, ...), then the counter size, then h0
            for k in range(1, len(acts) + 1):
                for who in it.combinations(acts, k):
                    for kinds in it.product(self.UPDATES[1:], repeat=k):
                        ups = {a: 'keep' for a in acts}
                        ups.update(dict(zip(who, kinds)))
                        for H in range(1, max_h + 1):
                            for h0 in (0, H):
                                yield 2 + k, H, h0, ups
        best_cost = self.cost
        for n, (base, H, h0, ups) in enumerate(hyps()):
            if n >= max_hyp or base >= best_cost + slack:
                break
            if _t.time() > deadline:
                self.timed_out = True
                break
            hs = {}
            for st, seq, tr in data:
                h = h0
                for t, a in enumerate(seq):
                    hs.setdefault(a, []).append([h])
                    h = self._step(h, ups[a], H)
            laws, cost = {}, base
            for a, prs in self.rows.items():
                if a not in inexact:          # exact memoryless laws stay; the counter only has to explain the rest
                    laws[a] = [dict(l, uses_h=False) for l in self.laws[a]]
                    cost += sum(l['cx'] for l in self.laws[a])
                    continue
                L = self._fit_action(prs, hs[a], split=True, split_features=[self.m],
                                     budget=best_cost + slack - cost)
                if L is None:
                    cost = float('inf')
                    break
                laws[a] = L
                cost += sum(l['cx'] for l in L)
            if cost < best_cost + slack:
                self.committee.append((cost, {'H': H, 'h0': h0, 'updates': ups, 'cost': cost}, laws))
                if cost < best_cost:
                    best_cost = cost
                    self.hidden = {'H': H, 'h0': h0, 'updates': ups, 'cost': cost}
                    self.hlaws = laws
        self.cost = best_cost
        self.committee = sorted([c for c in self.committee if c[0] < best_cost + slack], key=lambda c: c[0])[:committee_size]

    @staticmethod
    def _step(h, u, H):
        return min(H, h + 1) if u == 'inc' else max(0, h - 1) if u == 'dec' else 0 if u == 'reset' else h

    def _trace(self, st, seq, hidden, laws):
        z = list(self.start_readout(st))
        out = [tuple(z)]
        h = hidden['h0'] if hidden else 0
        for a in seq:
            if a in laws:
                z = [int(self._apply(law, self.np.array([z + [h] if hidden and law.get('uses_h', True) else z]))[0])
                     for law in laws[a]]
            if hidden:
                h = self._step(h, hidden['updates'].get(a, 'keep'), hidden['H'])
            out.append(tuple(z))
        return out

    def predict(self, st, seq):
        if not self.hidden:
            return super().predict(st, seq)
        return self._trace(st, seq, self.hidden, self.hlaws)[-1]


    def committee_traces(self, st, seq):
        return [tuple(self._trace(st, seq, hid, laws)) for _, hid, laws in self.committee]

    def describe(self):
        if not self.hidden:
            return super().describe()
        return {'hidden': {k: v for k, v in self.hidden.items()}, **{a: [l['d'] for l in L] for a, L in self.hlaws.items()}}


def adaptive(w, budget, rng, seed_frac=0.25, pool=150):
    """A public adaptive experimenter: a small sweep first, then, one at a time, the experiment on which its
    committee of near-cheapest explanations (SymbolicHidden) disagrees most. It never sees the world's law, only the
    traces of the experiments it runs. Returns the experiments."""
    from .fairness import sample_experiment
    exps = sweep(w, max(3, int(budget * seed_frac)), rng)
    while len(exps) < budget:
        model = SymbolicHidden(collect(w, exps))
        cands = [sample_experiment(w, rng) for _ in range(pool)]
        st0 = w.starts()
        for k in range(1, w.max_len + 1):     # also targeted probes: each action repeated, then each other action
            for i in range(len(w.inputs)):
                for j in range(len(w.inputs)):
                    if rng.random() < 0.3:
                        cands.append((st0[rng.randrange(len(st0))], [i] * (k - 1) + [j]))
        if len(model.committee) <= 1:
            exps.append(cands[rng.randrange(len(cands))])
            continue
        best = max(cands, key=lambda e: len(set(model.committee_traces(*e))))
        exps.append(best)
    return exps


LEARNERS['symbolic'] = lambda d: Symbolic(d)
LEARNERS['symbolic_hidden'] = lambda d: SymbolicHidden(d)
