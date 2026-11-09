"""Fairness and identifiability on compiled worlds (all table operations).

- first_difference: the shortest experiment (start, inputs) on which two worlds' readouts differ, by breadth-first
  search over the product machine from every settable start, up to a length limit. Exact.
- needle_fraction: the share of experiments from the public random sampler that separate a rival from the truth.
- eliminate: the R-aware eliminator (the local identifiability check). It knows the rival set, not which law is
  true; each step it runs the experiment from a candidate pool that splits the consistent rivals most evenly.
- sample_queries / answers: withheld queries drawn by code, and every world's answers to them.
"""
import random
import numpy as np


def readouts(w):
    return np.array(w.out_tuple, dtype=np.int64).reshape(w.n_states, -1)


def first_difference(w1, w2, max_len=None):
    """Shortest (start settings, input index list) separating w1 from w2 within max_len actions, or None."""
    L = w1.max_len if max_len is None else max_len
    r1, r2 = readouts(w1), readouts(w2)
    starts = w1.starts()
    P = np.array([[w1.start_state(st), w2.start_state(st)] for st in starts], dtype=np.int64)
    d = np.any(r1[P[:, 0]] != r2[P[:, 1]], axis=1)
    if d.any():
        return starts[int(d.argmax())], []
    n2 = w2.n_states
    seen = set((P[:, 0] * n2 + P[:, 1]).tolist())
    layers = []          # per depth: (parent index into the previous frontier, input index)
    n_in = len(w1.inputs)
    for _ in range(L):
        a = w1.next[:, P[:, 0]].reshape(-1)          # input-major: entry i*len(P)+j
        b = w2.next[:, P[:, 1]].reshape(-1)
        par = np.tile(np.arange(len(P)), n_in)
        inp = np.repeat(np.arange(n_in), len(P))
        d = np.any(r1[a] != r2[b], axis=1)
        if d.any():
            x = int(d.argmax())
            path, idx = [int(inp[x])], int(par[x])
            for lp, li in reversed(layers):
                path.append(int(li[idx]))
                idx = int(lp[idx])
            return starts[idx], path[::-1]
        keys = (a * n2 + b).tolist()
        keep = []
        for i, k in enumerate(keys):
            if k not in seen:
                seen.add(k)
                keep.append(i)
        if not keep:
            return None
        keep = np.array(keep)
        layers.append((par[keep], inp[keep]))
        P = np.stack([a[keep], b[keep]], 1)
    return None


# ------------------------------------------------------------------ experiments and the public sampler

def sample_experiment(w, rng, max_len=None):
    """The public random sampler: a uniform settable start, a uniform length 1..L, uniform actions."""
    L = w.max_len if max_len is None else max_len
    starts = w.starts()
    st = starts[rng.randrange(len(starts))]
    seq = [rng.randrange(len(w.inputs)) for _ in range(rng.randint(1, L))]
    return st, seq


def trace(w, exp):
    st, seq = exp
    return w.run_inputs(w.start_state(st), seq)


def needle_fraction(truth, rival, rng, n=400):
    """Share of public random experiments on which the rival's trace differs from the truth's."""
    hits = 0
    for _ in range(n):
        e = sample_experiment(truth, rng)
        hits += trace(truth, e) != trace(rival, e)
    return hits / n


# ------------------------------------------------------------------ the R-aware eliminator

def eliminate(truth, rivals, budget, rng, pool_size=300, extra_pool=()):
    """Greedy elimination: each step picks, from a fresh random pool (plus extra candidates), the experiment whose
    outcomes split the still-consistent worlds (truth included) into the most even partition, runs it on the truth,
    and keeps the consistent ones. Returns the experiments used and the rivals still consistent."""
    alive = list(range(len(rivals)))
    used = []
    for _ in range(budget):
        if not alive:
            break
        pool = [sample_experiment(truth, rng) for _ in range(pool_size)] + list(extra_pool)
        best, best_score = None, -1
        for e in pool:
            groups = {}
            groups[tuple(map(tuple, trace(truth, e)))] = 1
            for i in alive:
                k = tuple(map(tuple, trace(rivals[i], e)))
                groups[k] = groups.get(k, 0) + 1
            total = sum(groups.values())
            score = total - max(groups.values())      # how many worlds the worst outcome rules out
            if score > best_score:
                best, best_score = e, score
        if best_score <= 0:
            break                                     # nothing in the pool separates the survivors
        used.append(best)
        t = trace(truth, best)
        alive = [i for i in alive if trace(rivals[i], best) == t]
    return used, alive


# ------------------------------------------------------------------ queries

def sample_queries(w, rng, n_interp=20, n_comp=10, extra_len=0):
    """Withheld queries drawn by code. interpolation: like the public sampler; composition: longer action mixes up to
    the length limit, with every action appearing; extrapolation (optional): beyond the limit by extra_len."""
    out = []
    for _ in range(n_interp):
        st, seq = sample_experiment(w, rng)
        out.append({'regime': 'interpolation', 'start': st, 'inputs': seq})
    for _ in range(n_comp):
        st, _ = sample_experiment(w, rng)
        seq = list(range(len(w.inputs)))[: w.max_len]
        seq += [rng.randrange(len(w.inputs)) for _ in range(w.max_len - len(seq))]
        rng.shuffle(seq)
        out.append({'regime': 'composition', 'start': st, 'inputs': seq})
    for _ in range(extra_len and n_interp // 2):
        st, _ = sample_experiment(w, rng)
        seq = [rng.randrange(len(w.inputs)) for _ in range(w.max_len + extra_len)]
        out.append({'regime': 'extrapolation', 'start': st, 'inputs': seq})
    return out


def answer(w, q):
    """A query's answer: the readouts after the last action."""
    return w.run_inputs(w.start_state(q['start']), q['inputs'])[-1]
