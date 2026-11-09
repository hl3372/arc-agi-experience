"""Novelty for domain A: a behavioural fingerprint of a world, invariant to the usual disguises.

1. Restrict to states reachable from the settable starts.
2. Drop readouts that never vary, and inputs that never change the observable behaviour.
3. Minimize the observable Moore machine (partition refinement on readouts and successors).
4. Fingerprint the minimized machine up to renaming of inputs and relabelling of readout values: an iterated
   colour refinement (Weisfeiler-Lehman style) where a state's colour starts from the size of its readout class and is
   refined by the multiset, over inputs, of its successors' colours. The start states are marked.

Equal fingerprints are necessary, not sufficient, for equivalence up to these disguises (as WL hashing in domain B);
the fingerprint also reports the minimized size, inputs and readout classes. Known blind spots: product
decompositions (two textbook mechanisms side by side) and recodings that merge readout classes.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .world import World

ARCHIVE = Path(__file__).resolve().parents[1] / 'archive' / 'mechanisms.json'


def reachable(w):
    starts = sorted({w.start_state(st) for st in w.starts()})
    seen, frontier = set(starts), list(starts)
    while frontier:
        s = frontier.pop()
        for i in range(len(w.inputs)):
            t = int(w.next[i, s])
            if t not in seen:
                seen.add(t)
                frontier.append(t)
    return sorted(seen), starts


def minimize(w):
    """(states, inputs kept, readout classes, successor table, start blocks) of the minimized observable machine."""
    states, starts = reachable(w)
    idx = {s: k for k, s in enumerate(states)}
    outs = np.array([w.out_tuple[s] for s in states]).reshape(len(states), -1)
    varying = [j for j in range(outs.shape[1]) if len(set(outs[:, j])) > 1]
    obs = [tuple(outs[k, varying]) for k in range(len(states))]
    succ = np.array([[idx[int(w.next[i, s])] for s in states] for i in range(len(w.inputs))]).reshape(len(w.inputs), -1)
    # partition refinement (Moore)
    labels = {o: c for c, o in enumerate(sorted(set(obs)))}
    block = [labels[o] for o in obs]
    while True:
        sig = [(block[k],) + tuple(block[succ[i, k]] for i in range(len(w.inputs))) for k in range(len(states))]
        ids = {g: c for c, g in enumerate(sorted(set(sig)))}
        new = [ids[g] for g in sig]
        if len(set(new)) == len(set(block)):
            block = new
            break
        block = new
    nb = len(set(block))
    rep = {}
    for k, b in enumerate(block):
        rep.setdefault(b, k)
    table = np.array([[block[succ[i, rep[b]]] for b in range(nb)] for i in range(len(w.inputs))]).reshape(len(w.inputs), -1)
    out_class = [obs[rep[b]] for b in range(nb)]
    # inputs that act as the identity on the minimized machine change nothing observable: drop them
    keep = [i for i in range(len(w.inputs)) if not all(table[i, b] == b for b in range(nb))]
    # identical columns (two names for the same action) collapse too
    uniq, cols = [], set()
    for i in keep:
        key = tuple(table[i].tolist())
        if key not in cols:
            cols.add(key)
            uniq.append(i)
    start_blocks = sorted({block[idx[s]] for s in starts})
    return nb, uniq, out_class, table[uniq] if uniq else np.zeros((0, nb), dtype=int), start_blocks


def fingerprint(w, rounds=6):
    nb, inputs, out_class, table, start_blocks = minimize(w)
    # initial colour: readout class size (value-relabelling invariant) and start mark
    sizes = {}
    for o in out_class:
        sizes[o] = sizes.get(o, 0) + 1
    colour = [f'{sizes[out_class[b]]}|{int(b in start_blocks)}' for b in range(nb)]
    for _ in range(rounds):
        colour = [hashlib.sha1((colour[b] + '|' + ','.join(sorted(colour[table[i, b]] for i in range(len(inputs))))
                                ).encode()).hexdigest()[:10] for b in range(nb)]
    hist = sorted(colour)
    digest = hashlib.sha1(json.dumps([nb, len(inputs), len(set(out_class)), hist]).encode()).hexdigest()[:16]
    return {'states': nb, 'inputs': len(inputs), 'readout_classes': len(set(out_class)), 'digest': digest}


def same(fa, fb):
    return fa['digest'] == fb['digest']


def load_archive():
    if not ARCHIVE.exists():
        return []
    out = []
    for a in json.loads(ARCHIVE.read_text()):
        out.append({**a, 'fingerprint': fingerprint(World(a['world']))})
    return out


def archive_matches(w, archive=None):
    fp = fingerprint(w)
    return [a['id'] for a in (archive if archive is not None else load_archive()) if same(fp, a['fingerprint'])]


# ------------------------------------------------------------------ structure: the family, with constants abstracted

def _shape(src, state_vars):
    import ast

    class S(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id='V' if node.id in state_vars else ('K' if node.id == 'k' else node.id),
                                              ctx=ast.Load()), node)

        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant('C'), node)
    return ast.unparse(S().visit(ast.parse(src, mode='eval')))


def structure(spec):
    """A signature of the mechanism with names and constants abstracted: per action, the multiset of rule shapes
    (guard shape, sorted assignment shapes); the tick; readout shapes; boundary rules; numbers of hidden and
    settable variables. Equal signatures with different behaviour = instances of one family."""
    sv = set(spec['state'])
    def rules(rs):
        return sorted(json.dumps([_shape(r['if'], sv) if r.get('if') else '-',
                                  sorted(_shape(e, sv) for e in r.get('set', {}).values())]) for r in rs)
    acts = sorted(json.dumps([bool(d.get('param')), rules(d['rules'])]) for d in spec['actions'].values())
    sig = {'actions': acts, 'tick': rules(spec.get('tick') or []),
           'readouts': sorted(_shape(e, sv) for e in spec['readouts'].values()),
           'boundaries': sorted(d.get('boundary', 'clamp') for d in spec['state'].values()),
           'hidden': sum(1 for d in spec['state'].values() if not d.get('settable')),
           'settable': sum(1 for d in spec['state'].values() if d.get('settable'))}
    return hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]


def matches(spec, archive=None):
    """{'behaviour': [ids], 'structure': [ids]} against the archive."""
    w = World(spec)
    fp, st = fingerprint(w), structure(spec)
    arch = archive if archive is not None else load_archive()
    return {'behaviour': [a['id'] for a in arch if same(fp, a['fingerprint'])],
            'structure': [a['id'] for a in arch if a.get('structure') == st]}
