"""Canaries: textbook problems in disguise (renamed, dualised, restated, with redundant constraints added) must be
recognised by the novelty check. No model calls. Prints the hit rate; exits nonzero below 80%."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.construct import novelty

G, S = 'graph', 'set_system'


def spec(obj, cons, objective=None, kind=None):
    raw = {'object': obj, 'kind': kind or ('optimize' if objective else 'satisfy'), 'params': {},
           'constraints': [{'name': f'c{i}', 'expr': e} for i, e in enumerate(cons)]}
    if objective:
        raw['objective'] = {'expr': objective[0], 'direction': objective[1]}
    return raw


CANARIES = [  # (expected archive id, disguised spec, hint)
    ('mantel', spec(G, ['all(not (adj(a, b) and adj(b, c) and adj(a, c)) for a, b, c in combinations(V, 3))'],
                    ('sum(deg(v) for v in V) // 2', 'max')), None),
    ('mantel', spec(G, ['clique_number() < 3', 'components() <= n'], ('len(E)', 'max')), None),
    ('turan_k4', spec(G, ['not any(all(adj(x, y) for x, y in combinations(Q, 2)) for Q in combinations(V, 4))'],
                      ('len(E)', 'max')), None),
    ('ramsey_33', spec(G, ['triangles() == 0', 'all(any(adj(a, b) for a, b in combinations(T, 2)) for T in combinations(V, 3))']), None),
    ('girth5_max_edges', spec(G, ['triangles() == 0', 'all(len(nbrs(u) & nbrs(v)) <= 1 for u, v in combinations(V, 2))'],
                              ('len(E)', 'max')), None),
    ('bipartite_max', spec(G, ['not any(True for k in range(3, n + 1, 2) if False) and is_bipartite()'],
                           ('len(E)', 'max')), None),
    ('c4_free', spec(G, ['not any(adj(a, b) and adj(b, c) and adj(c, d) and adj(d, a) for a, b, c, d in combinations(V, 4))'
                         ' and not any(adj(a, c) and adj(c, b) and adj(b, d) and adj(d, a) for a, b, c, d in combinations(V, 4))'
                         ' and not any(adj(a, b) and adj(b, d) and adj(d, c) and adj(c, a) for a, b, c, d in combinations(V, 4))'],
                     ('len(E)', 'max')), None),
    ('diameter2_min_edges', spec(G, ['all(u == v or adj(u, v) or len(nbrs(u) & nbrs(v)) > 0 for u, v in combinations(V, 2))'],
                                 ('len(E)', 'min')), None),
    ('sperner', spec(S, ['all(len(A - B) > 0 and len(B - A) > 0 for A, B in combinations(F, 2))'], ('len(F)', 'max')), None),
    ('ekr_k3', spec(S, ['all(len(A) == 3 for A in F)', 'all(A & B for A, B in combinations(F, 2))'], ('len(F)', 'max')),
     {'set_sizes': [3]}),
    ('triple_packing', spec(S, ['all(len(A) == 3 for A in F)', 'not any(len(A & B) >= 2 for A, B in combinations(F, 2))'],
                            ('len(F)', 'max')), {'set_sizes': [3]}),
    ('separating', spec(S, ['len(set(tuple((a in B) for B in F) for a in ground)) == m'],
                        ('len(F)', 'min')), None),
]


def main():
    arch = novelty.load_archive()
    by_id = {a['id']: a for a in arch}
    hits = 0
    for want, raw, hint in CANARIES:
        try:
            fp = novelty.fingerprint(raw, hint)
            got = [a['id'] for a in arch if a['object'] == raw['object'] and novelty.same_family(fp, a['fingerprint'])]
        except Exception as e:
            got = [f'error {type(e).__name__}: {e}']
        ok = want in got
        hits += ok
        print(f"{'HIT ' if ok else 'MISS'} {want:22s} matched {got}")
    rate = hits / len(CANARIES)
    print(f'canary hit rate {hits}/{len(CANARIES)} = {rate:.0%}')
    sys.exit(0 if rate >= 0.8 else 1)


if __name__ == '__main__':
    main()
