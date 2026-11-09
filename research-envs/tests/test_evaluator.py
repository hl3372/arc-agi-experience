"""Evaluator trust: every domain function against brute force on random graphs, format checks, language whitelist,
timeouts, and verdict semantics per problem kind."""
import itertools
import random
import pytest
from research_envs.construct.evaluate import Spec
from research_envs.construct.lang import SpecError, compile_expr, run
from research_envs.construct.domain import GraphEnv, graph_from, sets_from, FormatError, INF


def rand_graph(rng, n, p):
    return n, frozenset((u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p)


def brute(n, edges):
    nb = {v: {u for e in edges for u in e if v in e and u != v} for v in range(n)}
    # all-pairs BFS distances
    dist = {}
    for s in range(n):
        d = {s: 0}; q = [s]
        for u in q:
            for v in nb[u]:
                if v not in d:
                    d[v] = d[u] + 1; q.append(v)
        dist[s] = d
    tri = sum(1 for a, b, c in itertools.combinations(range(n), 3) if b in nb[a] and c in nb[a] and c in nb[b])
    comps = len({frozenset(dist[s]) for s in range(n)})
    def clique(sub):
        return all(b in nb[a] for a, b in itertools.combinations(sub, 2))
    def indep(sub):
        return all(b not in nb[a] for a, b in itertools.combinations(sub, 2))
    om = max(k for k in range(n + 1) if any(clique(s) for s in itertools.combinations(range(n), k)))
    al = max(k for k in range(n + 1) if any(indep(s) for s in itertools.combinations(range(n), k)))
    # girth by brute force: shortest cycle through any simple cycle enumeration (small n)
    girth = INF
    for k in range(3, n + 1):
        for cyc in itertools.permutations(range(n), k):
            if cyc[0] == min(cyc) and cyc[1] < cyc[-1] and all(cyc[(i + 1) % k] in nb[cyc[i]] for i in range(k)):
                girth = k; break
        if girth < INF:
            break
    bip = any(all(((c >> a) & 1) != ((c >> b) & 1) for a, b in edges) for c in range(2 ** n))
    conn = comps == 1
    diam = max(max(d.values()) for d in dist.values()) if conn else INF
    return dict(nb=nb, dist=dist, tri=tri, comps=comps, om=om, al=al, girth=girth, bip=bip, conn=conn, diam=diam)


@pytest.mark.parametrize('seed', range(40))
def test_graph_functions_match_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 7)
    n, edges = rand_graph(rng, n, rng.choice([0.2, 0.4, 0.6, 0.9]))
    env, b = GraphEnv(n, edges).env(), brute(n, edges)
    assert env['n'] == n and set(env['E']) == set(edges)
    for v in range(n):
        assert env['nbrs'](v) == b['nb'][v] and env['deg'](v) == len(b['nb'][v])
        for u in range(n):
            assert env['adj'](u, v) == (u in b['nb'][v])
            assert env['dist'](u, v) == b['dist'][u].get(v, INF)
    assert env['triangles']() == b['tri']
    assert env['components']() == b['comps'] and env['connected']() == b['conn']
    assert env['clique_number']() == b['om'] and env['independence_number']() == b['al']
    assert env['girth']() == b['girth']
    assert env['is_bipartite']() == b['bip']
    assert env['diameter']() == b['diam']


def test_connectivity_on_known_graphs():
    cycle = GraphEnv(6, frozenset((i, (i + 1) % 6) if i < (i + 1) % 6 else ((i + 1) % 6, i) for i in range(6))).env()
    assert cycle['edge_connectivity']() == 2 and cycle['vertex_connectivity']() == 2
    k4 = GraphEnv(4, frozenset(itertools.combinations(range(4), 2))).env()
    assert k4['edge_connectivity']() == 3 and k4['vertex_connectivity']() == 3
    path = GraphEnv(4, frozenset([(0, 1), (1, 2), (2, 3)])).env()
    assert path['edge_connectivity']() == 1 and path['girth']() == INF


@pytest.mark.parametrize('obj', [
    {'n': 3, 'edges': [[0, 1], [1, 0]]}, {'n': 3, 'edges': [[0, 3]]}, {'n': 3, 'edges': [[1, 1]]},
    {'n': 3, 'edges': [[0, True]]}, {'n': 3, 'edges': [[0, 1, 2]]}, {'n': 3}, {'n': 3, 'edges': [], 'x': 1},
    {'n': 0, 'edges': []}, {'n': 999, 'edges': []}, [1, 2],
])
def test_bad_graph_formats(obj):
    with pytest.raises(FormatError):
        graph_from(obj, {})


@pytest.mark.parametrize('obj', [
    {'m': 3, 'sets': [[0, 1], [1, 0]]}, {'m': 3, 'sets': [[0, 0]]}, {'m': 3, 'sets': [[3]]},
    {'m': 3, 'sets': [[0], 'x']}, {'m': 3}, {'m': 40, 'sets': []},
])
def test_bad_set_formats(obj):
    with pytest.raises(FormatError):
        sets_from(obj, {})


def test_params_fix_and_bound_sizes():
    with pytest.raises(FormatError):
        graph_from({'n': 5, 'edges': []}, {'n': 6})
    with pytest.raises(FormatError):
        graph_from({'n': 9, 'edges': []}, {'n_max': 8})
    assert graph_from({'n': 6, 'edges': []}, {'n': 6})[0] == 6


@pytest.mark.parametrize('src', ['().__class__', "__import__('os')", 'open(1)', 'lambda: 1', 'E[0:1]', "'a'",
                                 '2**n', 'f(x=1)', '[*E]', '{1: 2}', 'x := 1', '_private', 'n.bit_length()',
                                 'E.__len__()', 'globals()', 'eval(1)', '1.5'])
def test_language_rejects(src):
    with pytest.raises(SpecError):
        compile_expr(src, ['n', 'E'])


def test_runtime_caps():
    from research_envs.construct.lang import StepBudget
    with pytest.raises(StepBudget):   # object-dependent sizes: a resource limit
        run(compile_expr('len(tuple(range(10**7)))', []), {})
    with pytest.raises(SpecError):
        run(compile_expr('len(tuple(combinations(range(10), 7)))', []), {})
    with pytest.raises(SpecError):
        run(compile_expr('pow(3, 1000)', []), {})


def test_timeout_makes_candidate_invalid_not_crash():
    s = Spec({'object': 'graph', 'kind': 'satisfy', 'params': {'n': 40},
              'constraints': [{'name': 'slow', 'expr': 'all(a + b + c >= 0 for a in range(400) for b in range(400) for c in range(400))'}]},
             timeout=0.3)
    r = s.evaluate({'n': 40, 'edges': []})
    assert not r['valid'] and ('timeout' in r['errors'] or 'iteration steps' in r['errors'].get('slow', ''))


def test_kinds_semantics():
    base = {'object': 'graph', 'params': {'n': 4}}
    c4 = {'n': 4, 'edges': [[0, 1], [1, 2], [2, 3], [0, 3]]}
    opt = Spec({**base, 'kind': 'optimize', 'constraints': [{'name': 'tf', 'expr': 'triangles() == 0'}],
                'objective': {'expr': 'len(E)', 'direction': 'max'}})
    assert opt.evaluate(c4) == {**opt.evaluate(c4), 'valid': True, 'objective': 4}
    ce = Spec({**base, 'kind': 'counterexample', 'constraints': [{'name': 'reg', 'expr': 'all(deg(v) == 2 for v in V)'}],
               'claim': {'expr': 'not is_bipartite()'}})
    r = ce.evaluate(c4)
    assert r['valid'] and r['claim_holds'] is False                     # C4 is 2-regular and bipartite: refutes the claim
    tri = {'n': 4, 'edges': [[0, 1], [1, 2], [0, 2], [2, 3]]}
    assert not ce.evaluate(tri)['valid']                                 # violates the assumption
    with pytest.raises(SpecError):
        Spec({**base, 'kind': 'satisfy', 'constraints': [{'name': 'bad', 'expr': 'len(E)'}]}).evaluate(c4)


def test_constraint_error_is_a_violation_with_reason():
    s = Spec({'object': 'graph', 'kind': 'satisfy', 'params': {'n': 3},
              'constraints': [{'name': 'bad_vertex', 'expr': 'deg(5) == 0'}]})
    r = s.evaluate({'n': 3, 'edges': []})
    assert not r['valid'] and 'bad_vertex' in r['errors']


def test_given_data_and_sets():
    s = Spec({'object': 'set_system', 'kind': 'optimize', 'params': {'m': 4},
              'given': {'COST': [1, 2, 3, 4]},
              'constraints': [{'name': 'pairwise_meet', 'expr': 'all(len(a & b) >= 1 for a, b in combinations(F, 2))'}],
              'objective': {'expr': 'len(F)', 'direction': 'max'}})
    star = {'m': 4, 'sets': [[0], [0, 1], [0, 2], [0, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3], [0, 1, 2, 3]]}
    r = s.evaluate(star)
    assert r['valid'] and r['objective'] == 8
    with pytest.raises(SpecError):
        Spec({'object': 'set_system', 'kind': 'satisfy', 'given': {'cost': [1]},
              'constraints': [{'name': 'x', 'expr': 'True'}]})


def test_an_object_too_large_for_the_language_is_invalid_not_a_crash():
    """Battery bug (preview p2): a 5000-set family made combinations(F, 2) exceed the pool limit and crashed search."""
    s = Spec({'object': 'set_system', 'kind': 'satisfy', 'params': {'m': 16},
              'constraints': [{'name': 'pairs', 'expr': 'all(len(A & B) <= 3 for A, B in combinations(F, 2))'}]})
    import itertools
    big = (16, tuple(frozenset(c) for c in itertools.combinations(range(16), 5)))   # 4368 sets, parsed directly
    r = s.evaluate_parsed(big)
    assert not r['valid'] and 'pool' in r['errors'].get('pairs', '')


def test_search_treats_families_beyond_the_format_cap_as_invalid():
    from research_envs.construct.search import Problem
    p = Problem(Spec({'object': 'set_system', 'kind': 'satisfy', 'params': {'m': 12},
                      'constraints': [{'name': 't', 'expr': 'True'}]}), {'set_sizes': [4]}, max_evals=10)
    sc, r = p.score(frozenset(range(495)))
    assert r['valid'] and sc[0] == 0
    sc, r = p.score(frozenset(range(len(p.universe))))
    assert len(p.universe) == 495 and r['valid']
    p2 = Problem(Spec({'object': 'set_system', 'kind': 'satisfy', 'params': {'m': 13},
                       'constraints': [{'name': 't', 'expr': 'True'}]}), {'set_sizes': [5]}, max_evals=10)
    sc, r = p2.score(frozenset(range(600)))
    assert not r['valid'] and sc[0] > 0
