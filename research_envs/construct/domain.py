"""Object types for domain B: their candidate format, format validation, and the functions the spec language exposes.

Candidates are explicit JSON data, never code. A malformed candidate is invalid with a reason; it is never repaired.
"""
import networkx as nx

INF = 10 ** 9
LIMITS = {'graph_n': 64, 'set_m': 32, 'set_family': 512}


class FormatError(ValueError):
    pass


def _int(x, what):
    if not isinstance(x, int) or isinstance(x, bool):
        raise FormatError(f'{what} must be an integer')
    return x


# ------------------------------------------------------------------ graphs

def graph_from(obj, params):
    """{"n": int, "edges": [[u, v], ...]} -> (n, frozenset of sorted pairs). Checks params n / n_min / n_max."""
    if not isinstance(obj, dict) or set(obj) - {'n', 'edges'} or 'n' not in obj or 'edges' not in obj:
        raise FormatError('a graph is {"n": int, "edges": [[u, v], ...]} and nothing else')
    n = _int(obj['n'], 'n')
    if not 1 <= n <= LIMITS['graph_n']:
        raise FormatError(f'n must be between 1 and {LIMITS["graph_n"]}')
    if 'n' in params and n != params['n']:
        raise FormatError(f'n must be {params["n"]}')
    if n < params.get('n_min', 1) or n > params.get('n_max', LIMITS['graph_n']):
        raise FormatError('n is outside the allowed range')
    if not isinstance(obj['edges'], list):
        raise FormatError('edges must be a list')
    edges = set()
    for e in obj['edges']:
        if not isinstance(e, list) or len(e) != 2:
            raise FormatError('each edge is a list [u, v]')
        u, v = _int(e[0], 'vertex'), _int(e[1], 'vertex')
        if not (0 <= u < n and 0 <= v < n):
            raise FormatError(f'edge {e} has a vertex outside 0..{n - 1}')
        if u == v:
            raise FormatError(f'edge {e} is a loop')
        p = (min(u, v), max(u, v))
        if p in edges:
            raise FormatError(f'edge {e} appears twice')
        edges.add(p)
    return n, frozenset(edges)


class GraphEnv:
    """The graph functions available to specs. Expensive helpers are computed once per object."""

    def __init__(self, n, edges):
        self.n, self.edges = n, edges
        self.nb = [set() for _ in range(n)]
        for u, v in edges:
            self.nb[u].add(v); self.nb[v].add(u)
        self.nb = [frozenset(s) for s in self.nb]
        self._g = None
        self._dist = None

    def g(self):
        if self._g is None:
            self._g = nx.Graph(); self._g.add_nodes_from(range(self.n)); self._g.add_edges_from(self.edges)
        return self._g

    def env(self):
        nb, n = self.nb, self.n

        def vertex(v):
            if not isinstance(v, int) or not 0 <= v < n:
                raise ValueError(f'{v!r} is not a vertex')
            return v

        def adj(u, v):
            return vertex(v) in nb[vertex(u)]

        def deg(v):
            return len(nb[vertex(v)])

        def nbrs(v):
            return nb[vertex(v)]

        def dist(u, v):
            if self._dist is None:
                self._dist = dict(nx.all_pairs_shortest_path_length(self.g()))
            return self._dist[vertex(u)].get(vertex(v), INF)

        cache = {}

        def once(name, f):
            def wrapped():
                if name not in cache:
                    cache[name] = f()
                return cache[name]
            return wrapped

        g = self.g
        return {
            'n': n, 'V': tuple(range(n)), 'E': tuple(sorted(self.edges)), 'INF': INF,
            'adj': adj, 'deg': deg, 'nbrs': nbrs, 'dist': dist,
            'connected': once('connected', lambda: n > 0 and nx.is_connected(g())),
            'components': once('components', lambda: nx.number_connected_components(g())),
            'triangles': once('triangles', lambda: sum(nx.triangles(g()).values()) // 3),
            'girth': once('girth', lambda: _girth(g())),
            'diameter': once('diameter', lambda: nx.diameter(g()) if n > 0 and nx.is_connected(g()) else INF),
            'clique_number': once('clique', lambda: max((len(c) for c in nx.find_cliques(g())), default=0)),
            'independence_number': once('indep', lambda: max((len(c) for c in nx.find_cliques(nx.complement(g()))), default=0)),
            'is_bipartite': once('bip', lambda: nx.is_bipartite(g())),
            'edge_connectivity': once('econn', lambda: nx.edge_connectivity(g()) if n > 1 else 0),
            'vertex_connectivity': once('vconn', lambda: nx.node_connectivity(g()) if n > 1 else 0),
        }


def _girth(g):
    """Length of a shortest cycle, INF for a forest."""
    best = INF
    for s in g.nodes:
        depth, parent, queue = {s: 0}, {s: None}, [s]
        for u in queue:
            for v in g[u]:
                if v not in depth:
                    depth[v], parent[v] = depth[u] + 1, u
                    queue.append(v)
                elif parent[u] != v:
                    best = min(best, depth[u] + depth[v] + 1)
    return best


# ------------------------------------------------------------------ set systems

def sets_from(obj, params):
    """{"m": int, "sets": [[...], ...]} -> (m, tuple of frozensets in a canonical order). No duplicate sets."""
    if not isinstance(obj, dict) or set(obj) - {'m', 'sets'} or 'm' not in obj or 'sets' not in obj:
        raise FormatError('a set system is {"m": int, "sets": [[...], ...]} and nothing else')
    m = _int(obj['m'], 'm')
    if not 1 <= m <= LIMITS['set_m']:
        raise FormatError(f'm must be between 1 and {LIMITS["set_m"]}')
    if 'm' in params and m != params['m']:
        raise FormatError(f'm must be {params["m"]}')
    if m < params.get('m_min', 1) or m > params.get('m_max', LIMITS['set_m']):
        raise FormatError('m is outside the allowed range')
    if not isinstance(obj['sets'], list) or len(obj['sets']) > LIMITS['set_family']:
        raise FormatError(f'sets must be a list of at most {LIMITS["set_family"]} sets')
    family = set()
    for s in obj['sets']:
        if not isinstance(s, list):
            raise FormatError('each set is a list of elements')
        elems = [_int(x, 'element') for x in s]
        if any(not 0 <= x < m for x in elems):
            raise FormatError(f'set {s} has an element outside 0..{m - 1}')
        if len(set(elems)) != len(elems):
            raise FormatError(f'set {s} repeats an element')
        fs = frozenset(elems)
        if fs in family:
            raise FormatError(f'set {sorted(fs)} appears twice')
        family.add(fs)
    return m, tuple(sorted(family, key=lambda f: (len(f), sorted(f))))


def set_env(m, family):
    return {'m': m, 'ground': tuple(range(m)), 'F': family, 'size': len}


OBJECTS = {
    'graph': {'parse': graph_from, 'env': lambda parsed: GraphEnv(*parsed).env()},
    'set_system': {'parse': sets_from, 'env': lambda parsed: set_env(*parsed)},
}
ENV_NAMES = {
    'graph': ['n', 'V', 'E', 'INF', 'adj', 'deg', 'nbrs', 'dist', 'connected', 'components', 'triangles', 'girth',
              'diameter', 'clique_number', 'independence_number', 'is_bipartite', 'edge_connectivity',
              'vertex_connectivity'],
    'set_system': ['m', 'ground', 'F', 'size'],
}


def to_json(kind, parsed):
    """The canonical JSON form of a parsed object (used by search and for hashing)."""
    if kind == 'graph':
        n, edges = parsed
        return {'n': n, 'edges': [list(e) for e in sorted(edges)]}
    m, family = parsed
    return {'m': m, 'sets': [sorted(f) for f in family]}
