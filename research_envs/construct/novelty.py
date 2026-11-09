"""Novelty by behaviour: fingerprints, a zoo of standard objects, and the textbook seed archive.

Fingerprint of a parametric spec: at each small size (graphs n = 4..7, set systems at small m with the task's set
sizes), the certified optimum (optimize) and the number of valid labelled objects, by exhaustive enumeration. Two
specs with equal fingerprints behave identically on every small object, whatever their syntax (duals, restatements,
redundant constraints), so they are the same family. Same fingerprint up to size parameters = an instance.

Zoo: standard named objects. A witness isomorphic to a zoo object, or a task whose fingerprint matches a textbook
problem, is calibration, not a new family.
"""
import json
import warnings
import networkx as nx
from pathlib import Path
from .sibling import enumerate_all, at_size
from .search import as_nx

warnings.filterwarnings('ignore', category=UserWarning, module='networkx')
ARCHIVE = Path(__file__).resolve().parents[1] / 'archive'
GRAPH_SIZES = (4, 5, 6, 7)


def fingerprint(raw, hint=None, sizes=None):
    key = 'n' if raw['object'] == 'graph' else 'm'
    if sizes is None:
        sizes = GRAPH_SIZES if key == 'n' else ((4, 5, 6) if (hint or {}).get('set_sizes') else (2, 3, 4))
    out = []
    for s in sizes:
        r = enumerate_all(at_size(raw, **{key: s}), hint)
        if r.get('skipped'):
            out.append(None)
        else:
            out.append((r['optimum'], r['valid']))
    return {'sizes': list(sizes), 'values': out, 'battery': battery_vector(raw, raw['object'])}


def _norm(x):
    return tuple(_norm(y) for y in x) if isinstance(x, (list, tuple)) else x


def same_family(fa, fb):
    """Same behaviour: equal exhaustive values on every size both cover (and they share at least three), and an equal
    probe-battery vector when both have one. Values are normalised (JSON turns tuples into lists)."""
    common = [(_norm(a), _norm(b)) for s, a in zip(fa['sizes'], fa['values']) for t, b in zip(fb['sizes'], fb['values'])
              if s == t and a is not None and b is not None]
    exhaustive_ok = len(common) >= 3 and all(a == b for a, b in common)
    if fa.get('battery') is not None and fb.get('battery') is not None:
        same_battery = _norm(fa['battery']) == _norm(fb['battery'])
        informative = any(v[0] for v in _norm(fa['battery']))
        return same_battery and informative and (exhaustive_ok or len(common) < 3)
    return exhaustive_ok


def battery_objects(kind, seed=7):
    """A fixed, seeded battery of small objects at several sizes. Set families are small (1-4 sets of mixed sizes),
    so pairwise constraints are often satisfiable and the battery is informative."""
    import random
    rng = random.Random(seed)
    out = []
    if kind == 'graph':
        for n in (5, 6, 7, 8):
            pairs = [[a, b] for a in range(n) for b in range(a + 1, n)]
            for k in range(60):
                dens = (k % 6 + 0.5) / 6
                out.append({'n': n, 'edges': [e for e in pairs if rng.random() < dens]})
    else:
        for m in (5, 6, 7, 8, 9, 13):
            for k in range(60):
                fam = set()
                for _ in range(rng.randint(1, 4)):
                    fam.add(tuple(sorted(rng.sample(range(m), rng.randint(1, min(m, 5))))))
                out.append({'m': m, 'sets': [list(x) for x in sorted(fam)]})
    return out


def battery_vector(raw, kind):
    """(valid, objective) of the parametric spec on each battery object (size params taken from the object)."""
    from .evaluate import Spec
    vec = []
    for obj in battery_objects(kind):
        key = 'n' if kind == 'graph' else 'm'
        spec = Spec({**raw, 'params': {**{k: v for k, v in raw.get('params', {}).items() if k not in (key, key + '_min', key + '_max')}, key: obj[key]}})
        r = spec.evaluate(obj)
        vec.append((r['valid'], r['objective'] if r['valid'] else None))
    return vec


# ------------------------------------------------------------------ zoo

def _cayley_z2(k, conn):
    g = nx.Graph(); g.add_nodes_from(range(2 ** k))
    g.add_edges_from((x, x ^ c) for x in range(2 ** k) for c in conn if x < x ^ c)
    return g


def graph_zoo():
    z = {
        'petersen': nx.petersen_graph(), 'heawood': nx.heawood_graph(), 'pappus': nx.pappus_graph(),
        'desargues': nx.desargues_graph(), 'dodecahedron': nx.dodecahedral_graph(), 'icosahedron': nx.icosahedral_graph(),
        'octahedron': nx.octahedral_graph(), 'moebius_kantor': nx.moebius_kantor_graph(), 'cube_q3': nx.hypercube_graph(3),
        'cube_q4': nx.hypercube_graph(4), 'hoffman_singleton': nx.hoffman_singleton_graph(),
        'clebsch': _cayley_z2(4, [1, 2, 4, 8, 15]), 'shrikhande': nx.circulant_graph(1, []),
    }
    sh = nx.Graph()  # Shrikhande: Cayley graph on Z4 x Z4 with connection set +-(1,0), +-(0,1), +-(1,1)
    for a in range(4):
        for b in range(4):
            for da, db in ((1, 0), (0, 1), (1, 1)):
                sh.add_edge((a, b), ((a + da) % 4, (b + db) % 4))
    z['shrikhande'] = sh
    for q in (5, 9, 13, 17):
        z[f'paley_{q}'] = nx.paley_graph(q).to_undirected() if q != 9 else _paley9()
    for n in range(5, 9):
        for k in range(2, n // 2):
            z[f'kneser_{n}_{k}'] = nx.kneser_graph(n, k) if hasattr(nx, 'kneser_graph') else None
    return {k: nx.convert_node_labels_to_integers(nx.Graph(v)) for k, v in z.items() if v is not None}


def _paley9():
    # GF(9) = Z3[i]/(i^2+1); x ~ y when x - y is a nonzero square
    els = [(a, b) for a in range(3) for b in range(3)]
    mul = lambda x, y: ((x[0] * y[0] - x[1] * y[1]) % 3, (x[0] * y[1] + x[1] * y[0]) % 3)
    sq = {mul(x, x) for x in els if x != (0, 0)}
    g = nx.Graph(); g.add_nodes_from(els)
    g.add_edges_from((x, y) for x in els for y in els if x < y and ((x[0] - y[0]) % 3, (x[1] - y[1]) % 3) in sq)
    return g


def set_zoo():
    fano = [[0, 1, 3], [1, 2, 4], [2, 3, 5], [3, 4, 6], [4, 5, 0], [5, 6, 1], [6, 0, 2]]
    pts = [(x, y) for x in range(3) for y in range(3)]
    idx = {p: i for i, p in enumerate(pts)}
    lines = set()
    for a in pts:
        for b in pts:
            if a < b:
                d = ((b[0] - a[0]) % 3, (b[1] - a[1]) % 3)
                lines.add(frozenset(idx[((a[0] + t * d[0]) % 3, (a[1] + t * d[1]) % 3)] for t in range(3)))
    pg23 = [sorted(((i + d) % 13 for d in (0, 1, 3, 9))) for i in range(13)]  # perfect difference set mod 13
    return {'fano': {'m': 7, 'sets': fano}, 'affine_plane_3': {'m': 9, 'sets': [sorted(l) for l in lines]},
            'projective_plane_3': {'m': 13, 'sets': pg23}}


def zoo_match(kind, obj):
    """Name of a zoo object isomorphic to `obj`, or None."""
    g = as_nx(kind, obj)
    if kind == 'graph':
        for name, z in graph_zoo().items():
            if z.number_of_nodes() == g.number_of_nodes() and z.number_of_edges() == g.number_of_edges() \
                    and nx.is_isomorphic(g, z):
                return name
        return None
    match = nx.algorithms.isomorphism.categorical_node_match('side', None)
    for name, z in set_zoo().items():
        zg = as_nx(kind, z)
        if zg.number_of_nodes() == g.number_of_nodes() and nx.is_isomorphic(g, zg, node_match=match):
            return name
    return None


def load_archive():
    path = ARCHIVE / 'textbook.json'
    return json.loads(path.read_text()) if path.exists() else []
