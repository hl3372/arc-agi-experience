"""The rival set R: laws near the true one, against which fairness and identifiability are computed.

R is an assumption, not a fact about the solver: laws far from the true one are not considered. It contains
- single-change mutants of every expression (guards, assignments, readouts): a flipped comparison, a constant +-1,
  a changed arithmetic operator, one variable reference replaced by another;
- dropped pieces: a guard (the rule always fires), an assignment, a whole rule;
- structural rivals: a boundary rule swapped (clamp <-> wrap), the tick removed, a hidden variable frozen (no
  memory), the tick's direction flipped;
- named rivals supplied by the planner.
Rivals keep the interface (state variables, settable starts, actions, readout names), so experiments are comparable.
"""
import ast
import copy
import json
import os
from .world import World

FLIPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
OPS = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult}


class _Mut(ast.NodeTransformer):
    def __init__(self, target, var_names):
        self.target, self.count, self.desc, self.vars = target, 0, None, var_names

    def _hit(self, desc):
        hit = self.count == self.target
        self.count += 1
        if hit:
            self.desc = desc
        return hit

    def visit_Compare(self, node):
        self.generic_visit(node)
        if type(node.ops[0]) in FLIPS and self._hit(f'flip {type(node.ops[0]).__name__}'):
            node.ops[0] = FLIPS[type(node.ops[0])]()
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in OPS and self._hit(f'{type(node.op).__name__} -> {OPS[type(node.op)].__name__}'):
            node.op = OPS[type(node.op)]()
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            if self._hit(f'{node.value} -> {node.value + 1}'):
                return ast.copy_location(ast.Constant(node.value + 1), node)
            if self._hit(f'{node.value} -> {node.value - 1}'):
                return ast.copy_location(ast.Constant(node.value - 1), node)
        return node

    def visit_Name(self, node):
        if node.id in self.vars:
            for other in self.vars:
                if other != node.id and self._hit(f'{node.id} -> {other}'):
                    return ast.copy_location(ast.Name(id=other, ctx=ast.Load()), node)
        return node


def _expr_mutants(src, var_names):
    tree = ast.parse(src, mode='eval')
    k = 0
    while True:
        m = _Mut(k, var_names)
        new = m.visit(copy.deepcopy(tree))
        if m.desc is None:
            return
        k += 1
        yield m.desc, ast.unparse(ast.fix_missing_locations(new))


def _sites(spec):
    """Every expression in the world, as (path, source)."""
    for a, d in spec['actions'].items():
        for i, r in enumerate(d['rules']):
            if r.get('if'):
                yield ('actions', a, 'rules', i, 'if'), r['if']
            for v, e in r.get('set', {}).items():
                yield ('actions', a, 'rules', i, 'set', v), e
    for i, r in enumerate(spec.get('tick') or []):
        if r.get('if'):
            yield ('tick', i, 'if'), r['if']
        for v, e in r.get('set', {}).items():
            yield ('tick', i, 'set', v), e
    for r, e in spec['readouts'].items():
        yield ('readouts', r), e


def _put(spec, path, value):
    s = copy.deepcopy(spec)
    node = s
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    return s


def _rules_at(spec):
    for a, d in spec['actions'].items():
        yield ('actions', a, 'rules'), d['rules']
    if spec.get('tick'):
        yield ('tick',), spec['tick']


def _get(spec, path):
    node = spec
    for p in path:
        node = node[p]
    return node


def candidates(spec):
    """Every rival spec (label, spec) before compilation."""
    var_names = list(spec['state'])
    out = []
    for path, src in _sites(spec):
        for desc, new in _expr_mutants(src, var_names):
            out.append((f"{'/'.join(map(str, path))}: {desc}", _put(spec, path, new)))
    for path, rules in _rules_at(spec):
        where = '/'.join(map(str, path))
        for i, r in enumerate(rules):
            if r.get('if'):
                s = copy.deepcopy(spec); _get(s, path)[i].pop('if')
                out.append((f'{where}/{i}: drop guard', s))
            for v in r.get('set', {}):
                s = copy.deepcopy(spec); _get(s, path)[i]['set'].pop(v)
                out.append((f'{where}/{i}: drop assignment to {v}', s))
            if len(rules) > 1 or path == ('tick',):
                s = copy.deepcopy(spec); del _get(s, path)[i]
                out.append((f'{where}/{i}: drop rule', s))
    # structural rivals
    for v, d in spec['state'].items():
        s = copy.deepcopy(spec)
        s['state'][v]['boundary'] = 'wrap' if d.get('boundary', 'clamp') == 'clamp' else 'clamp'
        out.append((f'structural: {v} boundary -> {s["state"][v]["boundary"]}', s))
        if not d.get('settable'):
            s = copy.deepcopy(spec)
            for _, rules in _rules_at(s):
                for r in rules:
                    r.get('set', {}).pop(v, None)
            out.append((f'structural: freeze hidden {v} (no memory)', s))
        if RANGE_RIVALS and isinstance(d.get('range'), list) and len(d['range']) == 2:
            lo, hi = d['range']             # "there is no cap here": the natural misconception about a limit
            s = copy.deepcopy(spec)
            s['state'][v]['range'] = [lo, hi + max(2, (hi - lo) // 2)]
            out.append((f'structural: {v} cap raised (no limit at {hi})', s))
    if spec.get('tick'):
        s = copy.deepcopy(spec); s.pop('tick')
        out.append(('structural: no tick', s))
    return out


MAX_MUTANTS = 150
RANGE_RIVALS = os.environ.get('RE_A_QUERY_MODE') == 'discriminating'  # demo v2: cap rivals


def rival_set(spec, named=(), seed=0):
    """Compiled rivals that differ from the truth somewhere in the state table: list of (label, World). All
    structural and named rivals are kept; single-change mutants are sampled down to MAX_MUTANTS (compiling a rival
    costs a pass over the whole state table)."""
    import random
    truth = World(spec)
    tkey = _table_key(truth)
    by_key, order = {}, []
    cands = list(candidates(spec))
    struct = [c for c in cands if c[0].startswith('structural')]
    muts = [c for c in cands if not c[0].startswith('structural')]
    if len(muts) > MAX_MUTANTS:
        muts = random.Random(seed).sample(muts, MAX_MUTANTS)
    for label, s in struct + muts + [(f"named: {n.get('label', i)}", n['world']) for i, n in enumerate(named)]:
        try:
            w = World(s)
        except Exception:
            continue
        key = _table_key(w)
        if key == tkey:
            continue      # the same law as the truth
        if key in by_key:
            by_key[key][0].append(label)    # the same law reached by another change: keep the label as an alias
            continue
        by_key[key] = ([label], w)
        order.append(key)
    return truth, [(' | '.join(by_key[k][0]), by_key[k][1]) for k in order]


def _table_key(w):
    return (w.next.tobytes(), json.dumps(w.out_tuple))
