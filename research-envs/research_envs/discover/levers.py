"""Mechanical checks that a world really has the property its request pinned (round 2 levers).

- time:    the tick changes behaviour (the world without its tick has a different table).
- cycle:   wrapping matters (changing every wrap boundary to clamp changes the table), or a state update uses %.
- alias:   a readout quantizes or thresholds the state (//, %, a comparison).
- aggregate (diagnostic): a readout combines two or more hidden variables.
- coupled: two state variables each update from the other (mutual influence).
"""
import ast
import copy
import json
from .world import World
from .novelty import reachable


def _table(w):
    return (w.next.tobytes(), json.dumps(w.out_tuple))


def _names(src):
    return {n.id for n in ast.walk(ast.parse(src, mode='eval')) if isinstance(n, ast.Name)}


def _rules(spec):
    for d in spec['actions'].values():
        yield from d['rules']
    yield from spec.get('tick') or []


def has_time(spec):
    if not spec.get('tick'):
        return False
    s = copy.deepcopy(spec); s.pop('tick')
    try:
        return _table(World(s)) != _table(World(spec))
    except Exception:
        return True


def has_cycle(spec):
    sv = set(spec['state'])
    if any('%' in e and (_names(e) & sv) for r in _rules(spec) for e in r.get('set', {}).values()):
        return True
    if not any(d.get('boundary') == 'wrap' for d in spec['state'].values()):
        return False
    s = copy.deepcopy(spec)
    for d in s['state'].values():
        d['boundary'] = 'clamp'
    try:
        return _table(World(s)) != _table(World(spec))
    except Exception:
        return True


def alias_ratio(spec):
    w = World(spec)
    states, _ = reachable(w)
    return len({tuple(w.out_tuple[s]) for s in states}) / max(1, len(states))


def has_alias(spec):
    """A readout is itself coarse or relational (division, modulo, a comparison, min/max/abs over the state): distinct
    states are folded into one reading by the sensor, not merely hidden. (min/max/abs are left out: they are usually
    clamps.)"""
    sv = set(spec['state'])
    for e in spec['readouts'].values():
        tree = ast.parse(e, mode='eval')
        if not (_names(e) & sv):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.FloorDiv, ast.Mod)):
                return True
            if isinstance(n, ast.Compare):
                return True
    return False


def has_aggregate(spec):
    """Diagnostic: some readout combines two or more hidden variables (a span, a separation, a total), so the solver
    only ever sees an aggregate of what it must infer."""
    hidden = {v for v, d in spec['state'].items() if not d.get('settable')}
    return any(len(_names(e) & hidden) >= 2 for e in spec['readouts'].values())


def has_coupled(spec):
    """Interacting parts: two different state variables that each update from the other (mutual influence), however
    the actions are arranged. (A narrower version requiring separately driven parts rejected real coupled worlds,
    such as two combs that one action moves and trims together.)"""
    sv = set(spec['state'])
    dep = {v: set() for v in sv}
    for r in _rules(spec):
        guard = _names(r['if']) & sv if r.get('if') else set()
        for v, e in r.get('set', {}).items():
            dep[v] |= (_names(e) & sv) | guard
    return any(b != a and a in dep.get(b, set()) for a in sv for b in dep[a])


CHECKS = {'time': has_time, 'cycle': has_cycle, 'alias': has_alias, 'coupled': has_coupled, 'aggregate': has_aggregate}


def profile(spec):
    """Which levers a world has (for every world, so arms and controls are compared on the same terms)."""
    out = {}
    for k, f in CHECKS.items():
        try:
            out[k] = bool(f(spec))
        except Exception:
            out[k] = None
    try:
        out['alias_ratio'] = round(alias_ratio(spec), 3)
    except Exception:
        out['alias_ratio'] = None
    return out
