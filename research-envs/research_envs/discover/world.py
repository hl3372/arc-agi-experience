"""Domain A worlds: small deterministic machines with integer state, compiled to explicit transition tables.

A world (JSON):
  {"state": {"x": {"range": [0, 9], "boundary": "clamp" | "wrap", "init": 0, "settable": true}, ...},
   "actions": {"A": {"rules": [{"if": "<bool expr>", "set": {"x": "<int expr>", ...}}, ...]},
               "C": {"param": [0, 3], "rules": [...]}},          # parameterised: input symbols C0..C3, name k
   "tick": [{"if": ..., "set": {...}}],                           # optional; after every action
   "readouts": {"r1": "<int or bool expr>", ...},
   "max_len": 6}

Semantics. An action applies its first rule whose guard holds (a rule without "if" always holds); its assignments are
simultaneous (every right-hand side reads the old state). Then each assigned variable is brought into range by its
boundary rule (clamp, or wrap modulo the range size). Then the tick, if any, applies the same way. Readouts are read
after reset and after every action. Expressions use domain B's validated language (construct/lang.py) with the state
variables (and the parameter k) as names.

Compilation enumerates the whole state grid (at most MAX_STATES states), so a world becomes:
  next[input][state] -> state,  out[state] -> readout tuple,  start(settable values) -> state.
Every later check (experiments, rivals, determinacy, minimization) is a table operation.
"""
import itertools
import json
import numpy as np
from ..construct.lang import compile_expr, run, SpecError, reset_steps

MAX_STATES = 2000
MAX_INPUTS = 12
MAX_LEN = 10


class WorldError(ValueError):
    pass


def _int(v, what):
    if isinstance(v, bool) or not isinstance(v, int):
        raise WorldError(f'{what} must be an integer')
    return v


class World:
    def __init__(self, spec):
        if isinstance(spec, str):
            spec = json.loads(spec)
        if not isinstance(spec, dict):
            raise WorldError('a world is a JSON object')
        self.raw = spec
        st = spec.get('state')
        if not isinstance(st, dict) or not 1 <= len(st) <= 6:
            raise WorldError('state: 1 to 6 variables')
        self.vars = list(st)
        self.lo, self.hi, self.boundary, self.init, self.settable = {}, {}, {}, {}, []
        for v, d in st.items():
            if not v.isidentifier() or v.startswith('_') or v == 'k':
                raise WorldError(f'bad variable name {v!r}')
            lo, hi = (_int(x, f'{v} range') for x in d.get('range', [None, None]))
            if hi < lo or hi - lo > 63:
                raise WorldError(f'{v}: range must be lo..hi with at most 64 values')
            self.lo[v], self.hi[v] = lo, hi
            self.boundary[v] = d.get('boundary', 'clamp')
            if self.boundary[v] not in ('clamp', 'wrap'):
                raise WorldError(f'{v}: boundary is clamp or wrap')
            self.init[v] = _int(d.get('init', lo), f'{v} init')
            if not lo <= self.init[v] <= hi:
                raise WorldError(f'{v}: init out of range')
            if d.get('settable'):
                self.settable.append(v)
        sizes = [self.hi[v] - self.lo[v] + 1 for v in self.vars]
        self.n_states = int(np.prod(sizes))
        if self.n_states > MAX_STATES:
            raise WorldError(f'{self.n_states} states exceed {MAX_STATES}')
        self.sizes = sizes
        self.max_len = _int(spec.get('max_len', 6), 'max_len')
        if not 1 <= self.max_len <= MAX_LEN:
            raise WorldError(f'max_len must be 1..{MAX_LEN}')
        names = set(self.vars)
        acts = spec.get('actions')
        if not isinstance(acts, dict) or not 1 <= len(acts) <= 8:
            raise WorldError('actions: 1 to 8')
        self.inputs = []          # (action, param or None)
        self.rules = {}
        for a, d in acts.items():
            if not a.isidentifier():
                raise WorldError(f'bad action name {a!r}')
            prm = d.get('param')
            if prm is not None:
                plo, phi = (_int(x, f'{a} param') for x in prm)
                if phi < plo or phi - plo > 9:
                    raise WorldError(f'{a}: param range of at most 10 values')
                self.inputs += [(a, k) for k in range(plo, phi + 1)]
            else:
                self.inputs.append((a, None))
            self.rules[a] = self._rules(d.get('rules'), names | ({'k'} if prm is not None else set()), a)
        if len(self.inputs) > MAX_INPUTS:
            raise WorldError(f'more than {MAX_INPUTS} input symbols')
        self.tick = self._rules(spec.get('tick') or [], names, 'tick', allow_empty=True)
        ro = spec.get('readouts')
        if not isinstance(ro, dict) or not 1 <= len(ro) <= 4:
            raise WorldError('readouts: 1 to 4')
        self.readout_names = list(ro)
        self.readouts = [(r, compile_expr(e, names), e) for r, e in ro.items()]
        self._compile()

    def _rules(self, rules, names, where, allow_empty=False):
        if not isinstance(rules, list) or (not rules and not allow_empty):
            raise WorldError(f'{where}: rules must be a non-empty list')
        out = []
        for r in rules:
            if not isinstance(r, dict) or not isinstance(r.get('set', {}), dict):
                raise WorldError(f'{where}: each rule is {{"if": expr, "set": {{var: expr}}}}')
            for v in r.get('set', {}):
                if v not in self.vars:
                    raise WorldError(f'{where}: assigns unknown variable {v!r}')
            guard = compile_expr(r['if'], names) if r.get('if') else None
            sets = [(v, compile_expr(e, names)) for v, e in r.get('set', {}).items()]
            out.append((guard, sets))
        return out

    # ------------------------------------------------------------------ state encoding
    def encode(self, vals):
        idx = 0
        for v, size in zip(self.vars, self.sizes):
            idx = idx * size + (vals[v] - self.lo[v])
        return idx

    def decode(self, idx):
        vals = {}
        for v, size in reversed(list(zip(self.vars, self.sizes))):
            idx, r = divmod(idx, size)
            vals[v] = r + self.lo[v]
        return {v: vals[v] for v in self.vars}

    def _bound(self, v, x):
        if isinstance(x, bool) or not isinstance(x, int):
            raise SpecError(f'{v} assigned a non-integer')
        lo, hi = self.lo[v], self.hi[v]
        if self.boundary[v] == 'wrap':
            return lo + (x - lo) % (hi - lo + 1)
        return min(hi, max(lo, x))

    def _apply(self, rules, vals, extra):
        env = {**vals, **extra}
        for guard, sets in rules:
            if guard is None or run(guard, env):
                new = dict(vals)
                for v, code in sets:
                    new[v] = self._bound(v, run(code, env))
                return new
        return vals

    def _compile(self):
        n = self.n_states
        self.next = np.zeros((len(self.inputs), n), dtype=np.int32)
        outs = []
        for s in range(n):
            vals = self.decode(s)
            reset_steps()
            row = []
            for r, code, _ in self.readouts:
                x = run(code, vals)
                if not isinstance(x, (int, bool)):
                    raise WorldError(f'readout {r} must be an integer or boolean')
                row.append(int(x))
            outs.append(tuple(row))
            for i, (a, k) in enumerate(self.inputs):
                new = self._apply(self.rules[a], vals, {} if k is None else {'k': k})
                if self.tick:
                    new = self._apply(self.tick, new, {})
                self.next[i, s] = self.encode(new)
        self.out_symbols = sorted(set(outs))
        sym = {o: j for j, o in enumerate(self.out_symbols)}
        self.out = np.array([sym[o] for o in outs], dtype=np.int32)
        self.out_tuple = outs

    # ------------------------------------------------------------------ the experiment interface
    def input_index(self, name, k=None):
        try:
            return self.inputs.index((name, k))
        except ValueError:
            raise WorldError(f'no action {name}' + (f' with k={k}' if k is not None else '')) from None

    def start_state(self, settings):
        settings = settings or {}
        extra = set(settings) - set(self.settable)
        if extra:
            raise WorldError(f'not settable: {sorted(extra)}')
        vals = dict(self.init)
        for v, x in settings.items():
            x = _int(x, v)
            if not self.lo[v] <= x <= self.hi[v]:
                raise WorldError(f'{v}={x} out of range {self.lo[v]}..{self.hi[v]}')
            vals[v] = x
        return self.encode(vals)

    def starts(self):
        """Every settable start (a list of settings dicts)."""
        grids = [range(self.lo[v], self.hi[v] + 1) for v in self.settable]
        return [dict(zip(self.settable, c)) for c in itertools.product(*grids)]

    def run_inputs(self, s, seq):
        """Readout tuples after reset and after each input index in seq."""
        trace = [self.out_tuple[s]]
        for i in seq:
            s = int(self.next[i, s])
            trace.append(self.out_tuple[s])
        return trace

    def experiment(self, exp, limit=None):
        """exp = {"start": {...}, "actions": [["A"], ["C", 2], ...]} -> list of readout dicts."""
        acts = exp.get('actions', [])
        if len(acts) > (limit or self.max_len):
            raise WorldError(f'at most {limit or self.max_len} actions per experiment')
        seq = [self.input_index(a[0], a[1] if len(a) > 1 else None) for a in acts]
        trace = self.run_inputs(self.start_state(exp.get('start')), seq)
        return [dict(zip(self.readout_names, t)) for t in trace]


def input_label(world, i):
    a, k = world.inputs[i]
    return a if k is None else f'{a}{k}'
