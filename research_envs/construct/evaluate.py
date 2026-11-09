"""The trusted evaluator for domain B.

A spec (JSON) names an object type, a problem kind, size params, optional given data, named constraints, and either an
objective (optimize) or a claim (counterexample). `Spec` validates and compiles it once; `Spec.evaluate(candidate)`
checks a candidate exactly.

Validity is separate from objective value: an invalid candidate reports which constraints fail (and its objective when
computable, for diagnostics only). Objective values are raw, never clamped or rescaled.
"""
import json
from .domain import OBJECTS, ENV_NAMES, FormatError
from .lang import compile_expr, graded, run, timed, SpecError, EvalTimeout, StepBudget

KINDS = ('optimize', 'satisfy', 'counterexample')
NAME_OK = set('abcdefghijklmnopqrstuvwxyz0123456789_')


def _freeze(x):
    if isinstance(x, list):
        return tuple(_freeze(y) for y in x)
    if isinstance(x, bool) or not isinstance(x, int):
        raise SpecError('given data must be integers or nested lists of integers')
    return x


class Spec:
    def __init__(self, spec, timeout=10.0):
        if not isinstance(spec, dict):
            raise SpecError('a spec is a JSON object')
        self.raw = spec
        self.object = spec.get('object')
        if self.object not in OBJECTS:
            raise SpecError(f'object must be one of {sorted(OBJECTS)}')
        self.kind = spec.get('kind')
        if self.kind not in KINDS:
            raise SpecError(f'kind must be one of {KINDS}')
        self.params = spec.get('params') or {}
        if not isinstance(self.params, dict) or any(not isinstance(v, int) or isinstance(v, bool) for v in self.params.values()):
            raise SpecError('params map names to integers')
        self.given = {}
        for k, v in (spec.get('given') or {}).items():
            if not k.isupper() or k.startswith('_') or not k.replace('_', '').isalnum():
                raise SpecError(f'given data name {k!r} must be UPPER_CASE')
            self.given[k] = _freeze(v)
        names = ENV_NAMES[self.object] + list(self.given)
        self.constraints = []
        for c in spec.get('constraints') or []:
            if not isinstance(c, dict) or not isinstance(c.get('name'), str) or set(c['name']) - NAME_OK:
                raise SpecError('each constraint is {"name": "snake_case", "expr": "..."}')
            self.constraints.append((c['name'], compile_expr(c.get('expr'), names), c['expr']))
        self.names = names
        self._graded = None
        if len({n for n, _, _ in self.constraints}) != len(self.constraints):
            raise SpecError('constraint names must be distinct')
        self.objective = self.claim = None
        if self.kind == 'optimize':
            o = spec.get('objective') or {}
            if o.get('direction') not in ('max', 'min'):
                raise SpecError('objective.direction must be "max" or "min"')
            self.objective = (compile_expr(o.get('expr'), names), o['direction'])
        elif 'objective' in spec:
            raise SpecError('only optimize specs have an objective')
        if self.kind == 'counterexample':
            self.claim = compile_expr((spec.get('claim') or {}).get('expr'), names)
        elif 'claim' in spec:
            raise SpecError('only counterexample specs have a claim')
        if self.kind != 'optimize' and not self.constraints and self.kind == 'satisfy':
            raise SpecError('a satisfy spec needs constraints')
        self.timeout = timeout

    # --------------------------------------------------------------
    def parse(self, candidate):
        return OBJECTS[self.object]['parse'](candidate, self.params)

    def evaluate(self, candidate):
        """Exact check of one candidate. Returns a JSON-serialisable verdict."""
        out = {'valid': False, 'format_error': None, 'violations': [], 'errors': {}, 'objective': None,
               'claim_holds': None}
        try:
            parsed = self.parse(candidate)
        except FormatError as e:
            out['format_error'] = str(e)
            return out
        return self.evaluate_parsed(parsed, out)

    def evaluate_parsed(self, parsed, out=None):
        out = out or {'valid': False, 'format_error': None, 'violations': [], 'errors': {}, 'objective': None,
                      'claim_holds': None}
        env = OBJECTS[self.object]['env'](parsed)
        env.update(self.given)
        try:
            with timed(self.timeout):
                for name, code, _ in self.constraints:
                    try:
                        ok = run(code, env)
                    except (EvalTimeout, SpecError):
                        raise
                    except StepBudget as e:
                        out['errors'][name] = str(e)
                        ok = False
                    except Exception as e:  # a constraint that cannot be evaluated on this object fails, with the reason
                        out['errors'][name] = f'{type(e).__name__}: {e}'
                        ok = False
                    if not isinstance(ok, bool):
                        raise SpecError(f'constraint {name} returned {type(ok).__name__}, not a boolean')
                    if not ok:
                        out['violations'].append(name)
                if self.objective:
                    try:
                        value = run(self.objective[0], env)
                    except (EvalTimeout, SpecError):
                        raise
                    except Exception as e:
                        out['errors']['objective'] = f'{type(e).__name__}: {e}'
                        value = None
                    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                        raise SpecError('the objective must be an integer')
                    out['objective'] = value
                if self.claim is not None:
                    try:
                        holds = run(self.claim, env)
                    except (EvalTimeout, SpecError):
                        raise
                    except Exception as e:  # a claim that cannot be evaluated makes the candidate invalid, with the reason
                        out['errors']['claim'] = f'{type(e).__name__}: {e}'
                        holds = None
                    if holds is not None and not isinstance(holds, bool):
                        raise SpecError('the claim must be a boolean')
                    out['claim_holds'] = holds
        except EvalTimeout as e:
            out['errors']['timeout'] = str(e)
            return out
        out['valid'] = (not out['violations'] and not out['errors']
                        and (self.kind != 'counterexample' or out['claim_holds'] is False)
                        and (self.kind != 'optimize' or out['objective'] is not None))
        return out

    def violation_degree(self, parsed):
        """Graded violation summed over constraints (search only; 0 iff every constraint holds)."""
        if self._graded is None:
            self._graded = [(name, graded(src, self.names)) for name, _, src in self.constraints]
        env = OBJECTS[self.object]['env'](parsed)
        env.update(self.given)
        total = 0
        try:
            with timed(self.timeout):
                for name, code in self._graded:
                    try:
                        total += max(0, int(run(code, env)))
                    except (EvalTimeout, SpecError):
                        raise
                    except Exception:
                        total += 1
        except EvalTimeout:
            total += 10 ** 6
        return total

    def better(self, a, b):
        """True if objective value a is strictly better than b (optimize specs)."""
        return a > b if self.objective[1] == 'max' else a < b


def load(path, timeout=10.0):
    return Spec(json.loads(open(path).read()), timeout)
