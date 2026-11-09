"""The experiment interface for pilots: public names in, traces out, budgets enforced, everything recorded.

Public names: actions A, B, C, ... (in the world's action order; a parameterised action C with k = 2 is written C2),
settable start values s1, s2, ... (in the world's settable order), readouts r1, r2, ... (in readout order). An invalid
experiment (unknown action, value out of range, too long) still uses one experiment from the budget and returns the
error, so the budget cannot be stretched by probing the interface.
"""
import json
from pathlib import Path
from . import prompts as P
from .fairness import answer


class PublicMap:
    def __init__(self, world):
        self.w = world
        self.labels = {}
        for i, (name, d) in enumerate(world.raw['actions'].items()):
            lab = chr(ord('A') + i)
            if d.get('param'):
                for k in range(d['param'][0], d['param'][1] + 1):
                    self.labels[f'{lab}{k}'] = (name, k)
            else:
                self.labels[lab] = (name, None)
        self.rev = {v: k for k, v in self.labels.items()}
        self.svars = ({v: v for v in world.settable} if P.V2 else {f's{j + 1}': v for j, v in enumerate(world.settable)})

    def to_world(self, exp):
        """Public experiment -> (start settings, input indices); raises ValueError with a public message."""
        if not isinstance(exp, dict):
            raise ValueError('an experiment is {"start": {...}, "actions": [...]}')
        start = exp.get('start') or {}
        if not isinstance(start, dict):
            raise ValueError('start must map s1, s2, ... to integers')
        st = {}
        for k, v in start.items():
            if k not in self.svars:
                raise ValueError(f'unknown settable value {k!r}')
            var = self.svars[k]
            if isinstance(v, bool) or not isinstance(v, int) or not self.w.lo[var] <= v <= self.w.hi[var]:
                raise ValueError(f'{k} must be an integer in {self.w.lo[var]}..{self.w.hi[var]}')
            st[var] = v
        for k, var in self.svars.items():      # unspecified settable values start at their default
            st.setdefault(var, self.w.init[var])
        acts = exp.get('actions') or []
        if not isinstance(acts, list) or len(acts) > self.w.max_len:
            raise ValueError(f'actions must be a list of at most {self.w.max_len} names')
        seq = []
        for a in acts:
            lab = self.norm_action(a)
            if lab not in self.labels:
                raise ValueError(f'unknown action {a!r}')
            seq.append(self.w.input_index(*self.labels[lab]))
        return st, seq

    @staticmethod
    def norm_action(a):
        """Accept the forms pilots write: "C2", "C(2)", "C 2", "c2", ["C", 2]."""
        if isinstance(a, list):
            a = ''.join(str(x) for x in a)
        if not isinstance(a, str):
            return a
        a = a.strip().replace('(', '').replace(')', '').replace(' ', '').replace('k=', '')
        return a[:1].upper() + a[1:]

    def public_start(self, st):
        inv = {v: k for k, v in self.svars.items()}
        return {inv[var]: val for var, val in st.items() if var in inv}

    def public_query(self, q):
        return {'start': self.public_start(q['start']),
                'actions': [self.rev[self.w.inputs[i]] for i in q['inputs']]}

    def run(self, exp):
        st, seq = self.to_world(exp)
        return [list(t) for t in self.w.run_inputs(self.w.start_state(st), seq)]


def run_pilot(world, description, queries, budget, calls, call, per_turn=8, max_turns=None, effort='high', name='pilot'):
    """One interactive pilot at a budget. call(prompt, schema, record_dir, name, effort=...) is the model call."""
    pm = PublicMap(world)
    pub = P.public_interface(world, description, budget)
    history, used, turns, beliefs = [], 0, 0, []
    max_turns = max_turns or (budget + per_turn - 1) // per_turn + 2
    turn_error = None
    while used < budget and turns < max_turns:
        turns += 1
        try:
            r = call(P.pilot_turn_prompt(pub, history, budget - used, min(per_turn, budget - used)), P.PILOT_TURN_SCHEMA,
                     calls, f'{name}_turn_{turns}', effort=effort)
        except Exception as e:           # a failed turn ends the experimenting; the pilot still answers the queries
            turn_error = f'{type(e).__name__}: {e}'
            break
        beliefs.append(r.get('belief'))
        try:
            exps = json.loads(r.get('experiments_json') or '[]')
        except json.JSONDecodeError:
            exps = []
        if not isinstance(exps, list):
            exps = []
        for e in exps[:min(per_turn, budget - used)]:
            used += 1
            try:
                tr = pm.run(e)
                history.append({'start': e.get('start'), 'actions': e.get('actions'), 'trace': tr})
            except ValueError as err:
                history.append({'start': e.get('start') if isinstance(e, dict) else None,
                                'actions': e.get('actions') if isinstance(e, dict) else None, 'trace': f'invalid: {err}'})
        if r.get('done') or not exps:
            break
    pq = [pm.public_query(q) for q in queries]
    fallback = None
    qrec = Path(calls) / f'{name}_queries' if calls else None
    tried = qrec is not None and qrec.exists() and not (qrec / 'response.json').exists() and any(qrec.glob('attempt-*.json'))
    try:
        if tried:
            raise RuntimeError('the full query call already failed')
        r = call(P.pilot_query_prompt(pub, history, pq), P.PILOT_QUERY_SCHEMA, calls, f'{name}_queries', effort=effort)
    except Exception as e:               # fallback: only the law as a program (the scored answer for every solver)
        fallback = f'{type(e).__name__}: {e}'
        r = call(P.pilot_program_prompt(pub, history, pq), P.PILOT_PROGRAM_SCHEMA, calls, f'{name}_queries_program',
                 effort=effort)
    try:
        direct = json.loads(r.get('answers_json') or '{}')
    except json.JSONDecodeError:
        direct = {}
    truths = [list(answer(world, q)) for q in queries]
    names = world.readout_names
    dhits = _score(direct, truths, names)
    prog, phits = None, None
    if (r.get('predictor_program') or '').strip():
        from ..construct.sandbox import run_program, SandboxUnavailable
        try:
            run = run_program(r['predictor_program'], {'queries': pq}, timeout=60)
        except SandboxUnavailable as e:   # never run unsandboxed: score the direct answers, and say so in the record
            run = {'ok': False, 'timed_out': False, 'stderr_tail': '', 'object': None, 'sandbox_unavailable': str(e)}
        prog = {'ran': run['ok'], 'timed_out': run['timed_out'], 'stderr_tail': run['stderr_tail'][-400:]}
        if run.get('sandbox_unavailable'):
            prog['sandbox_unavailable'] = run['sandbox_unavailable']
        if run['ok'] and isinstance(run['object'], dict):
            answered = sum(f'Q{i + 1}' in run['object'] for i in range(len(queries)))
            prog['answered'] = answered
            if answered >= 0.9 * len(queries):
                phits = _score(run['object'], truths, names)
    hits = phits if phits is not None else dhits
    crit = [h for h, q in zip(hits, queries) if q.get('disagree')]
    def acc(h, crit_only=False):
        hh = [x for x, q in zip(h, queries) if q.get('disagree')] if crit_only else h
        return sum(hh) / len(hh) if hh else None
    return {'budget': budget, 'experiments_used': used, 'turns': turns, 'history': history, 'beliefs': beliefs,
            'turn_error': turn_error, 'query_fallback': fallback,
            'law': r.get('law'), 'answers': direct, 'program': prog, 'scored_from': 'program' if phits is not None else 'answers',
            'hits': hits, 'accuracy': acc(hits), 'critical_accuracy': sum(crit) / len(crit) if crit else None,
            'direct_accuracy': acc(dhits), 'direct_critical_accuracy': acc(dhits, True),
            'program_accuracy': acc(phits) if phits is not None else None,
            'program_critical_accuracy': acc(phits, True) if phits is not None else None}


def _norm_answer(v, names):
    """Accept 5 for [5], true for 1, {"r1": 3, ...} for [3, ...], and a time series of readouts (a list of readout
    lists, or for one readout a flat list longer than one) for its final readouts: the question asks for the readouts
    after the last action, and some solvers answer with the whole trajectory."""
    n = len(names)
    if isinstance(v, dict):
        v = [v.get(f'r{j + 1}', v.get(nm)) for j, nm in enumerate(names)]
    if isinstance(v, (bool, int)):
        v = [v]
    if isinstance(v, list) and v and all(isinstance(x, list) for x in v):
        v = v[-1]
    elif isinstance(v, list) and len(v) > n and len(v) % n == 0 and all(isinstance(x, (int, bool, float)) for x in v):
        v = v[-n:]
    if not isinstance(v, list):
        return None
    out = []
    for x in v:
        if isinstance(x, bool):
            x = int(x)
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        if not isinstance(x, int):
            return None
        out.append(x)
    return out


def _score(ans, truths, names):
    return [isinstance(ans, dict) and _norm_answer(ans.get(f'Q{i + 1}'), names) == t for i, t in enumerate(truths)]
