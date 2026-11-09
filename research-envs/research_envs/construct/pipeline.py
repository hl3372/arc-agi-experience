"""Domain B pipeline: requests -> ideas -> mechanical prechecks -> select -> plan -> build and audit -> blind restatement
-> pilots -> mechanical adjudication -> dossiers.

Model calls run in parallel threads; compute-heavy checks (enumeration, battery, probes) run one task at a time, each
using all cores. Every call and verdict is recorded under the round directory; a rerun resumes from finished calls.
"""
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from . import llm, prompts, search, sibling, probes, novelty
from .evaluate import Spec
from .lang import SpecError
from .sandbox import run_program
from ..common import ROOT, FileSlots    # runs/ lives under ROOT (see research_envs/common.py)

B = int(os.environ.get('RE_BUDGET', 20000))   # base battery budget; persistence is checked at 10 * B
PROCESSES = 4                 # worker processes per compute job
# concurrent compute jobs on this machine (the paper's runs: 5 on 20 cores); RE_COMPUTE_SLOTS overrides
COMPUTE_SLOTS = int(os.environ.get('RE_COMPUTE_SLOTS') or max(1, min(5, (os.cpu_count() or 4) // PROCESSES)))
PROGRAM_SLOTS = int(os.environ.get('RE_PROGRAM_SLOTS', 3))   # concurrent sandboxed pilot programs
METHODS = {                   # demand -> (shortcuts that must fall short, methods that must reach the reference)
    'D1': (['relax_repair'], ['anneal', 'kick']),
    'D2': (['anneal', 'kick'], ['symmetric']),
    'D3': (['hill_climb'], ['kick']),
    'D4': (['margin'], []),
    None: (['greedy_add', 'hill_climb', 'kick', 'anneal', 'symmetric'], []),
}


def log(round_dir, msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(round_dir / 'log.txt', 'a') as f:
        f.write(line + '\n')


def jload(s, default=None):
    try:
        return json.loads(s) if isinstance(s, str) and s.strip() else default
    except json.JSONDecodeError:
        return default


def archive_lines(extra=()):
    rows = [f"- {a['title']}: {'; '.join(c['expr'] for c in a['constraints'])}"
            + (f"; {a['objective']['direction']} {a['objective']['expr']}" if a.get('objective') else '')
            for a in novelty.load_archive()]
    rows += [f"- {t}" for t in extra]
    return '\n'.join(rows)


# ------------------------------------------------------------------ mechanical prechecks of an idea

def precheck(cand):
    out = {'compiles': False}
    raw = jload(cand.get('spec_json'))
    sib = jload(cand.get('sibling_params_json'), {}) or {}
    hint = jload(cand.get('search_hint_json'), {}) or {}
    try:
        spec = Spec(raw)
        out['compiles'] = True
    except (SpecError, TypeError, AttributeError) as e:
        out['error'] = f'spec does not compile: {e}'
        return out, None
    out['kind'] = spec.kind
    out['object_type'] = spec.object
    try:
        s = sibling.enumerate_all(sibling.at_size(raw, **sib), hint)
        out['sibling'] = {k: s.get(k) for k in ('objects', 'valid', 'optimum', 'optimal_classes', 'redundant_constraints', 'skipped')}
    except Exception as e:
        out['sibling'] = {'error': f'{type(e).__name__}: {e}'}
    try:
        fp = novelty.fingerprint({**raw, 'params': {}}, hint)
        out['fingerprint'] = fp['values']
        out['textbook_matches'] = [a['id'] for a in novelty.load_archive()
                                   if a['object'] == raw['object'] and novelty.same_family(fp, a['fingerprint'])]
        out['_fp'] = fp
    except Exception as e:
        out['fingerprint_error'] = f'{type(e).__name__}: {e}'
    return out, raw


# ------------------------------------------------------------------ build and audit

def to_json_obj(prob, chosen):
    from .domain import to_json
    return to_json(prob.spec.object, prob.parsed(chosen))


def falls_short(spec, res, ref):
    if not res or not res.get('valid'):
        return True
    if spec.kind != 'optimize':
        return False
    gap = (ref - res['objective']) if spec.objective[1] == 'max' else (res['objective'] - ref)
    return gap >= max(2, 0.05 * abs(ref))


def reaches(spec, res, ref):
    if not res or not res.get('valid'):
        return False
    return spec.kind != 'optimize' or not spec.better(ref, res['objective'])


def build(task, round_dir):
    """Witness, sibling, battery at B and 10B, probes, novelty; returns the audit record."""
    raw, hint, demand = task['spec'], task['hint'], task['demand']
    spec = Spec(raw)
    a = {}
    w = run_program(task['witness_program'], raw, timeout=120, keep=round_dir / 'tasks' / task['id'] / 'witness_run')
    wv = spec.evaluate(w['object']) if w['ok'] else None
    a['witness'] = {'ran': w['ok'], 'valid': bool(wv and wv['valid']), 'objective': wv['objective'] if wv else None,
                    'verdict': wv, 'stderr_tail': w['stderr_tail'][-600:], 'is_construction': task.get('witness_is_construction')}
    if not a['witness']['valid']:
        return a, None
    witness = w['object']
    ref = a['witness']['objective']
    t0 = time.time()
    spec.evaluate(witness)
    per_eval = time.time() - t0
    a['seconds_per_evaluation'] = round(per_eval, 3)
    # cost is object-dependent (an objective over subfamilies is cheap on the witness and explodes on larger families):
    # also time objects drawn the way the search starts, since the battery and probes evaluate tens of thousands
    import random as _rnd
    try:
        prob = search.Problem(spec, hint, max_evals=10 ** 9)
        rng = _rnd.Random(0)
        times = []
        for _ in range(20):
            obj = to_json_obj(prob, search._random_start(prob, rng))
            t1 = time.time(); spec.evaluate(obj); times.append(time.time() - t1)
        a['seconds_per_search_evaluation'] = round(sorted(times)[len(times) // 2], 3)
    except Exception as e:
        a['seconds_per_search_evaluation'] = None
        a['search_timing_error'] = f'{type(e).__name__}: {e}'
    slow = max(per_eval, a.get('seconds_per_search_evaluation') or 0)
    if per_eval > MAX_EVAL_SECONDS or (a.get('seconds_per_search_evaluation') or 0) > MAX_SEARCH_EVAL_SECONDS:
        a['error'] = (f'evaluation too slow: {per_eval:.2f} s on the witness, {a.get("seconds_per_search_evaluation")} s '
                      f'(median) on search objects (limits {MAX_EVAL_SECONDS} s and {MAX_SEARCH_EVAL_SECONDS} s)')
        return a, None
    if raw['object'] == 'set_system':
        import math
        wsizes = sorted({len(x) for x in witness.get('sets', [])})
        m = raw['params'].get('m', 0)
        given = hint.get('set_sizes') or list(range(m + 1))
        if sum(math.comb(m, k) for k in given) > 500000 and wsizes:
            sizes = sorted(set(given) & set(wsizes)) or wsizes     # the search needs a universe: the witness's sizes
            hint = {**hint, 'set_sizes': sizes}
            a['hint_inferred_from_witness'] = sizes
            if sum(math.comb(m, k) for k in sizes) > 500000:      # still too many: a sampled universe (recorded)
                hint = {**hint, 'sample': 200000, 'include': witness.get('sets', [])}
                a['universe_sampled'] = 200000
            task['hint'] = hint
    s = sibling.enumerate_all(sibling.at_size(raw, **task['sibling']), hint, processes=PROCESSES)
    a['sibling'] = {k: v for k, v in s.items() if k != 'optimal_examples'}
    shortcuts, methods = METHODS[demand]
    for budget, cap in ((B, 240), (10 * B, 1200)):
        bat = search.battery(raw, hint, methods=sorted(set(shortcuts + methods + ['greedy_add'])), budget=budget,
                             seeds=(0, 1), processes=PROCESSES, max_seconds=cap)
        a[f'battery_{budget}'] = {m: {k: r.get(k) for k in ('valid', 'objective', 'evals', 'skipped', 'time_capped')}
                                  for m, r in bat['best'].items()}
        a.setdefault('time_capped', [])
        a['time_capped'] += [f"{r['method']}@{budget}" for r in bat['runs'] if r.get('time_capped')]
        best_objs = [r['object'] for r in bat['best'].values() if r.get('object')]
    ps = probes.probe_set(raw, [witness] + best_objs, hint)
    ad = probes.adequacy(raw, ps, processes=PROCESSES)
    a['probes'] = {'count': len(ps), 'adequacy': round(ad['score'], 3), 'survivors': ad['survivors'][:10],
                   'violated_alone': probes.violated_alone(raw, ps)}
    if ad['score'] < 0.9:      # targeted probes for the survivors (equivalent-at-sibling mutants leave the denominator)
        ta = probes.targeted_adequacy(raw, ps, hint, task['sibling'], processes=PROCESSES)
        a['probes'].update({'plain_adequacy': a['probes']['adequacy'], 'targeted': ta, 'adequacy': round(ta['score'], 3)})
    fp = novelty.fingerprint({**raw, 'params': {}}, hint)
    a['novelty'] = {'fingerprint': fp['values'],
                    'textbook_matches': [x['id'] for x in novelty.load_archive()
                                         if x['object'] == raw['object'] and novelty.same_family(fp, x['fingerprint'])],
                    'zoo_match': novelty.zoo_match(raw['object'], witness)}
    # the two-sided audit, at both budgets
    def res(b, m):
        return a[f'battery_{b}'].get(m)
    if demand:
        a['audit'] = {}
        for b in (B, 10 * B):
            sc_fail = all(falls_short(spec, res(b, m), ref) for m in shortcuts)
            mt_ok = (any(reaches(spec, res(b, m), ref) for m in methods) if methods else True)
            a['audit'][str(b)] = {'shortcut_falls_short': sc_fail, 'method_reaches': mt_ok}
        extra = []
        if demand == 'D1' and a['sibling'].get('redundant_constraints'):
            extra.append('redundant constraint at sibling size')
        if demand == 'D2' and not task.get('witness_is_construction'):
            extra.append('witness is not an explicit construction')
        if demand == 'D4' and (a['sibling'].get('valid') or 0) > 0:
            extra.append('claim already fails at the sibling size (not plausible)')
        a['audit']['extra_failures'] = extra
        a['audit']['passes'] = (all(v['shortcut_falls_short'] and v['method_reaches'] for k, v in a['audit'].items()
                                    if k.isdigit()) and not extra)
    else:
        best = [res(b, m) for b in (B, 10 * B) for m in shortcuts]
        a['audit'] = {'battery_falls_short_at_10B': all(falls_short(spec, res(10 * B, m), ref) for m in shortcuts),
                      'battery_falls_short_at_B': all(falls_short(spec, res(B, m), ref) for m in shortcuts)}
        a['audit']['passes'] = a['audit']['battery_falls_short_at_10B']
    return a, witness


# ------------------------------------------------------------------ pilots and adjudication

# Each task runs in its own process (the evaluator's timeouts use SIGALRM, which only works in a main thread).
_compute = FileSlots('compute', COMPUTE_SLOTS)
_program_slots = FileSlots('program', PROGRAM_SLOTS)


def run_pilots(task, round_dir, rounds_object=3, rounds_code=2, code_limit=240):
    raw = task['spec']
    spec = Spec(raw)
    public = prompts.public_package(task['statement'], raw)
    calls = round_dir / 'tasks' / task['id'] / 'calls'
    out = {}
    hist = []
    for i in range(rounds_object):
        r = llm.call(prompts.solve_object_prompt(public, hist), prompts.SOLVE_OBJECT_SCHEMA, calls, f'pilot_object_{i + 1}', effort='xhigh')
        obj = jload(r.get('object_json'))
        v = spec.evaluate(obj) if obj is not None else {'valid': False, 'format_error': 'not JSON'}
        hist.append({'object_json': r.get('object_json', ''), 'verdict': _brief(v), 'object': obj, 'full': v})
        if v.get('valid') and (spec.kind != 'optimize' or not spec.better(_target(task), v['objective'])):
            break
    out['object'] = [{'verdict': h['full'], 'object': h['object']} for h in hist]
    hist = []
    for i in range(rounds_code):
        r = llm.call(prompts.solve_code_prompt(public, hist, code_limit), prompts.SOLVE_CODE_SCHEMA, calls, f'pilot_code_{i + 1}', effort='xhigh')
        with _program_slots:
            run = run_program(r.get('program', ''), raw, timeout=code_limit)
        v = spec.evaluate(run['object']) if run['ok'] else {'valid': False, 'format_error': 'program printed no object',
                                                            'stderr': run['stderr_tail'][-500:], 'timed_out': run['timed_out']}
        hist.append({'verdict': _brief(v), 'object': run['object'] if run['ok'] else None, 'full': v})
        if v.get('valid') and (spec.kind != 'optimize' or not spec.better(_target(task), v['objective'])):
            break
    out['code'] = [{'verdict': h['full'], 'object': h['object']} for h in hist]
    return out


def _int_or_none(x):
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def closed(task):
    """An optimize task whose best known value meets the planner's own proven bound has no headroom left."""
    b, spec = task.get('stated_bound'), task.get('spec') or {}
    if b is None or spec.get('kind') != 'optimize':
        return False
    best = (task.get('best_known') or {}).get('objective', task.get('reference'))
    if best is None:
        return False
    return best >= b if spec['objective']['direction'] == 'max' else best <= b


def _target(task):
    """What pilots aim for: the best known value when the improver ran, else the reference."""
    return (task.get('best_known') or {}).get('objective', task['reference'])


def _brief(v):
    keys = ('valid', 'format_error', 'violations', 'errors', 'objective', 'claim_holds')
    return json.dumps({k: v.get(k) for k in keys if v.get(k) not in (None, [], {})})[:1500]


def adjudicate(task, audit, pilots, blind_raw):
    """Mechanical classes for pilot outcomes, then the task's class."""
    spec = Spec(task['spec'])
    blind = None
    try:
        blind = Spec(blind_raw) if blind_raw else None
    except SpecError:
        pass
    degenerate = []
    for e in task.get('degenerate_exprs') or []:
        try:
            base = {k: v for k, v in task['spec'].items() if k in ('object', 'params', 'given')}
            degenerate.append(Spec({**base, 'kind': 'satisfy', 'constraints': [{'name': 'degenerate', 'expr': e}]}))
        except Exception:
            pass
    outcomes = []
    for arm in ('object', 'code'):
        for att in pilots.get(arm, []):
            v, obj = att['verdict'], att['object']
            if not v.get('valid'):
                cls = 'invalid'
            elif (blind and _blind_rejects(blind, obj)) or any(d.evaluate(obj)['valid'] for d in degenerate):
                cls = 'loophole'
            elif spec.kind == 'optimize' and spec.better(v['objective'], task['reference']):
                cls = 'beats_reference'
            elif spec.kind == 'optimize' and spec.better(task['reference'], v['objective']):
                cls = 'valid_below_reference'
            else:
                cls = 'matches_reference'
            o = {'arm': arm, 'class': cls, 'objective': v.get('objective')}
            if task.get('best_known') and spec.kind == 'optimize' and cls in ('beats_reference', 'matches_reference',
                                                                              'valid_below_reference'):
                bk = task['best_known']['objective']
                o['vs_best'] = ('beats_best' if spec.better(v['objective'], bk) else
                                'below_best' if spec.better(bk, v['objective']) else 'matches_best')
            outcomes.append(o)
    return outcomes


def audit_mode(task, audit):
    """Which way a demand audit went: demand_verified (shortcut short, method reaches), hard_unattributed (every
    battery method short of the reference: hard, but not shown to be hard in the named way), or easy_for_shortcut."""
    au = audit.get('audit') or {}
    if not task.get('demand') or not au:
        return None
    if au.get('passes'):
        return 'demand_verified'
    big = au.get(str(10 * B)) or {}
    if big.get('shortcut_falls_short') and not big.get('method_reaches'):
        return 'hard_unattributed'
    return 'easy_for_shortcut' if not big.get('shortcut_falls_short') else 'extra_failure'


def _blind_rejects(blind, obj):
    """The blind spec rejects the object on its merits (a violation or a holding claim), not by failing to evaluate."""
    r = blind.evaluate(obj)
    return not r['valid'] and not r['errors'] and not r['format_error']


def classify(task, audit, restate, outcomes):
    reasons = []
    if audit.get('error'):
        return 'rejected', [f"build error: {audit['error']}"]
    if not audit.get('witness', {}).get('valid'):
        return 'rejected', ['witness invalid']
    if audit['probes']['adequacy'] < 0.9:
        reasons.append(f"probe adequacy {audit['probes']['adequacy']}")
    if any(c == 0 for c in audit['probes']['violated_alone'].values()) and task['demand'] == 'D1':
        reasons.append('a constraint is never violated alone (possibly implied)')
    if restate.get('disagree_sibling') or restate.get('disagree_probes'):
        reasons.append('blind restatement disagrees (spec or statement defect)')
    flags = task.setdefault('flags', [])
    if restate.get('unusable') and 'blind restatement unusable' not in flags:
        flags.append('blind restatement unusable')   # no evidence either way: a flag, not a rejection
    if any(o['class'] == 'loophole' for o in outcomes):
        reasons.append('a pilot found a loophole')
    if reasons:
        return 'rejected', reasons
    if audit['novelty']['textbook_matches'] or audit['novelty']['zoo_match']:
        return 'calibration', [f"matches {audit['novelty']['textbook_matches'] or audit['novelty']['zoo_match']}"]
    if not audit['audit']['passes']:
        return 'calibration', ['demand audit fails (easy for the named shortcut, or extra failures)']
    if closed(task):
        return 'closed', [f"best known value meets the planner's proven bound {task.get('stated_bound')}"]
    if outcomes and not any(o['class'] != 'invalid' for o in outcomes):
        return 'frontier', ['no pilot found a valid object']
    return 'accepted', []


# ------------------------------------------------------------------ naturalness (soft: measured and reported)

def mechanical_naturalness(raw, statement):
    """Signals of an assembled problem: many constraints, tuned constants, long specs and statements."""
    import ast as _ast
    consts = set()
    exprs = [c['expr'] for c in raw.get('constraints', [])] + [raw[k]['expr'] for k in ('objective', 'claim') if raw.get(k)]
    for e in exprs:
        for node in _ast.walk(_ast.parse(e, mode='eval')):
            if isinstance(node, _ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                if node.value not in (0, 1, 2):
                    consts.add(node.value)
    return {'constraints': len(raw.get('constraints', [])), 'constants_beyond_0_1_2': sorted(consts),
            'spec_chars': sum(len(e) for e in exprs), 'statement_words': len((statement or '').split())}


# ------------------------------------------------------------------ reference improvement (headroom)

IMPROVE = bool(int(os.environ.get('RE_IMPROVE', '0')))
IMPROVE_SECONDS = 600


def improve_reference(task, witness, calls):
    """A strong code agent (more time than the pilots) tries to beat the planner's witness. The best valid object is
    the task's best known value; the witness stays the reference for the battery audit."""
    spec = Spec(task['spec'])
    ref = task['reference']
    best = {'objective': ref, 'source': 'witness'}
    try:
        r = llm.call(prompts.improve_prompt(prompts.public_package(task['statement'], task['spec']), ref,
                                            spec.objective[1], json.dumps(witness), IMPROVE_SECONDS),
                     prompts.SOLVE_CODE_SCHEMA, calls, 'improve', effort='xhigh')
        with _program_slots:
            run = run_program(r.get('program', ''), task['spec'], timeout=IMPROVE_SECONDS + 30)
        v = spec.evaluate(run['object']) if run['ok'] else None
        best['attempt'] = {'ran': run['ok'], 'valid': bool(v and v['valid']), 'objective': v['objective'] if v else None,
                           'timed_out': run['timed_out'], 'note': r.get('note')}
        if v and v['valid'] and spec.better(v['objective'], ref):
            best.update({'objective': v['objective'], 'source': 'improver', 'object': run['object']})
    except Exception as e:
        best['attempt'] = {'error': f'{type(e).__name__}: {e}'}
    return best


# ------------------------------------------------------------------ the round registry (no duplicate families)

ROUND_REGISTRY = bool(int(os.environ.get('RE_ROUND_REGISTRY', '0')))


def round_titles(rd, task_id):
    """Families already taken in this round by other tasks, as avoid-list lines."""
    reg = rd / 'registry.jsonl'
    if not reg.exists():
        return ''
    rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    return '\n'.join(f"- (taken in this round) {r['title']}" for r in rows if r['id'] != task_id)


def register_family(rd, task_id, ranked):
    """Under a lock, take the first ranked candidate whose behavioural fingerprint matches no family another task
    has already taken in this round, and register it. Returns (index or None, [(index, duplicated task ids)])."""
    import fcntl
    reg, lock = rd / 'registry.jsonl', rd / '.registry.lock'
    dups = []
    with open(lock, 'a') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            entries = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()] if reg.exists() else []
            mine = [e for e in entries if e['id'] == task_id]
            if mine:                       # a resumed task keeps the family it registered
                return mine[0]['index'], []
            for i, (_, c, pc) in ranked:
                fp = pc.get('_fp')
                if fp is None:
                    continue
                same = [e['id'] for e in entries if e.get('object') == pc.get('object_type') and e.get('fp')
                        and novelty.same_family(fp, e['fp'])]
                if same:
                    dups.append((i, same))
                    continue
                with open(reg, 'a') as f:
                    f.write(json.dumps({'id': task_id, 'index': i, 'title': c.get('title'), 'object': pc.get('object_type'),
                                        'fp': fp}, default=str) + '\n')
                return i, dups
            return None, dups
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ------------------------------------------------------------------ blind restatement

def _parametric(raw):
    """Does the spec refer to its size parameter (n or m) anywhere?"""
    import ast as _ast
    exprs = [c['expr'] for c in raw.get('constraints', [])] + [raw[k]['expr'] for k in ('objective', 'claim') if raw.get(k)]
    key = 'n' if raw.get('object') == 'graph' else 'm'
    return any(isinstance(x, _ast.Name) and x.id == key for e in exprs for x in _ast.walk(_ast.parse(e, mode='eval')))


def blind_usable(blind, witness):
    """None if the blind spec compiles and evaluates on our witness without errors, else the reason."""
    try:
        r = Spec(blind).evaluate(witness)
    except Exception as e:
        return f'{type(e).__name__}: {e}'
    if r['format_error'] or r['errors']:
        return f"on the reference object: {r['format_error'] or r['errors']}"
    return None


def blind_restatement(task, calls, compare=True):
    """A blind model writes a spec from the statement alone; if its spec cannot be evaluated, it gets one retry with
    the error. Disagreement counts only objects on which both specs evaluate (errors are counted apart)."""
    tries = []
    blind, why = None, None
    for k in (1, 2):
        r = llm.call(prompts.restate_prompt(task['object'], task['statement'], task['spec'].get('params'), error=why),
                     prompts.RESTATE_SCHEMA, calls, 'restate' if k == 1 else 'restate_2', effort='high')
        blind = jload(r.get('spec_json'))
        why = blind_usable(blind, task['witness'])
        tries.append({'ambiguities': r.get('ambiguities'), 'unusable': why})
        if why is None:
            break
    rs = {'ambiguities': tries[-1]['ambiguities'], 'tries': tries}
    if why is not None:
        rs.update({'unusable': why})
        return rs, None
    if not compare:
        return rs, blind
    try:
        with _compute:
            if _parametric(blind):
                s = sibling.enumerate_all(sibling.at_size(task['spec'], **task['sibling']), task['hint'],
                                          other=sibling.at_size(blind, **task['sibling']), processes=PROCESSES)
            else:   # the statement fixes the target size, so a literal blind spec says nothing about other sizes
                s = {'disagree': None, 'skipped_reason': 'blind spec is not parametric in the size'}
            ps = probes.probe_set(task['spec'], [task['witness']], task['hint'])
            cmp_ = probes.compare(task['spec'], blind, ps)
        rs.update({'disagree_sibling': s.get('disagree'), 'sibling_examples': s.get('disagree_examples'),
                   'sibling_skipped': s.get('skipped_reason'),
                   'sibling_blind_errors': s.get('other_errors'), 'disagree_probes': cmp_['disagree'],
                   'probe_blind_errors': cmp_['b_errors'], 'probe_examples': cmp_['examples']})
    except Exception as e:
        rs.update({'unusable': f'comparison failed: {type(e).__name__}: {e}'})
        blind = None
    return rs, blind


# ------------------------------------------------------------------ one task, end to end

def _key(*parts):
    import hashlib
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def run_task(req, rd, arch):
    """ideas -> prechecks -> select -> plan -> build (cached) -> restatement -> naturalness -> pilots -> class.
    Every model call is resumable; the build is cached by the hash of what it depends on; the final dossier is
    written last, so a task with a final dossier is never redone."""
    rq = rd / 'requests' / req['id']
    tdir = rd / 'tasks' / req['id']
    tdir.mkdir(parents=True, exist_ok=True)
    final = tdir / 'dossier.json'
    if final.exists():
        d = json.loads(final.read_text())
        if d.get('final'):
            return d
    task = {'id': req['id'], 'request': req, 'demand': req['demand'], 'object': req['object']}
    pick, checked, dups_all = None, [], []
    for attempt in (1, 2):           # a second idea call only if every candidate duplicates a family in this round
        sfx = '' if attempt == 1 else '_2'
        avoid = arch if attempt == 1 else arch + '\n' + round_titles(rd, req['id'])
        ideas_out = llm.call(prompts.ideas_prompt(req['object'], req['demand'], avoid), prompts.IDEAS_SCHEMA, rq,
                             'ideas' + sfx, effort=IDEA_EFFORT, timeout=3600)
        checked = []
        with _compute:
            for i, c in enumerate(ideas_out.get('candidates', [])):
                pc, _ = precheck(c)
                checked.append((i, c, pc))
        _write(rq / f'prechecks{sfx}.json', [{'index': i, 'title': c.get('title'),
                                              'precheck': {k: v for k, v in pc.items() if k != '_fp'}} for i, c, pc in checked])
        log(rd, f"{req['id']}: {len(checked)} candidates{sfx}; compile {sum(pc['compiles'] for _, _, pc in checked)}; "
                f"textbook matches {[pc.get('textbook_matches') for _, _, pc in checked]}")
        listing = '\n\n'.join(
            f"### Candidate {i}: {c.get('title')}\nStatement: {c.get('statement')}\nSpec: {c.get('spec_json')}\n"
            f"Demand rationale: {c.get('demand_rationale')}\nNovelty note: {c.get('novelty_note')}\n"
            f"Checks: {json.dumps({k: v for k, v in pc.items() if k not in ('_fp', 'fingerprint')}, default=str)}"
            for i, c, pc in checked)
        sel = llm.call(prompts.select_prompt(req['object'], req['demand'], listing), prompts.SELECT_SCHEMA, rq,
                       'select' + sfx, effort='high')
        task['select_reason'] = sel.get('reason')
        order = [sel.get('chosen_index', -1)] + [i for i in (sel.get('ranking') or []) if i != sel.get('chosen_index')]
        order = [i for i in order if any(x[0] == i for x in checked)]
        if not order:
            task.update({'class': 'rejected', 'reasons': ['selector chose none'], 'final': True})
            _write(final, task)
            return task
        if not ROUND_REGISTRY:
            pick = order[0]
            break
        pick, dups = register_family(rd, req['id'], [(i, next(x for x in checked if x[0] == i)) for i in order])
        dups_all += dups
        if pick is not None:
            break
    task['round_duplicates_avoided'] = dups_all
    if pick is None:
        task.update({'class': 'rejected', 'reasons': [f'every candidate duplicates a family already in this round: {dups_all}'],
                     'final': True})
        _write(final, task)
        return task
    _, cand, pc = next(x for x in checked if x[0] == pick)
    checks = json.dumps({k: v for k, v in pc.items() if k not in ('_fp',)}, indent=1, default=str)
    plan = llm.call(prompts.plan_prompt(req['object'], req['demand'], json.dumps(cand, indent=1), checks),
                    prompts.PLAN_SCHEMA, rq, 'plan', effort='xhigh')
    task.update({'title': cand.get('title'), 'statement': plan.get('statement'), 'spec': jload(plan.get('spec_json')),
                 'hint': jload(plan.get('search_hint_json'), {}) or {}, 'sibling': jload(plan.get('sibling_params_json'), {}) or {},
                 'witness_program': plan.get('witness_program'), 'witness_is_construction': plan.get('witness_is_construction'),
                 'degenerate_exprs': plan.get('degenerate_exprs'), 'candidate': cand, 'plan_notes': plan.get('notes'),
                 'stated_bound': _int_or_none(plan.get('upper_bound')), 'bound_argument': plan.get('bound_argument')})
    try:
        Spec(task['spec'])
    except Exception as e:
        task.update({'class': 'rejected', 'reasons': [f'final spec does not compile: {e}'], 'final': True})
        _write(final, task)
        return task
    # build, cached
    bkey = _key(task['spec'], task['hint'], task['sibling'], task['witness_program'], BUILD_VERSION)
    cache = tdir / 'build.json'
    cached = json.loads(cache.read_text()) if cache.exists() else None
    if cached and cached.get('key') == bkey:
        audit, witness, task['hint'] = cached['audit'], cached['witness'], cached['hint']
    else:
        t0 = time.time()
        with _compute:
            try:
                audit, witness = build(task, rd)
            except Exception as e:
                audit, witness = {'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-2000:]}, None
        audit['seconds'] = round(time.time() - t0)
        _write(cache, {'key': bkey, 'audit': audit, 'witness': witness, 'hint': task['hint']})
    task['audit'], task['witness'] = audit, witness
    task['reference'] = audit.get('witness', {}).get('objective')
    log(rd, f"{task['id']}: built in {audit.get('seconds')}s; witness valid {audit.get('witness', {}).get('valid')}; "
            f"audit {audit.get('audit', {}).get('passes')}{'; ERROR ' + audit['error'] if audit.get('error') else ''}")
    task['naturalness'] = {'mechanical': mechanical_naturalness(task['spec'], task['statement'])}
    if witness and IMPROVE and Spec(task['spec']).kind == 'optimize':
        task['best_known'] = improve_reference(task, witness, tdir / 'calls')
    if not witness:
        task['class'], task['reasons'] = classify(task, audit, {}, [])
        task['final'] = True
        _write(final, task)
        return task
    calls = tdir / 'calls'
    rs, blind = blind_restatement(task, calls)
    task['restate'], task['blind_spec'] = rs, blind
    # naturalness, rated blind from the statement alone
    try:
        task['naturalness']['rater'] = llm.call(prompts.naturalness_prompt(task['statement']), prompts.NATURALNESS_SCHEMA,
                                                calls, 'naturalness', effort='high')
    except llm.LLMError as e:
        task['naturalness']['rater'] = {'error': str(e)}
    _write(final, task)  # provisional (not final): a restart resumes at the pilots
    task['pilots'] = run_pilots(task, rd)
    task['outcomes'] = adjudicate(task, audit, task['pilots'], blind)
    task['class'], task['reasons'] = classify(task, audit, rs, task['outcomes'])
    task['audit_mode'] = audit_mode(task, audit)
    task['code'] = code_hash()
    task['final'] = True
    _write(final, task)
    log(rd, f"{task['id']}: class {task['class']} {task['reasons']}; pilots {[o['class'] for o in task['outcomes']]}; "
            f"naturalness {task['naturalness'].get('rater', {}).get('score')}")
    return task


IDEA_EFFORT = 'high'
MAX_EVAL_SECONDS = 1.0
MAX_SEARCH_EVAL_SECONDS = 0.25
BUILD_VERSION = 4             # bump when build() changes, so cached builds are redone


def summarize(rd, requests):
    rows = []
    for req in requests:
        f = rd / 'tasks' / req['id'] / 'dossier.json'
        if not f.exists():
            rows.append({'id': req['id'], 'demand': req['demand'], 'object': req['object'], 'class': 'pending'})
            continue
        t = json.loads(f.read_text())
        rows.append({'id': t['id'], 'demand': t['demand'], 'object': t['object'], 'title': t.get('title'),
                     'class': t.get('class') if t.get('final') else 'pending', 'reasons': t.get('reasons'),
                     'reference': t.get('reference'), 'pilots': [o['class'] for o in t.get('outcomes', [])],
                     'naturalness': (t.get('naturalness') or {}).get('rater', {}).get('score'),
                     'constraints': (t.get('naturalness') or {}).get('mechanical', {}).get('constraints'),
                     'audit_passes': (t.get('audit') or {}).get('audit', {}).get('passes'),
                     'audit_mode': t.get('audit_mode')})
    return rows


def code_hash():
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob('*.py')):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _task_main(req, rd, arch):
    """Entry point of a task's own process."""
    try:
        run_task(req, rd, arch)
    except BaseException as e:
        (rd / 'tasks' / req['id']).mkdir(parents=True, exist_ok=True)
        (rd / 'tasks' / req['id'] / 'failure.txt').write_text(traceback.format_exc())
        log(rd, f"{req['id']}: FAILED {type(e).__name__}: {e}")
        raise SystemExit(1)


def run_round(name, requests, workers=12, avoid=()):
    """Run every request in its own process, at most `workers` at once. `avoid`: extra titles for the ideas prompt's
    list of known problems (for example earlier rounds' tasks)."""
    import multiprocessing as mp
    rd = ROOT / 'runs' / name
    (rd / 'tasks').mkdir(parents=True, exist_ok=True)
    (rd / 'requests.json').write_text(json.dumps(requests, indent=1))
    log(rd, f'round {name}: {len(requests)} requests, {workers} workers, budget B={B}')
    arch = archive_lines(avoid)
    ctx = mp.get_context('spawn')

    def one(req):
        fail = rd / 'tasks' / req['id'] / 'failure.txt'
        if fail.exists():
            fail.rename(fail.with_name(f'failure-{int(time.time())}.txt'))
        proc = ctx.Process(target=_task_main, args=(req, rd, arch), name=f"task-{req['id']}")
        proc.start()
        proc.join()
        try:
            _write(rd / 'progress.json', summarize(rd, requests))
        except Exception:
            pass
        if proc.exitcode != 0:
            if not fail.exists():
                log(rd, f"{req['id']}: process exited {proc.exitcode}")
            return False
        return True

    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(one, requests))
    rows = summarize(rd, requests)
    failed = [r['id'] for r, k in zip(requests, ok) if not k]
    if failed:
        log(rd, f'round incomplete: {len(failed)} tasks failed ({failed}); rerun to resume')
        return rows
    _write(rd / 'summary.json', rows)
    log(rd, 'round done')
    return rows


def _safe(f):
    def g(x):
        try:
            return f(x)
        except Exception as e:
            return e
    return g
