"""Task files: export a run's dossier to a self-contained task file, load task files, and validate them offline.

A task file is the published form of one generated task (the files under examples/ are task files):
- discover (domain A): {"format": "research-envs/discover-world/1", "public": {description, interface, queries},
  "private": {world, named_rivals, plan_experiments, queries with answers}, "results": ...};
- construct (domain B): {"format": "research-envs/construct-problem/1", "public": {statement, spec},
  "private": {sibling, search_hint, witness_program, witness, reference, best_known, ...}, "results": ...}.

"results" are what the pipeline recorded in the source run (class, pilot scores, audit summary). Validation re-derives
everything it can from the definition alone, with no model calls.

Domain A's public interface depends on RE_A_V2 (the published runs used RE_A_V2=1: settable values keep their world
names). Set it before importing this module; the command line does.
"""
import json
import time
from contextlib import contextmanager
from pathlib import Path

DISCOVER = 'research-envs/discover-world/1'
CONSTRUCT = 'research-envs/construct-problem/1'


# ------------------------------------------------------------------ loading

def find(ref):
    """A task file path from a path or an example id (examples/discover/<id>.json, examples/construct/<id>.json)."""
    from .common import EXAMPLES_DIR
    p = Path(ref)
    if p.is_file():
        return p
    if EXAMPLES_DIR:
        for sub in ('discover', 'construct'):
            q = EXAMPLES_DIR / sub / f'{ref}.json'
            if q.is_file():
                return q
    raise FileNotFoundError(f'no task file or example named {ref!r}' + ('' if EXAMPLES_DIR else
                            ' (example ids work from a source checkout; give the path to the task file)'))


def load(ref):
    """A task dict from a task file, an example id, or a run's dossier.json (exported on the fly)."""
    p = find(ref)
    d = json.loads(p.read_text())
    if d.get('format') in (DISCOVER, CONSTRUCT):
        return d
    if 'world' in d and 'queries' in d:
        return export_discover(d)
    if 'spec' in d and 'statement' in d:
        return export_construct(d)
    raise ValueError(f'{p}: neither a task file nor a dossier')


def domain(task):
    return 'discover' if task.get('format') == DISCOVER else 'construct'


# ------------------------------------------------------------------ export from a dossier

def _settings_a():
    from .discover import audit as Au, pipeline as Pp, prompts as P
    return {'budget': Pp.BUDGET, 'interface': 'v2' if P.V2 else 'v1', 'query_mode': Au.QUERY_MODE or 'critical+fill',
            'answer_cap': Au.ANSWER_CAP, 'trivial_cap': Au.TRIVIAL_CAP, 'quality_bar': P.QUALITY}


def export_discover(t, source_run=None, settings=None):
    from .discover.world import World
    from .discover.harness import PublicMap
    from .discover import prompts as P
    w = World(t['world'])
    pm = PublicMap(w)
    budget = (((t.get('pilots') or {}).get('full') or {}).get('budget')) or (settings or {}).get('budget') or 40
    pub = P.public_interface(w, t.get('description') or '', budget)
    queries, public_q = [], []
    for i, q in enumerate(t.get('queries') or []):
        qid = f'Q{i + 1}'
        public_q.append({'id': qid, **pm.public_query(q)})
        queries.append({'id': qid, 'regime': q.get('regime'), 'start': q['start'], 'inputs': q['inputs'],
                        'answer': list(q['answer']), 'critical': bool(q.get('disagree'))})
    a = t.get('audit') or {}
    ident = a.get('identifiability') or {}
    au = a.get('audit') or {}
    pilots = t.get('pilots') or {}
    return {
        'format': DISCOVER, 'id': t['id'], 'title': t.get('title'),
        'request': {k: t.get(k) for k in ('demand', 'lever')}, 'source_run': source_run,
        'settings': {**_settings_a(), **(settings or {}), 'budget': budget},
        'public': {'description': t.get('description'), 'interface': {k: v for k, v in pub.items() if k != 'description'},
                   'queries': public_q},
        'private': {'story': t.get('story'), 'world': t['world'], 'named_rivals': t.get('rivals') or [],
                    'plan_experiments': t.get('plan_experiments') or [], 'queries': queries},
        'results': {
            'class': t.get('class'), 'reasons': t.get('reasons'),
            'pilot_critical_accuracy': {k: v.get('critical_accuracy') for k, v in pilots.items()},
            'pilot_experiments_used': {k: v.get('experiments_used') for k, v in pilots.items()},
            'queries_dropped': t.get('queries_dropped'), 'silent_fraction': t.get('silent_fraction'),
            'lever_profile': t.get('lever_profile'), 'novelty': t.get('novelty'),
            'audit': {'states': a.get('states'), 'inputs': a.get('inputs'), 'starts': a.get('starts'),
                      'rivals': a.get('rivals'), 'interface_equivalent_rivals': len(a.get('interface_equivalent') or []),
                      'needles': len(a.get('needles') or []), 'rivals_left_after_identifiability': len(ident.get('rivals_left') or []),
                      'coverage': a.get('coverage'), 'queries': a.get('queries'),
                      'plan_reaches': au.get('plan_reaches'), 'shortcut_falls_short': au.get('shortcut_falls_short')},
        },
    }


def export_construct(t, source_run=None, settings=None):
    from .construct import prompts as Pb, pipeline as Pp
    spec = t['spec']
    a = t.get('audit') or {}
    sib = a.get('sibling') or {}
    bk = t.get('best_known')
    return {
        'format': CONSTRUCT, 'id': t['id'], 'title': t.get('title'), 'object': t.get('object') or spec.get('object'),
        'request': {'demand': t.get('demand')}, 'source_run': source_run,
        'settings': {'battery_budget': Pp.B, 'quality_bar': Pb.QUALITY_B, 'round_registry': Pp.ROUND_REGISTRY,
                     'improver': Pp.IMPROVE, **(settings or {})},
        'public': {'statement': t.get('statement'), 'spec': spec},
        'private': {'sibling': t.get('sibling'), 'search_hint': t.get('hint'), 'witness_program': t.get('witness_program'),
                    'witness_is_construction': t.get('witness_is_construction'), 'witness': t.get('witness'),
                    'reference': t.get('reference'), 'best_known': bk, 'stated_bound': t.get('stated_bound'),
                    'bound_argument': t.get('bound_argument'), 'degenerate_exprs': t.get('degenerate_exprs') or []},
        'results': {
            'class': t.get('class'), 'reasons': t.get('reasons'), 'audit_mode': t.get('audit_mode'),
            'audit_passes': (a.get('audit') or {}).get('passes'),
            'pilot_outcomes': t.get('outcomes'),
            'naturalness_rater_score': ((t.get('naturalness') or {}).get('rater') or {}).get('score'),
            'sibling': {k: sib.get(k) for k in ('objects', 'valid', 'optimum', 'optimal_classes', 'redundant_constraints')},
            'probe_adequacy': (a.get('probes') or {}).get('adequacy'),
            'novelty': {k: (a.get('novelty') or {}).get(k) for k in ('textbook_matches', 'zoo_match')},
            'battery_best': {b: {m: r.get('objective') if r.get('valid') else None for m, r in (a.get(b) or {}).items()}
                             for b in a if b.startswith('battery_')},
        },
    }


# ------------------------------------------------------------------ validation (offline)

class _Report:
    def __init__(self, task):
        self.task, self.checks, self.t0 = task, [], time.time()

    def add(self, name, ok, detail=''):
        """ok: True (pass), False (fail) or None (skipped)."""
        self.checks.append({'check': name, 'ok': ok, 'detail': detail})

    def run(self, name, f):
        try:
            ok, detail = f()
        except Exception as e:
            ok, detail = False, f'{type(e).__name__}: {e}'
        self.add(name, ok, detail)

    def result(self):
        return {'id': self.task.get('id'), 'domain': domain(self.task), 'ok': all(c['ok'] is not False for c in self.checks),
                'seconds': round(time.time() - self.t0, 1), 'checks': self.checks}


@contextmanager
def _query_rules(task):
    """Apply a world's own recorded query rules (module settings read when rivals and queries are built), then
    restore the process-wide ones."""
    from .discover import audit as Au, rivals as Rv
    st = task.get('settings') or {}
    saved = (Au.QUERY_MODE, Au.ANSWER_CAP, Au.TRIVIAL_CAP, Rv.RANGE_RIVALS)
    mode = st.get('query_mode', Au.QUERY_MODE)
    Au.QUERY_MODE = None if mode == 'critical+fill' else mode
    Au.ANSWER_CAP, Au.TRIVIAL_CAP = st.get('answer_cap', Au.ANSWER_CAP), st.get('trivial_cap', Au.TRIVIAL_CAP)
    Rv.RANGE_RIVALS = Au.QUERY_MODE == 'discriminating'
    try:
        yield
    finally:
        Au.QUERY_MODE, Au.ANSWER_CAP, Au.TRIVIAL_CAP, Rv.RANGE_RIVALS = saved


def validate_discover(task, full=False):
    """World compiles; the public interface and queries are what the world implies; every stored answer is the
    world's answer; critical queries are refuted by a named or structural rival; no archive match. full=True also
    reruns the whole build and audit (rivals, identifiability, baselines; minutes to tens of minutes)."""
    from .discover.world import World
    from .discover.harness import PublicMap
    from .discover import prompts as P, novelty as Nv, rivals as Rv, levers as Lv
    from .discover.fairness import answer
    from .discover.audit import _demand_rival
    rep = _Report(task)
    pub, prv = task['public'], task['private']
    try:
        w = World(prv['world'])
    except Exception as e:
        rep.add('world compiles', False, f'{type(e).__name__}: {e}')
        return rep.result()
    rep.add('world compiles', True, f'{w.n_states} states, {len(w.inputs)} input symbols, max_len {w.max_len}')
    pm = PublicMap(w)
    budget = task.get('settings', {}).get('budget', 40)

    def interface():
        got = {k: v for k, v in P.public_interface(w, pub.get('description') or '', budget).items() if k != 'description'}
        return got == pub['interface'], '' if got == pub['interface'] else f'expected {pub["interface"]}, got {got}'
    rep.run('public interface matches the world', interface)

    def queries():
        bad = []
        if len(pub['queries']) != len(prv['queries']):
            return False, 'public and private query lists differ in length'
        for qp, q in zip(pub['queries'], prv['queries']):
            st, seq = pm.to_world(qp)
            if (st, seq) != ({**{v: w.init[v] for v in w.settable}, **q['start']}, q['inputs']):
                bad.append(f"{q['id']}: public form does not map to the private query")
            elif list(answer(w, q)) != list(q['answer']):
                bad.append(f"{q['id']}: stored answer {q['answer']} != world's {list(answer(w, q))}")
        return not bad, f"{len(prv['queries'])} queries, {sum(q['critical'] for q in prv['queries'])} critical" + (
            '; ' + '; '.join(bad[:5]) if bad else '')
    rep.run('queries and answers reproduce', queries)

    def rivals():
        with _query_rules(task):
            _, R = Rv.rival_set(prv['world'], prv.get('named_rivals') or [])
        demand = [rw for label, rw in R if _demand_rival(label)]
        missing = [q['id'] for q in prv['queries'] if q['critical']
                   and not any(list(answer(rw, q)) != list(q['answer']) for rw in demand)]
        return not missing, (f'{len(R)} rivals ({len(demand)} named or structural); every critical query refutes one'
                             if not missing else f'critical queries no named or structural rival gets wrong: {missing}')
    rep.run('critical queries refute a rival', rivals)

    def novelty():
        m = Nv.matches(prv['world'])
        return not m['behaviour'], f"archive behaviour matches {m['behaviour']}, structure matches {m['structure']}"
    rep.run('not a textbook mechanism (archive)', novelty)
    lever = (task.get('request') or {}).get('lever')
    if lever:   # a diagnostic in the pipeline, never a gate (and only a rough indication for 'coupled')
        prof = Lv.profile(prv['world'])
        rep.add(f'pinned property detected ({lever})', True if prof.get(lever) else None,
                json.dumps(prof) + ('' if prof.get(lever) else '; not detected by the mechanical lever check, which is '
                                    'a diagnostic, not a gate'))
    if full:
        def build():
            from .discover import audit as Au, pipeline as Pp
            with _query_rules(task):
                _, _, qs, _, a = Au.build(prv['world'], (task.get('request') or {}).get('demand'),
                                          Pp.parse_plan(w, prv.get('plan_experiments') or []),
                                          named=prv.get('named_rivals') or [], budget=budget)
            ident = a.get('identifiability', {})
            left = set(ident.get('rivals_left_idx', [])) | set(ident.get('needle_idx', []))
            fair = [q for q in qs if not (set(q.get('any_rival', [])) & left)]
            same = [(q['start'], q['inputs']) for q in fair] == [(q['start'], q['inputs']) for q in prv['queries']]
            return a.get('error') is None, (f"rebuilt: {len(fair)} fair queries ({'identical to' if same else 'different from'} "
                                            f"the stored set), needles {len(a.get('needles', []))}, rivals left "
                                            f"{len(ident.get('rivals_left', []))}, plan reaches {a['audit'].get('plan_reaches')}")
        rep.run('full rebuild and audit', build)
    return rep.result()


def validate_construct(task, sandbox=True, full=False):
    """Spec compiles; the stored witness (and best known object) is valid with the stated value; the witness program
    reproduces it in the sandbox; the sibling instance enumerates to the recorded results; no textbook or zoo match.
    full=True also reruns the build (battery at B and 10B, probes; minutes to hours)."""
    from .construct.evaluate import Spec
    from .construct import sibling, novelty
    rep = _Report(task)
    pub, prv, res = task['public'], task['private'], task.get('results') or {}
    raw = pub['spec']
    try:
        spec = Spec(raw)
    except Exception as e:
        rep.add('spec compiles', False, f'{type(e).__name__}: {e}')
        return rep.result()
    rep.add('spec compiles', True, f'{spec.object}, {spec.kind}, params {spec.params}')

    def same_value(v, want):
        return spec.kind != 'optimize' or v.get('objective') == want

    def witness():
        v = spec.evaluate(prv['witness'])
        ok = bool(v['valid']) and same_value(v, prv.get('reference'))
        return ok, f"valid {v['valid']}, objective {v.get('objective')}, reference {prv.get('reference')}"
    rep.run('stored witness is valid with the reference value', witness)
    bk = prv.get('best_known') or {}
    if bk.get('object') is not None:
        def best():
            v = spec.evaluate(bk['object'])
            return bool(v['valid']) and same_value(v, bk.get('objective')), f"valid {v['valid']}, objective {v.get('objective')}"
        rep.run('best known object is valid with its value', best)
    if sandbox:
        from .construct.sandbox import run_program, landlock_abi, MIN_ABI
        if landlock_abi() < MIN_ABI:
            rep.add('witness program reproduces the witness (sandboxed)', None,
                    f'skipped: Landlock ABI {landlock_abi()} < {MIN_ABI}; the program is never run unsandboxed')
        else:
            def program():
                r = run_program(prv['witness_program'], raw, timeout=120)
                if not r['ok']:
                    return False, f"program failed: timed out {r['timed_out']}, stderr {r['stderr_tail'][-300:]}"
                v = spec.evaluate(r['object'])
                return bool(v['valid']) and same_value(v, prv.get('reference')), f"valid {v['valid']}, objective {v.get('objective')}"
            rep.run('witness program reproduces the witness (sandboxed)', program)
    else:
        rep.add('witness program reproduces the witness (sandboxed)', None, 'skipped (--no-sandbox)')

    def sib():
        s = sibling.enumerate_all(sibling.at_size(raw, **(prv.get('sibling') or {})), prv.get('search_hint'))
        if s.get('skipped'):
            return None, s['skipped']
        want = res.get('sibling') or {}
        diff = {k: (want[k], s.get(k)) for k in ('objects', 'valid', 'optimum', 'redundant_constraints')
                if want.get('objects') is not None and k in want and want[k] != s.get(k)}
        return not diff, (f"{s['objects']} objects, {s['valid']} valid, optimum {s['optimum']}, redundant "
                          f"{s['redundant_constraints']}" + (f'; differs from the record: {diff}' if diff else ''))
    rep.run('sibling instance enumerates to the recorded results', sib)

    def nov():
        fp = novelty.fingerprint({**raw, 'params': {}}, prv.get('search_hint'))
        tb = [a['id'] for a in novelty.load_archive() if a['object'] == raw['object'] and novelty.same_family(fp, a['fingerprint'])]
        zoo = novelty.zoo_match(raw['object'], prv['witness'])
        return not tb and not zoo, f'textbook matches {tb}, zoo match {zoo}'
    rep.run('not a textbook problem or zoo object', nov)
    if full:
        def build():
            from .construct import pipeline as Pp
            t = {'id': task['id'], 'spec': raw, 'hint': dict(prv.get('search_hint') or {}), 'sibling': prv.get('sibling') or {},
                 'demand': (task.get('request') or {}).get('demand'), 'witness_program': prv['witness_program'],
                 'witness_is_construction': prv.get('witness_is_construction')}
            from .common import ROOT
            a, _ = Pp.build(t, ROOT / 'runs' / 'validate')
            return a.get('error') is None and (a.get('witness') or {}).get('valid'), json.dumps(
                {'audit': a.get('audit'), 'probe_adequacy': (a.get('probes') or {}).get('adequacy')}, default=str)
        rep.run('full rebuild and audit', build)
    return rep.result()


def validate(task, **kw):
    if domain(task) == 'discover':
        return validate_discover(task, full=kw.get('full', False))
    return validate_construct(task, sandbox=kw.get('sandbox', True), full=kw.get('full', False))


# ------------------------------------------------------------------ pipeline forms for pilots

def discover_queries(task):
    """The private queries in the pipeline's form (critical queries carry a non-empty 'disagree')."""
    return [{'regime': q.get('regime'), 'start': q['start'], 'inputs': q['inputs'], 'answer': q['answer'],
             'disagree': [0] if q['critical'] else []} for q in task['private']['queries']]


def construct_task(task):
    """The pipeline's task dict for run_pilots and adjudicate."""
    prv = task['private']
    return {'id': task['id'], 'object': task.get('object'), 'demand': (task.get('request') or {}).get('demand'),
            'statement': task['public']['statement'], 'spec': task['public']['spec'], 'reference': prv.get('reference'),
            'best_known': prv.get('best_known'), 'degenerate_exprs': prv.get('degenerate_exprs') or [],
            'witness': prv.get('witness'), 'sibling': prv.get('sibling') or {}, 'hint': prv.get('search_hint') or {}}
