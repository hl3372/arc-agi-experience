"""Domain A pipeline: request -> ideas -> prechecks -> select -> plan -> build and audit -> pilots at three budgets ->
mechanical class. Reuses domain B's runner discipline (construct/pipeline.py): one process per task, machine-wide
compute slots, resumable model calls, cached builds, a final dossier written last, and no summary while any task
has failed."""
import json
import os
import time
import traceback
from pathlib import Path
from ..common import FileSlots, ROOT
from ..construct import llm
from ..construct.pipeline import log, _write, jload
from . import prompts as P, audit as Au, novelty as Nv, harness as H, levers as Lv
from .world import World

BUDGET = int(os.environ.get('RE_A_BUDGET', 40))
# A's builds are single-process: their own slots, not B's 4-process ones (RE_A_COMPUTE_SLOTS overrides the default 3)
_compute = FileSlots('compute-a', int(os.environ.get('RE_A_COMPUTE_SLOTS', 3)))
IDEA_EFFORT, PLAN_EFFORT, PILOT_EFFORT = 'high', 'xhigh', 'high'
BUILD_VERSION = 3
MIN_QUERIES, MIN_CRITICAL = 10, 3


def parse_plan(world, plan):
    """The planner's experiments in public-free world form: [(settings, input indices)]; invalid ones dropped."""
    out = []
    for e in plan if isinstance(plan, list) else []:
        try:
            seq = [world.input_index(x[0], x[1] if len(x) > 1 else None) for x in e.get('inputs', [])]
            st = {k: v for k, v in (e.get('start') or {}).items() if k in world.settable}
            world.start_state(st)
            if len(seq) <= world.max_len:
                out.append((st, seq))
        except Exception:
            continue
    return out


def named_rivals(rivals):
    out = []
    for r in rivals or []:
        w = jload(r.get('world_json'))
        if isinstance(w, dict):
            out.append({'label': r.get('label', 'rival'), 'world': w})
    return out


def precheck(cand, arch, lever=None):
    out = {'compiles': False}
    spec = jload(cand.get('world_json'))
    try:
        w = World(spec)
        out['compiles'] = True
    except Exception as e:
        out['error'] = f'world does not compile: {type(e).__name__}: {e}'
        return out
    out.update({'states': w.n_states, 'actions': len(w.inputs), 'starts': len(w.starts()), 'max_len': w.max_len})
    # True would be bad: a full sweep of starts and sequences would fit in 10x the budget
    out['trivially_sweepable'] = len(w.starts()) * sum(len(w.inputs) ** k for k in range(1, w.max_len + 1)) <= 10 * BUDGET
    m = Nv.matches(spec, arch)
    out['archive_behaviour_matches'], out['archive_structure_matches'] = m['behaviour'], m['structure']
    out['silent_fraction'] = silent_fraction(w)
    out['levers'] = Lv.profile(spec)
    if lever:
        out['pinned_lever_present'] = out['levers'].get(lever)
    out['_fp'], out['_struct'] = Nv.fingerprint(w)['digest'], Nv.structure(spec)
    return out


SILENT_MAX = 0.6
N_CANDIDATES = int(os.environ.get('RE_A_CANDIDATES', 3))


def silent_fraction(w, n=400):
    """Share of public random experiments whose final readouts equal the start readouts (nothing visible changed)."""
    import random as _r
    from .fairness import sample_experiment
    rng = _r.Random(0)
    same = 0
    for _ in range(n):
        st, seq = sample_experiment(w, rng)
        s0 = w.start_state(st)
        same += w.run_inputs(s0, seq)[-1] == w.out_tuple[s0]
    return round(same / n, 3)


def classify(task):
    """Revision 3 classes: gates first (sound, fair, new, not trivial); then the primary outcome, frontier difficulty
    (the full-budget pilot on critical queries). Demand audits and baselines are diagnostics, recorded as flags."""
    a = task.get('audit') or {}
    if a.get('error'):
        return 'rejected', [f"build error: {a['error']}"]
    au = a.get('audit') or {}
    qs = task.get('queries') or []
    n_crit = sum(bool(q.get('disagree')) for q in qs)
    if len(qs) < MIN_QUERIES or n_crit < MIN_CRITICAL:
        return 'rejected', [f'too few fair queries after dropping unfair ones ({len(qs)} published, {n_crit} critical)']
    if task['novelty']['behaviour']:
        return 'calibration', [f"behaves like archive {task['novelty']['behaviour']}"]
    if a.get('coverage'):
        return 'calibration', ['coverage: a full sweep fits in 10x budget']
    if P.QUALITY and (task.get('silent_fraction') or 0) > SILENT_MAX:
        return 'rejected', [f"silent readouts: {task['silent_fraction']:.0%} of random experiments end unchanged"]
    task['flags'] = {'public_discoverable': bool(au.get('plan_reaches')),
                     'shortcut_falls_short': bool(au.get('shortcut_falls_short')),
                     'demand_audit_passes': au.get('passes'),
                     'few_critical': (au.get('critical_queries') or 0) < Au.GAP_MIN}
    full = (task.get('pilots') or {}).get('full') or {}
    acc = full.get('critical_accuracy')
    if acc is None:
        return 'rejected', ['the full-budget pilot did not complete']
    if acc < Au.CRIT_OK:
        return 'frontier', [f'full-budget pilot critical accuracy {acc:.2f}']
    return 'solved', [f'full-budget pilot critical accuracy {acc:.2f}']


def run_task(req, rd, arch, arch_text):
    rq, tdir = rd / 'requests' / req['id'], rd / 'tasks' / req['id']
    tdir.mkdir(parents=True, exist_ok=True)
    final = tdir / 'dossier.json'
    if final.exists() and json.loads(final.read_text()).get('final'):
        return json.loads(final.read_text())
    lever = req.get('lever')
    naming = P.NAMING_V2 if P.V2 else None
    avoid = arch_text + ('\n' + round_titles(rd, req['id']) if REGISTRY else '')
    ideas = llm.call(P.ideas_prompt(req['demand'], avoid, BUDGET, lever=lever, naming=naming, count=N_CANDIDATES), P.IDEAS_SCHEMA, rq, 'ideas',
                     effort=IDEA_EFFORT, timeout=3600)
    checked = [(i, c, precheck(c, arch, lever)) for i, c in enumerate(ideas.get('candidates', []))]
    _write(rq / 'prechecks.json', [{'index': i, 'title': c.get('title'),
                                    'precheck': {k: v for k, v in pc.items() if not k.startswith('_')}} for i, c, pc in checked])
    log(rd, f"{req['id']}: {len(checked)} candidates; compile {sum(pc['compiles'] for _, _, pc in checked)}; "
            f"archive {[pc.get('archive_behaviour_matches') for _, _, pc in checked]}")
    import random as _r
    shown = list(checked)
    _r.Random(req['id']).shuffle(shown)       # shuffled order, against position bias; indices keep their labels
    listing = '\n\n'.join(f"### Candidate {i}: {c.get('title')}\nStory: {c.get('story')}\nWorld: {c.get('world_json')}\n"
                          f"Rationale: {c.get('demand_rationale')}\nChecks: {json.dumps({k: v for k, v in pc.items() if not k.startswith('_')})}"
                          for i, c, pc in shown)
    sel = llm.call(P.select_prompt(req['demand'], listing, lever=lever), P.SELECT_SCHEMA, rq, 'select', effort='high')
    task = {'id': req['id'], 'request': req, 'demand': req['demand'], 'lever': lever, 'select_reason': sel.get('reason')}
    ranked = [sel.get('chosen_index', -1)] + [i for i, _, _ in shown if i != sel.get('chosen_index', -1)]
    ranked = [x for i in ranked for x in checked if x[0] == i and x[2].get('compiles')]
    if sel.get('chosen_index', -1) < 0 or not ranked:
        task.update({'class': 'rejected', 'reasons': ['selector chose none'], 'final': True})
        _write(final, task)
        return task
    if REGISTRY:
        pick, dups = register(rd, req['id'], ranked)
        task['round_duplicates_avoided'] = dups
        if pick is None:
            task.update({'class': 'rejected', 'reasons': [f'every candidate duplicates a world already in this round: {dups}'],
                         'final': True})
            _write(final, task)
            return task
        ranked = [x for x in ranked if x[0] == pick]
    _, cand, pc = ranked[0]
    plan = llm.call(P.plan_prompt(req['demand'], json.dumps(cand, indent=1),
                                  json.dumps({k: v for k, v in pc.items() if not k.startswith('_')}, indent=1), BUDGET,
                                  lever=lever, naming=naming),
                    P.PLAN_SCHEMA, rq, 'plan', effort=PLAN_EFFORT, timeout=3600)
    spec = jload(plan.get('world_json'))
    task.update({'title': cand.get('title'), 'story': cand.get('story'), 'world': spec,
                 'description': plan.get('public_description'), 'rivals': named_rivals(plan.get('rivals')),
                 'plan_experiments': jload(plan.get('plan_json'), []), 'plan_notes': plan.get('notes')})
    try:
        w = World(spec)
    except Exception as e:
        task.update({'class': 'rejected', 'reasons': [f'final world does not compile: {e}'], 'final': True})
        _write(final, task)
        return task
    key = json.dumps([spec, task['rivals'], task['plan_experiments'], BUILD_VERSION], sort_keys=True)
    cache = tdir / 'build.json'
    cached = json.loads(cache.read_text()) if cache.exists() else None
    if cached and cached.get('key') == key:
        a, queries, extra = cached['audit'], cached['queries'], cached['extra']
    else:
        t0 = time.time()
        with _compute:
            try:
                _, _, queries, extra, a = Au.build(spec, req['demand'], parse_plan(w, task['plan_experiments']),
                                                   named=task['rivals'], budget=BUDGET)
            except Exception as e:
                a, queries, extra = {'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-2000:]}, [], []
        a['seconds'] = round(time.time() - t0)
        _write(cache, {'key': key, 'audit': a, 'queries': queries, 'extra': extra})
    task['audit'], task['novelty'] = a, Nv.matches(spec, arch)
    task['lever_profile'] = Lv.profile(spec)
    task['silent_fraction'] = silent_fraction(w)
    # publish only fair queries: none that a rival left by the identifiability check, or a needle rival, answers
    # differently from the truth
    ident = a.get('identifiability', {})
    left = set(ident.get('rivals_left_idx', [])) | set(ident.get('needle_idx', []))
    task['queries'] = [q for q in queries if not (set(q.get('any_rival', [])) & left)]
    task['queries_dropped'] = len(queries) - len(task['queries'])
    log(rd, f"{task['id']}: built in {a.get('seconds')}s; audit {(a.get('audit') or {}).get('passes')}; "
            f"reasons {(a.get('audit') or {}).get('reasons')}{'; ERROR ' + a['error'] if a.get('error') else ''}")
    if a.get('error') or len(task['queries']) < MIN_QUERIES:
        task['class'], task['reasons'] = classify(task)
        task['final'] = True
        _write(final, task)
        return task
    _write(final, task)   # provisional: a restart resumes at the pilots
    calls = tdir / 'calls'
    task['pilots'] = {}
    for label, b in (('none', 0), ('quarter', BUDGET // 4), ('full', BUDGET)):
        try:
            task['pilots'][label] = H.run_pilot(w, task['description'] or 'An unfamiliar device.', task['queries'], b,
                                                calls, llm.call, effort=PILOT_EFFORT, name=f'pilot_{label}')
        except llm.LLMError as e:
            task['pilots'][label] = {'error': str(e)}
        _write(final, task)
    task['class'], task['reasons'] = classify(task)
    task['code'] = _code_hash()
    task['final'] = True
    _write(final, task)
    pl = {k: (v.get('critical_accuracy'), v.get('accuracy')) for k, v in task['pilots'].items()}
    log(rd, f"{task['id']}: class {task['class']} {task['reasons']}; pilots (critical, all) {pl}")
    return task


REGISTRY = os.environ.get('RE_A_REGISTRY') == '1'


def round_titles(rd, task_id):
    reg = rd / 'registry.jsonl'
    if not reg.exists():
        return ''
    rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    return '\n'.join(f"- (taken in this round) {r['title']}: {r.get('story', '')}" for r in rows if r['id'] != task_id)


def register(rd, task_id, ranked):
    """Under a lock, take the first ranked candidate whose behaviour fingerprint and structure signature match no
    world already taken in this round. Returns (index or None, [(index, duplicated ids)])."""
    import fcntl
    reg, lock = rd / 'registry.jsonl', rd / '.registry.lock'
    dups = []
    with open(lock, 'a') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()] if reg.exists() else []
            mine = [r for r in rows if r['id'] == task_id]
            if mine:
                return mine[0]['index'], []
            for i, c, pc in ranked:
                same = [r['id'] for r in rows if r['fp'] == pc.get('_fp') or r['struct'] == pc.get('_struct')]
                if same:
                    dups.append((i, same))
                    continue
                with open(reg, 'a') as f:
                    f.write(json.dumps({'id': task_id, 'index': i, 'title': c.get('title'), 'story': c.get('story'),
                                        'fp': pc.get('_fp'), 'struct': pc.get('_struct')}) + '\n')
                return i, dups
            return None, dups
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _task_main(req, rd, arch, arch_text):
    try:
        run_task(req, rd, arch, arch_text)
    except BaseException as e:
        (rd / 'tasks' / req['id']).mkdir(parents=True, exist_ok=True)
        (rd / 'tasks' / req['id'] / 'failure.txt').write_text(traceback.format_exc())
        log(rd, f"{req['id']}: FAILED {type(e).__name__}: {e}")
        raise SystemExit(1)


def summarize(rd, requests):
    rows = []
    for req in requests:
        f = rd / 'tasks' / req['id'] / 'dossier.json'
        t = json.loads(f.read_text()) if f.exists() else {}
        rows.append({'id': req['id'], 'demand': req['demand'], 'lever': req.get('lever'), 'title': t.get('title'),
                     'lever_profile': t.get('lever_profile'),
                     'class': t.get('class') if t.get('final') else 'pending', 'reasons': t.get('reasons'),
                     'pilots': {k: v.get('critical_accuracy') for k, v in (t.get('pilots') or {}).items()}})
    return rows


def run_round(name, requests, workers=12, avoid=()):
    """Run every request in its own process, at most `workers` at once. `avoid`: extra lines for the ideas prompt's
    list of known mechanisms (for example earlier rounds' worlds, as "- title: story")."""
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor
    rd = ROOT / 'runs' / name
    (rd / 'tasks').mkdir(parents=True, exist_ok=True)
    (rd / 'requests.json').write_text(json.dumps(requests, indent=1))
    log(rd, f'round {name} (domain A): {len(requests)} requests, {workers} workers, budget {BUDGET}')
    arch = Nv.load_archive()
    arch_text = P.archive_lines(arch) + ''.join('\n' + x for x in avoid)
    ctx = mp.get_context('spawn')

    def one(req):
        fail = rd / 'tasks' / req['id'] / 'failure.txt'
        if fail.exists():
            fail.rename(fail.with_name(f'failure-{int(time.time())}.txt'))
        proc = ctx.Process(target=_task_main, args=(req, rd, arch, arch_text), name=f"task-{req['id']}")
        proc.start()
        proc.join()
        try:
            _write(rd / 'progress.json', summarize(rd, requests))
        except Exception:
            pass
        return proc.exitcode == 0

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


def _code_hash():
    import hashlib
    h = hashlib.sha256()
    for d in (Path(__file__).parent, Path(__file__).parent.parent / 'construct'):
        for f in sorted(d.glob('*.py')):
            h.update(f.read_bytes())
    return h.hexdigest()[:12]
