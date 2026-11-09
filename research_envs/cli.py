"""research-envs: generate, pilot and validate the two research-environment domains.

    research-envs info                                  configuration, Codex account, sandbox status
    research-envs discover round NAME [--preset demo]   generate a round of domain A worlds (model calls)
    research-envs discover requery NAME --from RUN --ids a,b   rebuild worlds of a run with the current query rules
    research-envs discover pilot TARGET... [--solver codex:high | or:MODEL]   run solvers on worlds (model calls)
    research-envs construct round NAME [--preset demo]  generate a round of domain B problems (model calls)
    research-envs construct pilot TARGET...             run the object and code pilots on problems (model calls)
    research-envs validate TARGET... [--full]           check worlds and problems offline (no model calls)
    research-envs export RUN ID...                      write task files from a run's dossiers
    research-envs check-isolation                       one model call: can a pilot read a file on this disk?

TARGET is a task file, an example id (examples/discover/*.json, examples/construct/*.json), a run's dossier.json,
or (for pilots) a run name under runs/. Presets set environment flags only where you have not set them yourself;
command-line options override both. Everything is written under runs/ (see research_envs/common.py).
"""
import argparse
import json
import os
import sys
import time

# ------------------------------------------------------------------ presets

A_PRESETS = {   # name: (environment defaults, requests, id prefix, workers)
    'demo': ({'RE_A_V2': '1', 'RE_A_REGISTRY': '1', 'RE_A_QUALITY': '1', 'RE_A_TRIVIAL_CAP': '0.3',
              'RE_A_CANDIDATES': '4', 'RE_A_QUERY_MODE': 'discriminating', 'RE_A_ANSWER_CAP': '0.4'},
             'time:3,cycle:3,alias:3,coupled:3', 'da', 12),
    'round2': ({'RE_A_V2': '1', 'RE_A_REGISTRY': '1'}, 'time:6,cycle:6,alias:6,coupled:6,control:8', 'a2', 16),
    'round1': ({}, 'A1:4,A2:4,A3:4,A4:4,control:16', 'a1', 16),
}
B_PRESETS = {
    'demo': ({'RE_ROUND_REGISTRY': '1', 'RE_IMPROVE': '1', 'RE_B_QUALITY': '1'},
             'D2:graph:2,D2:set:3,D4:graph:3,control:graph:2,control:set:2', 'db', 12),
    'round2': ({'RE_ROUND_REGISTRY': '1', 'RE_IMPROVE': '1'},
               ','.join(f'D{d}:{o}:2' for d in range(1, 5) for o in ('graph', 'set')) + ',control:graph:8,control:set:8',
               'r2', 32),
    'round1': ({}, ','.join(f'D{d}:{o}:2' for d in range(1, 5) for o in ('graph', 'set')) + ',control:graph:8,control:set:8',
               'r1', 32),
    'preview': ({}, 'D1:graph:1,D2:set:1,D3:graph:1,D4:graph:1,control:graph:1,control:set:1', 'p', 6),
}
LEVERS = ('time', 'cycle', 'alias', 'coupled')


def _defaults(env):
    for k, v in env.items():
        os.environ.setdefault(k, v)


def _set(key, value):
    if value is not None:
        os.environ[key] = str(value)


def a_requests(spec, prefix):
    out = []
    for item in spec.split(','):
        arm, n = item.strip().rsplit(':', 1)
        for k in range(1, int(n) + 1):
            if arm in LEVERS:
                out.append({'id': f'{prefix}_{arm}_{k}', 'demand': None, 'lever': arm})
            elif arm in ('A1', 'A2', 'A3', 'A4'):
                out.append({'id': f'{prefix}_{arm.lower()}_{k}', 'demand': arm})
            elif arm == 'control':
                out.append({'id': f'{prefix}_ctl_{k}', 'demand': None})
            else:
                raise SystemExit(f'unknown domain A arm {arm!r} (levers {LEVERS}, demands A1-A4, control)')
    return out


def b_requests(spec, prefix):
    out = []
    for item in spec.split(','):
        arm, obj, n = item.strip().split(':')
        tag, obj = {'graph': ('g', 'graph'), 'set': ('s', 'set_system'), 'set_system': ('s', 'set_system')}[obj]
        if arm not in ('D1', 'D2', 'D3', 'D4', 'control'):
            raise SystemExit(f'unknown domain B arm {arm!r} (D1-D4, control)')
        for k in range(1, int(n) + 1):
            name = 'ctl' if arm == 'control' else arm.lower()
            out.append({'id': f'{prefix}_{name}_{tag}{k}', 'object': obj, 'demand': None if arm == 'control' else arm})
    return out


def _avoid_a(runs, examples):
    from .common import ROOT, EXAMPLES_DIR
    lines = []
    for rnd in runs:
        for f in sorted((ROOT / 'runs' / rnd / 'tasks').glob('*/dossier.json')):
            t = json.loads(f.read_text())
            if t.get('class') in ('solved', 'frontier'):
                lines.append(f"- (from our earlier rounds) {t['title']}: {t.get('story', '')}")
    if examples and EXAMPLES_DIR:
        for f in sorted((EXAMPLES_DIR / 'discover').glob('*.json')):
            t = json.loads(f.read_text())
            lines.append(f"- (from our earlier rounds) {t['title']}: {t['private'].get('story', '')}")
    return lines


def _avoid_b(runs, examples):
    from .common import ROOT, EXAMPLES_DIR
    titles = set()
    for rnd in runs:
        for f in sorted((ROOT / 'runs' / rnd / 'tasks').glob('*/dossier.json')):
            t = json.loads(f.read_text())
            if t.get('title'):
                titles.add(t['title'])
    if examples and EXAMPLES_DIR:
        titles |= {json.loads(f.read_text())['title'] for f in (EXAMPLES_DIR / 'construct').glob('*.json')}
    return [f'(from our earlier rounds) {x}' for x in sorted(titles)]


def _attempts(run, rd, attempts, pause=120):
    """Rerun a round until it completes (each attempt resumes from finished calls and cached builds)."""
    for k in range(1, attempts + 1):
        run()
        if (rd / 'summary.json').exists():
            print(f'round complete: {rd / "summary.json"}')
            return 0
        if k < attempts:
            print(f'round incomplete after attempt {k}; retrying in {pause} s', flush=True)
            time.sleep(pause)
    print(f'round incomplete after {attempts} attempt(s); rerun the same command to resume', flush=True)
    return 1


# ------------------------------------------------------------------ commands

def cmd_info(args):
    import shutil
    from . import common
    from .construct import llm, sandbox
    home = llm.codex_home()
    abi = sandbox.landlock_abi()
    print(f'runs root:          {common.ROOT / "runs"}')
    print(f'source checkout:    {common.CHECKOUT or "(not installed from a checkout)"}')
    print(f'codex CLI:          {shutil.which(llm.CODEX) or "NOT FOUND (" + llm.CODEX + ")"}')
    print(f'codex home:         {home} ({"exists" if home.exists() else "missing: run `codex login`"})')
    print(f'model:              {llm.MODEL}; concurrency {llm.CONCURRENCY} (RE_CODEX_CONCURRENCY)')
    print(f'program sandbox:    Landlock ABI {abi} ({"ok" if abi >= sandbox.MIN_ABI else "UNAVAILABLE: domain B builds and code pilots will refuse to run"})')
    if abi >= sandbox.MIN_ABI:
        try:
            print(f'sandbox read paths: {", ".join(sandbox.read_only_paths())}')
        except sandbox.SandboxUnavailable as e:
            print(f'sandbox read paths: REFUSED: {e}')
    print(f'OpenRouter key:     {"set" if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_KEY_FILE") else "not set (only needed for or: solvers)"}')
    return 0


def cmd_a_round(args):
    env, reqs, prefix, workers = A_PRESETS[args.preset]
    _defaults(env)
    _set('RE_A_ANSWER_CAP', args.answer_cap)
    _set('RE_A_BUDGET', args.budget)
    from .common import ROOT
    from .discover import pipeline
    requests = a_requests(args.requests or reqs, args.prefix or prefix)
    avoid = _avoid_a(args.avoid or [], not args.no_avoid_examples)
    rd = ROOT / 'runs' / args.name
    return _attempts(lambda: pipeline.run_round(args.name, requests, workers=args.workers or workers, avoid=avoid),
                     rd, args.attempts)


def cmd_a_requery(args):
    import shutil
    _defaults(A_PRESETS['demo'][0])
    _set('RE_A_ANSWER_CAP', args.answer_cap)
    from .common import ROOT
    from .discover import pipeline
    runs = ROOT / 'runs'
    ids = args.ids.split(',')
    rd = runs / args.name
    (rd / 'requests').mkdir(parents=True, exist_ok=True)
    pool = []
    for src in args.sources:
        reqs = json.loads((runs / src / 'requests.json').read_text())
        regf = runs / src / 'registry.jsonl'
        reg = [json.loads(l) for l in regf.read_text().splitlines() if l.strip()] if regf.exists() else []
        for r in reqs:
            if r['id'] in ids and r['id'] not in [p['id'] for p in pool]:
                pool.append(r)
                if not (rd / 'requests' / r['id']).exists():     # the cached idea, select and plan calls: same world
                    shutil.copytree(runs / src / 'requests' / r['id'], rd / 'requests' / r['id'])
                    with open(rd / 'registry.jsonl', 'a') as f:
                        f.writelines(json.dumps(x) + '\n' for x in reg if x['id'] == r['id'])
    missing = set(ids) - {p['id'] for p in pool}
    if missing:
        raise SystemExit(f'unknown ids: {sorted(missing)}')
    return _attempts(lambda: pipeline.run_round(args.name, pool, workers=args.workers or len(pool)), rd, args.attempts)


def _solver(spec):
    """codex:EFFORT, codex:MODEL:EFFORT or or:MODEL -> (call, tag)."""
    kind, _, rest = spec.partition(':')
    if kind == 'codex':
        from .construct import llm
        model, _, effort = rest.rpartition(':')
        model = model or llm.MODEL

        def call(prompt, schema, record_dir, name, effort_=None, **k):
            return llm.call(prompt, schema, record_dir, name, effort=effort, timeout=1800, model=model)
        return call, f'codex__{model}__{effort}'.replace('/', '_')
    if kind == 'or':
        from .construct import openrouter
        return openrouter.make_call(rest), f'or__{rest}'.replace('/', '_').replace(':', '__')
    raise SystemExit(f'unknown solver {spec!r}: use codex:EFFORT, codex:MODEL:EFFORT or or:MODEL')


def cmd_a_pilot(args):
    _defaults(A_PRESETS['demo'][0])
    from concurrent.futures import ThreadPoolExecutor
    from .common import ROOT
    from . import tasks as T
    from .construct.pipeline import _write
    from .discover import harness as H
    from .discover.world import World
    jobs = []
    for target in args.targets:
        rd = ROOT / 'runs' / target
        if (rd / 'tasks').is_dir():          # a run: every finished, usable world in it
            for f in sorted((rd / 'tasks').glob('*/dossier.json')):
                t = json.loads(f.read_text())
                if t.get('final') and t.get('class') in ('solved', 'frontier') and t.get('queries'):
                    jobs.append((T.export_discover(t, source_run=target), rd / 'validation'))
        else:
            t = T.load(target)
            if T.domain(t) != 'discover':
                raise SystemExit(f'{target}: not a domain A world')
            jobs.append((t, ROOT / 'runs' / 'pilots_a'))
    solvers = [(s, *_solver(s)) for s in args.solver]
    work = [(s, call, tag, t, out) for s, call, tag in solvers for t, out in jobs]
    print(f'{len(jobs)} worlds x {len(solvers)} solvers = {len(work)} pilots', flush=True)

    def one(job):
        s, call, tag, t, out = job
        dest = out / tag / f"{t['id']}.json"
        if dest.exists():
            return json.loads(dest.read_text())
        w = World(t['private']['world'])
        budget = args.budget if args.budget is not None else t['settings'].get('budget', 40)

        def c(prompt, schema, record_dir, name, effort=None):
            return call(prompt, schema, record_dir, name)
        try:
            r = H.run_pilot(w, t['public'].get('description') or 'An unfamiliar device.', T.discover_queries(t), budget,
                            out / tag / t['id'] / 'calls', c, name='val')
        except Exception as e:
            r = {'error': f'{type(e).__name__}: {e}'}
        r = {k: v for k, v in r.items() if k != 'history'} | {'experiments': len(r.get('history', []))}
        res = {'solver': s, 'task': t['id'], 'demand': t['request'].get('demand'), 'lever': t['request'].get('lever'),
               'title': t.get('title'), **r}
        _write(dest, res)
        print(f"{s} {t['id']}: critical {r.get('critical_accuracy')} all {r.get('accuracy')} used "
              f"{r.get('experiments_used')}{' ERROR ' + r['error'] if r.get('error') else ''}", flush=True)
        return res

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(one, work))
    for s, _, _ in solvers:
        ok = [r['critical_accuracy'] for r in results if r['solver'] == s and r.get('critical_accuracy') is not None]
        print(f"{s}: mean critical accuracy {sum(ok) / len(ok) if ok else float('nan'):.2f} over {len(ok)}; "
              f"perfect {sum(x == 1.0 for x in ok)}; errors {sum(1 for r in results if r['solver'] == s and r.get('error'))}")
    return 0


def cmd_b_round(args):
    env, reqs, prefix, workers = B_PRESETS[args.preset]
    _defaults(env)
    _set('RE_BUDGET', args.budget)
    from .common import ROOT
    from .construct import pipeline
    requests = b_requests(args.requests or reqs, args.prefix or prefix)
    avoid = _avoid_b(args.avoid or [], not args.no_avoid_examples)
    rd = ROOT / 'runs' / args.name
    return _attempts(lambda: pipeline.run_round(args.name, requests, workers=args.workers or workers, avoid=avoid),
                     rd, args.attempts)


def _b_pilot_one(ref):
    """One problem's pilots, in its own process (the evaluator's timeouts need a main thread)."""
    from .common import ROOT
    from . import tasks as T
    from .construct import pipeline as Pp
    t = T.load(ref)
    task = T.construct_task(t)
    rd = ROOT / 'runs' / 'pilots_b'
    dest = rd / 'tasks' / task['id'] / 'pilots.json'
    if dest.exists():
        return json.loads(dest.read_text())
    pilots = Pp.run_pilots(task, rd)
    outcomes = Pp.adjudicate(task, {}, pilots, None)
    res = {'task': task['id'], 'title': t.get('title'), 'reference': task['reference'],
           'best_known': (task.get('best_known') or {}).get('objective'), 'outcomes': outcomes, 'pilots': pilots}
    Pp._write(dest, res)
    return res


def cmd_b_pilot(args):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    _defaults(B_PRESETS['demo'][0])
    from . import tasks as T
    for ref in args.targets:
        if T.domain(T.load(ref)) != 'construct':
            raise SystemExit(f'{ref}: not a domain B problem')
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as ex:
        for r in ex.map(_b_pilot_one, args.targets):
            print(f"{r['task']}: reference {r['reference']}, best known {r['best_known']}; pilots "
                  f"{[(o['arm'], o['class'], o['objective']) for o in r['outcomes']]}", flush=True)
    return 0


def cmd_validate(args):
    _defaults(A_PRESETS['demo'][0])
    _defaults(B_PRESETS['demo'][0])
    from . import tasks as T
    refs = []
    for ref in args.targets:
        if os.path.isdir(ref):
            refs += sorted(os.path.join(ref, f) for f in os.listdir(ref) if f.endswith('.json'))
        else:
            refs.append(ref)
    failed = 0
    out = []
    for ref in refs:       # one at a time in the main thread (the evaluator's timeouts use SIGALRM)
        t = T.load(ref)
        r = T.validate(t, full=args.full, sandbox=not args.no_sandbox)
        out.append(r)
        failed += not r['ok']
        if not args.json:
            print(f"{'PASS' if r['ok'] else 'FAIL'} {r['domain']}/{r['id']} ({r['seconds']} s)")
            for c in r['checks']:
                mark = {True: 'ok  ', False: 'FAIL', None: 'skip'}[c['ok']]
                print(f"   {mark} {c['check']}: {c['detail']}")
    if args.json:
        print(json.dumps(out, indent=1))
    print(f'{len(refs) - failed}/{len(refs)} passed')
    return 1 if failed else 0


def cmd_export(args):
    _defaults(A_PRESETS['demo'][0])
    from pathlib import Path
    from .common import ROOT
    from . import tasks as T
    rd = ROOT / 'runs' / args.run
    out = Path(args.out) if args.out else rd / 'export'
    out.mkdir(parents=True, exist_ok=True)
    for i in args.ids:
        t = T.load(str(rd / 'tasks' / i / 'dossier.json'))
        t['source_run'] = args.run
        (out / f'{i}.json').write_text(json.dumps(t, indent=1) + '\n')
        print(out / f'{i}.json')
    return 0


def cmd_check_isolation(args):
    """Plant a canary file under runs/ and ask the model, in the same mode the pilots use, to print it."""
    import secrets
    from .common import ROOT
    from .construct import llm
    token = secrets.token_hex(16)
    rd = ROOT / 'runs' / 'isolation_check' / time.strftime('%Y%m%d-%H%M%S')
    rd.mkdir(parents=True, exist_ok=True)
    canary = rd / 'canary.txt'
    canary.write_text(token + '\n')
    prompt = (f'This is a sandbox test. Run the shell command `cat {canary}` and report its exact output in the field '
              f'"output". If you cannot run shell commands or cannot read the file, say so in "output" and set '
              f'"could_read" to false.')
    schema = {'type': 'object', 'properties': {'output': {'type': 'string'}, 'could_read': {'type': 'boolean'}}}
    try:
        r = llm.call(prompt, schema, rd, 'probe', effort='low', timeout=600, retries=0)
    finally:
        canary.unlink(missing_ok=True)
    leaked = token in json.dumps(r)
    print(json.dumps(r, indent=1))
    if leaked:
        print('LEAK: the model read a file on this disk. Pilots here could read runs/ (answers, witnesses). Run them '
              'in a container or VM that holds only what they may see; see README, "Sandboxes".')
        return 1
    print('No leak in this probe: the model did not return the canary. This is one probe, not a proof of isolation.')
    return 0


# ------------------------------------------------------------------ parser

def build_parser():
    p = argparse.ArgumentParser(prog='research-envs', description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('info', help='show configuration').set_defaults(f=cmd_info)

    a = sub.add_parser('discover', help='domain A: discovery worlds').add_subparsers(dest='sub', required=True)
    r = a.add_parser('round', help='generate a round of worlds (model calls)')
    r.add_argument('name')
    r.add_argument('--preset', choices=sorted(A_PRESETS), default='demo', help='demo = the final settings (default)')
    r.add_argument('--requests', help='arms and counts, e.g. "time:3,alias:2,A2:4,control:2" (levers time, cycle, '
                                      'alias, coupled; demands A1-A4; control)')
    r.add_argument('--prefix', help='task id prefix')
    r.add_argument('--workers', type=int, help='task processes at once (model calls are limited by RE_CODEX_CONCURRENCY)')
    r.add_argument('--budget', type=int, help='experiment budget (RE_A_BUDGET, default 40)')
    r.add_argument('--answer-cap', type=float, help='max share of queries with one answer value (RE_A_ANSWER_CAP)')
    r.add_argument('--avoid', action='append', metavar='RUN', help="earlier run whose worlds go on the avoid list")
    r.add_argument('--no-avoid-examples', action='store_true', help='do not put the published examples on the avoid list')
    r.add_argument('--attempts', type=int, default=1, help='rerun until complete, up to this many times')
    r.set_defaults(f=cmd_a_round)
    q = a.add_parser('requery', help="rebuild a run's worlds with the current query rules (reuses design calls)")
    q.add_argument('name')
    q.add_argument('--from', dest='sources', action='append', required=True, metavar='RUN')
    q.add_argument('--ids', required=True, help='comma-separated task ids')
    q.add_argument('--answer-cap', type=float)
    q.add_argument('--workers', type=int)
    q.add_argument('--attempts', type=int, default=1)
    q.set_defaults(f=cmd_a_requery)
    pl = a.add_parser('pilot', help='run solvers on worlds (model calls)')
    pl.add_argument('targets', nargs='+', help='task files, example ids, dossiers, or run names')
    pl.add_argument('--solver', action='append', default=None,
                    help='codex:EFFORT, codex:MODEL:EFFORT or or:OPENROUTER_MODEL (repeatable; default codex:high)')
    pl.add_argument('--budget', type=int, help="experiment budget (default: the task's)")
    pl.add_argument('--workers', type=int, default=8)
    pl.set_defaults(f=cmd_a_pilot)

    b = sub.add_parser('construct', help='domain B: construction problems').add_subparsers(dest='sub', required=True)
    r = b.add_parser('round', help='generate a round of problems (model calls)')
    r.add_argument('name')
    r.add_argument('--preset', choices=sorted(B_PRESETS), default='demo', help='demo = the final settings (default)')
    r.add_argument('--requests', help='arms, objects and counts, e.g. "D2:graph:2,D4:graph:3,control:set:2"')
    r.add_argument('--prefix')
    r.add_argument('--workers', type=int)
    r.add_argument('--budget', type=int, help='battery budget B (RE_BUDGET, default 20000; persistence at 10 B)')
    r.add_argument('--avoid', action='append', metavar='RUN')
    r.add_argument('--no-avoid-examples', action='store_true')
    r.add_argument('--attempts', type=int, default=1)
    r.set_defaults(f=cmd_b_round)
    pl = b.add_parser('pilot', help='run the object and code pilots on problems (model calls)')
    pl.add_argument('targets', nargs='+', help='task files, example ids or dossiers')
    pl.add_argument('--workers', type=int, default=4)
    pl.set_defaults(f=cmd_b_pilot)

    v = sub.add_parser('validate', help='check worlds and problems offline (no model calls)')
    v.add_argument('targets', nargs='+', help='task files, example ids, dossiers, or folders of task files')
    v.add_argument('--full', action='store_true', help='also rerun the whole build and audit (slow)')
    v.add_argument('--no-sandbox', action='store_true', help='do not run witness programs')
    v.add_argument('--json', action='store_true')
    v.set_defaults(f=cmd_validate)

    e = sub.add_parser('export', help="write task files from a run's dossiers")
    e.add_argument('run')
    e.add_argument('ids', nargs='+')
    e.add_argument('--out', help='folder (default runs/RUN/export)')
    e.set_defaults(f=cmd_export)
    sub.add_parser('check-isolation', help='one model call: can a pilot read a file on this disk?').set_defaults(
        f=cmd_check_isolation)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, 'solver', 1) is None:
        args.solver = ['codex:high']
    try:
        return args.f(args) or 0
    except (FileNotFoundError, ValueError) as e:
        print(f'research-envs: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
