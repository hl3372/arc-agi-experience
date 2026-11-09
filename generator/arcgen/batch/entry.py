"""arcgen batch entry point.

Config selection happens here, before any pipeline module is imported:
  * a prepared run always uses its frozen `<root>/config.json` (plus recorded amendments);
  * otherwise `--config PATH` (a batch profile, such as one made by `bin/arcgen new`);
  * otherwise arcgen/batch/profiles/v1.json.

Commands: prepare | qualify | ready | quota | authorize | run | status | verify | test |
          amend (operations-only change with a reason) | show-config
"""
from pathlib import Path
import json
import shutil
import os
import subprocess
import sys
import time
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
if any(k == 'arcgen.lib' or k.startswith('arcgen.lib.') for k in sys.modules): raise RuntimeError('fresh process required')
sys.path.insert(0, str(REPO))
from arcgen.paths import WORKSPACE as ROOT, GATE  # noqa: E402  (data root: runs/ and the bank)
COMMANDS = ['prepare', 'test', 'qualify', 'ready', 'run', 'status', 'verify', 'quota', 'authorize', 'amend', 'show-config']
TEST_SUITES = {'v1': REPO / 'tests'}  # the offline suite, run under the v1 profile


def select(args):
    chosen = Path(args.config).resolve() if args.config else HERE / 'profiles/v1.json'
    root = Path(args.root).resolve() if args.root else None
    if root is None:
        root = ROOT / json.loads(chosen.read_text())['default_run']
    frozen_config = root / 'config.json'
    if frozen_config.exists():
        if args.config and json.loads(chosen.read_text()) != json.loads(frozen_config.read_text()):
            raise SystemExit('This run was prepared with a different config; use `amend` for operational changes.')
        chosen = frozen_config
    os.environ['ARCGEN_CONFIG'] = str(chosen)
    os.environ['ARCGEN_RUN_ROOT'] = str(root)
    return root


def amend(root, settings, reason):
    from arcgen.batch.config import apply_amendments, CONFIG, AMENDABLE_KEYS
    if not (root / 'config.json').exists(): raise SystemExit('amend applies to a prepared run')
    if not reason: raise SystemExit('--reason is required for an amendment')
    # Pool sizes and development windows may change on a started run too (2026-10-09): the scheduler compares its frozen
    # policy without them and the pools record the resize. Either takes effect when the run is next resumed.
    changes = {}
    for item in settings:
        key, _, raw = item.partition('=')
        top = key.split('.', 1)[0]
        if top not in AMENDABLE_KEYS: raise SystemExit(f'{top} cannot be amended on a prepared run (allowed: {sorted(AMENDABLE_KEYS)})')
        value = json.loads(raw)
        if '.' in key:
            base = dict(CONFIG['operations'][top]); base[key.split('.', 1)[1]] = value; value = base
        changes[top] = value
    directory = root / 'config_amendments'; directory.mkdir(exist_ok=True)
    path = directory / f'{len(list(directory.glob("*.json"))) + 1:03d}.json'
    body = {'operations': changes, 'reason': reason, 'at_epoch': time.time()}
    path.write_text(json.dumps(body, indent=1))
    try:
        apply_amendments(json.loads((root / 'config.json').read_text()), root)
    except Exception:
        path.unlink(); raise
    return {'status': 'amended', 'amendment': path.name, 'operations': changes,
            'note': 'Takes effect when the run is next started or resumed.'}


def _suite_env(profile):
    return dict(os.environ, ARCGEN_CONFIG=str(HERE / 'profiles' / (profile + '.json')), ARCGEN_RUN_ROOT='',
                PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get('PYTHONPATH')])))


def run_tests():
    """The offline suite, serially (ARCGEN_GATE_WORKERS=1) or split over parallel pytest processes (the default: 4)."""
    workers = int(os.environ.get('ARCGEN_GATE_WORKERS', '4'))
    GATE.mkdir(parents=True, exist_ok=True)
    if workers > 1:
        return run_tests_parallel(workers)
    results = {}
    for profile, suite in TEST_SUITES.items():
        if not suite.exists(): continue
        code = subprocess.call([sys.executable, '-B', '-m', 'pytest', str(suite), '-q', '-p', 'no:cacheprovider',
                                '--junitxml=' + str(GATE / f'focused_tests_{profile}.xml')], env=_suite_env(profile), cwd=REPO)
        results[profile] = 'passed' if code == 0 else 'failed'
    return results


SLOW_SECONDS = 60  # a test this slow in the last gate runs in a process of its own


def _node_id(classname, name):
    """A junit testcase's pytest node id: arcgen.batch.tests_v1.test_x[.TestC] + name -> arcgen/batch/tests_v1/test_x.py[::TestC]::name."""
    parts = classname.split('.')
    k = next((i for i, p in enumerate(parts) if p.startswith('test_')), len(parts) - 1)  # the module (suite folders are tests*)
    return '/'.join(parts[:k + 1]) + '.py' + ''.join('::' + c for c in parts[k + 1:]) + '::' + name


def _last_durations(profile):
    import xml.etree.ElementTree as ET
    path = GATE / f'focused_tests_{profile}.xml'
    try:
        return {_node_id(c.get('classname', ''), c.get('name', '')): float(c.get('time') or 0) for c in ET.parse(path).getroot().iter('testcase')}
    except (OSError, ET.ParseError):
        return {}


def _collect(profile, suite):
    out = subprocess.run([sys.executable, '-B', '-m', 'pytest', str(suite.relative_to(REPO)), '--collect-only', '-q', '-p', 'no:cacheprovider'],
                         env=_suite_env(profile), cwd=REPO, capture_output=True, text=True)
    ids = [line.strip() for line in out.stdout.splitlines() if '::' in line]
    if out.returncode or not ids:
        raise RuntimeError(f'test collection failed for {profile}:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}')
    return ids


def _merge_junit(parts, destination):
    """One junit file for a suite from its jobs' files: every testcase, the counts summed."""
    import xml.etree.ElementTree as ET
    merged = ET.Element('testsuite', name='pytest')
    totals = dict.fromkeys(('errors', 'failures', 'skipped', 'tests'), 0); seconds = 0.0
    for part in parts:
        for suite in ET.parse(part).getroot().iter('testsuite'):
            for key in totals:
                totals[key] += int(suite.get(key) or 0)
            seconds += float(suite.get('time') or 0)
            merged.extend(suite.findall('testcase'))
    for key, value in totals.items():
        merged.set(key, str(value))
    merged.set('time', f'{seconds:.3f}'); merged.set('timestamp', time.strftime('%Y-%m-%dT%H:%M:%S')); merged.set('hostname', os.uname().nodename)
    root = ET.Element('testsuites'); root.append(merged)
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)
    return {**totals, 'testcases': len(merged.findall('testcase'))}


def run_tests_parallel(workers):
    """Each suite's slow tests (by the last gate's times) in processes of their own, its other tests in one process,
    `workers` processes at a time, longest first, each with its own basetemp (deleted as soon as the job ends; its .log
    and .xml stay); the jobs' junit files merged into the
    usual evidence file per suite. A suite passes only when every job passes and every collected test was reported."""
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    GATE.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='arcgen-gate-'))
    jobs, collected = [], {}
    for profile, suite in TEST_SUITES.items():
        if not suite.exists(): continue
        ids = collected[profile] = _collect(profile, suite)
        times = _last_durations(profile)
        slow = [i for i in ids if times.get(i, 0) >= SLOW_SECONDS]
        fast = [i for i in ids if i not in slow]
        jobs += [(times[i], profile, [i]) for i in slow] + ([(sum(times.get(i, 1) for i in fast), profile, fast)] if fast else [])
    jobs.sort(key=lambda j: -j[0])

    def run(k, job):
        _, profile, ids = job
        xml = scratch / f'{profile}-{k:03d}.xml'
        basetemp = scratch / f'tmp-{k:03d}'
        try:
            with open(scratch / f'{profile}-{k:03d}.log', 'w') as log:
                code = subprocess.call([sys.executable, '-B', '-m', 'pytest', *ids, '-q', '-p', 'no:cacheprovider',
                                        f'--basetemp={basetemp}', f'--junitxml={xml}'],
                                       env=_suite_env(profile), cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        finally:
            # 2026-10-25: the job's temp data goes as soon as it ends (the gate peaked at ~12 GB of prepared runs and bank
            # copies with every basetemp kept to the end); its .log and .xml stay with the job logs
            shutil.rmtree(basetemp, ignore_errors=True)
        print(f'{time.strftime("%H:%M:%S")} {profile} job {k}: {len(ids)} test(s) {"passed" if code == 0 else "FAILED"}', flush=True)
        return profile, code, xml

    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(lambda kj: run(*kj), enumerate(jobs)))
    results = {}
    for profile, ids in collected.items():
        mine = [(code, xml) for p, code, xml in done if p == profile]
        counts = _merge_junit([xml for _, xml in mine if xml.exists()], GATE / f'focused_tests_{profile}.xml')
        ok = all(code == 0 for code, _ in mine) and counts['testcases'] == len(ids) and not counts['failures'] and not counts['errors']
        results[profile] = 'passed' if ok else 'failed'
        print(f'{profile}: {counts["testcases"]} of {len(ids)} tests reported, {counts["failures"]} failures, {counts["errors"]} errors '
              f'-> {results[profile]}', flush=True)
    print('job logs:', scratch, flush=True)
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=COMMANDS)
    p.add_argument('--root', type=Path, help='run root (default: the config default_run)')
    p.add_argument('--config', type=Path, help='profile or config JSON for a new run')
    p.add_argument('--authorization', help='Exact user launch instruction; required only for authorize')
    p.add_argument('--set', action='append', default=[], help='amend: KEY=JSON or KEY.SUBKEY=JSON (operations only)')
    p.add_argument('--reason', help='amend: why the operational setting changes')
    a = p.parse_args()
    root = select(a)
    if a.command == 'test':
        results = run_tests()
        from arcgen.batch.common import atomic, code_identity
        status = 'passed' if results and all(v == 'passed' for v in results.values()) else 'failed'
        GATE.mkdir(parents=True, exist_ok=True); atomic(GATE / 'offline_gate.json', dict(status=status, suites=results, code_hash=code_identity(), new_text_calls=0))
        print(json.dumps({'status': status, 'suites': results}, indent=2))
        raise SystemExit(0 if status == 'passed' else 1)
    if a.command == 'amend':
        print(json.dumps(amend(root, a.set, a.reason), indent=2)); raise SystemExit(0)
    from arcgen.batch.common import read
    from arcgen.batch.config import CONFIG, science_hash
    if a.command == 'show-config':
        print(json.dumps({**{k: v for k, v in CONFIG.items() if k not in ('science',)}, 'science_hash': science_hash(CONFIG),
                          'science': CONFIG['science']}, indent=1)); raise SystemExit(0)
    if a.command == 'status':
        from arcgen.batch.budget import Budget
        result = read(root / 'STATUS.json'); s = Budget(root).summary()
        result['accounting'] = {k: s[k] for k in ('logical_calls', 'physical_attempts', 'developments', 'by_phase_category')}
    elif a.command == 'quota':
        from arcgen.batch.next_batch import observe_quota
        result = observe_quota(root)
    elif a.command == 'authorize':
        from arcgen.batch.next_batch import authorize
        result = authorize(root, authorization=a.authorization)
    elif a.command == 'verify':
        from arcgen.batch.report import verify_terminal
        result = verify_terminal(root)
    else:
        from arcgen.batch import next_batch
        result = getattr(next_batch, a.command)(root)
    print(json.dumps({k: v for k, v in result.items() if k in ('status', 'revision', 'counts', 'accounting', 'logical_calls',
                      'physical_attempts', 'developments', 'accounts', 'account', 'reason', 'progress_preserved')}, indent=2))
