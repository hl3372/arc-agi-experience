"""Self-serve batch commands: create, start, watch, stop/resume, summarise and commit a batch.

  bin/arcgen new <name> [--size N]      profile + allocation from the v1 reference design (56 idea calls);
                                        --size N repeats the design N times; prints time and token estimates
  bin/arcgen start <name> [--note ..]   offline test gate if the code changed, prepare/qualify/ready, a usage reading,
                                        authorize, then run under a detached supervisor (survives the terminal closing)
  bin/arcgen progress <name> [--watch]  readable progress: stage, ideas, selection, development, verified, calls, tokens
  bin/arcgen stop <name> [--now]        pause: let running calls finish and stop dispatch (--now: stop immediately)
  bin/arcgen resume <name>              clear a pause and continue from saved state
  bin/arcgen workers <name> <N>         change to N concurrent Codex calls; a running batch picks it up after `stop`
                                        and `resume`
  bin/arcgen summary <name>             the end-of-run summary (written automatically when the run ends)
  bin/arcgen commit <name> [--dry-run]  commit the verified tasks to the bank (project_stage stage_3 by default);
                                        several names commit their runs as one bank transaction
  bin/arcgen list                       batches and their status
  bin/arcgen test                       the offline test suite (the gate `start` requires for the current code)

Everything a batch writes is in the workspace (arcgen/paths.py): the batch's profile in batches/, the run in
runs/<name>/ and its operations folder runs/<name>-ops/ (run.log with pipeline events, progress.log, usage readings,
SUMMARY.md and SUMMARY.json), and committed tasks in the bank. Running a command is your instruction for it;
the command line and any --note are recorded as the launch or commit authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob, hashlib, json, os, re, shlex, shutil, signal, subprocess, sys, time
from pathlib import Path

from arcgen.paths import WORKSPACE as ROOT, REPO, PACKAGE, BANK, BATCHES, rel

PY = Path(sys.executable)  # every subprocess runs the interpreter that runs this command
HERE = PACKAGE / 'batch'
TEMPLATE_PROFILE = HERE / 'profiles/v1.json'
TEMPLATE_SPEC = HERE / 'allocations/v1_reference.spec.json'  # the property-slot design at 56 calls (2026-10-07)
TERMINAL = ('verified_terminal', 'complete_pending_semantic_synthesis')
PROGRESS_EVERY = 300
# Scaling estimates from earlier production batches of this design (two accounts, about 30 concurrent calls each): an
# idea call yielded about 3.2 verified tasks (3.4 developments of about 0.53 hours each); idea calls took about 0.6 hours
# and development about 1.8 hours of Codex time per idea call; pools stay about 80% busy, plus a tail of slow last
# tasks and one wait per extra reference wave. A verified task took about 6 model calls and 220-270k tokens in all.
VERIFIED_PER_CALL = 3.2
UP_HOURS_PER_CALL, DOWN_HOURS_PER_CALL, BUSY, TAIL_HOURS, WAVE_HOURS = 0.60, 1.78, 0.8, 1.5, 0.5
# saved-only batches: per development, from the same batches, verifying at about 0.85 per development
DEVELOPMENTS_PER_CALL, DOWN_HOURS_PER_DEVELOPMENT, VERIFIED_PER_SAVED = 3.4, 0.53, 0.85
CALLS_PER_VERIFIED, TOKENS_PER_VERIFIED = 6, 245_000
MB_PER_CALL = 260  # resident memory of one Codex call (measured: 7.3 GB for 28 concurrent calls)


def mem_available_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return 8000


def split(total, group_size=1):
    """Idea and development workers for `total` concurrent Codex calls (config.split), as operations settings."""
    from .config import split as shape
    value = shape(total, group_size)
    return {'pool': value['pool'], 'development_windows': value['development_windows']}


def memory_cap():
    """Concurrent Codex calls that fit in 80% of the free memory now."""
    return max(4, int(mem_available_mb() * 0.8 / MB_PER_CALL))


def concurrency(size, workers=None, windows=5, group_size=1, default=30):
    """Concurrent Codex calls (the profile's `workers`, or --workers), capped by free memory, and the idea windows open
    at once. Idea windows: enough for the idea workers, plus, while no development is ready (the first hour of a batch),
    enough 4-call windows to fill every slot: idea calls borrow idle development slots, and the pool stops new loans once
    development is ready."""
    cap = memory_cap()
    total = workers or min(default, cap)
    up = split(total, group_size)['pool']['upstream_workers']
    supply = max(2, min(windows, -(-up // 4) + 1))
    return {'workers': total, 'continuous_supply': True, 'supply_windows': supply,
            'idle_supply_windows': max(supply, min(windows, -(-total // 4)))}, total, cap


def run_root(name):
    return ROOT / 'runs' / name


def ops(name):
    path = ROOT / 'runs' / (name + '-ops'); path.mkdir(parents=True, exist_ok=True); return path


def child_env(**extra):
    """Environment for a pipeline subprocess: the checkout importable, the same workspace and bank."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get('PYTHONPATH')])),
               ARCGEN_WORKSPACE=str(ROOT), ARCGEN_BANK=str(BANK))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(value, indent=1) + '\n')


def say(msg):
    print(msg, flush=True)


def entry(*args, config=None, check_status=None, timeout=None, env=None):
    """Run entry.py; return (status, stdout). Raises SystemExit with the tail of the output on failure."""
    env = child_env(**({'ARCGEN_CONFIG': config} if config else {}), **(env or {}))
    p = subprocess.run([str(PY), '-B', str(HERE / 'entry.py'), *map(str, args)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=timeout)
    statuses = re.findall(r'"status": "([a-z_]+)"', p.stdout)
    status = statuses[-1] if statuses else None
    if p.returncode != 0 or (check_status and status not in check_status):
        tail = '\n'.join((p.stdout + p.stderr).strip().splitlines()[-12:])
        raise SystemExit(f'entry.py {args[0]} failed (status {status}):\n{tail}')
    return status, p.stdout


# ------------------------------------------------------------------ new
def scale_spec(spec, k):
    """k copies of the template's windows and calls; each copy keeps the template's window sizes and accounts."""
    if k == 1:
        return spec
    per_account = {}
    for wid, w in spec['windows'].items():
        per_account.setdefault(w['account'], []).append(wid)
    # Each account's copies are numbered and listed by (reference wave, copy, position), so every copy's wave-A windows
    # come before any wave-B window (the scheduler and allocation.design require wave order); rows keep copy order.
    rename, listing = {}, []
    for account, wids in per_account.items():
        keyed = sorted((spec['windows'][wid].get('reference_wave') or '', c, j, wid) for c in range(k) for j, wid in enumerate(wids))
        for n, (_, c, j, wid) in enumerate(keyed, 1):
            rename[(c, wid)] = new = f"{wid.rsplit('-', 1)[0]}-{n:02d}"
            listing.append((new, wid))
    windows = {new: dict(spec['windows'][wid]) for new, wid in listing}
    rows = [[rename[(c, r[0])], *r[1:]] for c in range(k) for r in spec['rows']]
    return {**spec, 'windows': windows, 'rows': rows}


def estimate(rows, windows, workers=30, saved_rows=0):
    """Time, verified tasks, model calls and tokens. A saved-only batch has no idea calls: its developments (saved_rows,
    an upper bound) take an ordinary development's hours and verify at about 0.85 per development."""
    shape = split(workers)['pool']
    waves = len({w.get('reference_wave') for w in windows.values()} - {None})
    up = len(rows) * UP_HOURS_PER_CALL / (shape['upstream_workers'] * BUSY) + WAVE_HOURS * max(0, waves - 1)
    down = (len(rows) * DOWN_HOURS_PER_CALL + saved_rows * DOWN_HOURS_PER_DEVELOPMENT) / (shape['downstream_workers'] * BUSY)
    verified = round(len(rows) * VERIFIED_PER_CALL + saved_rows * VERIFIED_PER_SAVED)
    return {'idea_calls': len(rows), 'expected_verified_tasks': verified,
            'hours': round(TAIL_HOURS + max(up, down), 1), 'workers': workers,
            'memory_gb': round(workers * MB_PER_CALL / 1000, 1),
            'model_calls': verified * CALLS_PER_VERIFIED, 'tokens_millions': round(verified * TOKENS_PER_VERIFIED / 1e6, 1),
            **({'saved_developments': saved_rows} if saved_rows else {})}


def saved_rows(spec):
    """Developments a spec's saved windows can make: every saved queue row and every saved idea (the selector keeps some
    of the saved ideas, so this is an upper bound)."""
    return (sum(len(read(ROOT / w['queue'])['rows']) for w in (spec.get('saved_windows') or {}).values())
            + sum(len(read(ROOT / w['ideas'])['source_rows']) for w in (spec.get('saved_idea_windows') or {}).values()))


def shape_problem(s, p, size):
    """Why `new` refuses this spec's shape, or None. With idea calls: at least one window of exactly 24 ideas and none
    wider (the selector's context budget is calibrated on 24-idea windows), every window on the configured account. A
    saved-only spec (no idea calls) needs neither; it takes --size 1 only."""
    if not s['rows']:
        if not (s.get('saved_windows') or s.get('saved_idea_windows')):
            return 'the spec has neither idea calls nor saved windows'
        return None if size == 1 else 'a saved-only spec takes --size 1 (saved windows are never repeated)'
    widths = Counter()  # ideas per window: 4 per call, or 6 for kept-draft commit windows
    for row in s['rows']:
        widths[row[0]] += 6 if s['windows'][row[0]].get('commit') in ('kept', 'explore') else 4
    if max(widths.values()) != 24:
        return ('the allocation needs at least one window of exactly 24 ideas (6 calls, or 4 kept-draft calls) and '
                f'none wider; got {dict(widths)}')
    from .config import ACCOUNT
    used = {w['account'] for w in s['windows'].values()}
    if used != {ACCOUNT}:
        return f"every window of the allocation runs on the one Codex account ('account': '{ACCOUNT}'); got {sorted(used)}"
    return None


def new(name, spec=TEMPLATE_SPEC, profile=TEMPLATE_PROFILE, description=None, size=1, workers=None, profiles=BATCHES,
        batches=BATCHES):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,60}', name):
        raise SystemExit('name: lowercase letters, digits and dashes (3-61 characters)')
    if not 1 <= size <= 10:
        raise SystemExit('--size must be between 1 and 10')
    batches.mkdir(parents=True, exist_ok=True)
    out_profile, out_spec, out_alloc = profiles / f'{name}.json', batches / f'{name}.spec.json', batches / f'{name}.allocation.json'
    if out_profile.exists() or run_root(name).exists():
        raise SystemExit(f'{name} already exists')
    p = read(profile); s = scale_spec(read(spec), size)
    problem = shape_problem(s, p, size)
    if problem:
        raise SystemExit(problem)
    saved = {**(s.get('saved_windows') or {}), **(s.get('saved_idea_windows') or {})}
    revision = 'arcgen-' + name
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    s.update(revision=revision, seed=seed)
    p.update(revision=revision, profile=name, default_run=f'runs/{name}',
             description=description or f'Batch {name}: created by bin/arcgen new from {profile.name} and {spec.name}.')
    p['science']['allocation'] = rel(out_alloc)
    # idea windows open at once are bounded by the batch's windows (or a saved-only spec's saved windows)
    count = len(s['windows']) or len(saved)
    ops, total, cap = concurrency(size, workers, count, p['operations']['group_size'], p['operations'].get('workers', 30))
    for key in ('pool', 'development_windows'):  # derived from workers (config.split)
        p['operations'].pop(key, None)
    p['operations'].update(ops)
    write(out_spec, s); write(out_profile, p)
    env = child_env(ARCGEN_CONFIG=out_profile)
    r = subprocess.run([str(PY), '-m', 'arcgen.batch.allocations.make_allocation', str(out_spec), str(out_alloc)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        for f in (out_spec, out_profile): f.unlink(missing_ok=True)
        raise SystemExit('allocation failed:\n' + (r.stdout + r.stderr)[-2000:])
    entry('show-config', '--config', out_profile, config=out_profile)
    shares = modifier_shares(out_profile)
    if 'error' in shares:
        for f in (out_spec, out_profile, out_alloc): f.unlink(missing_ok=True)
        raise SystemExit('the property slots cannot be assigned: ' + shares['error'])
    e = estimate(s['rows'], s['windows'], total, saved_rows=saved_rows(s) if not s['rows'] else 0)
    write(out_spec.with_name(f'{name}.estimate.json'), e)
    shape = split(total, p['operations']['group_size'])['pool']
    say(f'Created {name}: profile {rel(out_profile)}, allocation {rel(out_alloc)}.')
    say(f"  {e['idea_calls']} idea calls (size x{size})" + (f", {e['saved_developments']} saved developments at most"
                                                            if e.get('saved_developments') else '')
        + f"; about {e['expected_verified_tasks']} verified tasks, about {e['hours']} hours, about {e['model_calls']} model "
        f"calls and {e['tokens_millions']}M tokens (scaled from earlier batches; actual yield varies).")
    say(f"  Up to {total} concurrent Codex calls ({shape['upstream_workers']} idea, {shape['downstream_workers']} development; "
        f"about {e['memory_gb']} GB). Free memory allows about {cap} now"
        + ('' if workers or p['operations'].get('workers', 30) <= cap else '; more free memory would let it run faster') + '.')
    if shares.get('counts'):
        say('  Property slots (calls of ' + str(shares['calls']) + '): ' + ', '.join(f'{m} {n}' for m, n in shares['counts'].items())
            + f"; no property on {shares['bare']} calls.")
    say(f'Start it with: bin/arcgen start {name}')
    return out_profile


def modifier_shares(profile):
    """The batch design's property-slot counts, computed as prepare will (allocation.design); an unmeetable share or
    reservation comes back as {'error': ...} so `new` can refuse the batch before anything runs."""
    code = ('import json\nfrom arcgen.batch.allocation import design\nfrom arcgen.batch.config import SCIENCE\n'
            'calls = [c for w in design()["windows"] if w["kind"] == "fresh_supply" for c in w["calls"]]\n'
            'print(json.dumps({"calls": len(calls), "bare": sum(not c.get("idea_modifiers") for c in calls),'
            ' "counts": {m: sum(m in c.get("idea_modifiers", ()) for c in calls) for m in SCIENCE.get("idea_modifiers") or {}}}))')
    r = subprocess.run([str(PY), '-B', '-c', code], cwd=REPO, env=child_env(ARCGEN_CONFIG=profile), capture_output=True, text=True)
    if r.returncode != 0:
        return {'error': (r.stderr.strip().splitlines() or ['design failed'])[-1]}
    return json.loads(r.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------ start / supervise
def code_hash(config):
    return subprocess.run([str(PY), '-B', '-c', 'from arcgen.batch.common import code_identity; print(code_identity())'],
                          cwd=REPO, env=child_env(ARCGEN_CONFIG=config), capture_output=True, text=True, check=True).stdout.strip()


def gate_is_current(config):
    from arcgen.paths import GATE
    gate = read(GATE / 'offline_gate.json', {})
    return gate.get('status') == 'passed' and gate.get('code_hash') == code_hash(config)


def discard_stale_preparation(root, code):
    """An unlaunched preparation is qualified and readied against the code of the moment; after a code change those
    frozen records no longer match and ready refuses the folder. Discard it so start prepares again. A launched run
    (launch_manifest present) is never touched. Returns True when the folder was removed."""
    if not root.exists() or (root / 'launch_manifest.private.json').exists():
        return False
    hashes = {(read(root / f) or {}).get('code_hash') for f in ('prepared_launch_manifest.private.json',
              'author_qualification.private.json', 'selector_qualification.private.json')} - {None}
    if not hashes or hashes == {code}:
        return False
    import shutil
    shutil.rmtree(root)
    return True


def start(name, note=None, skip_tests=False):
    profile = BATCHES / f'{name}.json'; root = run_root(name); o = ops(name)
    if not profile.exists():
        raise SystemExit(f'No profile for {name}; create it with: bin/arcgen new {name}')
    if alive(name):
        raise SystemExit(f'{name} is already running (bin/arcgen progress {name})')
    est = read(BATCHES / f'{name}.estimate.json') or {}
    if est.get('memory_gb') and est['memory_gb'] * 1000 > mem_available_mb():
        say(f"Note: this batch can use about {est['memory_gb']} GB at peak and {mem_available_mb() / 1000:.1f} GB is free now; "
            'close heavy apps or lower the workers (bin/arcgen workers).')
    gate = {}
    if skip_tests:
        say('Skipping the offline test gate (--skip-tests); the launch manifest records that it was skipped.')
        gate = {'ARCGEN_SKIP_TEST_GATE': '1'}
    elif not gate_is_current(profile):
        say('Code changed since the last test gate: running the offline test suite (a few minutes)...')
        entry('test', config=profile, check_status=('passed',))
    if discard_stale_preparation(root, code_hash(profile)):
        say('The earlier unlaunched preparation was made with older code; discarded it to prepare again.')
    if not (root / 'launch_manifest.private.json').exists():
        say('Preparing (frozen inputs, bank snapshot, reference embeddings, design)...')
        entry('prepare', '--config', profile, '--root', root, check_status=('prepared_not_launched',), env=gate)
        say('Qualifying and readying...')
        entry('ready', '--root', root, check_status=('ready_not_launched',), env=gate)
        command = 'bin/arcgen start ' + ' '.join(shlex.quote(a) for a in sys.argv[2:])
        authorization = f'Owner launch via `{command}` on {time.strftime("%Y-%m-%d %H:%M")}.' + (f' Note: {note}' if note else '')
        say('Authorizing (records the launch instruction and a usage reading)...')
        entry('authorize', '--root', root, '--authorization', authorization, check_status=('authorized',), env=gate)
        quota = read(root / 'capacity_quota_preflight.json')
        if quota:
            write(o / 'quota_before.json', quota)
            for a, row in quota['accounts'].items():
                if row.get('weekly_remaining_percent') is not None and row['weekly_remaining_percent'] < 20 \
                        and not (read(profile) or {}).get('operations', {}).get('usage_credits'):
                    say(f"Note: the account has {row['weekly_remaining_percent']}% of its weekly limit left; the run pauses "
                        f'itself at the limit (status paused_usage) and continues with bin/arcgen resume {name} after the reset.')
    launch_supervisor(name)
    say(f'{name} is running in the background. Watch it with: bin/arcgen progress {name} --watch')


def launch_supervisor(name):
    o = ops(name)
    with open(o / 'supervisor.log', 'a') as log:
        p = subprocess.Popen([str(PY), '-B', '-m', 'arcgen.batch.cli', '_supervise', name], cwd=REPO, stdout=log, env=child_env(),
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    write(o / 'supervisor.json', {'pid': p.pid, 'started_at': time.time()})


def alive(name):
    info = read(ops(name) / 'supervisor.json')
    if not info:
        return False
    try:
        os.kill(info['pid'], 0); return True
    except OSError:
        return False


AUTO_RESUMES = 3  # unexpected exits (killed by the out-of-memory killer, a signal, a crash) resumed at most this often


def run_command(root):
    """The pipeline run, kept awake while the laptop is open. systemd-inhibit blocks idle and requested sleep; GNOME's
    power manager ignores that lock for its idle auto-suspend (on battery after 15 minutes: v1-b9 slept 30 minutes on
    2026-10-05), so the run is also wrapped in a GNOME session inhibitor when a desktop session is available.
    Closing the lid still suspends (no handle-lid-switch lock; logind's LidSwitchIgnoreInhibited defaults to yes), so
    the laptop can go in a bag: the run pauses while suspended and continues on wake; calls cut off by the suspension
    fail and are retried like any failed call."""
    cmd = [str(PY), '-B', str(HERE / 'entry.py'), 'run', '--root', str(root)]
    why = f'arcgen batch {root.name} is running'
    gnome = shutil.which('gnome-session-inhibit')
    if gnome and os.environ.get('DBUS_SESSION_BUS_ADDRESS'):
        if subprocess.run([gnome, '--inhibit', 'suspend:idle', '--reason', why, 'true'], capture_output=True, timeout=20).returncode == 0:
            cmd = [gnome, '--inhibit', 'suspend:idle', '--reason', why] + cmd
    inhibit = shutil.which('systemd-inhibit')
    if inhibit and subprocess.run([inhibit, '--what=sleep:idle', '--mode=block', 'true'], capture_output=True).returncode == 0:
        return [inhibit, '--what=sleep:idle', '--who=arcgen', f'--why={why}', '--mode=block'] + cmd
    return cmd


def resume_decision(status, stop_reason, attempts, usage_reasons=()):
    """What the supervisor does when the run process exits.

    summary: the run finished. stop: a pause (yours, or the pipeline's own stop for a problem it found) or the weekly
    usage limit; these wait for you. resume: an unexpected exit, or a usage pause caused only by an unavailable usage
    reading (not a limit), resumed from saved state. give_up: too many automatic resumes.
    """
    if status in TERMINAL:
        return 'summary'
    transient = status == 'paused_usage' and usage_reasons and set(usage_reasons) == {'quota_observation_unavailable'}
    if stop_reason and 'database is locked' in stop_reason:  # a passing infrastructure stop, not a finding
        stop_reason = None
    if stop_reason is not None or (status == 'paused_usage' and not transient):
        return 'stop'
    return 'resume' if attempts < AUTO_RESUMES else 'give_up'


def supervise(name):
    """Run the pipeline to completion (kept awake, resumed after unexpected exits), log progress, then summarise."""
    root = run_root(name); o = ops(name)
    started = read(o / 'started.json') or {'first_started_at': time.time()}
    started['last_started_at'] = time.time()
    started['log_offset'] = (o / 'run.log').stat().st_size if (o / 'run.log').exists() else 0  # errors are reported per session
    write(o / 'started.json', started)
    env = child_env(ARCGEN_CONFIG=root / 'config.json', ARCGEN_RUN_ROOT=root)
    note = lambda text: open(o / 'progress.log', 'a').write(time.strftime('%Y-%m-%d %H:%M ') + text + '\n')
    attempts = 0
    while True:
        with open(o / 'run.log', 'a') as log:
            # its own process group, so leftover Codex calls of a killed run can be cleaned up before resuming
            p = subprocess.Popen(run_command(root), cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            write(o / 'run_process.json', {'pid': p.pid})
            last = 0
            while p.poll() is None:
                if time.time() - last >= PROGRESS_EVERY:
                    note(one_line(progress(name))); last = time.time()
                time.sleep(15)
        try:
            os.killpg(p.pid, signal.SIGTERM)  # orphaned calls of the exited run, if any
        except OSError:
            pass
        final = progress(name)
        note(f'run process exited ({p.returncode}): ' + one_line(final))
        stop_file = root / 'STOP_DISPATCH.json'
        usage_reasons = [(read(f) or {}).get('reason') for f in root.glob('PAUSED_USAGE_*.json')]
        decision = resume_decision(final['status'], (read(stop_file) or {}).get('reason', '') if stop_file.exists() else None, attempts,
                                   usage_reasons)
        if decision == 'resume':
            attempts += 1; wait = 60 * 2 ** (attempts - 1)
            note(f'unexpected exit: automatic resume {attempts}/{AUTO_RESUMES} in {wait} s')
            log_auto = read(o / 'auto_resumes.json', []); log_auto.append({'at': time.time(), 'returncode': p.returncode, 'status': final['status']})
            write(o / 'auto_resumes.json', log_auto)
            time.sleep(wait)
            continue
        if decision == 'give_up':
            note(f'stopped after {AUTO_RESUMES} automatic resumes; inspect run.log, then bin/arcgen resume {name}')
        elif decision == 'stop':
            note('waiting for you: ' + ('weekly usage limit reached; bin/arcgen resume after the reset' if final['status'] == 'paused_usage'
                                         else 'paused or stopped; see bin/arcgen progress, then bin/arcgen resume'))
        else:
            started['finished_at'] = time.time(); write(o / 'started.json', started)  # progress stops its clock here
            try:
                write(o / 'quota_after.json', quota_reading(root))
            except Exception as exc:  # a missing reading never blocks the summary
                write(o / 'quota_after.error.json', {'error': repr(exc)})
            summary(name, write_files=True)
        return


def quota_reading(root):
    code = ("import json,sys; from pathlib import Path; from arcgen.batch.quota import Quota; from arcgen.batch.common import read, HOMES\n"
            "root=Path(sys.argv[1]); t=read(root/'transport_preflight.json')['astra']; scratch=Path(sys.argv[2])\n"
            "print(json.dumps({'status':'observed','accounts':{a: Quota(scratch, t[a]['executable']['path'], account=a).observe(force=True) for a in sorted(HOMES)}}))")
    scratch = ops(root.name) / 'quota_scratch'; scratch.mkdir(exist_ok=True)
    env = child_env(ARCGEN_CONFIG=root / 'config.json', ARCGEN_RUN_ROOT=root)
    out = subprocess.run([str(PY), '-B', '-c', code, str(root), str(scratch)], cwd=REPO, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------ progress
def tokens(root):
    out = Counter()
    for p in glob.glob(str(root / 'accounts/*/**/transport_observations/response-*.stderr.log'), recursive=True):
        m = re.findall(r'tokens used\s*\n\s*([\d,]+)', Path(p).read_text(errors='replace'))
        if m:
            out[Path(p).relative_to(root).parts[1]] += int(m[-1].replace(',', ''))
    return dict(out)


def progress(name):
    root = run_root(name); o = ops(name)
    status = (read(root / 'STATUS.json') or {}).get('status', 'not_prepared')
    design = read(root / 'design/batch.private.json') or {'windows': []}
    planned = sum(len(w.get('calls', [])) for w in design['windows'])
    calls = {}
    for line in (o / 'run.log').read_text(errors='replace').splitlines() if (o / 'run.log').exists() else []:
        if line.startswith('{"account"'):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get('event') == 'call':
                calls[e['call_id']] = (e['state'], e['purpose'], e.get('at_epoch', 0))
    idea_done = sum(1 for cid, (st, pu, _) in calls.items() if cid.endswith('-commit') and pu.startswith('idea') and st == 'complete')
    queues = [read(p) for p in glob.glob(str(root / 'accounts/*/*/eligible_queue.private.json'))]
    queued = ('fresh_supply', 'saved_ideas', 'saved_queue')  # windows that freeze an eligible queue
    supply = sum(1 for w in design['windows'] if w.get('kind') in queued)
    wave_rows = {}  # reference waves: idea windows of a later wave wait for every earlier window's selection
    for w in design['windows']:
        if w.get('kind') in queued and w.get('reference_wave'):
            done = (root / 'accounts' / w['account'] / w['id'] / 'eligible_queue.private.json').exists()
            row = wave_rows.setdefault(w['reference_wave'], [0, 0]); row[0] += done; row[1] += 1
    records, finished = {}, set()
    for p in glob.glob(str(root / 'accounts/*/*-develop-*/downstream/state.json')):
        group = (read(p) or {}).get('semantic_records', {}); records.update(group)
        if (Path(p).parent / 'window_terminal.private.json').exists():  # every task of a finished group has ended
            finished.update(group)
    verified = sum(r.get('pipeline_usable_before_union_audit') is True for r in records.values())
    ended = sum((tid in finished or bool(((r.get('answer_evidence') or {}).get('difficulty_outcome'))) or r.get('status') in
                 ('realization_failure', 'rejected_technical', 'planning_rejected', 'plan_aborted', 'planning_operational_failure',
                  'realization_operational_failure', 'evidence_hold', 'serious_ambiguity'))
                and r.get('pipeline_usable_before_union_audit') is not True for tid, r in records.items())
    started = read(o / 'started.json') or {}
    since = started.get('last_started_at', 0)  # calls cut off by an earlier stop never log an end
    # a finished run's clock stops when it ended (the supervisor records it; older runs: their completion receipt)
    end_at = (started.get('finished_at') or (read(root / 'completion.json') or {}).get('completed_at_epoch')) if status in TERMINAL else None
    # run.log says 'started' before a call has a slot, so it counts queued calls too (v1-b14 showed 64 of 48 slots); the
    # live worker pools tell running (holding a slot) from queued (waiting for one)
    live = alive(name)
    pools = [read(p) or {} for p in glob.glob(str(root / 'accounts/*/worker_pool.json'))] if live else []
    errors = []
    if (o / 'run.log').exists():
        with open(o / 'run.log', errors='replace') as f:
            f.seek(started.get('log_offset', 0))
            errors = [l[:200] for l in f.read().splitlines()
                      if re.search(r'Traceback|IntegrityError|Error:|"state": "failed"', l)][-3:]
    return {'name': name, 'status': status, 'running': live,
            'paused': (root / 'STOP_DISPATCH.json').exists(),
            'elapsed_minutes': round(((end_at or time.time()) - started['first_started_at']) / 60) if started else None,
            'idea_calls': [idea_done, planned], 'selector_windows': [len(queues), supply],
            'eligible_ideas': sum(len(q.get('rows', [])) for q in queues if q),
            'developments': len(records), 'realized': sum(bool(r.get('complete_task')) for r in records.values()),
            'verified': verified, 'held_or_failed': ended, 'in_development': len(records) - verified - ended,
            'calls_in_flight': sum(1 for st, _, at in calls.values() if st == 'started' and at >= since),
            'calls_running': sum(len(p.get('active_calls') or {}) for p in pools),
            'calls_queued': sum(sum((p.get('waiting_by_category') or {}).values()) for p in pools),
            'waves': dict(sorted(wave_rows.items())),
            'tokens_by_account': tokens(root) if records or calls else {}, 'recent_errors': errors}


def one_line(p):
    tok = sum(p['tokens_by_account'].values()) / 1e6 if p['tokens_by_account'] else 0
    return (f"{p['status']}{' (paused)' if p['paused'] else ''} | ideas {p['idea_calls'][0]}/{p['idea_calls'][1]} calls | "
            f"selector {p['selector_windows'][0]}/{p['selector_windows'][1]} windows, {p['eligible_ideas']} eligible | "
            f"developing {p['developments']}: {p['verified']} verified, {p['held_or_failed']} held/failed, {p['in_development']} in progress | "
            f"{p['calls_running']} calls running, {p['calls_queued']} queued | {tok:.1f}M tokens" + _wave_note(p.get('waves') or {}))


def _wave_note(waves):
    """While an earlier reference wave is still selecting, the next wave's idea windows wait: say so, not 'stuck'."""
    for wave, (done, total) in waves.items():
        if done < total:
            later = [w for w in waves if w > wave]
            return f" | wave {wave} {done}/{total} windows selected" + (f"; wave {later[0]} starts after" if later else '')
    return ''


def show_progress(name, watch=None):
    while True:
        p = progress(name)
        say(time.strftime('%H:%M:%S ') + one_line(p))
        if p['elapsed_minutes'] is not None:
            say(f"  elapsed {p['elapsed_minutes']} min; supervisor {'running' if p['running'] else 'not running'}; "
                f"tokens by account {', '.join(f'{a} {t / 1e6:.1f}M' for a, t in sorted(p['tokens_by_account'].items())) or '-'}")
        for e in p['recent_errors']:
            say('  recent error: ' + e)
        if not watch or (not p['running'] and p['status'] in TERMINAL + ('paused_usage',)) or (not p['running'] and p['paused']):
            if p['status'] in TERMINAL and (ops(name) / 'SUMMARY.md').exists():
                say(f'Finished. Summary: runs/{name}-ops/SUMMARY.md; gallery: runs/{name}/gallery/index.html')
            return p
        time.sleep(watch)


# ------------------------------------------------------------------ stop / resume
def run_pids(root):
    """Python processes running this batch (entry.py run --root <root>)."""
    out = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = [a.decode(errors='replace') for a in (p / 'cmdline').read_bytes().split(b'\0')]
        except OSError:
            continue
        if any(a.endswith('entry.py') for a in argv) and 'run' in argv and str(root) in argv:
            out.append(int(p.name))
    return out


def stop(name, now=False):
    root = run_root(name)
    if now:
        # Order matters. The supervisor goes first so it does not auto-resume; the Python run process is then killed
        # outright, so it cannot record the calls it is about to lose as final failures (stopping the whole group at
        # once let it outlive its Codex calls and write ~20 v1-b8 tasks off on 2026-10-05); the calls go last and are
        # redone on resume.
        info = read(ops(name) / 'supervisor.json')
        if info:
            try:
                os.killpg(os.getpgid(info['pid']), signal.SIGTERM)
            except OSError:
                pass
        pids = run_pids(root)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(1)
        # The run process leads the process group of its Codex calls, which outlive it otherwise (v1-b16-t and -g,
        # 2026-10-14: 19 calls kept running after `stop --now`).
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                pass
        info = read(ops(name) / 'run_process.json')
        if info:
            try:
                os.killpg(os.getpgid(info['pid']), signal.SIGTERM)
            except OSError:
                pass
        say(f'{name} stopped immediately; saved progress is kept. Continue with: bin/arcgen resume {name}')
        return
    write(root / 'STOP_DISPATCH.json', {'reason': 'owner pause via bin/arcgen stop', 'at': time.time()})
    say(f'{name} is pausing: no new calls start, running calls finish (minutes). Continue with: bin/arcgen resume {name}')


def resume(name):
    root = run_root(name)
    if alive(name):
        raise SystemExit(f'{name} is still running; wait for the pause to finish (bin/arcgen progress {name})')
    if (read(root / 'STATUS.json') or {}).get('status') in TERMINAL:
        raise SystemExit(f'{name} has already finished; nothing to resume. Commit it with: bin/arcgen commit {name}')
    stop_file = root / 'STOP_DISPATCH.json'
    if stop_file.exists():
        write(ops(name) / f'resolved_stop_{int(time.time())}.json', read(stop_file)); stop_file.unlink()
    launch_supervisor(name)
    say(f'{name} resumed from saved state. Watch it with: bin/arcgen progress {name} --watch')


# ------------------------------------------------------------------ summary
def palette_varies(task):
    return len({frozenset(v for row in p['input'] for v in row) for p in task['train'] + task['test']}) > 1


def _properties(verified, rungs=None):
    """Achieved properties of the verified tasks (properties.py); a summary never fails on them."""
    try:
        from . import properties
        return properties.report(verified, rungs)
    except Exception as exc:  # noqa: BLE001 - the summary must still be written
        return {'error': repr(exc)[:300]} if verified else None


def summary(name, write_files=False):
    root = run_root(name); o = ops(name)
    report = read(root / 'analysis/terminal_report.private.json')
    if not report:
        raise SystemExit(f'{name} has no terminal report yet (bin/arcgen progress {name})')
    tasks = report['tasks']; ok = [t for t in tasks if t['usable_after_exact_palette_union']]
    records = {}
    for p in glob.glob(str(root / 'accounts/*/*-develop-*/downstream/state.json')):
        records.update(read(p)['semantic_records'])
    verified = [records[t['task_id']] for t in ok]
    before, after = read(o / 'quota_before.json'), read(o / 'quota_after.json')
    usage = {}
    if before and after:
        for a in before['accounts']:
            usage[a] = {'weekly_used_before': before['accounts'][a]['weekly_used_percent'], 'weekly_used_after': after['accounts'][a]['weekly_used_percent']}
    tok = tokens(root)
    held = Counter(((records[t['task_id']].get('answer_evidence') or {}).get('difficulty_outcome') or {}).get('reason') or t['status']
                   for t in tasks if not t['usable_after_exact_palette_union'])
    value = {'name': name, 'status': (read(root / 'STATUS.json') or {}).get('status'), 'developments': len(tasks),
             'verified_tasks': len(ok), 'not_verified': dict(held),
             'by_arm': dict(Counter(t['arm'] for t in ok)), 'by_idea_source': dict(Counter(f"{t['generation_model']} {t['effort']}" for t in ok)),
             'colour_instruction': dict(Counter(r['study'].get('colour_roles') or 'standard' for r in verified)),
             'colours_vary_across_examples': sum(palette_varies(r['complete_task']) for r in verified),
             'repairs': dict(Counter(int(r.get('repair_count') or 0) for r in verified)),
             'tokens_by_account': tok, 'tokens_total': sum(tok.values()), 'weekly_usage': usage,
             'properties': _properties(verified, ((read(root / 'config.json') or {}).get('science') or {}).get('rungs')),
             'gallery': f'runs/{name}/gallery/index.html', 'commit_with': f'bin/arcgen commit {name}'}
    checks = [read(p) for p in glob.glob(str(root / 'accounts/*/*-develop-*/downstream/rung_check/*.json'))]
    if checks:  # scale rungs (2026-10-12): verified tasks by family and rung, and the plan-stage floor check
        value['by_family_rung'] = dict(sorted(Counter(f"{t.get('family') or 'general'} {t['rung']}" for t in ok if t.get('rung')).items()))
        value['rung_plan_checks'] = {'plans_checked': len(checks), 'replanned': sum(c['replanned'] for c in checks),
                                     'replan_kept': sum(c['used'] == 'second' for c in checks),
                                     'reached_planned_rung': sum(bool(c['reached']) and c['reached'] >= c['rung'] for c in checks)}
        from statistics import median
        words = {}  # convention load: the plan's central rule, in words (screen bar: median 45 or fewer)
        for r in verified:
            if r['study'].get('rung'):
                rule = r['concrete_plan']['public_contract']['semantic_core'].get('central_rule') or ''
                words.setdefault(f"{r['study'].get('family') or 'general'} {r['study']['rung']}", []).append(len(rule.split()))
        value['central_rule_words_median'] = {k: median(v) for k, v in sorted(words.items())}
    reading = read_first(ok)
    value['blind_rule_words_median'] = reading['words_median']
    value['read_first'] = reading['flagged']
    if write_files:
        write(o / 'READ_FIRST.json', reading['flagged'])
        write(o / 'SUMMARY.json', value)
        lines = [f'# {name}', '', f"Status **{value['status']}**: {value['verified_tasks']} verified tasks from {value['developments']} developments.", '',
                 f"- Not verified: {value['not_verified'] or 'none'}",
                 f"- By arm: {value['by_arm']}", f"- By idea source: {value['by_idea_source']}",
                 f"- Colour instruction: {value['colour_instruction']}; colours vary across examples in {value['colours_vary_across_examples']} tasks",
                 f"- Repairs (verified tasks): {value['repairs']}",
                 f"- Tokens: {value['tokens_total'] / 1e6:.1f}M ({', '.join(f'{a} {t / 1e6:.1f}M' for a, t in sorted(tok.items()))})"]
        for a, u in usage.items():
            lines.append(f"- {a} weekly limit: {u['weekly_used_before']}% -> {u['weekly_used_after']}%")
        if 'rung_plan_checks' in value:
            c = value['rung_plan_checks']
            lines += [f"- By family and rung (verified): {value['by_family_rung']}",
                      f"- Plan-stage scale check: {c['plans_checked']} plans checked, {c['replanned']} re-planned "
                      f"({c['replan_kept']} re-plans kept), {c['reached_planned_rung']} planned at their rung",
                      f"- Central rule, median words: {value['central_rule_words_median']}"]
        if value['properties'] and 'error' not in value['properties']:
            from . import properties
            lines += properties.lines(value['properties'])
        why = Counter(r.split(':')[0] for r in value['read_first'].values())
        lines += [f"- Blind solver's inferred rule, median words: {value['blind_rule_words_median']}",
                  f"- To read before commit: {len(value['read_first'])} ({dict(why) or 'none'}), listed in runs/{name}-ops/READ_FIRST.json; "
                  f"hold any that fail with `bin/arcgen commit {name} --hold <file>`"]
        lines += ['', f"Gallery: {value['gallery']}", f"Commit to the bank (stage_3): `{value['commit_with']}`"]
        (o / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    return value


READ_RULE_WORDS = 80  # 2026-10-13 screen: human "no" tasks had a median of 86.5 words in the blind solver's inferred rule


def _blind_rule(row):
    """The primary blind solver's inferred rule for a task's final episode (candidates/<task>/blind.rK.json, latest K)."""
    state = (row.get('state') or {}).get('path')
    if not state:
        return ''
    folder = Path(state).parents[1] / 'candidates' / row['task_id']
    files = sorted(folder.glob('blind.r*.json'), key=lambda p: int(re.sub(r'\D', '', p.stem) or 0))
    blind = read(files[-1]) if files else None
    return ((blind or [{}])[0].get('response') or {}).get('primary_inferred_rule') or ''


def read_first(rows):
    """Tasks to read before commit (2026-10-13): no stage checks
    human solvability, and a blind failure never rejects. The screen's human "no" tasks were caught by a blind failure or a
    blind-inferred rule over READ_RULE_WORDS words (12 of 18), so those are flagged for a reader, never rejected."""
    from statistics import median
    flagged, words = {}, {}
    for t in rows:
        rule = _blind_rule(t); n = len(rule.split())
        words.setdefault(f"{t.get('family') or 'general'} {t.get('rung') or '-'}", []).append(n)
        if (t.get('blind_performance') or {}).get('any_all_test_success') is False:
            flagged[t['task_id']] = 'blind failed: the blind solver missed every test'
        elif n > READ_RULE_WORDS:
            flagged[t['task_id']] = f'long rule: the blind solver needed {n} words'
    return {'flagged': flagged, 'words_median': {k: median(v) for k, v in sorted(words.items())}}


# ------------------------------------------------------------------ workers
def set_workers(name, total):
    """Record a new number of concurrent Codex calls for a batch; a started batch picks it up when it is next resumed."""
    root = run_root(name)
    if not (root / 'config.json').exists():
        profile = BATCHES / f'{name}.json'
        value = read(profile)
        if value is None:
            raise SystemExit(f'No batch {name}')
        value['operations']['workers'] = total; write(profile, value)
        say(f'{name}: {total} concurrent Codex calls (recorded in its profile; takes effect at start).')
        return
    cap = memory_cap()
    if total > cap:
        say(f'Warning: free memory allows about {cap} concurrent Codex calls now; {total} may swap.')
    entry('amend', '--root', root, '--set', f'workers={total}', '--reason', f'bin/arcgen workers {name} {total}',
          config=root / 'config.json', check_status=('amended',))
    shape = split(total, read(root / 'config.json')['operations']['group_size'])
    say(f"{name}: {shape['pool']['upstream_workers']} idea and {shape['pool']['downstream_workers']} development workers, "
        f"{shape['development_windows']} development windows.")
    if alive(name):
        say(f'It takes effect at the next resume: `bin/arcgen stop {name}` (lets running calls finish), then `bin/arcgen resume {name}`.')


# ------------------------------------------------------------------ commit / list
def commit(name, stage='stage_3', note=None, dry_run=False, partial=False, hold=None, release_judge=None):
    if not isinstance(name, str):  # bin/arcgen commit A B C: one transaction; a single name is the usual commit
        names = list(name)
        if len(names) > 1:
            return commit_many(names, stage, note, dry_run, partial, hold, release_judge)
        name, = names
    root = run_root(name)
    status = (read(root / 'STATUS.json') or {}).get('status')
    if partial:  # a stopped run that will not resume: only verified tasks of its finished windows
        if status != 'stopped':
            raise SystemExit(f'{name} is {status}; --partial takes only a stopped run')
        process = (read(ops(name) / 'run_process.json') or {}).get('pid')
        if alive(name) or run_pids(root) or process and Path(f'/proc/{process}').exists():
            raise SystemExit(f'{name} is still running; stop it first (bin/arcgen stop {name})')
    elif status != 'verified_terminal':
        raise SystemExit(f'{name} is not finished and verified yet')
    held = None
    if hold:  # tasks held back by the owner (readers' verdicts): copied beside the run's ops files, which the commit pins
        held = ops(name) / 'held.json'
        value = read(Path(hold))
        if value is None:
            raise SystemExit(f'{hold} is not readable JSON')
        write(held, value)
    released = None
    if release_judge:  # weakness runs: JUDGE tasks that passed the offline consensus reads (otherwise held, as decoys are)
        released = ops(name) / 'judge_released.json'
        value = read(Path(release_judge))
        if value is None:
            raise SystemExit(f'{release_judge} is not readable JSON')
        write(released, value)
    commit_name = f"{name}-{'partial-' if partial else ''}task-commit-{time.strftime('%Y%m%d')}"
    staged = read(ROOT / 'runs' / commit_name / 'prepared.json')
    if staged and not (ROOT / 'runs' / commit_name / 'transaction.json').exists():  # never touch a commit in progress
        bank = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in BANK.glob('*.json')}
        pinned = (read(ROOT / 'runs' / commit_name / 'source_manifest.json') or {}).get('files', {})
        moved = [rel for rel, h in pinned.items() if not (ROOT / rel).is_file() or hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != h]
        stacked = (staged.get('transaction') or {}).get('after') or (staged.get('transaction') or {}).get('defer_sheet')
        if staged.get('old_bank_json_hashes') != bank or moved or staged.get('format') != COMMIT_FORMAT or stacked:
            # another commit landed, a pinned source changed, the previous commit code staged it, or it was staged as part
            # of a several-run transaction
            shutil.rmtree(ROOT / 'runs' / commit_name)
            say(('The bank' if staged.get('old_bank_json_hashes') != bank else f'{moved[0]} (and {len(moved) - 1} more)' if len(moved) > 1
                 else moved[0] if moved else 'The commit code or transaction') + ' changed since this batch was staged; staging it again.')
    command = 'bin/arcgen commit ' + ' '.join(shlex.quote(a) for a in sys.argv[2:])
    authorization = f'Owner commit via `{command}` on {time.strftime("%Y-%m-%d %H:%M")}; project stage {stage}.' + (f' Note: {note}' if note else '')
    env = child_env(ARCGEN_CONFIG=root / 'config.json', ARCGEN_RUN_ROOT=root)
    args = ([str(PY), '-B', '-m', 'arcgen.batch.commit', '--root', str(root), '--name', commit_name, '--authorization', authorization,
             '--stage-flag', stage] + (['--partial'] if partial else []) + (['--hold', str(held)] if held else [])
            + (['--release-judge', str(released)] if released else []) + ([] if dry_run else ['--commit']))
    say(('Preparing and verifying' if dry_run else 'Committing') + f' {name} into the bank as {commit_name} (project_stage {stage})...')
    from .bank_inputs import ensure_bank
    ensure_bank(BANK)
    r = subprocess.run(args, cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit('commit failed; the bank is unchanged or rolled back (see the output above)')
    if not dry_run:
        freed = prune_commit_staging(ROOT / 'runs' / commit_name)
        if freed:
            say(f'Removed the commit\'s working copies (staged bank and pre-commit backup, {freed / 1e9:.1f} GB); the bank and receipts stay.')
    say(f'Done. Receipts: runs/{commit_name}/' + ('' if dry_run else ' (report.md)'))


def prune_commit_staging(folder):
    """After a commit is recorded complete, its staged bank (a second copy of what was installed into committed_tasks)
    and pre-commit backup (used only to undo a failed install) are no longer used; each batch left about 3.4 GB of them.
    Receipts (proof, manifests, report) stay. Returns the bytes freed; does nothing unless the commit is complete.
    Since 2026-10-24 most staged files are hard links shared with the bank or the run; only files with no other link
    count as freed."""
    if (read(Path(folder) / 'transaction.json') or {}).get('status') != 'complete':
        return 0
    freed = 0
    for sub in ('staged_bank', 'backup'):
        path = Path(folder) / sub
        if path.is_dir():
            freed += sum(st.st_size for st in (f.stat() for f in path.rglob('*') if f.is_file()) if st.st_nlink == 1)
            shutil.rmtree(path)
    return freed


COMMIT_FORMAT = 2  # arcgen.batch.commit.FORMAT: prepared.json of the current commit code


def _hold_ids(value):
    """Task IDs of a hold file (commit.read_holds: a {task_id: reason} map or a list of IDs or {"task_id"} objects)."""
    return set(value) if isinstance(value, dict) else {x if isinstance(x, str) else x['task_id'] for x in value}


def commit_many(names, stage='stage_3', note=None, dry_run=False, partial=False, hold=None, release_judge=None):
    """Several runs committed as one bank transaction (2026-10-24): each run is prepared and verified in its own process
    and config exactly as a single commit, stacked on the previous run's staged bank; then one install puts all of them in
    under one lock with one backup, writes the bank-wide files and renders once and verifies the final bank once. Every run
    keeps its own commit set, manifest and receipts (runs/<name>-task-commit-<date>/). Nothing is installed unless every
    run prepares and verifies; a failed install is rolled back."""
    if len(set(names)) != len(names):
        raise SystemExit('a run is named twice')
    roots = {}
    for name in names:
        root = run_root(name)
        status = (read(root / 'STATUS.json') or {}).get('status')
        if partial:
            if status != 'stopped':
                raise SystemExit(f'{name} is {status}; --partial takes only a stopped run')
            process = (read(ops(name) / 'run_process.json') or {}).get('pid')
            if alive(name) or run_pids(root) or process and Path(f'/proc/{process}').exists():
                raise SystemExit(f'{name} is still running; stop it first (bin/arcgen stop {name})')
        elif status != 'verified_terminal':
            raise SystemExit(f'{name} is not finished and verified yet')
        roots[name] = root
    hold_ids = set()
    if hold:  # one file for the transaction: each run takes its own tasks; every listed task must be in one of the runs
        value = read(Path(hold))
        if value is None:
            raise SystemExit(f'{hold} is not readable JSON')
        hold_ids = _hold_ids(value)
        for name in names:
            write(ops(name) / 'held.json', value)
    if release_judge:
        value = read(Path(release_judge))
        if value is None:
            raise SystemExit(f'{release_judge} is not readable JSON')
        for name in names:
            write(ops(name) / 'judge_released.json', value)
    date = time.strftime('%Y%m%d')
    commit_names = [f"{n}-{'partial-' if partial else ''}task-commit-{date}" for n in names]
    folders = [ROOT / 'runs' / c for c in commit_names]
    if not any((f / 'transaction.json').exists() for f in folders):  # never touch a transaction in progress
        bank = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in BANK.glob('*.json')}
        stale = False
        for i, folder in enumerate(folders):
            staged = read(folder / 'prepared.json')
            if staged and not stale:
                t = staged.get('transaction') or {}
                pinned = (read(folder / 'source_manifest.json') or {}).get('files', {})
                moved = [rel for rel, h in pinned.items() if not (ROOT / rel).is_file() or hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != h]
                stale = (staged.get('format') != COMMIT_FORMAT or moved or t.get('after') != (commit_names[i - 1] if i else None)
                         or bool(t.get('defer_sheet')) != (i < len(names) - 1)
                         or (staged.get('old_bank_json_hashes') != bank if i == 0 else
                             t.get('base_prepared_sha256') != hashlib.sha256((folders[i - 1] / 'prepared.json').read_bytes()).hexdigest()))
                if stale:
                    say(f'{names[i]} was staged against another bank or transaction; staging it and the runs after it again.')
            elif not staged:
                stale = True  # the runs after it were stacked on a staging that is gone
                continue
            if stale and folder.exists():
                shutil.rmtree(folder)
    command = 'bin/arcgen commit ' + ' '.join(shlex.quote(a) for a in sys.argv[2:])
    authorization = f'Owner commit via `{command}` on {time.strftime("%Y-%m-%d %H:%M")}; project stage {stage}.' + (f' Note: {note}' if note else '')
    for i, (name, commit_name) in enumerate(zip(names, commit_names)):
        root = roots[name]
        env = child_env(ARCGEN_CONFIG=root / 'config.json', ARCGEN_RUN_ROOT=root)
        args = ([str(PY), '-B', '-m', 'arcgen.batch.commit', '--root', str(root), '--name', commit_name, '--authorization', authorization,
                 '--stage-flag', stage] + (['--partial'] if partial else [])
                + (['--hold', str(ops(name) / 'held.json'), '--hold-subset'] if hold else [])
                + (['--release-judge', str(ops(name) / 'judge_released.json')] if release_judge else [])
                + (['--after', commit_names[i - 1]] if i else []) + (['--defer-sheet'] if i < len(names) - 1 else [])
                + ([] if dry_run else ['--then-install']))
        say(f'Preparing and verifying {name} ({i + 1}/{len(names)}) as {commit_name} (project_stage {stage})...')
        if subprocess.run(args, cwd=REPO, env=env).returncode != 0:
            raise SystemExit(f'preparing {name} failed; nothing was installed (see the output above)')
    if hold:
        applied = {x['task_id'] for c in commit_names for x in read(ROOT / 'runs' / c / 'authorization.json')['excluded']
                   if str(x.get('reason', '')).startswith('held: ')}
        if hold_ids - applied:
            raise SystemExit('held tasks are in none of these runs: ' + ', '.join(sorted(hold_ids - applied)) + '; nothing was installed')
    if dry_run:
        say('Prepared and verified; nothing installed (dry run). Receipts: ' + ', '.join(f'runs/{c}/' for c in commit_names))
        return
    first = roots[names[0]]
    env = child_env(ARCGEN_CONFIG=first / 'config.json', ARCGEN_RUN_ROOT=first)
    say(f'Installing {len(names)} runs into the bank as one transaction...')
    if subprocess.run([str(PY), '-B', '-m', 'arcgen.batch.commit', '--install', *commit_names, '--commit'], cwd=REPO, env=env).returncode != 0:
        raise SystemExit('commit failed; the bank is unchanged or rolled back (see the output above)')
    freed = sum(prune_commit_staging(f) for f in folders)
    if freed:
        say(f'Removed the commits\' working copies (staged banks and pre-commit backup, {freed / 1e9:.1f} GB); the bank and receipts stay.')
    say('Done. Receipts: ' + ', '.join(f'runs/{c}/ (report.md)' for c in commit_names))


def list_batches():
    """Every batch made by `new` (workspace batches/) or run in this workspace, with its status."""
    names = {p.stem for p in BATCHES.glob('*.json') if '.' not in p.stem} | {o.name[:-4] for o in (ROOT / 'runs').glob('*-ops')}
    for name in sorted(names):
        s = (read(run_root(name) / 'STATUS.json') or {}).get('status', 'not_started')
        say(f"{name:40s} {s:30s} {'running' if alive(name) else ''}")


def main():
    p = argparse.ArgumentParser(prog='bin/arcgen', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('new'); a.add_argument('name'); a.add_argument('--spec', type=Path, default=TEMPLATE_SPEC)
    a.add_argument('--size', type=int, default=1, help='repeat the design N times (e.g. 3 or 5)')
    a.add_argument('--workers', type=int, help='concurrent Codex calls (default: the profile\'s workers, capped by free memory)')
    a.add_argument('--profile', type=Path, default=TEMPLATE_PROFILE); a.add_argument('--description')
    a = sub.add_parser('start'); a.add_argument('name'); a.add_argument('--note')
    a.add_argument('--skip-tests', action='store_true', help='launch without the offline test gate (recorded in the launch manifest)')
    a = sub.add_parser('progress'); a.add_argument('name'); a.add_argument('--watch', type=int, nargs='?', const=60)
    a = sub.add_parser('stop'); a.add_argument('name'); a.add_argument('--now', action='store_true')
    a = sub.add_parser('resume'); a.add_argument('name')
    a = sub.add_parser('summary'); a.add_argument('name')
    a = sub.add_parser('commit'); a.add_argument('name', nargs='+', help='one run, or several committed as one transaction')
    a.add_argument('--stage', default='stage_3')
    a.add_argument('--note'); a.add_argument('--dry-run', action='store_true')
    a.add_argument('--partial', action='store_true', help='a stopped run that will not resume: commit only verified tasks of finished windows')
    a.add_argument('--hold', help='JSON of tasks to hold back ({task_id: reason} or a list), e.g. runs/<name>-ops/READ_FIRST.json after reading')
    a.add_argument('--release-judge', help='weakness runs: JSON list of JUDGE task IDs that passed the offline consensus reads')
    sub.add_parser('list')
    a = sub.add_parser('workers', help='change the number of concurrent Codex calls (takes effect on the next start or resume)')
    a.add_argument('name'); a.add_argument('total', type=int)
    sub.add_parser('test', help='run the offline test suite and record the gate for the current code')
    a = sub.add_parser('_supervise'); a.add_argument('name')
    x = p.parse_args()
    if x.cmd == 'new': new(x.name, x.spec, x.profile, x.description, x.size, x.workers)
    elif x.cmd == 'start': start(x.name, x.note, x.skip_tests)
    elif x.cmd == 'progress': show_progress(x.name, x.watch)
    elif x.cmd == 'stop': stop(x.name, x.now)
    elif x.cmd == 'resume': resume(x.name)
    elif x.cmd == 'summary': say(json.dumps(summary(x.name, write_files=True), indent=1))
    elif x.cmd == 'commit': commit(x.name, x.stage, x.note, x.dry_run, x.partial, x.hold, x.release_judge)
    elif x.cmd == 'list': list_batches()
    elif x.cmd == 'workers': set_workers(x.name, x.total)
    elif x.cmd == 'test':
        r = subprocess.run([str(PY), '-B', str(HERE / 'entry.py'), 'test'], cwd=REPO, env=child_env())
        raise SystemExit(r.returncode)
    elif x.cmd == '_supervise': supervise(x.name)


if __name__ == '__main__':
    main()
