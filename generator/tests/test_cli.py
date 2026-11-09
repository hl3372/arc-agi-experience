"""Self-serve commands: progress reading on a synthetic run folder (no model calls, no subprocesses)."""
import json
from arcgen.batch import cli


def test_progress_reads_ideas_selection_and_development(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    run = tmp_path / 'runs/b'; ops = tmp_path / 'runs/b-ops'
    (run / 'design').mkdir(parents=True); ops.mkdir(parents=True)
    (run / 'STATUS.json').write_text(json.dumps({'status': 'running'}))
    calls = [{'opaque_id': f'p{i}'} for i in range(3)]
    (run / 'design/batch.private.json').write_text(json.dumps({'windows': [{'kind': 'fresh_supply', 'calls': calls}]}))
    queue = run / 'accounts/main/w1/eligible_queue.private.json'; queue.parent.mkdir(parents=True)
    queue.write_text(json.dumps({'rows': [{}, {}, {}]}))
    state = run / 'accounts/main/w1-develop-01/downstream/state.json'; state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'semantic_records': {
        't1': {'complete_task': {}, 'pipeline_usable_before_union_audit': True},
        't2': {'complete_task': {}, 'answer_evidence': {'difficulty_outcome': {'reason': 'repair_declined'}}},
        't3': {'status': 'planning'}}}))
    events = [{'account': 'main', 'event': 'call', 'call_id': f'p{i}-commit', 'purpose': 'idea_unconditioned', 'state': s}
              for i, s in ((0, 'complete'), (1, 'complete'), (2, 'started'))]
    events.append({'account': 'main', 'event': 'call', 'call_id': 't3-concrete-plan', 'purpose': 'planning', 'state': 'started'})
    (ops / 'run.log').write_text('\n'.join(json.dumps(e) for e in events) + '\n')
    p = cli.progress('b')
    assert p['idea_calls'] == [2, 3] and p['selector_windows'] == [1, 1] and p['eligible_ideas'] == 3
    assert (p['developments'], p['verified'], p['held_or_failed'], p['in_development']) == (3, 1, 1, 1)
    assert p['calls_in_flight'] == 2 and not p['running'] and not p['paused']
    line = cli.one_line(p)
    assert 'ideas 2/3 calls' in line and '1 verified, 1 held/failed, 1 in progress' in line
    # live: running and queued calls come from the worker pools, not from run.log (which logs a call before its slot)
    monkeypatch.setattr(cli, 'alive', lambda name: True)
    pool = run / 'accounts/main/worker_pool.json'; pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text(json.dumps({'active_calls': {f'c{i}': {} for i in range(3)}, 'waiting_by_category': {'downstream': 3, 'selection_codex': 1}}))
    p = cli.progress('b')
    assert (p['calls_running'], p['calls_queued']) == (3, 4) and '3 calls running, 4 queued' in cli.one_line(p)
    # a finished run's elapsed time stops when it ended
    (run / 'STATUS.json').write_text(json.dumps({'status': 'verified_terminal'}))
    now = cli.time.time()
    (ops / 'started.json').write_text(json.dumps({'first_started_at': now - 900 * 60, 'finished_at': now - 840 * 60}))
    assert cli.progress('b')['elapsed_minutes'] == 60
    (ops / 'started.json').write_text(json.dumps({'first_started_at': now - 900 * 60}))
    (run / 'completion.json').write_text(json.dumps({'completed_at_epoch': now - 800 * 60}))  # runs from before the record
    assert cli.progress('b')['elapsed_minutes'] == 100


def test_size_option_repeats_the_window_pattern():
    """--size k keeps every window's size and account, so qualification rules still hold, and multiplies the calls."""
    spec = {'windows': {'one-supply-01': {'account': 'main', 'wave': 'A', 'context': 'c01'},
                        'one-supply-02': {'account': 'main', 'wave': 'A', 'context': 'c02'},
                        'two-supply-01': {'account': 'main', 'wave': 'A', 'context': 'c01'}},
            'rows': [['one-supply-01', 'GEN', None, 'L', 'sol', 'high']] * 6 + [['one-supply-02', 'SEE', None, 'L', 'sol', 'max']] * 2
                    + [['two-supply-01', 'WORLD', None, 'L', 'sol6', 'xhigh']] * 6}
    s = cli.scale_spec(spec, 3)
    assert len(s['rows']) == 3 * len(spec['rows']) and len(s['windows']) == 9
    from collections import Counter
    widths = Counter(r[0] for r in s['rows'])
    assert max(widths.values()) == 6 and sorted(widths.values()) == sorted([6, 2, 6] * 3)
    assert {w['account'] for w in s['windows'].values()} == {'main'}
    assert all(s['windows'][r[0]]['account'] == spec['windows'][o[0]]['account'] for r, o in zip(s['rows'], spec['rows'] * 3))
    assert cli.scale_spec(spec, 1) == spec
    e = cli.estimate(s['rows'], s['windows'])
    assert e['idea_calls'] == 42 and e['workers'] == 30 and e['tokens_millions'] > 0


def test_supervisor_resumes_only_unexpected_exits():
    """Finished -> summary; your pause, a pipeline stop or the weekly limit -> wait for you; anything else -> resume, 3 times."""
    assert cli.resume_decision('verified_terminal', None, 0) == 'summary'
    assert cli.resume_decision('stopped', 'owner pause via bin/arcgen stop', 0) == 'stop'
    assert cli.resume_decision('stopped', 'frozen artifact changed: x', 0) == 'stop'   # a problem the pipeline found
    assert cli.resume_decision('paused_usage', None, 0) == 'stop'
    assert cli.resume_decision('running', None, 0) == 'resume'                          # e.g. killed by the out-of-memory killer
    assert cli.resume_decision('running', None, cli.AUTO_RESUMES - 1) == 'resume'
    assert cli.resume_decision('running', None, cli.AUTO_RESUMES) == 'give_up'
    # a usage pause caused only by an unavailable usage reading is transient and resumed; a real limit waits for you
    assert cli.resume_decision('paused_usage', None, 0, ['quota_observation_unavailable']) == 'resume'
    assert cli.resume_decision('paused_usage', None, 0, ['quota_observation_unavailable', 'main_usage_exhausted']) == 'stop'
    assert cli.resume_decision('paused_usage', None, 0, ['provider_usage_limit']) == 'stop'
    # a passing database lock is resumed; other pipeline stops wait for you
    assert cli.resume_decision('stopped', 'database is locked', 0) == 'resume'
    assert cli.resume_decision('stopped', 'frozen code changed', 0) == 'stop'


def test_run_is_kept_awake_when_possible():
    from pathlib import Path
    cmd = cli.run_command(Path('runs/example'))
    assert cmd[-6:-1] == [str(cli.PY), '-B', str(cli.HERE / 'entry.py'), 'run', '--root'] and cmd[-1] == 'runs/example'
    if cmd[0].endswith('systemd-inhibit'):
        assert any(a.startswith('--what=sleep') for a in cmd) and '--mode=block' in cmd
        assert not any('handle-lid-switch' in a for a in cmd)  # closing the lid still suspends (2026-10-07)


def test_supervisor_resumes_a_killed_run_then_summarises(tmp_path, monkeypatch):
    """First attempt is SIGKILLed (as by the out-of-memory killer); the supervisor resumes and the second attempt finishes."""
    import json, sys
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    run = tmp_path / 'runs/b'; run.mkdir(parents=True)
    (run / 'STATUS.json').write_text(json.dumps({'status': 'running'}))
    script = tmp_path / 'fake_run.py'
    script.write_text(
        "import json, os, signal, sys\n"
        f"run = {str(run)!r}\n"
        "marker = os.path.join(run, 'attempted')\n"
        "if not os.path.exists(marker):\n"
        "    open(marker, 'w').close(); os.kill(os.getpid(), signal.SIGKILL)\n"
        "json.dump({'status': 'verified_terminal'}, open(os.path.join(run, 'STATUS.json'), 'w'))\n")
    monkeypatch.setattr(cli, 'run_command', lambda root: [sys.executable, str(script)])
    monkeypatch.setattr(cli.time, 'sleep', lambda s: None)
    monkeypatch.setattr(cli, 'quota_reading', lambda root: {'status': 'observed', 'accounts': {}})
    summaries = []
    monkeypatch.setattr(cli, 'summary', lambda name, write_files=False: summaries.append(name))
    cli.supervise('b')
    assert summaries == ['b']
    resumes = json.loads((tmp_path / 'runs/b-ops/auto_resumes.json').read_text())
    assert len(resumes) == 1 and resumes[0]['returncode'] == -9
    log = (tmp_path / 'runs/b-ops/progress.log').read_text()
    assert 'automatic resume 1/3' in log and 'run process exited (0)' in log


def test_stale_unlaunched_preparation_is_discarded(tmp_path):
    """A readied but unlaunched folder made with other code is removed; the same code, or a launched run, is kept."""
    def folder(name, code, launched=False):
        root = tmp_path / name; root.mkdir()
        (root / 'prepared_launch_manifest.private.json').write_text(json.dumps({'code_hash': code}))
        if launched:
            (root / 'launch_manifest.private.json').write_text('{}')
        return root
    assert cli.discard_stale_preparation(folder('old', 'a'), 'b') and not (tmp_path / 'old').exists()
    assert not cli.discard_stale_preparation(folder('same', 'b'), 'b') and (tmp_path / 'same').exists()
    assert not cli.discard_stale_preparation(folder('launched', 'a', launched=True), 'b') and (tmp_path / 'launched').exists()
    assert not cli.discard_stale_preparation(tmp_path / 'missing', 'b')


def test_resume_refuses_a_finished_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path); monkeypatch.setattr(cli, 'alive', lambda name: False)
    (tmp_path / 'runs/done').mkdir(parents=True)
    (tmp_path / 'runs/done/STATUS.json').write_text(json.dumps({'status': 'verified_terminal'}))
    import pytest
    with pytest.raises(SystemExit, match='already finished'):
        cli.resume('done')


def test_commit_staging_is_pruned_only_after_a_complete_commit(tmp_path):
    def folder(name, status):
        d = tmp_path / name
        for sub in ('staged_bank/tasks', 'backup'):
            (d / sub).mkdir(parents=True)
        (d / 'staged_bank/tasks/t.json').write_text('{}'); (d / 'report.md').write_text('receipt')
        if status:
            (d / 'transaction.json').write_text(json.dumps({'status': status}))
        return d
    done, partial, staged = folder('done', 'complete'), folder('partial', 'installing'), folder('staged', None)
    assert cli.prune_commit_staging(done) > 0 and not (done / 'staged_bank').exists() and not (done / 'backup').exists()
    assert (done / 'report.md').exists()
    for d in (partial, staged):
        assert cli.prune_commit_staging(d) == 0 and (d / 'staged_bank').exists() and (d / 'backup').exists()


def test_saved_windows_pass_through_the_allocation_builder():
    from arcgen.batch.allocations.make_allocation import build
    spec = {'revision': 'r', 'seed': 1, 'windows': {'w1': {'account': 'main', 'wave': 'A', 'context': 'c01', 'commit': 'kept'}},
            'rows': [['w1', 'OPEN', None, 'L', 'sol', 'high']],
            'saved_windows': {'one-saved-01': {'account': 'main', 'wave': 'A', 'context': 'c01', 'seed': 7,
                                                 'queue': 'q.json', 'queue_sha256': 'x'}}}
    out = build(spec, {'sol': 'gpt-6.1-sol'}, {}, lambda arm, inv: True)
    assert out['saved_windows'] == spec['saved_windows']


def test_new_accepts_a_saved_only_spec_and_estimates_its_developments(tmp_path, monkeypatch):
    """`new` takes a spec of saved windows only (no idea calls) at --size 1; idea specs keep the 24-idea window rule and
    run every window on the one account. The estimate counts the saved developments, and the allocation builder passes
    saved idea windows through with no calls."""
    from arcgen.batch.allocations.make_allocation import build
    profile = {'operations': {'workers': 30}}
    for name, value in (('q.json', {'rows': [{}] * 5}), ('i.json', {'source_rows': {'k1': {}, 'k2': {}}})):
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    saved = {'revision': 'r', 'seed': 1, 'windows': {}, 'rows': [],
             'saved_windows': {'one-saved-x': {'account': 'main', 'wave': 'A', 'context': 'c01', 'seed': 7, 'queue': 'q.json',
                                                 'queue_sha256': 'x'}},
             'saved_idea_windows': {'two-carried-x1': {'account': 'main', 'wave': 'A', 'context': 'c01', 'seed': 8,
                                                         'ideas': 'i.json', 'ideas_sha256': 'y'}}}
    assert cli.shape_problem(saved, profile, 1) is None
    assert 'size 1' in cli.shape_problem(cli.scale_spec(saved, 2), profile, 2)
    assert 'neither idea calls nor saved windows' in cli.shape_problem({'windows': {}, 'rows': []}, profile, 1)
    ideas = {'windows': {'one-supply-01': {'account': 'main', 'wave': 'A', 'context': 'c01'},
                         'two-supply-01': {'account': 'main', 'wave': 'A', 'context': 'c01'}},
             'rows': [['one-supply-01', 'GEN', None, 'L', 'sol', 'high']] * 6 + [['two-supply-01', 'SEE', None, 'L', 'sol', 'high']] * 5}
    assert cli.shape_problem(ideas, profile, 1) is None
    assert 'exactly 24 ideas' in cli.shape_problem({**ideas, 'rows': ideas['rows'][:5]}, profile, 1)
    other = {**ideas, 'windows': {**ideas['windows'], 'two-supply-01': {'account': 'second', 'wave': 'A', 'context': 'c01'}}}
    assert 'one Codex account' in cli.shape_problem(other, profile, 1)
    assert cli.saved_rows(saved) == 7
    e = cli.estimate([], {}, 30, saved_rows=7)
    assert e['idea_calls'] == 0 and e['saved_developments'] == 7 and e['expected_verified_tasks'] == round(7 * cli.VERIFIED_PER_SAVED)
    assert e['hours'] > cli.TAIL_HOURS
    assert 'saved_developments' not in cli.estimate(ideas['rows'], ideas['windows'], 30)
    out = build(saved, {'sol': 'gpt-6.1-sol'}, {}, lambda arm, inv: True)
    assert (out['calls'], out['saved_windows'], out['saved_idea_windows']) == ([], saved['saved_windows'], saved['saved_idea_windows'])


def test_run_pids_finds_only_this_batch():
    import os, subprocess, sys, time
    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', 'entry.py', 'run', '--root', '/tmp/some-batch'])
    try:
        time.sleep(0.3)
        assert cli.run_pids('/tmp/no-such-batch') == [] and p.pid in cli.run_pids('/tmp/some-batch')
    finally:
        p.kill()
