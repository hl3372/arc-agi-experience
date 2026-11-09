"""Partial commit of a stopped run: the gate, the finished-window membership and the receipt (synthetic folders, no bank)."""
import fcntl, json
from types import SimpleNamespace
import pytest
from arcgen.batch import batch_report, cli, commit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)); return path


def run_folder(tmp_path, status):
    run = tmp_path / 'runs/b'
    write(run / 'config.json', {'revision': 'r'}); write(run / 'STATUS.json', {'status': status, 'reason': 'stopped for good'})
    for name in ('prepared_manifest.private.json', 'launch_manifest.private.json'):
        write(run / name, {})
    return run


def task(run, tid, window, verified, terminal=True):
    state = write(run / 'accounts/main' / window / 'downstream/state.json', {'semantic_records': {}})
    if terminal:
        write(state.parent / 'window_terminal.private.json', {'window': window})
    return {'task_id': tid, 'window': window, 'verified_before_union': verified, 'usable_after_exact_palette_union': verified,
            'state': {'path': str(state), 'sha256': ''}}


def test_partial_commit_takes_only_a_stopped_idle_run(tmp_path, monkeypatch):
    monkeypatch.setattr(commit, 'ROOT', tmp_path)
    for status in ('running', 'verified_terminal', 'paused_usage'):
        run = run_folder(tmp_path, status)
        with pytest.raises(ValueError, match='not stopped'):
            commit.Commit(run, 'c', 'ok', partial=True).prepare()
    run = run_folder(tmp_path, 'stopped')
    with pytest.raises(ValueError, match='not verified_terminal'):
        commit.Commit(run, 'c', 'ok').prepare()
    assert commit.idle(run)
    with (run / 'runner.lock').open('a+') as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not commit.idle(run)
        with pytest.raises(ValueError, match='may be alive'):
            commit.Commit(run, 'c', 'ok', partial=True).prepare()
    assert commit.idle(run)
    (run / 'budget.sqlite3-wal').write_bytes(b'x')
    assert not commit.idle(run)
    with pytest.raises(ValueError, match='may be alive'):
        commit.Commit(run, 'c', 'ok', partial=True).prepare()


def test_cli_partial_commit_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path); monkeypatch.setattr(cli, 'alive', lambda name: False)
    monkeypatch.setattr(cli, 'run_pids', lambda root: [])
    calls = []
    monkeypatch.setattr(cli.subprocess, 'run', lambda args, **kw: calls.append(args) or SimpleNamespace(returncode=0))
    for status in ('running', 'verified_terminal'):
        run_folder(tmp_path, status)
        with pytest.raises(SystemExit, match='only a stopped run'):
            cli.commit('b', partial=True, dry_run=True)
    run_folder(tmp_path, 'stopped')
    with pytest.raises(SystemExit, match='not finished and verified'):
        cli.commit('b', dry_run=True)
    monkeypatch.setattr(cli, 'alive', lambda name: True)
    with pytest.raises(SystemExit, match='still running'):
        cli.commit('b', partial=True, dry_run=True)
    monkeypatch.setattr(cli, 'alive', lambda name: False)
    assert not calls
    cli.commit('b', partial=True, dry_run=True)
    args, = calls
    assert '--partial' in args and '--commit' not in args and args[args.index('--name') + 1].startswith('b-partial-task-commit-')


def test_partial_selection_excludes_unfinished_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(commit, 'ROOT', tmp_path)
    run = run_folder(tmp_path, 'stopped')
    done, open_ = task(run, 't1', 'w-develop-01', True), task(run, 't2', 'w-develop-02', False, terminal=False)
    windows = [{'window': 'w-develop-01', 'terminal': True}, {'window': 'w-develop-02', 'terminal': False}]
    report = {'status': 'stopped_partial', 'windows': windows, 'tasks': [done, open_]}
    c = commit.Commit(run, 'c', 'ok', partial=True)
    assert [r['task_id'] for r in c.select_rows(report)] == ['t1']
    assert 'runs/b/accounts/main/w-develop-01/downstream/window_terminal.private.json' in c.sources
    assert [r['task_id'] for r in commit.Commit(run, 'c', 'ok').select_rows(report)] == ['t1']
    bad = [{**report, 'status': 'in_progress'},  # not a partial report
           {**report, 'tasks': [done, {**open_, 'usable_after_exact_palette_union': True}]},  # usable but unverified
           {**report, 'tasks': [done, {**open_, 'usable_after_exact_palette_union': True, 'verified_before_union': True}]}]
    (run / 'accounts/main/w-develop-01/downstream/window_terminal.private.json').unlink()
    bad.append(report)  # the window terminal is gone
    for value in bad:
        with pytest.raises(ValueError, match='finished windows'):
            commit.Commit(run, 'c', 'ok', partial=True).select_rows(value)


def test_partial_source_records_the_stop_and_unfinished_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(commit, 'ROOT', tmp_path); monkeypatch.setattr(commit, '__file__', str(write(tmp_path / 'commit.py', {})))
    run = run_folder(tmp_path, 'stopped'); write(run / 'STOP_DISPATCH.json', {'reason': 'stopped for good'})
    write(run / 'budget.sqlite3', {'ledger': 1})
    done, open_ = task(run, 't1', 'w-develop-01', True), task(run, 't2', 'w-develop-02', False, terminal=False)

    def fake(root, *, out):
        assert out == tmp_path / 'runs/c/partial_source' and root == run.resolve()
        value = {'status': 'stopped_partial', 'tasks': [done, open_], 'ledger_snapshot': [{'source': batch_report.ref(run / 'budget.sqlite3')}],
                 'windows': [{'window': 'w-develop-01', 'terminal': True}, {'window': 'w-develop-02', 'terminal': False},
                             {'window': 'w-supply-02', 'terminal': False}]}
        write(out / 'analysis/terminal_report.private.json', value); write(out / 'analysis/research_index.private.json', {})
        return value
    monkeypatch.setattr(batch_report, 'report', fake)
    c = commit.Commit(run, 'c', 'ok', partial=True)
    report, _ = c.partial_source(json.loads((run / 'STATUS.json').read_text()))
    assert c.partial_record == {'partial': True, 'run_status': 'stopped', 'stop_reason': 'stopped for good',
                                'windows': {'total': 3, 'finished': 1, 'unfinished': 2}, 'unfinished_task_ids': ['t2']}
    receipt = json.loads((tmp_path / 'runs/c/partial_source/partial_receipt.json').read_text())
    assert receipt['unfinished_windows'] == ['w-develop-02', 'w-supply-02'] and receipt['run_written'] is False
    assert {'runs/b/STATUS.json', 'runs/b/STOP_DISPATCH.json', 'runs/b/budget.sqlite3', 'runs/c/partial_source/partial_receipt.json',
            'runs/c/partial_source/analysis/terminal_report.private.json'} <= set(c.sources)
    assert 'partial commit of the stopped run runs/b' in c.authorization() and [r['task_id'] for r in c.select_rows(report)] == ['t1']
    assert 'every task of runs/b usable after' in commit.Commit(run, 'c', 'ok').authorization()


def test_partial_report_is_written_outside_the_run_from_ledger_copies(tmp_path):
    run = tmp_path / 'runs/b'
    for name in batch_report.LEDGERS:
        write(run / name, {'name': name})
    with pytest.raises(batch_report.IntegrityError, match='outside the run'):
        batch_report.report(run, out=run / 'analysis')
    rows = batch_report.snapshot_ledgers(run, tmp_path / 'out')
    assert [r['source']['sha256'] for r in rows] == [r['copy']['sha256'] for r in rows]
    assert (tmp_path / 'out/ledger_snapshot/inputs/palette_context.private.json').is_file()
    (run / 'palette.sqlite3-wal').write_bytes(b'x')
    with pytest.raises(batch_report.IntegrityError, match='write-ahead log'):
        batch_report.snapshot_ledgers(run, tmp_path / 'out2')
