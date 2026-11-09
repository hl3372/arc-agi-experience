"""End-to-end smoke test: a small batch through the whole pipeline, offline.

`new` (profile and allocation) -> prepare (bank snapshot, reference embeddings, wave A requests, Codex launcher) ->
qualify and ready -> authorize (usage reading) -> run (route probes, idea loop with explore turns, selector, wave B with
in-batch references, planning, realization, technical check, blind solve, terminal replay, report, gallery) -> summary ->
`bin/arcgen commit` into the bank. Model calls are answered by tests/support/e2e_driver.py through the mock transport;
embeddings use the local hashing provider; a stand-in `codex` file satisfies the executable binding and is never run.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
DRIVER = TESTS / 'support/e2e_driver.py'


def _run(args, env, timeout=900):
    out = subprocess.run([sys.executable, '-B', *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0, (out.stdout[-4000:], out.stderr[-4000:])
    return out.stdout


def test_a_small_batch_runs_from_new_to_commit(tmp_path):
    workspace = tmp_path / 'workspace'
    env = {k: v for k, v in os.environ.items() if not k.startswith('ARCGEN_')}
    env.update(ARCGEN_WORKSPACE=str(workspace), ARCGEN_BANK=str(workspace / 'committed_tasks'), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join([str(REPO)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])))
    _run([str(DRIVER), 'setup'], env)
    profile = workspace / 'batches/e2e.json'
    assert json.loads(profile.read_text())['operations']['workers'] == 8
    log = _run([str(DRIVER), 'run'], {**env, 'ARCGEN_CONFIG': str(profile)})
    run = workspace / 'runs/e2e'
    status = json.loads((run / 'STATUS.json').read_text())['status']
    assert status == 'verified_terminal', log[-3000:]
    report = json.loads((run / 'analysis/terminal_report.private.json').read_text())
    assert len(report['tasks']) == 2 and all(t['verified_before_union'] for t in report['tasks'])  # one idea per window
    # both developments realized the same fixture grids (the palette allocator may recolour one): the batch's
    # exact/palette union keeps one and holds the other
    verified = [t for t in report['tasks'] if t['usable_after_exact_palette_union']]
    held = [t for t in report['tasks'] if not t['usable_after_exact_palette_union']]
    first = verified[0]['task_id']
    assert len(verified) == 1 and {held[0]['exact_duplicate_of'], held[0]['palette_duplicate_of']} & {first, 'batch:' + first}
    assert (run / 'gallery/index.html').exists() and (workspace / 'runs/e2e-ops/SUMMARY.md').exists()
    assert (run / 'design/waves/B/batch_references.private.json').exists()  # wave B saw wave A's selection
    calls = [json.loads(line) for line in log.splitlines() if line.startswith('{"account"')]
    stages = {c['purpose'] for c in calls if c['state'] == 'complete'}
    assert {'transport_probe', 'full_window_selection', 'planning', 'realization', 'technical_verification', 'primary_blind'} <= stages
    before = len(json.loads((workspace / 'committed_tasks/active_registry.json').read_text())['entries'])
    _run(['-m', 'arcgen.batch.cli', 'commit', 'e2e', '--note', 'offline smoke test'], env)
    bank = json.loads((workspace / 'committed_tasks/active_registry.json').read_text())
    added = [e for e in bank['entries'] if e['commit_set_id'].startswith('e2e-task-commit-')]
    assert len(bank['entries']) == before + 1 and [e['source_candidate_id'] for e in added] == [verified[0]['task_id']]
    assert all(e['project_stage'] == 'stage_3' and (workspace / 'committed_tasks' / e['task_path']).is_file() for e in added)
    assert bank['commit_sets'][-1]['project_stage'] == 'stage_3'
