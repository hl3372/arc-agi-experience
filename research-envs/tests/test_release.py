"""Release checks with no model calls: one Codex account and a concurrency cap (against a fake `codex` executable),
the OpenRouter key lookup, the command-line presets, and a smoke test that validates shipped examples."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import pytest
from research_envs import cli, common
from research_envs.construct import llm, openrouter

HERE = Path(__file__).resolve().parents[1]

FAKE_CODEX = r'''#!/usr/bin/env python3
"""Stands in for the Codex CLI: records CODEX_HOME and its arguments, then writes a canned structured answer."""
import json, os, sys, time
args = sys.argv[1:]
log = os.environ['FAKE_CODEX_LOG']
with open(log, 'a') as f:
    f.write(json.dumps({'home': os.environ.get('CODEX_HOME'), 'args': args, 'start': time.time()}) + '\n')
time.sleep(float(os.environ.get('FAKE_CODEX_SLEEP', '0')))
if os.environ.get('FAKE_CODEX_FAIL'):
    sys.exit(3)
out = args[args.index('--output-last-message') + 1]
open(out, 'w').write(json.dumps({'answer': 'ok'}))
with open(log, 'a') as f:
    f.write(json.dumps({'end': time.time()}) + '\n')
'''
SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    exe = tmp_path / 'codex'
    exe.write_text(FAKE_CODEX)
    exe.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    monkeypatch.setattr(llm, 'CODEX', str(exe))
    monkeypatch.setenv('FAKE_CODEX_LOG', str(log))
    monkeypatch.setattr(llm.time, 'sleep', lambda s: None)       # no back-off waits between retries
    return log


def _rows(log):
    return [json.loads(l) for l in log.read_text().splitlines()]


def test_one_codex_account_default_home(fake_codex, tmp_path, monkeypatch):
    monkeypatch.delenv('CODEX_HOME', raising=False)
    assert llm.call('hi', SCHEMA, tmp_path / 'rec', 'a') == {'answer': 'ok'}
    starts = [r for r in _rows(fake_codex) if 'home' in r]
    assert starts[0]['home'] == str(Path.home() / '.codex')
    assert '--sandbox' in starts[0]['args'] and starts[0]['args'][starts[0]['args'].index('--sandbox') + 1] == 'read-only'
    rec = json.loads((tmp_path / 'rec' / 'a' / 'attempt-1.json').read_text())
    assert 'account' not in rec and rec['model'] == llm.MODEL


def test_codex_home_from_environment_and_no_rotation_across_retries(fake_codex, tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'acct'))
    monkeypatch.setenv('FAKE_CODEX_FAIL', '1')
    with pytest.raises(llm.LLMError):
        llm.call('hi', SCHEMA, tmp_path / 'rec', 'b', retries=2)
    homes = [r['home'] for r in _rows(fake_codex) if 'home' in r]
    assert homes == [str(tmp_path / 'acct')] * 3          # three attempts, one account


def test_codex_concurrency_cap(fake_codex, tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)           # slot files under tmp_path/runs
    monkeypatch.setattr(llm, 'CONCURRENCY', 1)
    monkeypatch.setenv('FAKE_CODEX_SLEEP', '0.3')
    ts = [threading.Thread(target=llm.call, args=('hi', SCHEMA, tmp_path / 'rec', f'c{i}')) for i in range(3)]
    [t.start() for t in ts]; [t.join() for t in ts]
    rows = _rows(fake_codex)
    spans = [(rows[i]['start'], rows[i + 1]['end']) for i in range(0, len(rows), 2)]
    assert len(spans) == 3 and all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), spans   # never overlapping


def test_openrouter_key_only_from_env_or_named_file(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path / '.openrouter_key').write_text('implicit')    # the old implicit location is no longer read
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.delenv('OR_KEY_FILE', raising=False)
    with pytest.raises(openrouter.ORError):
        openrouter._key()
    kf = tmp_path / 'k'
    kf.write_text('from-file\n')
    monkeypatch.setenv('OR_KEY_FILE', str(kf))
    assert openrouter._key() == 'from-file'
    monkeypatch.setenv('OPENROUTER_API_KEY', 'from-env')
    assert openrouter._key() == 'from-env'


def test_presets_reproduce_the_published_request_layouts():
    a = cli.a_requests(cli.A_PRESETS['demo'][1], 'da')
    assert [r['id'] for r in a] == [f'da_{lv}_{k}' for lv in ('time', 'cycle', 'alias', 'coupled') for k in (1, 2, 3)]
    assert all(r['demand'] is None and r['lever'] for r in a)
    b = cli.b_requests(cli.B_PRESETS['demo'][1], 'db')
    assert [r['id'] for r in b] == ['db_d2_g1', 'db_d2_g2', 'db_d2_s1', 'db_d2_s2', 'db_d2_s3', 'db_d4_g1', 'db_d4_g2',
                                    'db_d4_g3', 'db_ctl_g1', 'db_ctl_g2', 'db_ctl_s1', 'db_ctl_s2']
    r1 = cli.b_requests(cli.B_PRESETS['round1'][1], 'r1')
    assert len(r1) == 32 and sum(r['demand'] is None for r in r1) == 16
    a1 = cli.a_requests(cli.A_PRESETS['round1'][1], 'a1')
    assert len(a1) == 32 and sum(r['demand'] is None for r in a1) == 16
    env = cli.A_PRESETS['demo'][0]
    assert env['RE_A_QUERY_MODE'] == 'discriminating' and env['RE_A_V2'] == '1'


def _cli(*args, timeout=600):
    return subprocess.run([sys.executable, '-m', 'research_envs', *args], cwd=HERE, capture_output=True, text=True,
                          timeout=timeout, env={**os.environ, 'PYTHONPATH': str(HERE)})


def test_cli_info_runs_offline():
    r = _cli('info')
    assert r.returncode == 0, r.stderr
    assert 'runs root' in r.stdout and 'codex home' in r.stdout


def test_smoke_validate_one_world_and_one_problem():
    """Builds and validates one example world and one example problem from their task files, with no model calls:
    the world compiles and reproduces its public interface, queries and answers, and its critical queries refute
    rivals; the problem's spec compiles, its witness is valid, its witness program reproduces it in the sandbox
    (skipped without Landlock), and its sibling instance enumerates to the recorded results."""
    r = _cli('validate', '--json', 'da_time_3', 'db_ctl_s1')
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = json.loads(r.stdout[:r.stdout.rindex(']') + 1])
    assert [x['id'] for x in out] == ['da_time_3', 'db_ctl_s1'] and all(x['ok'] for x in out)
    checks = {c['check']: c['ok'] for x in out for c in x['checks']}
    assert checks['queries and answers reproduce'] and checks['sibling instance enumerates to the recorded results']


def test_every_example_world_validates():
    files = sorted((HERE / 'examples' / 'discover').glob('*.json'))
    assert len(files) == 10
    r = _cli('validate', *map(str, files))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_example_files_are_complete_and_hold_no_review_text():
    for sub, n in (('discover', 10), ('construct', 10)):
        files = sorted((HERE / 'examples' / sub).glob('*.json'))
        assert len(files) == n
        for f in files:
            t = json.loads(f.read_text())
            assert t['format'].startswith('research-envs/') and t['public'] and t['private'] and t['results']
            text = f.read_text()
            for k in ('"pilots"', '"history"', '"beliefs"', '"select_reason"', '"plan_notes"', '"verdict"'):
                assert k not in text, (f.name, k)
