"""Domain A round in task processes with a stubbed model: every stage runs, a failing task is isolated."""
import json, os, subprocess, sys
from pathlib import Path


def test_domain_a_round_dry(tmp_path):
    here = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'RE_LLM_STUB': str(here / 'tests' / 'llm_stub_a.py'), 'RE_ROOT': str(tmp_path), 'RE_A_BUDGET': '12'}
    code = ("import sys; sys.path.insert(0, %r)\nfrom research_envs.discover import pipeline\n"
            "if __name__ == '__main__':\n"
            "    pipeline.run_round('dry', [{'id': 'ok1', 'demand': 'A2'}, {'id': 'bad1', 'demand': None}], workers=2)\n") % str(here)
    script = tmp_path / 'dry.py'; script.write_text(code)
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    rd = tmp_path / 'runs' / 'dry'
    assert not (rd / 'summary.json').exists() and (rd / 'tasks' / 'bad1' / 'failure.txt').exists()
    t = json.loads((rd / 'tasks' / 'ok1' / 'dossier.json').read_text())
    assert t['final'] and t['class'] in ('frontier', 'solved', 'calibration', 'rejected'), t.get('reasons')
    assert set(t['pilots']) == {'none', 'quarter', 'full'} and t['queries']
    assert t['pilots']['full']['experiments_used'] >= 1 and t['pilots']['none']['experiments_used'] == 0


def test_domain_a_round2_switches_dry(tmp_path):
    """Round 2 switches: world-named interface, round registry, a lever-pinned request with its profile recorded."""
    here = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'RE_LLM_STUB': str(here / 'tests' / 'llm_stub_a.py'), 'RE_ROOT': str(tmp_path), 'RE_A_BUDGET': '12',
           'RE_A_V2': '1', 'RE_A_REGISTRY': '1'}
    code = ("import sys; sys.path.insert(0, %r)\nfrom research_envs.discover import pipeline\n"
            "if __name__ == '__main__':\n"
            "    pipeline.run_round('dry2', [{'id': 'l1', 'demand': None, 'lever': 'alias'},"
            " {'id': 'l2', 'demand': None, 'lever': 'time'}], workers=1)\n") % str(here)
    script = tmp_path / 'dry2.py'; script.write_text(code)
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    rd = tmp_path / 'runs' / 'dry2'
    t1 = json.loads((rd / 'tasks' / 'l1' / 'dossier.json').read_text())
    assert t1['lever'] == 'alias' and t1['lever_profile']['alias'] is True      # the stub's lamp readout is h > 0
    t2 = json.loads((rd / 'tasks' / 'l2' / 'dossier.json').read_text())
    assert 'duplicates a world already in this round' in t2['reasons'][0]        # the stub proposes the same world twice
    assert (rd / 'registry.jsonl').exists()
