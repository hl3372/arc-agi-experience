"""The sandbox blocks escapes; the sibling, probe-mutant and novelty checks catch what they are meant to catch.

The sandbox tests use only paths created or derived here (no machine-specific paths), a local listening socket instead
of an internet address, and a throwaway child process as the signal target."""
import os
import socket
import subprocess
import sys
from pathlib import Path
import pytest
from research_envs import common
from research_envs.construct import sandbox
from research_envs.construct.sandbox import run_program, landlock_abi, MIN_ABI, SandboxUnavailable
from research_envs.construct import sibling, probes, novelty
from research_envs.construct.evaluate import Spec

SPEC = {'object': 'graph', 'kind': 'satisfy', 'params': {'n': 5}, 'constraints': [{'name': 't', 'expr': 'True'}]}
needs_landlock = pytest.mark.skipif(landlock_abi() < MIN_ABI, reason=f'needs Linux with Landlock ABI >= {MIN_ABI}')


def last(code, timeout=10):
    return run_program(code, SPEC, timeout=timeout)


def probe_open(path):
    return f'try:\n    open({str(path)!r}).read(); print("OPEN")\nexcept Exception: print("blocked")'


def probe_list(path):
    return f'import os\ntry:\n    os.listdir({str(path)!r}); print("OPEN")\nexcept Exception: print("blocked")'


def test_sandbox_fails_closed_without_landlock(monkeypatch):
    monkeypatch.setattr(sandbox, 'landlock_abi', lambda: -1)
    with pytest.raises(SandboxUnavailable):
        run_program('print(1)', SPEC)


def test_sandbox_refuses_read_rules_that_would_expose_home_or_runs(monkeypatch):
    for bad in (str(Path.home()), str(common.ROOT / 'runs'), '/'):
        monkeypatch.setattr(sandbox, 'site_dirs', lambda bad=bad: [bad])
        with pytest.raises(SandboxUnavailable):
            sandbox.read_only_paths()


@needs_landlock
def test_sandbox_blocks_reading_private_files(tmp_path):
    secret = tmp_path / 'secret.txt'
    secret.write_text('private')
    runs = common.ROOT / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    (runs / 'answers.json').write_text('{}')
    targets = [probe_open(secret), probe_open(Path(__file__).resolve()), probe_open(runs / 'answers.json'),
               probe_list(Path.home()), probe_list(tmp_path), probe_list(Path(__file__).resolve().parents[1])]
    import tempfile
    targets.append(probe_list(tempfile.gettempdir()))
    if common.CHECKOUT:
        targets.append(probe_open(common.CHECKOUT / 'pyproject.toml'))
    for code in targets:
        assert open(secret).read() == 'private'           # readable from outside the sandbox
        r = last(code)
        assert r['stdout_tail'].strip() == 'blocked', (code, r)


@needs_landlock
def test_sandbox_blocks_writes_network_and_signals():
    assert last('try:\n    open("/usr/evil", "w"); print("OPEN")\nexcept Exception: print("blocked")')['stdout_tail'].strip() == 'blocked'
    srv = socket.socket()
    srv.bind(('127.0.0.1', 0)); srv.listen(1)
    port = srv.getsockname()[1]
    try:
        socket.create_connection(('127.0.0.1', port), timeout=3).close()   # reachable from outside the sandbox
        code = (f'import socket\ns = socket.socket(); s.settimeout(3)\ntry:\n    s.connect(("127.0.0.1", {port})); '
                f'print("OPEN")\nexcept Exception: print("blocked")')
        assert last(code)['stdout_tail'].strip() == 'blocked'
    finally:
        srv.close()
    victim = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        code = (f'import os, signal\ntry:\n    os.kill({victim.pid}, signal.SIGTERM); print("OPEN")\n'
                f'except Exception: print("blocked")')
        assert last(code)['stdout_tail'].strip() == 'blocked'
        assert victim.poll() is None                      # still alive
    finally:
        victim.kill(); victim.wait()


@needs_landlock
def test_sandbox_runs_programs_and_limits():
    r = last('import json, numpy, networkx\nprint(json.dumps({"n": 5, "edges": [[0, 1]]}))')
    assert r['ok'] and r['object'] == {'n': 5, 'edges': [[0, 1]]}
    assert last('while True: pass', timeout=2)['timed_out']
    assert not last('x = bytearray(4 * 1024**3); print(1)')['ok']


def test_sibling_finds_redundancy_optimum_and_misreading():
    raw = {'object': 'graph', 'kind': 'optimize', 'params': {'n': 6},
           'constraints': [{'name': 'triangle_free', 'expr': 'triangles() == 0'},
                           {'name': 'vacuous', 'expr': 'components() <= 6'}],
           'objective': {'expr': 'len(E)', 'direction': 'max'}}
    r = sibling.enumerate_all(raw)
    assert r['optimum'] == 9 and r['redundant_constraints'] == ['vacuous'] and r['optimal_classes'] == 1
    misread = {**raw, 'constraints': [{'name': 'triangle_free', 'expr': 'triangles() <= 1'}]}
    assert sibling.enumerate_all(raw, other=misread)['disagree'] > 0


def test_mutants_are_killed_by_a_good_probe_set():
    raw = {'object': 'graph', 'kind': 'optimize', 'params': {'n': 7},
           'constraints': [{'name': 'triangle_free', 'expr': 'triangles() == 0'},
                           {'name': 'min_deg', 'expr': 'all(deg(v) >= 2 for v in V)'}],
           'objective': {'expr': 'len(E)', 'direction': 'max'}}
    k34 = {'n': 7, 'edges': [[a, b] for a in range(3) for b in range(3, 7)]}
    ad = probes.adequacy(raw, probes.probe_set(raw, [k34]))
    assert ad['score'] >= 0.9, ad['survivors']


def test_fingerprint_separates_lookalikes_and_matches_restatements():
    mantel = {'object': 'graph', 'kind': 'optimize', 'params': {},
              'constraints': [{'name': 'tf', 'expr': 'triangles() == 0'}], 'objective': {'expr': 'len(E)', 'direction': 'max'}}
    restated = {**mantel, 'constraints': [{'name': 'no_k3', 'expr': 'clique_number() <= 2'}]}
    bipartite = {**mantel, 'constraints': [{'name': 'bip', 'expr': 'is_bipartite()'}]}
    fa, fb, fc = (novelty.fingerprint(x, sizes=(4, 5, 6)) for x in (mantel, restated, bipartite))
    assert novelty.same_family(fa, fb) and not novelty.same_family(fa, fc)


def test_zoo_recognises_relabelled_objects():
    import random
    rng = random.Random(1)
    perm = list(range(17)); rng.shuffle(perm)
    qr = {(x * x) % 17 for x in range(1, 17)}
    paley = {'n': 17, 'edges': [[perm[a], perm[b]] for a in range(17) for b in range(a + 1, 17) if (b - a) % 17 in qr]}
    assert novelty.zoo_match('graph', paley) == 'paley_17'
    fano = {'m': 7, 'sets': [[(x + 2) % 7 for x in s] for s in [[0, 1, 3], [1, 2, 4], [2, 3, 5], [3, 4, 6], [4, 5, 0], [5, 6, 1], [6, 0, 2]]]}
    assert novelty.zoo_match('set_system', fano) == 'fano'


def test_canary_matches_archive_after_json_round_trip():
    """Gate bug 1: fresh fingerprints (tuples) must match archived ones (lists)."""
    arch = {a['id']: a for a in novelty.load_archive()}
    fresh = {'sizes': [4, 5, 6, 7], 'values': [(None, 62), (None, 892), (None, 22484), (None, 923012)]}
    assert novelty.same_family(fresh, arch['ramsey_44']['fingerprint'])
    assert not novelty.same_family(fresh, arch['ramsey_34']['fingerprint'])


def test_probes_exercise_size_constraints_hidden_by_the_search_hint():
    """Gate bug 2: with a hint restricting search to 4-sets, probes must still violate the size constraint alone."""
    pg = {'object': 'set_system', 'kind': 'optimize', 'params': {'m': 13},
          'constraints': [{'name': 'blocks_of_4', 'expr': 'all(len(B) == 4 for B in F)'},
                          {'name': 'meet_exactly_once', 'expr': 'all(len(A & B) == 1 for A, B in combinations(F, 2))'}],
          'objective': {'expr': 'len(F)', 'direction': 'max'}}
    w = {'m': 13, 'sets': [sorted((i + d) % 13 for d in (0, 1, 3, 9)) for i in range(13)]}
    ps = probes.probe_set(pg, [w], {'set_sizes': [4]})
    assert probes.adequacy(pg, ps)['score'] >= 0.9
    assert all(c > 0 for c in probes.violated_alone(pg, ps).values())
