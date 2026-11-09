"""Start-up ramp: rate-limit (429) waves came at a run's start, when all pool slots filled at once."""
import copy, json, threading, time
import pytest
from arcgen.batch import config
from arcgen.lib.e18_pre_e20_pool import DedicatedCallPool


def _pool(path, **ramp):
    return DedicatedCallPool(path, downstream_workers=4, upstream_workers=2, global_ceiling=6, codex_cap=6, allow_resize=True, **ramp)


def _acquire_times(pool, n):
    times, lock = [], threading.Lock()
    def one():
        with pool.bound('downstream'):
            with lock:
                times.append(time.time())
    threads = [threading.Thread(target=one) for _ in range(n)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=30)
    return sorted(times)


def test_ramp_spaces_the_first_acquisitions(tmp_path):
    times = _acquire_times(_pool(tmp_path / 'pool.json', ramp_gap_seconds=0.2, ramp_window_seconds=30), 4)
    assert len(times) == 4 and all(b - a >= 0.18 for a, b in zip(times, times[1:]))


def test_no_ramp_by_default_or_after_the_window(tmp_path):
    start = time.time()
    assert _acquire_times(_pool(tmp_path / 'a.json'), 4)[-1] - start < 1
    start = time.time()
    pool = _pool(tmp_path / 'b.json', ramp_gap_seconds=5, ramp_window_seconds=0.01)
    time.sleep(0.05)
    _acquire_times(pool, 4)
    assert time.time() - start < 2


def test_ramp_up_config_validation():
    base = json.loads((config.PROFILES / 'v1.json').read_text())
    for good in ({'gap_seconds': 20, 'window_seconds': 900}, {'gap_seconds': 0, 'window_seconds': 0}):
        value = copy.deepcopy(base); value['operations']['ramp_up'] = good
        config.validate(value)
    for bad in ({'gap_seconds': 20}, {'gap_seconds': -1, 'window_seconds': 900}, {'gap_seconds': 20, 'window_seconds': 9000},
                {'gap_seconds': '20', 'window_seconds': 900}, {'gap_seconds': True, 'window_seconds': 900}, [20, 900]):
        value = copy.deepcopy(base); value['operations']['ramp_up'] = bad
        with pytest.raises(config.ConfigError):
            config.validate(value)
