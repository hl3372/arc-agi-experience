"""Opt-in step schedule and revisit age cap of the frontier-mix prompt order (e.g. the non-bank share 50% -> 10% over
steps 1-50 -> 0% by step 100, frontier revisits from step 51). Without them the order draws exactly as the
unscheduled class (checkpointed frontier_mix states keep loading and resuming unchanged)."""
import hashlib
import json
import random

import pytest

from arcsolver.rl import config as RT
from arcsolver.rl import mixorder as rmo

BANK, REST = list(range(700)), list(range(700, 800))
COUNTS = [1] * 800
SCHED_EX = [(1, 0.0, 0.5), (50, 0.0, 0.1), (51, 0.2, 0.1), (100, 0.2, 0.0)]


def _run(kw, steps=40, k=64, reload_at=17):
    """Draw `steps` steps with random outcomes and a save/load round trip at `reload_at` (as a resumed run)."""
    o = rmo.FrontierMixOrder(COUNTS, [BANK, REST], 11, **kw)
    rng, seq = random.Random(5), []
    for s in range(steps):
        if s == reload_at:
            st = json.loads(json.dumps(o.state_dict()))
            o = rmo.FrontierMixOrder(COUNTS, [BANK, REST], 11, **kw)
            o.load_state_dict(st)
        got = o.take(k)
        seq.append(got)
        o.update([(i, rng.randint(0, 8), 8, 0) for i in got])
    return seq, o


def _h(seq):
    return hashlib.sha256(json.dumps(seq).encode()).hexdigest()[:16]


@pytest.mark.parametrize("kw, golden", [
    ({}, "12540d8b710c82fa"),
    ({"frontier_share": 0.3, "rest_share": 0.1, "min_gap": 3, "lo": 2, "hi": 5}, "77794e7973be6882"),
])
def test_no_schedule_draws_match_the_original_class(kw, golden):
    # golden = the draws of the class before the schedule existed, incl. the save/load round trip
    seq, o = _run(kw)
    assert _h(seq) == golden
    assert "schedule" not in o.state_dict()["params"] and "max_age" not in o.state_dict()["params"]
    assert "frontier_mix/frontier_share" not in o.metrics()


def test_a_state_without_schedule_keys_loads_and_continues():
    _, o = _run({})
    st = json.loads(json.dumps(o.state_dict()))
    assert set(st["params"]) == {"frontier_share", "rest_share", "min_gap", "lo", "hi"}
    o2 = rmo.FrontierMixOrder(COUNTS, [BANK, REST], 11, schedule=SCHED_EX, max_age=25)
    o2.load_state_dict(st)
    assert o2.t == o.t and o2.queue == o.queue


def test_schedule_shares_and_quotas():
    o = rmo.FrontierMixOrder(COUNTS, [BANK, REST], 11, schedule=SCHED_EX)
    assert o.shares(1) == (0.0, 0.5) and o.shares(0) == (0.0, 0.5)
    f, r = o.shares(25)
    assert f == 0.0 and r == pytest.approx(0.5 - 0.4 * 24 / 49)
    assert o.shares(50) == (0.0, 0.1)
    assert o.shares(51) == (0.2, 0.1)
    assert o.shares(100) == (0.2, 0.0) and o.shares(500) == (0.2, 0.0)
    assert o.quotas(64, 1) == (32, 0, 32)
    assert o.quotas(64, 50) == (58, 0, 6)
    assert o.quotas(64, 51) == (45, 13, 6)
    assert o.quotas(64, 100) == (51, 13, 0)


def test_schedule_steers_draws_and_resumes_exactly():
    seq, o = _run({"schedule": SCHED_EX, "max_age": 25}, steps=60, reload_at=33)
    ref, _ = _run({"schedule": SCHED_EX, "max_age": 25}, steps=60, reload_at=10 ** 9)
    assert seq == ref
    rest = set(REST)
    assert sum(i in rest for i in seq[0]) == 32                       # step 1: half non-bank
    assert sum(i in rest for i in seq[49]) == 6                       # step 50
    assert o.metrics()["frontier_mix/frontier_share"] == 0.2
    assert o.state_dict()["params"]["schedule"] == [list(x) for x in SCHED_EX]


def test_no_revisits_while_the_frontier_share_is_zero_but_the_queue_builds():
    o = rmo.FrontierMixOrder(COUNTS, [BANK, REST], 11, schedule=SCHED_EX)
    rng = random.Random(1)
    for _ in range(50):
        got = o.take(64)
        assert o.metrics()["frontier_mix/revisits"] == 0
        o.update([(i, rng.randint(1, 6), 8, 0) for i in got])
    assert len(o.queue) == len(COUNTS)                                # every task drawn landed 1-6 of 8
    o.take(64)
    assert o.metrics()["frontier_mix/revisits"] == 13


def test_max_age_only_revisits_recent_queue_entries():
    o = rmo.FrontierMixOrder(COUNTS, [BANK, REST], 11, frontier_share=0.5, rest_share=0.0, min_gap=2, max_age=4)
    o.seed_history([(1, [(10, 3, 8, 0)]), (5, [(11, 3, 8, 0)]), (6, [(12, 3, 8, 0)])])
    got = o.take(4)                                                   # step 7: entries from steps 3..5 are revisitable
    assert got[:1] == [11] and 10 not in got and 12 not in got
    assert o.metrics()["frontier_mix/queue_ready"] == 1


def test_bad_schedules_are_rejected():
    for bad in ([(5, 0, 0.5), (5, 0, 0.1)], [(1, 0.6, 0.5)], [(1, -0.1, 0.1)]):
        with pytest.raises(ValueError):
            rmo.FrontierMixOrder(COUNTS, [BANK, REST], 11, schedule=bad)
    with pytest.raises(ValueError):
        rmo.FrontierMixOrder(COUNTS, [BANK, REST], 11, max_age=-1)


def test_parse_mix_schedule():
    assert RT.parse_mix_schedule("") is None
    assert RT.parse_mix_schedule("1:0:0.5,50:0:0.1,51:0.2:0.1,100:0.2:0") == SCHED_EX
    with pytest.raises(ValueError):
        RT.parse_mix_schedule("1:0.5")
