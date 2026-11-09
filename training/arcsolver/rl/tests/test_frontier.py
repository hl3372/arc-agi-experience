"""frontier.FrontierOrder: draws by each task's last group outcome (--pool-order frontier)."""
import json
from collections import Counter

import pytest

from arcsolver.rl.frontier import FrontierOrder


def test_take_is_distinct_seeded_and_covers_unseen_pool():
    a, b = FrontierOrder(100, seed=3, group_size=8), FrontierOrder(100, seed=3, group_size=8)
    for _ in range(5):
        x, y = a.take(16), b.take(16)
        assert x == y and len(set(x)) == 16
    assert len(set(FrontierOrder(5, 0, 8).take(16))) == 5          # k > n: every item once


def test_states_from_group_outcomes():
    o = FrontierOrder(5, 0, 8)
    o.take(5)
    o.update([(0, 3, 8, 2), (1, 8, 8, 0), (2, 0, 8, 7), (3, 0, 8, 1), (4, 0, 8, 4)])
    assert o.state == ["frontier", "solved", "dead", "failed", "dead"]       # trunc_hi .5: 4/8 counts as dead
    assert o.counts() == {"unseen": 0, "frontier": 1, "solved": 1, "dead": 2, "failed": 1}
    with pytest.raises(ValueError):
        o.update([(0, 9, 8, 0)])


def test_frontier_tasks_are_preferred_and_dead_ones_revisited_later():
    o = FrontierOrder(40, seed=1, group_size=8, revisit_after=50)
    o.take(40)
    o.update([(i, 4, 8, 0) if i < 10 else (i, 0, 8, 8) for i in range(40)])   # 10 frontier, 30 dead
    picks = Counter()
    for _ in range(25):                                        # within revisit_after: dead tasks stay down-weighted
        picks.update(o.take(8))
    front = sum(picks[i] for i in range(10)) / 10
    dead = sum(picks[i] for i in range(10, 40)) / 30
    assert front > 4 * dead
    assert o.weight(10) < 0.6 * o.p["w_unseen"]
    for _ in range(30):                                        # 55 steps after the result: back at full weight
        o.take(8)
    assert o.weight(10) == pytest.approx(o.p["w_unseen"])


def test_state_dict_roundtrip_resumes_identically_and_rejects_other_kinds():
    o = FrontierOrder(30, seed=7, group_size=8)
    for s in range(3):
        items = o.take(6)
        o.update([(i, (i + s) % 9, 8, i % 3) for i in items])
    saved = json.loads(json.dumps(o.state_dict()))
    r = FrontierOrder(30, seed=7, group_size=8)
    r.load_state_dict(saved)
    assert r.take(6) == o.take(6)
    with pytest.raises(ValueError, match="weighted"):
        r.load_state_dict({"order": {}, "carry": [], "n_expanded": 30})      # a WeightedOrder checkpoint
    with pytest.raises(ValueError, match="mismatch"):
        FrontierOrder(31, seed=7, group_size=8).load_state_dict(saved)
    with pytest.raises(ValueError, match="unknown"):
        FrontierOrder(3, 0, 8, w_bogus=1.0)
