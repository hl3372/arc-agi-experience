"""--pool-order frontier in the trainer: flags and their validation, make_order (the default weighted path unchanged),
the per-group outcome builder fed to FrontierOrder.update, and rollout_step -> update -> state_dict resume with a fake
sampler."""
import json
from types import SimpleNamespace as NS

import pytest

from arcsolver.prompt import arc_render as ar
from arcsolver.rl import config as RT
from arcsolver.rl import frontier as rf
from arcsolver.rl import pool as rp
from arcsolver.rl import rewards as rr
from arcsolver.rl import train as rd

from .helpers import G, OUTCOME, rollout_harness

OUT = ["--out", "/tmp/frontier-integration-test", "--student", "/tmp/none"]
STATES = {"unseen", "frontier", "solved", "dead", "failed"}
EXPECT = {0: "frontier", 1: "solved", 2: "dead", 3: "failed", 4: "frontier", 5: "dead"}


# --- flags -----------------------------------------------------------------------------------------------------------

def test_flags_default_to_weighted_and_frontier_defaults():
    a = RT.parse_args(OUT)
    assert a.pool_order == "weighted"
    assert RT.frontier_params(a) == {k: rf.DEFAULTS[k] for k in RT.FRONTIER_PARAMS}
    assert set(RT.FRONTIER_PARAMS) == set(rf.DEFAULTS) - {"w_unseen"}
    a = RT.parse_args(OUT + ["--bank-share", "0.5", "--pool-oversample", "3"])
    assert a.pool_order == "weighted" and a.bank_share == 0.5 and a.pool_oversample == 3


def test_frontier_flags_parse_and_validate():
    a = RT.parse_args(OUT + ["--pool-order", "frontier", "--frontier-w-frontier", "3", "--frontier-w-solved", "0",
                             "--frontier-w-dead", "0.05", "--frontier-w-failed", "0.4", "--frontier-trunc-hi", "0.75",
                             "--frontier-revisit-after", "50"])
    assert RT.frontier_params(a) == dict(w_frontier=3.0, w_solved=0.0, w_dead=0.05, w_failed=0.4, trunc_hi=0.75,
                                         revisit_after=50)
    assert isinstance(a.frontier_revisit_after, int) and isinstance(a.frontier_w_frontier, float)
    fr = OUT + ["--pool-order", "frontier"]
    for bad in (["--bank-share", "0.5"], ["--pool-oversample", "2"], ["--frontier-trunc-hi", "0"],
                ["--frontier-trunc-hi", "1.5"], ["--frontier-w-dead", "-0.1"], ["--frontier-revisit-after", "0"]):
        with pytest.raises(SystemExit):
            RT.parse_args(fr + bad)
    with pytest.raises(SystemExit):
        RT.parse_args(OUT + ["--pool-order", "curriculum"])
    assert rd.parse_args(OUT + ["--pool-order", "frontier"]).pool_order == "frontier"


# --- make_order -------------------------------------------------------------------------------------------------------

def _pool(n_bank=6, n_arc=4):
    return [NS(source="bank")] * n_bank + [NS(source="arc_train")] * n_arc


def _dump(order):
    return json.dumps(order.state_dict())


def test_make_order_default_path_is_the_weighted_order():
    pool, counts = _pool(), [2, 1, 1, 3, 1, 1, 1, 1, 2, 1]
    for args in (RT.parse_args(OUT + ["--seed", "3"]), NS(bank_share=0.0, seed=3)):   # callers without pool_order
        o = RT.make_order(args, pool, counts)
        ref = rp.WeightedOrder(counts, 3)
        assert type(o) is rp.WeightedOrder and _dump(o) == _dump(ref)
        assert [o.take(4) for _ in range(5)] == [ref.take(4) for _ in range(5)] and _dump(o) == _dump(ref)
    a = RT.parse_args(OUT + ["--seed", "3", "--bank-share", "0.6"])
    o = RT.make_order(a, pool, counts)
    ref = rp.StratifiedOrder(counts, [list(range(6)), list(range(6, 10))], [0.6, 1.0 - 0.6], 3)
    assert type(o) is rp.StratifiedOrder and _dump(o) == _dump(ref)
    assert [o.take(5) for _ in range(3)] == [ref.take(5) for _ in range(3)] and _dump(o) == _dump(ref)


def test_make_order_builds_a_frontier_order_only_when_asked():
    a = RT.parse_args(OUT + ["--seed", "5", "--group-size", "4", "--pool-order", "frontier", "--frontier-w-dead", "0.05"])
    o = RT.make_order(a, _pool(), [1] * 10)
    assert isinstance(o, rf.FrontierOrder)
    s = o.state_dict()
    assert (s["kind"], s["n"], s["seed"], s["group_size"]) == (rf.KIND, 10, 5, 4)
    assert s["params"] == {**rf.DEFAULTS, "w_dead": 0.05}
    assert len(set(o.take(8))) == 8


# --- group outcomes -> FrontierOrder.update ----------------------------------------------------------------------------

def _r(group, pool_index, outcome, finish="stop", prefix=None):
    # decoys: em_all and r_seq contradict the outcome reward, which alone decides "correct"
    r = {"group": group, "pool_index": pool_index, "finish": finish, "seq": {"outcome": outcome},
         "r_seq": float(not outcome), "stats": {"em_all": not outcome, "finish": finish}}
    if prefix is not None:
        r["prefix"] = prefix
    return r


def test_frontier_results_count_correct_and_truncated_per_group():
    rs = ([_r(0, 7, float(i < 2), "length" if i in (5, 6) else "stop") for i in range(8)]    # 2/8 right, 2 cut
          + [_r(1, 3, 0.0, "length" if i < 5 else "stop") for i in range(7)]
          + [_r(2, 11, 0.25) for _ in range(8)]                                          # partial credit is > 0
          + [_r(3, 4, 1.0, prefix={"cut": 3000}) for _ in range(8)])                     # prefix group: not a draw
    rs = rs[1::2] + rs[0::2]                                                                 # order does not matter
    res = rd.frontier_results(rs)
    assert res == [(7, 2, 8, 2), (3, 0, 7, 5), (11, 8, 8, 0)]
    o = rf.FrontierOrder(12, 0, 8)
    o.take(4)
    o.update(res)
    assert [o.state[i] for i in (7, 3, 11, 4)] == ["frontier", "dead", "solved", "unseen"]   # 5/7 cut >= trunc_hi
    assert rd.frontier_results([]) == []


def _scored(task, text, finish, cfg, group, pool_index):
    tt = rr.TokenText(text, list(range(len(text))), list(range(1, len(text) + 1)))
    return {"group": group, "pool_index": pool_index, "finish": finish, **rr.score_rollout(task, tt, finish, cfg)}


def test_frontier_results_follow_the_outcome_reward_not_em_all_or_r_seq():
    ar.configure_arm(RT.parse_args(OUT).arm)
    t2 = {"train": [{"input": [[0]], "output": [[1]]}],
          "test": [{"input": [[0]], "output": [[1]]}, {"input": [[0]], "output": [[2]]}]}
    both, half, none = (f"Rule.\n</think>\n\n{x}" for x in ("1\n\n2", "1\n\n3", "3\n\n3"))
    partial = rr.RewardConfig(outcome_all_tests=False, beta_loop=0)          # the default: em_frac
    strict = rr.RewardConfig(outcome_all_tests=True, beta_loop=0)
    no_stop = rr.RewardConfig(outcome_all_tests=True, outcome_require_stop=False, beta_loop=0)
    fmt = rr.RewardConfig(outcome_all_tests=True, w_format=0.5, beta_loop=0)
    rs = ([_scored(t2, half, "stop", partial, 0, 10) for _ in range(4)]                   # equal partial credit
          + [_scored(t2, x, "stop", partial, 1, 11) for x in (half, half, none, none)]    # 0.5 vs 0: mixed
          + [_scored(t2, half, "stop", strict, 2, 12) for _ in range(4)]                  # all tests: 0.5 -> 0
          + [_scored(t2, both, "length", strict, 3, 13) for _ in range(4)]                # right but cut: 0 ...
          + [_scored(t2, both, "length", no_stop, 4, 14) for _ in range(4)]               # ... unless EOS not needed
          + [_scored(t2, none, "stop", fmt, 5, 15) for _ in range(4)])                    # format reward only
    assert [r["seq"]["outcome"] for r in rs[:8]] == [0.5] * 6 + [0.0] * 2
    assert all(r["stats"]["em_all"] for r in rs[12:20])                       # em_all would call groups 3/4 solved
    assert all(r["r_seq"] == 0.5 and r["seq"]["outcome"] == 0 for r in rs[20:])
    want = [(10, 4, 4, 0), (11, 2, 4, 0), (12, 0, 4, 0), (13, 0, 4, 4), (14, 4, 4, 4), (15, 0, 4, 0)]
    assert rd.frontier_results(rs) == want
    for r in rs:
        r["r_seq"] = 0.123                                                     # length shaping rewrites r_seq only
    assert rd.frontier_results(rs) == want
    o = rf.FrontierOrder(16, 0, 4)
    o.take(6)
    o.update(want)
    assert [o.state[i] for i in range(10, 16)] == ["solved", "frontier", "failed", "dead", "solved", "failed"]


# --- rollout_step end to end (fake sampler) ---------------------------------------------------------------------------

class Recorder:
    """Wraps an order, recording what it hands out and what it is told."""

    def __init__(self, order):
        self.order, self.taken, self.updates = order, [], []

    def take(self, k):
        self.taken.append(self.order.take(k))
        return self.taken[-1]

    def update(self, results):
        self.updates.append(list(results))
        self.order.update(results)


def test_rollout_step_feeds_frontier_order_and_resumes_from_state_dict(monkeypatch):
    a, pool, step = rollout_harness(monkeypatch, ["--pool-order", "frontier"])
    rec = Recorder(RT.make_order(a, pool, [1] * len(pool)))
    for n in (1, 2):
        rollouts = step(rec, n)[0]
        assert [r["pool_index"] for r in rollouts] == [i for i in rec.taken[-1] for _ in range(G)]
    for taken, upd in zip(rec.taken, rec.updates):
        assert len(set(taken)) == 3
        assert upd == [(k, OUTCOME[k][0], G, OUTCOME[k][1]) for k in taken]
    o = rec.order
    seen = {k for t in rec.taken for k in t}
    assert {k: o.state[k] for k in seen} == {k: EXPECT[k] for k in seen}
    assert all(o.state[k] == "unseen" for k in set(OUTCOME) - seen)
    assert o.epoch == pytest.approx(6 / 6) and sum(o.counts().values()) == len(pool)
    assert {f"frontier/n_{s}" for s in o.counts()} == {f"frontier/n_{s}" for s in STATES}    # the metric names
    # checkpoint after step 2 (state.json is JSON), resume into a fresh order, run steps 3-4 on both
    state = json.loads(json.dumps({"step": 2, "order": o.state_dict(), "args": vars(a)}))
    resumed = RT.make_order(a, pool, [1] * len(pool))
    logs = []
    rd.restore_prompt_order(a, state, resumed, logs.append)
    assert not logs and resumed.state_dict() == state["order"]                 # what --restore-only compares
    rec2 = Recorder(resumed)
    for n in (3, 4):
        step(rec, n)
        step(rec2, n)
    assert rec2.taken == rec.taken[2:] and rec2.updates == rec.updates[2:]
    assert _dump(rec2.order) == _dump(rec.order)


def test_weighted_rollout_step_never_updates_the_order(monkeypatch):
    a, pool, step = rollout_harness(monkeypatch)
    order = RT.make_order(a, pool, [1] * len(pool))
    ref = rp.WeightedOrder([1] * len(pool), a.seed)
    rollouts = step(order, 1)[0]
    assert [r["pool_index"] for r in rollouts] == [i for i in ref.take(3) for _ in range(G)]
    assert not hasattr(order, "update") and _dump(order) == _dump(ref)


def test_rollout_step_advantages_and_dropped_groups(monkeypatch):
    """GRPO std advantages per group on the fake sampler's binary outcomes; all-right / all-wrong groups dropped."""
    a, pool, step = rollout_harness(monkeypatch, ["--drop-zero-var", "outcome", "--beta-loop", "0"])
    order = rp.WeightedOrder([1] * len(pool), a.seed)
    rollouts, kept, *_ = step(order, 1)
    for gi in {r["group"] for r in rollouts}:
        rs = [r for r in rollouts if r["group"] == gi]
        k = OUTCOME[rs[0]["pool_index"]][0]
        if k in (0, G):
            assert all(r["dropped"] and r["seq_adv"] == 0.0 for r in rs)
        else:
            assert not any(r["dropped"] for r in rs) and sum(r["seq_adv"] for r in rs) == pytest.approx(0.0, abs=1e-9)
    assert all(not r["dropped"] for r in kept)


def test_restore_prompt_order_handles_a_pool_order_switch():
    loaded = []
    order = NS(load_state_dict=loaded.append)
    weighted_state = {"step": 5, "args": {"pool_file": "p.json"}, "order": {"order": {}, "carry": []}}
    frontier_state = {"step": 5, "args": {"pool_file": "p.json", "pool_order": "frontier"}, "order": {"kind": rf.KIND}}
    args = NS(new_pool_on_resume=False, pool_file="p.json", pool_order="weighted")
    rd.restore_prompt_order(args, weighted_state, order, print)               # unchanged: loads the saved order
    assert loaded == [weighted_state["order"]]
    with pytest.raises(SystemExit, match="new-pool-on-resume"):
        rd.restore_prompt_order(args, frontier_state, order, print)           # frontier -> weighted: no KeyError
    args.pool_order = "frontier"
    rd.restore_prompt_order(args, frontier_state, order, print)
    assert loaded[-1] == frontier_state["order"]
    with pytest.raises(SystemExit, match="new-pool-on-resume"):
        rd.restore_prompt_order(args, weighted_state, order, print)
    logs = []
    args.new_pool_on_resume = True                                             # a switch with the flag: fresh order
    rd.restore_prompt_order(args, weighted_state, order, logs.append)
    assert len(loaded) == 2 and logs == ["new pool on resume: fresh prompt order"]
