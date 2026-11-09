"""--pool-new-per-step: after a --pool-extend the never-drawn bank tasks wait in new_queue and each step takes at most
new_per_step of them inside its bank quota; the rest of the quota comes from pending (with --pool-extend-tiers: the tasks
not yet drawn in the pass in progress) and then the bank order, which never draws a task still waiting in new_queue.
Checkpointed; a resume without the quota drains the queue first; a later extension keeps undrawn new tasks queued."""
import gzip
import inspect
import json
import os
import random
import types

import pytest

from arcsolver.rl import config as RT
from arcsolver.rl import mixorder as M
from arcsolver.rl import train as TD

SCHED = [(1, 0.0, 0.0)]
K = 6                                                   # prompts per step in these tests
STEPS = 7                                               # 30 bank tasks: pass 1 = steps 1-5, pass 2 started (12 drawn twice)


def make(bank_ids, rest_ids, seed=7, new_per_step=0):
    ids = list(bank_ids) + list(rest_ids)
    strata = [list(range(len(bank_ids))), list(range(len(bank_ids), len(ids)))]
    return M.FrontierMixOrder([1] * len(ids), strata, seed, schedule=SCHED, min_gap=1, new_per_step=new_per_step), ids


def write_step(out_dir, step, ids, got, rng):
    res, rows = [], []
    for g, i in enumerate(got):
        c = rng.choice([0, 0, 1, 3, 8])
        res.append((i, c, 8, 0))
        rows += [{"task_id": ids[i], "group": g, "seq": {"outcome": 1.0 if s < c else 0.0}, "finish": "stop"}
                 for s in range(8)]
    os.makedirs(os.path.join(out_dir, "rollouts"), exist_ok=True)
    with gzip.open(os.path.join(out_dir, "rollouts", f"step_{step:06d}.jsonl.gz"), "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    return res


def run(o, ids, steps, out_dir, rng):
    for _ in range(steps):
        got = o.take(K)
        o.update(write_step(out_dir, o.t, ids, got, rng))


def counts(out_dir, steps):
    n = {}
    for s in range(1, steps + 1):
        with gzip.open(os.path.join(out_dir, "rollouts", f"step_{s:06d}.jsonl.gz"), "rt") as f:
            for t, _ in {(r["task_id"], r["group"]) for r in map(json.loads, f)}:
                n[t] = n.get(t, 0) + 1
    return n


def base(tmp_path):
    o, ids = make([f"b{n:03d}" for n in range(30)], ["r0", "r1", "r2"])
    run(o, ids, STEPS, str(tmp_path), random.Random(1))
    return ids, o.state_dict()


def grown(old_ids, new_names, new_per_step):
    bank = [t for t in old_ids if not t.startswith("r")] + list(new_names)
    random.Random(3).shuffle(bank)
    return make(bank, [t for t in old_ids if t.startswith("r")], new_per_step=new_per_step)


NEW9 = [f"n{n:03d}" for n in range(9)]


def test_new_tasks_spread_at_most_q_per_step(tmp_path):
    old_ids, s = base(tmp_path)
    n0 = counts(str(tmp_path), STEPS)
    o, ids = grown(old_ids, NEW9, new_per_step=2)
    info = o.extend_from_records(s, str(tmp_path), STEPS, {t: i for i, t in enumerate(ids)}, seed=7, tiers=True)
    assert info["new_queue"] == 9 and info["pending"] == 18 and info["tiers"] == {0: 9, 1: 18}
    assert {ids[i] for i in o.new_queue} == set(NEW9)
    assert {ids[i] for i in o.pending} == {t for t, v in n0.items() if v == 1}
    per_step_new, drawn_before_pass_done = [], []
    for _ in range(5):                                   # 5 steps x 6 = 30 = 9 new + 18 behind + 3 from the bank order
        waiting = set(o.new_queue)
        got = o.take(K)
        names = [ids[i] for i in got]
        assert len(set(got)) == len(got)
        per_step_new.append(o.stats["frontier_mix/fresh_new"])
        assert sum(i in waiting for i in got) == per_step_new[-1]   # a waiting task only comes via the quota
        drawn_before_pass_done += names
        o.update([(i, 0, 8, 0) for i in got])
    assert per_step_new == [2, 2, 2, 2, 1]               # spread: at most 2 per step until the 9 are drawn
    twice = {t for t, v in n0.items() if v == 2}
    first27 = drawn_before_pass_done[:27]
    # every new and every behind task is drawn before any task that was already drawn twice
    assert not (twice & set(drawn_before_pass_done[:24]))
    assert set(NEW9) <= set(drawn_before_pass_done) and len(set(first27)) == 27


def test_bank_order_never_draws_a_waiting_new_task(tmp_path):
    old_ids, s = base(tmp_path)
    o, ids = grown(old_ids, NEW9, new_per_step=1)
    o.extend_from_records(s, str(tmp_path), STEPS, {t: i for i, t in enumerate(ids)}, seed=7)   # no tiers: pending []
    assert o.pending == [] and len(o.new_queue) == 9
    first_seen = []
    for _ in range(9):
        waiting = set(o.new_queue)
        got = o.take(K)
        fresh = [ids[i] for i in got if i in waiting]
        assert len(fresh) == 1 == o.stats["frontier_mix/fresh_new"]   # a waiting task only via the quota
        first_seen += fresh
        o.update([(i, 0, 8, 0) for i in got])
    assert sorted(first_seen) == sorted(NEW9) and o.new_queue == []


def test_quota_zero_is_the_drain_first_behaviour(tmp_path):
    old_ids, s = base(tmp_path)
    a, ids = grown(old_ids, NEW9, new_per_step=0)
    b, _ = grown(old_ids, NEW9, new_per_step=0)
    idx = {t: i for i, t in enumerate(ids)}
    a.extend_from_records(s, str(tmp_path), STEPS, idx, seed=7, tiers=True)
    b.extend_from_records(s, str(tmp_path), STEPS, idx, seed=7, tiers=True)
    assert a.new_queue == [] and "new_queue" not in a.state_dict() and a.pending == b.pending
    got = a.take(K) + a.take(K)
    assert {ids[i] for i in got[:9]} == set(NEW9)        # all new first, as --pool-extend-tiers alone


def test_checkpoint_round_trip_and_resume_without_quota(tmp_path):
    old_ids, s = base(tmp_path)
    a, ids = grown(old_ids, NEW9, new_per_step=2)
    a.extend_from_records(s, str(tmp_path), STEPS, {t: i for i, t in enumerate(ids)}, seed=7, tiers=True)
    a.update([(i, 1, 8, 0) for i in a.take(K)])
    st = json.loads(json.dumps(a.state_dict()))
    assert st["new_queue"] == a.new_queue and st["params"]["new_per_step"] == 2 and len(st["new_queue"]) == 7
    c, _ = grown(old_ids, NEW9, new_per_step=2)
    c.load_state_dict(st)
    for _ in range(3):
        x, y = a.take(K), c.take(K)
        assert x == y
        a.update([(i, 0, 8, 0) for i in x])
        c.update([(i, 0, 8, 0) for i in y])
    assert a.state_dict() == c.state_dict()
    d, _ = grown(old_ids, NEW9, new_per_step=0)          # resumed without the flag: the waiting tasks go first
    d.load_state_dict(st)
    assert d.new_queue == [] and d.pending[:7] == st["new_queue"]


def test_second_extension_keeps_undrawn_new_tasks_queued(tmp_path):
    old_ids, s = base(tmp_path)
    o, ids = grown(old_ids, NEW9, new_per_step=2)
    o.extend_from_records(s, str(tmp_path), STEPS, {t: i for i, t in enumerate(ids)}, seed=7, tiers=True)
    rng = random.Random(5)
    for _ in range(2):                                   # 4 of the 9 new tasks drawn, records written
        got = o.take(K)
        o.update(write_step(str(tmp_path), o.t, ids, got, rng))
    s2 = o.state_dict()
    more = [f"m{n:03d}" for n in range(5)]
    o2, ids2 = grown(ids, more, new_per_step=2)
    info = o2.extend_from_records(s2, str(tmp_path), STEPS + 2, {t: i for i, t in enumerate(ids2)}, seed=7, tiers=True)
    waiting = {ids2[i] for i in o2.new_queue}
    drawn_new = {ids[i] for i in range(len(ids)) if ids[i].startswith("n")} - {ids[i] for i in o.new_queue}
    assert waiting == (set(NEW9) - drawn_new) | set(more) and info["new_queue"] == 5 + 5


def test_flag_wiring():
    src_tr = inspect.getsource(RT)
    src_td = inspect.getsource(TD)
    assert 'new_per_step=getattr(args, "pool_new_per_step", 0)' in src_tr
    assert '"--pool-new-per-step"' in src_td and "needs --pool-order frontier_mix" in src_td
    with pytest.raises(ValueError):
        M.FrontierMixOrder([1, 1], [[0], [1]], 0, new_per_step=-1)


def test_restore_prompt_order_with_quota(tmp_path):
    old_ids, s = base(tmp_path)
    o, ids = grown(old_ids, NEW9, new_per_step=3)
    logs = []
    args = types.SimpleNamespace(pool_extend=True, pool_extend_tiers=True, pool_order="frontier_mix",
                                 new_pool_on_resume=False, out=str(tmp_path), seed=7)
    TD.restore_prompt_order(args, {"args": {"pool_order": "frontier_mix"}, "order": s}, o, logs.append,
                            pool=[types.SimpleNamespace(task_id=t) for t in ids], start_step=STEPS)
    assert '"new_queue": 9' in logs[0] and '"new_per_step": 3' in logs[0] and len(o.pending) == 18
