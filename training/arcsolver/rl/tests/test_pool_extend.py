"""--pool-extend: FrontierMixOrder.extend_from_records continues a frontier_mix checkpoint on a grown bank: never-drawn
bank tasks first (pending, seeded shuffle), the revisit queue re-seeded from the rollout records by task id, unchanged
strata continued, pending checkpointed; train.restore_prompt_order's branch."""
import gzip
import json
import os
import random
import types

import pytest

from arcsolver.rl import mixorder as M
from arcsolver.rl import train as TD

SCHED = [(1, 0.0, 0.0), (3, 0.0, 0.0), (4, 0.3, 0.0)]


def make(bank_ids, rest_ids, seed=7):
    """An order over pool = bank ids then rest ids (index = position), as config.make_order builds it."""
    ids = list(bank_ids) + list(rest_ids)
    strata = [list(range(len(bank_ids))), list(range(len(bank_ids), len(ids)))]
    o = M.FrontierMixOrder([1] * len(ids), strata, seed, schedule=SCHED, max_age=25, min_gap=1)
    return o, ids


def run_steps(o, ids, steps, k, out_dir, rng):
    """take + update for `steps` steps, writing each step's rollout record like train.py (8 samples/group)."""
    os.makedirs(os.path.join(out_dir, "rollouts"), exist_ok=True)
    for _ in range(steps):
        got = o.take(k)
        res, rows = [], []
        for g, i in enumerate(got):
            c = rng.choice([0, 0, 1, 3, 5, 8])
            res.append((i, c, 8, 0))
            for s in range(8):
                rows.append({"task_id": ids[i], "group": g, "seq": {"outcome": 1.0 if s < c else 0.0},
                             "finish": "stop"})
        o.update(res)
        with gzip.open(os.path.join(out_dir, "rollouts", f"step_{o.t:06d}.jsonl.gz"), "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def old_run(tmp_path, steps=4):
    bank = [f"b{n:03d}" for n in range(30)]
    rest = [f"r{n}" for n in range(5)]
    o, ids = make(bank, rest)
    run_steps(o, ids, steps, 6, str(tmp_path), random.Random(1))
    return o, ids, o.state_dict()


def grown_pool(old_ids, n_new=12):
    bank = [t for t in old_ids if t.startswith("b")] + [f"n{n:03d}" for n in range(n_new)]
    random.Random(3).shuffle(bank)                     # the grown pool re-indexes every task
    rest = [t for t in old_ids if t.startswith("r")]
    return make(bank, rest)


def drawn_ids(out_dir, steps):
    out = set()
    for s in range(1, steps + 1):
        with gzip.open(os.path.join(out_dir, "rollouts", f"step_{s:06d}.jsonl.gz"), "rt") as f:
            out |= {json.loads(line)["task_id"] for line in f}
    return out


def test_extend_restores_queue_and_draws_unseen_first(tmp_path):
    old, old_ids, s = old_run(tmp_path)
    new, ids = grown_pool(old_ids)
    index_of = {t: i for i, t in enumerate(ids)}
    info = new.extend_from_records(s, str(tmp_path), 4, index_of, seed=7)
    assert new.t == 4 and info["t"] == 4 and info["bank"] == 42 and info["bank_before"] == 30
    # the queue, by task id, is the checkpoint's
    assert {ids[i]: v for i, v in new.queue.items()} == {old_ids[i]: v for i, v in old.queue.items()}
    seen = drawn_ids(str(tmp_path), 4)
    bank_ids = {t for t in ids if not t.startswith("r")}
    assert {ids[i] for i in new.pending} == bank_ids - seen and info["pending"] == len(bank_ids - seen)
    assert new.orders[1].state_dict() == s["orders"][1]          # the unchanged stratum continues
    # fresh bank draws: every pending task once, before any other bank task
    fresh = []
    while new.pending:
        got = new.take(6)
        fresh += [ids[i] for i in got if new.last[i] == "bank"]
        new.update([(i, 0, 8, 0) for i in got])
    unseen = bank_ids - seen
    assert len(fresh) >= len(unseen) and set(fresh[:len(unseen)]) == unseen and len(set(fresh[:len(unseen)])) == len(unseen)


def test_extend_is_deterministic_and_pending_checkpoints(tmp_path):
    _, old_ids, s = old_run(tmp_path)
    a, ids = grown_pool(old_ids)
    b, _ = grown_pool(old_ids)
    index_of = {t: i for i, t in enumerate(ids)}
    a.extend_from_records(s, str(tmp_path), 4, index_of, seed=7)
    b.extend_from_records(s, str(tmp_path), 4, index_of, seed=7)
    assert a.pending == b.pending and a.queue == b.queue
    a.update([(i, 2, 8, 0) for i in a.take(6)])                  # one step in, then a checkpoint
    st = json.loads(json.dumps(a.state_dict()))
    assert st["pending"] == a.pending and len(st["pending"]) > 0
    c, _ = grown_pool(old_ids)
    c.load_state_dict(st)
    for _ in range(4):
        x, y = a.take(6), c.take(6)
        assert x == y
        a.update([(i, 1, 8, 0) for i in x])
        c.update([(i, 1, 8, 0) for i in y])
    assert a.state_dict() == c.state_dict()


def test_no_pending_key_without_extension(tmp_path):
    o, _, s = old_run(tmp_path)
    assert "pending" not in s and o.pending == []


def test_missing_record_raises(tmp_path):
    _, old_ids, s = old_run(tmp_path)
    os.remove(os.path.join(str(tmp_path), "rollouts", "step_000002.jsonl.gz"))
    new, ids = grown_pool(old_ids)
    with pytest.raises(ValueError, match="no rollout records for steps \\[2\\]"):
        new.extend_from_records(s, str(tmp_path), 4, {t: i for i, t in enumerate(ids)}, seed=7)


def test_restore_prompt_order_branch(tmp_path):
    _, old_ids, s = old_run(tmp_path)
    new, ids = grown_pool(old_ids)
    pool = [types.SimpleNamespace(task_id=t) for t in ids]
    logs = []
    args = types.SimpleNamespace(pool_extend=True, pool_order="frontier_mix", new_pool_on_resume=False,
                                 out=str(tmp_path), seed=7)
    TD.restore_prompt_order(args, {"args": {"pool_order": "frontier_mix"}, "order": s}, new, logs.append, pool=pool,
                            start_step=4)
    assert logs and "pool extended" in logs[0] and new.t == 4 and new.pending
    same, _ = make([t for t in old_ids if t.startswith("b")], [t for t in old_ids if t.startswith("r")])
    TD.restore_prompt_order(args, {"args": {"pool_order": "frontier_mix"}, "order": s}, same, logs.append,
                            pool=None, start_step=4)                   # same size: the usual load, no extension
    assert same.state_dict() == s and len(logs) == 1
