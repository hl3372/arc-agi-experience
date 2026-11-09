"""--pool-extend-tiers: extend_from_records(tiers=True) puts the never-drawn bank tasks first (identical to tiers=False),
then the tasks drawn fewer times than the most-drawn ones, so the pass in progress finishes before any task is drawn
again; restore_prompt_order passes the flag; the guard."""
import gzip
import inspect
import json
import os
import random
import types

from arcsolver.rl import mixorder as M
from arcsolver.rl import train as TD

SCHED = [(1, 0.0, 0.0)]                               # bank only, no revisits (--mix-schedule 1:0:0)


def make(bank_ids, rest_ids, seed=7):
    ids = list(bank_ids) + list(rest_ids)
    strata = [list(range(len(bank_ids))), list(range(len(bank_ids), len(ids)))]
    return M.FrontierMixOrder([1] * len(ids), strata, seed, schedule=SCHED, min_gap=1), ids


def run_steps(o, ids, steps, k, out_dir, rng):
    os.makedirs(os.path.join(out_dir, "rollouts"), exist_ok=True)
    for _ in range(steps):
        got = o.take(k)
        res, rows = [], []
        for g, i in enumerate(got):
            c = rng.choice([0, 0, 1, 3, 8])
            res.append((i, c, 8, 0))
            rows += [{"task_id": ids[i], "group": g, "seq": {"outcome": 1.0 if s < c else 0.0}, "finish": "stop"}
                     for s in range(8)]
        o.update(res)
        with gzip.open(os.path.join(out_dir, "rollouts", f"step_{o.t:06d}.jsonl.gz"), "wt") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))


def draws_by_id(out_dir, steps):
    n = {}
    for s in range(1, steps + 1):
        with gzip.open(os.path.join(out_dir, "rollouts", f"step_{s:06d}.jsonl.gz"), "rt") as f:
            seen = {(r["task_id"], r["group"]) for r in map(json.loads, f)}
        for t, _ in seen:
            n[t] = n.get(t, 0) + 1
    return n


STEPS = 7                                              # 30 bank tasks, 6 per step: pass 1 = steps 1-5, pass 2 started


def second_pass_run(tmp_path):
    bank = [f"b{n:03d}" for n in range(30)]
    rest = [f"r{n}" for n in range(4)]
    o, ids = make(bank, rest)
    run_steps(o, ids, STEPS, 6, str(tmp_path), random.Random(1))
    return ids, o.state_dict()


def grown(old_ids, n_new=9):
    bank = [t for t in old_ids if t.startswith("b")] + [f"n{n:03d}" for n in range(n_new)]
    random.Random(3).shuffle(bank)
    return make(bank, [t for t in old_ids if t.startswith("r")])


def test_tiers_finish_the_pass_in_progress(tmp_path):
    old_ids, s = second_pass_run(tmp_path)
    n = draws_by_id(str(tmp_path), STEPS)
    assert max(n.values()) == 2 and sum(v == 2 for v in n.values()) == 12     # 42 draws: 30 once, 12 twice
    new, ids = grown(old_ids)
    info = new.extend_from_records(s, str(tmp_path), STEPS, {t: i for i, t in enumerate(ids)}, seed=7, tiers=True)
    new_ids = {t for t in ids if t.startswith("n")}
    once = {t for t, v in n.items() if v == 1}
    p = [ids[i] for i in new.pending]
    assert set(p[:9]) == new_ids and set(p[9:]) == once and len(p) == len(set(p)) == 9 + 18
    assert info["tiers"] == {0: 9, 1: 18} and info["max_draws"] == 2 and info["at_max_draws"] == 12
    assert info["pending"] == 27
    plain, _ = grown(old_ids)                          # tier 0 is exactly the tiers=False pending
    plain.extend_from_records(s, str(tmp_path), STEPS, {t: i for i, t in enumerate(ids)}, seed=7)
    assert new.pending[:9] == plain.pending and "tiers" not in plain.extend_from_records(
        s, str(tmp_path), STEPS, {t: i for i, t in enumerate(ids)}, seed=7)
    # the draws: all 27 pending tasks once, new ones first, before any task drawn twice comes back
    got = []
    while new.pending:
        x = new.take(6)
        got += [ids[i] for i in x]
        new.update([(i, 0, 8, 0) for i in x])
    assert set(got[:9]) == new_ids and set(got[:27]) == new_ids | once and len(set(got[:27])) == 27
    assert not ({t for t, v in n.items() if v == 2} & set(got[:27]))


def test_tiers_deterministic_and_checkpointed(tmp_path):
    old_ids, s = second_pass_run(tmp_path)
    a, ids = grown(old_ids)
    b, _ = grown(old_ids)
    idx = {t: i for i, t in enumerate(ids)}
    a.extend_from_records(s, str(tmp_path), STEPS, idx, seed=7, tiers=True)
    b.extend_from_records(s, str(tmp_path), STEPS, idx, seed=7, tiers=True)
    assert a.pending == b.pending
    a.update([(i, 1, 8, 0) for i in a.take(6)])
    st = json.loads(json.dumps(a.state_dict()))
    c, _ = grown(old_ids)
    c.load_state_dict(st)                              # later resumes: same pool size, the usual load
    for _ in range(6):
        x, y = a.take(6), c.take(6)
        assert x == y
        a.update([(i, 0, 8, 0) for i in x])
        c.update([(i, 0, 8, 0) for i in y])
    assert a.state_dict() == c.state_dict()


def test_first_pass_tiers_equal_plain(tmp_path):
    """Before any task repeats (max draws 1) tiers add nothing: a first-pass extension is unchanged."""
    bank = [f"b{n:03d}" for n in range(30)]
    o, ids0 = make(bank, ["r0", "r1"])
    run_steps(o, ids0, 3, 6, str(tmp_path), random.Random(2))
    s = o.state_dict()
    a, ids = grown(ids0)
    b, _ = grown(ids0)
    idx = {t: i for i, t in enumerate(ids)}
    ia = a.extend_from_records(s, str(tmp_path), 3, idx, seed=7, tiers=True)
    b.extend_from_records(s, str(tmp_path), 3, idx, seed=7)
    assert a.pending == b.pending and ia["tiers"] == {0: len(a.pending)} and ia["max_draws"] == 1


def test_restore_prompt_order_passes_tiers(tmp_path):
    old_ids, s = second_pass_run(tmp_path)
    new, ids = grown(old_ids)
    logs = []
    args = types.SimpleNamespace(pool_extend=True, pool_extend_tiers=True, pool_order="frontier_mix",
                                 new_pool_on_resume=False, out=str(tmp_path), seed=7)
    TD.restore_prompt_order(args, {"args": {"pool_order": "frontier_mix"}, "order": s}, new, logs.append,
                            pool=[types.SimpleNamespace(task_id=t) for t in ids], start_step=STEPS)
    assert logs and "pool extended" in logs[0] and '"tiers"' in logs[0] and len(new.pending) == 27


def test_tiers_flag_needs_pool_extend():
    src = inspect.getsource(TD)
    assert '"--pool-extend-tiers needs --pool-extend"' in src and 'tiers=getattr(args, "pool_extend_tiers", False)' in src
