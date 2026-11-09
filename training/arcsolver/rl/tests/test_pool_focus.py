"""--pool-focus-*: every take holds at least focus_per_step focus tasks: never-drawn ones from new_queue first, then
revisits (fewest draws, then oldest latest draw); fresh draws never pick a focus task; draw counts are seeded from the
rollout records (prefix rows skipped); the shuffled-cycle order."""
import gzip
import inspect
import json
import os

import pytest

from arcsolver.rl import mixorder as M
from arcsolver.rl import train as D

K = 72


def make(n_bank=400, n_rest=20, new_per_step=8):
    strata = [list(range(n_bank)), list(range(n_bank, n_bank + n_rest))]
    return M.FrontierMixOrder([1] * (n_bank + n_rest), strata, 3, schedule=[(1, 0.0, 0.0)], min_gap=1,
                              new_per_step=new_per_step)


def test_quota_new_first_then_revisits():
    o = make()
    focus = list(range(300, 352))                                  # 52 focus tasks
    o.new_queue = focus[:30]                                      # 30 never drawn
    drawn = {i: [1, 100 + (i % 7)] for i in focus[30:]}           # 22 drawn once, at steps 100..106
    o.set_focus(focus, 20, drawn)
    o.t = 134
    a = o.take(K)
    assert len(a) == len(set(a)) == K
    fa = [i for i in a if i in o.focus_set]
    assert len(fa) == 20 and set(fa) <= set(focus[:30])            # all from the never-drawn queue
    assert o.stats["frontier_mix/focus"] == 20 and o.stats["frontier_mix/focus_revisits"] == 0
    b = o.take(K)
    fb = [i for i in b if i in o.focus_set]
    assert len(fb) == 20 and len(set(fb) & set(focus[:30])) == 10  # the last 10 new ones + 10 revisits
    rev = [i for i in fb if i not in focus[:30]]
    assert o.stats["frontier_mix/focus_revisits"] == 10
    # revisits: drawn-once tasks with the oldest latest draw first (steps 100, 101, ...)
    oldest = sorted(focus[30:], key=lambda i: (100 + (i % 7), o._tie(i)))[:10]
    assert sorted(rev) == sorted(oldest)
    for _ in range(10):                                           # keeps cycling the focus set evenly
        c = o.take(K)
        assert sum(i in o.focus_set for i in c) == 20 and len(c) == len(set(c)) == K
    d = [o.focus_draws[i][0] for i in focus]
    assert max(d) - min(d) <= 1                                   # 52 tasks, 240 focus draws: balanced counts


def test_fresh_draws_never_focus_and_off():
    o = make(n_bank=300, new_per_step=0)
    focus = list(range(0, 50))
    o.set_focus(focus, 20, {i: [1, 5] for i in focus})
    o.t = 10
    for _ in range(30):
        a = o.take(40)
        assert sum(i in o.focus_set for i in a) == 20              # exactly the quota: fresh draws skip focus tasks
    p = make(n_bank=300, new_per_step=0)
    p.t = 10
    p.take(40)
    assert p.focus_per_step == 0 and "frontier_mix/focus" not in p.stats


def test_set_focus_rejects_a_quota_the_pool_cannot_fill():
    o = make(n_bank=60, new_per_step=0)
    with pytest.raises(ValueError):
        o.set_focus(list(range(50)), 20, {})                       # 10 non-focus bank tasks cannot fill fresh draws


def test_draws_from_records(tmp_path):
    os.makedirs(tmp_path / "rollouts")
    rows = {
        3: [("t1", 0, None, 'plain'), ("t1", 0, None, 'x'), ("t2", 1, None, 'has "prefix": { inside')],
        5: [("t1", 2, {"cut": 9}, 'p'), ("t3", 4, None, 'y')],
        7: [("t2", 1, None, 'z')],
    }
    for step, rr in rows.items():
        with gzip.open(tmp_path / "rollouts" / f"step_{step:06d}.jsonl.gz", "wt") as f:
            for t, g, pre, text in rr:
                r = {"task_id": t, "source": "bank", "seen": False, "group": g, "seq": {"outcome": 0.0}, "text": text}
                if pre:
                    r["prefix"] = pre
                f.write(json.dumps(r) + "\n")
    idx = {"t1": 10, "t2": 11, "t3": 12}
    d = M.focus_draws_from_records(str(tmp_path), 1, 7, idx)
    assert d == {10: [1, 3], 11: [2, 7], 12: [1, 5]}              # t1: the step-5 prefix row is skipped
    assert M.focus_draws_from_records(str(tmp_path), 6, 7, idx) == {11: [1, 7]}


def test_flags_and_wiring(tmp_path):
    src = inspect.getsource(D)
    assert src.index("restore_prompt_order(args, state, order, log, pool=pool") < src.index("setup_focus(args, order, pool")
    f = tmp_path / "f.json"
    f.write_text(json.dumps(["a.json", "b"]))
    base = ["--out", str(tmp_path / "o"), "--student", "/tmp/none", "--pool-order", "frontier_mix",
            "--prompts-per-step", "96"]
    a = D.parse_args(base + ["--pool-focus-file", str(f), "--pool-focus-per-step", "20"])
    assert a.pool_focus_per_step == 20 and a.pool_focus_from_step == 1
    a = D.parse_args(base + ["--pool-focus-file", str(f), "--pool-focus-per-step", "30", "--pool-focus-order", "shuffle",
                             "--pool-focus-shuffle-from", "163"])
    assert a.pool_focus_order == "shuffle" and a.pool_focus_shuffle_from == 163
    for bad in (["--pool-focus-per-step", "20"], ["--pool-focus-file", str(f), "--pool-focus-per-step", "-1"],
                ["--pool-focus-file", str(f), "--pool-focus-per-step", "500"],
                ["--pool-focus-file", str(f), "--pool-focus-per-step", "30", "--pool-focus-order", "shuffle"]):
        with pytest.raises(SystemExit):
            D.parse_args(base + bad)


def test_shuffle_order_mixes_new_and_drawn_and_resumes():
    """--pool-focus-order shuffle (new and drawn focus tasks shuffled together, not new first): one take per step draws
    the next positions of a seeded shuffled cycle over the whole focus set."""
    o = make(n_bank=600)
    focus = list(range(100, 160))                                 # 60 focus tasks, given in a stable order
    o.new_queue = focus[:30] + [500, 501]                          # 30 never-drawn focus tasks + 2 other new tasks
    o.set_focus(focus, 10, {i: [1, 120] for i in focus[30:]}, order="shuffle", cursor=0)
    assert o.new_queue == [500, 501]                               # the cycle draws the focus tasks, not the queue
    o.t = 162
    takes = [o.take(K) for _ in range(6)]
    fs = [[i for i in a if i in o.focus_set] for a in takes]
    assert all(len(f) == 10 for f in fs) and all(len(a) == len(set(a)) == K for a in takes)
    assert sorted(i for f in fs for i in f) == sorted(focus)       # one full cycle: every focus task exactly once
    assert any(i in focus[:30] for i in fs[0]) and any(i in focus[30:] for i in fs[0])   # mixed, not new first
    assert o.focus_cursor == 60 and o.stats["frontier_mix/focus_cursor"] == 60
    assert 500 in takes[0] + takes[1] or 501 in takes[0] + takes[1]  # other new tasks keep their own quota
    nxt = [i for i in o.take(K) if i in o.focus_set]
    assert nxt == o.focus_cycle(1)[:10] != o.focus_cycle(0)[:10]   # cycle 2 is a new shuffle
    # a restart re-seeded at cursor 30 continues exactly where the cycle was
    r = make(n_bank=600)
    r.new_queue = []
    r.set_focus(focus, 10, {}, order="shuffle", cursor=30)
    r.t = 165
    assert [[i for i in r.take(K) if i in r.focus_set] for _ in range(3)] == fs[3:6]


def test_focus_draw_count_from_records(tmp_path):
    os.makedirs(tmp_path / "rollouts")
    for step, rr in {5: [("a", 0), ("b", 1), ("x", 2)], 6: [("a", 0), ("c", 3)], 7: [("b", 0)]}.items():
        with gzip.open(tmp_path / "rollouts" / f"step_{step:06d}.jsonl.gz", "wt") as f:
            for t, g in rr:
                for _ in range(8):
                    f.write(json.dumps({"task_id": t, "source": "bank", "seen": False, "group": g, "text": "w"}) + "\n")
    idx = {"a": 0, "b": 1, "c": 2}
    assert M.focus_draw_count(str(tmp_path), 5, 7, idx) == 5 and M.focus_draw_count(str(tmp_path), 6, 7, idx) == 3
