"""--prefix-groups: source choice from the previous step (no all-right groups, at most max_zero sources from no-right
groups, one per task, cut in [min_start, R - end_margin] snapped back to a newline), the rebuilt groups (exact prompt +
prefix, the scored task view), dense rewards shifted to the continuation, metrics, and prefix groups kept out of the
order's draw history."""
import gzip
import inspect
import json
import os
import random
import types

import pytest

from arcsolver.rl import augment as ra
from arcsolver.rl import mixorder as M
from arcsolver.rl import train as TD

THINK, NL = 7, 9                                     # fake token ids: </think> and a newline token


def comp(R, closed=True, nl_every=100):
    """A completion of R reasoning tokens (newline token every nl_every), then </think> and 50 answer tokens."""
    c = [NL if (q + 1) % nl_every == 0 else 1 for q in range(R)]
    return c + ([THINK] + [2] * 50 if closed else [])


def rows_and_comps():
    rows, comps, rng = [], [], random.Random(1)
    spec = ([("all_right", 8)] + [(f"zero{z}", 0) for z in range(12)] + [(f"mix{m}", 3) for m in range(20)]
            + [("mix0", 5)])                           # the last group repeats task mix0 (must not be chosen twice)
    for g, (task, k) in enumerate(spec):
        for j in range(8):
            R = rng.randint(2000, 40000)
            rows.append({"task_id": task, "group": g, "seq": {"outcome": 1.0 if j < k else 0.0},
                         "finish": "stop", "aug": {"d8": "id", "perm": "0123456789"}})
            comps.append(comp(R))
    # a prefix group of the previous step: never a source
    for j in range(8):
        rows.append({"task_id": "pre", "group": len(spec), "seq": {"outcome": 0.0}, "prefix": {"cut": 3000}})
        comps.append(comp(30000))
    return rows, comps


def test_choose_prefix_sources_rules():
    rows, comps = rows_and_comps()
    out = TD.choose_prefix_sources(rows, comps, 24, 8, 2500, 3000, THINK, {NL}, random.Random(3))
    assert len(out) == 24
    tasks = [rows[i]["task_id"] for i, _, _ in out]
    assert len(set(tasks)) == 24 and "all_right" not in tasks and "pre" not in tasks
    assert sum(k == 0 for _, _, k in out) <= 8 and sum(t.startswith("zero") for t in tasks) <= 8
    for i, cut, k in out:
        R = comps[i].index(THINK)
        assert 2500 <= cut <= R - 3000
        assert comps[i][cut - 1] == NL                  # snapped to just after a newline (one every 100 tokens)
    again = TD.choose_prefix_sources(rows, comps, 24, 8, 2500, 3000, THINK, {NL}, random.Random(3))
    assert again == out                                   # deterministic per seed


def test_choose_prefix_sources_truncated_and_short():
    rows = [{"task_id": "a", "group": 0, "seq": {"outcome": 0.0}}] * 2 + [{"task_id": "b", "group": 1, "seq": {"outcome": 0.0}}] * 2
    comps = [comp(70000, closed=False), comp(70000, closed=False), comp(5000), comp(5400)]   # b: too short for a cut
    out = TD.choose_prefix_sources(rows, comps, 5, 8, 2500, 3000, THINK, {NL}, random.Random(0))
    assert [rows[i]["task_id"] for i, _, _ in out] == ["a"] and 2500 <= out[0][1] <= 67000


def test_build_prefix_groups_rebuilds_exact_inputs(tmp_path):
    np = pytest.importorskip("numpy")
    rows, comps = rows_and_comps()
    rows = rows[:16 * 8]                                  # all_right + zero0..11 + mix0..2
    comps = comps[:16 * 8]
    rows[9 * 8]["test_idx"] = 1                           # one split source
    for r in rows:
        r["aug"] = {"d8": "rot90", "perm": "0912345678"}
    prompts = [[100 + i % 5] * (3 + i % 4) for i in range(len(rows))]
    os.makedirs(tmp_path / "rollouts")
    with gzip.open(tmp_path / "rollouts" / "step_000004.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    np.savez_compressed(tmp_path / "rollouts" / "step_000004.tokens.npz",
                        prompt=np.concatenate([np.asarray(p, np.int32) for p in prompts]),
                        prompt_len=np.asarray([len(p) for p in prompts]),
                        completion=np.concatenate([np.asarray(c, np.int32) for c in comps]),
                        completion_len=np.asarray([len(c) for c in comps]))
    task = {"train": [{"input": [[1, 0]], "output": [[0, 1]]}],
            "test": [{"input": [[2, 0]], "output": [[0, 2]]}, {"input": [[3]], "output": [[3]]}]}
    names = sorted({r["task_id"] for r in rows})
    pool = [types.SimpleNamespace(task_id=t, path=t) for t in names]
    stok = types.SimpleNamespace(get_vocab=lambda: {"a": 1, "Ċ": NL, "</think>": THINK},
                                 convert_tokens_to_ids=lambda t: THINK)
    TD._TOKEN_SETS.clear()
    args = types.SimpleNamespace(out=str(tmp_path), seed=5, prefix_groups=6, prefix_max_zero=2, prefix_min_start=2500,
                                 prefix_end_margin=3000)
    groups = TD.build_prefix_groups(args, 5, pool, lambda p: task, stok, log=lambda m: None)
    assert 1 <= len(groups) <= 6 and sum(g["meta"]["src_group_right"] == 0 for g in groups) <= 2
    for g in groups:
        m = g["meta"]
        src = next(i for i, r in enumerate(rows) if r["group"] == m["src_group"] and comps[i][:m["cut"]] == g["prefix"]
                   and prompts[i] + comps[i][:m["cut"]] == g["prompt"])
        view = dict(task)
        if rows[src].get("test_idx") is not None:
            view["test"], view["_test_index"], view["_n_test"] = [task["test"][1]], 1, 2
        assert g["task"] == ra.Aug("rot90", (0, 9, 1, 2, 3, 4, 5, 6, 7, 8)).task(view)
        assert g["item"].task_id == rows[src]["task_id"] and m["src_step"] == 4 and m["src_len"] == len(comps[src])


def test_shift_dense_and_metrics():
    sc = {"dense": {3: -1.0, 10: 0.5, 12: 0.25}, "dense_sum": -0.25, "r_seq": 1.0}
    out = TD.shift_dense(sc, 10)
    assert out["dense"] == {0: 0.5, 2: 0.25} and out["dense_sum"] == 0.75 and TD.shift_dense(sc, 0) is sc
    rs = [{"group": 0, "seq": {"outcome": 1.0}, "dropped": False, "prefix": {"src_group_right": 0, "cut": 4000, "src_outcome": 0.0}},
          {"group": 0, "seq": {"outcome": 0.0}, "dropped": False, "prefix": {"src_group_right": 0, "cut": 4000, "src_outcome": 0.0}},
          {"group": 1, "seq": {"outcome": 0.5}}]
    m = TD.prefix_metrics(rs)
    assert m["prefix/groups"] == 1 and m["prefix/from_zero"] == 1 and m["prefix/outcome"] == 0.5
    assert m["prefix/informative"] == 1 and m["reward/outcome_fresh"] == 0.5 and m["prefix/outcome_src_wrong"] == 0.5
    assert TD.prefix_metrics(rs[2:]) == {}


def test_prefix_groups_are_not_draws(tmp_path):
    rs = [{"group": 0, "pool_index": 4, "seq": {"outcome": 1.0}, "finish": "stop"},
          {"group": 1, "pool_index": 9, "seq": {"outcome": 1.0}, "finish": "stop", "prefix": {"cut": 3000}}]
    assert TD.frontier_results(rs) == [(4, 1, 1, 0)]


def test_mixorder_history_skips_prefix(tmp_path):
    os.makedirs(tmp_path / "rollouts")
    rows = [{"task_id": "a", "group": 0, "seq": {"outcome": 1.0}, "finish": "stop"},
            {"task_id": "b", "group": 1, "seq": {"outcome": 0.0}, "finish": "stop", "prefix": {"cut": 3000}}]
    with gzip.open(tmp_path / "rollouts" / "step_000001.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    hist = M.history_from_records(str(tmp_path), 1, {"a": 0, "b": 1})
    assert hist == [(1, [(0, 1, 1, 0)])]


def test_zero_source_cut_limited_to_a_fraction():
    """--prefix-zero-max-frac: a source from a group with no right rollout is cut in the first half of its reasoning."""
    rows, comps = rows_and_comps()
    base = TD.choose_prefix_sources(rows, comps, 24, 8, 2500, 3000, THINK, {NL}, random.Random(3))
    assert TD.choose_prefix_sources(rows, comps, 24, 8, 2500, 3000, THINK, {NL}, random.Random(3), zero_max_frac=1.0) == base
    for seed in range(20):
        out = TD.choose_prefix_sources(rows, comps, 24, 8, 2500, 3000, THINK, {NL}, random.Random(seed), zero_max_frac=0.5)
        assert sum(k == 0 for _, _, k in out) > 0
        for i, cut, k in out:
            R = comps[i].index(THINK)
            assert 2500 <= cut <= R - 3000
            if k == 0:
                assert cut <= 0.5 * R
    trunc = [{"task_id": "z", "group": 0, "seq": {"outcome": 0.0}}] * 2
    tc = [comp(78000, closed=False), comp(78000, closed=False)]
    cuts = [TD.choose_prefix_sources(trunc, tc, 2, 8, 2500, 3000, THINK, {NL}, random.Random(s), zero_max_frac=0.5)[0][1]
            for s in range(30)]
    assert max(cuts) <= 39000 and min(cuts) >= 2500
    short = [{"task_id": "s", "group": 0, "seq": {"outcome": 0.0}}]
    assert TD.choose_prefix_sources(short, [comp(4800)], 1, 8, 2500, 3000, THINK, {NL}, random.Random(0), zero_max_frac=0.5) == []


def test_early_weighted_cuts():
    """--prefix-early-frac / --prefix-early-prob: cuts land in the first 60% of the trace with probability 0.8, later
    otherwise."""
    rows, comps = rows_and_comps()
    base = TD.choose_prefix_sources(rows, comps, 24, 8, 2500, 3000, THINK, {NL}, random.Random(3))
    assert TD.choose_prefix_sources(rows, comps, 24, 8, 2500, 3000, THINK, {NL}, random.Random(3), early_frac=0.0) == base
    early = late = 0
    for seed in range(60):
        for i, cut, k in TD.choose_prefix_sources(rows, comps, 24, 0, 2500, 3000, THINK, {NL}, random.Random(seed),
                                                  early_frac=0.6, early_prob=0.8):
            R = comps[i].index(THINK)
            assert 2500 <= cut <= R - 3000
            if 0.6 * R > 2500 and 0.6 * R < R - 3000:
                early += cut <= 0.6 * R
                late += cut > 0.6 * R
    assert late > 0 and 0.72 < early / (early + late) < 0.88       # ~80% early, some late


def test_prefix_every_and_extra_prompts():
    # prefix groups once every 3 steps; such steps add 6 fresh prompts (96 fresh otherwise)
    a = types.SimpleNamespace(prefix_every=3)
    assert [s for s in range(204, 214) if TD.is_prefix_step(a, s)] == [204, 207, 210, 213]
    assert all(TD.is_prefix_step(types.SimpleNamespace(prefix_every=1), s) for s in range(1, 9))
    assert all(TD.is_prefix_step(types.SimpleNamespace(), s) for s in range(1, 9))     # old args: every step
    rs = inspect.getsource(TD.rollout_step)
    assert 'order.take(args.prompts_per_step + (getattr(args, "prefix_extra_prompts", 0) if pre else 0) - len(pre))' in rs
    assert "pstep = is_prefix_step(args, step)" in rs and "if pstep and getattr(args, \"prefix_groups\", 0) > 0" in rs


def test_prefix_every_flags(tmp_path):
    base = ["--out", str(tmp_path / "o"), "--student", "/tmp/none", "--prompts-per-step", "96", "--save-rollouts", "1",
            "--prefix-groups", "24"]
    a = TD.parse_args(base + ["--prefix-every", "3", "--prefix-extra-prompts", "6"])
    assert a.prefix_every == 3 and a.prefix_extra_prompts == 6
    d = TD.parse_args(base)
    assert d.prefix_every == 1 and d.prefix_extra_prompts == 0
    for bad in (["--prefix-every", "0"], ["--prefix-extra-prompts", "-1"]):
        with pytest.raises(SystemExit):
            TD.parse_args(base + bad)
