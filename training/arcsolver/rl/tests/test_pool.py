"""Prompt pools (never an evaluation task), pool files with measured rates, the weighted / stratified orders,
config.make_order / augment_group / slice_metrics and the per-budget pool scoring, on synthetic tasks."""
import json
import os
import random
import types

import pytest

from arcsolver.rl import augment as ra
from arcsolver.rl import config as RT
from arcsolver.rl import core as oc
from arcsolver.rl import pool as rp

from .helpers import StubTok, token_text


@pytest.fixture
def stub():
    return StubTok()


def _task(seed):
    rng = random.Random(seed)
    g = lambda: [[rng.randrange(10) for _ in range(3)] for _ in range(3)]  # noqa: E731
    return {"train": [{"input": g(), "output": g()} for _ in range(2)], "test": [{"input": g(), "output": g()}]}


def _write(d, tid, task):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, tid + ".json"), "w") as f:
        json.dump(task, f)


@pytest.fixture
def dirs(tmp_path):
    ev, bank, tr = (str(tmp_path / x) for x in ("evaluation", "bank", "training"))
    for i in range(4):
        _write(ev, f"ev{i}", _task(1000 + i))
    for i in range(5):
        _write(bank, f"bank{i}", _task(2000 + i))
    for i in range(4):
        _write(tr, f"tr{i}", _task(3000 + i))
    _write(tr, "ev0", _task(1000))                                  # an eval id in training
    _write(tr, "renamed_ev1", _task(1001))                          # eval content under another name
    _write(bank, "tr_dup", _task(3002))                             # a bank copy of a training task
    return ev, bank, tr


def test_rl_pools_by_source_never_hold_evaluation_tasks(dirs):
    ev, bank, tr = dirs
    arc, rep = rp.build_rl_pool("arc_train", bank, tr, ev, trained_ids={"tr1"})
    assert {it.task_id for it in arc} == {"tr0", "tr1", "tr2", "tr3"} and all(it.source == "arc_train" for it in arc)
    assert rep["dropped_eval_id"] == ["ev0"] and rep["dropped_eval_content"][0][0] == "renamed_ev1"
    assert {it.task_id: it.seen for it in arc}["tr1"] is True
    b, _ = rp.build_rl_pool("bank", bank, tr, ev, set(), holdout_ids={"bank4"})
    assert {it.task_id for it in b} == {"bank0", "bank1", "bank2", "bank3", "tr_dup"}
    mixed, rep_m = rp.build_rl_pool("mixed", bank, tr, ev, set())
    assert len(mixed) == 5 + 1 + 4 - 1 and ("tr2", "tr_dup") in rep_m["duplicates"]   # the dup is kept once
    for items in (arc, b, mixed):
        oc.assert_no_eval(items, ev)
        assert not {it.task_id for it in items} & {f"ev{i}" for i in range(4)}
    with pytest.raises(ValueError, match="evaluation directory"):
        rp.build_rl_pool("arc_train", bank, ev, ev, set())
    with pytest.raises(ValueError):
        rp.build_rl_pool("eval", bank, tr, ev, set())


def test_pool_file_formats_filter_and_oversample(tmp_path, dirs):
    ev, bank, tr = dirs
    items, _ = rp.build_rl_pool("arc_train", bank, tr, ev, set())
    meas = {"meta": {}, "summary": {}, "tasks": {
        "tr0": {"budgets": {"8192": {"success": 0.0}, "12288": {"success": 0.25}}},
        "tr1": {"budgets": {"8192": {"success": 0.5}, "12288": {"success": 1.0}}},
        "tr2": {"budgets": {"8192": {"success": 0.125}, "12288": {"success": 0.125}}}}}
    p = tmp_path / "pool.json"
    p.write_text(json.dumps(meas))
    with pytest.raises(ValueError, match="pool-budget"):
        rp.load_pool_rates(str(p))
    with pytest.raises(ValueError, match="no budget 4096"):
        rp.load_pool_rates(str(p), 4096)
    rates = rp.load_pool_rates(str(p), 12288)
    assert rates == {"tr0": 0.25, "tr1": 1.0, "tr2": 0.125}
    kept, rep = rp.filter_pool(items, rates, "learnable", "drop")
    assert [it.task_id for it in kept] == ["tr0", "tr2"] and rep == {"unmeasured": 1, "outside": 1, "kept": 2}
    kept_u, _ = rp.filter_pool(items, rates, "learnable", "keep")
    assert [it.task_id for it in kept_u] == ["tr0", "tr2", "tr3"]
    assert rp.load_pool_rates(str(p), 8192)["tr0"] == 0.0
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"tr0": 0.5, "tr1": {"success": 0.0}}))
    assert rp.load_pool_rates(str(flat)) == {"tr0": 0.5, "tr1": 0.0}
    assert rp.oversample_counts(items, rates, 3) == [3, 1, 3, 1]


def test_positive_pool_filter_keeps_always_solved_tasks(dirs):
    ev, bank, tr = dirs
    items, _ = rp.build_rl_pool("arc_train", bank, tr, ev, set())
    rates = {"tr0": 0.0, "tr1": 1.0, "tr2": 0.125}
    kept, rep = rp.filter_pool(items, rates, "positive", "drop")
    assert [it.task_id for it in kept] == ["tr1", "tr2"] and rep == {"unmeasured": 1, "outside": 1, "kept": 2}


def test_weighted_order_is_deterministic_resumable_and_distinct_within_a_step():
    counts = [3, 1, 1, 3, 1]
    a = rp.WeightedOrder(counts, seed=4)
    steps = [a.take(3) for _ in range(3)]
    state = a.state_dict()
    rest = [a.take(3) for _ in range(6)]
    b = rp.WeightedOrder(counts, seed=4)
    for s in steps:
        assert b.take(3) == s
    c = rp.WeightedOrder(counts, seed=4)
    c.load_state_dict(state)
    assert [c.take(3) for _ in range(6)] == rest
    for s in steps + rest:
        assert len(set(s)) == 3                                       # distinct items within a step
    flat = [i for s in steps + rest for i in s]
    assert flat.count(0) > flat.count(1)                              # oversampled items come up more often
    small = rp.WeightedOrder([2, 2], seed=0)
    assert len(small.take(3)) == 3                                    # fewer items than prompts: repeats allowed
    with pytest.raises(ValueError):
        rp.WeightedOrder([1, 1], seed=0).load_state_dict(state)


def test_stratified_order_meets_the_quota_every_step_and_resumes():
    bank, rest = list(range(0, 30)), list(range(30, 50))
    counts = [1] * 50
    o = rp.StratifiedOrder(counts, [bank, rest], [0.67, 0.33], seed=3)
    assert o.quotas(64) == [43, 21] and o.quotas(10) == [7, 3]
    steps = [o.take(10) for _ in range(4)]
    for s in steps:
        assert len(s) == 10 and len(set(s)) == 10
        assert sum(i in bank for i in s) == 7                         # the bank quota holds on every step
    state = o.state_dict()
    rest_steps = [o.take(10) for _ in range(5)]
    p = rp.StratifiedOrder(counts, [bank, rest], [0.67, 0.33], seed=3)
    assert [p.take(10) for _ in range(4)] == steps                    # deterministic
    q = rp.StratifiedOrder(counts, [bank, rest], [0.67, 0.33], seed=3)
    q.load_state_dict(state)
    assert [q.take(10) for _ in range(5)] == rest_steps               # resumable
    flat = [i for s in steps + rest_steps for i in s]
    assert all(flat.count(i) >= 1 for i in bank)                      # every bank item drawn within ~one epoch
    with pytest.raises(ValueError):
        rp.StratifiedOrder(counts, [bank[:-1], rest], [0.67, 0.33], seed=3).load_state_dict(state)
    with pytest.raises(ValueError):
        rp.WeightedOrder(counts, seed=3).load_state_dict(state)       # a stratified state never loads silently
    with pytest.raises(ValueError):
        rp.StratifiedOrder(counts, [bank, rest], [0.6, 0.6], seed=3)
    with pytest.raises(ValueError):
        rp.StratifiedOrder(counts, [bank, []], [0.67, 0.33], seed=3)


class _Item:
    def __init__(self, source, ids=(1, 2, 3), rate=None):
        self.source, self.student_ids, self.rate = source, list(ids), rate


def _rl_args(**kw):
    base = dict(bank_share=0.0, augment=1, aug_original_prob=0.0, aug_colors="nonzero", aug_identity_prob=-1.0,
                aug_max_tries=8, max_prompt_tokens=10_000, seed=0, arm="render_think")
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_make_order_stratifies_by_source_only_when_asked():
    pool = [_Item("bank")] * 6 + [_Item("arc_train")] * 4
    assert isinstance(RT.make_order(_rl_args(), pool, [1] * 10), rp.WeightedOrder)
    o = RT.make_order(_rl_args(bank_share=0.67), pool, [1] * 10)
    assert isinstance(o, rp.StratifiedOrder) and o.strata == [list(range(6)), list(range(6, 10))]
    assert sum(i < 6 for i in o.take(6)) == 4
    with pytest.raises(SystemExit):
        RT.make_order(_rl_args(bank_share=0.67), [_Item("arc_train")] * 3, [1] * 3)


def test_augment_group_original_prob_and_unchanged_default(monkeypatch):
    monkeypatch.setattr(oc, "student_messages", lambda task, arm: task)
    monkeypatch.setattr(oc, "render_ids", lambda tok, task: [v for p in task["train"] + task["test"]
                                                             for g in p.values() for row in g for v in row])
    task = {"train": [{"input": [[1, 2, 0]], "output": [[3, 0, 0]]}], "test": [{"input": [[4, 5, 6]], "output": [[6]]}]}
    it = _Item("bank", ids=[9, 9])
    # default (0): the group's RNG goes straight to sample_aug
    for gi in range(20):
        atask, ids, aug, n = RT.augment_group(_rl_args(), task, it, None, 5, gi)
        want = ra.sample_aug(ra.group_rng(0, 5, gi), "nonzero", -1.0)
        assert aug == want and atask == want.task(task) and n == 0
    # 1: every group sees the task as given (the pool's rendered prompt, no transform)
    for gi in range(5):
        atask, ids, aug, n = RT.augment_group(_rl_args(aug_original_prob=1.0), task, it, None, 5, gi)
        assert aug.is_identity() and atask is task and ids == [9, 9] and n == 0
    # 0.5: about half, deterministic per (seed, step, group)
    got = [RT.augment_group(_rl_args(aug_original_prob=0.5), task, it, None, s, g)[2].is_identity()
           for s in range(40) for g in range(10)]
    assert 0.4 < sum(got) / len(got) < 0.6
    assert got == [RT.augment_group(_rl_args(aug_original_prob=0.5), task, it, None, s, g)[2].is_identity()
                   for s in range(40) for g in range(10)]
    # over-long augmented prompts still fall back to the identity dihedral, keeping the colours
    atask, ids, aug, n = RT.augment_group(_rl_args(max_prompt_tokens=1), task, it, None, 5, 0)
    assert aug.d8 == "id" and n == 8 and not aug.is_identity()
    # no augmentation at all
    assert RT.augment_group(_rl_args(augment=0), task, it, None, 5, 0)[2] == ra.IDENTITY


def test_slice_metrics_split_bank_and_original_views():
    def r(source, ident, em, rate):
        return {"item": _Item(source, rate=rate), "aug": ra.IDENTITY if ident else ra.Aug("rot90"),
                "stats": {"em_frac": em}}
    rs = [r("bank", True, 1.0, 0.25), r("bank", False, 0.0, 0.25), r("bank", False, 0.0, 0.01),
          r("arc_train", False, 1.0, 0.5), r("arc_train", True, 0.0, 0.5)]
    m = RT.slice_metrics(rs)
    assert m["slice/bank/rollouts"] == 3 and abs(m["slice/bank/em_frac"] - 1 / 3) < 1e-9
    assert abs(m["slice/bank/vs_screen"] - 1.0 / 0.5) < 1e-9          # the never-solved marker rate (0.01) is left out
    assert m["slice/orig/rollouts"] == 2 and abs(m["slice/orig/vs_screen"] - 1.0 / 0.75) < 1e-9
    assert m["slice/arc/em_frac"] == 0.5 and m["slice/aug/rollouts"] == 3
    assert "bank 33.3% (x2.00 screen) n=3" in RT.slice_line(m)


def test_sample_budget_stats_and_summary(stub):
    task = {"train": [], "test": [{"input": [[0]], "output": [[1, 2], [3, 4]]}]}
    right = token_text(stub, "Short.</think>\n\n12\n34<|im_end|>")
    n = len(right)
    budgets = [n - 1, n, n + 100]
    s = rp.sample_budget_stats(task, right, "stop", budgets)
    assert s[str(n - 1)]["finished"] is False and s[str(n - 1)]["em_frac"] == 0.0 and s[str(n - 1)]["closed"] is True
    assert s[str(n)]["finished"] is True and s[str(n)]["em_frac"] == 1.0 and s[str(n)]["em_all"] is True
    loop_text = "Intro.\n\n" + "\n\n".join(f"Check cell {i}." for i in range(12))
    lt = token_text(stub, loop_text)
    k = lt.token_range(loop_text.index("Check cell 2."), len(loop_text))[0]    # prefix with only 2 templated paras
    ls = rp.sample_budget_stats(task, lt, "length", [k, len(lt)])
    assert ls[str(k)]["loop"] is False and ls[str(len(lt))]["loop"] is True and ls[str(len(lt))]["loop_verbatim"] is False
    samples = [s, s, rp.sample_budget_stats(task, token_text(stub, "x</think>\n\n99<|im_end|>"), "stop", budgets),
               rp.sample_budget_stats(task, lt, "length", budgets)]
    tb = rp.task_budget_stats(samples, budgets)
    assert tb[str(n)]["success"] == 0.5 and tb[str(n)]["n_correct"] == 2 and tb[str(n - 1)]["success"] == 0.0
    tasks = {"a": {"budgets": tb, "n_prompt_tokens": 100},
             "b": {"budgets": rp.task_budget_stats([s] * 4, budgets), "n_prompt_tokens": 20000},
             "c": {"budgets": rp.task_budget_stats(samples[2:], budgets), "n_prompt_tokens": 50}}
    summ = rp.summarise_pool(tasks, budgets, max_prompt_tokens=12288)
    B = summ[str(n)]
    assert (B["n_tasks"], B["n_zero"], B["n_learnable"], B["n_one"], B["n_any_correct"]) == (3, 1, 1, 1, 2)
    assert B["n_learnable_fit"] == 1 and B["mean_success"] == pytest.approx(0.5)


def test_build_pool_and_eval_set_from_cli_paths(tmp_path, dirs):
    """config.build_pool through parse_args with explicit paths; the eval set skips a source with no directory."""
    from arcsolver.rl import data as ot
    ev, bank, tr = dirs
    out = tmp_path / "run"
    out.mkdir()

    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            return [1] * (10 + len(msgs[-1]["content"]) // 100)

        def decode(self, ids):
            return "<|im_start|>assistant\n<think>\n"

    a = RT.parse_args(["--out", str(out), "--student", "/tmp/none", "--bank-dir", bank, "--arc-train-dir", tr, "--arc-eval-dir", ev,
                       "--pool", "mixed", "--pool-filter", "none", "--eval-bank", "4", "--eval-arc", "2"])
    tasks = ot.TaskCache()
    items = ot.build_eval_set(a, Tok(), tasks, lambda *m: None)          # no --bank200-dir: ARC eval tasks only
    assert [it.source for it in items] == ["arc_eval", "arc_eval"]
    pool, counts = RT.build_pool(a, Tok(), tasks, set(), lambda *m: None)
    assert len(pool) == 9 and counts == [1] * 9 and not {it.task_id for it in pool} & {f"ev{i}" for i in range(4)}
    assert json.load(open(out / "pool.json"))["items"][0]["copies"] == 1
