"""Forced answers for truncated rollouts (--train-budget-force / --forced-credit) on floored MaxRL with
--trunc-neg-scale: natural success > forced success > failure in every group, a forced-only group gets half of a
lone natural success, truncated rollouts whose forced answer is wrong keep the scaled push-down, credited ones are
not scaled, and the forced tokens are scored, never trained."""
import types

import pytest

from arcsolver.rl import rewards as rr
from arcsolver.rl import sampler_dispatch
from arcsolver.rl import train as D

FLOOR = 0.25
BASE = ["--out", "/tmp/forced_answers_test", "--student", "/tmp/none"]


def _adv(rewards, gids):
    return rr.group_advantages(rewards, gids, "maxrl_floor", 1e-4, floor=FLOOR)


def _credit(natural, forced, gids, credit=0.5):
    rewards, mask = rr.forced_credit_rewards(natural, forced, gids, credit)
    return rr.scale_forced_advantages(_adv(rewards, gids), mask, credit, gids, natural), rewards, mask


def test_forced_rewards_only_fill_in_for_failed_rollouts():
    out, mask = rr.forced_credit_rewards([1.0, 0.0, 0.0, 0.0, 0.5], [1.0, 1.0, 0.0, None, 1.0])
    assert out == [1.0, 1.0, 0.0, 0.0, 0.5]            # natural rewards (incl. partial credit) are never replaced
    assert mask == [False, True, False, False, False]


def test_half_credit_ranks_natural_over_forced_over_failure():
    adv, rewards, _ = _credit([1.0] + [0.0] * 7, [None, 1.0] + [0.0] * 6, [0] * 8)
    assert rewards[:2] == [1.0, 0.5]
    m = 1.5 / 8                                        # below the floor: divide by 0.25
    assert adv[0] == pytest.approx((1 - m) / FLOOR) and adv[1] == pytest.approx((0.5 - m) / FLOOR)
    assert adv[0] > adv[1] > 0 > adv[2]
    assert adv[2:] == pytest.approx([-m / FLOOR] * 6)


def test_a_forced_only_group_gets_half_a_lone_success():
    adv, _, mask = _credit([0.0] * 8, [1.0] + [0.0] * 7, [0] * 8)
    lone = _adv([1.0] + [0.0] * 7, [0] * 8)            # 3.5 under the floor
    assert mask[0] and adv[0] == pytest.approx(lone[0] / 2)
    assert adv[1:] == pytest.approx(lone[1:])           # the failures are pushed as they would be by a natural success


def test_a_forced_answer_cannot_make_a_group_look_all_correct():
    adv, rewards, _ = _credit([1.0] * 7 + [0.0], [None] * 7 + [1.0], [0] * 8)
    assert 0 not in rr.zero_variance_groups(rewards, [0] * 8)
    assert adv[7] < 0 < adv[0]


def test_credit_is_decided_per_group():
    gids = [0, 0, 0, 0, 1, 1, 1, 1]
    adv, rewards, mask = _credit([1.0, 0, 0, 0, 0, 0, 0, 0], [None, 1.0, 0.0, None, 1.0, 0.0, None, None], gids)
    assert rewards == [1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert mask == [False, True, False, False, True, False, False, False]
    assert adv[4] == pytest.approx(_adv([1.0, 0.0, 0.0, 0.0], [1] * 4)[0] / 2)


def test_credit_bounds_and_identity():
    assert rr.scale_forced_advantages([2.0, -1.0], [True, True], 1.0) == [2.0, -1.0]
    assert rr.scale_forced_advantages([2.0, -1.0], [True, True], 0.0) == [0.0, -1.0]
    with pytest.raises(ValueError):
        rr.scale_forced_advantages([1.0], [True], 1.5)


def test_without_forced_outcomes_nothing_changes():
    rewards = [1.0, 0.0, 0.0, 1.0]
    out, mask = rr.forced_credit_rewards(rewards, [None] * 4)
    assert out == rewards and not any(mask)
    assert rr.scale_forced_advantages(_adv(rewards, [0] * 4), mask, 0.5) == _adv(rewards, [0] * 4)


def test_truncation_scale_skips_credited_rollouts_and_keeps_the_push_down_on_the_rest():
    # one natural success (stop), one truncated with a right forced answer, one truncated with a wrong one, five
    # natural failures (stop): the end-to-end advantage path of rollout_step for this group
    finish = ["stop", "length", "length"] + ["stop"] * 5
    natural = [1.0] + [0.0] * 7
    forced = [None, 1.0, 0.0] + [None] * 5
    adv, rewards, mask = _credit(natural, forced, [0] * 8)
    ro = [dict(finish=f, dropped=False, seq_adv=a, completion=[0] * 10, forced_credit=m)
          for f, a, m in zip(finish, adv, mask)]
    n, _ = D.scale_truncated_advantages(ro, 0.3)
    assert n == 1                                      # only the truncated rollout whose forced answer was wrong
    assert ro[1]["seq_adv"] == pytest.approx(adv[1]) and ro[1]["seq_adv"] > 0
    assert ro[2]["seq_adv"] == pytest.approx(0.3 * adv[2]) and ro[2]["seq_adv"] < 0
    assert ro[0]["seq_adv"] > ro[1]["seq_adv"] > 0 > ro[2]["seq_adv"] > ro[3]["seq_adv"]


@pytest.mark.parametrize("k", range(1, 7))
def test_after_truncation_scaling_natural_outranks_right_forced_outranks_wrong_forced(k):
    """With k natural successes of 8, the right forced answer (reward 0.5) is below the group mean for k >= 5 and gets a
    negative advantage; it must still end above the truncated wrong one after the 0.3 scale."""
    natural = [1.0] * k + [0.0] * (8 - k)
    forced = [None] * k + [1.0, 0.0] + [None] * (6 - k)
    finish = ["stop"] * k + ["length", "length"] + ["stop"] * (6 - k)
    adv, _, mask = _credit(natural, forced, [0] * 8)
    ro = [dict(finish=f, dropped=False, seq_adv=a, completion=[0], forced_credit=m) for f, a, m in zip(finish, adv, mask)]
    D.scale_truncated_advantages(ro, 0.3)
    nat, right, wrong = ro[0]["seq_adv"], ro[k]["seq_adv"], ro[k + 1]["seq_adv"]
    assert nat > right > wrong
    assert wrong == pytest.approx(0.3 * adv[k + 1])
    assert right == pytest.approx(adv[k] if adv[k] > 0 else 0.3 * adv[k])


def test_seven_of_eight_right_forced_is_pushed_less_than_a_wrong_one_would_be():
    gids = [0] * 8
    finish = ["stop"] * 7 + ["length"]
    out = {}
    for f in (1.0, 0.0):                               # the eighth rollout's forced answer right, or wrong
        adv, _, mask = _credit([1.0] * 7 + [0.0], [None] * 7 + [f], gids)
        ro = [dict(finish=x, dropped=False, seq_adv=a, completion=[0], forced_credit=m) for x, a, m in zip(finish, adv, mask)]
        D.scale_truncated_advantages(ro, 0.3)
        out[f] = ro[7]["seq_adv"]
    assert out[1.0] == pytest.approx(0.3 * (0.5 - 7.5 / 8) / (7.5 / 8)) and out[0.0] == pytest.approx(-0.3)
    assert 0 > out[1.0] > out[0.0]


def test_resume_eval_pending_with_eval_steps(tmp_path):
    a = types.SimpleNamespace(eval_every=20, eval_steps="70", out=str(tmp_path))
    assert D.resume_eval_pending(a, 70) and D.resume_eval_pending(a, 80) and not D.resume_eval_pending(a, 71)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "step_000070.COMPLETE").write_text("ok")
    assert not D.resume_eval_pending(a, 70)
    with pytest.raises(SystemExit):
        D.parse_args(["--out", str(tmp_path), "--student", "/tmp/none", "--eval-steps", "7O"])


class _Sampler:
    def __init__(self):
        self.calls = []

    def call(self, cmd, prompts, params, max_tokens, seeds, stream_target=None):
        assert cmd == "generate"
        self.calls.append(dict(prompts=prompts, params=params, max_tokens=max_tokens, seeds=seeds,
                               stream_target=stream_target))
        return {"outputs": [[{"token_ids": [7, 8, 9], "finish": "stop", "text": ""}] for _ in prompts]}


def _stok(text, add_special_tokens=False):
    return {"input_ids": [101, 102]}


def test_force_answers_targets_only_truncated_rollouts():
    args = types.SimpleNamespace(train_budget_force=2048, sampler_max_model_len=200, eval_temperature=1.0,
                                 eval_top_p=1.0, eval_top_k=0, seed=0)
    ro = [dict(prompt_ids=[1] * 10, completion=[2] * 50, finish="length"),     # forced
          dict(prompt_ids=[1] * 10, completion=[2] * 50, finish="stop"),       # finished: no
          dict(prompt_ids=[1] * 10, completion=[], finish="length"),           # empty: no
          dict(prompt_ids=[1] * 10, completion=[2] * 150, finish="length"),    # room 200 - 162 = 38 < 64: no
          dict(prompt_ids=[1] * 10, completion=[2] * 100, finish="length")]    # forced, room 200 - 112 = 88
    s = _Sampler()
    out = D.force_answers(args, 3, s, _stok, ro)
    assert sorted(out) == [0, 4]
    assert ro[3].get("forced_skipped") and not any(ro[i].get("forced_skipped") for i in (0, 1, 2, 4))
    c = s.calls[0]
    assert c["stream_target"] == D.FORCE_STREAM_SEQS
    assert c["max_tokens"] == [200 - 62, 88] and c["prompts"][0][-2:] == [101, 102]
    assert c["params"] == {"n": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "logprobs": False}
    assert out[0][0] == [7, 8, 9] and out[0][1] == "stop"
    assert ro[0]["completion"] == [2] * 50             # the trained sequence is untouched


def test_forced_stats_counts_rescued_groups():
    ro = [dict(group=0, seq={"outcome": 0.0}, forced_outcome=1.0, forced_finish="stop"),
          dict(group=0, seq={"outcome": 0.0}, forced_outcome=0.0, forced_finish="length"),
          dict(group=1, seq={"outcome": 1.0}), dict(group=1, seq={"outcome": 0.0}, forced_outcome=1.0, forced_finish="stop"),
          dict(group=1, seq={"outcome": 0.0}, forced_skipped=True)]
    m = D.forced_stats(ro, [True, False, False, True, False])
    assert m["forced/tried"] == 3 and m["forced/correct"] == 2 and m["forced/credited"] == 2
    assert m["forced/rescued_groups"] == 1
    assert m["forced/cut_at_budget"] == 1 and m["forced/skipped_no_room"] == 1


def test_multiengine_caps_stream_concurrency_for_one_call(monkeypatch):
    seen = []
    monkeypatch.setattr(sampler_dispatch, "generate_stream",
                        lambda engines, pool, kw, seqs, kv_high=0.9, on_sample=None:
                        seen.append((dict(kw), seqs)) or {"outputs": []})
    me = D.MultiEngine([object(), object()], stream_seqs=48)
    me.call("generate", prompts=[], params={}, stream_target=24)
    me.call("generate", prompts=[], params={})
    me.call("generate", prompts=[], params={}, stream_target=96)       # a cap never raises concurrency
    assert [s for _, s in seen] == [24, 48, 48]
    assert all("stream_target" not in kw for kw, _ in seen)


def test_flags():
    a = D.parse_args(BASE)
    assert a.train_budget_force == 0 and a.forced_credit == 0.5 and a.train_budget_force_from == 0
    assert a.eval_steps == "" and not D.eval_due(types.SimpleNamespace(eval_every=20, eval_steps=""), 70)
    assert D.eval_due(types.SimpleNamespace(eval_every=20, eval_steps="70"), 70)
    assert D.eval_due(types.SimpleNamespace(eval_every=20, eval_steps="70"), 80)
    assert not D.eval_due(types.SimpleNamespace(eval_every=20, eval_steps="70"), 71)
    a = D.parse_args(BASE + ["--train-budget-force", "2048", "--trunc-neg-scale", "0.3"])   # the two combine
    assert a.train_budget_force == 2048 and a.trunc_neg_scale == 0.3
    for bad in (["--train-budget-force", "-1"], ["--forced-credit", "1.5"], ["--forced-credit", "0"],
                ["--train-budget-force", "2048", "--mask-truncated", "1"]):
        with pytest.raises(SystemExit):
            D.parse_args(BASE + bad)
