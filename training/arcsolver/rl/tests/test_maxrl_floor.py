"""Floored MaxRL + down-weighted truncated rollouts (the 4B recipe): rewards.group_advantages(mode="maxrl_floor"),
train.py's truncation scaling, group dropping, weight metrics and flags."""
import random

from arcsolver.rl import rewards as rr
from arcsolver.rl import train as TD

FLOOR, LAM = 0.25, 0.3


def adv(values, groups=None, floor=FLOOR):
    groups = groups if groups is not None else [0] * len(values)
    return rr.group_advantages(values, groups, "maxrl_floor", 1e-4, floor=floor)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def make_step(groups):
    """groups: [[(reward, finish), ...] per group] -> rollouts as rollout_step builds them, then its advantage
    sequence: group statistics, outcome drop, truncation scaling, kept rollouts."""
    rollouts = []
    for g, rs in enumerate(groups):
        for R, fin in rs:
            rollouts.append({"group": g, "r_seq": R, "r_outcome": R, "seq": {"outcome": R}, "finish": fin,
                             "completion": [1] * (65536 if fin == "length" else 20000), "dense": []})
    gids = [r["group"] for r in rollouts]
    seq_adv = rr.group_advantages([r["r_seq"] for r in rollouts], gids, "maxrl_floor", 1e-4, floor=FLOOR)
    dropped = TD.dropped_groups("outcome", rollouts, gids)
    for r, a_ in zip(rollouts, seq_adv):
        r["dropped"] = r["group"] in dropped
        r["seq_adv"] = 0.0 if r["dropped"] else a_
    n_scaled = TD.scale_truncated_advantages(rollouts, LAM)
    return rollouts, TD.kept_rollouts(rollouts), dropped, n_scaled


def test_groups_sum_to_zero_without_truncation():
    rnd = random.Random(0)
    for G in (8, 16):
        for _ in range(500):
            if rnd.random() < 0.5:
                v = [float(rnd.random() < rnd.random()) for _ in range(G)]
            else:
                v = [rnd.choice([0.0, 0.0, 0.25, 0.5, 1.0]) for _ in range(G)]
            assert abs(sum(adv(v))) < 1e-9


def test_reference_table_g8():
    # correct of 8 -> (A correct, A failed)
    table = {1: (3.50, -0.50), 2: (3.00, -1.00), 3: (5 / 3, -1.00), 4: (1.00, -1.00), 7: (1 / 7, -1.00)}
    for k, (ac, af) in table.items():
        a = adv([1.0] * k + [0.0] * (8 - k))
        assert all(close(x, ac) for x in a[:k]), (k, a)
        assert all(close(x, af) for x in a[k:]), (k, a)


def test_g16_and_fractional():
    a = adv([1.0] + [0.0] * 15)                       # m = 1/16 < floor
    assert close(a[0], (1 - 1 / 16) / 0.25) and close(a[1], -(1 / 16) / 0.25)
    assert close(a[0], 3.75)
    a = adv([1.0] * 6 + [0.0] * 10)                   # m = 0.375 > floor: plain MaxRL
    assert close(a[0], (1 - 0.375) / 0.375) and close(a[-1], -1.0)
    v = [1.0, 0.5, 0, 0, 0, 0, 0, 0]                  # fractional, m = 0.1875 < floor
    m = sum(v) / 8
    assert all(close(x, (r - m) / 0.25) for x, r in zip(adv(v), v))
    v = [1.0, 1.0, 1.0, 0.5, 0, 0, 0, 0]              # fractional, m = 0.4375 > floor
    m = sum(v) / 8
    assert all(close(x, (r - m) / m) for x, r in zip(adv(v), v))
    # the floor is a fixed probability: the same 0.25 for every group size
    assert close(adv([1.0] + [0.0] * 3)[0], (1 - 0.25) / 0.25)      # G = 4, m = 0.25 = floor


def test_floor_required_and_other_modes_unchanged():
    try:
        rr.group_advantages([1.0, 0.0], [0, 0], "maxrl_floor", 1e-4, floor=0.0)
        raise AssertionError("floor 0 accepted")
    except ValueError:
        pass
    v, g = [1.0, 0.0, 0.0, 0.0], [0] * 4
    assert rr.group_advantages(v, g, "maxrl", 1e-4) == rr.group_advantages(v, g, "maxrl", 1e-4, floor=0.25)
    assert rr.group_advantages(v, g, "std", 1e-4) == rr.group_advantages(v, g, "std", 1e-4, floor=0.25)


def test_one_correct_seven_truncated_is_kept():
    rollouts, kept, dropped, n_scaled = make_step([[(1.0, "stop")] + [(0.0, "length")] * 7])
    assert not dropped and len(kept) == 8 and n_scaled[0] == 7
    raw = adv([1.0] + [0.0] * 7)
    assert close(raw[0], 3.5)                          # before scaling
    assert close(rollouts[0]["seq_adv"], 3.5)          # finished: unchanged
    assert all(close(r["seq_adv"], LAM * -0.5) for r in rollouts[1:])


def test_truncation_scaling_only_truncated():
    spec = [(1.0, "stop"), (1.0, "stop"), (0.0, "stop"), (0.0, "length"), (0.0, "length"), (0.0, "stop"),
            (0.0, "length"), (0.0, "stop")]
    rollouts, kept, dropped, n_scaled = make_step([spec])
    raw = adv([R for R, _ in spec])
    for r, a0, (_, fin) in zip(rollouts, raw, spec):
        assert close(r["seq_adv"], a0 * (LAM if fin == "length" else 1.0))
    assert n_scaled == (3, 3 * 65536)
    # the group's net push = (1 - lambda) x sum |A| of its truncated rollouts
    net = sum(r["seq_adv"] for r in rollouts)
    assert close(net, (1 - LAM) * sum(abs(a0) for a0, (_, f) in zip(raw, spec) if f == "length"))
    assert all(r in kept for r in rollouts)            # truncated rollouts stay in the loss


def test_zero_variance_groups_dropped():
    spec = [[(0.0, "length")] * 4 + [(0.0, "stop")] * 4,    # all failed, truncations included
            [(1.0, "stop")] * 8,                             # all correct
            [(1.0, "stop")] + [(0.0, "length")] * 7]         # informative
    rollouts, kept, dropped, n_scaled = make_step(spec)
    assert dropped == {0, 1}
    assert {r["group"] for r in kept} == {2}
    assert all(r["seq_adv"] == 0.0 and not r.get("trunc_scaled") for r in rollouts if r["group"] in (0, 1))
    assert n_scaled[0] == 7                            # only the kept group's truncations


def test_adv_weight_metrics():
    spec = [[(1.0, "stop")] + [(0.0, "length")] * 7,                         # k = 1
            [(1.0, "stop")] * 3 + [(0.0, "stop")] * 5,                       # k = 3
            [(1.0, "stop")] * 5 + [(0.0, "length")] * 3,                     # k = 5
            [(1.0, "stop")] * 7 + [(0.0, "stop")],                           # k = 7
            [(1.0, "stop")] * 8]                                             # dropped
    rollouts, kept, _, _ = make_step(spec)
    m = TD.adv_weight_metrics(rollouts, kept, 8)
    shares = [m[f"adv/weight_share_{b}"] for b in ("k1", "k2_3", "k4_5", "k6_7", "other")]
    assert close(sum(shares), 1.0)
    w = {}
    for r in kept:
        w[r["group"]] = w.get(r["group"], 0.0) + abs(r["seq_adv"])
    tot = sum(w.values())
    assert close(m["adv/weight_share_k1"], w[0] / tot) and close(m["adv/weight_share_k2_3"], w[1] / tot)
    assert close(m["adv/weight_share_k4_5"], w[2] / tot) and close(m["adv/weight_share_k6_7"], w[3] / tot)
    assert m["adv/groups_k1"] == m["adv/groups_k2_3"] == m["adv/groups_k4_5"] == m["adv/groups_k6_7"] == 1
    tr = sum(abs(r["seq_adv"]) for r in kept if r["finish"] == "length")
    assert close(m["adv/weight_share_truncated"], tr / tot)
    a = [r["seq_adv"] for r in kept]
    assert close(m["adv/ess_rollouts"], sum(map(abs, a)) ** 2 / sum(x * x for x in a))
    assert close(m["adv/net_sum"], sum(a)) and m["adv/net_sum"] > 0


def test_entropy_estimate():
    rs = [{"sampler_lp": [-1.0, -2.0]}, {"sampler_lp": [-3.0]}, {"sampler_lp": []}]
    assert close(TD.sampled_nll_mean(rs), 2.0)


BASE = ["--out", "/tmp/x", "--student", "/tmp/m"]


def test_flags():
    args = TD.parse_args(BASE + ["--adv-norm", "maxrl_floor", "--adv-floor", "0.25", "--mask-truncated", "0",
                                 "--trunc-neg-scale", "0.3", "--loss-agg", "seq", "--minibatch-max", "1"])
    assert (args.adv_norm, args.adv_floor, args.mask_truncated, args.trunc_neg_scale, args.loss_agg,
            args.minibatch_max) == ("maxrl_floor", 0.25, 0, 0.3, "seq", 1)
    d = TD.parse_args(BASE)
    assert d.trunc_neg_scale == 1.0 and d.adv_norm == "std" and d.mask_truncated == 0   # defaults unchanged
    for bad in (["--mask-truncated", "1", "--trunc-neg-scale", "0.3"], ["--trunc-neg-scale", "1.5"]):
        try:
            TD.parse_args(BASE + bad)
            raise AssertionError(f"accepted {bad}")
        except SystemExit:
            pass
