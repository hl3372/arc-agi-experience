"""--len-pair-lambda: thresholded pairwise length shaping on floored-MaxRL advantages, only in groups with >= min_correct
fully right rollouts. Properties: zero sum per group, |B| <= 1, non-right rollouts unchanged, right rollouts stay
>= (1 - gamma) * a_plus in groups with failures, no signal below min_correct or within delta, the ramp, all-right
groups returned for keeping, permutation invariance."""
import inspect
import random

import pytest

from arcsolver.rl import rewards as rr
from arcsolver.rl import train as TD


def _train_src():
    return inspect.getsource(TD)


FLOOR = 0.2


def maxrl(rewards):
    m = sum(rewards) / len(rewards)
    return [(r - m) / max(m, FLOOR) for r in rewards]


def shape(outcomes, lengths, lam0=0.1, **kw):
    adv = maxrl(outcomes)
    out, allr, stats = rr.pairwise_length_shaping(adv, outcomes, lengths, [0] * len(outcomes), lam0, **kw)
    return adv, out, allr, stats


def test_note_example_mixed_group():
    # 2 short (10k) + 5 long (40k) right, 1 wrong: B = 5/6 and -1/3; a_plus = 1/7 -> lambda = min(0.1, 0.5/7)
    outc = [1] * 7 + [0]
    lens = [10000, 10000] + [40000] * 5 + [50000]
    adv, out, allr, st = shape(outc, lens)
    lam = min(0.1, 0.5 / 7)
    assert out[0] == pytest.approx(1 / 7 + lam * 5 / 6) and out[2] == pytest.approx(1 / 7 - lam / 3)
    assert out[7] == adv[7] and sum(out) == pytest.approx(0.0, abs=1e-12) and allr == set()
    assert min(out[:7]) >= (1 - 0.5) * (1 / 7) - 1e-12 and st["lenpair/groups"] == 1


def test_note_example_all_right_group():
    outc = [1] * 8
    lens = [10000, 10000] + [40000] * 6
    adv, out, allr, st = shape(outc, lens)
    assert out[0] == pytest.approx(0.1 * 6 / 7) and out[2] == pytest.approx(-0.1 * 2 / 7)
    assert sum(out) == pytest.approx(0.0, abs=1e-12) and allr == {0} and st["lenpair/allright_groups"] == 1


def test_gates_and_ramp():
    # 3 right (< min_correct 4): unchanged
    outc = [1, 1, 1, 0, 0, 0, 0, 0]
    adv, out, _, st = shape(outc, [10000, 50000, 30000, 1, 2, 3, 4, 5])
    assert out == adv and st["lenpair/groups"] == 0
    # 4 right but all within delta of each other: unchanged
    outc = [1, 1, 1, 1, 0, 0, 0, 0]
    adv, out, _, _ = shape(outc, [30000, 35000, 39000, 40000, 1, 2, 3, 4])
    assert out == adv
    # gap exactly delta -> 0, delta + span/2 -> 0.5, delta + span -> 1, beyond -> 1 (two right rollouts compared)
    for gap, q in ((10000, 0.0), (20000, 0.5), (30000, 1.0), (60000, 1.0)):
        outc = [1, 1, 1, 1]
        lens = [10000, 10000, 10000 + gap, 10000 + gap]   # pairs: short-short 0, short-long q
        adv, out, _, _ = shape(outc, lens)
        assert out[0] == pytest.approx(0.1 * (2 * q) / 3) and out[2] == pytest.approx(-0.1 * (2 * q) / 3)


def test_properties_random():
    rng = random.Random(0)
    for _ in range(300):
        G = 8
        outc = [rng.choice([0, 0.5, 1, 1, 1]) for _ in range(G)]
        lens = [rng.randint(2000, 78000) for _ in range(G)]
        adv = maxrl(outc)
        out, allr, _ = rr.pairwise_length_shaping(adv, outc, lens, [0] * G, 0.1)
        assert sum(out) == pytest.approx(sum(adv), abs=1e-9)
        right = [i for i in range(G) if outc[i] >= 1]
        for i in range(G):
            if i not in right:
                assert out[i] == adv[i]
        if len(right) < G and len(right) >= 4:
            ap = min(adv[i] for i in right)
            assert all(out[i] >= (1 - 0.5) * ap - 1e-12 for i in right)
        for i in right:
            assert abs(out[i] - adv[i]) <= 0.1 + 1e-12
        perm = list(range(G)); rng.shuffle(perm)
        out2, _, _ = rr.pairwise_length_shaping([adv[p] for p in perm], [outc[p] for p in perm], [lens[p] for p in perm],
                                                [0] * G, 0.1)
        assert out2 == pytest.approx([out[p] for p in perm])


def test_groups_independent_and_off():
    outc = [1] * 8 + [1] * 4 + [0] * 4
    lens = [10000] * 4 + [50000] * 4 + [10000, 10000, 50000, 50000, 5, 5, 5, 5]
    gids = [0] * 8 + [1] * 8
    adv = maxrl(outc[:8]) + maxrl(outc[8:])
    out, allr, st = rr.pairwise_length_shaping(adv, outc, lens, gids, 0.1)
    assert allr == {0} and st["lenpair/groups"] == 2 and st["lenpair/mixed_groups"] == 1
    assert sum(out[:8]) == pytest.approx(0.0) and sum(out[8:]) == pytest.approx(sum(adv[8:]))
    assert rr.pairwise_length_shaping(adv, outc, lens, gids, 0.0) == (adv, set(), {})
    with pytest.raises(ValueError):
        rr.pairwise_length_shaping(adv, outc, lens, gids, 0.1, gamma=1.0)


def test_trainer_wiring():
    src = _train_src()
    assert '"--len-pair-lambda"' in src and "rr.pairwise_length_shaping(" in src
    assert "allc = set(allc) | pair_allright" in src
    # shaping happens before the forced-answer scaling and before the dropped-group computation
    assert src.index("rr.pairwise_length_shaping(") < src.index("seq_adv = rr.scale_forced_advantages(")
    assert src.index("rr.pairwise_length_shaping(") < src.index("dropped = dropped_groups(args.drop_zero_var")


def test_allright_off():
    # 15k tolerance, no shaping for all-right groups; mixed groups unchanged by the switch
    outc = [1] * 8 + [1] * 4 + [0] * 4
    lens = [10000] * 4 + [50000] * 4 + [10000, 10000, 50000, 50000, 5, 5, 5, 5]
    gids = [0] * 8 + [1] * 8
    adv = maxrl(outc[:8]) + maxrl(outc[8:])
    on, allr_on, _ = rr.pairwise_length_shaping(adv, outc, lens, gids, 0.1, delta=15000)
    off, allr, st = rr.pairwise_length_shaping(adv, outc, lens, gids, 0.1, delta=15000, shape_allright=False)
    assert allr_on == {0} and allr == set() and off[:8] == adv[:8]
    assert st["lenpair/groups"] == 1 and st["lenpair/allright_groups"] == 0 and st["lenpair/mixed_groups"] == 1
    assert off[8:] == on[8:] and off[8:] != adv[8:]
    # 15k tolerance: a 14k gap gives nothing, 25k gives half the ramp
    o2 = [1, 1, 1, 1, 0]
    a2 = maxrl(o2)
    same, _, _ = rr.pairwise_length_shaping(a2, o2, [10000, 10000, 24000, 24000, 1], [0] * 5, 0.1, delta=15000)
    assert same == a2


def test_allright_flag_wiring():
    src = _train_src()
    assert '"--len-pair-allright"' in src and 'shape_allright=bool(getattr(args, "len_pair_allright", 1))' in src
