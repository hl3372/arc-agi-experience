"""Opt-in --mask-truncated (DAPO overlong filtering): truncated rollouts keep their place in the group statistics but
leave the loss and its token-mean normaliser. Off by default: kept rollouts, normalisers and the rollout record are
unchanged."""
import pytest

from arcsolver.rl import rewards as rr
from arcsolver.rl import train as D

BASE = ["--out", "/tmp/x", "--student", "/tmp/m", "--pg-loss", "tis_sc", "--adv-norm", "maxrl"]


def _rollouts():
    """Three groups of 4: g0 mixed (1 correct, 1 closed wrong, 2 truncated), g1 all wrong (dropped, 3 truncated),
    g2 mixed (2 correct, 2 truncated)."""
    spec = [(0, 1, "stop", 40), (0, 0, "stop", 60), (0, 0, "length", 100), (0, 0, "length", 100),
            (1, 0, "length", 100), (1, 0, "length", 100), (1, 0, "length", 100), (1, 0, "stop", 70),
            (2, 1, "stop", 30), (2, 1, "stop", 50), (2, 0, "length", 100), (2, 0, "length", 100)]
    rs = [{"group": g, "r_seq": float(c), "r_outcome": float(c), "finish": f, "completion": [7] * n}
          for g, c, f, n in spec]
    gids = [r["group"] for r in rs]
    adv = rr.group_advantages([r["r_seq"] for r in rs], gids, "maxrl")
    dropped = rr.zero_variance_groups([r["r_outcome"] for r in rs], gids)
    for r, a in zip(rs, adv):                 # as the trainer's dropped-groups loop
        r["dropped"] = r["group"] in dropped
        r["seq_adv"] = 0.0 if r["dropped"] else a
    return rs


def test_flag_default_off_and_parses():
    assert D.parse_args(BASE).mask_truncated == 0
    assert D.parse_args(BASE + ["--mask-truncated", "1"]).mask_truncated == 1


def test_off_is_unchanged():
    rs = _rollouts()
    kept = D.kept_rollouts(rs)
    assert [r["group"] for r in kept] == [0] * 4 + [2] * 4          # the dropped all-wrong group only
    assert not any("trunc_masked" in r for r in rs)
    assert sum(len(r["completion"]) for r in kept) == 580               # the token-mean normaliser


def test_mask_truncated():
    rs = _rollouts()
    before = [r["seq_adv"] for r in rs]
    n, toks = D.mask_truncated_rollouts(rs)
    assert (n, toks) == (4, 400)                                    # g0's and g2's truncated rows; g1 is dropped
    assert [r["seq_adv"] for r in rs] == before                     # group statistics untouched
    assert not any(r.get("trunc_masked") for r in rs if r["group"] == 1)
    kept = D.kept_rollouts(rs)
    assert [(r["group"], r["finish"]) for r in kept] == [(0, "stop"), (0, "stop"), (2, "stop"), (2, "stop")]
    assert sum(len(r["completion"]) for r in kept) == 180              # normaliser: masked tokens left out
    # MaxRL kept advantages: g0 p=1/4 -> +3 (correct), -1 (closed wrong); g2 p=1/2 -> +1, +1
    assert [round(r["seq_adv"], 2) for r in kept] == [3.0, -1.0, 1.0, 1.0]


@pytest.mark.parametrize("extra", [["--train-budget-force", "1024"]])
def test_rejected_combinations(extra):
    try:
        D.parse_args(BASE + extra)                                  # must parse on its own for the test to mean anything
    except SystemExit:
        pytest.skip(f"{extra} needs more flags on its own")
    with pytest.raises(SystemExit, match="--mask-truncated"):
        D.parse_args(BASE + ["--mask-truncated", "1"] + extra)
