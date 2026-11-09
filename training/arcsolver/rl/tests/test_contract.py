"""--objective-contract grpo_token_v1 and the resume objective check."""
from types import SimpleNamespace as NS

import pytest

from arcsolver.rl import contract as C
from arcsolver.rl import train as D

from .helpers import CONTRACT_BASE as BASE
from .helpers import CONTRACT_FLAGS


def parse(extra):
    return D.parse_args(BASE + CONTRACT_FLAGS + extra)


def test_contract_flags_accepted():
    a = parse([])
    assert C.check_contract(a, "grpo_token_v1") == []
    assert (a.prompts_per_step, a.group_size, a.beta_loop, a.drop_zero_var) == (250, 6, 0.0, "outcome")


@pytest.mark.parametrize("drop,key", [("--beta-loop", "beta_loop"), ("--drop-zero-var", "drop_zero_var"),
                                      ("--train-split-tests", "train_split_tests"),
                                      ("--per-request-seeds", "per_request_seeds")])
def test_inherited_defaults_refused(drop, key):
    flags = list(CONTRACT_FLAGS)
    i = flags.index(drop)
    del flags[i:i + (1 if drop in ("--train-split-tests", "--per-request-seeds") else 2)]
    with pytest.raises(SystemExit) as e:
        D.parse_args(BASE + flags)
    assert key in str(e.value)


@pytest.mark.parametrize("extra,key", [(["--pg-loss", "tis_sc"], "pg_loss"), (["--clip-eps-high", "0.28"], "clip_eps_high"),
                                       (["--adv-norm", "maxrl"], "adv_norm"), (["--minibatch-max", "2"], "minibatch_max"),
                                       (["--len-share-lambda", "0.25"], "len_share_lambda")])
def test_deviations_refused(extra, key):
    with pytest.raises(SystemExit) as e:
        parse(extra)
    assert key in str(e.value)


def test_every_contract_key_is_a_parsed_argument():
    a = parse([])
    for name, c in C.CONTRACTS.items():
        missing = [k for k in c if not hasattr(a, k)]
        assert not missing, (name, missing)


def test_resume_snapshot_compare():
    a = parse([])
    cfg = NS(w_outcome=1.0, beta_loop=0.0)
    s0 = C.objective_snapshot(a, cfg)
    assert C.check_resume({"objective": s0}, s0, "") == []
    b = parse(["--lr", "5e-7"])
    s1 = C.objective_snapshot(b, cfg)
    diffs = C.check_resume({"objective": s0}, s1, "")
    assert diffs and "lr" in diffs[0]
    assert C.check_resume({"objective": s0}, s1, "lr") == []
    s2 = C.objective_snapshot(a, NS(w_outcome=1.0, beta_loop=0.3))
    assert any("reward_config" in d for d in C.check_resume({"objective": s0}, s2, ""))
    assert C.check_resume({}, s0, "")                       # a checkpoint without a snapshot is refused ...
    assert C.check_resume({}, s0, "legacy") == []           # ... unless explicitly accepted
