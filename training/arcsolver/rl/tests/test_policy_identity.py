"""Policy identity in the sampler: records carry the engine's policy, the coordinator accepts only its committed policy,
a reload commits a policy only when every engine acknowledges it, and the eval store manifest distinguishes policies
and full logit-bias maps."""
from types import SimpleNamespace as NS

import pytest

from arcsolver.rl import train as D


class FakeEngine:
    def __init__(self, name, policy, ack=None):
        self.name, self.policy, self.ack = name, policy, ack

    def call(self, cmd, **kw):
        if cmd == "reload":
            self.policy = self.ack if self.ack is not None else kw.get("policy")
            return {"seconds": 0.1, "policy": self.policy}
        if cmd == "generate":
            outs = [[{"token_ids": [1, 2], "logprobs": [-0.1, -0.2], "policy": self.policy}
                     for _ in range(n)] for n in (kw.get("n_per_prompt") or [1] * len(kw["prompts"]))]
            return {"outputs": outs, "gen_tokens": 2, "seconds": 0.0}
        return {"seconds": 0.0}


def multi(policies, committed):
    return D.MultiEngine([FakeEngine(f"e{i}", p) for i, p in enumerate(policies)], policy=committed)


def test_mixed_policy_records_are_rejected():
    m = multi(["P", "P"], "P")
    r = m.call("generate", prompts=[[1], [2], [3]], params={}, max_tokens=8, seeds=[1, 2, 3], n_per_prompt=[2, 2, 2])
    assert len(r["outputs"]) == 3
    m2 = multi(["P", "Q"], "P")                      # engine 1 still serves another policy
    with pytest.raises(RuntimeError, match="mixed-policy"):
        m2.call("generate", prompts=[[1], [2]], params={}, max_tokens=8, seeds=[1, 2], n_per_prompt=[1, 1])


def test_reload_commits_only_on_every_acknowledgement():
    m = multi(["P", "P"], "P")
    m.call("reload", weights_path="/x", policy="Q")
    assert m.committed_policy == "Q"
    bad = D.MultiEngine([FakeEngine("e0", "Q"), FakeEngine("e1", "Q", ack="STALE")], policy="Q")
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        bad.call("reload", weights_path="/y", policy="R")
    assert bad.committed_policy is None              # nothing is committed after a failed acknowledgement
    m.call("reload", weights_path="/z")              # a reload without an identity leaves the sampler untracked
    assert m.committed_policy is None


def test_eval_manifest_policy_and_full_bias_map():
    args = NS(eval_temperature=1.0, eval_top_p=1.0, eval_top_k=0, eval_max_tokens=100, sampler_max_model_len=200,
              eval_budget_force=0)
    m = lambda pol, lb: D.eval_store_manifest(args, "t", "/m", pol, ["a", "b"], 4, 1, 2, lb)  # noqa: E731
    assert m("P11", None) != m("P22", None)                       # another live policy: no reuse
    assert m("P", {"5": -1.0}) != m("P", {"5": -10.0})              # another bias map: no reuse
    assert m("P", {"5": -1.0}) == m("P", {"5": -1.0})


def test_behaviour_logprobs_checked():
    m = multi(["P"], "P")
    ok = {"token_ids": [1, 2], "logprobs": [-0.1, -2.0], "policy": "P"}
    D.check_behaviour_logprobs(ok, m)
    D.check_behaviour_logprobs({"token_ids": [], "logprobs": None}, m)                  # empty completion: nothing
    for bad, msg in (({"token_ids": [1, 2], "logprobs": None}, "misaligned"),
                     ({"token_ids": [1, 2], "logprobs": [-0.1]}, "misaligned"),
                     ({"token_ids": [1, 2], "logprobs": [-0.1, float("nan")], "policy": "P"}, "non-finite"),
                     ({"token_ids": [1, 2], "logprobs": [-0.1, float("-inf")], "policy": "P"}, "non-finite"),
                     ({"token_ids": [1, 2], "logprobs": [-0.1, -0.2], "policy": "OLD"}, "mixed-policy")):
        with pytest.raises(RuntimeError, match=msg):
            D.check_behaviour_logprobs(bad, m)
