"""The policy-gradient loss (loss.pg_loss), advantage assembly, token / sequence aggregation, the reference KL term
(loss.kl_k3), TIS + score centering aggregation (sc.sc_tis_loss) and the AdamW parameter groups. Needs torch."""
import math

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "tensor") or not hasattr(torch, "autograd"):
    pytest.skip("real torch not installed", allow_module_level=True)

from arcsolver.rl import loss as rl  # noqa: E402
from arcsolver.rl import rewards as rr  # noqa: E402
from arcsolver.rl import sc as rl_sc  # noqa: E402

from .helpers import StubTok, token_text  # noqa: E402


def _toy(T=6, V=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(T, V, generator=g, dtype=torch.float64).requires_grad_(True)
    x = torch.multinomial(torch.softmax(z.detach(), -1), 1, generator=g).squeeze(-1)
    return z, x


def test_pg_loss_uses_exactly_the_given_advantage():
    """On-policy (sampler == trainer) the gradient is -(1/N) sum_t A_t grad log p(x_t): REINFORCE with our A_t."""
    z, x = _toy()
    T = z.shape[0]
    lp = torch.log_softmax(z, -1).gather(-1, x[:, None]).squeeze(-1)
    A = torch.tensor([1.0, -0.5, 0.0, 2.0, -1.0, 0.3], dtype=torch.float64)
    loss, stats = rl.pg_loss(lp, lp.detach(), A, torch.ones(T, dtype=torch.float64), n_total=10.0)
    loss.backward()
    onehot = torch.nn.functional.one_hot(x, z.shape[1]).double()
    expected = -(A[:, None] * (onehot - torch.softmax(z.detach(), -1))) / 10.0
    assert torch.allclose(z.grad, expected, atol=1e-12)
    assert math.isclose(float(loss), -float(A.sum()) / 10.0, rel_tol=1e-12)
    assert float(stats["adv_sq_sum"]) == pytest.approx(float((A * A).sum()))
    assert float(stats["clipped"]) == 0.0 and float(stats["mis_abs_sum"]) == 0.0


def test_pg_loss_ratio_clipping():
    V = 4
    z = torch.zeros(4, V, dtype=torch.float64, requires_grad=True)
    lp = torch.log_softmax(z, -1)[:, 0]
    rho = torch.tensor([1.5, 1.5, 0.5, 0.5], dtype=torch.float64)
    A = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    loss, stats = rl.pg_loss(lp, lp.detach() - torch.log(rho), A, torch.ones(4, dtype=torch.float64), 4.0, clip_eps=0.2)
    loss.backward()
    dlogp = torch.tensor([0.75, -0.25, -0.25, -0.25], dtype=torch.float64)
    for t, active in enumerate([False, True, True, False]):          # A>0 & rho>1.2 and A<0 & rho<0.8 are clipped
        want = -(rho[t] * A[t] * dlogp) / 4 if active else torch.zeros(V, dtype=torch.float64)
        assert torch.allclose(z.grad[t], want, atol=1e-12)
    assert float(stats["clipped"]) == 2


def test_clip_higher_asymmetric_bound():
    """--clip-eps-high: a positive-advantage token whose ratio rose to 1.25 still gets gradient with an upper bound of
    1.28, but not with the symmetric 0.2 bound; the lower bound is unchanged."""
    lp_s = torch.log(torch.tensor([0.2, 0.2]))
    for eps_hi, want_grad in ((None, False), (0.28, True)):
        lp_new = torch.log(torch.tensor([0.25, 0.2])).clone().requires_grad_(True)   # ratio 1.25 and 1.0
        adv = torch.tensor([1.0, 1.0])
        mask = torch.tensor([1.0, 0.0])
        loss, _ = rl.pg_loss(lp_new, lp_s, adv, mask, 1.0, clip_eps=0.2, clip_eps_high=eps_hi)
        loss.backward()
        assert (lp_new.grad[0].abs() > 0) == want_grad
    lp_new = torch.log(torch.tensor([0.15, 0.2])).clone().requires_grad_(True)       # ratio 0.75, negative adv
    loss, _ = rl.pg_loss(lp_new, lp_s, torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0]), 1.0, clip_eps=0.2,
                         clip_eps_high=0.28)
    loss.backward()
    assert lp_new.grad[0].abs() == 0                                               # below 1 - 0.2: still clipped


def test_assemble_advantage_terms_and_clip():
    dense = rl.dense_tensor(5, {1: -0.3, 3: 0.5}, dtype=torch.float64)
    assert dense.tolist() == [0.0, -0.3, 0.0, 0.5, 0.0]
    a = rl.assemble_advantage(0.7, dense)
    assert torch.allclose(a, torch.tensor([0.7, 0.4, 0.7, 1.2, 0.7], dtype=torch.float64))
    c = rl.assemble_advantage(2.0, dense, clip=1.0)
    assert c.max() == 1.0 and c.min() == 1.0
    assert torch.equal(rl.assemble_advantage(-0.5, dense, 0.0), dense - 0.5)


def test_token_mean_and_sequence_mean_aggregation():
    """token: one backward on the concatenation / all tokens; seq: mean over rollouts of per-rollout token means."""
    z, x = _toy(T=9, V=5, seed=3)
    A = torch.linspace(-1, 1, 9, dtype=torch.float64)
    parts = [slice(0, 2), slice(2, 9)]
    lens = [2, 7]

    def grad(norms):
        z.grad = None
        for sl, n in zip(parts, norms):
            lp = torch.log_softmax(z[sl], -1).gather(-1, x[sl, None]).squeeze(-1)
            loss, _ = rl.pg_loss(lp, lp.detach(), A[sl], torch.ones_like(lp), n)
            loss.backward()
        return z.grad.clone()

    g_tok = grad(rl.normalisers(lens, "token"))
    lp_all = torch.log_softmax(z, -1).gather(-1, x[:, None]).squeeze(-1)
    z.grad = None
    rl.pg_loss(lp_all, lp_all.detach(), A, torch.ones(9, dtype=torch.float64), 9.0)[0].backward()
    assert torch.allclose(g_tok, z.grad, atol=1e-12)
    g_seq = grad(rl.normalisers(lens, "seq"))
    z.grad = None
    tot = 0
    for sl, n in zip(parts, lens):
        lp = torch.log_softmax(z[sl], -1).gather(-1, x[sl, None]).squeeze(-1)
        tot = tot + (-(A[sl] * lp).sum() / n) / 2
    tot.backward()
    assert torch.allclose(g_seq, z.grad, atol=1e-12)
    assert rl.normalisers([3, 5], "token") == [8.0, 8.0] and rl.normalisers([3, 5], "seq") == [6.0, 10.0]


def test_end_to_end_advantages_from_rewards():
    """Rewards -> group advantages -> dynamic sampling -> per-token advantages: a correct rollout pushes up every
    token, a looping wrong one pushes down (more on its loop tokens), a dropped group contributes nothing."""
    stub = StubTok()
    task = {"train": [], "test": [{"input": [[0]], "output": [[1]]}]}
    texts = [("Rule.</think>\n\n1<|im_end|>", "stop"),
             ("Intro.\n\n" + "\n\n".join(["Check cell 3."] * 6), "length"),
             ("Other.</think>\n\n2<|im_end|>", "stop"), ("More.</think>\n\n3<|im_end|>", "stop")]
    cfg = rr.RewardConfig(beta_loop=0.3)
    recs = [rr.score_rollout(task, token_text(stub, t), f, cfg) for t, f in texts]
    gids = [0, 0, 1, 1]
    adv = rr.group_advantages([r["r_seq"] for r in recs], gids, "std")
    dropped = rr.zero_variance_groups([r["r_seq"] + r["dense_sum"] for r in recs], gids)
    assert dropped == {1}                                             # both wrong, no loops: nothing to learn
    a0 = rl.assemble_advantage(adv[0], rl.dense_tensor(recs[0]["stats"]["n_tokens"], recs[0]["dense"], torch.float64))
    a1 = rl.assemble_advantage(adv[1], rl.dense_tensor(recs[1]["stats"]["n_tokens"], recs[1]["dense"], torch.float64))
    assert adv[0] > 0 > adv[1] and bool((a0 == adv[0]).all())
    lo, hi = recs[1]["loop_range"]
    assert hi > lo and torch.allclose(a1[lo:hi], torch.full((hi - lo,), adv[1] - 0.3, dtype=torch.float64))
    assert torch.allclose(a1[:lo], torch.full((lo,), adv[1], dtype=torch.float64))


def test_kl_k3_value_gradient_and_normaliser():
    ref = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    new = torch.tensor([-1.0, -1.5, -0.7, -3.0], requires_grad=True)
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    loss, st = rl.kl_k3(new, ref, mask, n_total=10.0, coef=0.001)
    r = ref - new.detach()
    k3 = torch.exp(r) - r - 1
    assert torch.isclose(loss, 0.001 * (k3 * mask).sum() / 10.0)
    assert st["kl_n"] == 3.0 and abs(st["kl_sum"] - float((k3 * mask).sum())) < 1e-6
    loss.backward()
    # d/dlp k3 = 1 - exp(lp_ref - lp): zero where the policy equals the reference, masked tokens get nothing
    want = 0.001 * (1 - torch.exp(r)) * mask / 10.0
    assert torch.allclose(new.grad, want)
    assert new.grad[0] == 0.0 and new.grad[3] == 0.0
    same = torch.tensor([-1.0, -2.0], requires_grad=True)             # equal policies: exactly zero
    l0, s0 = rl.kl_k3(same, same.detach().clone(), torch.ones(2), 2.0, 0.001)
    assert float(l0) == 0.0 and s0["kl_sum"] == 0.0
    big = torch.tensor([-30.0], requires_grad=True)                   # k3 >= 0 and clamped
    lb, sb = rl.kl_k3(big, torch.tensor([0.0]), torch.ones(1), 1.0, 1.0)
    assert float(lb) == 10.0 and sb["kl_max"] == 10.0
    m = rl.kl_metrics({"kl_sum": 3.0, "kl_n": 6.0, "kl_k1_sum": 1.5, "kl_max": 0.9})
    assert m["kl/k3_mean"] == 0.5 and m["kl/k1_mean"] == 0.25 and m["kl/k3_max"] == 0.9
    assert rl.kl_metrics({}) == {}


def _sc_grad(T, A, norm):
    th = torch.zeros((), requires_grad=True)
    lp = th + torch.full((T,), -1.0)                   # every token's log-prob moves with th
    e = torch.zeros((T, 0))
    loss, _ = rl_sc.sc_tis_loss(lp, e, lp.detach(), e, e.bool(), torch.full((T,), float(A)), torch.ones(T), norm,
                                tis_cap=2.0, score_centering=False)
    loss.backward()
    return float(th.grad)


def test_seq_aggregation_equal_weight_per_rollout():
    lens = [100, 98000]
    seq = rl.normalisers(lens, "seq")
    g = [_sc_grad(T, -0.5, n) for T, n in zip(lens, seq)]
    assert abs(g[0] - g[1]) < 1e-6 and abs(g[0] - 0.5 / len(lens)) < 1e-6      # -A / n_rollouts each
    tok = rl.normalisers(lens, "token")
    gt = [_sc_grad(T, -0.5, n) for T, n in zip(lens, tok)]
    assert abs(gt[1] / gt[0] - lens[1] / lens[0]) < 1e-3 * lens[1] / lens[0]    # token-mean: weight ~ length


def test_optimizer_beta2_and_matrix_weight_decay():
    from arcsolver.rl import fsdp as F
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    o = F.optimizer(m, 1e-5, 1e-8)
    assert len(o.param_groups) == 1 and o.param_groups[0]["weight_decay"] == 0.0 and o.param_groups[0]["betas"] == (0.9, 0.999)
    o = F.optimizer(m, 1e-5, 1e-8, beta2=0.99, weight_decay=0.01)
    assert [g["weight_decay"] for g in o.param_groups] == [0.01, 0.0]
    assert [len(g["params"]) for g in o.param_groups] == [1, 3]       # the Linear weight only; biases + norm: none
    assert all(g["betas"] == (0.9, 0.99) for g in o.param_groups)
