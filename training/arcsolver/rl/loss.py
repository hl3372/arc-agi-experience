"""Torch pieces of the RL trainer: sampled-token log-probs, per-token advantage assembly, the PPO-clipped
policy-gradient loss, the reference KL term and the step's loss normalisers.

    A_t = A_seq                                   (group-normalised sequence reward, every completion token)
        + dense_t                                 (token rewards, e.g. the loop penalty: rewards.score_rollout)
    optionally clipped to [-c, c].

    loss_i = -(1/N_i) sum_t mask_t * min(rho_t A_t, clip(rho_t, 1-eps, 1+eps_high) A_t),   rho_t = pi_theta / pi_old
      token mean: N_i = all loss tokens of the step's kept rollouts (DAPO token-level mean)
      seq mean:   N_i = n_kept_rollouts * (loss tokens of rollout i) (GRPO: mean over rollouts of token means)

pi_old is the sampler's behaviour log-prob (or, with --ppo-old-logp trainer, the trainer's own at the start of the
step). The TIS + score-centering alternative to the PPO ratio is in sc.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# --- log-probs of the sampled tokens ------------------------------------------------------------------------------

def _chunk_logprobs(h, weight, targets):
    # bf16 matmul, fp32 log-softmax: the same precision path as vLLM's logprobs (bf16 logits .float());
    # fp32/fp64 inputs (tests) keep their precision
    logits = F.linear(h, weight)
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()
    return logits.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def token_logprobs(hidden, weight, targets, chunk: int = 2048, use_checkpoint: bool = True):
    """log softmax(hidden @ weight.T)[i, targets[i]] in fp32, shape [T].

    Never materialises more than `chunk` rows of vocab-size logits (248k x 2048 x 4 B = 2 GB); with
    checkpointing, backward recomputes each chunk's logits instead of keeping them."""
    if hidden.shape[0] != targets.shape[0]:
        raise ValueError(f"{hidden.shape[0]} hidden states for {targets.shape[0]} targets")
    outs = []
    need_grad = torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad)
    for s in range(0, hidden.shape[0], chunk):
        h, t = hidden[s:s + chunk], targets[s:s + chunk]
        if use_checkpoint and need_grad:
            outs.append(checkpoint(_chunk_logprobs, h, weight, t, use_reentrant=False))
        else:
            outs.append(_chunk_logprobs(h, weight, t))
    return torch.cat(outs)


def reduce_stats(acc: dict, stats: dict) -> dict:
    """Accumulate one rollout's stats (python floats; max for keys ending in _max)."""
    for k, v in stats.items():
        v = float(v)
        acc[k] = max(acc.get(k, 0.0), v) if k.endswith("_max") else acc.get(k, 0.0) + v
    return acc


def dense_tensor(n: int, dense: dict, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """{token index: reward} -> [n] tensor."""
    out = torch.zeros(n, dtype=dtype, device=device)
    if dense:
        idx = torch.tensor(list(dense.keys()), dtype=torch.long, device=device)
        out[idx] = torch.tensor(list(dense.values()), dtype=dtype, device=device)
    return out


def assemble_advantage(seq_adv: float, dense: torch.Tensor, clip: float = 0.0) -> torch.Tensor:
    """A_t for one rollout ([T]): the sequence advantage on every token plus the token rewards, clipped to [-clip, clip]
    when clip > 0."""
    adv = dense + float(seq_adv)
    if clip and clip > 0:
        adv = adv.clamp(-clip, clip)
    return adv


def pg_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
            n_total: float, clip_eps: float = 0.2, clip_eps_high: float | None = None):
    """One rollout's share of the step loss: -(1/n_total) sum_t mask_t * PPO-clipped surrogate (see module doc).

    Summing this over all rollouts of a step (one backward each) gives the step's loss. logp_new: [T] fp32 with grad
    (trainer); logp_old (behaviour log-probs), adv, mask: [T]. clip_eps_high (DAPO clip-higher): the ratio's upper clip
    bound is 1 + clip_eps_high instead of 1 + clip_eps; None = symmetric. Returns (loss, stats) with stats = detached
    sums over the loss tokens (pg_metrics)."""
    if not (logp_new.shape == logp_old.shape == adv.shape == mask.shape):
        raise ValueError(f"shape mismatch {logp_new.shape} {logp_old.shape} {adv.shape} {mask.shape}")
    mask = mask.to(logp_new.dtype)
    lp_det = logp_new.detach()
    adv = adv.to(logp_new.dtype)
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * adv
    hi = clip_eps if clip_eps_high is None else clip_eps_high
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + hi) * adv
    surr = torch.minimum(surr1, surr2)
    loss = -(surr * mask).sum() / n_total
    with torch.no_grad():
        on = mask > 0
        mis = (logp_old - lp_det) * mask                 # k1 estimate of KL(behaviour || trainer)
        r = ratio.detach()
        stats = {
            "n_tok": mask.sum(),
            "slp_sum": (lp_det * mask).sum(),
            "adv_sum": (adv * mask).sum(),
            "adv_sq_sum": (adv * adv * mask).sum(),
            "mis_sum": mis.sum(),
            "mis_abs_sum": mis.abs().sum(),
            "mis_abs_max": mis.abs().max() if mis.numel() else mis.new_zeros(()),
            "clipped": ((surr2 < surr1) & on).sum().to(mask.dtype),   # clipped branch active: zero grad
            "rho_lo": ((r < 1.0 - clip_eps) & on).sum().to(mask.dtype),
            "rho_hi": ((r > 1.0 + hi) & on).sum().to(mask.dtype),
        }
    return loss, stats


def kl_k3(logp_new: torch.Tensor, logp_ref: torch.Tensor, mask: torch.Tensor, n_total: float, coef: float,
          clamp: float = 10.0):
    """Reference KL penalty: per token the low-variance k3 estimator of
    KL(pi_theta || pi_ref), r = lp_ref - lp_theta, k3 = exp(r) - r - 1 >= 0 (GRPO, DeepSeekMath eq. 4), clamped at
    `clamp` as verl's low_var_kl; the gradient flows through lp_theta only. Returns (coef * sum_t mask_t k3_t / N,
    stats) with the same normaliser N as the policy-gradient term (token mean: all loss tokens of the step)."""
    r = (logp_ref.detach().to(logp_new.dtype) - logp_new).clamp(-30.0, 30.0)     # exp never overflows (inf*0 = NaN)
    k3 = (torch.exp(r) - r - 1.0).clamp(max=clamp)
    m = mask.to(logp_new.dtype)
    kl = (k3 * m).sum()
    stats = {"kl_sum": float(kl.detach()), "kl_n": float(m.sum()), "kl_max": float((k3.detach() * m).max()) if m.numel() else 0.0,
             "kl_k1_sum": float((-r.detach() * m).sum())}
    return coef * kl / n_total, stats


def kl_metrics(acc: dict) -> dict:
    """Token means of the reference KL terms (reduce_stats sums; kl_max takes the max)."""
    if not acc.get("kl_n"):
        return {}
    n = acc["kl_n"]
    return {"kl/k3_mean": acc.get("kl_sum", 0.0) / n, "kl/k1_mean": acc.get("kl_k1_sum", 0.0) / n,
            "kl/k3_max": acc.get("kl_max", 0.0), "kl/tokens": n}


def normalisers(mask_sums: list, agg: str) -> list:
    """Per kept rollout: the N_i of the module doc."""
    if agg == "token":
        tot = float(sum(mask_sums))
        return [tot] * len(mask_sums)
    if agg == "seq":
        k = len(mask_sums)
        return [float(k * max(m, 1)) for m in mask_sums]
    raise ValueError(agg)


def adv_stats(adv: torch.Tensor, mask: torch.Tensor) -> dict:
    """Sums for the step's token-level advantage statistics."""
    m = mask.to(adv.dtype)
    on = m > 0
    a = adv[on]
    if a.numel() == 0:
        return {"n": 0.0, "sum": 0.0, "sq": 0.0, "pos": 0.0, "neg": 0.0, "min": 0.0, "max": 0.0}
    return {"n": float(a.numel()), "sum": float(a.sum()), "sq": float((a * a).sum()),
            "pos": float((a > 0).sum()), "neg": float((a < 0).sum()), "min": float(a.min()), "max": float(a.max())}


def reduce_adv_stats(acc: dict, s: dict) -> dict:
    for k, v in s.items():
        if k == "min":
            acc[k] = min(acc.get(k, v), v)
        elif k == "max":
            acc[k] = max(acc.get(k, v), v)
        else:
            acc[k] = acc.get(k, 0.0) + v
    return acc


def adv_metrics(acc: dict) -> dict:
    n = max(acc.get("n", 0.0), 1.0)
    mean = acc.get("sum", 0.0) / n
    return {"adv/token_mean": mean, "adv/token_std": max(acc.get("sq", 0.0) / n - mean * mean, 0.0) ** 0.5,
            "adv/token_pos_frac": acc.get("pos", 0.0) / n, "adv/token_neg_frac": acc.get("neg", 0.0) / n,
            "adv/token_min": acc.get("min", 0.0), "adv/token_max": acc.get("max", 0.0)}


def pg_metrics(acc: dict) -> dict:
    """The ratio / mismatch statistics of pg_loss (reduce_stats sums)."""
    n = max(acc.get("n_tok", 0.0), 1.0)
    return {"pg/loss_tokens": acc.get("n_tok", 0.0),
            "pg/mismatch_kl_k1": acc.get("mis_sum", 0.0) / n,
            "pg/mismatch_abs": acc.get("mis_abs_sum", 0.0) / n,
            "pg/mismatch_abs_max": acc.get("mis_abs_max", 0.0),
            "pg/clip_frac": acc.get("clipped", 0.0) / n,
            "pg/rho_below_frac": acc.get("rho_lo", 0.0) / n,
            "pg/rho_above_frac": acc.get("rho_hi", 0.0) / n,
            "pg/entropy_est": -acc.get("slp_sum", 0.0) / n}
