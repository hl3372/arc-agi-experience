"""Score centering for RL under a training-inference mismatch (Marek & Ryabinin, "Score Centering Stabilizes
Off-policy Reinforcement Learning", arXiv 2609.20807; reference code github.com/martin-marek/score-centering).

Rollouts come from the sampler's distribution q (vLLM, FP8 weights / FP8 KV cache), gradients from the trainer's p
(bf16 compute). Per token, E_q[A * grad log p(y)] = E_q[A] * s_bar + Cov_q(A, grad log p(y)) with
s_bar = sum_v q_v grad log p_v, which is not 0 when q != p: the "drift" term, which compounds because the sampler is
synced from the trainer every step. Score centering subtracts s_bar, keeping only the covariance (the signal).

s_bar is estimated from the sampler's top-k (ids + log-probs, returned by vLLM at every generated position) and the
trainer's own distribution for the tail, rescaled by rho = q_tail / p_tail. Composed with truncated importance
sampling (TIS, weight f(r) = min(r, C) with r = p/q) as in the paper's Eq. 11-12:

    g_t = w_y * log p(y_t) - sum_{v in head} sg[q_v * w_v - alpha * p_v] * log p(v),
    w_v = f(p_v / q_v),  alpha = rho * f(1 / rho),  rho = (1 - sum_head q) / (1 - sum_head p)
    loss = -(1/N) sum_t mask_t * A_t * g_t        (token mean over the step, A_t detached)

With TIS off f = 1 (alpha = rho); with SC off the correction term is dropped (plain TIS policy gradient). One update
per batch (the trainer's old policy is the current one), so no PPO ratio is needed: the importance weight carries
the trainer/sampler difference instead (the PPO-ratio-against-the-sampler form collapses under quantized rollouts,
FlashRL 2025).

Pure torch: shared by the FSDP trainer (torch 2.10, GPU) and the CPU tests.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# --- trainer log-probs at the sampled token and at the sampler's top-k ----------------------------------------

def _chunk_token_and_head(h, weight, targets, head_ids, vocab=None):
    logits = F.linear(h, weight)
    if vocab is not None and vocab < logits.shape[-1]:
        logits = logits[:, :vocab]                       # drop the alignment rows (pad_vocab)
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()
    logz = torch.logsumexp(logits, dim=-1)
    lp_y = logits.gather(-1, targets[:, None]).squeeze(-1) - logz
    lp_h = logits.gather(-1, head_ids.clamp(min=0)) - logz[:, None]
    return lp_y, lp_h


def pad_vocab(weight, multiple: int = 128):
    """The lm_head with zero rows appended up to a multiple of `multiple`. A pruned vocab of odd size (e.g. 128,459)
    gives [rows, V] logits a misaligned row stride and cuBLAS falls back to slow SM75 kernels for the head GEMMs; the
    padded rows are sliced off before the softmax, so the values are unchanged and their gradient is zero."""
    pad = -weight.shape[0] % multiple
    return F.pad(weight, (0, 0, 0, pad)) if pad else weight


def token_and_head_logprobs(hidden, weight, targets, head_ids, chunk: int = 2048, use_checkpoint: bool = True):
    """(log p(targets[i]) [T], log p(head_ids[i, j]) [T, k]) under softmax(hidden @ weight.T), fp32, chunked over
    positions with checkpointing so at most `chunk` rows of vocab-size logits exist at once. head_ids entries < 0
    are padding (their log-probs are garbage; mask them with head_ids >= 0)."""
    if not (hidden.shape[0] == targets.shape[0] == head_ids.shape[0]):
        raise ValueError(f"{hidden.shape[0]} hidden states, {targets.shape[0]} targets, {head_ids.shape[0]} head rows")
    ys, hs = [], []
    need_grad = torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad)
    vocab = weight.shape[0]
    weight = pad_vocab(weight)
    for s in range(0, hidden.shape[0], chunk):
        args = (hidden[s:s + chunk], weight, targets[s:s + chunk], head_ids[s:s + chunk], vocab)
        y, h = (checkpoint(_chunk_token_and_head, *args, use_reentrant=False) if use_checkpoint and need_grad
                else _chunk_token_and_head(*args))
        ys.append(y)
        hs.append(h)
    return torch.cat(ys), torch.cat(hs)


def head_valid(head_ids):
    """Usable head entries [T, k]: not padding (ids >= 0) and the first listing of a token in its row. Top-k arrays
    may list the sampled token twice (vLLM's FlatLogprobs puts it before the top-k); counting it twice would inflate
    the head sums and push q_tail and p_tail to their clamp."""
    s, idx = head_ids.sort(dim=-1)
    rep = torch.zeros_like(s, dtype=torch.bool)
    rep[..., 1:] = s[..., 1:] == s[..., :-1]
    return (head_ids >= 0) & ~torch.zeros_like(rep).scatter_(-1, idx, rep)


# --- the loss --------------------------------------------------------------------------------------------------

SC_STAT_KEYS = ("n_tok", "mis_k1_sum", "mis_k3_sum", "mis_abs_sum", "w_sum", "w_clipped", "rho_sum", "q_tail_sum",
                "p_tail_sum", "corr_abs_sum", "adv_sum", "slp_sum")


def sc_tis_loss(lp_y, lp_head, q_lp_y, q_lp_head, head_valid, adv, mask, n_total: float, tis_cap: float = 2.0,
                score_centering: bool = True, eps: float = 1e-6):
    """One rollout's share of the step loss (see the module docstring). Shapes: lp_y, q_lp_y, adv, mask [T];
    lp_head, q_lp_head, head_valid [T, k]. lp_* carry the trainer's gradient; q_* are the sampler's log-probs
    (constants); tis_cap <= 0 turns TIS off (w = 1). Returns (loss, stats) with stats = detached sums."""
    if not (lp_y.shape == q_lp_y.shape == adv.shape == mask.shape):
        raise ValueError(f"shape mismatch {lp_y.shape} {q_lp_y.shape} {adv.shape} {mask.shape}")
    if not (lp_head.shape == q_lp_head.shape == head_valid.shape):
        raise ValueError(f"head shape mismatch {lp_head.shape} {q_lp_head.shape} {head_valid.shape}")
    mask = mask.to(lp_y.dtype)
    adv = adv.to(lp_y.dtype)
    valid = head_valid.to(lp_y.dtype)
    log_cap = math.log(tis_cap) if tis_cap and tis_cap > 0 else None
    with torch.no_grad():
        d_y = lp_y.detach() - q_lp_y                       # log(p/q) at the sampled token
        w_y = torch.exp(d_y.clamp(max=log_cap)) if log_cap is not None else torch.ones_like(d_y)
        if score_centering:
            p_h = torch.exp(lp_head.detach()) * valid
            q_h = torch.exp(q_lp_head) * valid
            p_tail = (1.0 - p_h.sum(-1)).clamp(min=eps)
            q_tail = (1.0 - q_h.sum(-1)).clamp(min=eps)
            rho = q_tail / p_tail
            if log_cap is not None:
                w_h = torch.exp((lp_head.detach() - q_lp_head).clamp(max=log_cap)) * valid
                alpha = rho * torch.clamp(1.0 / rho, max=tis_cap)
            else:
                w_h = valid
                alpha = rho
            residual = (q_h * w_h - alpha[:, None] * p_h) * valid
        else:
            residual = None
    g = w_y * lp_y
    if residual is not None:
        g = g - (residual * lp_head).sum(-1)
    loss = -(mask * adv * g).sum() / n_total
    with torch.no_grad():
        on = mask > 0
        stats = {
            "n_tok": mask.sum(),
            "mis_k1_sum": ((-d_y) * mask).sum(),                       # k1 estimate of KL(q || p), samples from q
            "mis_k3_sum": ((torch.exp(d_y) - 1 - d_y) * mask).sum(),   # k3 estimate (>= 0)
            "mis_abs_sum": (d_y.abs() * mask).sum(),
            "w_sum": (w_y * mask).sum(),
            "w_clipped": (((d_y > log_cap) & on).sum().to(mask.dtype) if log_cap is not None else mask.new_zeros(())),
            "rho_sum": ((rho * mask).sum() if residual is not None else mask.new_zeros(())),
            "q_tail_sum": ((q_tail * mask).sum() if residual is not None else mask.new_zeros(())),
            "p_tail_sum": ((p_tail * mask).sum() if residual is not None else mask.new_zeros(())),
            "corr_abs_sum": ((residual.abs().sum(-1) * mask).sum() if residual is not None else mask.new_zeros(())),
            "adv_sum": (adv * mask).sum(),
            "slp_sum": (lp_y.detach() * mask).sum(),                  # -> pg/entropy_est (loss.pg_metrics)
        }
    return loss, stats


def sc_metrics(acc: dict) -> dict:
    """Step metrics from summed stats (python floats)."""
    n = max(acc.get("n_tok", 0.0), 1.0)
    return {"sc/mismatch_k1": acc.get("mis_k1_sum", 0.0) / n, "sc/mismatch_k3": acc.get("mis_k3_sum", 0.0) / n,
            "sc/mismatch_abs": acc.get("mis_abs_sum", 0.0) / n, "sc/tis_w_mean": acc.get("w_sum", 0.0) / n,
            "sc/tis_clip_frac": acc.get("w_clipped", 0.0) / n, "sc/rho_mean": acc.get("rho_sum", 0.0) / n,
            "sc/q_tail_mean": acc.get("q_tail_sum", 0.0) / n, "sc/p_tail_mean": acc.get("p_tail_sum", 0.0) / n,
            "sc/correction_l1_mean": acc.get("corr_abs_sum", 0.0) / n, "sc/loss_tokens": n}
