"""Verifiable rewards for RL on ARC (pure python: no torch, no vLLM).

Per rollout (a sampled completion of one task prompt; thinking mode, the answer follows "</think>"):

  sequence-level (summed into R_seq, then group-normalised per prompt, broadcast to every completion token)
    outcome   w_outcome * mean_j [attempt-1 grid j == test output j]   (arc_render parsing and attempt-1 rule;
              by default only if the sample finished cleanly, finish == "stop"); with outcome_all_tests 1 only if
              every test output is right. A test pair's "alt_outputs" (optional) count as right too.
    format    w_format if the sample finished within budget (finish == "stop"), closed </think>, and wrote exactly
              one rectangular grid per test input, each with the SHAPE of the true test output (--format-shape
              output; "none" = well-formedness only), and no grid is a verbatim copy of its test input (unless the
              true output equals the input). Guards against "close early with junk".
    loop_seq  -w_loop_seq if the reasoning loops (see loop region below)
  token-level (added at their own tokens only; no return propagation)
    loop      -beta_loop on every token of the loop region: from the first char of the loop-onset paragraph to the
              end of the reasoning (the "</think>" char, or the end of a truncated sample). Onset = core.loop_kind's
              paragraph detector (paragraph i repeats an earlier one and >= 80% of the >= 3 paragraphs from i on are
              repeats). detect="templated" compares digit-masked paragraphs, which also finds every verbatim loop
              (its onset is never later than the verbatim onset); detect="verbatim" = exact repeats only.
    overlong  -w_len per token past len_soft, ramping up to len_cap (off by default).

The released recipes use the binary outcome only (w_outcome 1, every other weight 0). The group-level shapings
(MaxRL / floored MaxRL advantages, length shapings, forced-answer credit) are below.

Token positions come from TokenAligner: the completion ids are detokenised byte by byte (byte-level BPE), so every
char of the text belongs to exactly one token; special tokens (<|im_end|>) get an empty span, which reproduces
vLLM's text (skip_special_tokens=True). Loops are found in that text and mapped back through the spans.
"""
from __future__ import annotations

import bisect
import codecs
import math
import re
import statistics as st
from dataclasses import dataclass

from arcsolver.prompt import arc_render as ar
from arcsolver.rl import core as oc

THINK_CLOSE = "</think>"


# --- token <-> char alignment ------------------------------------------------------------------------------------

def _bytes_to_unicode() -> dict:
    """GPT-2 / Qwen byte-level BPE: byte -> printable unicode char used in the vocab strings."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}


@dataclass
class TokenText:
    """A detokenised completion: text plus the [starts[j], ends[j]) char span of token j (spans tile the text)."""
    text: str
    starts: list
    ends: list

    def __len__(self):
        return len(self.starts)

    def token_range(self, c0: int, c1: int) -> tuple:
        """[lo, hi): the tokens whose span overlaps chars [c0, c1) (plus empty-span tokens strictly inside)."""
        if c1 <= c0:
            return 0, 0
        lo = bisect.bisect_right(self.ends, c0)
        hi = bisect.bisect_left(self.starts, c1)
        return lo, max(lo, hi)

    def token_at(self, pos: int) -> int:
        """The token whose span contains char pos."""
        j = bisect.bisect_right(self.ends, pos)
        if j >= len(self.ends) or self.starts[j] > pos:
            raise IndexError(f"char {pos} is not covered by any token")
        return j

    def prefix_text(self, n_tokens: int) -> str:
        """The text of the first n_tokens tokens."""
        return self.text[:self.ends[n_tokens - 1]] if n_tokens > 0 else ""


class TokenAligner:
    """Byte-level detokeniser with per-token char spans (Qwen2/3/3.5 tokenizers).

    Ordinary tokens -> their bytes via the byte-level vocab strings; non-special added tokens ("<think>",
    "</think>") -> their content; special tokens (tokenizer.all_special_ids, e.g. <|im_end|>) -> "" when
    skip_special (vLLM's default detokenisation). An incremental UTF-8 decoder attributes a character split
    across tokens to the token that completes it (the earlier token gets an empty span)."""

    def __init__(self, tokenizer, skip_special: bool = True):
        self.tok = tokenizer
        self.special = {int(i) for i in tokenizer.all_special_ids} if skip_special else set()
        self.added = {int(i): t.content for i, t in tokenizer.added_tokens_decoder.items()}
        self._bytes = {}

    def token_bytes(self, i: int) -> bytes:
        b = self._bytes.get(i)
        if b is None:
            if i in self.special:
                b = b""
            elif i in self.added:
                b = self.added[i].encode("utf-8")
            else:
                s = self.tok.convert_ids_to_tokens(i)
                if s is None:
                    # an id with no token string: the model's vocab (e.g. 248,320) is padded past the tokenizer's
                    # (248,077 tokens), and unrestricted T=1.0 sampling can emit one; it carries no text
                    # (vLLM's detokeniser also emits nothing for it).
                    b = b""
                else:
                    try:
                        b = bytes(BYTE_DECODER[ch] for ch in s)
                    except KeyError as e:
                        raise ValueError(f"token {i} ({s!r}) is not a byte-level BPE token") from e
            self._bytes[i] = b
        return b

    def __call__(self, ids) -> TokenText:
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        parts, starts, ends = [], [], []
        n = 0
        for i in ids:
            s = dec.decode(self.token_bytes(int(i)))
            starts.append(n)
            n += len(s)
            ends.append(n)
            parts.append(s)
        tail = dec.decode(b"", final=True)
        if tail and ends:
            parts.append(tail)
            ends[-1] += len(tail)
        return TokenText("".join(parts), starts, ends)


# --- loops ------------------------------------------------------------------------------------------------------

def reasoning_text(text: str) -> str:
    """The reasoning part: everything before the first "</think>" (the whole text if never closed); for the
    render_output arm (arc_render default channel "output"), everything before the last "Final answer:" line."""
    return ar.reasoning_part(text)


def paragraphs(think: str) -> list:
    """core.loop_kind's paragraphs (split on "\\n\\n", stripped, empty ones dropped) with the char span of
    each stripped paragraph: [(start, end, text)]."""
    out, pos = [], 0
    for piece in think.split("\n\n"):
        s = piece.strip()
        if s:
            a = pos + (len(piece) - len(piece.lstrip()))
            out.append((a, a + len(s), s))
        pos += len(piece) + 2
    return out


def _mask_digits(p: str) -> str:
    return re.sub(r"\d+", "#", p)


def _first_repeat(keys, key):
    seen = set()
    for i, p in enumerate(keys):
        k = key(p)
        if k in seen:
            return i
        seen.add(k)
    return None


def loop_info(text: str, detect: str = "templated", onset_rule: str = "sustained") -> dict:
    """kind: "verbatim" | "templated" | "none" (exactly core.loop_kind); onset/c0/c1: the loop region
    [c0, c1) in chars under the chosen detector (None if it finds no loop).
    onset_rule "sustained" (default): the core.loop_kind onset (a repeat from which >= 80% of the >= 3 remaining
    paragraphs are repeats); "first_repeat": the first paragraph that repeats an earlier one (much more aggressive, and
    with detect "templated" it fires on any two paragraphs of the same digit-masked template)."""
    if detect not in ("templated", "verbatim") or onset_rule not in ("sustained", "first_repeat"):
        raise ValueError((detect, onset_rule))
    think = reasoning_text(text)
    paras = paragraphs(think)
    keys = [p[2] for p in paras]
    on_v = oc._loop_onset(keys, lambda p: p)
    on_t = oc._loop_onset(keys, _mask_digits)
    kind = "verbatim" if on_v is not None else ("templated" if on_t is not None else "none")
    if onset_rule == "first_repeat":
        onset = _first_repeat(keys, (lambda p: p) if detect == "verbatim" else _mask_digits)
    else:
        onset = on_v if detect == "verbatim" else on_t
    if onset is None:
        return {"kind": kind, "onset": None, "c0": None, "c1": None, "n_paras": len(paras)}
    return {"kind": kind, "onset": onset, "c0": paras[onset][0], "c1": len(think), "n_paras": len(paras)}


# --- answers ----------------------------------------------------------------------------------------------------

def _shape(g):
    return (len(g), len(g[0]) if g else 0)


def accepted_keys(test: dict) -> set:
    """grid_keys of every accepted answer of one test pair: its output plus any "alt_outputs" (credible alternative
    readings of an ambiguous task, all counted as correct)."""
    return {ar.grid_key(g) for g in [test["output"], *test.get("alt_outputs", ())]}


def answer_eval(task: dict, text: str, finish: str | None = None) -> dict:
    """Attempt-1 answer checks against the task JSON (arc_render parsing: the text after the LAST "</think>",
    then the arm's answer mode: strict = "Attempt 1:" block if present, blank-line separated digit-row blocks,
    ragged block -> None; lenient = the last n_test digit-row runs, prose ignored).
      closed       wrote "</think>"                   finished  finish == "stop" (ended by EOS within budget)
      em           per test output: exact match (of the output or any of the pair's "alt_outputs")
      em_frac      mean of em (0 if no ground truth)
      well_formed  exactly one rectangular grid per test input
      shape_ok     well_formed and every grid has its true output's shape
      copy         some grid equals its test input although the true output differs from it"""
    tests = task["test"]
    n = len(tests)
    ans, closed = ar.final_answer_text(text)
    grids = ar.answer_grids(ans, n)[0] if closed else None      # strict or lenient per the arm (configure_arm)
    truth = all("output" in t for t in tests)
    em = [False] * n
    if truth and grids is not None:
        em = [j < len(grids) and grids[j] is not None and ar.grid_key(grids[j]) in accepted_keys(tests[j])
              for j in range(n)]
    well_formed = grids is not None and len(grids) == n and all(g is not None for g in grids)
    shape_ok = well_formed and truth and all(_shape(grids[j]) in {_shape(g) for g in [tests[j]["output"],
                                                                         *tests[j].get("alt_outputs", ())]}
                                             for j in range(n))
    copy = well_formed and any(grids[j] == tests[j]["input"] and tests[j].get("output") != tests[j]["input"]
                               for j in range(n))
    return {"closed": closed, "finished": finish == "stop", "n_test": n, "n_grids": 0 if grids is None else len(grids),
            "em": em, "em_frac": sum(em) / n if n else 0.0, "em_all": bool(n) and all(em),
            "well_formed": well_formed, "shape_ok": shape_ok, "copy": copy}


# --- per-rollout rewards ----------------------------------------------------------------------------------------

@dataclass
class RewardConfig:
    w_outcome: float = 1.0
    outcome_require_stop: bool = True
    outcome_all_tests: bool = False         # True: 1 only if every test output is right (else the mean over outputs)
    w_format: float = 0.0
    format_shape: str = "output"            # "output" (true test-output shapes) | "none" (well-formed only)
    format_reject_copy: bool = True
    beta_loop: float = 0.3                  # per-token penalty in the loop region
    w_loop_seq: float = 0.0                 # sequence-level penalty if the reasoning loops
    loop_detect: str = "templated"
    loop_onset: str = "sustained"           # "sustained" | "first_repeat" (see loop_info)
    # Overlong penalty (DAPO-style soft overlong shaping, applied per token so it works outside MaxRL's
    # normalisation and on groups with no correct rollout): token j >= len_soft gets
    # -w_len * min(1, (j - len_soft + 1) / (len_cap - len_soft)). 0 = off.
    len_soft: int = 0
    len_cap: int = 0
    w_len: float = 0.0

def score_rollout(task: dict, tt: TokenText, finish: str, cfg: RewardConfig) -> dict:
    """All reward components of one rollout. tt = TokenAligner(completion ids).
    Returns seq (component -> value), r_seq (their sum), dense (token index -> reward: loop, overlong), and
    per-rollout statistics for logging."""
    n = len(tt)
    text = tt.text
    ans = answer_eval(task, text, finish)
    lp = loop_info(text, cfg.loop_detect, cfg.loop_onset)
    think = reasoning_text(text)
    closed_tok = None
    if ans["closed"]:
        closed_tok = tt.token_at(len(think)) if len(think) < len(text) else None
    seq = {"outcome": 0.0, "format": 0.0, "loop_seq": 0.0}
    if not cfg.outcome_require_stop or ans["finished"]:
        seq["outcome"] = cfg.w_outcome * (float(ans["em_all"]) if cfg.outcome_all_tests else ans["em_frac"])
    format_ok = (ans["finished"] and ans["closed"] and ans["well_formed"]
                 and (cfg.format_shape == "none" or ans["shape_ok"])
                 and not (cfg.format_reject_copy and ans["copy"]))
    if cfg.format_shape not in ("output", "none"):
        raise ValueError(cfg.format_shape)
    seq["format"] = cfg.w_format * float(format_ok)
    looped = lp["onset"] is not None
    seq["loop_seq"] = -cfg.w_loop_seq * float(looped)
    dense = {}
    loop_lo = loop_hi = 0
    if looped:
        loop_lo, loop_hi = tt.token_range(lp["c0"], lp["c1"])
        if cfg.beta_loop:
            for j in range(loop_lo, loop_hi):
                dense[j] = -cfg.beta_loop
    len_pen = 0.0
    if cfg.w_len and cfg.len_soft and n > cfg.len_soft:
        span = max(1, (cfg.len_cap or n) - cfg.len_soft)
        for j in range(cfg.len_soft, n):
            v = -cfg.w_len * min(1.0, (j - cfg.len_soft + 1) / span)
            dense[j] = dense.get(j, 0.0) + v
            len_pen += v
    pre_loop_think_tokens = (loop_lo if looped else
                             (closed_tok if closed_tok is not None else n))
    stats = {"n_tokens": n, "finish": finish, **{k: ans[k] for k in ("closed", "finished", "em_frac", "em_all",
                                                                       "well_formed", "shape_ok", "copy", "n_grids")},
             "format_ok": format_ok, "loop_kind": lp["kind"], "looped": looped,
             "loop_tokens": loop_hi - loop_lo, "loop_onset_tok": loop_lo if looped else None,
             "closed_tok": closed_tok, "think_tokens_pre_loop": pre_loop_think_tokens,
             "loop_reward_sum": -cfg.beta_loop * (loop_hi - loop_lo), "len_reward_sum": len_pen}
    return {"seq": seq, "r_seq": sum(seq.values()), "dense": dense, "dense_sum": sum(dense.values()),
            "loop_range": (loop_lo, loop_hi), "stats": stats}


# --- groups -----------------------------------------------------------------------------------------------------

def _groups(group_ids):
    by = {}
    for i, g in enumerate(group_ids):
        by.setdefault(g, []).append(i)
    return by


def group_advantages(values: list, group_ids: list, mode: str = "std", eps: float = 1e-4, floor: float = 0.0) -> list:
    """GRPO: (R - mean_group) / (std_group + eps) with the sample std (ddof 1, as TRL); "mean": R - mean_group
    (Dr. GRPO); "maxrl": (R - mean_group) / (mean_group + eps), the MaxRL estimator (Maximum Likelihood RL,
    arXiv 2602.02710, eq. 10 / Alg. 1: sum_i (r_i/K - 1/N) S_i = 1/N sum_i ((r_i - r_hat)/r_hat) S_i), i.e. dividing
    by the mean reward instead of the std. It weights rarely-solved prompts up (1/p instead of GRPO's
    1/sqrt(p(1-p))) and is an unbiased estimate of sum_k 1/k grad pass@k; a group with mean 0 gets 0.
    "maxrl_floor": (R - mean_group) / max(mean_group, floor) with a fixed floor (0.25 = at
    most +3.5 for 1 correct of 8): MaxRL for groups solved at least floor of the time, Dr. GRPO scaled by 1/floor
    below it. The mean is over every rollout of the group (truncated ones count as R = 0), so each group still sums
    to 0. A group of one gets 0."""
    if mode not in ("std", "mean", "maxrl", "maxrl_floor"):
        raise ValueError(mode)
    if mode == "maxrl_floor" and not floor > 0:
        raise ValueError("maxrl_floor needs floor > 0")
    out = [0.0] * len(values)
    for idx in _groups(group_ids).values():
        v = [float(values[i]) for i in idx]
        if len(v) < 2:
            continue
        m = sum(v) / len(v)
        if mode == "maxrl" and m <= 0:
            continue                                   # K = 0: no correct rollout, zero gradient (Alg. 1)
        if mode == "maxrl_floor":
            d = max(m, floor)
        else:
            d = st.stdev(v) + eps if mode == "std" else (m + eps if mode == "maxrl" else 1.0)
        for i in idx:
            out[i] = (values[i] - m) / d if d > 0 else 0.0
    return out


def length_scaled_rewards(values: list, lengths: list, group_ids: list, alpha: float) -> list:
    """Kimi k1.5-style group-relative length preference for correct rollouts: a rollout with reward > 0 gets
    reward * (1 - alpha * (len - min_len) / (max_len - min_len)) over its group's lengths (shorter right answers
    score higher); rewards of 0 are unchanged (MaxRL needs rewards >= 0). alpha = 0 returns the values unchanged."""
    if not alpha:
        return list(values)
    out = list(values)
    for idx in _groups(group_ids).values():
        lo = min(lengths[i] for i in idx); hi = max(lengths[i] for i in idx)
        if hi <= lo:
            continue
        for i in idx:
            if values[i] > 0:
                out[i] = values[i] * (1.0 - alpha * (lengths[i] - lo) / (hi - lo))
    return out


def overlong_penalty(values: list, lengths: list, truncated: list, outcomes: list, group_ids: list, cap: int,
                     start_frac: float, p0: float, p1: float, mixed_only: bool = True) -> tuple:
    """DAPO-style soft overlong punishment (experimental): a rollout of L tokens (of a cap of `cap`)
    gets 0 for L <= start_frac * cap, then -(p0 + (p1 - p0) * (L - L0) / (cap - L0)) up to the cap (L0 = start_frac *
    cap), and -p1 if it was truncated at the cap. Added to every rollout of a group (right or wrong) with mixed_only
    only when the group's natural outcomes are mixed (some > 0, some not): all-right / all-wrong groups keep their
    rewards (and stay zero-variance on the outcome). Returns (new values, penalised rollouts, penalised groups,
    penalty sum)."""
    if not (0 < start_frac < 1 and 0 <= p0 <= p1 and cap > 0):
        raise ValueError("overlong penalty needs 0 < start_frac < 1, 0 <= p0 <= p1 and cap > 0")
    lo = start_frac * cap
    mixed = None
    if mixed_only:
        by = {}
        for o, g in zip(outcomes, group_ids):
            r = by.setdefault(g, [0, 0])
            r[0 if o > 0 else 1] += 1
        mixed = {g for g, (ok, bad) in by.items() if ok and bad}
    out, n, groups, tot = [], 0, set(), 0.0
    for v, L, t, g in zip(values, lengths, truncated, group_ids):
        pen = 0.0
        if mixed is None or g in mixed:
            if t:
                pen = p1
            elif L > lo:
                pen = p0 + (p1 - p0) * min(1.0, (L - lo) / max(cap - lo, 1.0))
        if pen:
            n += 1
            groups.add(g)
            tot += pen
        out.append(v - pen)
    return out, n, len(groups), tot


def length_reward_absolute(values: list, lengths: list, alpha: float, budget: int) -> list:
    """Length-aware MaxRL, variant A (experimental): a rewarded rollout's reward r becomes
    r * (1 - alpha * min(L / budget, 1)); unrewarded rollouts stay 0 whatever their length. With MaxRL's
    (r - mean) / mean a lone success keeps advantage G - 1 however long it is (its scale cancels); in groups with
    several successes the shorter ones gain. alpha = 0 returns the values unchanged."""
    if not alpha:
        return list(values)
    if budget <= 0:
        raise ValueError("budget must be > 0")
    return [v * (1.0 - alpha * min(L / budget, 1.0)) if v > 0 else v for v, L in zip(values, lengths)]


def length_share_positive(adv: list, rewards: list, lengths: list, group_ids: list, lam: float,
                          budget: int) -> list:
    """DRPO-style length weighting on MaxRL (experimental): in each group the rollouts with
    reward > 0 and advantage > 0 share their summed advantage with weights w = exp((1 - L / budget) / lam), i.e.
    adv_i * w_i * sum(adv) / sum(adv * w) over those rollouts. The group's positive mass is unchanged, so correctness
    is reinforced as before and only its split among the successes moves; none turns negative. Two successes'
    shares differ by exp((L_long - L_short) / (budget * lam)), so near-equal lengths stay near-equal. Wrong
    rollouts, a lone success and equal lengths are unchanged. Only the group's top-reward rollouts share (with partial
    credit a short half-right answer could otherwise outrank a long full solve); lower partial
    rewards keep their advantage. lam = 0 returns adv unchanged."""
    if not lam:
        return list(adv)
    if lam < 0 or budget <= 0:
        raise ValueError(f"need lam > 0 and budget > 0: {lam}, {budget}")
    out = list(adv)
    for idx in _groups(group_ids).values():
        top = max(rewards[i] for i in idx)
        pos = [i for i in idx if rewards[i] > 0 and adv[i] > 0 and rewards[i] >= top - 1e-9]
        if len(pos) < 2:
            continue
        lo = min(lengths[i] for i in pos)
        if max(lengths[i] for i in pos) <= lo:
            continue
        w = {i: math.exp(-(lengths[i] - lo) / (budget * lam)) for i in pos}   # exp((1 - L/B)/lam) up to a constant
        scale = sum(adv[i] for i in pos) / sum(adv[i] * w[i] for i in pos)
        for i in pos:
            out[i] = adv[i] * w[i] * scale
    return out


def length_advantage_all_correct(adv: list, rewards: list, lengths: list, group_ids: list, beta: float, full: float,
                                 floor: int = 1024) -> tuple:
    """ConciseR-style length signal for all-correct groups (experimental): in a group where every
    rollout has the full outcome reward (>= full; MaxRL gives it zero advantage and --drop-zero-var drops it), a
    rollout gains beta * max(0, (mean L - L) / max(std L, floor)): shorter than the group's average is reinforced in
    proportion, the rest get nothing, so no correct answer is pushed down. The floor (tokens) keeps near-equal
    lengths from being stretched to a full standard deviation. Other groups are unchanged. Returns (advantages, ids
    of the groups that received a signal, which the caller must keep)."""
    if not beta:
        return list(adv), set()
    if beta < 0 or floor <= 0:
        raise ValueError(f"need beta >= 0 and floor > 0: {beta}, {floor}")
    out, hit = list(adv), set()
    for g, idx in _groups(group_ids).items():
        if len(idx) < 2 or min(rewards[i] for i in idx) < full - 1e-9:
            continue
        ls = [lengths[i] for i in idx]
        mean, sd = sum(ls) / len(ls), max(st.pstdev(ls), floor)
        for i in idx:
            a = beta * max(0.0, (mean - lengths[i]) / sd)
            if a > 0:
                out[i] = adv[i] + a
                hit.add(g)
    return out, hit


def pairwise_length_shaping(adv: list, outcomes: list, lengths: list, group_ids: list, lam0: float, min_correct: int = 4,
                            delta: float = 10000, span: float = 20000, gamma: float = 0.5, full: float = 1.0,
                            shape_allright: bool = True) -> tuple:
    """Thresholded pairwise length shaping (the 'Thresholded Pairwise Length Shaping for GRPO' idea adapted to floored
    MaxRL, applied to groups with >= min_correct fully right rollouts). Right = outcome >= full
    (natural, completed answers; forced-credited and partially right rollouts are not). For every pair of right
    rollouts q_ij = sign(L_j - L_i) * clip((|L_j - L_i| - delta) / span, 0, 1) (positive favours the shorter i);
    B_i = mean of q_ij over the other right rollouts (|B_i| <= 1, sum over the group = 0). A_i += lambda_g * B_i with
    lambda_g = min(lam0, gamma * a_plus) in a group that also has non-right rollouts (a_plus = the smallest advantage
    of its right rollouts, so no right rollout drops below (1 - gamma) * a_plus > 0) and lam0 in an all-right group
    (where the baseline advantage is 0 and length is the only signal). Non-right rollouts are unchanged and every
    group's advantage sum is unchanged; no re-standardisation. Returns (advantages, all-right groups that got a
    signal (the caller keeps them, as for length_advantage_all_correct), stats). shape_allright=False leaves all-right
    groups untouched (no shaping, not returned for keeping)."""
    if not lam0:
        return list(adv), set(), {}
    if lam0 < 0 or not 0 < gamma < 1 or span <= 0 or delta < 0 or min_correct < 2:
        raise ValueError(f"need lam0 >= 0, 0 < gamma < 1, span > 0, delta >= 0, min_correct >= 2: "
                         f"{lam0}, {gamma}, {span}, {delta}, {min_correct}")
    out, allright, groups, n_mixed, adj, gaps = list(adv), set(), 0, 0, [], []
    for g, idx in _groups(group_ids).items():
        right = [i for i in idx if outcomes[i] >= full - 1e-9]
        m = len(right)
        if m < min_correct:
            continue
        is_all = m == len(idx)
        if is_all and not shape_allright:
            continue
        B = {i: 0.0 for i in right}
        for a_ in range(m):
            for b_ in range(a_ + 1, m):
                i, j = right[a_], right[b_]
                gap = lengths[j] - lengths[i]
                q = (1 if gap > 0 else -1 if gap < 0 else 0) * min(1.0, max(0.0, (abs(gap) - delta) / span))
                B[i] += q
                B[j] -= q
        if not any(B.values()):
            continue
        lam = lam0 if is_all else min(lam0, gamma * min(adv[i] for i in right))
        if lam <= 0:
            continue
        for i in right:
            d = lam * B[i] / (m - 1)
            out[i] = adv[i] + d
            adj.append(abs(d))
        groups += 1
        n_mixed += not is_all
        gaps.append(max(lengths[i] for i in right) - min(lengths[i] for i in right))
        if is_all:
            allright.add(g)
    stats = ({"lenpair/groups": groups, "lenpair/mixed_groups": n_mixed, "lenpair/allright_groups": len(allright),
              "lenpair/adj_abs_mean": sum(adj) / len(adj), "lenpair/adj_abs_max": max(adj),
              "lenpair/right_len_range_mean": sum(gaps) / len(gaps)} if groups else
             {"lenpair/groups": 0, "lenpair/mixed_groups": 0, "lenpair/allright_groups": 0})
    return out, allright, stats


def forced_credit_rewards(rewards: list, forced: list, group_ids: list = None, credit: float = 1.0) -> tuple:
    """Budget-forced answers (--train-budget-force): a
    truncated rollout's reward is 0, but if the answer the model gives after its reasoning was force-closed at the cap
    is right, it earns credit for the group statistics. forced[i] is that outcome or None; natural rewards are never
    replaced. In a group with a natural success the forced outcome enters at credit * f, so the truncated rollout still
    ranks below the natural successes and cannot make the group look all-correct; in a group without one it enters at
    f, and scale_forced_advantages scales its advantage instead (MaxRL, and floored MaxRL whenever the floor binds as it
    does for such a group, are linear in the rewards). Returns (rewards, forced-credited mask)."""
    nat = {g for g, r in zip(group_ids, rewards) if r > 0} if group_ids is not None else set()
    out, mask = list(rewards), [False] * len(rewards)
    for i, f in enumerate(forced):
        if f is not None and f > 0 and rewards[i] <= 0:
            out[i] = credit * f if group_ids is not None and group_ids[i] in nat else f
            mask[i] = True
    return out, mask


def scale_forced_advantages(adv: list, mask: list, credit: float, group_ids: list = None, natural: list = None) -> list:
    """A forced-credited rollout in a group without a natural success keeps `credit` of the positive advantage it would
    get had it finished naturally (groups that have a natural success are left alone: forced_credit_rewards already
    scaled their rewards). With group_ids, a forced advantage is also capped at the smallest positive natural advantage
    in its group, so a finished answer always outranks a truncated one."""
    if not 0.0 <= credit <= 1.0:
        raise ValueError(f"credit must be in [0, 1]: {credit}")
    nat = ({g for g, r in zip(group_ids, natural) if r > 0}
           if group_ids is not None and natural is not None else set())
    out = [a * credit if m and a > 0 and not (group_ids is not None and group_ids[i] in nat) else a
           for i, (a, m) in enumerate(zip(adv, mask))]
    if group_ids is not None:
        for idx in _groups(group_ids).values():
            pos = [adv[i] for i in idx if not mask[i] and adv[i] > 0]
            if pos:
                cap = min(pos)
                for i in idx:
                    if mask[i] and out[i] > cap:
                        out[i] = cap
    return out


def zero_variance_groups(values: list, group_ids: list, tol: float = 1e-9) -> set:
    """Groups whose values are all equal (within tol): no learning signal (DAPO dynamic sampling drops them)."""
    return {g for g, idx in _groups(group_ids).items() if max(values[i] for i in idx) - min(values[i] for i in idx) <= tol}


# --- step metrics -----------------------------------------------------------------------------------------------

def _pct(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))] if xs else 0


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def step_metrics(recs: list) -> dict:
    """recs: per rollout {"stats", "seq", "r_seq", "dense_sum", "seen", "group", "dropped", "seq_adv"}."""
    m = {}
    if not recs:
        return m
    S = [r["stats"] for r in recs]
    n = len(recs)
    lens = [s["n_tokens"] for s in S]
    m.update({"rollout/n": n, "rollout/len_mean": _mean(lens), "rollout/len_median": st.median(lens),
              "rollout/len_p90": _pct(lens, 0.9), "rollout/len_max": max(lens), "rollout/tokens": sum(lens),
              "rollout/truncated_rate": _mean(s["finish"] == "length" for s in S),
              "rollout/closed_think_rate": _mean(s["closed"] for s in S),
              "rollout/close_rate": _mean(s["closed"] and s["finished"] for s in S),
              "rollout/well_formed_rate": _mean(s["well_formed"] for s in S),
              "rollout/shape_ok_rate": _mean(s["shape_ok"] for s in S),
              "rollout/copy_rate": _mean(s["copy"] for s in S),
              "rollout/format_ok_rate": _mean(s["format_ok"] for s in S),
              "rollout/em_rate": _mean(s["em_all"] for s in S),
              "rollout/em_frac": _mean(s["em_frac"] for s in S),
              "rollout/loop_rate": _mean(s["looped"] for s in S),
              "rollout/loop_verbatim_rate": _mean(s["loop_kind"] == "verbatim" for s in S),
              "rollout/loop_templated_only_rate": _mean(s["loop_kind"] == "templated" for s in S),
              "rollout/loop_token_share": sum(s["loop_tokens"] for s in S) / max(1, sum(lens))})
    closed_lens = [s["n_tokens"] for s in S if s["closed"] and s["finished"]]
    if closed_lens:
        m["rollout/len_closed_mean"] = _mean(closed_lens)
    onsets = [s["loop_onset_tok"] for s in S if s["loop_onset_tok"] is not None]
    if onsets:
        m["rollout/loop_onset_tok_median"] = st.median(onsets)
    for tag in ("seen", "unseen"):
        sub = [r for r in recs if r["seen"] == (tag == "seen")]
        m[f"rollout/n_{tag}"] = len(sub)
        if sub:
            m[f"rollout/em_rate_{tag}"] = _mean(r["stats"]["em_all"] for r in sub)
            m[f"rollout/close_rate_{tag}"] = _mean(r["stats"]["closed"] and r["stats"]["finished"] for r in sub)
    for k in ("outcome", "format", "loop_seq"):
        m[f"reward/{k}"] = _mean(r["seq"][k] for r in recs)
    m["reward/seq_total"] = _mean(r["r_seq"] for r in recs)
    m["reward/loop_tok_sum"] = _mean(s["loop_reward_sum"] for s in S)
    m["reward/total"] = _mean(r["r_seq"] + r["dense_sum"] for r in recs)
    # groups
    groups = _groups([r["group"] for r in recs])
    m["group/n"] = len(groups)
    m["group/dropped_frac"] = _mean(recs[idx[0]]["dropped"] for idx in groups.values())
    m["group/any_em_frac"] = _mean(any(recs[i]["stats"]["em_frac"] > 0 for i in idx) for idx in groups.values())
    m["group/outcome_var_frac"] = _mean(len({recs[i]["seq"]["outcome"] for i in idx}) > 1 for idx in groups.values())
    kept = [r for r in recs if not r["dropped"]]
    m["rollout/kept"] = len(kept)
    if kept:
        adv = [r["seq_adv"] for r in kept]
        m["adv/seq_abs_mean"] = _mean(abs(a) for a in adv)
        m["adv/seq_max"] = max(adv)
        m["adv/seq_min"] = min(adv)
    return m
