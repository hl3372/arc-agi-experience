"""Rewards on CPU: token alignment, loop regions, answers / outcome / format, group advantages (GRPO, Dr. GRPO, MaxRL),
the overlong token penalty, length scaling and step metrics."""
import math
import os

import pytest

from arcsolver.rl import core as oc
from arcsolver.rl import rewards as rr

from .helpers import StubTok, token_text


@pytest.fixture
def stub():
    return StubTok()


def _tt(stub, text):
    return token_text(stub, text)


# --- token alignment ----------------------------------------------------------------------------------------------

def test_aligner_spans_tile_the_text_and_skip_special_tokens(stub):
    text = "pair 1 was é-ok\n\nrow ✓</think>\n\n12\n34<|im_end|>"
    ids = stub.encode(text)
    tt = rr.TokenAligner(stub)(ids)
    assert tt.text == text.replace("<|im_end|>", "")               # special token -> "" (vLLM's text)
    assert len(tt) == len(ids)
    assert tt.starts[0] == 0 and tt.ends[-1] == len(tt.text)
    assert all(tt.ends[j] == tt.starts[j + 1] for j in range(len(ids) - 1))   # spans tile the text
    j_think = ids.index(StubTok.THINK)
    assert tt.text[tt.starts[j_think]:tt.ends[j_think]] == "</think>"
    assert tt.starts[-1] == tt.ends[-1]                              # <|im_end|>: empty span
    # a 3-byte char split over 3 tokens: the first two get empty spans, the completing one gets the char
    k = tt.text.index("✓")
    j = tt.token_at(k)
    assert tt.text[tt.starts[j]:tt.ends[j]] == "✓" and tt.starts[j - 1] == tt.ends[j - 1]
    assert tt.prefix_text(3) == "pair 1"[:tt.ends[2]]


def test_token_range_and_token_at(stub):
    tt = _tt(stub, "Check cell 3.\n\npair 2 was 45")
    lo, hi = tt.token_range(tt.text.index("pair"), len(tt.text))
    assert "".join(tt.text[tt.starts[j]:tt.ends[j]] for j in range(lo, hi)) == "pair 2 was 45"
    c = tt.text.index("cell") + 2                                     # inside the " cell" word token
    j = tt.token_at(c)
    assert tt.text[tt.starts[j]:tt.ends[j]] == " cell"
    assert tt.token_range(5, 5) == (0, 0)
    with pytest.raises(IndexError):
        tt.token_at(len(tt.text))


def _tokenizer_dir():
    d = os.environ.get("ARC_RL_TEST_TOKENIZER")                       # a Qwen3.5 tokenizer dir, optional
    return d if d and os.path.exists(os.path.join(d, "tokenizer.json")) else None


def _has_transformers():
    import importlib.util
    return importlib.util.find_spec("transformers") is not None


@pytest.mark.skipif(_tokenizer_dir() is None or not _has_transformers(),
                    reason="set ARC_RL_TEST_TOKENIZER to a Qwen3.5 tokenizer dir (needs transformers)")
def test_aligner_matches_the_real_tokenizer_decode():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_tokenizer_dir())
    al = rr.TokenAligner(tok)
    texts = ["I compared pair 1: r3 = 0012300, so colour 4 at r2 c5.\n\nThe rule is a flip.\n</think>\n\n12\n34",
             "Unicode: café × ✓ → 世界 and emoji 🙂 end.\n\n\n\nsplit</think>\n\n0",
             "<think>\nnested tags </think> twice </think>\n\n7"]
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
        tt = al(ids)
        assert tt.text == tok.decode(ids, skip_special_tokens=True)
        assert tt.text == text
        assert all(tt.ends[j] == tt.starts[j + 1] for j in range(len(ids) - 1))
        for j, i in enumerate(ids):                                   # ASCII tokens: span text == the token's decode
            piece = tok.decode([i], skip_special_tokens=True)
            if piece.isascii():
                assert tt.text[tt.starts[j]:tt.ends[j]] == piece


# --- loops ----------------------------------------------------------------------------------------------------------

LOOP_TEXTS = [
    "Look at row 1.\n\nCompare pairs.\n\nDone.",
    "Look at row 1.\n\nCompare pairs.\n\n" + "\n\n".join(["Check cell 3."] * 6),
    "Look at row 1.\n\nCompare pairs.\n\n" + "\n\n".join(f"Check cell {i}." for i in range(8)),
    "Intro.\n\n" + "\n\n".join(["A", "B"] * 5) + "\n\n</think>\n\n12",
    "Intro.\n\n\n\n  Check cell 3.  \n\n\n" + "\n\n".join(["Check cell 3."] * 5),
]


@pytest.mark.parametrize("text", LOOP_TEXTS)
def test_loop_kind_matches_core_and_regions_nest(text):
    v, t = rr.loop_info(text, "verbatim"), rr.loop_info(text, "templated")
    assert v["kind"] == t["kind"] == oc.loop_kind(text)
    if v["onset"] is not None:                                        # templated region contains the verbatim one
        assert t["onset"] is not None and t["c0"] <= v["c0"] and t["c1"] == v["c1"]
    if t["onset"] is not None:
        think = rr.reasoning_text(text)
        assert t["c1"] == len(think)
        paras = [p.strip() for p in think.split("\n\n") if p.strip()]
        assert think[t["c0"]:].startswith(paras[t["onset"]])


def test_loop_onset_rules():
    text = "Intro.\n\nCheck cell 3.\n\nNew idea.\n\nCheck cell 3.\n\nMore.\n\nEven more.\n\n" + "\n\n".join(["Loop."] * 5)
    sus = rr.loop_info(text, "verbatim", "sustained")
    first = rr.loop_info(text, "verbatim", "first_repeat")
    assert text[sus["c0"]:].startswith("Loop.\n\nLoop.") and sus["onset"] == 7    # the 2nd "Loop." paragraph
    assert first["onset"] == 3 and text[first["c0"]:].startswith("Check cell 3.\n\nMore.")
    assert rr.loop_info("A.\n\nB.\n\nC.", "verbatim", "first_repeat")["onset"] is None
    with pytest.raises(ValueError):
        rr.loop_info(text, "verbatim", "other")


def test_loop_region_tokens_get_the_penalty(stub):
    body = "Look at row 1.\n\nCompare pairs.\n\n" + "\n\n".join(f"Check cell {i}." for i in range(8))
    tt = _tt(stub, body)                                             # truncated (never closed), templated loop
    cfg = rr.RewardConfig(beta_loop=0.3, w_loop_seq=0.1)
    sc = rr.score_rollout({"train": [], "test": [{"input": [[0]], "output": [[1]]}]}, tt, "length", cfg)
    lo, hi = sc["loop_range"]
    c0 = body.index("Check cell 0")
    info = rr.loop_info(body)                                        # the first paragraph that repeats a template
    assert info["kind"] == "templated" and body[info["c0"]:].startswith("Check cell 1.")
    assert c0 < info["c0"]
    assert "".join(tt.text[tt.starts[j]:tt.ends[j]] for j in range(lo, hi)) == body[info["c0"]:]
    assert sc["dense"] == {j: -0.3 for j in range(lo, hi)}
    assert sc["seq"]["loop_seq"] == -0.1 and sc["stats"]["loop_tokens"] == hi - lo
    verb = rr.score_rollout({"train": [], "test": [{"input": [[0]]}]}, tt, "length",
                            rr.RewardConfig(loop_detect="verbatim"))
    assert verb["dense"] == {} and verb["stats"]["looped"] is False  # digit-masked detection is what finds it


def test_loop_region_ends_at_think_close(stub):
    text = "Intro.\n\n" + "\n\n".join(["A", "B"] * 5) + "\n\n</think>\n\n12<|im_end|>"
    tt = _tt(stub, text)
    sc = rr.score_rollout({"train": [], "test": [{"input": [[0]], "output": [[1, 2]]}]}, tt, "stop", rr.RewardConfig())
    lo, hi = sc["loop_range"]
    region = "".join(tt.text[tt.starts[j]:tt.ends[j]] for j in range(lo, hi))
    assert region.startswith("A") and region.rstrip().endswith("B") and "</think>" not in region
    assert sc["seq"]["outcome"] == 1.0                               # looped, then answered right: still rewarded


# --- answers, outcome and format ------------------------------------------------------------------------------------

TASK2 = {"train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
         "test": [{"input": [[1, 1], [0, 0]], "output": [[0, 0], [1, 1]]},
                  {"input": [[5, 6, 7]], "output": [[7], [6], [5]]}]}


def test_answer_eval_exact_match_per_test_output():
    a = rr.answer_eval(TASK2, "think</think>\n\n00\n11\n\n7\n6\n5", "stop")
    assert a["em"] == [True, True] and a["em_frac"] == 1.0 and a["em_all"] and a["shape_ok"] and a["well_formed"]
    b = rr.answer_eval(TASK2, "think</think>\n\n00\n11\n\n765", "stop")
    assert b["em"] == [True, False] and b["em_frac"] == 0.5 and not b["em_all"] and b["well_formed"] and not b["shape_ok"]
    c = rr.answer_eval(TASK2, "x</think>\nAttempt 1:\n11\n00\n\n7\n6\n5\n\nAttempt 2:\n00\n11\n\n7\n6\n5", "stop")
    assert c["em"] == [False, True]                                  # attempt 1 only
    assert c["copy"] is True                                         # attempt 1 copies test input 1 verbatim
    d = rr.answer_eval(TASK2, "x</think>\n\n22\n00\n\n7\n6\n5", "stop")
    assert d["copy"] is False and d["em"] == [False, True] and d["shape_ok"]
    e = rr.answer_eval(TASK2, "no close\n\n00\n11", "length")
    assert not e["closed"] and e["em"] == [False, False] and not e["well_formed"]
    f = rr.answer_eval(TASK2, "x</think>\n\n00\n1\n\n7\n6\n5", "stop")   # ragged first grid
    assert f["em"] == [False, True] and not f["well_formed"]


def test_outcome_requires_a_clean_stop_and_format_guards(stub):
    base = rr.RewardConfig(w_outcome=1.0, w_format=0.2, beta_loop=0.0)
    right = _tt(stub, "Rule found.</think>\n\n00\n11\n\n7\n6\n5<|im_end|>")
    sc = rr.score_rollout(TASK2, right, "stop", base)
    assert sc["seq"] == {"outcome": 1.0, "format": 0.2, "loop_seq": -0.0} and sc["r_seq"] == pytest.approx(1.2)
    trunc = rr.score_rollout(TASK2, right, "length", base)            # hit the cap: no outcome, no format
    assert trunc["seq"]["outcome"] == 0.0 and trunc["seq"]["format"] == 0.0
    lenient = rr.RewardConfig(w_outcome=1.0, outcome_require_stop=False, beta_loop=0.0)
    assert rr.score_rollout(TASK2, right, "length", lenient)["seq"]["outcome"] == 1.0
    wrong_shape = _tt(stub, "x</think>\n\n00\n11\n\n765<|im_end|>")
    assert rr.score_rollout(TASK2, wrong_shape, "stop", base)["seq"]["format"] == 0.0
    no_shape = rr.RewardConfig(w_format=0.2, format_shape="none", beta_loop=0.0)
    assert rr.score_rollout(TASK2, wrong_shape, "stop", no_shape)["seq"]["format"] == 0.2
    copied = _tt(stub, "x</think>\n\n11\n00\n\n7\n6\n5<|im_end|>")
    assert rr.score_rollout(TASK2, copied, "stop", base)["seq"]["format"] == 0.0
    allow = rr.RewardConfig(w_format=0.2, format_reject_copy=False, beta_loop=0.0)
    assert rr.score_rollout(TASK2, copied, "stop", allow)["seq"]["format"] == 0.2
    one_grid = _tt(stub, "x</think>\n\n00\n11<|im_end|>")            # missing the second output
    s1 = rr.score_rollout(TASK2, one_grid, "stop", base)
    assert s1["seq"]["format"] == 0.0 and s1["seq"]["outcome"] == 0.5
    all_t = rr.RewardConfig(outcome_all_tests=True, beta_loop=0.0)
    assert rr.score_rollout(TASK2, one_grid, "stop", all_t)["seq"]["outcome"] == 0.0
    assert rr.score_rollout(TASK2, right, "stop", all_t)["seq"]["outcome"] == 1.0


def test_binary_outcome_config_has_no_dense_rewards(stub):
    """The released recipes' reward: w_outcome 1, every shaping weight 0 -> no token rewards at all."""
    cfg = rr.RewardConfig(beta_loop=0.0, w_loop_seq=0.0, w_format=0.0, w_len=0.0)
    looped = _tt(stub, "Intro.\n\n" + "\n\n".join(["A"] * 6))
    sc = rr.score_rollout(TASK2, looped, "length", cfg)
    assert sc["dense"] == {} and sc["dense_sum"] == 0.0 and sc["r_seq"] == 0.0 and sc["stats"]["looped"]


# --- groups ---------------------------------------------------------------------------------------------------------

def test_group_normalisation_std_and_mean():
    vals = [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 2.0, 0.0]
    g = [0, 0, 0, 0, 1, 1, 2, 2]
    a = rr.group_advantages(vals, g, "std", eps=0.0)
    s0 = math.sqrt(sum((v - 0.25) ** 2 for v in vals[:4]) / 3)       # sample std (ddof 1)
    assert a[:4] == pytest.approx([(v - 0.25) / s0 for v in vals[:4]])
    assert a[4:6] == [0.0, 0.0]                                      # all equal: 0 (no division by zero with eps>0 too)
    assert rr.group_advantages(vals, g, "std")[4:6] == [0.0, 0.0]
    assert a[6:] == pytest.approx([1 / math.sqrt(2), -1 / math.sqrt(2)])
    m = rr.group_advantages(vals, g, "mean")
    assert m == pytest.approx([0.75, -0.25, -0.25, -0.25, 0.0, 0.0, 1.0, -1.0])
    assert sum(a[:4]) == pytest.approx(0.0) and sum(m) == pytest.approx(0.0)
    assert rr.group_advantages([3.0], [0]) == [0.0]
    assert rr.zero_variance_groups(vals, g) == {1}
    assert rr.zero_variance_groups([0.0, 1e-12, 5.0, 5.0], [0, 0, 1, 1]) == {0, 1}


def test_grpo_std_advantages_of_a_binary_group_of_six():
    """The 9B recipe: group of 6, binary reward, sample std (ddof 1) + 1e-4."""
    vals = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = rr.group_advantages(vals, [0] * 6, "std", eps=1e-4)
    m, sd = 1 / 6, math.sqrt((25 / 36 + 5 / 36) / 5)
    assert a[0] == pytest.approx((1 - m) / (sd + 1e-4)) and a[1] == pytest.approx(-m / (sd + 1e-4))
    assert rr.zero_variance_groups([0.0] * 6 + [1.0] * 6, [0] * 6 + [1] * 6) == {0, 1}   # 0/6 and 6/6 dropped


def test_step_metrics_rates(stub):
    recs = []
    texts = [("x</think>\n\n0<|im_end|>", "stop"), ("x</think>\n\n1<|im_end|>", "stop"),
             ("Intro.\n\n" + "\n\n".join(["A"] * 6), "length"), ("Some reasoning.\n\nMore.", "length")]
    task = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[5]], "output": [[0]]}]}
    for k, (t, f) in enumerate(texts):
        sc = rr.score_rollout(task, _tt(stub, t), f, rr.RewardConfig())
        recs.append({**sc, "seen": k < 2, "group": k // 2, "dropped": k // 2 == 1, "seq_adv": 0.5})
    m = rr.step_metrics(recs)
    assert m["rollout/em_rate"] == 0.25 and m["rollout/close_rate"] == 0.5 and m["rollout/loop_rate"] == 0.25
    assert m["rollout/em_rate_seen"] == 0.5 and m["rollout/em_rate_unseen"] == 0.0
    assert m["group/dropped_frac"] == 0.5 and m["group/outcome_var_frac"] == 0.5 and m["rollout/kept"] == 2
    assert m["rollout/truncated_rate"] == 0.5 and not any(k.startswith("claims/") for k in m)


def test_maxrl_advantages():
    """MaxRL (arXiv 2602.02710): (r - mean) / mean per group; 1 success of 8 -> +7 / -1; all-wrong -> 0; 4 of 8 -> +1/-1;
    unlike GRPO the weight grows as the success rate falls."""
    g = [0] * 8 + [1] * 8 + [2] * 8
    vals = [1.0] + [0.0] * 7 + [0.0] * 8 + [1.0] * 4 + [0.0] * 4
    a = rr.group_advantages(vals, g, "maxrl", eps=0.0)
    assert math.isclose(a[0], 7.0) and all(math.isclose(x, -1.0) for x in a[1:8])
    assert all(x == 0.0 for x in a[8:16])
    assert all(math.isclose(x, 1.0) for x in a[16:20]) and all(math.isclose(x, -1.0) for x in a[20:24])
    for grp in (range(0, 8), range(16, 24)):
        assert abs(sum(a[i] for i in grp)) < 1e-9                    # zero-mean within a group
    s = rr.group_advantages(vals, g, "std", eps=0.0)
    assert a[0] > s[0] > 0                                           # rare successes weigh more than under GRPO
    with pytest.raises(ValueError):
        rr.group_advantages(vals, g, "bogus")


def test_overlong_penalty_and_length_scaling():
    """Dense overlong penalty (ramps from len_soft to len_cap, per token, independent of the outcome) and the
    Kimi-style group-relative length scaling of correct rollouts."""
    cfg = rr.RewardConfig(len_soft=4, len_cap=8, w_len=1.0, beta_loop=0.0)

    class TT:  # minimal TokenText stand-in: 10 tokens of plain text, no answer
        text = "thinking " * 10

        def __len__(self):
            return 10

        def token_at(self, c):
            return 0

        def token_range(self, a, b):
            return (0, 0)
    task = {"train": [], "test": [{"input": [[1]], "output": [[2]]}]}
    out = rr.score_rollout(task, TT(), "length", cfg)
    d = out["dense"]
    assert set(d) == set(range(4, 10))
    assert math.isclose(d[4], -0.25) and math.isclose(d[7], -1.0) and math.isclose(d[9], -1.0)
    assert math.isclose(out["stats"]["len_reward_sum"], sum(d.values()))
    off = rr.score_rollout(task, TT(), "length", rr.RewardConfig(beta_loop=0.0))
    assert off["dense"] == {}
    vals = [1.0, 1.0, 0.0, 0.0]
    lens = [100, 300, 200, 50]
    g = [0, 0, 0, 0]
    sv = rr.length_scaled_rewards(vals, lens, g, 0.2)
    assert math.isclose(sv[0], 1.0 * (1 - 0.2 * 50 / 250)) and math.isclose(sv[1], 0.8) and sv[2:] == [0.0, 0.0]
    assert rr.length_scaled_rewards(vals, lens, g, 0.0) == vals
