"""--eval-logit-bias: token biases for eval decoding only. vLLM ignores logit_bias under speculative decoding, so the flag
requires --mtp-tokens 0 and the worker refuses it with MTP on."""
import inspect
import json
import os

import pytest

from arcsolver.rl import train as D


def test_flag_and_guard(tmp_path):
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"logit_bias": {"13373": -10, "3655": -10.0}}))
    base = ["--out", str(tmp_path / "o"), "--student", "/tmp/none"]
    assert D.parse_args(base).eval_logit_bias == ""
    a = D.parse_args(base + ["--eval-logit-bias", str(f), "--mtp-tokens", "0"])
    assert a.eval_logit_bias_map == {"13373": -10.0, "3655": -10.0}
    with pytest.raises(SystemExit):
        D.parse_args(base + ["--eval-logit-bias", str(f), "--mtp-tokens", "3"])
    (tmp_path / "e.json").write_text("{}")
    with pytest.raises(SystemExit):
        D.load_logit_bias(str(tmp_path / "e.json"))


def test_wiring():
    src = inspect.getsource(D)
    ev = src[src.index("def make_eval("):]
    gen = ev[ev.index("    def generate("):ev.index("    def score_prompt(")]
    assert gen.count("**bias}") == 2                      # the main eval call and the forced-answer call
    fa = src[src.index("def force_answers("):src.index("def forced_stats(")]
    assert "bias" not in fa                               # training forced answers are never biased
    w = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vllm_worker.py")).read()
    sp = w[w.index("    def sampling_params("):w.index("    def stream(")]
    assert "logit_bias={int(k): float(v) for k, v in lb.items()} if lb else None" in sp
    assert "if lb and args.mtp_tokens > 0:" in sp and "raise ValueError" in sp
