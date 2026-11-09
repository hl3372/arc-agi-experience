"""The companion reading question for rows whose questions are all yes/no (build_dataset.add_companion):
triggered by question kinds only, never by answers; a cell / line / run of a plain grid; never next to a
"nothing else" wording, on a proposed or candidate grid, or on a literal item's grid."""
import copy
import dataclasses

import numpy as np

import pv2.build_dataset as B
from pv2 import families as F
from pv2.model import FLAG_NOTHING_ELSE, Answer


def _flip(part):
    """The same part with every yes/no answer flipped (and verification counts changed)."""
    p = copy.deepcopy(part)
    out = []
    for q in p.queries:
        a = q.answer
        if a.kind == "bool":
            a = Answer("bool", not a.value, "no" if a.value else "yes")
        elif q.qid in B._VERIFY_COUNTS:
            a = Answer("number", a.value + 3, str(a.value + 3))
        elif a.kind == "token" and a.text in ("yes", "no"):
            a = Answer("token", "no" if a.text == "yes" else "yes", "no" if a.text == "yes" else "yes")
        out.append(dataclasses.replace(q, answer=a))
    p.queries = out
    return p


def test_trigger_reads_question_kinds_not_answers():
    rng = np.random.default_rng(11)
    seen = 0
    for _ in range(300):
        part = F.comparison_part(rng)
        yn = B._yes_no_only([part])
        assert yn == B._yes_no_only([_flip(part)])
        seen += yn
    assert seen > 20


def test_companion_is_a_reading_question_on_a_plain_grid():
    rng = np.random.default_rng(12)
    added = 0
    for _ in range(400):
        part = F.comparison_part(rng)
        if not B._yes_no_only([part]):
            before = len(part.queries)
            assert not B.add_companion(rng, [part]) and len(part.queries) == before
            continue
        before = [q.qid for q in part.queries]
        if B.add_companion(rng, [part]):
            added += 1
            q = part.queries[-1]
            assert [x.qid for x in part.queries[:-1]] == before
            assert q.source == "lib" and q.qid[4:] in F.READING_QIDS
            assert all(part.slots[i].role == "grid" for i in q.reads)
            assert part.prov["companion"] == q.qid
            part.validate_shape()
    assert added > 20


def test_no_companion_next_to_a_nothing_else_wording_or_a_literal_item():
    rng = np.random.default_rng(13)
    for _ in range(300):
        part = F.reframed_part(rng)
        literal = part.queries[0].qid.split(":")[1] in F.LITERAL_ITEM_TASKS
        single = part.single_answer or FLAG_NOTHING_ELSE in part.queries[0].flags
        got = B.add_companion(rng, [part])
        if literal or single:
            assert not got


def test_yes_no_items_trigger_both_ways():
    from types import SimpleNamespace
    for text in ("yes", "no", "true", "false"):
        assert B._yes_no(SimpleNamespace(qid="item:56:canon", answer=Answer("token", text, text)))
    # a count that happens to be 0 or 1 never triggers (a companion would then tell its value)
    for n in (0, 1, 7):
        assert not B._yes_no(SimpleNamespace(qid="lib:count", answer=Answer("number", n, str(n))))


def test_rows_build_with_companions():
    from _tok import tokenizer_or_skip
    cfg = B.BuildConfig(classic_share=0.0)
    cfg.arc["questions"] = "reading"
    cfg.assemble["single_grid_encoding"] = "compact"
    tok = tokenizer_or_skip()
    rows = with_companion = 0
    for s in range(250):
        built = B.build_row(5_300_000 + s, cfg, tok)
        if built is None:
            continue
        rows += 1
        with_companion += any("companion" in (p.get("prov") or {}) for p in built[2]["assembled"]["parts"])
    assert rows > 200 and with_companion >= 1
