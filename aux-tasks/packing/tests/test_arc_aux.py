"""Aux tasks as ARC rules: rule keys, colour alignment, allowed wordings, library equivalents."""
import numpy as np
import pytest

import prompt_variants
from pv2.arc import aux_rules as A
from pv2.arc.grids import Info
from registry import generate


def test_palette_symbols_never_occur_in_prompts():
    for tid in A.STATED:
        for s in range(3):
            meta = generate(tid, s)["metadata"]
            for template in A.wordings(tid):
                text = A._render(tid, meta, template)
                assert not set(A.SYM) & set(text), (tid, template)


def test_wording_filter():
    flagged = {"11_v005", "44_v006", "45_v005", "66_v006", "80_v005"}    # DESIGN 13.3 placement wordings
    store = prompt_variants.load()["tasks"]
    for vid in flagged:
        tid = vid.split("_")[0]
        v = next(v for v in store[tid] if v["id"] == vid)
        text = A._render(tid, generate(tid, 1)["metadata"], v["template"])
        assert not A.wording_ok(text), vid
    assert not A.wording_ok("Output the smallest rectangle containing the object.")
    assert not A.wording_ok("The input shows two subgrids.")
    assert not A.wording_ok("Answer with the number.")
    assert A.wording_ok("Two objects match when one becomes the other after a rotation (mirror images do not count).")
    assert A.wording_ok("Translate each loose piece, arriving from above or below.")


def test_rule_key_normalises_colours():
    m1 = dict(generate("29", 0)["metadata"])
    m2 = dict(m1, source_color=(m1["source_color"] + 3) % 10, target_color=(m1["target_color"] + 5) % 10)
    if m2["source_color"] == m2["target_color"]:
        m2["target_color"] = (m2["target_color"] + 1) % 10
    k1, o1 = A.rule_key("29", m1)
    k2, o2 = A.rule_key("29", m2)
    assert k1 == k2 and o1 != o2
    assert o1 == (m1["source_color"], m1["target_color"])


@pytest.mark.parametrize("tid", ["10", "29", "70", "12", "31", "88"])
def test_collect_aligns_examples_to_one_rule(tid):
    rng = np.random.default_rng(int(tid[:2]))
    exs = None
    for _ in range(6):
        exs = A.collect(rng, tid, 4, max_cells=400, budget=A.budget_for(tid))
        if exs:
            break
    assert exs, f"task {tid}: no collection"
    canon = A.render_under(tid, exs[0].meta, exs[0].sigma)
    for e in exs:
        assert sorted(e.sigma) == list(range(10))
        assert A.render_under(tid, e.meta, e.sigma) == canon
        # the aligned example is exactly the regenerated example under sigma
        ex = generate(tid, e.seed)
        lut = np.array(e.sigma, dtype=np.int8)
        a, b = e.mapped()
        assert np.array_equal(a, lut[np.asarray(ex["input"], dtype=np.int8)])
        assert np.array_equal(b, lut[np.asarray(ex["output"], dtype=np.int8)])
    assert exs[0].sigma == list(range(10))


@pytest.mark.parametrize("tid", A.HIDDEN)
def test_library_rules_reproduce_the_generator(tid):
    """Every hidden-mode task's library rule gives the frozen generator's output on collected
    examples (colours aligned); collections where it does not are never used in hidden mode."""
    rng = np.random.default_rng(7)
    hits = tried = 0
    for _ in range(8):
        exs = A.collect(rng, tid, 3, max_cells=400, budget=A.budget_for(tid))
        if not exs:
            continue
        tried += 1
        rule = A.library_rule(tid, exs)
        if rule is None:
            continue
        hits += 1
        for e in exs:
            a, b = e.mapped()
            out = rule.apply_info(Info(a))
            assert out is not None and np.array_equal(out, b)
    assert tried and hits >= tried * 0.75, (tid, hits, tried)


def test_pools():
    assert set(A.HIDDEN) <= set(A.LIBRARY)
    assert not set(A.STATED_OK) & A.NO_WORDING
    assert len(A.STATED_OK) >= 30 and len(A.HIDDEN) >= 15
    for tid in A.STATED:
        out = generate(tid, 0)["output"]
        assert isinstance(out, list) and isinstance(out[0], list), tid


def test_stated_rules_are_never_questions():
    """Some stored wordings are questions ("What does this grid look like after ...?"); the stated
    frame presents the rule as an instruction, so they are skipped."""
    asked = 0
    for tid in ("10", "11", "66", "29", "70"):
        rng = np.random.default_rng(3)
        for _ in range(12):
            exs = A.collect(rng, tid, 3, max_cells=400, budget=A.budget_for(tid))
            if not exs:
                continue
            got = A.stated_text(rng, tid, exs)
            assert got is not None and "?" not in got[1]
        asked += sum("?" in A._render(tid, generate(tid, 1)["metadata"], w) for w in A.wordings(tid))
    assert asked > 0          # the store does hold question wordings for these tasks
