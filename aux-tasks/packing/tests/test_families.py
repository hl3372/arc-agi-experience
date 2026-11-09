"""Family builders: shape-valid parts, unasked grids in galleries, no literal item next to a library
question on its grid, the 'as shown' context whenever an edit shares a grid."""
from collections import Counter

import numpy as np

from pv2 import families as F
from pv2.library.core import AS_SHOWN
from pv2.model import FLAG_EDIT


def test_reframed_parts():
    rng = np.random.default_rng(0)
    for _ in range(60):
        p = F.reframed_part(rng)
        p.validate_shape()
        assert len(p.slots) == 1 and len(p.queries) == 1 and p.queries[0].lead_clause


def test_galleries_have_unasked_grids_and_mixed_sources():
    rng = np.random.default_rng(1)
    unasked, sources = 0, Counter()
    for _ in range(80):
        n = int(rng.integers(2, 9))
        k = int(rng.integers(1, n + 1))
        p = F.gallery_part(rng, n_grids=n, n_asked=k)
        p.validate_shape()
        read = {i for q in p.queries for i in q.reads}
        unasked += len(p.slots) - len(read)
        sources.update(q.source for q in p.queries)
        assert all(len(q.reads) == 1 for q in p.queries)
    assert unasked > 0 and sources["item"] and sources["lib"]


def test_one_grid_parts_never_put_a_library_question_next_to_a_literal_item():
    rng = np.random.default_rng(2)
    mixed = 0
    for _ in range(150):
        p = F.one_grid_part(rng, n_questions=int(rng.integers(2, 6)))
        p.validate_shape()
        items = [q for q in p.queries if q.source == "item"]
        if items:
            tid = items[0].qid.split(":")[1]
            assert tid not in F.LITERAL_ITEM_TASKS
            for q in p.queries:
                if q.source == "lib":
                    assert q.qid.split(":")[-1] in F.READING_QIDS
            mixed += len(p.queries) > 1
        if any(FLAG_EDIT in q.flags for q in p.queries) and len(p.queries) > 1:
            assert AS_SHOWN in p.context
    assert mixed > 10


def test_runs_encoded_grids_are_introduced_by_a_note():
    from pv2 import library as L
    from pv2.assemble import assemble_row
    from pv2.assemble.row import ENCODING_NOTES
    from pv2.model import Part, Slot
    rng = np.random.default_rng(4)
    for _ in range(40):
        g = rng.integers(0, 4, size=(int(rng.integers(3, 9)), int(rng.integers(3, 9)))).tolist()
        slots = [Slot(g, "runs")]
        q = L.pose(rng, slots, (0,), qid="n_colours")
        row = assemble_row(rng, [Part("multi_question", slots, [q])])
        first_run_line = row.prompt.index("r0: ")
        before = row.prompt[:first_run_line]
        assert any(n.split("{how}")[0] in before for n in ENCODING_NOTES)


def test_a_raising_generator_seed_costs_only_that_draw(monkeypatch):
    """The reviewed generators raise (never fall back) on a rare unbuildable seed; v2's own seed
    draws skip it."""
    import pv2.items as I
    import pv2.pictures as P
    real = P.generate
    calls = {"n": 0}

    def flaky(tid, seed):
        calls["n"] += 1
        if calls["n"] % 3 == 1:
            raise RuntimeError("cannot build the fixed plan")
        return real(tid, seed)
    monkeypatch.setattr(P, "generate", flaky)
    rng = np.random.default_rng(6)
    for _ in range(10):
        assert P.sample_picture(rng) is not None
        assert I.item_query(rng) is not None


def test_arc_options_default_to_todays_parts_and_modes_are_respected():
    from pv2.arc import bundle
    a = F.arc_part(np.random.default_rng(9))
    b = bundle.arc_part(np.random.default_rng(9))
    assert [q.text for q in a.queries] == [q.text for q in b.queries] and a.prov == b.prov
    rng = np.random.default_rng(10)
    for _ in range(12):
        p = F.arc_part(rng, modes=["stated"])
        assert p.prov.get("mode") in (None, "stated") or "stated" in str(p.prov)
    import pytest
    with pytest.raises(ValueError):
        F.arc_part(rng, modes=["nonsense"])


def test_single_grid_answer_encoding_option():
    from pv2.build_dataset import BuildConfig, build_row
    from _tok import tokenizer_or_skip
    tok = tokenizer_or_skip()
    cfg = BuildConfig(classic_share=0.0)
    cfg.assemble["single_grid_encoding"] = "compact"
    styles, grid_parts = set(), 0
    for s in range(80):
        b = build_row(6_600_000 + s, cfg, tok)
        if not b:
            continue
        for p in b[2]["assembled"]["parts"]:
            styles.add(p["style"])
            if p["grid_answer_encoding"] is not None:
                grid_parts += 1
                assert p["grid_answer_encoding"] == "compact" and p["style"] != "json"
    assert grid_parts > 20 and {"sentence", "labelled"} <= styles
