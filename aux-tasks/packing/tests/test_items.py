"""Registry items as v2 queries: property tests over thousands of random items."""
import time
from collections import Counter

import numpy as np
import pytest

from pv2 import encodings as enc
from pv2 import items as I
from pv2 import model
from pv2 import wordings as W
import prompt_variants as pv
from registry import generate

LABEL_CHOICES = ("image", "panel", "map", "figure", "#", "picture", "grid_number", "L", "X")


def _random_needs(rng) -> I.Needs:
    """A random mix of part constraints (most draws leave most fields at their defaults)."""
    def coin(p):
        return bool(rng.random() < p)
    kinds = None
    if coin(0.15):
        kinds = frozenset(rng.choice(["grid", "number", "color", "token", "text"],
                                     size=int(rng.integers(1, 4)), replace=False).tolist())
    encs = None
    if coin(0.15):
        encs = frozenset(rng.choice(sorted(enc.ENCODINGS), size=int(rng.integers(1, 4)), replace=False).tolist())
    labels = frozenset(w for w in LABEL_CHOICES if coin(0.2))
    return I.Needs(role_bearing=coin(0.3), multi_answer=coin(0.5), question_after_grid=coin(0.3),
                   multi_grid=coin(0.4), forbidden_label_words=labels, ordinal_refs=coin(0.3),
                   forbidden_style_labels=frozenset(w for w in ("answer", "question", "task") if coin(0.2)),
                   block_capable=not coin(0.2), short_only=coin(0.05), answer_kinds=kinds,
                   allow_stage2=not coin(0.2), encodings=encs)


def _check_pick(pick, needs):
    slot, query = pick
    prov = pick.prov
    tid, seed = prov["task_id"], prov["seed"]
    ex = generate(tid, seed)
    meta = ex.get("metadata", {})
    # the grid is the generator's input, and it round-trips through its encoding
    assert slot.grid == ex["input"]
    text = enc.render(slot.grid, slot.encoding)
    assert enc.parse(text, slot.encoding) == slot.grid
    # the answer is the generator's answer, verbatim for values
    a = query.answer
    if W.answer_kind(tid) == "grid":
        assert a.kind == "grid" and a.value == ex["output"] and a.text is None
        assert model.FLAG_EDIT in query.flags
    else:
        assert a.text == str(ex["output"]) and a.kind == W.answer_kind(tid)
    # the wording, its neutral text and its flags
    w = W.wordings(tid)[prov["wording_index"]]
    assert w.wid == prov["wording"]
    original = w.render(meta)
    assert query.text == W.neutral_text(tid, original)
    assert query.flags == W.text_flags(original, query.text) | W.task_flags(tid)
    assert not (query.flags & needs.forbidden()), (query.qid, query.flags & needs.forbidden())
    assert not W.REPLY_SHAPE.search(query.text) and not pv._ANSWER_SENTENCE.search(query.text)
    assert "⟪" not in query.text and query.text[:1].isupper()
    assert query.lead_clause and query.reads == (0,) and query.source == "item" and query.statement is None
    # role / placement / multi-answer / labels, read off the text itself
    if needs.role_bearing:
        assert not W._IO_NOUN.search(original)
    if needs.multi_grid or needs.question_after_grid:
        assert not W._PLACEMENT.search(original)
    if needs.multi_answer:
        assert not W._NOTHING_ELSE.search(query.text)     # the text v2 shows (the neutral text)
    for word in needs.forbidden_label_words:
        if word in W.LABEL_WORDS:
            assert not W.LABEL_WORDS[word].search(original)
        else:
            assert word not in W._LABEL_LETTER.findall(original)
    if not needs.block_capable:
        assert tid not in W.STRUCTURED_TASKS
    if needs.answer_kinds is not None:
        assert a.kind in needs.answer_kinds
    # encoding rules
    e = enc.ENCODINGS[slot.encoding]
    assert not e.uses_names and not e.reveals_background
    if tid in W.COORD_TASKS and e.reveals_coords:
        assert slot.encoding == "stage2" and needs.allow_stage2
    if tid in W.DIMS_TASKS:
        assert not e.reveals_dims and not e.reveals_coords and slot.dims_header_ok is False
    else:
        assert slot.dims_header_ok is True
    if needs.encodings is not None:
        assert slot.encoding in needs.encodings
    assert len(slot.grid) * len(slot.grid[0]) <= needs.max_cells
    # provenance re-creates the item exactly
    again = I.item_from_prov(prov)
    assert again[0] == slot and again[1] == query and again.prov == prov


def test_property_2000_random_items():
    rng = np.random.default_rng(20260929)
    checked, none, tasks, wordings_seen, encodings_seen = 0, 0, Counter(), set(), Counter()
    while checked < 2000:
        needs = _random_needs(rng) if rng.random() < 0.7 else I.Needs()
        pick = I.item_query(rng, needs=needs)
        if pick is None:
            none += 1
            assert none < 200
            continue
        _check_pick(pick, needs)
        checked += 1
        tasks[pick.prov["task_id"]] += 1
        wordings_seen.add(pick.prov["wording"])
        encodings_seen[pick[0].encoding] += 1
    assert len(tasks) == len(W.TASKS)                   # every task shows up
    assert len(wordings_seen) > 450                     # most of the 704 wordings show up
    assert encodings_seen["compact"] > encodings_seen["spaced"] > 0 and encodings_seen["stage2"] > 0
    assert "names" not in encodings_seen and "sparse" not in encodings_seen


def test_every_task_serves_default_needs():
    rng = np.random.default_rng(1)
    for tid in W.TASKS:
        pick = I.item_query(rng, tid)
        assert pick is not None, tid
        _check_pick(pick, I.Needs())


def test_pick_pickles():
    import pickle
    pick = I.item_query(np.random.default_rng(2))
    again = pickle.loads(pickle.dumps(pick))
    assert again == pick and again.prov == pick.prov and isinstance(again, I.ItemPick)


def test_determinism():
    def run(seed):
        rng = np.random.default_rng(seed)
        out = []
        for k in range(60):
            needs = _random_needs(rng) if k % 2 else I.Needs()
            pick = I.item_query(rng, needs=needs)
            out.append(None if pick is None else (pick.prov, pick[1].text, pick[0].encoding, pick[1].answer))
        return out
    assert run(7) == run(7)
    assert run(7) != run(8)


def test_role_bearing_multi_answer_and_placement():
    rng = np.random.default_rng(3)
    arc = I.Needs(role_bearing=True)
    assert set(I.task_pool(arc)) == set(W.TASKS) - {"40", "71", "83", "93", "95", "96", "98", "101"}
    for _ in range(300):
        slot, query = I.item_query(rng, "42", needs=arc)
        assert model.FLAG_IO_NOUNS not in query.flags
    assert I.item_query(rng, "102", needs=I.Needs(multi_answer=True)) is not None   # its demand is not shown
    assert I.item_query(rng, "102", needs=I.Needs(multi_answer=True, block_capable=False)) is None
    assert I.item_query(rng, "40", needs=arc) is None
    for _ in range(100):
        _, query = I.item_query(rng, "77", needs=I.Needs(multi_answer=True))
        assert "nothing else" not in query.text          # 77_v006 may come, its demand removed
        _, query = I.item_query(rng, "80", needs=I.Needs(multi_grid=True))
        assert query.qid != "item:80:v005"
    part = I.item_part(rng, task_id="102")     # no longer single-answer: its demand is not shown
    assert not part.single_answer and part.family == "reframed" and len(part.slots) == len(part.queries) == 1
    assert "structured" in part.queries[0].flags   # its label-line answer still limits the styles
    part = I.item_part(rng, task_id="30")
    assert not part.single_answer
    assert I.item_query(rng, "9", needs=I.Needs(forbidden_label_words=frozenset({"image"}))) is None


def test_encoding_rules_by_task():
    rng = np.random.default_rng(5)
    seen = {t: Counter() for t in ("1", "77", "2", "30")}
    for tid in seen:
        for _ in range(250):
            seen[tid][I.item_query(rng, tid)[0].encoding] += 1
    coord_revealing = {n for n, e in enc.ENCODINGS.items() if e.reveals_coords}
    assert not set(seen["1"]) & coord_revealing
    assert set(seen["77"]) & coord_revealing == {"stage2"}
    assert not set(seen["2"]) & coord_revealing          # its runs listing would print the answer
    assert {"runs", "row_labelled"} & set(seen["30"])       # other tasks may use row labels
    no_stage2 = I.Needs(allow_stage2=False)
    for _ in range(100):
        assert I.item_query(rng, "77", needs=no_stage2)[0].encoding not in coord_revealing
    assert I.item_query(rng, "1", needs=I.Needs(encodings=frozenset({"stage2"}))) is None
    for _ in range(30):
        assert I.item_query(rng, "30", needs=I.Needs(encodings=frozenset({"stage2"})))[0].encoding == "stage2"


def test_wording_choice_is_uniform_among_eligible():
    rng = np.random.default_rng(11)
    counts = Counter(I.item_query(rng, "1")[1].qid for _ in range(1400))
    assert len(counts) == len(W.wordings("1")) == 7
    assert min(counts.values()) > 140 and max(counts.values()) < 260     # 200 expected each
    # with a restriction, uniform among the rest: 11_v005 (placement) is never drawn
    counts = Counter(I.item_query(rng, "11", needs=I.Needs(multi_grid=True))[1].qid for _ in range(700))
    assert "item:11:v005" not in counts and len(counts) == len(W.wordings("11")) - 1
    assert min(counts.values()) > 60 and max(counts.values()) < 145      # 100 expected each


def test_sample_item_task_and_pools():
    rng = np.random.default_rng(19)
    assert Counter(I.sample_item_task(rng) for _ in range(2000)).keys() == set(W.TASKS)
    grids_only = I.Needs(answer_kinds=frozenset({"grid"}))
    assert all(W.answer_kind(I.sample_item_task(rng, grids_only)) == "grid" for _ in range(200))
    assert set(I.task_pool(I.Needs(block_capable=False))) == set(W.TASKS) - W.STRUCTURED_TASKS
    assert set(I.task_pool(I.Needs(tasks=("1", "2")))) == {"1", "2"}
    with pytest.raises(ValueError):
        I.sample_item_task(rng, I.Needs(tasks=("14",)))


def test_retargeting_reads_cleanly_with_a_lead_clause():
    """'About the third grid: <text>' -- the text starts a sentence, never names another grid by
    position, and a multi-grid part never gets a text that places its grid."""
    rng = np.random.default_rng(13)
    needs = I.Needs(multi_grid=True, question_after_grid=True, multi_answer=True, ordinal_refs=True)
    for _ in range(400):
        slot, query = I.item_query(rng, needs=needs)
        line = "About the third grid: " + query.text
        assert query.text[0].isupper()
        low = line.lower()
        for bad in ("the grid below", "grid above", "picture below", "the following grid", "subgrid",
                    "first grid", "second grid", "other grid"):
            assert bad not in low, (query.qid, bad)


def test_item_query_is_fast():
    rng = np.random.default_rng(17)
    I.task_pool(I.Needs())            # warm the probe cache
    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        I.item_query(rng)
        times.append(time.perf_counter() - t0)
    times.sort()
    assert times[len(times) // 2] < 0.010, times[len(times) // 2]


def test_stage2_never_hands_a_coordinate_list_task_its_answer():
    import numpy as np
    from pv2 import items
    from registry import generate
    for tid in sorted(items.STAGE2_GIVES_AWAY):
        g = generate(tid, 0)["input"]
        assert "stage2" not in items.encoding_weights(tid, g, items.Needs())
    assert "stage2" in items.encoding_weights("77", generate("77", 0)["input"], items.Needs())
