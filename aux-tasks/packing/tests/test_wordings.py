"""Wordings, their flags and the answer-wrapper table, measured over every wording of every
in-scope task on the coverage seeds (one seed per distinct canonical wording over seeds 0-399,
stored in tests/data/coverage_seeds_400.json, plus seeds 0-7)."""
import json
import os
import re
from collections import defaultdict

import pytest

from pv2 import model
from pv2 import wordings as W
import prompt_variants as pv
from registry import generate

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_TABLE = os.path.join(HERE, "data", "coverage_seeds_400.json")


def regenerate_seed_table(path=SEED_TABLE):  # pragma: no cover - maintenance helper
    """Rebuild the stored seed table from the generator package (about two minutes)."""
    seeds = {t: sorted(set(pv.coverage_seeds(t, 400)) | set(range(8))) for t in W.TASKS}
    with open(path, "w") as f:
        json.dump({"note": "coverage_seeds(tid, 400) plus seeds 0-7", "seeds": seeds}, f, indent=0, sort_keys=True)


@pytest.fixture(scope="module")
def seed_table():
    with open(SEED_TABLE) as f:
        table = json.load(f)["seeds"]
    assert set(table) == set(W.TASKS)
    return table


@pytest.fixture(scope="module")
def renderings(seed_table):
    """[(task, seed, example, [Rendered per wording])] over the coverage seeds."""
    out = []
    for tid in W.TASKS:
        for seed in seed_table[tid]:
            ex = generate(tid, seed)
            out.append((tid, seed, ex, W.render_all(tid, ex.get("metadata", {}))))
    return out


@pytest.fixture(scope="module")
def table(renderings):
    flags = defaultdict(set)
    for _, _, _, rendered in renderings:
        for r in rendered:
            flags[r.wording.wid] |= r.flags
    return flags


def _ids(table, flag):
    return {wid for wid, f in table.items() if flag in f}


# --- the wordings ---------------------------------------------------------------------------------

def test_scope_and_store():
    assert len(W.TASKS) == 95
    assert not {"14", "41", "78"} & set(W.TASKS)
    payload = pv.load()
    total = 0
    for tid in W.TASKS:
        ws = W.wordings(tid)
        stored = pv.variants_for(tid, payload)
        assert ws[0].index == 0 and ws[0].template is None and ws[0].wid == f"{tid}:canon"
        assert [w.variant_id for w in ws[1:]] == [v["id"] for v in stored]
        assert [w.index for w in ws] == list(range(len(ws)))
        total += len(stored)
    assert total == 609          # 628 stored wordings minus the 19 of tasks 14, 41 and 78


def test_render_matches_the_old_paths(seed_table):
    """Canonical = get_prompt; variants = prompt_variants.render (compound tokens expanded)."""
    from prompts import get_prompt
    for tid in ("3", "9", "13", "77", "102"):
        meta = generate(tid, seed_table[tid][-1]).get("metadata", {})
        ws = W.wordings(tid)
        assert ws[0].render(meta) == get_prompt(tid, meta)
        for w in ws[1:]:
            assert w.render(meta) == pv.render(tid, w.template, meta=meta)
            assert "{" not in w.render(meta)


# --- answers ----------------------------------------------------------------------------------------

def test_answer_kinds_match_every_generated_output(renderings):
    for tid, seed, ex, _ in renderings:
        a = W.make_answer(tid, ex["output"])
        if a.kind == "grid":
            assert a.value == ex["output"]
        else:
            assert a.text == str(ex["output"]), (tid, seed)
            if a.kind in ("number", "color"):
                assert str(a.value) == a.text
    kinds = {t: W.answer_kind(t) for t in W.TASKS}
    assert sum(k == "grid" for k in kinds.values()) == 60
    assert {t for t, k in kinds.items() if k == "color"} == {"25", "57", "75"}
    assert {t for t, k in kinds.items() if k == "token"} == {"36", "38", "56"}


def test_vocabularies_are_the_stated_ones(renderings):
    seen = defaultdict(set)
    for tid, _, ex, rendered in renderings:
        if tid in W.VOCABULARY:
            seen[tid].add(ex["output"])
            for r in rendered:    # every wording names every word of its vocabulary
                for word in W.VOCABULARY[tid]:
                    assert word in r.neutral, (r.wording.wid, word)
    for tid, vocab in W.VOCABULARY.items():
        assert seen[tid] <= set(vocab)
        assert len(seen[tid]) >= 2


def test_dims_task_answers_the_size_and_no_other_task_does(renderings):
    for tid, seed, ex, _ in renderings:
        g = ex["input"]
        size = f"{len(g)}x{len(g[0])}"
        if tid in W.DIMS_TASKS:
            assert ex["output"] == size
        else:
            assert ex["output"] != size, (tid, seed)


def test_structured_tasks_are_measured(renderings):
    structured = set()
    for tid, _, ex, _ in renderings:
        if W.answer_kind(tid) == "text" and re.search(r"[\[\n:]|, ", ex["output"]):
            structured.add(tid)
    assert structured == set(W.STRUCTURED_TASKS)


def test_coordinate_tasks_are_measured(renderings):
    measured = {tid for tid, _, _, rendered in renderings for r in rendered if W.coords_in_text(r.text)}
    assert measured == set(W.COORD_TASKS)


# --- the wrapper table ------------------------------------------------------------------------------

REQUIRED_NOTATION = {   # notation specifics every neutral clause keeps verbatim
    "Answer with the number (0 if nothing is enclosed).": ["number", "(0 if nothing is enclosed)"],
    "Answer with the number.": ["number"],
    "Answer as HxW.": ["HxW"],
    "Answer as MxN.": ["MxN"],
    "Answer as 90, 180, or 270.": ["90, 180, or 270"],
    "Answer as a single color label.": ["single color label"],
    "Answer 'closed' or 'open'.": ["'closed' or 'open'"],
    "Answer 'true' or 'false'.": ["'true' or 'false'"],
    "Answer 'yes' or 'no'.": ["'yes' or 'no'"],
    W.WRAPPERS[9][0]: ["row-major order", "[(r,c),...]"],
    W.WRAPPERS[10][0]: ["first:[(row,col),(row,col),...], second:[(row,col),(row,col),...]"],
    W.WRAPPERS[11][0]: ["[(row,col),(row,col),...]"],
    W.WRAPPERS[12][0]: ["color:count, color:count, ..."],
    W.WRAPPERS[13][0]: ["one line for each label in that order", "label: yes or label: no"],
}


def test_wrapper_table_keeps_the_notation():
    assert set(REQUIRED_NOTATION) == {o for o, _ in W.WRAPPERS}
    for orig, neutral in W.WRAPPERS:
        for piece in REQUIRED_NOTATION[orig]:
            assert piece in orig and piece in neutral, (orig, piece)
        assert not W.REPLY_SHAPE.search(neutral), neutral
        assert not pv._ANSWER_SENTENCE.search(neutral), neutral


def test_every_wording_has_a_known_wrapper_or_none(renderings):
    """Completeness: grid tasks carry no wrapper; every value task's wording carries a table
    wrapper (tasks 2 and 94 may state the notation inline instead); nothing that the generator
    package's own answer-sentence pattern finds is left outside the table."""
    used = set()
    for tid, seed, _, rendered in renderings:
        for r in rendered:
            found = W.wrappers_in(r.text)
            used.update(found)
            if W.answer_kind(tid) == "grid":
                assert not found and not pv._ANSWER_SENTENCE.search(r.text), (r.wording.wid, seed)
            elif tid not in W.INLINE_NOTATION_TASKS:
                assert found, (r.wording.wid, seed)
            for m in pv._ANSWER_SENTENCE.finditer(r.text):
                assert any(m.start() >= r.text.find(o) and m.end() <= r.text.find(o) + len(o) for o in found), \
                    (r.wording.wid, m.group(0))
    assert used == {o for o, _ in W.WRAPPERS}      # no dead entries


def test_neutral_text_has_no_reply_shape(renderings):
    for tid, seed, _, rendered in renderings:
        for r in rendered:
            assert r.neutral == W.neutral_text(tid, r.text)
            assert not W.REPLY_SHAPE.search(r.neutral), (r.wording.wid, W.REPLY_SHAPE.search(r.neutral).group(0))
            assert not pv._ANSWER_SENTENCE.search(r.neutral)
            for phrase in ("Answer with", "Answer as", "nothing else", "extra text", "Do not add"):
                if phrase == "nothing else" and phrase in r.neutral:
                    # content uses remain ("to nothing else", "nothing else about it changes")
                    assert not re.search(r"\banswer\b[^.]*nothing else", r.neutral, re.I), r.wording.wid
                    continue
                assert phrase not in r.neutral, (r.wording.wid, phrase)
            # only the wrapper changed
            rebuilt = r.text
            for orig, neutral in W.WRAPPERS:
                rebuilt = rebuilt.replace(orig, neutral)
            if r.wording.wid != "77_v006":
                assert rebuilt == r.neutral, r.wording.wid


def test_value_tasks_neutral_texts_are_exact(seed_table):
    meta = generate("1", 0).get("metadata", {})
    assert W.neutral_text("1", W.wordings("1")[0].render(meta)) == \
        "What are the dimensions of this grid? Write the size as HxW."
    meta = generate("77", seed_table["77"][0]).get("metadata", {})
    text = W.wordings("77")[6].render(meta)
    assert "and nothing else" in text
    assert "and nothing else" not in W.neutral_text("77", text)
    assert W.neutral_text("77", text).startswith("Your answer is ")
    with pytest.raises(ValueError):
        W.neutral_text("30", "How many color 5 cells are in this grid?")        # wrapper missing
    with pytest.raises(ValueError):
        W.neutral_text("30", "How many cells? Answer with a single digit.")      # unknown wrapper


# --- flags ------------------------------------------------------------------------------------------

def test_placement_wordings(table):
    assert _ids(table, model.FLAG_PLACEMENT) == {"11_v005", "44_v006", "45_v005", "66_v006", "80_v005"}


def test_nothing_else_wordings(table):
    # measured on the neutral text v2 shows: the demands of 102 (all wordings) and 77_v006 are removed
    # there, so no wording keeps the flag (the raw texts still carry them)
    assert _ids(table, model.FLAG_NOTHING_ELSE) == set()
    for tid in ("102", "77"):
        for r in W.render_all(tid, generate(tid, 0)["metadata"]):
            assert not W._NOTHING_ELSE.search(r.neutral), (tid, r.wording)


def test_io_nouns(table):
    io = _ids(table, model.FLAG_IO_NOUNS)
    canon = {W.task_of(w) for w in io if w.endswith(":canon")}
    # the Opus review's canonical list, measured
    assert canon == {"40", "42", "44", "64", "65", "71", "83", "93", "95", "96", "98", "101"}
    assert len({W.task_of(w) for w in io}) >= 22
    # verb uses ("Output the whole grid.", "and output the full grid", "Output the result.") are not nouns
    for wid in ("5a:canon", "32:canon", "11_v008", "80_v006", "77:canon", "79:canon", "86:canon"):
        assert wid in table and wid not in io, wid
    for wid in ("33_v005", "94_v003", "42_v006", "77_v004", "44:canon", "64:canon"):
        assert wid in io, wid


def test_label_words(table):
    assert "label_word:image" in table["9:canon"]
    assert "label_word:panel" in table["96_v004"]
    assert "label_word:figure" in table["72_v003"]
    assert "label_word:map" in table["99:canon"]
    assert "label_word:#" in table["4:canon"] and "label_word:#" in table["59:canon"]
    assert {"label_word:L", "label_word:X"} <= table["4:canon"] and "label_word:T" in table["59:canon"]
    assert "label_word:grid_number" in table["10:canon"] and "label_word:grid_number" in table["44:canon"]
    assert "label_word:grid" not in {f for flags in table.values() for f in flags}
    assert "label_word:A" not in {f for flags in table.values() for f in flags}


def test_style_labels_and_inner_ordinals(table):
    assert _ids(table, "style_label:answer") == {"26_v003"}
    assert "36_v003" in _ids(table, "style_label:question")
    assert {W.task_of(w) for w in _ids(table, W.FLAG_INNER_ORDINALS)} == {"42"}
    assert len(_ids(table, W.FLAG_INNER_ORDINALS)) == len(W.wordings("42"))


def test_task_flags_follow_the_tables(table):
    for wid, flags in table.items():
        tid = W.task_of(wid)
        assert (model.FLAG_COORDS in flags) == (tid in W.COORD_TASKS)
        assert (model.FLAG_DIMS in flags) == (tid in W.DIMS_TASKS)
        assert (model.FLAG_STRUCTURED in flags) == (tid in W.STRUCTURED_TASKS)
        assert (model.FLAG_EDIT in flags) == (W.answer_kind(tid) == "grid")


def test_flags_are_consistent_with_the_text(renderings):
    for tid, _, _, rendered in renderings:
        for r in rendered:
            assert r.flags == W.text_flags(r.text, r.neutral) | W.task_flags(tid)
            assert (model.FLAG_LONG in r.flags) == (len(r.neutral) > W.LONG_CHARS)
            for word, rx in W.LABEL_WORDS.items():
                assert (model.LABEL_WORD_PREFIX + word in r.flags) == bool(rx.search(r.text))
            letters = {f for f in r.flags if f.startswith(model.LABEL_WORD_PREFIX) and len(f) == len("label_word:X")
                       and f[-1].isupper()}
            assert letters == {model.LABEL_WORD_PREFIX + c for c in W._LABEL_LETTER.findall(r.text)}


def test_flag_counts_report(table, capsys):
    counts = W.flag_counts({k: frozenset(v) for k, v in table.items()})
    assert counts[model.FLAG_PLACEMENT] == (5, 5)
    assert model.FLAG_NOTHING_ELSE not in counts
    assert sum(1 for w in table) == 95 + 609
    with capsys.disabled():
        print("\nflag counts (wordings, tasks):", json.dumps(counts))
