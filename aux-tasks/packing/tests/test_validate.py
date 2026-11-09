"""Mutation tests: every validator check (V1-V5) catches a deliberately corrupted row."""
import copy
import re

import numpy as np
import pytest

import stub_producers as sp
from pv2.assemble import AssembleConfig, Row, assemble_row
from pv2.assemble import row as row_module
from pv2.model import FLAG_IO_NOUNS, LABEL_WORD_PREFIX, Answer, Part, Query, Slot
from pv2.validate import ValidationError, validate_row


def _expect(check, row, parts):
    with pytest.raises(ValidationError) as e:
        validate_row(row, parts)
    assert e.value.check == check, str(e.value)


def edit_prompt(row, start, end, new):
    """Replace prompt[start:end] by ``new`` and shift every recorded offset after it (so only the
    content changes, never the bookkeeping)."""
    delta = len(new) - (end - start)
    prompt = row.prompt[:start] + new + row.prompt[end:]
    meta = copy.deepcopy(row.meta)
    ch = meta["checks"]
    if ch["row_header"] is not None and start < len(ch["row_header"]):
        ch["row_header"] = prompt[:len(ch["row_header"]) + delta]
    for pc in ch["parts"]:
        a, b = pc["span"]
        pc["span"] = [a + delta if a >= end else a, b + delta if b >= end else b]
        for bl in pc["blocks"]:
            if bl["start"] >= end:
                bl["start"] += delta
                bl["end"] += delta
    return Row(prompt, row.target, meta)


def insert_in_part(row, pi, text):
    """Insert a paragraph right after the part's first paragraph."""
    a, b = row.meta["checks"]["parts"][pi]["span"]
    cut = row.prompt.find("\n\n", a, b)
    cut = b if cut < 0 else cut
    return edit_prompt(row, cut, cut, "\n\n" + text)


def find(pred, cfg=None, seeds=range(4000), maker=sp.random_parts):
    for seed in seeds:
        rng = np.random.default_rng(seed)
        parts = maker(rng)
        row = assemble_row(rng, parts, config=cfg)
        if pred(parts, row):
            return parts, row
    raise AssertionError("no row with the wanted shape")


def test_clean_rows_pass():
    for seed in range(50):
        rng = np.random.default_rng(seed)
        parts = sp.random_parts(rng)
        validate_row(assemble_row(rng, parts), parts)


# --- V1 ---------------------------------------------------------------------------------------------------

def test_v1_changed_cell():
    parts, row = find(lambda p, r: r.meta["checks"]["parts"][0]["blocks"][0]["encoding"] == "compact")
    bl = row.meta["checks"]["parts"][0]["blocks"][0]
    i = bl["start"]
    new = "1" if row.prompt[i] != "1" else "2"
    _expect("V1", edit_prompt(row, i, i + 1, new), parts)


def test_v1_grid_glued_to_the_next_paragraph():
    parts, row = find(lambda p, r: r.meta["checks"]["parts"][0]["delimiter"] in ("label_line", "blank", "banner"))
    bl = row.meta["checks"]["parts"][0]["blocks"][0]
    e = bl["end"]
    assert row.prompt[e:e + 2] == "\n\n"
    _expect("V1", edit_prompt(row, e, e + 2, "\n"), parts)


def test_v1_wrong_size_in_header():
    parts, row = find(lambda p, r: any(b["header"] and "rows x" in b["header"] for b in r.meta["checks"]["parts"][0]["blocks"]))
    bl = next(b for b in row.meta["checks"]["parts"][0]["blocks"] if b["header"] and "rows x" in b["header"])
    hs = row.prompt.rfind(bl["header"], 0, bl["start"])
    m = re.search(r"\((\d+) rows", bl["header"])
    s = hs + m.start(1)
    wrong = str(int(m.group(1)) % 9 + 1)
    mutated = edit_prompt(row, s, s + len(m.group(1)), wrong)
    for pc in mutated.meta["checks"]["parts"]:
        for b in pc["blocks"]:
            if b is not None and b["header"] == bl["header"]:
                b["header"] = b["header"].replace(f"({m.group(1)} rows", f"({wrong} rows")
    _expect("V1", mutated, parts)


def test_v1_missing_boundary():
    parts, row = find(lambda p, r: r.meta["n_parts"] > 1)
    pc = row.meta["checks"]["parts"][1]
    a = pc["span"][0]
    _expect("V1", edit_prompt(row, a, a + len(pc["boundary"]), "Next"), parts)


# --- V2 ---------------------------------------------------------------------------------------------------

def test_v2_unqualified_ordinals_in_a_later_part(monkeypatch):
    """An assembler that forgot to qualify: 'the first grid' in part 2 also reads as the row's first."""
    monkeypatch.setattr(row_module, "_qualify", lambda plans, noun: None)
    cfg = AssembleConfig(validate=False, scheme_multi={"label": 0.0, "ordinal": 1.0})

    def two(rng):
        return [sp.reframed(rng), sp.gallery(rng, lib_only=True)]
    parts, row = find(lambda p, r: len(p[1].slots) > 1, cfg=cfg, maker=two)
    _expect("V2", row, parts)


def test_v2_reference_to_another_grid():
    cfg = AssembleConfig(scheme_multi={"label": 0.0, "ordinal": 1.0}, ordinal_end_p=0.0)

    def one(rng):
        grids = [sp.rand_grid(rng, 4, 4) for _ in range(6)]
        return [Part("gallery", [Slot(g) for g in grids], [sp.lib_count(rng, grids[2], 2)])]
    parts, row = find(lambda p, r: True, cfg=cfg, maker=one)
    i = row.prompt.find("the third grid")
    assert i >= 0
    mutated = edit_prompt(row, i, i + len("the third grid"), "the fifth grid")
    d = mutated.meta["checks"]["parts"][0]["questions"][0]
    d["text"] = d["text"].replace("the third grid", "the fifth grid")
    d["refs"] = [["the fifth grid", 2]]
    _expect("V2", mutated, parts)


def test_v2_stray_reference():
    cfg = AssembleConfig(scheme_multi={"label": 0.0, "ordinal": 1.0})

    def one(rng):
        grids = [sp.rand_grid(rng, 4, 4) for _ in range(3)]
        return [Part("gallery", [Slot(g) for g in grids], [sp.lib_count(rng, grids[0], 0)])]
    parts, row = find(lambda p, r: True, cfg=cfg, maker=one)
    _expect("V2", insert_in_part(row, 0, "Ignore the ninth grid."), parts)


def test_v2_duplicate_label():
    cfg = AssembleConfig(scheme_single={"none": 0.0, "label": 1.0}, scheme_multi={"label": 1.0, "ordinal": 0.0})
    parts, row = find(lambda p, r: r.meta["n_parts"] > 1 and all(m["scheme"] == "label" for m in r.meta["parts"]),
                      cfg=cfg)
    # the question of part 2 names part 1's grid label instead of its own
    lab1 = row.meta["checks"]["parts"][0]["labels"][0]
    pc = row.meta["checks"]["parts"][1]
    d = pc["questions"][0]
    phrase, slot = d["refs"][0]
    i = row.prompt.find(d["text"], pc["span"][0])
    j = row.prompt.find(phrase, i)
    mutated = edit_prompt(row, j, j + len(phrase), lab1)
    md = mutated.meta["checks"]["parts"][1]["questions"][0]
    md["text"] = md["text"].replace(phrase, lab1, 1)
    md["refs"][0] = [lab1, slot]
    _expect("V2", mutated, parts)


# --- V3 ---------------------------------------------------------------------------------------------------

def test_v3_wrong_gold_number():
    parts, row = find(lambda p, r: any(q.answer.kind == "number" for q in p[0].queries))
    q = next(q for q in parts[0].queries if q.answer.kind == "number")
    mutated_parts = copy.deepcopy(parts)
    i = parts[0].queries.index(q)
    mutated_parts[0].queries[i].answer = Answer("number", q.answer.value + 1, str(q.answer.value + 1))
    _expect("V3", row, mutated_parts)


def test_v3_wrong_gold_grid():
    parts, row = find(lambda p, r: any(q.answer.kind == "grid" for q in p[0].queries))
    mutated_parts = copy.deepcopy(parts)
    q = next(q for q in mutated_parts[0].queries if q.answer.kind == "grid")
    g = [list(r) for r in q.answer.value]
    g[0][0] = (g[0][0] + 1) % 10
    q.answer = Answer("grid", g)
    _expect("V3", row, mutated_parts)


def test_v3_answer_names_the_wrong_grids():
    parts, row = find(lambda p, r: any(q.qid == "lib:which" and q.answer.value for q in p[0].queries),
                      maker=lambda rng: [sp.compare(rng)])
    mutated_parts = copy.deepcopy(parts)
    q = next(q for q in mutated_parts[0].queries if q.qid == "lib:which")
    q.answer = Answer("list", [s + 1 for s in q.answer.value], q.answer.text)
    _expect("V3", row, mutated_parts)


# --- V4 ---------------------------------------------------------------------------------------------------

def test_v4_answers_swapped_in_the_target():
    cfg = AssembleConfig(force={"style": "labelled"})

    def one(rng):
        g = sp.rand_grid(rng, 5, 5)
        return [Part("x", [Slot(g)], [sp.lib_count(rng, g, 0), sp.lib_present(rng, g, 0)])]
    parts, row = find(lambda p, r: True, cfg=cfg, maker=one)
    a1, a2 = [d["answer"] for d in row.meta["checks"]["parts"][0]["questions"]]
    target = row.target.replace(f" {a1}\n", " @@\n", 1).replace(f" {a2}", f" {a1}").replace(" @@\n", f" {a2}\n")
    assert target != row.target
    _expect("V4", Row(row.prompt, target, row.meta), parts)


def test_v4_json_type_changed():
    cfg = AssembleConfig(force={"style": "json"})

    def one(rng):
        g = sp.rand_grid(rng, 5, 5)
        return [Part("x", [Slot(g)], [sp.lib_count(rng, g, 0), sp.lib_present(rng, g, 0)])]
    parts, row = find(lambda p, r: True, cfg=cfg, maker=one)
    n = row.meta["checks"]["parts"][0]["questions"][0]["answer"]
    target = re.sub(rf": {n}\b", f': "{n}"', row.target, count=1)
    _expect("V4", Row(row.prompt, target, row.meta), parts)


def test_v4_missing_heading():
    parts, row = find(lambda p, r: r.meta["n_parts"] > 1)
    h = row.meta["checks"]["parts"][1]["heading"]
    _expect("V4", Row(row.prompt, row.target.replace(h + "\n", "", 1), row.meta), parts)


def test_v4_trailing_text():
    parts, row = find(lambda p, r: r.meta["n_parts"] == 1)
    _expect("V4", Row(row.prompt, row.target + "\nThat is all.", row.meta), parts)


# --- V5 ---------------------------------------------------------------------------------------------------

def _single(parts, row):
    return row.meta["n_parts"] == 1


@pytest.mark.parametrize("text", [
    "The background is color 0.",
    "Note that the background color is black.",
    "What is the size of Test input 1? Answer as HxW (rows x columns) only.",
    "How many cells of colour 3 does Training pair 1 input have? Answer with the number only.",
    "You will see 9 grids.",
    "Leftover ⟪g0⟫ here.",
])
def test_v5_bad_sentences(text):
    parts, row = find(_single)
    _expect("V5", insert_in_part(row, 0, text), parts)


def test_v5_contradicting_reply_shape():
    cfg = AssembleConfig(force={"style": "json"})
    parts, row = find(_single, cfg=cfg)
    _expect("V5", insert_in_part(row, 0, "Reply in full sentences."), parts)
    _expect("V5", insert_in_part(row, 0, "Answer with the number only."), parts)


def _remove_sentence(row, sentences):
    for s in sentences:
        i = row.prompt.find(s)
        if i >= 0:
            j = i + len(s)
            if row.prompt[j:j + 1] == " ":
                j += 1
            mutated = edit_prompt(row, i, j, "")
            for pc in mutated.meta["checks"]["parts"]:
                pc["instructions"] = pc["instructions"].replace(s + " ", "").replace(s, "")
            return mutated
    raise AssertionError("sentence not found")


def test_v5_missing_zero_index_sentence():
    from pv2.assemble.layouts import ZERO_INDEX_SENTENCES

    def one(rng):
        g = sp.rand_grid(rng, 5, 5)
        return [Part("x", [Slot(g)], [sp.lib_cell(rng, g, 0), sp.lib_count(rng, g, 0)])]
    parts, row = find(lambda p, r: True, maker=one)
    _expect("V5", _remove_sentence(row, ZERO_INDEX_SENTENCES), parts)


def test_v5_missing_as_shown_sentence():
    from pv2.assemble.layouts import AS_SHOWN_SENTENCES

    def one(rng):
        g = sp.rand_grid(rng, 5, 5)
        return [Part("x", [Slot(g)], [sp.lib_recolour(rng, g, 0), sp.lib_count(rng, g, 0)])]
    parts, row = find(lambda p, r: True, maker=one)
    _expect("V5", _remove_sentence(row, AS_SHOWN_SENTENCES), parts)


def test_v5_size_header_on_a_sized_grid():
    cfg = AssembleConfig(scheme_single={"none": 0.0, "label": 1.0}, force={"dims": True})

    def one(rng):
        g = sp.rand_grid(rng, 5, 5)
        return [Part("x", [Slot(g)], [sp.lib_count(rng, g, 0)])]
    parts, row = find(lambda p, r: "rows x" in r.prompt, cfg=cfg, maker=one)
    mutated = copy.deepcopy(parts)
    mutated[0].slots[0].dims_header_ok = False
    _expect("V5", row, mutated)


def test_v5_io_nouns_in_a_role_part():
    parts, row = find(lambda p, r: True, maker=lambda rng: [sp.arc(rng)])
    mutated = copy.deepcopy(parts)
    q = mutated[0].queries[0]
    mutated[0].queries[0] = Query(q.source, q.qid, q.reads, q.text, q.answer, q.statement,
                                  q.flags | {FLAG_IO_NOUNS}, q.lead_clause)
    _expect("V5", row, mutated)


def test_v5_label_letter_next_to_a_shape_letter():
    cfg = AssembleConfig(scheme_multi={"label": 1.0, "ordinal": 0.0}, label_nouns={"Grid": 1.0},
                         force={"label_style": "letter"})

    def one(rng):
        grids = [sp.rand_grid(rng, 4, 4) for _ in range(3)]
        return [Part("x", [Slot(g) for g in grids], [sp.lib_count(rng, grids[0], 0)])]
    parts, row = find(lambda p, r: True, cfg=cfg, maker=one)
    letter = row.meta["checks"]["parts"][0]["labels"][1][-1]
    mutated = copy.deepcopy(parts)
    q = mutated[0].queries[0]
    mutated[0].queries[0] = Query(q.source, q.qid, q.reads, q.text, q.answer, q.statement,
                                  q.flags | {LABEL_WORD_PREFIX + letter}, q.lead_clause)
    _expect("V5", row, mutated)


def test_v5_questions_not_numbered_continuously():
    parts, row = find(lambda p, r: r.meta["n_questions"] >= 3 and r.meta["question_tags"] == "number")
    mutated = copy.deepcopy(row.meta)
    qs = [d for pc in mutated["checks"]["parts"] for d in pc["questions"]]
    qs[-1]["tag"] = str(len(qs) + 1)
    with pytest.raises(ValidationError):
        validate_row(Row(row.prompt, row.target, mutated), parts)


def test_meta_survives_json_and_revalidates():
    import json
    for seed in range(200):
        rng = np.random.default_rng(seed)
        parts = sp.random_parts(rng)
        row = assemble_row(rng, parts)
        meta = json.loads(json.dumps(row.meta))
        validate_row(Row(row.prompt, row.target, meta), parts)


# --- follow-up checks ------------------------------------------------------------------------------------

def test_v3_context_run_together():
    def one(rng):
        part = sp.arc(rng, n_pairs=2, n_tests=1)
        part.context = ["Possible rules:", "A. Flip each row.", "B. Keep the grid."]
        return [part]
    parts, row = find(lambda p, r: True, maker=one)
    i = row.prompt.find("Possible rules:\nA. Flip each row.")
    assert i >= 0
    j = i + len("Possible rules:")
    _expect("V3", edit_prompt(row, j, j + 1, " "), parts)


def test_v5_grid_placeholder_next_to_tag_frames():
    cfg = AssembleConfig(force={"delimiter": "tags", "style": "labelled"})

    def one(rng):
        g = sp.rand_grid(rng, 5, 5)
        return [Part("x", [Slot(g)], [sp.lib_row(rng, g, 0), sp.lib_count(rng, g, 0)])]
    parts, row = find(lambda p, r: r.meta["parts"][0]["delimiter"] == "tags", cfg=cfg, maker=one)
    i = row.prompt.find("[grid]")
    mutated = edit_prompt(row, i, i + len("[grid]"), "<grid>")
    pc = mutated.meta["checks"]["parts"][0]
    pc["instructions"] = pc["instructions"].replace("[grid]", "<grid>")
    pc["style"]["variant"].pop("grid_ph")
    _expect("V5", mutated, parts)


def test_v5_mixed_arc_bases(monkeypatch):
    calls = iter([0, 1] * 10)
    orig = row_module.chance

    def flip(rng, p):
        return next(calls) == 0 if p == AssembleConfig().role_zero_based_p else orig(rng, p)
    monkeypatch.setattr(row_module, "chance", flip)
    old = row_module._plan_refs

    def per_part(rng, cfg, plan, book, nouns, gok, row_state):
        row_state.clear()                    # the old behaviour: a base per part
        return old(rng, cfg, plan, book, nouns, gok, row_state)
    monkeypatch.setattr(row_module, "_plan_refs", per_part)
    parts, row = find(lambda p, r: len({m["base"] for m in r.meta["parts"]}) == 2,
                      cfg=AssembleConfig(validate=False), maker=lambda rng: [sp.arc(rng), sp.arc(rng)])
    _expect("V5", row, parts)


def test_v5_letter_tag_without_parentheses():
    cfg = AssembleConfig(force={"tags": "letter", "layout": "reminder"})

    def one(rng):
        grids = [sp.rand_grid(rng, 4, 4) for _ in range(3)]
        return [Part("x", [Slot(g) for g in grids], [sp.lib_count(rng, grids[0], 0), sp.lib_present(rng, grids[1], 1)])]
    parts, row = find(lambda p, r: "Now answer questions (a) and (b)." in r.prompt, cfg=cfg, maker=one)
    i = row.prompt.find("Now answer questions (a) and (b).")
    _expect("V5", edit_prompt(row, i, i + len("Now answer questions (a) and (b)."),
                              "Now answer questions a and b."), parts)


def test_v5_bare_exclusivity_under_a_heading():
    cfg = AssembleConfig(force={"style": "bare"})

    def two(rng):
        g = sp.rand_grid(rng)
        return [Part("x", [Slot(g)], [sp.lib_count(rng, g, 0)]), sp.gallery(rng)]
    parts, row = find(lambda p, r: r.meta["parts"][0]["style"] == "bare", cfg=cfg, maker=two)
    pc = row.meta["checks"]["parts"][0]
    instr = pc["instructions"]
    i = row.prompt.find(instr)
    new = instr.rstrip(".") + ", with no other text."
    mutated = edit_prompt(row, i, i + len(instr), new)
    mutated.meta["checks"]["parts"][0]["instructions"] = new
    _expect("V5", mutated, parts)


def test_rows_never_carry_solver_channel_markers_or_prompt_sentences():
    """Training session (2026-09-29): "Final answer:" / "Attempt k:" lines and think tags are the solver
    arms' split points, and its system-prompt sentences mark the solver context."""
    import numpy as np
    from pv2.build_dataset import BuildConfig, build_row
    from _tok import tokenizer_or_skip
    from pv2 import validate as V
    if not V._solver_sentences():
        pytest.skip("the solver system prompts (../training/arcsolver/prompt/prompts) are not present")
    tok = tokenizer_or_skip()
    for s in range(60):
        b = build_row(4_400_000 + s, BuildConfig(classic_share=0.0), tok)
        if not b:
            continue
        for text in b[:2]:
            assert not V.SOLVER_MARKER_LINE.search(text)
            assert "</think>" not in text and "<think>" not in text
        assert not any(sent in b[0] for sent in V._solver_sentences())
    assert V.SOLVER_MARKER_LINE.search("x\nFinal answer:\n7") and V.SOLVER_MARKER_LINE.search("Attempt 2:")


def test_names_grids_with_digit_grid_answers_carry_the_mapping():
    """Training-side blind solve (row 5070000949): a grid in colour names, a JSON grid answer in
    digits, no mapping in the prompt."""
    import numpy as np
    from pv2 import library as Lb
    from pv2.assemble import assemble_row, AssembleConfig
    from pv2.assemble import layouts as LL
    from pv2.model import Part, Slot
    rng = np.random.default_rng(12)
    seen = 0
    for style in ("json", "labelled", "bare"):
        for enc_ in ("compact", "json_rows"):
            for _ in range(10):
                g = rng.integers(0, 6, size=(6, 7)).tolist()
                slots = [Slot(g, "names")]
                q = Lb.pose(rng, slots, (0,), qid="row")
                if q is None:
                    continue
                row = assemble_row(rng, [Part("multi_question", slots, [q])],
                                   config=AssembleConfig(force={"style": style, "grid_enc": enc_}))
                pc = row.meta["checks"]["parts"][0]
                if pc["grid_answer_encoding"] != "names":
                    assert LL.NAMES_TO_DIGITS_MAP in row.prompt
                    seen += 1
    assert seen >= 20
