"""Rows: property tests over thousands of assembled rows from stub producers, every allowed
scheme x layout x style combination, fidelity to the stage-2 format, and the contract errors."""
import importlib.util
import itertools
import os
import re
from collections import Counter

import numpy as np
import pytest

import stub_producers as sp
from pv2.assemble import AssembleConfig, AssemblyError, assemble_row
from pv2.assemble import layouts as L
from pv2.assemble import styles as S
from pv2.model import FLAG_COORDS, FLAG_EDIT, Answer, Part, Query, Slot
from pv2.paths import SOLVER_PROMPT_DIR
from pv2.validate import PROBE_PATTERNS, validate_row

N_ROWS = 3000


def _arc_render():
    path = os.path.join(SOLVER_PROMPT_DIR, "arc_render.py")
    if not os.path.isfile(path):
        pytest.skip("the solver renderer (../training/arcsolver/prompt/arc_render.py) is not present")
    spec = importlib.util.spec_from_file_location("arc_render_rows", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make(seed):
    rng = np.random.default_rng(seed)
    parts = sp.random_parts(rng)
    return parts, assemble_row(rng, parts)


@pytest.fixture(scope="module")
def sample():
    return [_make(seed) for seed in range(N_ROWS)]


def test_every_row_validates_and_round_trips(sample):
    for parts, row in sample:
        validate_row(row, parts)                       # also run inside assemble_row
        for pc in row.meta["checks"]["parts"]:
            spec = S.StyleSpec.from_dict(pc["style"])
            assert S.parse(spec, S.render(spec, [d["answer"] for d in pc["questions"]])) == \
                [d["answer"] for d in pc["questions"]]


def test_rows_are_deterministic(sample):
    for seed in range(0, N_ROWS, 37):
        parts, row = _make(seed)
        assert (row.prompt, row.target, row.meta) == (sample[seed][1].prompt, sample[seed][1].target,
                                                        sample[seed][1].meta)


def test_multi_part_rows_mix_families_and_number_questions_continuously(sample):
    multi = [(p, r) for p, r in sample if len(p) > 1]
    assert len(multi) > N_ROWS // 3
    assert sum(len({x.family for x in p}) > 1 for p, _ in multi) > len(multi) // 2
    for parts, row in multi:
        tags = [d["tag"] for pc in row.meta["checks"]["parts"] for d in pc["questions"]]
        n = len(tags)
        assert tags in ([str(i + 1) for i in range(n)], [chr(97 + i) for i in range(n)])
        for k, pc in enumerate(row.meta["checks"]["parts"]):
            assert pc["boundary"] in row.prompt and f"{k + 1}" in pc["boundary"]


def test_variety_and_frequencies(sample):
    c = {k: Counter() for k in ("scheme", "layout", "delimiter", "style", "grid_answer_encoding")}
    for _, row in sample:
        for m in row.meta["parts"]:
            for k in c:
                c[k][m[k]] += 1
    assert set(c["scheme"]) == {"label", "ordinal", "none", "role"}
    assert set(c["layout"]) == set(L.SINGLE_LAYOUTS) | set(L.MULTI_LAYOUTS)
    assert set(c["delimiter"]) == set(L.DELIMITERS)
    assert set(c["style"]) == set(S.STYLES)
    total = sum(c["style"].values())
    for style, n in c["style"].items():
        assert n / total > 0.02, (style, n)


def test_bare_compact_is_the_majority_of_grid_answers(sample):
    counts = Counter()
    for _, row in sample:
        for pc in row.meta["checks"]["parts"]:
            n = sum(1 for s in pc["style"]["shapes"] if s["grid"])
            if n:
                counts[pc["grid_answer_encoding"]] += n
    assert counts["compact"] / sum(counts.values()) >= 0.60, counts


def test_bare_grid_answers_are_readable_by_stage2(sample):
    ar = _arc_render()
    seen = 0
    for parts, row in sample:
        if len(parts) == 1 and row.meta["parts"][0]["style"] == "bare":
            q = parts[0].queries[0]
            if q.answer.kind == "grid":
                assert ar.parse_grids(row.target) == [q.answer.value]
                seen += 1
    assert seen > 20


def test_scoped_sentences_and_hygiene(sample):
    for parts, row in sample:
        assert "⟪" not in row.prompt + row.target
        for pat in PROBE_PATTERNS:
            assert not pat.search(row.prompt)
        for part, pc in zip(parts, row.meta["checks"]["parts"]):
            seg = row.prompt[pc["span"][0]:pc["span"][1]]
            if any(FLAG_COORDS in q.flags for q in part.queries):
                assert re.search(r"0-indexed|numbered from 0|start at 0", seg)
            edits = [q for q in part.queries if FLAG_EDIT in q.flags]
            if any(set(e.reads) & set(o.reads) for e in edits for o in part.queries if o is not e):
                assert any(s in seg or s[:1].lower() + s[1:] in seg for s in L.AS_SHOWN_SENTENCES)
            # the assembler's own framing never says above/below
            for d in pc["questions"]:
                seg = seg.replace(d["text"], "")
            for b in pc["blocks"]:
                if b["header"]:
                    seg = seg.replace(b["header"], "")
            assert " above" not in seg and " below" not in seg


def test_long_rows_scale(sample):
    rng = np.random.default_rng(99)
    parts = [f(rng) for f in (sp.gallery, sp.arc, sp.one_grid_many, sp.compare, sp.arc, sp.gallery, sp.reframed,
                              sp.arc, sp.one_grid_many, sp.gallery)]
    row = assemble_row(rng, parts)
    assert row.meta["n_parts"] == 10
    assert len(row.prompt) > 3000
    # repeated ARC parts are qualified, so role labels stay unambiguous
    arcs = [m for m in row.meta["parts"] if m["family"] == "arc"]
    assert sum(m["qualified"] for m in arcs) >= 2


# --- exhaustive forced combinations ---------------------------------------------------------------------

def _lib_part(rng, n_slots, n_q, lib_only=True, role=False, grids_ok=True):
    if role:
        part = sp.arc(rng, n_pairs=2, n_tests=1)
        grids = [s.grid for s in part.slots]
        qs = []
        while len(qs) < n_q:
            j = int(rng.integers(len(grids)))
            qs.append(sp.lib_row(rng, grids[j], j) if len(qs) % 2 == 0 and grids_ok else sp.lib_count(rng, grids[j], j))
        part.queries = qs
        return part
    grids = [sp.rand_grid(rng, 6, 6) for _ in range(n_slots)]
    qs = []
    makers = (sp.lib_count, sp.lib_present, sp.lib_cell, sp.lib_colours) + ((sp.lib_row,) if grids_ok else ())
    while len(qs) < n_q:
        i = int(rng.integers(n_slots))
        qs.append(makers[len(qs) % len(makers)](rng, grids[i], i))
    slots = [Slot(g, encoding="compact") for g in grids]
    return Part("forced", slots, qs)


def _allowed_combos():
    for scheme, style in itertools.product(("none", "label", "ordinal", "role"), S.STYLES):
        sizes = [1] if scheme == "none" else ([2, 5] if scheme in ("ordinal", "role") else [1, 4])
        for n_slots in sizes:
            layouts = L.SINGLE_LAYOUTS if n_slots == 1 else L.MULTI_LAYOUTS
            if scheme == "role":
                layouts = L.MULTI_LAYOUTS
            for layout in layouts:
                yield scheme, n_slots, layout, style


def test_every_scheme_layout_style_combination():
    rng = np.random.default_rng(7)
    seen = set()
    for scheme, n_slots, layout, style in _allowed_combos():
        n_q = 1 if style == "bare" else (3 if style in S.MIN_ANSWERS else int(rng.integers(1, 4)))
        for multi in (False, True):
            part = _lib_part(rng, n_slots, n_q, role=scheme == "role",
                             grids_ok="grid" in S.KIND_MATRIX[style])
            parts = [part] + ([sp.gallery(rng, lib_only=True)] if multi else [])
            if multi and style == "bare" and n_q != 1:
                continue
            cfg = AssembleConfig(force={"scheme": scheme, "layout": layout, "style": style})
            row = assemble_row(rng, parts, config=cfg)
            m = row.meta["parts"][0]
            assert (m["scheme"], m["layout"], m["style"]) == (scheme, layout, style)
            seen.add((scheme, layout, style))
    expected = {(s, lay, st) for s, _, lay, st in _allowed_combos()}
    assert seen == expected and len(expected) == 9 * (3 + 7 + 5 + 5)      # "split" serves both grid counts


def test_every_delimiter_per_scheme():
    rng = np.random.default_rng(8)
    for scheme in ("none", "label", "ordinal", "role"):
        for delim in L.delimiters_for("label" if scheme == "role" else scheme):
            for dims in (False, True):
                part = _lib_part(rng, 1 if scheme == "none" else 3, 2, role=scheme == "role")
                cfg = AssembleConfig(force={"scheme": scheme, "delimiter": delim, "dims": dims})
                row = assemble_row(rng, [part], config=cfg)
                assert row.meta["parts"][0]["delimiter"] == delim
                # (the stub ARC part turns a slot's size header off when it drew a size question)
                if dims and scheme in ("label", "role") and all(s.dims_header_ok for s in part.slots):
                    assert "rows x" in row.prompt


def test_grid_answer_encodings_are_declared():
    rng = np.random.default_rng(9)
    for name in AssembleConfig().grid_answer_encodings:
        part = _lib_part(rng, 1, 2)
        part.queries = [sp.lib_row(rng, part.slots[0].grid, 0), sp.lib_count(rng, part.slots[0].grid, 0)]
        if name == "names":
            part.slots[0].encoding = "names"
        row = assemble_row(rng, [part], config=AssembleConfig(force={"style": "labelled", "grid_enc": name}))
        assert row.meta["parts"][0]["grid_answer_encoding"] == name
        if name != "compact":
            assert any(d in row.prompt for d in __import__("pv2.encodings").encodings.ENCODINGS[name].describe)


# --- fidelity ------------------------------------------------------------------------------------------------

def test_arc_part_is_byte_identical_to_the_stage2_task_block():
    ar = _arc_render()
    rng = np.random.default_rng(10)
    for base in (0, 1):
        part = sp.arc(rng, n_pairs=3, n_tests=2)
        for s in part.slots:
            s.encoding = "stage2"
            s.dims_header_ok = True
        part.context = []
        g = part.slots[-1].grid
        part.queries = [Query("arc", "arc:predict", (len(part.slots) - 1,), "Give the output grid for ⟪g0⟫.",
                              Answer("grid", [list(reversed(r)) for r in g]))]
        cfg = AssembleConfig(role_zero_based_p=1.0 if base == 0 else 0.0,
                             force={"delimiter": "label_line", "dims": True, "layout": "grids_first"})
        row = assemble_row(rng, [part], config=cfg)
        task = {"train": [{"input": part.slots[2 * k].grid, "output": part.slots[2 * k + 1].grid} for k in range(3)],
                "test": [{"input": part.slots[6].grid}, {"input": part.slots[7].grid}]}
        assert ar.task_block(task, zero_based=base == 0) in row.prompt


# --- contract errors -----------------------------------------------------------------------------------------

def test_contract_errors():
    rng = np.random.default_rng(11)
    lone = sp.lone_item(rng)
    assemble_row(rng, [lone])
    with pytest.raises(AssemblyError):
        assemble_row(rng, [lone, sp.reframed(rng)])
    g = sp.rand_grid(rng)
    io = sp.item_io(rng, g, 0)
    arc = sp.arc(rng)
    arc.queries.append(Query("item", io.qid, (0,), io.text, Answer("grid", [[1]]), flags=io.flags, lead_clause=True))
    with pytest.raises(AssemblyError):
        assemble_row(rng, [arc])
    grids = [sp.rand_grid(rng), sp.rand_grid(rng)]
    pl = sp.item_placement(rng, grids[0], 0)
    with pytest.raises(AssemblyError):
        assemble_row(rng, [Part("gallery", [Slot(x) for x in grids], [pl])])


def test_placement_and_nothing_else_parts():
    rng = np.random.default_rng(12)
    for _ in range(30):
        part = sp.placement_single(rng)
        row = assemble_row(rng, [part])
        assert row.meta["parts"][0]["layout"] == "q_first"
        pc = row.meta["checks"]["parts"][0]
        assert row.prompt.find(pc["questions"][0]["text"]) < pc["blocks"][0]["start"]
        row = assemble_row(rng, [sp.lone_item(rng)])
        assert row.meta["parts"][0]["style"] == "bare"
        assert row.target == parts_answer(row)


def parts_answer(row):
    return row.meta["checks"]["parts"][0]["questions"][0]["answer"]


def test_label_words_are_never_labels():
    rng = np.random.default_rng(13)
    for _ in range(200):
        g = [sp.rand_grid(rng, 5, 5) for _ in range(3)]
        q = sp.item_image(rng, g[1], 1)
        part = Part("gallery", [Slot(x) for x in g], [q, sp.lib_count(rng, g[0], 0)])
        row = assemble_row(rng, [part], config=AssembleConfig(scheme_multi={"label": 1.0, "ordinal": 0.0}))
        assert row.meta["parts"][0]["label_noun"] != "Image"


# --- wording flags from the items producer (pv2.wordings on main) ---------------------------------------

def _gallery_with(rng, maker, n=13):
    grids = [sp.rand_grid(rng, 4, 4) for _ in range(n)]
    i = int(rng.integers(n))
    q = maker(rng, grids[i], i)
    j = int(rng.integers(n))
    return Part("gallery", [Slot(g) for g in grids], [q, sp.lib_count(rng, grids[j], j)])


def test_flagged_letters_are_skipped_in_labels():
    rng = np.random.default_rng(20)
    seen_m = False
    for _ in range(150):
        part = _gallery_with(rng, sp.item_shape_letter)
        row = assemble_row(rng, [part], config=AssembleConfig(force={"scheme": "label", "label_style": "letter"}))
        labels = row.meta["checks"]["parts"][0]["labels"]
        assert not any(lab.endswith(" L") for lab in labels)
        seen_m |= any(lab.endswith(" M") for lab in labels)
    assert seen_m


def test_grid_number_wordings_keep_numeric_grid_labels_out_of_the_row():
    rng = np.random.default_rng(21)
    for _ in range(150):
        parts = [_gallery_with(rng, sp.item_tile, n=4), sp.gallery(rng, lib_only=True)]
        row = assemble_row(rng, parts, config=AssembleConfig(scheme_multi={"label": 1.0, "ordinal": 0.0}))
        labels = [lab for pc in row.meta["checks"]["parts"] for lab in pc["labels"] if lab]
        assert not any(re.fullmatch(r"Grid \d+", lab) for lab in labels)
        assert not any(re.fullmatch(r"(?:\w+ )?#?\d+", lab) for lab in row.meta["checks"]["parts"][0]["labels"])


def test_style_label_words_stay_out_of_reply_labels_and_tags():
    rng = np.random.default_rng(22)
    styles = Counter()
    for _ in range(300):
        parts = [_gallery_with(rng, sp.item_style_label, n=3), sp.one_grid_many(rng)]
        row = assemble_row(rng, parts)
        ch = row.meta["checks"]
        assert "Question" not in (ch["tag_format"] or "")
        pc = ch["parts"][0]
        styles[pc["style"]["style"]] += 1
        section = row.target.split("\n\n")[0]
        assert not re.search(r"^(?:Answer|Question)\b", "\n".join(section.split("\n")[1:]), re.M)
        for d in pc["questions"]:
            assert not d["text"].startswith("Question about")
    assert styles["labelled"] > 10


def test_inner_ordinal_wordings_are_never_referred_to_by_ordinals():
    rng = np.random.default_rng(23)
    for _ in range(150):
        part = _gallery_with(rng, sp.item_inner, n=5)
        row = assemble_row(rng, [part])
        assert row.meta["parts"][0]["scheme"] == "label"


def test_all_flags_together_still_pack():
    rng = np.random.default_rng(24)
    for _ in range(100):
        grids = [sp.rand_grid(rng, 4, 4) for _ in range(14)]
        qs = [m(rng, grids[k], k) for k, m in enumerate((sp.item_shape_letter, sp.item_tile, sp.item_inner,
                                                           sp.item_style_label))]
        row = assemble_row(rng, [Part("gallery", [Slot(g) for g in grids], qs)])
        labels = row.meta["checks"]["parts"][0]["labels"]
        assert all(re.fullmatch(r"\w+ [A-KM-Z]", lab) for lab in labels), labels


# --- follow-up fixes (verifier findings, 2026-09-29) ---------------------------------------------------

def _table_menu_part(rng):
    part = sp.arc(rng, n_pairs=3, n_tests=1)
    ins = [i for i, s in enumerate(part.slots) if s.role == "train_input"]
    part.context = ["Each training input with its value:"] + [f"⟪G{s}⟫ -> {3 + k}" for k, s in enumerate(ins)] + \
        ["Possible rules:", "A. Count the cells of color 1.", "B. Count the rows.", "C. Count the columns."]
    return part


def test_context_lines_keep_their_own_lines():
    rng = np.random.default_rng(30)
    for _ in range(100):
        parts = [_table_menu_part(rng)] + ([sp.gallery(rng)] if rng.random() < 0.5 else [])
        row = assemble_row(rng, parts)
        lines = row.prompt.split("\n")
        for want in ("Each training input with its value:", "Possible rules:", "A. Count the cells of color 1.",
                     "B. Count the rows.", "C. Count the columns."):
            assert want in lines
        assert sum(1 for line in lines if re.fullmatch(r"(?:[Tt]his \w+'s )?Training pair \d+ input -> \d+", line)) == 3


def test_real_arc_value_tables_and_menus_stay_lists():
    from pv2.arc import bundle
    rng = np.random.default_rng(31)
    seen = 0
    for _ in range(40):
        part = bundle.arc_part(rng, mode="scalar")
        row = assemble_row(rng, [part])
        a, b = row.meta["checks"]["parts"][0]["span"]
        lines = row.prompt[a:b].split("\n")
        # value-table lines only: the chain answer style's own skeleton ("Q1 -> <answer> | Q2 -> ...")
        # legitimately carries several arrows on one line
        arrows = [line for line in lines if " -> " in line and "<answer>" not in line and "<grid>" not in line
                  and "[grid]" not in line]
        seen += len(arrows)
        assert all(line.count(" -> ") == 1 for line in arrows)
    assert seen >= 40 * 3


def test_tags_parts_never_show_a_grid_tag_placeholder():
    rng = np.random.default_rng(32)
    for style in ("labelled", "json", "restate"):
        for _ in range(40):
            part = _lib_part(rng, 3, 2)
            g = part.slots[0].grid
            part.queries = [sp.lib_row(rng, g, 0), sp.lib_count(rng, g, 0)]
            row = assemble_row(rng, [part], config=AssembleConfig(
                scheme_multi={"label": 1.0, "ordinal": 0.0}, force={"delimiter": "tags", "style": style}))
            pc = row.meta["checks"]["parts"][0]
            assert pc["delimiter"] == "tags" and pc["style"]["style"] == style
            assert "<grid>" not in pc["instructions"]
            if "exactly like this" in pc["instructions"] or style != "labelled":   # a skeleton is shown
                assert "[grid]" in pc["instructions"]
            assert row.prompt.count("<grid>") == row.prompt.count("</grid>") == 3
    # other delimiters keep the usual placeholder
    part = _lib_part(rng, 1, 2)
    part.queries = [sp.lib_row(rng, part.slots[0].grid, 0), sp.lib_count(rng, part.slots[0].grid, 0)]
    row = assemble_row(rng, [part], config=AssembleConfig(force={"delimiter": "fenced", "style": "labelled"}))
    assert "[grid]" not in row.prompt


def test_reminder_prints_letter_tags_in_parentheses():
    for seed in range(50):
        rng = np.random.default_rng(seed)
        assert L.reminder_line(rng, ["a"]) in ("Now answer question (a).", "Now answer the question.")
        assert L.reminder_line(rng, ["b", "c"]) in ("Now answer questions (b) and (c).", "Now answer the 2 questions.")
        assert L.reminder_line(rng, ["4"]) in ("Now answer question 4.", "Now answer the question.")


def test_one_arc_numbering_base_per_row():
    # the default prints the verbatim stage-2 labels 0-based only (the live solver arms' base); a
    # config that allows both still keeps one base per row
    for p, expected in ((None, {0}), (0.5, {0, 1})):
        rng = np.random.default_rng(33)
        cfg = AssembleConfig() if p is None else AssembleConfig(role_zero_based_p=p)
        both = set()
        for _ in range(150):
            parts = [sp.arc(rng), sp.reframed(rng), sp.arc(rng), sp.arc(rng)]
            row = assemble_row(rng, parts, config=cfg)
            bases = {m["base"] for m in row.meta["parts"] if m["scheme"] == "role"}
            assert len(bases) == 1
            both |= bases
        assert both == expected


def test_bare_parts_under_a_heading_claim_no_exclusivity():
    rng = np.random.default_rng(34)
    seen = Counter()
    for _ in range(200):
        g = sp.rand_grid(rng)
        lone = Part("reframed", [Slot(g)], [sp.lib_count(rng, g, 0)])
        row = assemble_row(rng, [lone, sp.gallery(rng)], config=AssembleConfig(force={"style": "bare"}))
        pc = row.meta["checks"]["parts"][0]
        assert pc["style"]["style"] == "bare"
        assert not re.search(r"alone|no other text|nothing else", pc["instructions"])
        seen[pc["instructions"].split(", ", 1)[-1].split(": ", 1)[-1].lower()] += 1
    assert len(seen) == 3
    # a single-part row keeps all its wordings
    words = set()
    for _ in range(60):
        g = sp.rand_grid(rng)
        row = assemble_row(rng, [Part("reframed", [Slot(g)], [sp.lib_count(rng, g, 0)])],
                           config=AssembleConfig(force={"style": "bare"}))
        words.add(row.meta["checks"]["parts"][0]["instructions"])
    assert any("alone" in w for w in words) and any("no other text" in w for w in words)


def test_question_tags_are_never_capital_letters(sample):
    for _, row in sample[:1500]:
        ch = row.meta["checks"]
        for pc in ch["parts"]:
            for d in pc["questions"]:
                if d["tag"] is None:
                    continue
                assert d["tag"].isdigit() or (len(d["tag"]) == 1 and d["tag"].islower())
                printed = ch["tag_format"].format(t=d["tag"])
                assert printed + " " + d["text"] in row.prompt
                assert not re.match(r"\(?[A-Z][.):]", printed)
    for kind in L.QUESTION_TAGS:
        for fmt in L.QUESTION_TAGS[kind]:
            assert not re.fullmatch(r"\(?[A-Z][.):]*", fmt.format(t="a"))
