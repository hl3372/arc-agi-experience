"""Declared sentences (DESIGN 9, 13.7, 15): a share of sentence / paragraph parts writes every
question's exact, answer-neutral sentence into the prompt.

(a) every declared form -- library phrasings, the verification and chain phrasings, ARC builders -- is
    the same whatever the answer (case, plural, yes / no, none);
(b) in declared parts the target sentence is the declared text with the answer in place, the style
    parser round-trips and eval_v2 gives the target full marks;
(c) the validator refuses a declared part whose target sentence differs from its declaration;
(d) the prose share of answers does not move;
(e) rows build with 0 failures in the default and the reading / compact configurations.
"""
import copy
import re
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
import pytest

from pv2 import families as FAM
from pv2.arc import questions as Q
from pv2.assemble import AssembleConfig, assemble_row
from pv2.assemble import styles as S
from pv2.eval_v2 import score
from pv2.library import QUESTIONS, pose
from pv2.library import parts as LP
from pv2.library.core import (NEUTRAL_ALWAYS, SAME_VALUE, Phrasing, declared_template, inflect,
                              neutral_template)
from pv2.model import Part, Slot, fill
from pv2.validate import ValidationError, validate_row

FIELD = re.compile(r"\{(\w+)\}")
PLACEHOLDER = re.compile(r"⟪([gG]\d+s?|answer)⟫")


# --- (a) answer-neutral forms ----------------------------------------------------------------------------

def _op_phrasings():
    """The verification / choice phrasings over every operation kind, and the chain phrasings in all their
    variants (same grid, 0-indexed sentence added)."""
    rng = np.random.default_rng(5)
    g = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]])
    cw = LP._cw(False)
    out = []
    for kind in LP.OP_KINDS:
        op = None
        while op is None:
            op = LP.make_op(rng, g, kind, cw)
        out += [(f"verify:{kind}", p, "bool") for p in LP.VERIFY_IS(op)]
        out += [(f"verify_diff:{kind}", p, "number") for p in LP.VERIFY_DIFF(op)]
        out += [(f"verify_redo:{kind}", p, "grid") for p in LP.VERIFY_REDO(op)]
        out += [(f"choice:{kind}", p, "token") for p in LP.CHOICE(op, "⟪g1⟫ and ⟪g2⟫", "⟪g1⟫ or ⟪g2⟫")]
    for table, kinds in ((LP.CHAIN_LINE, {"row": "grid", "col": "grid", "top_rows": "grid"}),
                         (LP.CHAIN_COLOUR, {"count": "number", "has": "bool", "recolour": "grid",
                                            "rows_containing": "list"})):
        for hop2, phrasings in table.items():
            for variant in (list(phrasings), LP._same_grid(phrasings), LP._with_zero(phrasings),
                            LP._same_grid(LP._with_zero(phrasings))):
                out += [(f"chain:{hop2}", p, kinds[hop2]) for p in variant]
    return out


def _all_phrasings():
    for spec in QUESTIONS.values():
        for p in spec.phrasings:
            yield spec.qid, p, spec.kind
    yield from _op_phrasings()


def test_every_phrasing_has_a_declared_form_that_the_answer_cannot_change():
    """Template check: every phrasing has one declared STRING (never a per-case dict) using only fields its
    question states, so formatting it with any answer's fields (1 cell / 5 cells, is / are, a chain's k and
    X, found / none) gives one text; grid answers declare a lead-in ending in ':'."""
    n = 0
    for qid, p, kind in _all_phrasings():
        form = declared_template(p)
        assert isinstance(form, str), (qid, p.text)
        assert neutral_template(form, p.text), (qid, form)
        if kind == "grid":
            assert "⟪answer⟫" not in form and form.endswith(":"), (qid, form)
        else:
            assert form.count("⟪answer⟫") == 1, (qid, form)
        # every grid the declared sentence names is one the question names
        said = set(PLACEHOLDER.findall(p.text)) | {"answer"}
        used = set(PLACEHOLDER.findall(form))
        assert {u.lower().rstrip("s") for u in used} <= {s.lower().rstrip("s") for s in said} | {"answer"} \
            or "{refs}" in p.text or "{Refs}" in p.text, (qid, form)
        # one text whatever the answer: the question's own fields fixed, every answer field varied
        base = {f: f"<{f}>" for f in FIELD.findall(p.text) + list(NEUTRAL_ALWAYS)}
        for group in SAME_VALUE:
            if group & set(base):
                base.update({f: "<same>" for f in group})
        texts = set()
        for count in (0, 1, 2, 7):
            answer_fields = {**inflect(count), "k": count, "idx": count - 1, "Xw": f"color {count}",
                             "n": count, "case": count}
            texts.add(form.format(**{**answer_fields, **base}))
        assert len(texts) == 1, (qid, texts)
        n += 1
    assert n >= 245 + 8 * (4 + 3 + 2 + 4)


def test_declared_fields_are_policed():
    """A declared form naming a field its question does not state is refused (never reaches a row)."""
    bad = Phrasing("terse", "⟪G0⟫: number of rows?", "⟪G0⟫ has ⟪answer⟫ {rows}.",
                   declared="⟪G0⟫ has ⟪answer⟫ {rows}.")
    with pytest.raises(RuntimeError, match="does not state"):
        declared_template(bad)
    per_case = Phrasing("terse", "Q ⟪g0⟫?", {"yes": "Yes ⟪answer⟫.", "no": "No ⟪answer⟫."})
    assert declared_template(per_case) is None
    assert declared_template(Phrasing("terse", "Q ⟪g0⟫?", "⟪G0⟫ has ⟪answer⟫ {cells}.")) is None


def _grids(rng, n):
    out = []
    for _ in range(n):
        h, w = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        k = int(rng.integers(1, 5))
        colours = rng.choice(10, size=k, replace=False)
        g = colours[rng.integers(0, k, size=(h, w))]
        if rng.random() < 0.3:
            g = np.concatenate([g, g[:, ::-1]], axis=1)       # symmetric ones (yes answers)
        out.append([[int(v) for v in row] for row in g])
    return out


def test_library_declared_text_is_a_function_of_the_question_text():
    """Empirical check over many draws: one question text, one declared text, while the answers (and the
    statements, which are inflected) vary."""
    rng = np.random.default_rng(11)
    by_text = defaultdict(set)
    answers = defaultdict(set)
    statements = defaultdict(set)
    grids = _grids(rng, 160)
    for i in range(1600):
        a, b = grids[i % 160], grids[(i * 7 + 3) % 160]
        if rng.random() < 0.3:
            b = copy.deepcopy(a)
            if rng.random() < 0.5:
                b[0][0] = (b[0][0] + 1) % 10
        slots = [Slot(a, "compact"), Slot(b, "compact"), Slot(grids[(i * 13 + 5) % 160], "compact")]
        reads = pick_reads(rng)
        q = pose(rng, slots, reads)
        if q is None:
            continue
        key = (q.qid, q.params.get("phrasing"), q.text)
        by_text[key].add(q.declared)
        answers[key].add(q.answer.text if q.answer.text is not None else repr(q.answer.value))
        statements[key].add(q.statement)
    assert all(len(v) == 1 and None not in v for v in by_text.values()), \
        [(k, v) for k, v in by_text.items() if len(v) != 1][:5]
    varied = [k for k in by_text if len(answers[k]) > 1]
    inflected = [k for k in by_text if len(statements[k]) > 1]
    assert len(varied) >= 150 and len(inflected) >= 40, (len(varied), len(inflected))
    assert len({k[0] for k in inflected}) >= 15


def pick_reads(rng):
    r = rng.random()
    if r < 0.6:
        return (0,)
    if r < 0.85:
        return (0, 1)
    return (0, 1, 2)


def _arc_pairs():
    """(builder name, query for answer case A, query for case B) with equal parameters and rng state."""
    one = [[1, 2], [3, 3]]
    many = [[1, 1], [1, 1]]
    none_ = [[2, 2], [3, 3]]
    other = [[4, 5], [6, 7]]
    r = lambda: np.random.default_rng(3)  # noqa: E731
    yield "predict_value", Q.predict_scalar(r(), 0, "number", 1), Q.predict_scalar(r(), 0, "number", 12)
    yield "predict_value:color", Q.predict_scalar(r(), 0, "color", 1), Q.predict_scalar(r(), 0, "color", 7)
    yield "identify", Q.identify(r(), (0, 1), "A"), Q.identify(r(), (0, 1), "C")
    yield "verify", Q.verify(r(), 2, 1, True), Q.verify(r(), 2, 1, False)
    yield "verify_diff", Q.verify_diff(r(), 2, 1, 1), Q.verify_diff(r(), 2, 1, 6)
    yield "verify_value", Q.verify_scalar(r(), 0, 5, True), Q.verify_scalar(r(), 0, 5, False)
    yield "verify_value:color", Q.verify_scalar(r(), 0, 5, True, "color"), Q.verify_scalar(r(), 0, 5, False, "color")
    for x_grid, y_grid in ((one, many), (none_, many), (one, none_)):
        yield "count_color", Q.count_color(r(), 0, x_grid, 1, "compact"), Q.count_color(r(), 0, y_grid, 1, "compact")
    yield "n_colors", Q.distinct_colors(r(), 0, many), Q.distinct_colors(r(), 0, other)
    yield "mode_color", Q.most_frequent(r(), 0, many, "compact"), Q.most_frequent(r(), 0, [[3, 3], [3, 1]], "compact")
    yield "dims", Q.dims(r(), 0, many), Q.dims(r(), 0, [[1, 2, 3]])
    yield "color_at", Q.color_at(r(), 0, many), Q.color_at(r(), 0, other)
    yield "changed_cells", Q.changed_cells(r(), 0, 1, many, one), Q.changed_cells(r(), 0, 1, many, [[1, 1], [1, 2]])
    yield "identical", Q.identical(r(), 0, 1, many, many), Q.identical(r(), 0, 1, many, other)


def test_arc_declared_forms_do_not_depend_on_the_answer():
    for name, a, b in _arc_pairs():
        assert a.text == b.text, name
        assert a.answer.text != b.answer.text, name                  # the two cases really differ
        assert a.declared is not None and a.declared == b.declared, (name, a.declared, b.declared)
        assert a.declared.count("⟪answer⟫") == 1, name
    # the inflected ones do differ in their statements (the case the declared form exists for)
    differing = {name for name, a, b in _arc_pairs() if a.statement != b.statement}
    assert {"verify", "verify_diff", "verify_value", "count_color", "n_colors", "changed_cells",
            "identical"} <= differing


def test_arc_parts_declare_every_statement_they_have():
    """Every ARC query with a statement form carries a declared form (grid answers and name / per-grid
    lists have neither and use the generic form)."""
    from pv2.arc import bundle as B
    seen = Counter()
    for seed in range(60):
        part = B.build(seed, questions="reading" if seed % 2 else "all")
        for q in part.queries:
            seen[q.qid] += 1
            if q.statement is not None:
                assert q.declared is not None, q.qid
            if q.declared is not None:
                assert q.declared.count("⟪answer⟫") == 1 or (q.answer.kind == "grid" and q.declared.endswith(":"))
    assert len(seen) >= 8, seen


# --- (b) declared parts --------------------------------------------------------------------------------------

def _part(rng, k):
    maker = k % 6
    if maker == 0:
        return LP.library_part(rng, n_questions=int(rng.integers(2, 6)))
    if maker == 1:
        return LP.comparison_part(rng)
    if maker == 2:
        return LP.chain_part(rng)
    if maker == 3:
        n = int(rng.integers(2, 6))
        return FAM.gallery_part(rng, n_grids=n, n_asked=int(rng.integers(1, n + 1)), max_cells=300)
    if maker == 4:
        return FAM.one_grid_part(rng, n_questions=int(rng.integers(2, 5)), max_cells=300)
    return FAM.arc_part(rng, questions="reading" if k % 12 == 5 else "all")


@pytest.fixture(scope="module")
def declared_rows():
    rows = []
    for k in range(180):
        rng = np.random.default_rng(900 + k)
        part = _part(rng, k)
        if part is None:
            continue
        style = "paragraph" if k % 3 == 0 else "sentence"
        cfg = AssembleConfig(force={"style": style}, declare_statements_p=1.0)
        row = assemble_row(rng, [part], config=cfg)
        rows.append((part, row))
    return rows


def _expected_sentence(q, refs, answer, shape):
    """(the target sentence, the text the prompt states for it), rebuilt from the producer's declared form
    alone (a question without one: from the shape's generic form)."""
    if q.declared is None or shape.generic:
        assert q.declared is None or (shape.grid and "⟪answer⟫" in q.declared), q.qid
        if shape.grid:
            return shape.prefix + "\n" + answer, shape.prefix
        return shape.prefix + answer + shape.suffix, shape.prefix + "<answer>" + shape.suffix
    if shape.grid:
        lead = fill(q.declared, refs).rstrip()
        return lead + "\n" + answer, lead
    return fill(q.declared, refs, answer=answer), fill(q.declared, refs, answer="<answer>")


def test_declared_parts_answer_with_the_stated_sentences(declared_rows):
    prose = [(p, r) for p, r in declared_rows if r.meta["parts"][0]["style"] in ("sentence", "paragraph")]
    assert len(prose) >= 120, len(prose)
    kinds, sources = Counter(), Counter()
    for part, row in prose:
        pc = row.meta["checks"]["parts"][0]
        spec = S.StyleSpec.from_dict(pc["style"])
        assert spec.variant.get("declared") is True
        answers = [d["answer"] for d in pc["questions"]]
        sentences = []
        for d, shape in zip(pc["questions"], spec.shapes):
            q = part.queries[d["query"]]
            refs = [pc["phrases"][i] for i in q.reads]
            sentence, stated = _expected_sentence(q, refs, d["answer"], shape)
            sentences.append(sentence)
            # the prompt states this very sentence, with <answer> in the answer's place
            assert f'"{stated}"' in row.prompt, (q.qid, stated)
            kinds[q.answer.kind] += 1
            sources[(q.source, q.declared is not None)] += 1
        sep = " " if spec.style == "paragraph" else ("\n\n" if any(s.multiline for s in spec.shapes) else "\n")
        assert row.target == sep.join(sentences)
        assert S.parse(spec, row.target) == answers
        sc = score({"id": "x", "target": row.target, "checks": row.meta["checks"]}, row.target)
        assert sc["strict"] and sc["lenient"] == 1.0 and sc["all_ok"]
    assert {"number", "bool", "list", "grid", "color"} <= set(kinds), kinds
    assert sources[("lib", True)] > 100 and sources[("arc", True)] > 10 and sources[("item", False)] > 5, sources


def test_undeclared_parts_keep_their_sentences_and_show_an_example():
    for k in range(40):
        rng = np.random.default_rng(1900 + k)
        part = LP.library_part(rng, n_questions=3)
        cfg = AssembleConfig(force={"style": "sentence"}, declare_statements_p=0.0)
        row = assemble_row(rng, [part], config=cfg)
        pc = row.meta["checks"]["parts"][0]
        if pc["style"]["style"] != "sentence":
            continue
        spec = S.StyleSpec.from_dict(pc["style"])
        assert "declared" not in spec.variant
        for d, shape in zip(pc["questions"], spec.shapes):
            q = part.queries[d["query"]]
            if not shape.generic:                                  # today's content-aware statement
                refs = [pc["phrases"][i] for i in q.reads]
                assert shape.prefix + d["answer"] + shape.suffix == fill(q.statement, refs, answer=d["answer"])
        if not all(s.generic for s in spec.shapes):
            assert any(ex in pc["instructions"] for ex in S.SENTENCE_EXAMPLES)
        else:
            assert not any(ex in pc["instructions"] for ex in S.SENTENCE_EXAMPLES)


# --- (c) the validator -----------------------------------------------------------------------------------------

def _tampered(row, pi, qi, new_prefix, *, instruction_too=False):
    row2 = copy.deepcopy(row)
    pc = row2.meta["checks"]["parts"][pi]
    spec = S.StyleSpec.from_dict(pc["style"])
    old = spec.shapes[qi].prefix
    spec.shapes[qi].prefix = new_prefix
    pc["style"] = spec.to_dict()
    row2.target = S.render(spec, [d["answer"] for d in pc["questions"]])     # V4 stays happy
    if instruction_too:
        assert len(new_prefix) == len(old)
        row2.prompt = row2.prompt.replace(f'"{old}', f'"{new_prefix}')
        pc["instructions"] = pc["instructions"].replace(f'"{old}', f'"{new_prefix}')
    return row2


def test_validator_refuses_a_sentence_that_differs_from_its_declaration(declared_rows):
    checked = 0
    for part, row in declared_rows:
        pc = row.meta["checks"]["parts"][0]
        if pc["style"]["style"] not in ("sentence", "paragraph"):
            continue
        spec = S.StyleSpec.from_dict(pc["style"])
        qi = next((i for i, s in enumerate(spec.shapes) if not s.generic and s.prefix.startswith("The ")), None)
        if qi is None:
            continue
        validate_row(row, [part])
        # the target says something else than the prompt declared
        bad = _tampered(row, 0, qi, spec.shapes[qi].prefix.replace("The ", "Its ", 1))
        with pytest.raises(ValidationError, match="stated sentence") as e:
            validate_row(bad, [part])
        assert e.value.check == "V5"
        # prompt and target agree, but not with the question's declared form
        bad = _tampered(row, 0, qi, spec.shapes[qi].prefix.replace("The ", "Thy ", 1), instruction_too=True)
        with pytest.raises(ValidationError, match="declared form") as e:
            validate_row(bad, [part])
        checked += 1
    assert checked >= 60, checked


def test_validator_refuses_an_undeclared_part_whose_generic_sentence_differs():
    checked = 0
    for k in range(60):
        rng = np.random.default_rng(3000 + k)
        part = FAM.reframed_part(rng, multi_answer=True, max_cells=200)
        row = assemble_row(rng, [part], config=AssembleConfig(force={"style": "sentence"}, declare_statements_p=0.0))
        spec = S.StyleSpec.from_dict(row.meta["checks"]["parts"][0]["style"])
        if spec.style != "sentence" or not spec.shapes[0].generic:
            continue
        validate_row(row, [part])
        bad = _tampered(row, 0, 0, spec.shapes[0].prefix + "then ")
        with pytest.raises(ValidationError, match="stated sentence"):
            validate_row(bad, [part])
        checked += 1
    assert checked >= 5, checked


# --- (d) the prose share -----------------------------------------------------------------------------------------

def test_prose_share_is_unchanged():
    """The style is drawn before the declared / undeclared draw, so the same parts assembled from the same rng
    state pick the same styles whether sentences are declared or not (up to round-trip refusals)."""
    counts = {p: Counter() for p in (0.0, 0.5)}
    for k in range(240):
        part_rng = np.random.default_rng(5000 + k)
        part = _part(part_rng, k)
        if part is None:
            continue
        for p in (0.0, 0.5):
            row = assemble_row(np.random.default_rng(7000 + k), [copy.deepcopy(part)],
                               config=AssembleConfig(declare_statements_p=p))
            for pm in row.meta["parts"]:
                n = pm["questions"]
                counts[p]["all"] += n
                counts[p]["prose"] += n if pm["style"] in ("sentence", "paragraph") else 0
    share = {p: c["prose"] / c["all"] for p, c in counts.items()}
    assert share[0.0] > 0.15, share
    assert abs(share[0.5] - share[0.0]) <= 0.01, share


# --- (e) builds -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reading", [False, True])
def test_rows_build_without_failures(reading):
    from pv2 import build_dataset as B
    from _tok import tokenizer_or_skip
    cfg = B.BuildConfig(classic_share=0.0)
    if reading:
        cfg.arc["questions"] = "reading"
        cfg.assemble["single_grid_encoding"] = "compact"
    tok = tokenizer_or_skip()
    examples, stats, failures = B.build(80, 4242 + reading, cfg, tok, log_every=0)
    assert not failures, failures
    assert len(examples) == 80
