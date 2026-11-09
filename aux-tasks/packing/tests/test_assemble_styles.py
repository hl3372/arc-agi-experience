"""Answer styles: the style x kind matrix, render/parse round trips, exact skeletons, signatures."""
import itertools

import numpy as np
import pytest

from pv2.assemble import styles as S

SAMPLES = {
    "number": ["7", "0", "123"],
    "color": ["3", "green"],
    "bool": ["yes", "no", "true"],
    "token": ["clockwise", "4-connected"],
    "text": ["5x7", "(1,-2)", "first:[(0,1),(0,2)], second:[(3,3)]", "[(0,1),(2,3)]"],
    "list": ["1, 3, 5", "the third grid and the fifth grid", "none", "Grid A, Grid C"],
    "grid": ["0120\n0001", "5", "[[1,2],[3,4]]"],
    "multiline": ["horizontal_reflection: yes\nvertical_reflection: no"],
}
JSON_GRIDS = ["[[0,1,2,0],[0,0,0,1]]", "[[5]]"]      # a JSON-style grid answer is the json_flat rendering


def _samples(style, cls):
    return JSON_GRIDS if style == "json" and cls == "grid" else SAMPLES[cls]


def _shape(tag, cls, text, style):
    kind = "text" if cls == "multiline" else cls
    grid = cls == "grid"
    s = S.Shape(tag=tag, kind=kind, grid=grid, multiline=grid or "\n" in text,
                json_type="grid" if grid else ("number" if cls == "number" and text.isdigit() else "string"))
    if style in ("sentence", "paragraph"):
        s.prefix, s.suffix = (f"Row 1 of Grid {tag} is:", "") if grid else (f"The answer for Grid {tag} is ", ".")
    if style == "restate":
        s.question = f"What is asked in question {tag}?"
    return s


def _spec(rng, style, classes, texts, tags):
    shapes = [_shape(t, c, x, style) for t, c, x in zip(tags, classes, texts)]
    return S.StyleSpec(style, shapes, S.choose_variant(rng, style, shapes))


def test_matrix_is_complete_and_explicit():
    assert set(S.KIND_MATRIX) == set(S.STYLES)
    for style in S.STYLES:
        assert S.KIND_MATRIX[style] <= set(S.ANSWER_CLASSES)
    for cls in S.ANSWER_CLASSES:
        carriers = [s for s in S.STYLES if cls in S.KIND_MATRIX[s]]
        assert {"bare", "labelled", "json"} <= set(carriers), cls
    # grids and multi-line answers never go into one-line styles
    for style in ("paragraph", "kv", "chain", "bullets"):
        assert not {"grid", "multiline"} & S.KIND_MATRIX[style]
    assert S.STRUCTURED_OK == {"bare", "labelled", "json"}
    assert S.NOTHING_ELSE_OK == {"bare"}


def test_part_rules():
    one = ["number"]
    assert S.style_allows("bare", one, structured=[False], nothing_else=[False], restate_ok=True)
    assert not S.style_allows("bare", one * 2, structured=[False] * 2, nothing_else=[False] * 2, restate_ok=True)
    for style in ("chain", "bullets", "paragraph"):
        assert not S.style_allows(style, one, structured=[False], nothing_else=[False], restate_ok=True)
    assert not S.style_allows("restate", one, structured=[False], nothing_else=[False], restate_ok=False)
    for style in S.STYLES:
        ok = S.style_allows(style, ["text"], structured=[True], nothing_else=[False], restate_ok=True)
        assert ok == (style in S.STRUCTURED_OK and style not in S.MIN_ANSWERS)
        ok = S.style_allows(style, ["text"], structured=[False], nothing_else=[True], restate_ok=True)
        assert ok == (style == "bare")


@pytest.mark.parametrize("style", S.STYLES)
def test_every_allowed_combination_round_trips(style):
    rng = np.random.default_rng(0)
    for cls in S.KIND_MATRIX[style]:
        for text in _samples(style, cls):
            for n in (1, 2, 3):
                if not S.style_allows(style, [cls] * n, structured=[cls == "multiline"] * n,
                                      nothing_else=[False] * n, restate_ok=True):
                    continue
                for tags in ([None] if n == 1 else []) + [[str(i + 4) for i in range(n)],
                                                          [chr(97 + i) for i in range(n)]]:
                    tags = tags if isinstance(tags, list) else [tags]
                    for _ in range(6):
                        spec = _spec(rng, style, [cls] * n, [text] * n, tags)
                        assert S.round_trips(spec, [text] * n), (style, cls, text, spec.variant)


@pytest.mark.parametrize("style", S.STYLES)
def test_mixed_answers_round_trip(style):
    rng = np.random.default_rng(1)
    classes = [c for c in S.KIND_MATRIX[style] if c != "multiline"]
    for _ in range(200):
        n = int(rng.integers(S.MIN_ANSWERS.get(style, 1), S.MAX_ANSWERS.get(style, 6) + 1))
        cls = [classes[int(rng.integers(len(classes)))] for _ in range(n)]
        texts = [_samples(style, c)[int(rng.integers(len(_samples(style, c))))] for c in cls]
        tags = [None] if n == 1 and rng.random() < 0.5 else [str(i + 1) for i in range(n)]
        spec = _spec(rng, style, cls, texts, tags)
        target = S.render(spec, texts)
        if style == "chain" and any("; " in t or " | " in t for t in texts):
            continue            # the assembler's round trip rejects such variants
        assert S.parse(spec, target) == texts, (style, spec.variant, target)


def test_parser_rejects_deviations():
    rng = np.random.default_rng(2)
    spec = _spec(rng, "labelled", ["number", "grid"], ["7", "01\n10"], ["1", "2"])
    spec.variant["label"] = "Answer {t}:"
    good = S.render(spec, ["7", "01\n10"])
    assert good == "Answer 1: 7\nAnswer 2:\n01\n10"
    for bad in ("Answer 1: 7\nAnswer 3:\n01\n10", "Answer 1: 7", good + "\nextra", "Answer 1 7\nAnswer 2:\n01\n10"):
        try:
            got = S.parse(spec, bad)
        except ValueError:
            continue
        assert got != ["7", "01\n10"]          # the validator compares with the gold (V4)
    js = _spec(rng, "json", ["number", "text"], ["7", "5x7"], ["1", "2"])
    js.variant.update(key="q{t}", layout="inline")
    assert S.render(js, ["7", "5x7"]) == '{"q1": 7, "q2": "5x7"}'
    for bad in ('{"q1": "7", "q2": "5x7"}', '{"q2": "5x7", "q1": 7}', '{"q1": 7}', '{"q1": 7, "q1": 7, "q2": "x"}'):
        with pytest.raises(ValueError):
            S.parse(js, bad)


def test_paragraph_is_statements_joined_by_single_spaces():
    rng = np.random.default_rng(3)
    spec = _spec(rng, "paragraph", ["number", "bool"], ["4", "yes"], ["1", "2"])
    assert S.render(spec, ["4", "yes"]) == "The answer for Grid 1 is 4. The answer for Grid 2 is yes."


def test_sentence_grid_answers_stand_alone():
    rng = np.random.default_rng(4)
    spec = _spec(rng, "sentence", ["number", "grid"], ["4", "01\n10"], ["1", "2"])
    target = S.render(spec, ["4", "01\n10"])
    assert target == "The answer for Grid 1 is 4.\n\nRow 1 of Grid 2 is:\n01\n10"
    for line in target.split("\n"):
        assert not (any(ch.isdigit() for ch in line) and any(ch.isalpha() for ch in line)) or "Grid" in line


def test_skeletons_are_the_rendering_with_placeholders():
    rng = np.random.default_rng(5)
    for style in ("json", "kv", "chain", "bullets", "labelled", "restate"):
        cls = ["number", "text"] if style != "labelled" else ["number", "grid"]
        spec = _spec(rng, style, cls, ["1", "x"], ["3", "4"])
        sk = S.skeleton(spec)
        assert "<answer>" in sk or "<number>" in sk
        if style in ("kv", "chain", "bullets"):
            assert S.parse(spec, sk) == ["<answer>", "<answer>"]


def _all_instructions():
    rng = np.random.default_rng(6)
    for style in S.STYLES:
        for n, grid, tags_kind in itertools.product((1, 2, 3), (False, True), ("number", "letter", None)):
            if tags_kind is None and n != 1:
                continue
            cls = ["grid" if grid and i == 0 else "number" for i in range(n)]
            if not S.style_allows(style, cls, structured=[False] * n, nothing_else=[False] * n, restate_ok=True):
                continue
            tags = [None] if tags_kind is None else ([str(i + 2) for i in range(n)] if tags_kind == "number"
                                                     else [chr(98 + i) for i in range(n)])
            for _ in range(12):
                spec = _spec(rng, style, cls, ["1"] * n, tags)
                if style in ("sentence", "paragraph"):
                    spec.shapes[0].generic = bool(rng.random() < 0.5)
                    if rng.random() < 0.5:
                        spec.variant["declared"] = True       # every sentence stated (else: the example)
                yield style, S.instruction(rng, spec)


def test_signatures_identify_each_style_exactly():
    seen = set()
    for style, text in _all_instructions():
        hits = {s for s, sig in S.SIGNATURES.items() if sig.search(text)}
        assert hits == {style}, (style, text, hits)
        assert not S.BARE_ONLY_PHRASES.search(text) or style == "bare", text
        assert "above" not in text and "below" not in text
        seen.add(style)
    assert seen == set(S.STYLES)
