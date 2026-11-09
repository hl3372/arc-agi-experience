"""Questions asked inside ARC parts: solve, identify the rule, verify, and reading questions about
one grid, the train inputs, the train outputs or every grid of the bundle.

Every phrasing has exactly one statement form (DESIGN 13.7), written for the actual answer
(pluralised, negated) and repeating the question's reference placeholders.  Grid answers carry no
statement (the assembler never puts a grid on a line with prose).  Answers that name grids write
placeholders (``"⟪g0⟫ and ⟪g2⟫"`` / ``"none"``) with the part's slot indices in ``value``.

``Query.declared`` is the answer-neutral form a declared sentence / paragraph part writes into the
prompt: the statement itself where nothing in it depends on the answer, else a neutral rewrite ("The
number of cells of color 3 in ⟪g0⟫ is ⟪answer⟫."); a builder without one leaves it None (the part then
declares the generic form).  It is built from the builder's parameters only, never from the answer.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pv2.encodings import colour_word
from pv2.model import FLAG_COORDS, FLAG_DIMS, Answer, Query


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def _cells(n: int) -> str:
    return "1 cell" if n == 1 else f"{n} cells"


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _ph(n: int) -> list[str]:
    return [f"⟪g{i}⟫" for i in range(n)]


# ---------------------------------------------------------------------------------------------
# solving, rules, verification
# ---------------------------------------------------------------------------------------------

PREDICT = (
    "What is the output grid for ⟪g0⟫?",
    "Apply the same rule to ⟪g0⟫ and give the resulting grid.",
    "Predict the output for ⟪g0⟫.",
    "Which grid does the rule produce from ⟪g0⟫?",
    "Solve ⟪g0⟫: give its output grid.",
    "Transform ⟪g0⟫ the way every training input was transformed. What do you get?",
)


def predict(rng, test_slot: int, grid) -> Query:
    return Query("arc", "arc:predict", (test_slot,), _pick(rng, PREDICT), Answer("grid", grid))


SCALAR_PREDICT = {
    "number": ("What number does the rule give for ⟪g0⟫?", "Which value belongs to ⟪g0⟫?",
               "Following the same rule, what is the value for ⟪g0⟫?"),
    "color": ("Which color does the rule give for ⟪g0⟫?", "Which value belongs to ⟪g0⟫?",
              "Following the same rule, what is the value for ⟪g0⟫?"),
    "size": ("Which size (rows x columns) does the rule give for ⟪g0⟫?", "Which value belongs to ⟪g0⟫?",
             "Following the same rule, what is the value for ⟪g0⟫?"),
}


def predict_scalar(rng, test_slot: int, kind: str, value) -> Query:
    text = _pick(rng, SCALAR_PREDICT[kind])
    ans_kind = {"number": "number", "color": "color", "size": "text"}[kind]
    stmt = "The value for ⟪g0⟫ is color ⟪answer⟫." if kind == "color" else "The value for ⟪g0⟫ is ⟪answer⟫."
    return Query("arc", "arc:predict_value", (test_slot,), text, Answer(ans_kind, value, str(value)),
                 statement=stmt, declared=stmt)


# the statements cover every training pair, as the questions do (not just the pair they name)
IDENTIFY = (
    ("Which rule from the list turns ⟪g0⟫ into ⟪g1⟫, and every other training input into its output? "
     "Give its label.", "The rule that turns ⟪g0⟫ into ⟪g1⟫, and every other training input into its output, "
                        "is ⟪answer⟫."),
    ("⟪G0⟫ becomes ⟪g1⟫ under one of the listed rules, and so does every other training pair. Which label "
     "is it?", "⟪G0⟫ becomes ⟪g1⟫ under rule ⟪answer⟫, and so does every other training pair."),
    ("Name the listed rule, by its label, that maps ⟪g0⟫ to ⟪g1⟫ and fits every training pair.",
     "The listed rule that maps ⟪g0⟫ to ⟪g1⟫ and fits every training pair is ⟪answer⟫."),
)

IDENTIFY_SCALAR = (
    ("Which listed rule gives the value shown for ⟪g0⟫ and for every other training input? Give its label.",
     "The rule that gives the value shown for every training input, ⟪g0⟫ included, is ⟪answer⟫."),
    ("One listed rule produces the value of ⟪g0⟫ and of every other training input. Which label is it?",
     "The value of every training input, ⟪g0⟫ included, comes from rule ⟪answer⟫."),
)


def identify(rng, reads: tuple, label: str, scalar: bool = False) -> Query:
    text, stmt = _pick(rng, IDENTIFY_SCALAR if scalar else IDENTIFY)
    return Query("arc", "arc:identify", reads, text, Answer("token", label, label), statement=stmt, declared=stmt)


# Always capital letters: the assembler tags questions with lowercase letters or numbers, never
# capitals, so a menu entry can never be mistaken for a question tag.
MENU_LABELS = (tuple("ABCDEF"),)
MENU_HEADS = (
    "Candidate rules (exactly one of them fits every training pair):",
    "Possible rules:",
    "One of these rules produced the training outputs:",
)


def menu_lines(rng, descriptions: list[str]) -> tuple[list[str], list[str]]:
    labels = MENU_LABELS[0][:len(descriptions)]
    head = _pick(rng, MENU_HEADS)
    lines = [head] + [f"{lab}. {d[0].upper() + d[1:]}." for lab, d in zip(labels, descriptions)]
    return lines, list(labels)


VERIFY = (
    ("Is ⟪g0⟫ the correct output for ⟪g1⟫?", "⟪G0⟫ is the correct output for ⟪g1⟫.",
     "⟪G0⟫ is not the correct output for ⟪g1⟫."),
    ("Does the rule turn ⟪g1⟫ into ⟪g0⟫?", "The rule turns ⟪g1⟫ into ⟪g0⟫.",
     "The rule does not turn ⟪g1⟫ into ⟪g0⟫."),
    ("Someone proposes ⟪g0⟫ as the answer for ⟪g1⟫. Is that right?", "⟪G0⟫ is the right answer for ⟪g1⟫.",
     "⟪G0⟫ is not the right answer for ⟪g1⟫."),
)
# the answer-neutral forms, in VERIFY's order (a yes / no statement says the answer)
VERIFY_DECLARED = (
    "Asked whether ⟪g0⟫ is the correct output for ⟪g1⟫, the answer is ⟪answer⟫.",
    "On whether the rule turns ⟪g1⟫ into ⟪g0⟫, the answer is ⟪answer⟫.",
    "Asked whether ⟪g0⟫ is the right answer for ⟪g1⟫, the answer is ⟪answer⟫.",
)


def verify(rng, proposed_slot: int, test_slot: int, correct: bool) -> Query:
    text, yes, no = _pick(rng, VERIFY)
    neutral = VERIFY_DECLARED[[v[0] for v in VERIFY].index(text)]
    return Query("arc", "arc:verify", (proposed_slot, test_slot), text,
                 Answer("bool", correct, "yes" if correct else "no"), statement=yes if correct else no,
                 declared=neutral)


def verify_diff(rng, proposed_slot: int, test_slot: int, n: int) -> Query:
    text = _pick(rng, ("How many cells of ⟪g0⟫ differ from the correct output for ⟪g1⟫?",
                       "In how many cells does ⟪g0⟫ disagree with the correct output for ⟪g1⟫?"))
    stmt = ("⟪G0⟫ differs from the correct output for ⟪g1⟫ in ⟪answer⟫ cell." if n == 1 else
            "⟪G0⟫ differs from the correct output for ⟪g1⟫ in ⟪answer⟫ cells.")
    return Query("arc", "arc:verify_diff", (proposed_slot, test_slot), text, Answer("number", n, str(n)),
                 statement=stmt,
                 declared="The number of cells in which ⟪g0⟫ differs from the correct output for ⟪g1⟫ is ⟪answer⟫.")


def verify_scalar(rng, test_slot: int, proposed, correct: bool, kind: str = "number") -> Query:
    text = _pick(rng, (f"Does the rule give {proposed} for ⟪g0⟫?", f"Is {proposed} the value for ⟪g0⟫?"))
    said = f"color {proposed}" if kind == "color" else f"{proposed}"
    stmt = f"The value for ⟪g0⟫ is {said}." if correct else f"The value for ⟪g0⟫ is not {said}."
    return Query("arc", "arc:verify_value", (test_slot,), text,
                 Answer("bool", correct, "yes" if correct else "no"), statement=stmt,
                 declared=f"Asked whether the rule gives {proposed} for ⟪g0⟫, the answer is ⟪answer⟫.")


# ---------------------------------------------------------------------------------------------
# reading questions over the bundle's grids
# ---------------------------------------------------------------------------------------------

# the reply names grids exactly as the question lists them (qualifiers such as "this part's" included):
# unstated, 4 of 4 blind solves wrote them differently (parity review)
_EACH_FORMAT = ("Give each grid, named exactly as in the list, with its number, in that order, as grid: number, "
                "separated by commas.")

GROUP_WORDS = {"train_input": "training inputs", "train_output": "training outputs", "all": "grids shown"}


def count_color(rng, slot: int, grid, x: int, enc: str) -> Query:
    n = int((np.asarray(grid) == x).sum())
    cw = colour_word(x, enc)
    text = _pick(rng, (f"How many cells of {cw} does ⟪g0⟫ contain?", f"Count the {cw} cells in ⟪g0⟫."))
    stmt = f"⟪G0⟫ contains ⟪answer⟫ {'cell' if n == 1 else 'cells'} of {cw}."
    return Query("arc", "arc:count_color", (slot,), text, Answer("number", n, str(n)), statement=stmt,
                 declared=f"The number of cells of {cw} in ⟪g0⟫ is ⟪answer⟫.")


def distinct_colors(rng, slot: int, grid) -> Query:
    n = len(np.unique(np.asarray(grid)))
    text = _pick(rng, ("How many different colors appear in ⟪g0⟫?", "How many distinct colors does ⟪g0⟫ use?"))
    stmt = f"⟪G0⟫ uses ⟪answer⟫ different {'color' if n == 1 else 'colors'}."
    return Query("arc", "arc:n_colors", (slot,), text, Answer("number", n, str(n)), statement=stmt,
                 declared="The number of different colors in ⟪g0⟫ is ⟪answer⟫.")


def most_frequent(rng, slot: int, grid, enc: str) -> Optional[Query]:
    a = np.asarray(grid)
    counts = np.bincount(a.ravel(), minlength=10)
    top = np.nonzero(counts == counts.max())[0]
    if len(top) != 1:
        return None
    c = int(top[0])
    text = _pick(rng, ("Which color is most frequent in ⟪g0⟫?", "Which color covers the most cells of ⟪g0⟫?"))
    return Query("arc", "arc:mode_color", (slot,), text, Answer("color", c, str(c)),
                 statement="The most frequent color in ⟪g0⟫ is color ⟪answer⟫.",
                 declared="The most frequent color in ⟪g0⟫ is color ⟪answer⟫.")


def dims(rng, slot: int, grid) -> Query:
    a = np.asarray(grid)
    t = f"{a.shape[0]}x{a.shape[1]}"
    text = _pick(rng, ("What are the dimensions of ⟪g0⟫ (rows x columns)?",
                       "How big is ⟪g0⟫? Give rows x columns."))
    return Query("arc", "arc:dims", (slot,), text, Answer("text", t, t), statement="⟪G0⟫ is ⟪answer⟫.",
                 flags=frozenset({FLAG_DIMS}), declared="⟪G0⟫ is ⟪answer⟫.")


def color_at(rng, slot: int, grid) -> Query:
    a = np.asarray(grid)
    r, c = int(rng.integers(a.shape[0])), int(rng.integers(a.shape[1]))
    v = int(a[r, c])
    text = _pick(rng, (f"What color is the cell at row {r}, column {c} of ⟪g0⟫?",
                       f"Which color does ⟪g0⟫ have at row {r}, column {c}?"))
    return Query("arc", "arc:color_at", (slot,), text, Answer("color", v, str(v)),
                 statement=f"The cell at row {r}, column {c} of ⟪g0⟫ is color ⟪answer⟫.",
                 flags=frozenset({FLAG_COORDS}), declared=f"The cell at row {r}, column {c} of ⟪g0⟫ is color ⟪answer⟫.")


def changed_cells(rng, in_slot: int, out_slot: int, a, b) -> Optional[Query]:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return None
    n = int((a != b).sum())
    # never the perception probe's own sentence ("How many cells differ between ... and ...?")
    text = _pick(rng, ("Count the positions where ⟪g0⟫ and ⟪g1⟫ hold different colors.",
                       "In how many cells does ⟪g1⟫ differ from ⟪g0⟫?"))
    stmt = f"⟪G0⟫ and ⟪g1⟫ differ in ⟪answer⟫ {'cell' if n == 1 else 'cells'}."
    return Query("arc", "arc:changed_cells", (in_slot, out_slot), text, Answer("number", n, str(n)),
                 statement=stmt, declared="The number of cells in which ⟪g0⟫ and ⟪g1⟫ differ is ⟪answer⟫.")


def identical(rng, s0: int, s1: int, a, b) -> Query:
    a, b = np.asarray(a), np.asarray(b)
    same = a.shape == b.shape and bool(np.array_equal(a, b))
    text = _pick(rng, ("Is ⟪g0⟫ identical to ⟪g1⟫?", "Are ⟪g0⟫ and ⟪g1⟫ exactly the same grid?"))
    stmt = "⟪G0⟫ is identical to ⟪g1⟫." if same else "⟪G0⟫ is not identical to ⟪g1⟫."
    return Query("arc", "arc:identical", (s0, s1), text, Answer("bool", same, "yes" if same else "no"),
                 statement=stmt, declared="Asked whether ⟪g0⟫ is identical to ⟪g1⟫, the answer is ⟪answer⟫.")


def which_contain(rng, slots: list[int], grids: list, x: int, group: str, enc: str) -> Query:
    cw = colour_word(x, enc)
    ph = _ph(len(slots))
    hits = [i for i, g in enumerate(grids) if bool((np.asarray(g) == x).any())]
    words = GROUP_WORDS[group]
    names = "copy their names from that list, in that order and joined the same way, or answer none."
    text = _pick(rng, (f"Which of the {words} ({_join(ph)}) contain {cw}? To answer, {names}",
                       f"Among {_join(ph)}, which grids contain at least one cell of {cw}? To answer, {names}"))
    ans = _join([ph[i] for i in hits]) if hits else "none"
    return Query("arc", "arc:which_contain", tuple(slots), text,
                 Answer("list", [slots[i] for i in hits], ans))


def each_count(rng, slots: list[int], grids: list, x: int, group: str, enc: str) -> Query:
    cw = colour_word(x, enc)
    ph = _ph(len(slots))
    counts = [int((np.asarray(g) == x).sum()) for g in grids]
    words = GROUP_WORDS[group]
    text = _pick(rng, (f"For each of the {words} ({_join(ph)}), how many cells of {cw} does it contain? "
                       f"{_EACH_FORMAT}",
                       f"Count the cells of {cw} in each of {_join(ph)}. {_EACH_FORMAT}"))
    ans = ", ".join(f"{p}: {n}" for p, n in zip(ph, counts))
    return Query("arc", "arc:each_count", tuple(slots), text, Answer("list", counts, ans))


def each_n_colors(rng, slots: list[int], grids: list, group: str) -> Query:
    ph = _ph(len(slots))
    counts = [len(np.unique(np.asarray(g))) for g in grids]
    words = GROUP_WORDS[group]
    text = f"For each of the {words} ({_join(ph)}), how many different colors does it use? {_EACH_FORMAT}"
    ans = ", ".join(f"{p}: {n}" for p, n in zip(ph, counts))
    return Query("arc", "arc:each_n_colors", tuple(slots), text, Answer("list", counts, ans))


# ---------------------------------------------------------------------------------------------
# grid-writing reading questions: copy a window of one shown grid (row, rows a..b, column, crop)
# ---------------------------------------------------------------------------------------------
# The solver's own path: read stage-2 labelled rows, write compact digit rows (grid-writing rate,
# DESIGN 15).  Never about a grid that is not shown; the window never is the whole grid, and the
# caller refuses an answer equal to a shown grid or to a test output.  Coordinates are 0-based
# (FLAG_COORDS: the assembler adds the 0-index sentence).  Statements are answer-neutral lead-ins
# ending in ":" (the library's grid-answer convention).

GRID_ROW = (
    ("Write out row {a} of ⟪g0⟫.", "Row {a} of ⟪g0⟫ is:"),
    ("Copy row {a} of ⟪g0⟫ as a grid with one row.", "Row {a} of ⟪g0⟫, as a grid with one row, is:"),
    ("What does row {a} of ⟪g0⟫ hold, from left to right?", "From left to right, row {a} of ⟪g0⟫ holds:"),
)
GRID_ROWS = (
    ("Write out rows {a} to {b} of ⟪g0⟫, both included.", "Rows {a} to {b} of ⟪g0⟫ are:"),
    ("Copy rows {a} through {b} of ⟪g0⟫ (inclusive), in their order.", "Rows {a} through {b} of ⟪g0⟫ are:"),
    ("Which cells make up rows {a}-{b} of ⟪g0⟫, both ends included? Give them as a grid.",
     "Rows {a}-{b} of ⟪g0⟫ are made up of:"),
)
GRID_COL = (
    ("Write out column {c} of ⟪g0⟫ as a grid with one column, top to bottom.",
     "Column {c} of ⟪g0⟫, top to bottom, is:"),
    ("Copy column {c} of ⟪g0⟫ from top to bottom, as a one-column grid.",
     "From top to bottom, column {c} of ⟪g0⟫ is:"),
)
GRID_CROP = (
    ("Write out the block of ⟪g0⟫ at rows {a}-{b} and columns {c0}-{c1}, both ranges inclusive.",
     "The block of ⟪g0⟫ at rows {a}-{b} and columns {c0}-{c1} is:"),
    ("Crop ⟪g0⟫ to rows {a} to {b} and columns {c0} to {c1}, ends included, and give the result.",
     "⟪G0⟫ cropped to rows {a} to {b} and columns {c0} to {c1} is:"),
    ("Copy the rectangle of ⟪g0⟫ covering rows {a}..{b} and columns {c0}..{c1} (inclusive).",
     "The rectangle of ⟪g0⟫ covering rows {a}..{b} and columns {c0}..{c1} is:"),
)

GRID_READ_QIDS = ("arc:row", "arc:rows", "arc:col", "arc:crop")


def _window_query(rng, qid: str, slot: int, value, forms, **fmt) -> Query:
    text, stmt = _pick(rng, forms)
    grid = [[int(v) for v in row] for row in np.asarray(value)]
    lead = stmt.format(**fmt)          # an answer-neutral lead-in: declarable as it stands
    return Query("arc", qid, (slot,), text.format(**fmt), Answer("grid", grid), statement=lead, declared=lead,
                 flags=frozenset({FLAG_COORDS}))


def grid_row(rng, slot: int, grid, a: int) -> Query:
    return _window_query(rng, "arc:row", slot, np.asarray(grid)[a:a + 1], GRID_ROW, a=a)


def grid_rows(rng, slot: int, grid, a: int, b: int) -> Query:
    return _window_query(rng, "arc:rows", slot, np.asarray(grid)[a:b + 1], GRID_ROWS, a=a, b=b)


def grid_col(rng, slot: int, grid, c: int) -> Query:
    return _window_query(rng, "arc:col", slot, np.asarray(grid)[:, c:c + 1], GRID_COL, c=c)


def grid_crop(rng, slot: int, grid, a: int, b: int, c0: int, c1: int) -> Query:
    return _window_query(rng, "arc:crop", slot, np.asarray(grid)[a:b + 1, c0:c1 + 1], GRID_CROP,
                         a=a, b=b, c0=c0, c1=c1)
