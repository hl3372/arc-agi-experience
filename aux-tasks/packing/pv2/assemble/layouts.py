"""Layouts: where grids, questions and instructions go, and how grid blocks are delimited.

Placements (per part):
  single grid   q_first (question before the grid) | q_after (after it) | split (short context line
                before the grid, question after)
  several grids grids_first | questions_first | split | interleaved (grid, its questions, next grid)
                | reminder (question list on top, grids, a one-line reminder at the bottom)

Delimiters (per part): ``label_line`` ("Grid B:" / stage-2 "Training pair 1 input (5 rows x 7 cols):"
header line, body, blank line), ``fenced`` (optional header line + ``` fence), ``tags`` (optional
header line + <grid> ... </grid>), ``banner`` ("=== Grid B ===" line), ``blank`` (body alone, blank
lines around; unlabelled only -- canonical encodings never contain blank lines, so grids stay
separable).

Also here: part boundaries and answer headings of multi-part rows, question tags, lead clauses for
item questions, the scoped sentences (0-indexed, "as shown", grid-answer encoding), intro lines and
the row header.  Nothing here says "above" or "below".
"""
from __future__ import annotations

import re

from pv2.assemble._util import count_phrase, join_tags, pick
from pv2.assemble.flags import contains_word
from pv2.encodings import ENCODINGS

DELIMITERS = ("label_line", "fenced", "tags", "banner", "blank")
SINGLE_LAYOUTS = ("q_first", "q_after", "split")
MULTI_LAYOUTS = ("grids_first", "questions_first", "split", "interleaved", "reminder")
GENERIC_HEADERS = ("Input", "Grid")          # headers of a lone unlabelled grid (not labels)


def delimiters_for(scheme: str) -> tuple[str, ...]:
    if scheme == "ordinal":
        return ("fenced", "tags", "banner", "blank")      # no header line to hang a label on
    if scheme == "none":
        return DELIMITERS
    return ("label_line", "fenced", "tags", "banner")      # labels are always printed


def block_frame(header: str | None, delimiter: str) -> tuple[str, str]:
    """(prefix, suffix) around a grid body.  ``header`` is the full header text (label and optional
    dims) or None."""
    if delimiter == "label_line":
        if header is None:
            raise ValueError("label_line needs a header")
        return f"{header}:\n", ""
    head = f"{header}:\n" if header is not None else ""
    if delimiter == "fenced":
        return head + "```\n", "\n```"
    if delimiter == "tags":
        return head + "<grid>\n", "\n</grid>"
    if delimiter == "banner":
        return f"=== {header if header is not None else 'grid'} ===\n", ""
    if delimiter == "blank":
        if header is not None:
            raise ValueError("blank delimiter carries no header")
        return "", ""
    raise ValueError(f"unknown delimiter {delimiter!r}")


# --- row structure ------------------------------------------------------------------------------------

BOUNDARY_FORMATS = ("{N} {k} of {n}", "{N} {k} of {n}:", "## {N} {k} of {n}", "=== {N} {k} of {n} ===",
                    "--- {N} {k}/{n} ---")
HEADING_FORMATS = ("{N} {k}:", "[{N} {k}]", "## {N} {k}", "{N} {k} answers:")


def boundary_line(fmt: str, noun: str, k: int, n: int) -> str:
    return fmt.format(N=noun.capitalize(), k=k, n=n)


def heading_line(fmt: str, noun: str, k: int) -> str:
    return fmt.format(N=noun.capitalize(), k=k)


def heading_formats(banned) -> tuple[str, ...]:
    return tuple(f for f in HEADING_FORMATS if not ("answer" in banned and "answers" in f))


def row_header(rng, noun: str, n_parts: int, n_questions: int, heading_fmt: str) -> str:
    parts = count_phrase(n_parts, noun, rng)
    h1, h2 = heading_line(heading_fmt, noun, 1), heading_line(heading_fmt, noun, 2)
    first = pick(rng, (f"This prompt has {parts}.",
                       f"This prompt has {parts} with {count_phrase(n_questions, 'question', rng)} in all.",
                       f"There are {parts}."))
    order = pick(rng, (f"Answer the {noun}s in order, each in the format it asks for.",
                       f"Work through the {noun}s in order and answer each one in the format it asks for."))
    head = pick(rng, (f'Begin the answers of each {noun} with its heading on a line of its own, exactly '
                      f'"{h1}", "{h2}" and so on, and leave a blank line between {noun}s.',
                      f'Put each {noun}\'s answers under a heading line of the form "{h1}" (then "{h2}", '
                      f'and so on), with a blank line between {noun}s.'))
    return " ".join((first, order, head))


# --- questions ----------------------------------------------------------------------------------------

QUESTION_TAGS = {
    "number": ("{t}.", "{t})", "Q{t}.", "Q{t}:", "Question {t}:", "({t})"),
    "letter": ("({t})", "{t})", "{t}."),
}
LEAD_FORMS = ("About {refs}:", "For {refs}:", "Question about {refs}:", "Regarding {refs}:", "Look at {refs}.")


def question_tag_formats(kind: str, banned) -> tuple[str, ...]:
    return tuple(f for f in QUESTION_TAGS[kind] if not contains_word(f, banned))


def lead_forms(banned) -> tuple[str, ...]:
    return tuple(f for f in LEAD_FORMS if not contains_word(f, banned))


def lead_clause(form: str, n_reads: int) -> str:
    """A lead clause with placeholders, e.g. "About ⟪g0⟫:" (several reads: "⟪g0⟫ and ⟪g1⟫")."""
    refs = join_tags([f"⟪g{i}⟫" for i in range(n_reads)])
    return form.format(refs=refs)


def question_line(tag_fmt: str | None, tag: str | None, text: str) -> str:
    if tag is None:
        return text
    return tag_fmt.format(t=tag) + " " + text


def question_block(lines: list[str]) -> str:
    sep = "\n\n" if any("\n" in line for line in lines) else "\n"
    return sep.join(lines)


# --- scoped sentences ----------------------------------------------------------------------------------

# How a text shows it states the 0-indexed convention (the library's markers included).
ZERO_INDEX_RE = re.compile(r"0-indexed|zero-based|zero-indexed|\bfrom 0\b|\bstart(?:s|ing)? at 0\b|"
                           r"[Rr]ow 0 is the top row|\(0,0\) (?:is )?the top-left")
AS_SHOWN_RE = re.compile(r"\bas (?:they are )?shown\b", re.IGNORECASE)

ZERO_INDEX_SENTENCES = (
    "Rows and columns are 0-indexed from the top-left corner.",
    "Row and column numbers start at 0 at the top-left corner.",
    "Rows and columns are numbered from 0, starting at the top-left cell.",
)
AS_SHOWN_SENTENCES = (
    "Every question refers to the grid as shown, not to the result of another question.",
    "Each question is about the grids exactly as shown, not as changed by another question.",
    "Answer every question from the grids as shown; no question builds on another question's result.",
)
SCOPE_FORMS = ("For this {noun}: ", "In this {noun}, ")


def scoped(form: str, noun: str, text: str) -> str:
    """"For this part: Reply ..." / "In this part, reply ..." (multi-part rows scope every instruction)."""
    head = form.format(noun=noun)
    return head + (text[:1].lower() + text[1:] if head.endswith(", ") else text)


# A grid shown in colour NAMES whose grid answer is written in digits: the prompt must give the
# mapping (training-side blind solve, 2026-09-30: row 5070000949 answered a names grid in digits
# with no mapping stated, so the answer needed the ARC palette from memory).
NAMES_TO_DIGITS_MAP = ("stand for digits: black 0, blue 1, red 2, green 3, yellow 4, grey 5, magenta 6, "
                       "orange 7, cyan 8, maroon 9")
NAMES_TO_DIGITS = f"Color names {NAMES_TO_DIGITS_MAP}."      # scoped instructions lowercase the first word


GRID_ANSWER_FORMS = ("Write grid answers {d}.", "Give every grid answer {d}.",
                     "When an answer is a grid, write it {d}.")


def grid_answer_sentence(rng, encoding: str) -> str:
    return pick(rng, GRID_ANSWER_FORMS).format(d=pick(rng, ENCODINGS[encoding].describe))


def intro_line(rng, n_grids: int, n_questions: int | None) -> str:
    if n_grids == 1:
        text = pick(rng, ("Here is a grid.", "Consider the following grid.", "Look at this grid.",
                          "Take this grid."))
    else:
        g = count_phrase(n_grids, "grid", rng)
        text = pick(rng, (f"Here are {g}.", f"You will see {g}.", f"Consider these {g}.",
                          "Study the following grids.", "Look at the grids that follow."))
    if n_questions is not None:
        text += f" Then answer {count_phrase(n_questions, 'question', rng)}." if n_questions > 1 \
            else " Then answer the question."
    return text


def reminder_line(rng, tags: list[str | None]) -> str:
    if tags[0] is None:
        return "Now answer the question."
    shown = [t if t.isdigit() else f"({t})" for t in tags]          # a letter tag is printed "(a)"
    if len(tags) == 1:
        return pick(rng, (f"Now answer question {shown[0]}.", "Now answer the question."))
    return pick(rng, (f"Now answer questions {join_tags(shown)}.",
                      f"Now answer the {count_phrase(len(tags), 'question')}."))
