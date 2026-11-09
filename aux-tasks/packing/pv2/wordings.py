"""Every wording of every in-scope registry task, its flags, and the answer-wrapper table.

A task's WORDINGS are its canonical builder (``prompts.get_prompt``; index 0) and every approved
stored paraphrase in the generator package's ``prompt_variants.json`` (index k = the k-th approved
variant in store order, rendered through ``prompt_variants.render``).  A wording's text depends on
the example's metadata, so flags are computed on the text actually rendered for an example
(``text_flags``); ``flag_table`` aggregates them over the coverage seeds for reports and tests.

Flags (``pv2.model`` FLAG_* strings, DESIGN.md 13.3) and how each is measured:

* ``io_nouns``      "input" / "output" used as a NOUN ("the input", "the output's top row",
                    "every input cell", "Output: ...").  The imperative verb ("Output the whole grid.",
                    "and output the result") is not flagged.  Never in role-bearing (ARC) parts.
* ``placement``     the text locates its grid relative to itself ("The grid below", "The picture
                    below").  Only in single-grid, question-first parts.  Spatial uses inside the grid
                    ("from above or below", "below the previous bounding box") and references to the
                    text's own list ("each allowed symmetry below", task 102) are not placement.
* ``label_word:w``  the text uses a word the assembler may use as a grid label: image, panel, map,
                    figure, picture (any form: "mirror images", "maps to", "Figure:"); ``#`` (the
                    pattern glyph of 4/5/58/59, which would sit badly next to "#3" labels); a
                    standalone capital letter (``label_word:L`` -- the L / X / H / U / T shape names of
                    4/5/59 -- so letter labels "Grid L" avoid it); ``grid_number`` ("Tile this grid 4
                    times", "Rotate this grid 90 degrees": no numeric "Grid 4" labels next to it).
                    "grid" itself is never flagged: every item says "grid" for its own grid, which the
                    lead clause names.
* ``nothing_else``  the wording demands nothing but its answer ("Do not add extra lines or extra
                    text.", "Your answer is ... and nothing else").  Only in single-answer parts.  The
                    neutral text drops the demand as well (see ``neutral_text``), so no part ever shows
                    it next to an answer style; the flag stays as the review asked (13.3).
* ``style_label:w`` the text uses "Answer:", "Question:" or "Task:" as an inline label; the part should
                    not use that word for its own question / answer labels.  (Added here: 26_v003 has
                    "Answer:", seven wordings "Question:", seven "Task:".)
* ``inner_ordinals`` the text numbers parts of its own grid with ordinals or calls them grids ("In the
                    first subgrid ... in the second", "two grids of true/false values": every task-42
                    wording).  A part holding such an item should name its grids by label, not by
                    ordinal words ("the second grid"), so "the second" never has two readings.  (Added.)
* ``long``          the neutral text is longer than ``LONG_CHARS`` characters (no restate-then-answer).
* ``structured``    task-level: a verbatim structured answer (coordinate lists, per-colour counts, the
                    labelled lines of 102); only block-capable styles.
* ``coords``        task-level: the task's wordings read or write row / column numbers (coordinates,
                    0-indexed rows / columns / blocks, mirror formulas).
* ``dims``          task-level: the answer is the grid's size (task 1).
* ``edit``          task-level: the answer is a grid (the read grid changed or cut); the part then says
                    every question refers to the grid as shown.

The ANSWER-WRAPPER TABLE (``WRAPPERS``) maps every answer-format sentence to a neutral notation
clause that states the value's notation without dictating the reply's shape; ``neutral_text``
swaps it.  v2 parts always get their reply shape from a part-level instruction written by the
assembler, so item texts carry only the notation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

from pv2 import model
from pv2.paths import ensure_generator_package

ensure_generator_package()
import prompt_variants as _pv  # noqa: E402
from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402

from pv2.pictures import picture_tasks  # noqa: E402


def _task_key(tid: str):
    m = re.match(r"(\d+)(.*)", tid)
    return (int(m.group(1)), m.group(2))


#: every task whose input is one grid (1-102 minus 14, 41, 78), natural order
TASKS: tuple[str, ...] = tuple(sorted(picture_tasks(), key=_task_key))

# --- answers ------------------------------------------------------------------------------------

NUMBER_TASKS = frozenset({"4", "5", "9", "15", "15a", "22", "24", "30", "35", "46a", "51", "52", "54",
                          "59", "68a", "68b", "72", "73", "89", "90", "92"})
COLOR_TASKS = frozenset({"25", "57", "75"})                   # one colour label (a digit)
TOKEN_TASKS = {"36": ("closed", "open"), "38": ("true", "false"), "56": ("yes", "no")}
TEXT_TASKS = frozenset({"1", "2", "45", "60", "62", "76", "94", "102"})   # verbatim notation
#: answers from a closed vocabulary (checked against every generated answer)
VOCABULARY = {"9": ("90", "180", "270"), **TOKEN_TASKS}


def answer_kind(tid: str) -> str:
    """number / color / token / text / grid -- the ``Answer.kind`` of the task's items."""
    if tid in NUMBER_TASKS:
        return "number"
    if tid in COLOR_TASKS:
        return "color"
    if tid in TOKEN_TASKS:
        return "token"
    if tid in TEXT_TASKS:
        return "text"
    return "grid"


def make_answer(tid: str, output) -> model.Answer:
    """The gold ``Answer`` of one generated example (text = the old builder's ``str(output)``)."""
    kind = answer_kind(tid)
    if kind == "grid":
        if not (isinstance(output, list) and output and isinstance(output[0], list)):
            raise ValueError(f"task {tid}: expected a grid output, got {output!r}")
        return model.Answer("grid", [list(map(int, row)) for row in output])
    if not isinstance(output, str):
        raise ValueError(f"task {tid}: expected a string output, got {type(output).__name__}")
    text = str(output)
    if tid in VOCABULARY and text not in VOCABULARY[tid]:
        raise ValueError(f"task {tid}: {text!r} is outside the vocabulary {VOCABULARY[tid]}")
    if kind in ("number", "color"):
        if not re.fullmatch(r"0|[1-9]\d*", text):
            raise ValueError(f"task {tid}: {text!r} is not a number")
        value = int(text)
        if kind == "color" and not 0 <= value <= 9:
            raise ValueError(f"task {tid}: {text!r} is not a colour")
        return model.Answer(kind, value, text)
    return model.Answer(kind, text, text)


# --- task-level flags ---------------------------------------------------------------------------

#: verbatim structured answers (coordinate lists, per-colour counts, labelled lines)
STRUCTURED_TASKS = frozenset({"2", "60", "62", "76", "94", "102"})
#: wordings read or write row / column numbers (measured: ``coords_in_text``; 61/63 recolour path
#: cells by count and never print a coordinate, so they are not coordinate tasks)
COORD_TASKS = frozenset({"2", "3", "60", "62", "77", "79", "86", "87", "94", "98", "99", "101", "102"})
#: the answer is the grid's size
DIMS_TASKS = frozenset({"1"})

_COORD_TEXT = re.compile(r"0-indexed|\(r,c\)|\(row,\s?col\)|\brows? \d|\bcolumns? \d|position \(\d|"
                         r"\(\d+, \d+\)|\brow r\b")


def coords_in_text(text: str) -> bool:
    """Whether a prompt text prints row / column numbers (used to measure ``COORD_TASKS``)."""
    return bool(_COORD_TEXT.search(text))


def task_flags(tid: str) -> frozenset:
    flags = set()
    if tid in STRUCTURED_TASKS:
        flags.add(model.FLAG_STRUCTURED)
    if tid in COORD_TASKS:
        flags.add(model.FLAG_COORDS)
    if tid in DIMS_TASKS:
        flags.add(model.FLAG_DIMS)
    if answer_kind(tid) == "grid":
        flags.add(model.FLAG_EDIT)
    return frozenset(flags)


# --- the answer-wrapper table ---------------------------------------------------------------------
# Every answer-format sentence of tasks 1-102 (the 13 distinct sentences that
# ``prompt_variants._ANSWER_SENTENCE`` finds, every stored variant carries its task's sentence
# verbatim) plus task 102's inline format.  Tasks 2 and 94 state their notation inside the question
# itself ("List ... as [(r,c),...]", "Return one box per object as [...]") and have no wrapper,
# except 2_v004, whose wrapper sentence is in the table.

WRAPPERS: tuple[tuple[str, str], ...] = (
    ("Answer with the number (0 if nothing is enclosed).", "The answer is a number (0 if nothing is enclosed)."),
    ("Answer with the number.", "The answer is a number."),
    ("Answer as HxW.", "Write the size as HxW."),
    ("Answer as MxN.", "Write the counts as MxN."),
    ("Answer as 90, 180, or 270.", "The angle is 90, 180, or 270."),
    ("Answer as a single color label.", "The answer is a single color label."),
    ("Answer 'closed' or 'open'.", "The answer is 'closed' or 'open'."),
    ("Answer 'true' or 'false'.", "The answer is 'true' or 'false'."),
    ("Answer 'yes' or 'no'.", "The answer is 'yes' or 'no'."),
    ("Answer with the coordinates of all of its cells in row-major order, written [(r,c),...].",
     "The answer lists the coordinates of all of its cells in row-major order, written [(r,c),...]."),
    ("Format the answer exactly as first:[(row,col),(row,col),...], second:[(row,col),(row,col),...].",
     "Write the two runs exactly as first:[(row,col),(row,col),...], second:[(row,col),(row,col),...]."),
    ("Format the answer exactly as [(row,col),(row,col),...].",
     "Write the cells exactly as [(row,col),(row,col),...]."),
    ("Format: color:count, color:count, ...", "Write the counts as color:count, color:count, ..."),
    ("Return exactly one line for each label in that order, formatted as label: yes or label: no. "
     "Do not add extra lines or extra text.",
     "The answer has exactly one line for each label in that order, formatted as label: yes or label: no."),
)

#: tasks with a text answer whose notation lives in the question sentence itself (no wrapper)
INLINE_NOTATION_TASKS = frozenset({"2", "94"})

# Reply-shape demands inside a question sentence (not wrapper sentences): removed from the neutral
# text; the wording keeps its ``nothing_else`` flag.
_INLINE_DEMANDS = (
    (re.compile(r"(Your answer is [^.:]*?) and nothing else:"), r"\1:"),     # 77_v006
)

#: phrases that dictate the reply's shape; none may survive in a neutral text
REPLY_SHAPE = re.compile(
    r"\bAnswer (?:with|as)\b|\bAnswer '|\bFormat the answer\b|\bFormat:|\bDo not add\b|"
    r"\bextra (?:text|lines)\b|\b(?:answer|reply|response)\b[^.]*\b(?:nothing else|only)\b|"
    r"\b(?:reply|respond)\b", re.I)


def wrappers_in(text: str) -> list[str]:
    return [orig for orig, _ in WRAPPERS if orig in text]


def neutral_text(tid: str, wording_text: str) -> str:
    """The instruction with its answer wrapper swapped for the neutral notation clause.

    Raises when a value-answer task shows no known wrapper (and is not an inline-notation task), or
    when any reply-shape phrase survives -- the completeness guarantee of the table."""
    out = wording_text
    found = False
    for orig, neutral in WRAPPERS:
        if orig in out:
            out = out.replace(orig, neutral)
            found = True
    for rx, rep in _INLINE_DEMANDS:
        out = rx.sub(rep, out)
    if answer_kind(tid) != "grid" and not found and tid not in INLINE_NOTATION_TASKS:
        raise ValueError(f"task {tid}: no known answer wrapper in {wording_text!r}")
    if answer_kind(tid) == "grid" and found:
        raise ValueError(f"task {tid}: a grid task carries a value-answer wrapper")
    hit = REPLY_SHAPE.search(out) or _pv._ANSWER_SENTENCE.search(out)
    if hit:
        raise ValueError(f"task {tid}: reply-shape phrase {hit.group(0)!r} survives in {out!r}")
    return out


# --- text flags ---------------------------------------------------------------------------------

LONG_CHARS = 150

_IO_NOUN = re.compile(
    r"\b(?:the|this|that|an?|each|every|other|original|its|their|your|same)\s+(?:inputs?|outputs?)\b"
    r"|\b(?:inputs?|outputs?)(?:'s|\s+(?:grids?|cells?|is|are|has|have|shows?|keeps?|matches)\b)"
    r"|\b(?:[Ii]nputs?|[Oo]utputs?)\s*:", re.I)
_PLACEMENT = re.compile(
    r"\b(?:grid|picture|image|drawing|figure|input|pattern)\s+(?:below|above|beneath)\b"
    r"|\b(?:shown|given|printed|displayed|provided|written)\s+(?:below|above)\b"
    r"|\bthe following (?:grid|picture|image)\b", re.I)
_NOTHING_ELSE = re.compile(
    r"\bDo not add extra (?:lines|text)|\bextra text\b|\b(?:answer|reply|response)\b[^.]*\bnothing else\b"
    r"|\b(?:answer|reply|respond)\s+(?:with\s+)?only\b", re.I)
LABEL_WORDS = {
    "image": re.compile(r"\bimages?\b", re.I),
    "panel": re.compile(r"\bpanels?\b", re.I),
    "map": re.compile(r"\bmap(?:s|ped|ping)?\b", re.I),
    "figure": re.compile(r"\bfigures?\b", re.I),
    "picture": re.compile(r"\bpictures?\b", re.I),
    "#": re.compile(r"#"),
    "grid_number": re.compile(r"\bgrid \d", re.I),
}
_LABEL_LETTER = re.compile(r"\b([B-HJ-Z])\b")     # "A" is the article, "I" never occurs
_STYLE_LABEL = re.compile(r"\b(Answer|Question|Task)\s*:")
_INNER_ORDINALS = re.compile(r"\bsubgrids?\b|\btwo grids\b", re.I)

FLAG_STYLE_LABEL_PREFIX = "style_label:"
FLAG_INNER_ORDINALS = "inner_ordinals"


def text_flags(text: str, neutral: Optional[str] = None) -> frozenset:
    """Flags measured on one rendered wording (``neutral``: its neutral text, for ``long``)."""
    flags = set()
    if _IO_NOUN.search(text):
        flags.add(model.FLAG_IO_NOUNS)
    if _PLACEMENT.search(text):
        flags.add(model.FLAG_PLACEMENT)
    # measured on the text v2 SHOWS: the neutral text drops the reply-shape demands (102's "Do not add
    # extra lines or extra text.", 77_v006's "and nothing else"), so those wordings may join
    # multi-answer parts (items verifier, 2026-09-29: 102 was in 6 of 14,170 item slots)
    if _NOTHING_ELSE.search(neutral if neutral is not None else text):
        flags.add(model.FLAG_NOTHING_ELSE)
    for word, rx in LABEL_WORDS.items():
        if rx.search(text):
            flags.add(model.LABEL_WORD_PREFIX + word)
    for m in _LABEL_LETTER.finditer(text):
        flags.add(model.LABEL_WORD_PREFIX + m.group(1))
    for m in _STYLE_LABEL.finditer(text):
        flags.add(FLAG_STYLE_LABEL_PREFIX + m.group(1).lower())
    if _INNER_ORDINALS.search(text):
        flags.add(FLAG_INNER_ORDINALS)
    if len(neutral if neutral is not None else text) > LONG_CHARS:
        flags.add(model.FLAG_LONG)
    return frozenset(flags)


# --- wordings -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Wording:
    """One way a task's instruction is worded: index 0 = canonical builder, k = k-th approved variant."""
    task_id: str
    index: int
    variant_id: Optional[str] = None
    template: Optional[str] = None

    @property
    def wid(self) -> str:
        """Stable id: ``"77:canon"`` or the store id ``"77_v003"``."""
        return f"{self.task_id}:canon" if self.index == 0 else str(self.variant_id)

    def render(self, meta: Optional[dict]) -> str:
        """The instruction for one example, exactly as the old builder renders it (no palette)."""
        if self.template is None:
            return get_prompt(self.task_id, meta)
        return _pv.render(self.task_id, self.template, meta=meta)


@lru_cache(maxsize=1)
def _store() -> dict:
    return _pv.load()


@lru_cache(maxsize=None)
def wordings(tid: str) -> tuple[Wording, ...]:
    """Canonical + every approved stored variant of an in-scope task, in store order."""
    tid = str(tid)
    if tid not in TASKS:
        raise KeyError(f"task {tid} is not an in-scope single-grid task")
    rows = _pv.variants_for(tid, _store())
    return (Wording(tid, 0),) + tuple(Wording(tid, k, v["id"], v["template"]) for k, v in enumerate(rows, 1))


@dataclass(frozen=True)
class Rendered:
    """One wording rendered for one example: original text, neutral text and all flags."""
    wording: Wording
    text: str
    neutral: str
    flags: frozenset


def render_all(tid: str, meta: Optional[dict]) -> list[Rendered]:
    """Every wording of the task rendered for one example's metadata, with its flags."""
    out = []
    tflags = task_flags(tid)
    for w in wordings(tid):
        text = w.render(meta)
        neutral = neutral_text(tid, text)
        out.append(Rendered(w, text, neutral, text_flags(text, neutral) | tflags))
    return out


# --- measurement over coverage seeds -----------------------------------------------------------------

def sample_seeds(tid: str, sweep: int = 400, extra: Iterable[int] = range(4)) -> list[int]:
    """One seed per distinct canonical wording (``prompt_variants.coverage_seeds``) plus ``extra``."""
    return sorted(set(_pv.coverage_seeds(tid, sweep)) | set(extra))


def flag_table(seed_table: Optional[dict[str, list[int]]] = None, tasks: Optional[Iterable[str]] = None,
               sweep: int = 400) -> dict[str, frozenset]:
    """{wording id: union of its flags over the task's seeds} (for reports and tests)."""
    table: dict[str, set] = {}
    for tid in (tasks or TASKS):
        seeds = seed_table[tid] if seed_table and tid in seed_table else sample_seeds(tid, sweep)
        for seed in seeds:
            meta = generate(tid, int(seed)).get("metadata", {})
            for r in render_all(tid, meta):
                table.setdefault(r.wording.wid, set()).update(r.flags)
    return {k: frozenset(v) for k, v in table.items()}


def task_of(wid: str) -> str:
    """The task id of a wording id ("77:canon" / "77_v003")."""
    return wid.split(":")[0] if ":" in wid else wid.rsplit("_v", 1)[0]


def flag_counts(table: dict[str, frozenset]) -> dict[str, tuple[int, int]]:
    """{flag: (number of wordings, number of tasks)} from a ``flag_table``."""
    per: dict[str, tuple[set, set]] = {}
    for wid, flags in table.items():
        tid = task_of(wid)
        for f in flags:
            ws, ts = per.setdefault(f, (set(), set()))
            ws.add(wid)
            ts.add(tid)
    return {f: (len(ws), len(ts)) for f, (ws, ts) in sorted(per.items())}


if __name__ == "__main__":  # pragma: no cover - report helper
    import json
    import sys

    sweep = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    table = flag_table(sweep=sweep)
    print(json.dumps({f: {"wordings": n, "tasks": t} for f, (n, t) in flag_counts(table).items()}, indent=1))
