"""The contract between producers and the assembler (DESIGN.md sections 3 and 14).

Producers -- registry items (``pv2.items``), the question library (``pv2.library``) and ARC bundles
(``pv2.arc``) -- return ``Part``s: grids (``Slot``s) plus questions (``Query``s) with gold
``Answer``s.  They never decide how a grid is referred to, labelled or placed, nor the answer style;
the assembler (``pv2.assemble``) does, and renders, and validates.

Placeholders.  Question texts and statement forms name grids through placeholders that the
assembler fills with reference phrases.  They use ⟪...⟫, which never occurs in task prompts (plain
braces would collide with JSON and with prompt text):

    ⟪g0⟫  reference phrase for the query's first slot      "the third grid" / "Grid B" / "Training pair 0 input"
    ⟪G0⟫  the same, capitalised for a sentence start       "The third grid"
    ⟪g0s⟫ possessive                                      "the third grid's" / "Grid B's"
    ⟪g1⟫ ⟪G1⟫ ⟪g1s⟫ ...  further slots in ``Query.reads`` order
    ⟪answer⟫  (statement and declared forms only) the rendered answer value

Everything a question says about a grid goes through these; a producer never writes "the grid"
for a slot it does not own alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

Grid = list[list[int]]

# --- answers -------------------------------------------------------------------------------------

KINDS = (
    "number",   # an integer                                       value: int
    "color",    # one colour                                       value: int 0-9
    "grid",     # a grid (text rendered in the part's grid-answer encoding)   value: Grid
    "bool",     # yes/no, true/false, open/closed ... (text is the task's own token)   value: bool
    "token",    # a word or short code from a closed set (text verbatim)          value: str
    "text",     # an item answer in the task's own notation (2x3, (dr,dc), lists) value: str
    "list",     # several values, e.g. "for each grid ..." (text is a joined form)  value: list
)


@dataclass(frozen=True)
class Answer:
    """A gold answer.

    ``text`` is the bare answer exactly as the question asks for it (for items: the task's own
    rendering, verbatim).  For ``kind == "grid"`` leave ``text`` as None: the assembler renders
    ``value`` in the part's grid-answer encoding (compact unless the part declares another).
    ``value`` is the typed value (JSON style and lenient eval use it).

    An answer that NAMES grids ("which grids contain colour 4?") writes them as placeholders in
    ``text`` (``"⟪g0⟫ and ⟪g2⟫"``, or ``"none"``) and lists the slot indices in ``value``; the
    assembler fills them with the same reference phrases the question uses.
    """
    kind: str
    value: Any
    text: Optional[str] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown answer kind {self.kind!r}")
        if self.kind == "grid":
            if self.text is not None:
                raise ValueError("grid answers are rendered by the assembler; leave text None")
        elif self.text is None:
            raise ValueError("non-grid answers carry their bare text")

    @property
    def multiline(self) -> bool:
        return self.kind == "grid" or "\n" in (self.text or "")


# --- grids and questions -------------------------------------------------------------------------

ROLES = ("grid", "train_input", "train_output", "test_input", "candidate", "proposed")


@dataclass
class Slot:
    """One grid shown in the prompt.

    ``encoding`` is chosen by the producer (it knows what its questions must not be handed:
    coordinates, dimensions, the background) through ``pv2.encodings`` flags.  ``role`` and ``pair``
    matter for ARC parts (stage-2 labels are derived from them).  ``dims_header_ok`` is False when a
    question on this slot asks for its size, so no "(H rows x W cols)" header may be printed.
    """
    grid: Grid
    encoding: str = "compact"
    role: str = "grid"
    pair: Optional[int] = None
    dims_header_ok: bool = True

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"unknown slot role {self.role!r}")


# Query flags (strings) that the assembler and validator act on.
FLAG_EDIT = "edit"                    # the answer is the read grid changed (an edit, not a read)
FLAG_COORDS = "coords"                # the question or answer uses row/column numbers -> 0-indexed sentence
FLAG_DIMS = "dims"                    # asks for a size -> no dims header on the read slot
FLAG_IO_NOUNS = "io_nouns"            # the text uses "input"/"output" as plain nouns -> never in role-bearing parts
FLAG_PLACEMENT = "placement"          # says "below"/"above" -> only single-grid, question-first parts
FLAG_NOTHING_ELSE = "nothing_else"    # demands nothing but its answer -> only single-answer parts
FLAG_STRUCTURED = "structured"        # verbatim multi-line / structured answer -> only block-capable styles
FLAG_LONG = "long"                    # long question text -> no restate-then-answer
LABEL_WORD_PREFIX = "label_word:"     # e.g. "label_word:image" -> the part may not label grids with that word


@dataclass
class Query:
    """One question with its gold answer.

    ``text``: the question, with ⟪g0⟫-style placeholders for every grid it mentions (``reads`` order).
    For registry items this is the task instruction (answer wrapper swapped, see pv2.items) and the
    assembler adds the leading reference clause; ``lead_clause=True`` asks for it.
    ``statement``: the ONE full-sentence form of this phrasing, with placeholders and ⟪answer⟫, or
    None (the assembler then uses its generic form for the source).
    ``flags``: see the FLAG_* constants.
    ``source``/``qid``: provenance ("item", "lib", "arc") and a stable id ("item:77:v3", "lib:row").
    ``declared``: the ANSWER-NEUTRAL sentence form, written verbatim into the prompt when a sentence /
    paragraph part declares its sentences (``AssembleConfig.declare_statements_p``): the same
    placeholders, ⟪answer⟫ once (a grid answer: a lead-in ending in ":" and no ⟪answer⟫), and not one
    word that depends on the answer (no "1 cell" / "cells", no yes- or none-sentence).  None: not
    declarable; a declared part then states the generic form for this question.  A builder that
    knows its sentence is neutral only sets this field.  Kept out of ``repr`` so the pinned digest of
    default ARC parts (tests/test_arc_bundle.py) measures the parts' content, not this text.
    """
    source: str
    qid: str
    reads: tuple[int, ...]
    text: str
    answer: Answer
    statement: Optional[str] = None
    flags: frozenset = frozenset()
    lead_clause: bool = False
    declared: Optional[str] = field(default=None, repr=False)

    def label_words(self) -> set[str]:
        return {f[len(LABEL_WORD_PREFIX):] for f in self.flags if f.startswith(LABEL_WORD_PREFIX)}


@dataclass
class Part:
    """One episode: grids, questions and optional context lines owned by the producer.

    ``context``: sentences the producer needs before the questions (e.g. an ARC mode-A rule
    statement), with placeholders allowed.  ``labels_required``: the assembler must label every
    slot (ARC roles always are).  ``single_answer``: the part must be packed with one answer only.
    ``prov``: whatever re-creates the part (task ids, seeds, variant indices, transform names).
    """
    family: str
    slots: list[Slot]
    queries: list[Query]
    context: list[str] = field(default_factory=list)
    labels_required: bool = False
    single_answer: bool = False
    prov: dict = field(default_factory=dict)

    def validate_shape(self) -> None:
        if not self.slots or not self.queries:
            raise ValueError(f"{self.family}: a part needs slots and queries")
        for q in self.queries:
            if not q.reads or any(not 0 <= i < len(self.slots) for i in q.reads):
                raise ValueError(f"{self.family}: query {q.qid} reads a slot that does not exist")
            for i in range(len(q.reads)):
                if not any(f"⟪{tok}⟫" in q.text for tok in (f"g{i}", f"G{i}", f"g{i}s", f"G{i}s")) \
                        and not q.lead_clause:
                    raise ValueError(f"{self.family}: query {q.qid} never names its slot {i}")


# --- placeholders ----------------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"⟪(answer|[gG]\d+s?)⟫")


def fill(text: str, refs: list[str], answer: Optional[str] = None) -> str:
    """Replace placeholders; ``refs[i]`` is the lower-case reference phrase for reads[i]."""
    def sub(m):
        token = m.group(1)
        if token == "answer":
            if answer is None:
                raise ValueError("⟪answer⟫ outside a statement form")
            return answer
        i = int(re.sub(r"\D", "", token))
        if i >= len(refs):
            raise ValueError(f"placeholder {token} has no reference")
        phrase = refs[i]
        if token.endswith("s"):
            phrase = phrase + "'s"
        if token[0] == "G":
            phrase = phrase[:1].upper() + phrase[1:]
        return phrase
    out = _PLACEHOLDER.sub(sub, text)
    if "⟪" in out or "⟫" in out:
        raise ValueError(f"unresolved placeholder in {out!r}")
    return out
