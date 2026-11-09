"""Machinery of the question library: question specs, phrasings, encoding checks and ``pose``.

A question (``Spec``) owns a sampler and 2-6 phrasings.  The sampler looks at the grid(s) it is
given, draws the question's parameters (a row, a colour, a direction ...) under the question's
guards, computes the LITERAL answer and returns a ``Draw``; ``pose`` then picks one phrasing and
fills its question text and its one statement form.  Nothing here decides how a grid is referred
to: texts carry the ⟪g0⟫ / ⟪G0⟫ / ⟪g0s⟫ placeholders of ``pv2.model``.

Conventions every question follows (tested in tests/test_library.py):

* coordinates are 0-indexed from the top-left; a question that uses a row or column number, or
  answers with one, says so in its own text and carries ``FLAG_COORDS``;
* a size question carries ``FLAG_DIMS``, switches off the dims header of every slot it reads, and
  is never asked of a grid whose encoding prints row or column labels or its size (stage2, runs,
  row_labelled, indexed_table, sparse): the last label gives the size away;
* a coordinate question may use the stage-2 rendering but no other coordinate-labelled encoding;
  questions whose answer the runs listing prints almost verbatim (positions, runs, run lengths and
  ends) use neither stage2 nor runs;
* colours in prose are ``encodings.colour_word(v, encoding)`` ("color 3" / "green"); a bare colour
  answer is the colour's label in the grid's own notation ("3" / "green"), so the statement form
  reads "... is color 3." / "... is green.";
* yes/no answers are the tokens "yes" / "no"; answers that name grids carry placeholders and list
  the slot indices in ``value``;
* statements are inflected by the answer (1 cell / 2 cells, found / none, yes / no) but each
  phrasing has exactly one statement form per answer case, and one answer-neutral declared form
  ("The number of rows of ⟪g0⟫ is ⟪answer⟫.") that a declared part writes into the prompt
  (``declared_template``: the statement itself when neutral, else ``Phrasing.declared``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from pv2 import encodings as enc
from pv2.model import FLAG_COORDS, FLAG_DIMS, FLAG_EDIT, FLAG_LONG, LABEL_WORD_PREFIX, Answer, Query

REGISTERS = ("imperative", "interrogative", "request", "terse", "embedded")

# Every coordinate question carries one of these clauses (drawn per question) or an inline clause
# of its own ("counting from 0 at the top").
ZERO_SENTENCES = (
    "Rows and columns are 0-indexed from the top-left corner.",
    "Rows and columns are numbered from 0, starting at the top-left corner.",
    "Row 0 is the top row and column 0 is the leftmost column.",
    "Count rows from the top and columns from the left, both starting at 0.",
)
ZERO_PARENS = (
    "(0-indexed from the top-left)",
    "(rows and columns counted from 0 at the top-left)",
    "(counting from 0, with (0,0) the top-left cell)",
)
# Markers by which a text shows it states the 0-indexed convention (the tests use them too).
ZERO_MARKERS = ("0-indexed", "from 0", "starting at 0", "Row 0 is the top row", "(0,0) the top-left")

LABEL_WORDS = ("image", "panel", "map", "figure", "picture", "left", "right", "top", "bottom")

# Placed in a part's context when an edit question shares a grid with another question.
AS_SHOWN = "Every question refers to the grid as shown, not to the result of another question."

LONG_TEXT = 170     # characters (placeholders counted as a typical reference) -> FLAG_LONG


@dataclass
class LibQuery(Query):
    """A library ``Query`` that also carries its drawn parameters (``params``): provenance, and
    what ``recompute`` needs to check the answer again from the grids (validator V3)."""
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Phrasing:
    """One wording.  ``text`` is the question; ``statement`` its one full-sentence form (a dict
    when the sentence is inflected by an answer case).  ``tag`` restricts the phrasing to draws of
    that tag (e.g. an ordering the answer depends on); ``only`` to "digits" or "names" grids.
    ``declared``: the answer-neutral sentence a declared part writes into the prompt (``Query.declared``);
    None = the statement itself when it is neutral (``declared_template``)."""
    register: str
    text: str
    statement: Any
    tag: Optional[str] = None
    only: Optional[str] = None
    declared: Optional[str] = None


@dataclass(frozen=True)
class Spec:
    qid: str                    # without the "lib:" prefix
    arity: Any                  # 1, 2 or "n" (two or more grids)
    kind: str                   # answer kind (pv2.model.KINDS)
    sample: Callable            # (rng, ctx) -> Draw | None
    phrasings: tuple
    coords: bool = False        # uses / answers with row or column numbers
    dims: bool = False          # asks about the size
    edit: bool = False          # grid answer that changes the grid
    forbid: frozenset = frozenset()   # encodings never used with this question
    names_ok: bool = True       # False when the answer notation needs digit colours
    weight: float = 1.0
    grid_answer: bool = False
    bounds_size: bool = False   # the text names a row / column index or a half / scale that bounds the size


@dataclass
class Draw:
    value: Any
    text: Optional[str]                 # None for grid answers
    fmt: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    case: str = "main"
    tag: Optional[str] = None
    presupposes: tuple = ()             # colours the question text takes to be present
    asks_presence: tuple = ()           # colours whose presence the answer settles


class Ctx:
    """The grids a question reads, with the colour vocabulary of their encoding."""

    def __init__(self, grids, encodings, reads=None):
        self.lists = [enc.check_grid(g) for g in grids]
        self.grids = [np.asarray(g, dtype=int) for g in self.lists]
        self.encodings = list(encodings)
        self.reads = tuple(reads) if reads is not None else tuple(range(len(grids)))
        self.names = enc.ENCODINGS[self.encodings[0]].uses_names
        self.mode = "names" if self.names else "digits"

    @property
    def g(self) -> np.ndarray:
        return self.grids[0]

    def cw(self, v) -> str:
        """Colour in prose: "color 3" / "green"."""
        return enc.colour_word(int(v), self.encodings[0])

    def cwc(self, v) -> str:
        w = self.cw(v)
        return w[:1].upper() + w[1:]

    def label(self, v) -> str:
        """A colour as a bare answer: "3" / "green"."""
        return enc.COLOR_NAMES[int(v)] if self.names else str(int(v))


# --- small helpers ----------------------------------------------------------------------------------

def plural(n: int, word: str, plural_form: Optional[str] = None) -> str:
    return word if n == 1 else (plural_form or word + "s")


def inflect(n: int) -> dict:
    """Format fields inflected by a count answer."""
    return {"cells": plural(n, "cell"), "rows": plural(n, "row"), "columns": plural(n, "column"),
            "colors": plural(n, "color"), "positions": plural(n, "position"),
            "is_are": "is" if n == 1 else "are", "them": "it" if n == 1 else "them"}


def pick(rng, seq, weights=None):
    seq = list(seq)
    if not seq:
        raise ValueError("pick from an empty sequence")
    if weights is None:
        return seq[int(rng.integers(len(seq)))]
    w = np.asarray(weights, dtype=float)
    return seq[int(rng.choice(len(seq), p=w / w.sum()))]


def colour_counts(g: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(g, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def first_appearance(g: np.ndarray) -> list[int]:
    seen, out = set(), []
    for v in g.ravel().tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def ref_list(indices, conj: str = "and") -> str:
    """"⟪g0⟫, ⟪g1⟫ and ⟪g2⟫" for placeholder indices."""
    refs = [f"⟪g{i}⟫" for i in indices]
    if len(refs) == 1:
        return refs[0]
    return ", ".join(refs[:-1]) + f" {conj} " + refs[-1]


def range_rivals(length: int, start: int, span: int) -> list[int]:
    """Start positions a misreading of lines start..start+span-1 lands on (the task-77 guard):
    1-indexed from the near edge, and counted from the far edge 0- and 1-indexed."""
    end = start + span - 1
    starts = (start - 1, length - 1 - end, length - end)
    return sorted({s for s in starts if 0 <= s <= length - span and s != start})


def cell_rivals(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    """Cells a misreading of (r, c) lands on: 1-indexed on both axes, far edges on both axes, and
    the transposed reading."""
    out = {(r - 1, c - 1), (h - 1 - r, w - 1 - c), (c, r)}
    return sorted((a, b) for a, b in out if 0 <= a < h and 0 <= b < w and (a, b) != (r, c))


def grid_list(a: np.ndarray) -> list[list[int]]:
    return [[int(v) for v in row] for row in np.asarray(a)]


# --- encoding checks ----------------------------------------------------------------------------------

def encoding_ok(spec: Spec, encodings) -> bool:
    """Whether a question may be asked of grids shown in these encodings."""
    uses = {enc.ENCODINGS[e].uses_names for e in encodings}
    if len(uses) != 1:
        return False            # colour words would be ambiguous across the grids
    for e in encodings:
        info = enc.ENCODINGS[e]
        if e in spec.forbid:
            return False
        if info.uses_names and not spec.names_ok:
            return False
        if spec.coords and info.reveals_coords and e != "stage2":
            return False
        if spec.dims and (info.reveals_dims or info.reveals_coords):
            return False        # row / column labels give the size away (stage2 included)
    return True


def encoding_weights(*, coords: bool = False, dims: bool = False, forbid=()) -> dict:
    """``encodings.DEFAULT_WEIGHTS`` without colour names, restricted to what a planned set of
    questions allows: the stage-2 rendering stays allowed for coordinate questions (DESIGN 13.1),
    but no labelled encoding is allowed for a size question (the labels give the size away)."""
    out = {}
    for name, w in enc.DEFAULT_WEIGHTS.items():
        info = enc.ENCODINGS[name]
        if name in forbid or info.uses_names:
            continue
        if coords and info.reveals_coords and name != "stage2":
            continue
        if dims and (info.reveals_dims or info.reveals_coords):
            continue
        out[name] = w
    return out


def choose_encoding(rng, grid, *, coords=False, dims=False, forbid=(), names=False) -> str:
    """One grid's encoding for questions planned in advance; ``names`` picks colour names (a
    part-level decision, so every grid of a question speaks the same colour vocabulary)."""
    if names:
        return "names"
    grid = grid_list(np.asarray(grid))
    weights = encoding_weights(coords=coords, dims=dims, forbid=forbid)
    return enc.choose(rng, grid, names_ok=False, coords_ok=True, dims_ok=not dims, background_ok=True,
                      weights=weights)


# --- building queries ---------------------------------------------------------------------------------

_FIELD = re.compile(r"\{(\w+)\}")


def _tidy(text: str) -> str:
    text = re.sub(r" {2,}", " ", text).strip()
    return text.replace(" ?", "?").replace(" .", ".").replace(" ,", ",").replace(" :", ":")


def label_word_flags(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return {LABEL_WORD_PREFIX + w for w in LABEL_WORDS if w in words or w + "s" in words}


def _long(text: str) -> bool:
    return len(re.sub(r"⟪[gG]\d+s?⟫", "the second grid", text)) > LONG_TEXT


# Declared sentences (DESIGN 13.7 / 15).  A sentence written into the prompt before the answer is known
# may only say what the question says: every format field it uses is one the question text uses (or
# the same value in another case), plus the colour-word prefix, which the encoding decides.  Fields
# that only statements use ({cells}, {is_are}, a chain's {k} / {Xw} ...) carry the answer, and a
# per-case dict is chosen by the answer, so neither is ever declared.
NEUTRAL_ALWAYS = frozenset({"C", "Cs"})
SAME_VALUE = (frozenset({"X", "Xc"}), frozenset({"refs", "Refs"}), frozenset({"def", "Def", "defs"}))


def neutral_template(template: str, question: str) -> bool:
    """Whether ``template`` uses only fields whose values ``question`` already states."""
    said = set(_FIELD.findall(question))
    for group in SAME_VALUE:
        if said & group:
            said |= group
    return set(_FIELD.findall(template)) <= said | NEUTRAL_ALWAYS


def declared_template(ph: "Phrasing") -> Optional[str]:
    """The phrasing's answer-neutral sentence template: its ``declared`` form, else its statement when
    that is one string using only neutral fields, else None.  Raises on a ``declared`` form that is not
    neutral (a wording bug, never a draw's)."""
    if ph.declared is not None:
        if not isinstance(ph.declared, str) or not neutral_template(ph.declared, ph.text):
            raise RuntimeError(f"declared form {ph.declared!r} uses a field its question does not state")
        return ph.declared
    if isinstance(ph.statement, str) and neutral_template(ph.statement, ph.text):
        return ph.statement
    return None


def build_query(rng, spec: Spec, draw: Draw, reads, slots, phrasing: Optional[int] = None,
                extra_flags=()) -> Optional[LibQuery]:
    """The query for a drawn question; None when a forced ``phrasing`` does not fit the draw."""
    ctx_mode = "names" if enc.ENCODINGS[slots[reads[0]].encoding].uses_names else "digits"
    allowed = [i for i, p in enumerate(spec.phrasings)
               if (p.tag is None or p.tag == draw.tag) and (p.only is None or p.only == ctx_mode)]
    if not allowed:
        raise RuntimeError(f"lib:{spec.qid}: no phrasing for tag {draw.tag!r} / {ctx_mode}")
    if phrasing is None:
        index = pick(rng, allowed)
    elif phrasing in allowed:
        index = phrasing
    else:
        return None
    ph = spec.phrasings[index]
    fields = {"C": "" if ctx_mode == "names" else "color ",
              "Cs": "" if ctx_mode == "names" else "colors ",
              "ZS": pick(rng, ZERO_SENTENCES), "ZP": pick(rng, ZERO_PARENS)}
    fields.update(draw.fmt)
    text = _tidy(ph.text.format(**fields))
    template = ph.statement if isinstance(ph.statement, str) else ph.statement[draw.case]
    statement = _tidy(template.format(**fields))
    neutral = declared_template(ph)
    declared = _tidy(neutral.format(**fields)) if neutral is not None else None
    flags = set(extra_flags)
    if spec.coords:
        flags.add(FLAG_COORDS)
    if spec.dims:
        flags.add(FLAG_DIMS)
        for i in reads:
            slots[i].dims_header_ok = False
    if spec.edit:
        flags.add(FLAG_EDIT)
    flags |= label_word_flags(text)
    if _long(text):
        flags.add(FLAG_LONG)
    answer = Answer(spec.kind, draw.value, draw.text)
    params = dict(draw.params)
    params.update(phrasing=index, register=ph.register, case=draw.case, tag=draw.tag,
                  presupposes=list(draw.presupposes), asks_presence=list(draw.asks_presence),
                  bounds_size=bool(spec.bounds_size))
    return LibQuery(source="lib", qid="lib:" + spec.qid, reads=tuple(reads), text=text, answer=answer,
                    statement=statement, flags=frozenset(flags), params=params, declared=declared)


def arity_ok(spec: Spec, n: int) -> bool:
    return spec.arity == n or (spec.arity == "n" and n >= 2)


def weighted_order(rng, specs):
    """A weighted random order (Efraimidis-Spirakis keys), deterministic from ``rng``."""
    if not specs:
        return []
    keys = rng.random(len(specs)) ** (1.0 / np.asarray([s.weight for s in specs], dtype=float))
    return [specs[i] for i in np.argsort(-keys, kind="stable")]
