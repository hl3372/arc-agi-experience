"""Wording flags the assembler honours beyond ``pv2.model``'s FLAG_* constants.

The items producer measures them per wording (``pv2.wordings`` on main documents how):

* ``label_word:<w>`` for w in image / panel / map / figure / picture / #: never a label with that word.
* ``label_word:<X>`` with a single capital letter (the L / X / H / T / U shape names): that letter is
  never a label in that part ("Grid L" is skipped: Grid A .. Grid K, Grid M).
* ``label_word:grid_number`` ("Tile this grid 4 times"): no numeric labels in that part, and no
  "Grid N" label anywhere in the row (the text's "grid 4" must never read as a label).
* ``style_label:<answer|question|task>``: the text uses "Answer:" / "Question:" / "Task:" as an inline
  label, so the part's answer labels (and the row's question tags / lead clauses) avoid that word.
* ``inner_ordinals``: the text numbers parts of its own grid ("the first subgrid ... the second"), so
  the part names its grids by label, never by ordinal words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pv2.wordings import FLAG_INNER_ORDINALS, FLAG_STYLE_LABEL_PREFIX  # noqa: E402  (one source of truth)
GRID_NUMBER = "grid_number"
STYLE_LABEL_WORDS = ("answer", "question", "task")


def label_letters(q) -> set[str]:
    return {w for w in q.label_words() if len(w) == 1 and w.isalpha() and w.isupper()}


def style_words(q) -> set[str]:
    return {f[len(FLAG_STYLE_LABEL_PREFIX):].lower() for f in q.flags if f.startswith(FLAG_STYLE_LABEL_PREFIX)}


@dataclass(frozen=True)
class PartRules:
    banned_letters: frozenset  # letters that never end a label in this part (Grid L next to "an L shape")
    numbers_ok: bool          # numeric labels (Grid 4, #2, Image 3); role labels are exempt
    ordinals_ok: bool         # ordinal words for this part's grids
    banned_words: frozenset   # style-label words the part's own labels avoid


def part_rules(part) -> PartRules:
    qs = part.queries
    return PartRules(
        banned_letters=frozenset(x for q in qs for x in label_letters(q)),
        numbers_ok=not any(GRID_NUMBER in q.label_words() for q in qs),
        ordinals_ok=not any(FLAG_INNER_ORDINALS in q.flags for q in qs),
        banned_words=frozenset(w for q in qs for w in style_words(q)),
    )


def row_banned_words(parts) -> frozenset:
    return frozenset(w for p in parts for q in p.queries for w in style_words(q))


def row_grid_numbers_ok(parts) -> bool:
    return not any(GRID_NUMBER in q.label_words() for p in parts for q in p.queries)


def uses_word_as_label(text: str, words) -> bool:
    """"Answer 3:", "Question (b):", "Answer:" -- a word used as a label (followed by a colon)."""
    return any(re.search(rf"\b{w}\b(?:\s+(?:\d+|\(?[a-z]\)?|\(\d+\)))?\s*:", text, re.IGNORECASE) for w in words)


def contains_word(text: str, words) -> bool:
    return any(re.search(rf"\b{w}", text, re.IGNORECASE) for w in words)
