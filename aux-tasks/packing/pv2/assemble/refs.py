"""Reference schemes: how a part's grids are named, and how a name resolves back to a grid.

Schemes (chosen per part by ``pv2.assemble.row``):

* ``label``   printed labels -- "Grid 1".."Grid N", "Grid A".., "Image N", "Panel N", "Map N",
              "Figure N", "#N" (numbers or letters); the reference phrase is the label itself.
              A label noun never appears in the row's texts (``Query.label_words()`` and a word
              scan), and labels are unique across the row (numbering continues in later parts).
* ``ordinal`` UNLABELLED galleries (2+ grids): ordinal WORDS only -- "the first grid" ..
              "the twelfth grid", "the last grid", "the second-to-last grid".  Printed numbers
              only ever refer to printed labels.
* ``none``    one unlabelled grid: "the grid" / "this grid" (scoped by its part).
* ``role``    ARC roles with the stage-2 labels verbatim: "Training pair k input/output",
              "Test input k" (0- or 1-based, as printed), plus "Candidate A", "Proposed grid"
              and letter labels for plain grids inside a role-bearing part.

In a multi-part row a phrase is QUALIFIED ("this part's third grid", "this part's Training pair 1
input") whenever its row-wide reading could differ from its part reading (``resolve``).
``resolve`` is shared by the assembler (to decide qualification) and the validator (to check
every phrase found in the rendered text).
"""
from __future__ import annotations

import re
from typing import Optional

from pv2.assemble._util import LETTERS, ORDINAL_WORDS, letter, pick, pick_weighted

LABEL_NOUNS = ("Grid", "Image", "Panel", "Map", "Figure", "#")
QUALIFIER_NOUNS = ("part", "section", "exercise")
MAX_ORDINAL = len(ORDINAL_WORDS)          # "the twelfth grid"

_ORD_ALT = "second-to-last|last|" + "|".join(ORDINAL_WORDS)
_QUAL = r"[Tt]his (?:part|section|exercise)'s"
# a reference phrase found in running text (the validator scans prompt and target with these)
ORDINAL_IN_TEXT = re.compile(rf"(?<![\w'-])(?:(?P<qual>{_QUAL}) |(?P<the>[Tt]he) )(?P<ord>{_ORD_ALT}) grid(?![\w-])")
QUALIFIER_BEFORE = re.compile(rf"(?:{_QUAL}) $")
_PHRASE_QUAL = re.compile(rf"^(?P<qual>{_QUAL}) (?P<body>.+)$")
_PHRASE_ORD = re.compile(rf"^(?:[Tt]he )?(?P<ord>{_ORD_ALT}) grid$")
SINGLE_PHRASES = ("the grid", "this grid")


# --- labels ----------------------------------------------------------------------------------------

def label_text(noun: str, style: str, k: int) -> str:
    """The k-th label (0-based position) of a noun and index style ("number" 1-based, "letter")."""
    tag = str(k + 1) if style == "number" else letter(k)
    return f"#{tag}" if noun == "#" else f"{noun} {tag}"


def noun_in_texts(noun: str, texts: list[str]) -> bool:
    """Whether a label noun occurs as a word (any case, singular or plural) in any of the texts."""
    if noun == "#":
        return any("#" in t for t in texts)
    pat = re.compile(rf"\b{re.escape(noun.lower())}s?\b")
    return any(pat.search(t.lower()) for t in texts)


class LabelBook:
    """Row-wide label registry: labels never repeat within a row."""

    def __init__(self):
        self.next: dict[tuple[str, str], int] = {}
        self.used_nouns: set[str] = set()
        self.used: set[str] = set()

    def free_letters(self, noun: str, banned=frozenset()) -> int:
        return sum(1 for x in LETTERS[self.next.get((noun, "letter"), 0):] if x not in banned)

    def take(self, noun: str, style: str, n: int, banned=frozenset()) -> list[str]:
        """The next n labels of a noun and style (letters skip ``banned``)."""
        k = self.next.get((noun, style), 0)
        out = []
        while len(out) < n:
            if style == "letter" and k >= len(LETTERS):
                raise ValueError("out of letters")
            if style == "number" or LETTERS[k] not in banned:
                out.append(label_text(noun, style, k))
            k += 1
        self.next[(noun, style)] = k
        self.used_nouns.add(noun)
        self.used.update(out)
        return out

    def assign(self, rng, n: int, nouns: list[str], weights: dict, letter_p: float,
               force_style: Optional[str] = None, numbers_ok: bool = True, banned_letters=frozenset(),
               grid_numbers_ok: bool = True) -> tuple[str, str, list[str]]:
        """Labels for n grids: a noun among ``nouns`` and an index style ("Grid N" also needs
        ``grid_numbers_ok``; letters skip ``banned_letters``); raises ValueError when nothing is left."""
        options: dict[str, list[str]] = {}
        for noun in nouns:
            styles = []
            if numbers_ok and not (noun == "Grid" and not grid_numbers_ok):
                styles.append("number")
            if noun != "#" and self.free_letters(noun, banned_letters) >= n:
                styles.append("letter")
            if styles:
                options[noun] = styles
        if not options:
            raise ValueError("no label noun and index style left for this part")
        table = {}
        for noun in options:
            w = weights.get(noun, 0.0)
            if noun not in self.used_nouns:
                w *= 3.0                      # prefer a fresh noun, else continue its numbering
            table[noun] = w
        noun = pick_weighted(rng, table) if any(table.values()) else pick(rng, sorted(options))
        styles = options[noun]
        if force_style in styles:
            style = force_style
        elif len(styles) == 1:
            style = styles[0]
        else:
            style = "letter" if rng.random() < letter_p else "number"
        return noun, style, self.take(noun, style, n, banned_letters if style == "letter" else frozenset())

    def reserve(self, labels: list[str]) -> None:
        self.used.update(labels)


def role_labels(slots, base: int, book: LabelBook, banned_letters=frozenset()) -> list[str]:
    """Stage-2 role labels for a role-bearing part (0- or 1-based as printed).  Candidates, proposed
    grids and plain grids take letters ("Candidate A", "Grid B"), skipping letters the part's texts
    use as words."""
    labels: list[Optional[str]] = [None] * len(slots)
    tests = [i for i, s in enumerate(slots) if s.role == "test_input"]
    cands = [i for i, s in enumerate(slots) if s.role == "candidate"]
    props = [i for i, s in enumerate(slots) if s.role == "proposed"]
    plain = [i for i, s in enumerate(slots) if s.role == "grid"]
    free = [x for x in LETTERS if x not in banned_letters]
    for i, s in enumerate(slots):
        if s.role in ("train_input", "train_output"):
            if s.pair is None:
                raise ValueError("a training slot needs its pair index")
            labels[i] = f"Training pair {s.pair + base} {'input' if s.role == 'train_input' else 'output'}"
    for k, i in enumerate(tests):
        pair = slots[i].pair if slots[i].pair is not None else k
        labels[i] = f"Test input {pair + base}"
    for k, i in enumerate(cands):
        labels[i] = f"Candidate {free[k]}"
    for k, i in enumerate(props):
        labels[i] = "Proposed grid" if len(props) == 1 else f"Proposed grid {free[k]}"
    if plain:
        # plain grids continue the row's "Grid" letters so a label never repeats
        for i, lab in zip(plain, book.take("Grid", "letter", len(plain), banned_letters)):
            labels[i] = lab
    if len(set(labels)) != len(labels):
        raise ValueError(f"role labels repeat within a part: {labels}")
    book.reserve(labels)
    return labels


# --- phrases ---------------------------------------------------------------------------------------

def ordinal_options(i: int, n: int) -> list[str]:
    opts = []
    if i < MAX_ORDINAL:
        opts.append(f"the {ORDINAL_WORDS[i]} grid")
    if n >= 2 and i == n - 1:
        opts.append("the last grid")
    if n >= 3 and i == n - 2:
        opts.append("the second-to-last grid")
    return opts


def ordinal_phrases(rng, n: int, end_p: float = 0.35) -> list[Optional[str]]:
    """One phrase per slot of an unlabelled gallery (None where no ordinal word reaches the slot)."""
    out = []
    for i in range(n):
        opts = ordinal_options(i, n)
        words = [o for o in opts if not o.startswith(("the last", "the second-to-last"))]
        ends = [o for o in opts if o not in words]
        if not opts:
            out.append(None)
        elif words and ends:
            out.append(pick(rng, ends) if rng.random() < end_p else words[0])
        else:
            out.append(words[0] if words else pick(rng, ends))
    return out


def qualify(phrase: str, noun: str) -> str:
    """"the third grid" -> "this part's third grid"; "Grid B" -> "this part's Grid B"."""
    body = phrase[4:] if phrase.startswith("the ") else phrase
    return f"this {noun}'s {body}"


# --- resolution ------------------------------------------------------------------------------------

def _ordinal_index(word: str, n: int) -> Optional[int]:
    if word == "last":
        return n - 1 if n >= 1 else None
    if word == "second-to-last":
        return n - 2 if n >= 2 else None
    i = ORDINAL_WORDS.index(word)
    return i if i < n else None


def resolve(phrase: str, part: Optional[int], row_labels: list[list[Optional[str]]]) -> set[tuple[int, int]]:
    """Every (part, slot) a careful reader could take ``phrase`` to mean, read in ``part`` (None: the
    row header, outside every part).

    ``row_labels[p][s]`` is the printed label of slot s of part p, or None.  An unqualified ordinal is
    read both within its part and across the whole row, and a label "Grid k" / "Grid <k-th letter>"
    anywhere in the row is a third reading of "the k-th grid"; a qualified phrase ("this part's ...")
    is read within its part only; a label is its printed label (unqualified: anywhere in the row).
    """
    flat = [(p, s) for p, labels in enumerate(row_labels) for s in range(len(labels))]
    qualified = False
    body = phrase
    m = _PHRASE_QUAL.match(phrase)
    if m:
        qualified, body = True, m.group("body")
        if part is None:
            return set()
    if body.lower() in SINGLE_PHRASES and not qualified:
        if part is None:
            return set(flat)
        return {(part, s) for s in range(len(row_labels[part]))}
    if qualified and body == "grid":
        return {(part, s) for s in range(len(row_labels[part]))}
    om = _PHRASE_ORD.match(body)
    if om and (qualified or body[:4].lower() == "the "):
        word = om.group("ord")
        found = set()
        if part is not None:
            i = _ordinal_index(word, len(row_labels[part]))
            if i is not None:
                found.add((part, i))
        if not qualified:
            i = _ordinal_index(word, len(flat))
            if i is not None:
                found.add(flat[i])
            if word not in ("last", "second-to-last"):
                k = ORDINAL_WORDS.index(word)
                same = {f"Grid {k + 1}", f"Grid {letter(k)}"}
                found |= {(p, s) for p, s in flat if row_labels[p][s] in same}
        return found
    scope = [(part, s) for s in range(len(row_labels[part]))] if qualified else flat
    return {(p, s) for p, s in scope if row_labels[p][s] == body}


def find_label_mentions(text: str, labels: list[str]) -> list[tuple[int, int, str, bool]]:
    """(start, end, label, qualified) for every mention of a printed label in ``text``."""
    out = []
    for label in sorted(set(labels), key=len, reverse=True):
        pat = re.compile(rf"(?<![A-Za-z0-9#]){re.escape(label)}(?![A-Za-z0-9])")
        for m in pat.finditer(text):
            if any(a <= m.start() < b for a, b, _, _ in out):
                continue          # inside a longer label already found
            qualified = bool(QUALIFIER_BEFORE.search(text[max(0, m.start() - 20):m.start()]))
            out.append((m.start(), m.end(), label, qualified))
    return sorted(out)


def header_text(label: str, grid=None) -> str:
    """"Grid 3" or, with the grid, "Grid 3 (5 rows x 7 cols)" (the stage-2 size notation)."""
    if grid is None:
        return label
    return f"{label} ({len(grid)} rows x {len(grid[0])} cols)"


DIMS_IN_HEADER = re.compile(r"\((\d+) rows x (\d+) cols\)")
