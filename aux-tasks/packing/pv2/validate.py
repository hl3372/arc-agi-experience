"""validate_row: the checks every assembled row passes before it is emitted (DESIGN.md sections 10, 13).

The validator reads the RENDERED prompt and target.  ``row.meta["checks"]`` only says where things
are claimed to be (part spans, block offsets, the reference phrases each question uses, the answer
style's shape); every claim is verified against the text and against the producers' gold ``parts``.

V1  every grid block parses back from the prompt text to its slot (canonical encoding, frame and
    header intact, header sizes true, blocks separated by blank lines, parts contiguous).
V2  every reference phrase in the prompt and the target -- ordinals, printed labels, role labels, with
    or without "this part's" -- resolves to exactly one grid, read both within its part and across the
    whole row; each question's phrases resolve to the slots it reads.
V3  question texts and answer strings equal the producers' gold after filling (grid answers parse back
    to the gold grid; answers naming grids name exactly the gold slots).
V4  the target parses back with each part's style parser to the expected answer strings, in order, and
    re-renders byte-identically; headings of multi-part rows in order; bare grid answers stay readable
    by stage-2's ``parse_grids``.
V5  hygiene: no leftover placeholders; exactly one reply shape per part and no contradicting phrases;
    the 0-indexed and "as shown" sentences where needed; no background naming; stated counts true;
    no perception-probe sentence; no dims header on a sized question; wording flags respected;
    questions numbered continuously; every sentence a sentence / paragraph instruction states is the
    target's sentence (in a declared part: each question's answer-neutral declared form).
Any failure raises ``ValidationError`` (``.check`` is "V1".."V5").
"""
from __future__ import annotations

import importlib.util
import os
import re

from pv2 import encodings as enc
from pv2.assemble import flags as F
from pv2.assemble import layouts as L
from pv2.assemble import refs as R
from pv2.assemble import styles as S
from pv2.assemble._util import WORD_TO_NUMBER, letter
from pv2.model import (FLAG_COORDS, FLAG_DIMS, FLAG_EDIT, FLAG_IO_NOUNS, FLAG_LONG, FLAG_NOTHING_ELSE,
                       FLAG_PLACEMENT, FLAG_STRUCTURED, fill)


class ValidationError(ValueError):
    def __init__(self, check: str, message: str):
        super().__init__(f"{check}: {message}")
        self.check = check


def _fail(check: str, message: str):
    raise ValidationError(check, message)


# --- banned and checked phrases ------------------------------------------------------------------------

PROBE_PATTERNS = tuple(re.compile(p) for p in (
    r"How many cells differ between Training pair \d+ input and Training pair \d+ output\?",
    r"What is the colou?r of the cell at row \d+, column \d+ \(zero-based\) of [^?\n]+\?",
    r"Write row \d+ \(zero-based\) of [^\n]+? as consecutive digits\.",
    r"What is the size of [^?\n]+\? Answer as HxW",
    r"Answer as HxW \(rows x columns\) only\.",
    r"How many cells of colou?r \d+ does [^?\n]+ have\? Answer with the number only\.",
    r"Answer the question about the grids below\. They are shown with zero-based row labels",
))
_COLOUR_WORD = r"(?:\d|black|blue|red|green|yellow|gr[ae]y|magenta|orange|cyan|maroon)"
BACKGROUND_NAMING = re.compile(
    rf"\bbackground(?: colou?r)?(?: of [^.\n]{{1,40}}?)? (?:is|was|=|will be) (?:colou?r )?{_COLOUR_WORD}\b",
    re.IGNORECASE)
_NUM = r"(?P<n>\d+|" + "|".join(WORD_TO_NUMBER) + r")"
COUNT_CLAIMS = tuple(re.compile(p) for p in (
    rf"\b(?:Here are|You will see|Consider these|There are|This prompt has) {_NUM} (?P<noun>grids?|questions?|parts?|sections?|exercises?)\b",
    rf"\bwith {_NUM} (?P<noun>questions?) in all\b",
    rf"\bThen answer {_NUM} (?P<noun>questions?)\b",
    rf"\bNow answer the {_NUM} (?P<noun>questions?)\b",
))
SINGLE_GRID_CLAIMS = re.compile(r"\b(?:Here is a grid|Consider the following grid|Look at this grid|Take this grid)\.")
BOUNDARY_CLAIM = re.compile(r"\b(?P<noun>Part|Section|Exercise) (?P<k>\d+)(?: of |/)(?P<n>\d+)\b")
ZERO_INDEX = L.ZERO_INDEX_RE


def _number(word: str) -> int:
    return int(word) if word.isdigit() else WORD_TO_NUMBER[word]


# --- helpers -------------------------------------------------------------------------------------------

def _part_segments(row) -> list[tuple[int, int]]:
    return [tuple(p["span"]) for p in row.meta["checks"]["parts"]]


def _block_regions(row) -> list[tuple[int, int]]:
    out = []
    for p in row.meta["checks"]["parts"]:
        for b in p["blocks"]:
            pre, suf = L.block_frame(b["header"], b["delimiter"])
            out.append((b["start"] - len(pre), b["end"] + len(suf)))
    return out


def _text_outside(text: str, start: int, end: int, holes: list[tuple[int, int]]) -> list[tuple[int, str]]:
    """Pieces of text[start:end] outside the holes, as (offset, piece)."""
    out, cur = [], start
    for a, b in sorted(holes):
        if b <= start or a >= end:
            continue
        if a > cur:
            out.append((cur, text[cur:a]))
        cur = max(cur, b)
    if cur < end:
        out.append((cur, text[cur:end]))
    return out


def _target_sections(row) -> list[str]:
    ch = row.meta["checks"]
    if not ch["multi"]:
        return [row.target]
    headings = [p["heading"] for p in ch["parts"]]
    t = row.target
    sections, pos = [], 0
    for i, h in enumerate(headings):
        if not t.startswith(h + "\n", pos):
            _fail("V4", f"heading {h!r} missing or out of place")
        pos += len(h) + 1
        if i + 1 < len(headings):
            nxt = "\n\n" + headings[i + 1] + "\n"
            end = t.find(nxt, pos)
            if end < 0:
                _fail("V4", f"heading {headings[i + 1]!r} missing")
            sections.append(t[pos:end])
            pos = end + 2
        else:
            sections.append(t[pos:])
    for h in headings:
        if sum(1 for line in t.split("\n") if line == h) != 1:
            _fail("V4", f"heading {h!r} does not occur exactly once")
    return sections


def _capital(p: str) -> str:
    return p[:1].upper() + p[1:]


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase in text or _capital(phrase) in text


# --- V1 -------------------------------------------------------------------------------------------------

def _v1(row, parts):
    ch = row.meta["checks"]
    prompt = row.prompt
    if len(ch["parts"]) != len(parts):
        _fail("V1", "checks describe a different number of parts")
    cur = 0
    if ch["multi"]:
        header = ch["row_header"]
        if not header or not prompt.startswith(header + "\n\n"):
            _fail("V1", "row header missing")
        cur = len(header) + 2
    for pi, (pc, part) in enumerate(zip(ch["parts"], parts)):
        a, b = pc["span"]
        if a != cur:
            _fail("V1", f"part {pi + 1} does not start where the previous one ended")
        if ch["multi"]:
            if not prompt.startswith(pc["boundary"] + "\n\n", a):
                _fail("V1", f"part {pi + 1} boundary line missing")
        cur = b + 2
        blocks = pc["blocks"]
        if [bl["slot"] for bl in blocks] != list(range(len(part.slots))):
            _fail("V1", f"part {pi + 1}: blocks do not show every slot once, in order")
        last = a
        for bl, slot in zip(blocks, part.slots):
            s, e = bl["start"], bl["end"]
            if not (a <= s <= e <= b) or s < last:
                _fail("V1", f"part {pi + 1}: block of slot {bl['slot']} outside its part or out of order")
            if bl["encoding"] != slot.encoding:
                _fail("V1", "block encoding differs from the slot's")
            try:
                grid = enc.parse(prompt[s:e], slot.encoding)
            except (ValueError, TypeError) as err:
                _fail("V1", f"part {pi + 1} slot {bl['slot']}: grid does not parse ({err})")
            if grid != slot.grid:
                _fail("V1", f"part {pi + 1} slot {bl['slot']}: grid differs from the slot")
            pre, suf = L.block_frame(bl["header"], bl["delimiter"])
            if prompt[s - len(pre):s] != pre or prompt[e:e + len(suf)] != suf:
                _fail("V1", f"part {pi + 1} slot {bl['slot']}: block frame broken")
            bs, be = s - len(pre), e + len(suf)
            if not (bs == 0 or prompt[bs - 2:bs] == "\n\n") or not (be == len(prompt) or prompt[be:be + 2] == "\n\n"):
                _fail("V1", f"part {pi + 1} slot {bl['slot']}: block not separated by blank lines")
            header = bl["header"]
            if bl["label"] is not None:
                if header is None or not (header == bl["label"] or header.startswith(bl["label"] + " (")):
                    _fail("V1", f"label {bl['label']!r} not printed in its header")
            if header is not None:
                m = R.DIMS_IN_HEADER.search(header)
                if m and (int(m.group(1)), int(m.group(2))) != (len(slot.grid), len(slot.grid[0])):
                    _fail("V1", f"header {header!r} states a wrong size")
            last = be
    if cur - 2 != len(prompt):
        _fail("V1", "text after the last part")


# --- V2 -------------------------------------------------------------------------------------------------

def _row_labels(row) -> list[list]:
    return [[bl["label"] for bl in pc["blocks"]] for pc in row.meta["checks"]["parts"]]


def _scan(text: str, offset_scope, row_labels, noun, where: str):
    """Resolve every ordinal / label phrase in ``text``; ``offset_scope(i)`` gives the part of offset i."""
    labels = sorted({lab for part in row_labels for lab in part if lab is not None})
    found = []
    for m in R.ORDINAL_IN_TEXT.finditer(text):
        found.append((m.start(), m.group(0), m.group("qual")))
    for s, e, label, qualified in R.find_label_mentions(text, labels):
        if qualified:
            q = R.QUALIFIER_BEFORE.search(text[max(0, s - 20):s])
            qual = q.group(0).strip()
            found.append((s, f"{qual} {label}", qual))
        else:
            found.append((s, label, None))
    for off, phrase, qual in found:
        part = offset_scope(off)
        if qual is not None:
            if part is None:
                _fail("V2", f"{where}: qualified phrase {phrase!r} outside any part")
            if noun is None or qual.split()[1] != f"{noun}'s":
                _fail("V2", f"{where}: qualifier {qual!r} does not name the row's {noun!r}s")
        got = R.resolve(phrase, part, row_labels)
        if len(got) != 1:
            _fail("V2", f"{where}: {phrase!r} resolves to {sorted(got)} (part {part})")


def _v2(row, parts):
    ch = row.meta["checks"]
    rl = _row_labels(row)
    prompt = row.prompt
    for pi, (pc, part) in enumerate(zip(ch["parts"], parts)):
        a, b = pc["span"]
        seg = prompt[a:b]
        for d in pc["questions"]:
            if d["text"] not in seg:
                _fail("V2", f"question {d['tag']} not printed in its part")
            q = part.queries[d["query"]]
            if not d["refs"]:
                if len(part.slots) != 1:
                    _fail("V2", f"question {d['tag']} names no grid in a part with several")
            elif {s for _, s in d["refs"]} != set(q.reads):
                _fail("V2", f"question {d['tag']} does not name every grid it reads")
            for phrase, slot in d["refs"] + d.get("answer_refs", []):
                if not _contains_phrase(d["text"], phrase) and not _contains_phrase(d["answer"], phrase):
                    _fail("V2", f"phrase {phrase!r} not in question {d['tag']}")
                if phrase.lower() in R.SINGLE_PHRASES:
                    if len(part.slots) != 1:
                        _fail("V2", f"{phrase!r} in a part with several grids")
                    continue
                got = R.resolve(phrase, pi, rl)
                if got != {(pi, slot)}:
                    _fail("V2", f"{phrase!r} in question {d['tag']} resolves to {sorted(got)}, not {(pi, slot)}")
    spans = _part_segments(row)

    def scope(off):
        for i, (a, b) in enumerate(spans):
            if a <= off < b:
                return i
        return None
    for off, piece in _text_outside(prompt, 0, len(prompt), _block_regions(row)):
        _scan(piece, lambda i, off=off: scope(off + i), rl, ch["noun"], "prompt")
    for pi, section in enumerate(_target_sections(row)):
        _scan(section, lambda i, pi=pi: pi, rl, ch["noun"], "target")


# --- V3 -------------------------------------------------------------------------------------------------

_NAMES_ONLY = re.compile(r"⟪[gG]\d+⟫(?:(?:, | and |, and )⟪[gG]\d+⟫)*")


def _v3(row, parts):
    ch = row.meta["checks"]
    for pi, (pc, part) in enumerate(zip(ch["parts"], parts)):
        phrases = pc["phrases"]
        if sorted(d["query"] for d in pc["questions"]) != list(range(len(part.queries))):
            _fail("V3", f"part {pi + 1}: questions missing or repeated")
        grid_enc = pc["grid_answer_encoding"]
        # the producer's context lines, filled, each on a line of its own (tables and menus stay lists)
        a, b = pc["span"]
        seg_lines = set(row.prompt[a:b].split("\n"))
        for c in part.context:
            line = fill(c, [ph if ph is not None else "" for ph in phrases])
            if any(x not in seg_lines for x in line.split("\n")):
                _fail("V3", f"part {pi + 1}: context line {line[:60]!r} is not printed on a line of its own")
        for d in pc["questions"]:
            q = part.queries[d["query"]]
            refs = [phrases[s] for s in q.reads]
            body = fill(q.text, refs)
            if d["text"] != body:
                leads = [fill(L.lead_clause(f, len(q.reads)), refs) + " " + body for f in L.LEAD_FORMS]
                if not (q.lead_clause and len(part.slots) > 1 and d["text"] in leads):
                    _fail("V3", f"question {d['tag']} is not the producer's text")
            if q.answer.kind == "grid":
                if grid_enc is None:
                    _fail("V3", "grid answer without a grid-answer encoding")
                try:
                    ok = enc.parse(d["answer"], grid_enc) == q.answer.value
                except (ValueError, TypeError):
                    ok = False
                if not ok or d["answer"] != enc.render(q.answer.value, grid_enc):
                    _fail("V3", f"answer {d['tag']} is not the gold grid")
            else:
                if d["answer"] != fill(q.answer.text, refs):
                    _fail("V3", f"answer {d['tag']} differs from the gold {q.answer.text!r}")
                named = [i for i in range(len(q.reads))
                         if any(f"⟪{t}⟫" in q.answer.text for t in (f"g{i}", f"G{i}", f"g{i}s", f"G{i}s"))]
                # only an answer made of grid names alone ("⟪g0⟫ and ⟪g2⟫") carries slot ids in its value;
                # "⟪g0⟫: 3, ⟪g1⟫: 5" (a value per grid) carries the values
                names_only = bool(_NAMES_ONLY.fullmatch(q.answer.text))
                if named and names_only and isinstance(q.answer.value, (list, tuple)) and \
                        all(isinstance(v, int) for v in q.answer.value):
                    if {q.reads[i] for i in named} != set(q.answer.value):
                        _fail("V3", f"answer {d['tag']} names other grids than its gold value")


# --- V4 -------------------------------------------------------------------------------------------------

_STAGE2 = []


def _stage2_parse_grids():
    if not _STAGE2:
        from pv2.paths import SOLVER_PROMPT_DIR
        path = os.path.join(SOLVER_PROMPT_DIR, "arc_render.py")
        fn = None
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("_pv2_arc_render", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = mod.parse_grids
        _STAGE2.append(fn)
    return _STAGE2[0]


def _v4(row, parts):
    ch = row.meta["checks"]
    sections = _target_sections(row)
    for pi, (pc, part, section) in enumerate(zip(ch["parts"], parts, sections)):
        spec = S.StyleSpec.from_dict(pc["style"])
        expected = [d["answer"] for d in pc["questions"]]
        if [s.tag for s in spec.shapes] != [d["tag"] for d in pc["questions"]]:
            _fail("V4", f"part {pi + 1}: the style's tags differ from the questions'")
        for s, d in zip(spec.shapes, pc["questions"]):
            q = part.queries[d["query"]]
            if s.kind != q.answer.kind or s.grid != (q.answer.kind == "grid"):
                _fail("V4", f"part {pi + 1}: the style's answer kinds differ from the gold")
        try:
            got = S.parse(spec, section)
        except (ValueError, KeyError, TypeError) as err:
            _fail("V4", f"part {pi + 1}: target does not parse as {spec.style} ({err})")
        if got != expected:
            _fail("V4", f"part {pi + 1}: parsed answers {got!r} differ from {expected!r}")
        if S.render(spec, got) != section:
            _fail("V4", f"part {pi + 1}: target is not the canonical {spec.style} rendering")
        if spec.style == "bare" and spec.shapes[0].grid:
            pg = _stage2_parse_grids()
            if pg is not None and pg(section) != [part.queries[pc["questions"][0]["query"]].answer.value]:
                _fail("V4", "bare grid answer unreadable by stage-2 parse_grids")


# --- V5 -------------------------------------------------------------------------------------------------

# The solver's own channels (training session, 2026-09-29): the render_output arm splits the reply on
# the last line that is exactly "Final answer:", the two-attempt arm on "Attempt k:" lines, every arm
# on </think>; aux rows must never contain those, nor any sentence of the solver's system prompts
# (the system prompt is what tells the model it is in the solver context).
SOLVER_MARKER_LINE = re.compile(r"^(?:Final answer:|Attempt \d+:)\s*$", re.M)
SOLVER_THINK_TAGS = ("<think>", "</think>")
_SOLVER_SENTENCES: list = []


def _solver_sentences() -> list:
    if not _SOLVER_SENTENCES:
        from pv2.paths import SOLVER_PROMPT_DIR
        d = os.path.join(SOLVER_PROMPT_DIR, "prompts")
        found = []
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.endswith(".md"):
                    text = open(os.path.join(d, name), encoding="utf-8").read()
                    found += [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n", text) if len(x.strip()) >= 40]
        _SOLVER_SENTENCES.append(tuple(sorted(set(found))))
    return list(_SOLVER_SENTENCES[0])


def _v5(row, parts):
    ch = row.meta["checks"]
    prompt, target = row.prompt, row.target
    for where, text in (("prompt", prompt), ("target", target)):
        if "⟪" in text or "⟫" in text:
            _fail("V5", f"leftover placeholder in the {where}")
        if SOLVER_MARKER_LINE.search(text) or any(tag in text for tag in SOLVER_THINK_TAGS):
            _fail("V5", f"a solver channel marker in the {where}")
    for sentence in _solver_sentences():
        if sentence in prompt:
            _fail("V5", f"a sentence of the solver's system prompt in the prompt: {sentence[:60]!r}")
    # a grid answer in digits read from a grid shown in colour names needs the mapping in its part
    for pc, part in zip(row.meta["checks"]["parts"], parts):
        genc = pc.get("grid_answer_encoding")
        if genc is None or genc == "names":
            continue
        if any(part.slots[i].encoding == "names" for q in part.queries if q.answer.kind == "grid"
               for i in q.reads) and L.NAMES_TO_DIGITS_MAP not in pc["instructions"]:
            _fail("V5", "a names-encoded grid feeds a digit grid answer without the name-to-digit mapping")
    for pat in PROBE_PATTERNS:
        if pat.search(prompt):
            _fail("V5", f"perception-probe sentence in the prompt: {pat.pattern}")
    m = BACKGROUND_NAMING.search(prompt)
    if m:
        _fail("V5", f"the prompt names the background: {m.group(0)!r}")
    blocks = _block_regions(row)
    spans = _part_segments(row)
    n_q = sum(len(p.queries) for p in parts)

    # stated counts
    for off, piece in _text_outside(prompt, 0, len(prompt), blocks):
        for pat in COUNT_CLAIMS:
            for mm in pat.finditer(piece):
                n = _number(mm.group("n"))
                noun = mm.group("noun").rstrip("s")
                start = off + mm.start()
                pi = next((i for i, (a, b) in enumerate(spans) if a <= start < b), None)
                if noun == "grid":
                    true = len(parts[pi].slots) if pi is not None else sum(len(p.slots) for p in parts)
                elif noun == "question":
                    true = len(parts[pi].queries) if pi is not None else n_q
                else:
                    true = len(parts)
                if n != true:
                    _fail("V5", f"false count {mm.group(0)!r} (true: {true})")
        for mm in SINGLE_GRID_CLAIMS.finditer(piece):
            pi = next((i for i, (a, b) in enumerate(spans) if a <= off + mm.start() < b), None)
            if pi is None or len(parts[pi].slots) != 1:
                _fail("V5", f"{mm.group(0)!r} but the part shows several grids")
    for pi, pc in enumerate(ch["parts"]):
        if pc["boundary"] is not None:
            mm = BOUNDARY_CLAIM.search(pc["boundary"])
            if not mm or int(mm.group("k")) != pi + 1 or int(mm.group("n")) != len(parts) \
                    or mm.group("noun").lower() != ch["noun"]:
                _fail("V5", f"boundary {pc['boundary']!r} is wrong")

    # continuous numbering
    tags = [d["tag"] for pc in ch["parts"] for d in pc["questions"]]
    if n_q == 1:
        if tags != [None]:
            _fail("V5", "a lone question carries a tag")
    elif tags != [str(i + 1) for i in range(n_q)] and tags != [letter(i).lower() for i in range(n_q)]:
        _fail("V5", f"questions are not numbered continuously: {tags}")

    row_labels = _row_labels(row)
    label_nouns = {lab.split(" ")[0] if not lab.startswith("#") else "#" for part in row_labels for lab in part
                   if lab is not None}
    words = {w.lower() for p in parts for q in p.queries for w in q.label_words()}
    for noun in label_nouns:
        if any(re.search(rf"(?:^|\W){re.escape(noun.lower())}s?(?:$|\W)", w) for w in words):
            _fail("V5", f"label noun {noun!r} is a flagged label word of the row")

    _flag_rules(row, parts)
    _row_conventions(row, parts)
    _stated_sentences(row, parts)

    for pi, (pc, part) in enumerate(zip(ch["parts"], parts)):
        a, b = pc["span"]
        seg_pieces = _text_outside(prompt, a, b, blocks)
        seg = "\n".join(p for _, p in seg_pieces)
        spec = S.StyleSpec.from_dict(pc["style"])
        style = spec.style
        instr = pc["instructions"]
        if instr not in seg:
            _fail("V5", f"part {pi + 1}: its answer-style instruction is not printed")
        if not S.SIGNATURES[style].search(instr):
            _fail("V5", f"part {pi + 1}: the instruction does not declare the {style} style")
        for other, sig in S.SIGNATURES.items():
            if other != style and sig.search(seg):
                _fail("V5", f"part {pi + 1}: reply-shape phrase of style {other!r} next to {style!r}")
        if style != "bare" and S.BARE_ONLY_PHRASES.search(seg.replace(instr, "")):
            _fail("V5", f"part {pi + 1}: a bare-answer demand next to the {style} style")
        if seg.count(instr) != 1 or sum(1 for s2 in S.STYLES if S.SIGNATURES[s2].search(instr)) != 1:
            _fail("V5", f"part {pi + 1}: the reply shape is declared ambiguously")

        qs = [part.queries[d["query"]] for d in pc["questions"]]
        if any(FLAG_COORDS in q.flags for q in qs) and not ZERO_INDEX.search(seg):
            _fail("V5", f"part {pi + 1}: coordinates without the 0-indexed sentence")
        shared_edit = any(FLAG_EDIT in q.flags and any(set(q.reads) & set(o.reads) for o in qs if o is not q)
                          for q in qs)
        if shared_edit and not L.AS_SHOWN_RE.search(seg):
            _fail("V5", f"part {pi + 1}: an edit shares a grid without the 'as shown' sentence")

        classes = [S.answer_class(q.answer.kind, q.answer.text) for q in qs]
        structured = [FLAG_STRUCTURED in q.flags or c == "multiline" for q, c in zip(qs, classes)]
        nothing = [FLAG_NOTHING_ELSE in q.flags for q in qs]
        restate_ok = all(q.source == "lib" and FLAG_LONG not in q.flags for q in qs)
        if not S.style_allows(style, classes, structured=structured, nothing_else=nothing, restate_ok=restate_ok):
            _fail("V5", f"part {pi + 1}: style {style!r} outside the style x kind matrix")
        if any(nothing) and len(parts) != 1:
            _fail("V5", "a nothing-else wording in a multi-part row")
        if style == "bare" and spec.shapes[0].grid and pc["grid_answer_encoding"] != "compact":
            _fail("V5", "a bare grid answer that is not compact")
        roles = any(s.role != "grid" for s in part.slots)
        if roles and any(FLAG_IO_NOUNS in q.flags for q in qs):
            _fail("V5", f"part {pi + 1}: io_nouns wording in a role-bearing part")
        for d, q in zip(pc["questions"], qs):
            if FLAG_PLACEMENT in q.flags:
                if len(part.slots) != 1:
                    _fail("V5", "a placement wording in a multi-grid part")
                qpos = prompt.find(d["text"], a, b)
                if not (0 <= qpos < pc["blocks"][0]["start"]):
                    _fail("V5", "a placement wording after its grid")
        for bl in pc["blocks"]:
            if bl["header"] and R.DIMS_IN_HEADER.search(bl["header"]):
                slot = part.slots[bl["slot"]]
                sized = any(FLAG_DIMS in q.flags and bl["slot"] in q.reads for q in qs)
                if not slot.dims_header_ok or sized:
                    _fail("V5", f"part {pi + 1}: a size header on a grid whose size is asked")


_LETTER_LABEL = re.compile(r"(?:^| )[A-Z]$")
_NUMBER_LABEL = re.compile(r"^(?:(?:Grid|Image|Panel|Map|Figure|Candidate|Proposed grid) \d+|#\d+)$")


def _flag_rules(row, parts):
    """The wording flags of pv2.assemble.flags: label letters / numbers, style-label words, inner ordinals."""
    ch = row.meta["checks"]
    banned_row = F.row_banned_words(parts)
    if ch["tag_format"] and F.contains_word(ch["tag_format"], banned_row):
        _fail("V5", f"question tags {ch['tag_format']!r} use a word an item text uses as a label")
    for pc in ch["parts"]:
        if pc["heading"] and "answer" in banned_row and "answers" in pc["heading"]:
            _fail("V5", "a heading uses 'answers' next to an item's own 'Answer:' label")
    all_labels = [bl["label"] for pc in ch["parts"] for bl in pc["blocks"] if bl["label"]]
    if not F.row_grid_numbers_ok(parts) and any(re.fullmatch(r"Grid \d+", lab) for lab in all_labels):
        _fail("V5", "a 'Grid N' label in a row whose texts say 'grid <number>'")
    for pi, (pc, part) in enumerate(zip(ch["parts"], parts)):
        rules = F.part_rules(part)
        labels = [bl["label"] for bl in pc["blocks"] if bl["label"]]
        for lab in labels:
            m = _LETTER_LABEL.search(lab)
            if m and m.group(0).strip() in rules.banned_letters:
                _fail("V5", f"part {pi + 1}: label {lab!r} next to a text that uses that letter as a word")
        if not rules.numbers_ok and any(_NUMBER_LABEL.match(lab) for lab in labels):
            _fail("V5", f"part {pi + 1}: numeric labels next to a text that says 'grid <number>'")
        if not rules.ordinals_ok:
            for d in pc["questions"]:
                if any(R.ORDINAL_IN_TEXT.search(ph) for ph, _ in d["refs"]):
                    _fail("V5", f"part {pi + 1}: ordinal references next to a text with its own ordinals")
        if rules.banned_words:
            spec = S.StyleSpec.from_dict(pc["style"])
            templates = [v for k, v in spec.variant.items() if k in ("label", "key", "q", "a", "item", "tag")]
            if F.uses_word_as_label(S.skeleton(spec), rules.banned_words) or \
                    any(F.contains_word(v, rules.banned_words) for v in templates):
                _fail("V5", f"part {pi + 1}: the reply labels use a word the item text uses as a label")
        if banned_row:
            for d in pc["questions"]:
                q = part.queries[d["query"]]
                body = fill(q.text, [pc["phrases"][s] for s in q.reads])
                lead = d["text"][:len(d["text"]) - len(body)]
                if F.contains_word(lead, banned_row):
                    _fail("V5", f"question {d['tag']}: the lead clause uses a word an item text uses as a label")


_ROLE_NUMBER = re.compile(r"^(?:Training pair (\d+) (?:input|output)|Test input (\d+))$")


def _row_conventions(row, parts):
    """One stage-2 numbering base per row; letter question tags printed "(a)" in prose; no "<grid>"
    placeholder next to <grid> tag frames; a bare instruction under a heading claims no exclusivity."""
    ch = row.meta["checks"]
    bases = set()
    for pc, part in zip(ch["parts"], parts):
        for bl in pc["blocks"]:
            m = _ROLE_NUMBER.match(bl["label"] or "")
            slot = part.slots[bl["slot"]]
            if m and slot.pair is not None:
                bases.add(int(m.group(1) or m.group(2)) - slot.pair)
    if len(bases) > 1:
        _fail("V5", f"stage-2 labels of one row mix numbering bases {sorted(bases)}")
    tags = [d["tag"] for pc in ch["parts"] for d in pc["questions"] if d["tag"]]
    if tags and not tags[0].isdigit():
        letters = "".join(tags)
        if re.search(rf"\bquestions? [{letters}](?![\w)])", row.prompt):
            _fail("V5", "a letter question tag is named without its parentheses")
    for pi, pc in enumerate(ch["parts"]):
        if pc["delimiter"] == "tags" and S.GRID_PLACEHOLDER in pc["instructions"]:
            _fail("V5", f"part {pi + 1}: a {S.GRID_PLACEHOLDER} placeholder next to <grid> tag frames")
        if ch["multi"] and pc["style"]["style"] == "bare" and \
                re.search(r"\balone\b|no other text|nothing else", pc["instructions"]):
            _fail("V5", f"part {pi + 1}: a bare instruction claims exclusivity under a heading line")


def _stated_sentences(row, parts):
    """Sentence / paragraph parts: every sentence the instruction states (all of them in a declared part,
    the generic ones otherwise) is printed there verbatim, so the target's sentence -- the style's
    rendering, V4 -- is exactly the stated text with the answer in place; in a declared part each
    question's sentence is its producer's answer-neutral ``declared`` form, the generic form only for a
    question without one."""
    for pi, (pc, part) in enumerate(zip(row.meta["checks"]["parts"], parts)):
        spec = S.StyleSpec.from_dict(pc["style"])
        if spec.style not in ("sentence", "paragraph"):
            continue
        stated = S.declarations(spec)
        if stated and stated.strip() not in pc["instructions"]:
            _fail("V5", f"part {pi + 1}: a stated sentence is not in the instruction or differs from the target's")
        if not spec.variant.get("declared"):
            continue
        for s, d in zip(spec.shapes, pc["questions"]):
            q = part.queries[d["query"]]
            refs = [pc["phrases"][i] for i in q.reads]
            want = S.sentence_pieces(q.declared, refs, s.grid, lead_in=True)
            if want is None:
                if not s.generic:
                    _fail("V5", f"question {d['tag']}: a declared part states a sentence its question has no "
                                "declared form for")
            elif s.generic or (s.prefix, s.suffix) != want:
                _fail("V5", f"question {d['tag']}: the declared sentence is not the question's declared form")


def validate_row(row, parts) -> None:
    """Raise ValidationError unless the row passes V1-V5."""
    parts = list(parts)
    _v1(row, parts)
    _v2(row, parts)
    _v3(row, parts)
    _v4(row, parts)
    _v5(row, parts)
