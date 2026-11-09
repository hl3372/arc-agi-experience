"""Answer styles: the reply-shape instruction, the target rendering, and its parser.

A style works on a ``StyleSpec``: everything about the reply's SHAPE (question tags, per-answer kind
and JSON type, statement prefix/suffix, restated question texts, the chosen variant) but no answer
text.  ``render(spec, texts)`` writes the target from the answer strings; ``parse(spec, target)``
reads them back (it never sees the gold); ``instruction(rng, spec)`` declares the shape in the
prompt (structured styles show their exact skeleton, rendered from the same code as the target).

Every style except JSON is a sequence of literal pieces around the answers
(``target = L0 + a1 + L1 + ... + an + Ln``); the parser matches the literals in order.  The assembler
renders, parses and compares before it accepts a style for a part, so an answer that would collide
with a literal (e.g. contains "; " in a chain line) never reaches a row.

The style x answer-kind matrix (``KIND_MATRIX``, ``PART_RULES``) is explicit and completeness-tested.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from pv2.assemble._util import join_tags, pick
from pv2.assemble.flags import contains_word
from pv2.encodings import parse as parse_grid, render as render_grid
from pv2.model import fill

STYLES = ("bare", "labelled", "sentence", "paragraph", "json", "kv", "chain", "bullets", "restate")

SCALAR_KINDS = ("number", "color", "bool", "token", "text", "list")
ANSWER_CLASSES = SCALAR_KINDS + ("grid", "multiline")   # "multiline": non-grid text spanning lines

# Which answer classes each style can carry.  Grid answers always stand alone on their own lines
# (after a label, a statement ending in a colon, or "A:"), never inside prose; the one-line styles
# (paragraph, key: value, chain, bullets) take one-line answers only.
KIND_MATRIX = {
    "bare": set(ANSWER_CLASSES),
    "labelled": set(ANSWER_CLASSES),
    "json": set(ANSWER_CLASSES),
    "sentence": set(SCALAR_KINDS) | {"grid"},
    "restate": set(SCALAR_KINDS) | {"grid"},
    "paragraph": set(SCALAR_KINDS),
    "kv": set(SCALAR_KINDS),
    "chain": set(SCALAR_KINDS),
    "bullets": set(SCALAR_KINDS),
}
STRUCTURED_OK = frozenset({"bare", "labelled", "json"})      # FLAG_STRUCTURED: verbatim blocks only
NOTHING_ELSE_OK = frozenset({"bare"})                          # FLAG_NOTHING_ELSE: the bare answer only
MIN_ANSWERS = {"chain": 2, "bullets": 2, "paragraph": 2}
MAX_ANSWERS = {"bare": 1}

# Distinctive phrases: every instruction wording of a style matches its own signature and no other
# (tested), so the validator can tell which reply shapes a part declares.
SIGNATURES = {
    "bare": re.compile(r"\b(?:the (?:answer|grid) alone|only the (?:answer|grid)|just the (?:answer|grid))\b"),
    "labelled": re.compile(r"\b[Ll]abel (?:each answer|the answers?)\b"),
    "sentence": re.compile(r"\b(?:full|complete) sentences?\b"),
    "paragraph": re.compile(r"\b(?:single|one) paragraph\b"),
    "json": re.compile(r"\bJSON object\b"),
    "kv": re.compile(r"\bkey(?::| =) value\b"),
    "chain": re.compile(r"\ball the answers on a single line\b|\bsingle line that chains\b"),
    "bullets": re.compile(r"\bbullet"),
    "restate": re.compile(r"\b[Rr]estate each question\b"),
}
# Phrases that demand a bare reply: contradictory next to any other style (query texts are scanned).
# Only reply-directed uses count ("Answer with the number only.", "your answer and nothing else"), not
# content ("the odd object, and nothing else, is recolored") -- the same reading as pv2.wordings.
BARE_ONLY_PHRASES = re.compile(
    r"\bno other (?:text|words)\b|\bdo not add (?:any )?extra (?:lines|text)\b|\bextra text\b|"
    r"\b(?:answer|reply|response)\b[^.\n]*\bnothing else\b|\b(?:answer|reply|respond)\s+(?:with\s+)?only\b|"
    r"\b(?:answer|reply|respond) with (?:just |only )?(?:the |a )?"
    r"(?:number|digit|grid|colou?r|word|value|answer)s?(?: only)?\.", re.IGNORECASE)


@dataclass
class Shape:
    """One answer as a style sees it (no answer text)."""
    tag: Optional[str]              # printed question tag ("3", "b"), None for a lone unnumbered question
    kind: str
    grid: bool = False
    multiline: bool = False         # grid, or text with a line break
    json_type: str = "string"       # "number" | "string" | "grid"
    prefix: Optional[str] = None    # statement text before the answer (sentence / paragraph)
    suffix: Optional[str] = None    # statement text after it
    generic: bool = False           # the statement is the part's declared generic form
    question: Optional[str] = None  # the question text as printed (restate)


@dataclass
class StyleSpec:
    style: str
    shapes: list[Shape]
    variant: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"style": self.style, "variant": dict(self.variant), "shapes": [asdict(s) for s in self.shapes]}

    @staticmethod
    def from_dict(d: dict) -> "StyleSpec":
        return StyleSpec(d["style"], [Shape(**s) for s in d["shapes"]], dict(d["variant"]))


def answer_class(kind: str, text_or_none: Optional[str]) -> str:
    if kind == "grid":
        return "grid"
    if text_or_none is not None and "\n" in text_or_none:
        return "multiline"
    return kind


def style_allows(style: str, classes: list[str], *, structured: list[bool], nothing_else: list[bool],
                 restate_ok: bool) -> bool:
    """The matrix: can ``style`` carry these answers (one per question of the part)?"""
    n = len(classes)
    if n < MIN_ANSWERS.get(style, 1) or n > MAX_ANSWERS.get(style, 10 ** 9):
        return False
    if any(c not in KIND_MATRIX[style] for c in classes):
        return False
    if any(structured) and style not in STRUCTURED_OK:
        return False
    if any(nothing_else) and style not in NOTHING_ELSE_OK:
        return False
    if style == "restate" and not restate_ok:
        return False
    return True


# --- variants --------------------------------------------------------------------------------------

LABELS = {
    "number": ("Answer {t}:", "A{t}:", "Q{t}:", "{t}.", "{t})", "Question {t}:", "({t})"),
    "letter": ("({t})", "{t})", "{t}.", "Answer ({t}):", "Answer {t}:"),
    None: ("Answer:", "Result:"),
}
JSON_KEYS = {"number": ("q{t}", "Q{t}", "answer_{t}", "{t}"), "letter": ("{t}", "q_{t}", "answer_{t}"),
             None: ("answer", "result")}
KV_KEYS = {"number": ("q{t}", "Q{t}", "answer_{t}", "a{t}"), "letter": ("{t}", "answer_{t}", "q_{t}"),
           None: ("answer", "result")}
CHAIN_ITEMS = {"number": ("Q{t} -> ", "{t} -> ", "Q{t}: "), "letter": ("({t}) -> ", "{t} -> ")}
CHAIN_JOINERS = ("; ", " | ")
BULLET_MARKERS = ("- ", "* ")
BULLET_TAGS = {"number": ("", "Q{t}: ", "({t}) "), "letter": ("", "({t}) ")}
RESTATE = {"number": (("Q: ", "A:"), ("Question: ", "Answer:"), ("Q{t}: ", "A{t}:")),
           "letter": (("Q: ", "A:"), ("Question: ", "Answer:")),
           None: (("Q: ", "A:"), ("Question: ", "Answer:"))}


def tag_kind(shapes: list[Shape]) -> Optional[str]:
    tags = [s.tag for s in shapes]
    if any(t is None for t in tags):
        if len(tags) != 1:
            raise ValueError("untagged answers only for a lone question")
        return None
    return "number" if all(t.isdigit() for t in tags) else "letter"


def _ok(options, banned):
    """Templates that do not use a banned style-label word (an item text's own "Answer:" etc.)."""
    kept = [o for o in options if not contains_word(o if isinstance(o, str) else " ".join(o), banned)]
    if not kept:
        raise ValueError(f"every variant uses a banned word {sorted(banned)}")
    return kept


def choose_variant(rng, style: str, shapes: list[Shape], banned: frozenset = frozenset()) -> dict:
    tk = tag_kind(shapes)
    if style == "labelled":
        return {"label": pick(rng, _ok(LABELS[tk], banned)), "wording": int(rng.integers(2))}
    if style == "json":
        return {"key": pick(rng, _ok(JSON_KEYS[tk], banned)), "layout": pick(rng, ("inline", "lines")),
                "wording": int(rng.integers(2))}
    if style == "kv":
        return {"key": pick(rng, _ok(KV_KEYS[tk], banned)), "sep": pick(rng, (": ", " = ")),
                "wording": int(rng.integers(2))}
    if style == "chain":
        return {"item": pick(rng, CHAIN_ITEMS[tk]), "joiner": pick(rng, CHAIN_JOINERS),
                "wording": int(rng.integers(2))}
    if style == "bullets":
        return {"marker": pick(rng, BULLET_MARKERS), "tag": pick(rng, BULLET_TAGS[tk]),
                "wording": int(rng.integers(2))}
    if style == "restate":
        q, a = pick(rng, _ok(RESTATE[tk], banned))
        return {"q": q, "a": a}
    return {"wording": int(rng.integers(3))}


# --- pieces ----------------------------------------------------------------------------------------

def _fmt(template: str, tag: Optional[str]) -> str:
    return template.format(t=tag) if "{t}" in template else template


def _sentence_sep(spec: StyleSpec) -> str:
    return "\n\n" if any(s.multiline for s in spec.shapes) else "\n"


def pieces(spec: StyleSpec) -> list[str]:
    """Literal pieces L0..Ln of a non-JSON style (target = L0 + a1 + L1 + ... + an + Ln)."""
    st, sh, v = spec.style, spec.shapes, spec.variant
    if st == "bare":
        return ["", ""]
    befores = []                       # befores[i]: the text between answer i-1 (or the start) and answer i
    for i, s in enumerate(sh):
        if st == "labelled":
            before = ("\n" if i else "") + _fmt(v["label"], s.tag) + ("\n" if s.multiline else " ")
        elif st in ("sentence", "paragraph"):
            sep = _sentence_sep(spec) if st == "sentence" else " "
            before = (sh[i - 1].suffix + sep if i else "") + s.prefix + ("\n" if s.multiline else "")
        elif st == "kv":
            before = ("\n" if i else "") + _fmt(v["key"], s.tag) + v["sep"]
        elif st == "chain":
            before = (v["joiner"] if i else "") + _fmt(v["item"], s.tag)
        elif st == "bullets":
            before = ("\n" if i else "") + v["marker"] + _fmt(v["tag"], s.tag)
        elif st == "restate":
            before = (("\n\n" if i else "") + _fmt(v["q"], s.tag) + s.question + "\n" + _fmt(v["a"], s.tag)
                      + ("\n" if s.multiline else " "))
        else:
            raise ValueError(f"no pieces for style {st!r}")
        befores.append(before)
    final = sh[-1].suffix if st in ("sentence", "paragraph") else ""
    return befores + [final]


def _interleave(lits: list[str], texts: list[str]) -> str:
    out = [lits[0]]
    for t, lit in zip(texts, lits[1:]):
        out += [t, lit]
    return "".join(out)


def _parse_pieces(text: str, lits: list[str]) -> list[str]:
    n = len(lits) - 1
    if not text.startswith(lits[0]):
        raise ValueError("reply does not start as declared")
    pos, out = len(lits[0]), []
    for i in range(1, n + 1):
        lit = lits[i]
        if i == n:
            if not text.endswith(lit) or len(text) - len(lit) < pos:
                raise ValueError("reply does not end as declared")
            end = len(text) - len(lit)
        else:
            if not lit:
                raise ValueError("empty separator between answers")
            end = text.find(lit, pos)
            if end < 0:
                raise ValueError(f"separator {lit!r} not found")
        out.append(text[pos:end])
        pos = end + len(lit)
    if pos != len(text):
        raise ValueError("trailing text after the last answer")
    return out


# --- JSON --------------------------------------------------------------------------------------------

def _json_value(s: Shape, text: str, raw: bool) -> str:
    if raw:
        return text
    if s.json_type == "number":
        if not re.fullmatch(r"-?\d+", text):
            raise ValueError(f"JSON number expected, got {text!r}")
        return text
    if s.json_type == "grid":
        return text                     # already the one-line JSON rendering (json_flat)
    return json.dumps(text, ensure_ascii=False)


def _json_render(spec: StyleSpec, texts: list[str], raw: bool = False) -> str:
    keys = [_fmt(spec.variant["key"], s.tag) for s in spec.shapes]
    items = [f'"{k}": {_json_value(s, t, raw)}' for k, s, t in zip(keys, spec.shapes, texts)]
    if spec.variant["layout"] == "inline":
        return "{" + ", ".join(items) + "}"
    return "{\n" + ",\n".join("  " + it for it in items) + "\n}"


def _json_parse(spec: StyleSpec, text: str) -> list[str]:
    def pairs_hook(pairs):
        keys = [k for k, _ in pairs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate JSON key")
        return pairs
    pairs = json.loads(text, object_pairs_hook=pairs_hook)
    keys = [_fmt(spec.variant["key"], s.tag) for s in spec.shapes]
    if not isinstance(pairs, list) or [k for k, _ in pairs] != keys:
        raise ValueError("JSON keys differ from the declared ones")
    out = []
    for (k, v), s in zip(pairs, spec.shapes):
        if s.json_type == "number":
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{k}: a JSON number was declared")
            out.append(str(v))
        elif s.json_type == "grid":
            out.append(render_grid(v, "json_flat"))
        else:
            if not isinstance(v, str):
                raise ValueError(f"{k}: a JSON string was declared")
            out.append(v)
    return out


# --- public API ----------------------------------------------------------------------------------------

def render(spec: StyleSpec, texts: list[str]) -> str:
    if len(texts) != len(spec.shapes):
        raise ValueError("one text per answer")
    if spec.style == "json":
        return _json_render(spec, texts)
    return _interleave(pieces(spec), texts)


def parse(spec: StyleSpec, text: str) -> list[str]:
    """The answer strings, in question order (raises on any deviation from the declared shape)."""
    if spec.style == "json":
        return _json_parse(spec, text)
    return _parse_pieces(text, pieces(spec))


def round_trips(spec: StyleSpec, texts: list[str]) -> bool:
    try:
        target = render(spec, texts)
        return parse(spec, target) == texts and render(spec, parse(spec, target)) == target
    except (ValueError, KeyError, TypeError):
        return False


# The grid placeholder of skeletons.  A part whose input grids are framed by <grid> ... </grid> tags
# uses "[grid]" instead (``variant["grid_ph"]``), so a skeleton never looks like one of those frames.
GRID_PLACEHOLDER = "<grid>"
TAGS_GRID_PLACEHOLDER = "[grid]"


def grid_placeholder(spec: "StyleSpec") -> str:
    return spec.variant.get("grid_ph", GRID_PLACEHOLDER)


def _placeholder(s: Shape, grid_ph: str = GRID_PLACEHOLDER) -> str:
    if s.grid:
        return grid_ph
    if s.multiline:
        return "<answer lines>"
    return "<answer>"


def skeleton(spec: StyleSpec) -> str:
    """The exact reply shape with placeholders in place of the answers."""
    gp = grid_placeholder(spec)
    if spec.style == "json":
        vals = [gp if s.grid else ("<number>" if s.json_type == "number" else '"<answer>"')
                for s in spec.shapes]
        return _json_render(spec, vals, raw=True)
    if spec.style == "restate":
        shapes = [Shape(**{**asdict(s), "question": f"<{qref(s.tag)}>" if s.tag else "<question>"})
                  for s in spec.shapes]
        return _interleave(pieces(StyleSpec("restate", shapes, spec.variant)), [_placeholder(s, gp) for s in shapes])
    return _interleave(pieces(spec), [_placeholder(s, gp) for s in spec.shapes])


def qref(tag: str) -> str:
    return f"question {tag}" if tag.isdigit() else f"question ({tag})"


def sentence_pieces(form: Optional[str], refs: list[str], grid: bool, *, lead_in: bool = False):
    """(prefix, suffix) of a statement form filled with ``refs``, or None when the form cannot carry
    this answer: ⟪answer⟫ must occur once (a grid answer: at the end, and the sentence then ends in
    ":" with the grid on the lines after it); ``lead_in`` also takes a grid lead-in that already ends
    in ":" and has no ⟪answer⟫ (the library's grid forms, used by declared parts)."""
    if form is None:
        return None
    if grid and lead_in and "⟪answer⟫" not in form and form.rstrip().endswith(":"):
        return fill(form.rstrip(), refs), ""
    if form.count("⟪answer⟫") != 1:
        return None
    pre, suf = form.split("⟪answer⟫")
    pre, suf = fill(pre, refs), fill(suf, refs)
    if not grid:
        return pre, suf
    if suf.strip() not in ("", "."):
        return None
    pre = pre.rstrip()
    return (pre if pre.endswith(":") else pre + ":"), ""


# Undeclared sentence / paragraph parts show the SHAPE with one fixed example; its placeholders are
# never a claim about a grid of the row (no reference phrase, no "<grid>" next to tag frames).  The
# examples name no measure: a concrete one ("The number of rows of ... is ...") read as the wording
# for a question of that measure in the same part, whose own sentence may differ (23 of 1200 rows,
# blind recheck 2026-09-30).
SENTENCE_EXAMPLES = (
    'For example: "The <thing asked for> of <grid name> is <answer>."',
    'A sentence might read: "In <grid name>, the <thing asked for> is <answer>."',
    'For instance: "<grid name> has <answer> as its <thing asked for>."',
)


def declared_shapes(spec: StyleSpec) -> list[bool]:
    """Which answers have their exact sentence written in the prompt: every one in a declared part
    (``variant["declared"]``), else the generic ones."""
    every = bool(spec.variant.get("declared"))
    return [every or s.generic for s in spec.shapes]


def declarations(spec: StyleSpec) -> str:
    """The sentences a sentence / paragraph instruction states verbatim (see ``declared_shapes``)."""
    decls = []
    for s, stated in zip(spec.shapes, declared_shapes(spec)):
        if not stated:
            continue
        if s.grid:
            body = f'"{s.prefix}" followed by the grid on the lines after it'
        else:
            body = f'"{s.prefix}<answer>{s.suffix}"'
        decls.append(f"for {qref(s.tag)}, write {body}" if s.tag else f"write {body}")
    if not decls:
        return ""
    text = "; ".join(decls)
    if spec.variant.get("declared") and len(decls) > 1:
        return " Word each sentence exactly as follows: " + text + "."
    return " " + text[0].upper() + text[1:] + "."


def _prose_extras(spec: StyleSpec) -> str:
    """The example (when some sentence's exact wording is not stated) and the stated sentences."""
    example = ""
    if not all(declared_shapes(spec)):
        example = " " + SENTENCE_EXAMPLES[spec.variant.get("wording", 0) % len(SENTENCE_EXAMPLES)]
    return example + declarations(spec)


def instruction(rng, spec: StyleSpec) -> str:
    """The reply-shape declaration for a part (one of several wordings; exact skeletons where the
    shape is structured)."""
    st, sh, v = spec.style, spec.shapes, spec.variant
    n = len(sh)
    tags = [s.tag for s in sh]
    many = n > 1
    grid = any(s.grid for s in sh)
    w = v.get("wording", 0)
    if st == "bare":
        noun = "grid" if grid else "answer"
        if v.get("in_row"):
            # a multi-part row puts a heading line before the part's answer: no "alone" / "no other text"
            return (f"Give just the {noun}.", f"Write just the {noun}.", f"Reply with just the {noun}.")[w % 3]
        return (f"Reply with the {noun} alone.", f"Give only the {noun}, with no other text.",
                f"Write just the {noun}.")[w % 3]
    if st == "labelled":
        if w == 0 and tags[0] is not None:
            first = _fmt(v["label"], tags[0])
            text = (f'Label each answer with its question {"number" if tags[0].isdigit() else "letter"} '
                    f'exactly like "{first}", one answer per line, in question order.')
            if any(s.multiline for s in sh):
                text += " An answer that is a grid or spans several lines starts on the line after its label."
            return text
        return ("Label the answers exactly like this:\n" if many else "Label the answer exactly like this:\n") \
            + skeleton(spec)
    if st == "sentence":
        if many:
            text = ("Answer each question with one full sentence, in question order, each sentence on its own line.",
                    "Reply in complete sentences: one sentence per question, in order, each on its own line.",
                    "Write one full sentence per question, in question order, one sentence per line.")[w % 3]
        else:
            text = ("Answer with one full sentence.", "Reply with a complete sentence.",
                    "Write your answer as one full sentence.")[w % 3]
        if grid:
            text += " When the answer is a grid, end its sentence with a colon and write the grid on the lines after it."
            if many:
                text += " Leave a blank line between answers."
        return text + _prose_extras(spec)
    if st == "paragraph":
        text = ("Reply with a single paragraph, one sentence per question, in question order.",
                "Write one paragraph that answers every question in order, one sentence per question.",
                "Answer in a single paragraph: one sentence for each question, in question order, "
                "separated by single spaces.")[w % 3]
        return text + _prose_extras(spec)
    if st == "json":
        head = ("Reply with a JSON object with exactly these keys, in this order:\n",
                "Give the answers as one JSON object, filled in from this template:\n")[w % 2]
        rules = []
        if any(s.json_type == "number" for s in sh):
            rules.append("each <number> is a JSON number")
        if any(s.json_type == "string" for s in sh):
            rules.append('each "<answer>" is a JSON string')
        if grid:
            rules.append(f"each {grid_placeholder(spec)} is the grid as a JSON list of rows on one line")
        rule = ", ".join(rules)
        return head + skeleton(spec) + "\n" + rule[0].upper() + rule[1:] + "."
    if st == "kv":
        form = "key" + v["sep"] + "value"
        head = (f"Reply with one {form} line per question, exactly in this form:\n",
                f"Write the answers as {form} lines like this:\n")[w % 2]
        return head + skeleton(spec)
    if st == "chain":
        head = ("Give all the answers on a single line, in this form:\n",
                "Reply with a single line that chains the answers like this:\n")[w % 2]
        return head + skeleton(spec)
    if st == "bullets":
        head = ("Reply with a bulleted list, one bullet per question, in question order:\n",
                "List the answers as markdown bullets in question order, like this:\n")[w % 2]
        return head + skeleton(spec)
    if st == "restate":
        text = "Restate each question before answering it, exactly in this form:\n" + skeleton(spec)
        if many:
            text += "\nLeave a blank line between questions."
        return text
    raise ValueError(f"unknown style {st!r}")


def check_grid_text(text: str, encoding: str, grid) -> bool:
    try:
        return parse_grid(text, encoding) == grid
    except (ValueError, TypeError, json.JSONDecodeError):
        return False


__all__ = ["STYLES", "KIND_MATRIX", "STRUCTURED_OK", "NOTHING_ELSE_OK", "SIGNATURES", "BARE_ONLY_PHRASES",
           "Shape", "StyleSpec", "answer_class", "style_allows", "choose_variant", "render", "parse",
           "round_trips", "skeleton", "instruction", "qref", "join_tags", "sentence_pieces", "declared_shapes",
           "declarations", "SENTENCE_EXAMPLES"]
