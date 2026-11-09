"""assemble_row: render any list of Parts (families may differ) into one (prompt, target) row.

Per part the assembler chooses a reference scheme, labels and reference phrases (qualified with
"this part's" where a row-wide reading could differ), a layout, a delimiter, the answer style (from
the style x kind matrix, with a render/parse round trip before acceptance), the grid-answer encoding,
and where the scoped instructions go; it fills every placeholder, numbers questions continuously
across the row, renders, records a ``checks`` block in ``meta`` and runs ``pv2.validate.validate_row``
(unless the config turns it off).  Deterministic for a given numpy Generator state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pv2 import encodings as enc
from pv2.assemble import layouts as L
from pv2.assemble import refs as R
from pv2.assemble import flags as F
from pv2.assemble import styles as S
from pv2.assemble._util import chance, letter, pick, pick_weighted
from pv2.model import (FLAG_COORDS, FLAG_DIMS, FLAG_EDIT, FLAG_IO_NOUNS, FLAG_LONG, FLAG_NOTHING_ELSE,
                       FLAG_PLACEMENT, FLAG_STRUCTURED, Part, fill)


class AssemblyError(ValueError):
    """The parts cannot be packed as asked (a producer/lead contract problem, not bad luck)."""


@dataclass
class Row:
    prompt: str
    target: str
    meta: dict


def _w(**kw):
    return field(default_factory=lambda: dict(kw))


@dataclass
class AssembleConfig:
    scheme_multi: dict = _w(label=0.5, ordinal=0.5)
    scheme_single: dict = _w(none=0.55, label=0.45)
    label_nouns: dict = _w(**{"Grid": 0.40, "Image": 0.12, "Panel": 0.12, "Map": 0.10, "Figure": 0.12, "#": 0.14})
    letter_labels_p: float = 0.4
    delimiters: dict = _w(label_line=0.35, fenced=0.2, tags=0.15, banner=0.15, blank=0.15)
    role_delimiters: dict = _w(label_line=0.8, fenced=0.08, tags=0.06, banner=0.06)
    generic_header_p: float = 0.6        # "Input:" / "Grid:" over a lone unlabelled grid (fenced / tags / banner)
    dims_p: float = 0.35                 # "(5 rows x 7 cols)" in label headers, when allowed
    role_dims_p: float = 0.9             # stage-2 headers carry the size
    # stage-2 role labels are printed verbatim, and the live solver arms (render_think / render_output)
    # number pairs from 0; a 1-based "Training pair 1" was misread as the second pair by the training
    # side's claim checker (training session, 2026-09-29), so the verbatim labels are 0-based only
    role_zero_based_p: float = 1.0       # drawn once per row: every ARC part of a row uses the same base
    single_layouts: dict = _w(q_first=0.35, q_after=0.40, split=0.25)
    multi_layouts: dict = _w(grids_first=0.30, questions_first=0.20, split=0.15, interleaved=0.20, reminder=0.15)
    # paragraph can only carry one-line answers, so its weight is diluted by eligibility; 0.30 makes it a
    # real share of the parts that can take it (the user asked for paragraph practice; 0.08 gave 2%)
    styles: dict = _w(bare=0.20, labelled=0.16, sentence=0.12, paragraph=0.30, json=0.10, kv=0.07, chain=0.07,
                      bullets=0.07, restate=0.07)
    # share of sentence / paragraph parts that write every question's exact, answer-neutral sentence into
    # the prompt (``Query.declared``; the generic form where a question has none).  Drawn per part from the
    # rng, never from the answers; the other parts keep their content-aware sentences and show one fixed
    # example of the shape.  Training side, 2026-09-30: half declared, half undeclared.
    declare_statements_p: float = 0.5
    grid_answer_encodings: dict = _w(compact=0.76, spaced=0.05, json_rows=0.04, comma=0.04, brackets=0.03,
                                     pipes=0.03, json_flat=0.03, names=0.02)
    json_grid_factor: float = 0.4        # JSON carries grids as JSON arrays: damp it where grids are answers
    # one encoding for every grid ANSWER (e.g. "compact"); JSON is then not used where a part has a grid
    # answer (it would write the grid as an array).  None = the weighted mix above.  Answer STYLES
    # (sentences, paragraphs, labels, ...) are unaffected.
    single_grid_encoding: Optional[str] = None
    compact_sentence_p: float = 0.3      # also state the compact grid-answer format when nothing requires it
    head_instructions_p: float = 0.4     # scoped instructions at the part's start (else at its end)
    letter_questions_p: float = 0.15     # question tags (a), (b) ... instead of numbers
    intro_p: float = 0.3                 # an intro line in layouts other than split
    ordinal_end_p: float = 0.35          # "the last grid" / "the second-to-last grid" where they apply
    validate: bool = True
    force: dict = field(default_factory=dict)   # test hook: scheme / layout / delimiter / style / grid_enc / tags


_NUMERAL = re.compile(r"-?\d+")

# Encodings whose text does not explain itself (the stage-2 rendering carries its own runs title,
# the sparse listing and colour names read plainly): a note names the encoding before the grid.
NOTED_ENCODINGS = frozenset({"runs"})
ENCODING_NOTES = ("The next grid is written {how}.", "(The following grid is given {how}.)",
                  "Note: the next grid is written {how}.")


# --- row-level checks ------------------------------------------------------------------------------------

def _role_bearing(part: Part) -> bool:
    return any(s.role != "grid" for s in part.slots)


def check_parts(parts: list[Part]) -> None:
    """Contract rules that no choice of layout or style can repair (raise AssemblyError)."""
    if not parts:
        raise AssemblyError("a row needs at least one part")
    for part in parts:
        try:
            part.validate_shape()
        except ValueError as e:
            raise AssemblyError(str(e)) from e
        lone = part.single_answer or any(FLAG_NOTHING_ELSE in q.flags for q in part.queries)
        if lone and (len(parts) != 1 or len(part.queries) != 1):
            raise AssemblyError(f"{part.family}: a single-answer / nothing-else part must be the row's only "
                                "part and have one question")
        if _role_bearing(part) and any(FLAG_IO_NOUNS in q.flags for q in part.queries):
            raise AssemblyError(f"{part.family}: an io_nouns wording in a role-bearing part")
        if any(FLAG_PLACEMENT in q.flags for q in part.queries) and len(part.slots) != 1:
            raise AssemblyError(f"{part.family}: a placement wording needs a single-grid part")
        for q in part.queries:
            if q.answer.kind == "grid":
                enc.check_grid(q.answer.value)
        for s in part.slots:
            enc.check_grid(s.grid)
            if s.encoding not in enc.ENCODINGS:
                raise AssemblyError(f"unknown slot encoding {s.encoding!r}")


def _label_words(parts: list[Part]) -> set[str]:
    return {w.lower() for p in parts for q in p.queries for w in q.label_words()}


def _row_texts(parts: list[Part]) -> list[str]:
    out = []
    for p in parts:
        out += list(p.context)
        for q in p.queries:
            out.append(q.text)
            if q.statement:
                out.append(q.statement)
            if q.declared:
                out.append(q.declared)
            if q.answer.text:
                out.append(q.answer.text)
    return out


def _allowed_nouns(parts: list[Part]) -> list[str]:
    words = _label_words(parts)
    texts = _row_texts(parts)
    out = []
    for noun in R.LABEL_NOUNS:
        low = noun.lower()
        if any(re.search(rf"(?:^|\W){re.escape(low)}s?(?:$|\W)", w) for w in words):
            continue
        if noun != "Grid" and R.noun_in_texts(noun, texts):
            continue                      # "Grid" is exempt: item texts say "the grid" of their own slot
        out.append(noun)
    return out


# --- per-part planning ------------------------------------------------------------------------------------

@dataclass
class _Plan:
    part: Part
    scheme: str = ""
    labels: list = field(default_factory=list)
    phrases: list = field(default_factory=list)
    label_noun: Optional[str] = None
    base: Optional[int] = None
    qualified: bool = False
    layout: str = ""
    order: list = field(default_factory=list)
    delimiter: str = ""
    headers: list = field(default_factory=list)
    dims: bool = False
    tags: list = field(default_factory=list)


def _referenced_slots(part: Part) -> set[int]:
    used = set()
    for q in part.queries:
        used.update(q.reads)
    if part.context:
        used.update(range(len(part.slots)))
    return used


def _choose_scheme(rng, cfg: AssembleConfig, part: Part, label_ok: bool) -> str:
    if _role_bearing(part):
        return "role"
    n = len(part.slots)
    rules = F.part_rules(part)
    if n == 1:
        options = {"label"} if part.labels_required else {"none", "label"}
        table = cfg.scheme_single
    else:
        options = {"label"}
        reach = all(R.ordinal_options(i, n) for i in _referenced_slots(part))
        if not part.labels_required and reach and rules.ordinals_ok:
            options.add("ordinal")
        table = cfg.scheme_multi
    if not label_ok and options != {"label"}:
        options.discard("label")
    if not options or (options == {"label"} and not label_ok):
        raise AssemblyError(f"{part.family}: no reference scheme fits the part's wording flags")
    forced = cfg.force.get("scheme")
    if forced in options:
        return forced
    return pick_weighted(rng, table, options)


def _plan_refs(rng, cfg, plan: _Plan, book: R.LabelBook, nouns: list[str], grid_numbers_ok: bool,
               row_state: dict) -> None:
    part = plan.part
    n = len(part.slots)
    rules = F.part_rules(part)
    label_ok = any((rules.numbers_ok and not (noun == "Grid" and not grid_numbers_ok))
                   or (noun != "#" and book.free_letters(noun, rules.banned_letters) >= n) for noun in nouns)
    plan.scheme = _choose_scheme(rng, cfg, part, label_ok)
    if plan.scheme == "role":
        if "base" not in row_state:           # one stage-2 numbering base per row, drawn at its first ARC part
            row_state["base"] = 0 if chance(rng, cfg.role_zero_based_p) else 1
        plan.base = row_state["base"]
        try:
            plan.labels = R.role_labels(part.slots, plan.base, book, rules.banned_letters)
        except (ValueError, IndexError) as e:
            raise AssemblyError(f"{part.family}: {e}") from e
        plan.phrases = list(plan.labels)
    elif plan.scheme == "label":
        try:
            plan.label_noun, style, plan.labels = book.assign(
                rng, n, nouns, cfg.label_nouns, cfg.letter_labels_p, cfg.force.get("label_style"),
                numbers_ok=rules.numbers_ok, banned_letters=rules.banned_letters, grid_numbers_ok=grid_numbers_ok)
        except ValueError as e:
            raise AssemblyError(f"{part.family}: {e}") from e
        plan.phrases = list(plan.labels)
    elif plan.scheme == "ordinal":
        plan.labels = [None] * n
        plan.phrases = R.ordinal_phrases(rng, n, cfg.ordinal_end_p)
    else:
        plan.labels = [None]
        plan.phrases = [pick(rng, R.SINGLE_PHRASES)]


def _qualify(plans: list[_Plan], noun: Optional[str]) -> None:
    row_labels = [p.labels for p in plans]
    for pi, plan in enumerate(plans):
        bad = any(ph is not None and R.resolve(ph, pi, row_labels) != {(pi, s)} for s, ph in enumerate(plan.phrases))
        if not bad:
            continue
        if noun is None:
            raise AssemblyError(f"ambiguous reference in a single-part row: {plan.phrases}")
        plan.phrases = [R.qualify(ph, noun) if ph is not None else None for ph in plan.phrases]
        plan.qualified = True
        for s, ph in enumerate(plan.phrases):
            if ph is not None and R.resolve(ph, pi, row_labels) != {(pi, s)}:
                raise AssemblyError(f"reference {ph!r} stays ambiguous")


def _plan_layout(rng, cfg, plan: _Plan) -> None:
    part = plan.part
    n = len(part.slots)
    forced = cfg.force.get("layout")
    if any(FLAG_PLACEMENT in q.flags for q in part.queries):
        plan.layout = "q_first"
    elif n == 1:
        plan.layout = forced if forced in L.SINGLE_LAYOUTS else pick_weighted(rng, cfg.single_layouts)
    else:
        plan.layout = forced if forced in L.MULTI_LAYOUTS else pick_weighted(rng, cfg.multi_layouts)
    order = list(range(len(part.queries)))
    if plan.layout == "interleaved":
        order.sort(key=lambda i: (max(part.queries[i].reads), i))
    plan.order = order


def _plan_blocks(rng, cfg, plan: _Plan) -> None:
    part = plan.part
    allowed = L.delimiters_for("label" if plan.scheme == "role" else plan.scheme)
    table = cfg.role_delimiters if plan.scheme == "role" else cfg.delimiters
    forced = cfg.force.get("delimiter")
    plan.delimiter = forced if forced in allowed else pick_weighted(rng, table, set(allowed))
    dims_ok = (plan.scheme in ("label", "role") and all(s.dims_header_ok for s in part.slots)
               and not any(FLAG_DIMS in q.flags for q in part.queries))
    plan.dims = dims_ok and chance(rng, cfg.role_dims_p if plan.scheme == "role" else cfg.dims_p)
    if "dims" in cfg.force:
        plan.dims = dims_ok and bool(cfg.force["dims"])
    headers = []
    for s, label in zip(part.slots, plan.labels):
        if label is not None:
            headers.append(R.header_text(label, s.grid if plan.dims else None))
        elif plan.scheme == "none" and (plan.delimiter == "label_line" or (
                plan.delimiter in ("fenced", "tags", "banner") and chance(rng, cfg.generic_header_p))):
            headers.append(pick(rng, L.GENERIC_HEADERS))
        else:
            headers.append(None)
    plan.headers = headers


# --- filling -------------------------------------------------------------------------------------------------

_GENERIC_FORMS = (
    ("ref", "For ⟪g0⟫, the answer is ⟪answer⟫."),
    ("ref", "The answer about ⟪g0⟫ is ⟪answer⟫."),
    ("tag", "The answer to {qref} is ⟪answer⟫."),
    ("tag", "{Qref}: the answer is ⟪answer⟫."),
)


def _placeholders_used(text: str, n: int) -> list[int]:
    return [i for i in range(n) if any(f"⟪{p}⟫" in text for p in (f"g{i}", f"G{i}", f"g{i}s", f"G{i}s"))]


def _statement(q, refs, grid: bool, generic: str, tag: Optional[str],
               declared: bool = False) -> tuple[str, str, bool]:
    """(prefix, suffix, generic) of a question's sentence: its content-aware statement, or in a declared
    part its answer-neutral ``declared`` form (a grid lead-in ending in ":" included); the part's generic
    form where the question has no usable one."""
    pieces = S.sentence_pieces(q.declared if declared else q.statement, refs, grid, lead_in=declared)
    if pieces is not None:
        return pieces[0], pieces[1], False
    form = generic
    if len(refs) > 1:                     # a generic statement names every grid the question reads
        form = form.replace("⟪g0⟫", L.join_tags([f"⟪g{i}⟫" for i in range(len(refs))]))
    if tag is not None:
        qr = S.qref(tag)
        form = form.replace("{qref}", qr).replace("{Qref}", qr[:1].upper() + qr[1:])
    pre, suf = form.split("⟪answer⟫")
    pre, suf = fill(pre, refs), fill(suf, refs)
    if grid:
        pre = pre.rstrip()
        return (pre if pre.endswith(":") else pre + ":"), "", True
    return pre, suf, True


def _json_type(q, text: str) -> str:
    if q.answer.kind == "grid":
        return "grid"
    if q.answer.kind in ("number", "color") and _NUMERAL.fullmatch(text) and not isinstance(q.answer.value, bool) \
            and isinstance(q.answer.value, int) and int(text) == q.answer.value:
        return "number"
    return "string"


def _grid_encoding(rng, cfg, style: str, part: Part) -> str:
    if style == "json":
        return "json_flat"
    if style == "bare":
        return "compact"                  # stage-2's parse_grids reads every bare grid answer
    if cfg.single_grid_encoding:
        return cfg.single_grid_encoding
    forced = cfg.force.get("grid_enc")
    allowed = set(cfg.grid_answer_encodings)
    if any(q.source == "item" for q in part.queries) or any(s.encoding != "names" for s in part.slots):
        allowed.discard("names")          # item texts (and questions about digit grids) say "color 5"
    if forced in allowed:
        return forced
    return pick_weighted(rng, cfg.grid_answer_encodings, allowed)


def _choose_style(rng, cfg, plan: _Plan, questions: list[dict]) -> tuple[S.StyleSpec, Optional[str], list[str]]:
    part = plan.part
    qs = [part.queries[i] for i in plan.order]
    classes = [S.answer_class(q.answer.kind, q.answer.text) for q in qs]
    structured = [FLAG_STRUCTURED in q.flags or c == "multiline" for q, c in zip(qs, classes)]
    nothing = [FLAG_NOTHING_ELSE in q.flags for q in qs]
    restate_ok = all(q.source == "lib" and FLAG_LONG not in q.flags and "\n" not in d["text"]
                     for q, d in zip(qs, questions))
    candidates = [s for s in S.STYLES if cfg.styles.get(s, 0) > 0
                  and S.style_allows(s, classes, structured=structured, nothing_else=nothing, restate_ok=restate_ok)]
    forced = cfg.force.get("style")
    if forced in candidates:
        candidates = [forced]
    tags = [plan.tags[k] for k in range(len(qs))]
    banned = F.part_rules(part).banned_words
    usable_generic = [f for kind, f in _GENERIC_FORMS if (kind == "ref" or tags[0] is not None)
                      and not F.uses_word_as_label(f.replace("{Qref}", "Question 1"), banned)]
    weights = dict(cfg.styles)
    if any(q.answer.kind == "grid" for q in qs):
        weights["json"] = weights.get("json", 0) * cfg.json_grid_factor
        if cfg.single_grid_encoding:
            weights["json"] = 0.0
            candidates = [c for c in candidates if c != "json"]
    declare = None                        # sentence / paragraph: state every exact sentence (drawn once per part)
    while candidates:
        style = pick_weighted(rng, weights, set(candidates))
        grid_enc = _grid_encoding(rng, cfg, style, part) if any(q.answer.kind == "grid" for q in qs) else None
        prose = style in ("sentence", "paragraph")
        if prose and declare is None:
            declare = cfg.declare_statements_p > 0 and chance(rng, cfg.declare_statements_p)
        texts, shapes = [], []
        generic = pick(rng, usable_generic)
        for k, (q, d) in enumerate(zip(qs, questions)):
            if q.answer.kind == "grid":
                text = enc.render(q.answer.value, grid_enc)
            else:
                text = fill(q.answer.text, d["read_phrases"])
            shape = S.Shape(tag=tags[k], kind=q.answer.kind, grid=q.answer.kind == "grid",
                            multiline=q.answer.kind == "grid" or "\n" in text, json_type=_json_type(q, text))
            if prose:
                shape.prefix, shape.suffix, shape.generic = _statement(q, d["read_phrases"], shape.grid, generic,
                                                                       tags[k], declare)
            if style == "restate":
                shape.question = d["text"]
            texts.append(text)
            shapes.append(shape)
        if style in ("sentence", "paragraph") and tags[0] is not None:
            # two questions sharing one generic sentence ("The answer about Grid A is ...") align only by
            # their order: switch the part to a form that names the question (54 of 1082 parts, parity review)
            said = [s.prefix.rstrip(" :") for s in shapes if s.generic]     # a grid's lead-in ends in ":"
            tag_forms = [f for f in usable_generic if "{qref}" in f or "{Qref}" in f]
            if len(said) != len(set(said)) and tag_forms:
                generic = pick(rng, tag_forms)
                for k, (q, d) in enumerate(zip(qs, questions)):
                    if shapes[k].generic:
                        shapes[k].prefix, shapes[k].suffix, shapes[k].generic = _statement(
                            q, d["read_phrases"], shapes[k].grid, generic, tags[k], declare)
        try:
            spec = S.StyleSpec(style, shapes, S.choose_variant(rng, style, shapes, banned))
        except ValueError:
            candidates.remove(style)
            continue
        if prose and declare:
            spec.variant["declared"] = True
        if S.round_trips(spec, texts):
            return spec, grid_enc, texts
        candidates.remove(style)
    raise AssemblyError(f"{part.family}: no answer style fits the part's answers")


# --- the row ---------------------------------------------------------------------------------------------------

def assemble_row(rng, parts: list[Part], *, config: Optional[AssembleConfig] = None) -> Row:
    cfg = config or AssembleConfig()
    parts = list(parts)
    check_parts(parts)
    multi = len(parts) > 1
    n_q = sum(len(p.queries) for p in parts)
    numbered = n_q > 1

    # row-level choices
    tag_kind = None
    if numbered:
        tag_kind = "letter" if n_q <= 26 and chance(rng, cfg.letter_questions_p) else "number"
        if cfg.force.get("tags") in ("number", "letter") and (cfg.force["tags"] == "number" or n_q <= 26):
            tag_kind = cfg.force["tags"]
    banned_row = F.row_banned_words(parts)
    tag_fmt = pick(rng, L.question_tag_formats(tag_kind, banned_row)) if numbered else None
    noun = pick(rng, R.QUALIFIER_NOUNS) if multi else None
    boundary_fmt = pick(rng, L.BOUNDARY_FORMATS) if multi else None
    heading_fmt = pick(rng, L.heading_formats(banned_row)) if multi else None
    lead_form = pick(rng, L.lead_forms(banned_row))
    scope_form = pick(rng, L.SCOPE_FORMS) if multi else None

    # references
    nouns = _allowed_nouns(parts)
    book = R.LabelBook()
    plans = [_Plan(part=p) for p in parts]
    grid_numbers_ok = F.row_grid_numbers_ok(parts)
    row_state: dict = {}
    for plan in plans:
        _plan_refs(rng, cfg, plan, book, nouns, grid_numbers_ok, row_state)
    _qualify(plans, noun)

    # layouts, blocks, continuous question tags
    k = 0
    for plan in plans:
        _plan_layout(rng, cfg, plan)
        _plan_blocks(rng, cfg, plan)
        plan.tags = []
        for _ in plan.order:
            k += 1
            plan.tags.append(None if not numbered else (str(k) if tag_kind == "number" else letter(k - 1).lower()))

    # render
    out: list[str] = []
    pos = [0]

    def emit(text: str) -> tuple[int, int]:
        if out:
            out.append("\n\n")
            pos[0] += 2
        start = pos[0]
        out.append(text)
        pos[0] += len(text)
        return start, pos[0]

    row_header = None
    if multi:
        row_header = L.row_header(rng, noun, len(parts), n_q, heading_fmt)
        emit(row_header)

    checks_parts, meta_parts, targets = [], [], []
    for pi, plan in enumerate(plans):
        part = plan.part
        part_start = pos[0] + (2 if out else 0)
        boundary = L.boundary_line(boundary_fmt, noun, pi + 1, len(parts)) if multi else None
        heading = L.heading_line(heading_fmt, noun, pi + 1) if multi else None

        # questions (display order = plan.order)
        questions = []
        for k_, qi in enumerate(plan.order):
            q = part.queries[qi]
            refs = [plan.phrases[s] for s in q.reads]
            if any(r is None for r in refs):
                raise AssemblyError(f"{part.family}: query {q.qid} reads a slot with no reference phrase")
            body = fill(q.text, refs)
            lead = None
            if q.lead_clause and len(part.slots) > 1:
                lead = fill(L.lead_clause(lead_form, len(q.reads)), refs)
            text = f"{lead} {body}" if lead else body
            named = list(range(len(q.reads))) if lead else _placeholders_used(q.text, len(q.reads))
            questions.append({"query": qi, "qid": q.qid, "tag": plan.tags[k_], "text": text,
                              "read_phrases": refs, "refs": [[refs[i], q.reads[i]] for i in named]})

        spec, grid_enc, answers = _choose_style(rng, cfg, plan, questions)
        if plan.delimiter == "tags" and any(s.grid for s in spec.shapes):
            spec.variant["grid_ph"] = S.TAGS_GRID_PLACEHOLDER   # never "<grid>" next to <grid> ... </grid> frames
        if multi and spec.style == "bare":
            spec.variant["in_row"] = True                     # a heading line precedes this part's answer
        for d, a, qi in zip(questions, answers, plan.order):
            q = part.queries[qi]
            d["answer"] = a
            d["answer_refs"] = [] if q.answer.kind == "grid" else \
                [[d["read_phrases"][i], q.reads[i]] for i in _placeholders_used(q.answer.text, len(q.reads))]

        # scoped instructions
        qs = [part.queries[i] for i in plan.order]
        coords = [q for q in qs if FLAG_COORDS in q.flags]
        needs_zero = bool(coords)
        sentences = []
        if needs_zero and not all(L.ZERO_INDEX_RE.search(q.text) for q in coords):
            sentences.append(pick(rng, L.ZERO_INDEX_SENTENCES))
        needs_as_shown = any(FLAG_EDIT in q.flags and any(set(q.reads) & set(o.reads) for o in qs if o is not q)
                             for q in qs)
        if needs_as_shown and not any(L.AS_SHOWN_RE.search(c) for c in part.context):
            sentences.append(pick(rng, L.AS_SHOWN_SENTENCES))      # (library parts carry it as context)
        if grid_enc is not None and grid_enc != "names" and any(
                part.slots[i].encoding == "names" for q in qs if q.answer.kind == "grid" for i in q.reads):
            sentences.append(L.NAMES_TO_DIGITS)
        if grid_enc is not None and spec.style != "json":
            shown_other = any(s.encoding != "compact" for s in part.slots)
            if grid_enc != "compact" or shown_other or chance(rng, cfg.compact_sentence_p):
                sentences.append(L.grid_answer_sentence(rng, grid_enc))
        style_text = S.instruction(rng, spec)
        instructions = " ".join(sentences + [style_text])
        if scope_form is not None:
            instructions = L.scoped(scope_form, noun, instructions)
        instr_head = chance(rng, cfg.head_instructions_p)

        # paragraphs of the part
        refs_all = [ph if ph is not None else "" for ph in plan.phrases]
        # one line per context line: value tables ("Training pair 1 input -> 23") and rule menus stay lists
        context = "\n".join(fill(c, refs_all) for c in part.context) if part.context else None
        lines = [L.question_line(tag_fmt, d["tag"], d["text"]) for d in questions]
        paras: list[tuple[str, object]] = []
        n_slots = len(part.slots)
        if plan.layout == "split":
            paras.append(("text", L.intro_line(rng, n_slots, len(qs))))
        elif chance(rng, cfg.intro_p):
            paras.append(("text", L.intro_line(rng, n_slots, None)))
        ctx = [("text", context)] if context else []
        qblock = [("text", L.question_block(lines))]
        blocks = [("block", s) for s in range(n_slots)]
        if plan.layout == "q_first":
            paras += ctx + qblock + blocks
        elif plan.layout in ("q_after", "split", "grids_first"):
            paras += blocks + ctx + qblock
        elif plan.layout == "questions_first":
            paras += ctx + qblock + blocks
        elif plan.layout == "reminder":
            paras += ctx + qblock + blocks + [("text", L.reminder_line(rng, plan.tags))]
        elif plan.layout == "interleaved":
            paras += ctx
            for s in range(n_slots):
                paras.append(("block", s))
                mine = [line for line, qi in zip(lines, plan.order) if max(part.queries[qi].reads) == s]
                if mine:
                    paras.append(("text", L.question_block(mine)))
        else:
            raise AssertionError(plan.layout)
        if instr_head:
            paras.insert(0, ("text", instructions))
        else:
            paras.append(("text", instructions))
        if boundary:
            paras.insert(0, ("text", boundary))

        block_checks = []
        for kind, val in paras:
            if kind == "text":
                emit(val)
                continue
            s = val
            slot = part.slots[s]
            if slot.encoding in NOTED_ENCODINGS:
                # an encoding a reader cannot recognise from the text alone is named right before it
                emit(ENCODING_NOTES[int(rng.integers(len(ENCODING_NOTES)))].format(
                    how=enc.ENCODINGS[slot.encoding].describe[0]))
            body = enc.render(slot.grid, slot.encoding)
            pre, suf = L.block_frame(plan.headers[s], plan.delimiter)
            start, _ = emit(pre + body + suf)
            block_checks.append({"slot": s, "header": plan.headers[s], "label": plan.labels[s],
                                 "delimiter": plan.delimiter, "encoding": slot.encoding,
                                 "start": start + len(pre), "end": start + len(pre) + len(body)})
        part_end = pos[0]

        target = S.render(spec, answers)
        targets.append((heading, target))
        checks_parts.append({
            "family": part.family, "boundary": boundary, "heading": heading, "span": [part_start, part_end],
            "scheme": plan.scheme, "labels": plan.labels, "phrases": plan.phrases, "qualified": plan.qualified,
            "layout": plan.layout, "delimiter": plan.delimiter, "dims": plan.dims, "blocks": block_checks,
            "questions": [{k2: v for k2, v in d.items() if k2 != "read_phrases"} for d in questions],
            "style": spec.to_dict(), "grid_answer_encoding": grid_enc, "instructions": instructions,
            "needs_zero_index": needs_zero, "needs_as_shown": needs_as_shown,
        })
        meta_parts.append({
            "family": part.family, "scheme": plan.scheme, "label_noun": plan.label_noun, "base": plan.base,
            "qualified": plan.qualified, "layout": plan.layout, "delimiter": plan.delimiter, "dims": plan.dims,
            "style": spec.style, "variant": dict(spec.variant), "encodings": [s.encoding for s in part.slots],
            "grid_answer_encoding": grid_enc, "questions": len(qs), "instructions_at": "head" if instr_head else "tail",
            "query_order": [part.queries[i].qid for i in plan.order], "prov": part.prov,
        })

    prompt = "".join(out)
    if multi:
        target = "\n\n".join(f"{h}\n{t}" for h, t in targets)
    else:
        target = targets[0][1]
    meta = {
        "n_parts": len(parts), "n_questions": n_q, "families": [p.family for p in parts],
        "question_tags": tag_kind, "tag_format": tag_fmt, "part_noun": noun, "parts": meta_parts,
        "checks": {"multi": multi, "row_header": row_header, "tag_format": tag_fmt, "noun": noun,
                   "parts": checks_parts},
    }
    row = Row(prompt, target, meta)
    if cfg.validate:
        from pv2.validate import validate_row
        validate_row(row, parts)
    return row
