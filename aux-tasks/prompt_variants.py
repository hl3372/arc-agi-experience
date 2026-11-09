"""Extra prompt wordings per task, kept in ``prompt_variants.json`` next to this file.

A variant is an alternative wording of a task's instruction with the same objective and the
same answer format as the canonical prompt in ``prompts.py``.  Nothing here changes what the
generators produce; the canonical prompt is always available and, when variants are enabled
for a dataset, one wording is drawn uniformly from ``canonical + approved variants``.

Store shape (the same shape the parent workspace's ``prompt_variant_store`` reads, so the file
can move there unchanged)::

    {"version": 1,
     "tasks": {"4": [{"id": "4_v001", "template": "...", "note": "...", "status": "approved"}]}}

Templates use the mini-template language of ``prompts.render_prompt_variant_template``
(``{ci:key}``, ``{m:key}``, ``{shape_query_phrase}``, ...) plus a few compound tokens defined
here so a variant never has to restate a policy sentence by hand:

    {shape_object_rule}     the tasks 4/5 object rule (background ignored, >= 2 cells, isolated cells)
    {pair_object_rule}      the tasks 7/8 object rule (connectivity-aware)
    {isolated_cell_rule}    the isolated-cell clause alone
    {shape_size_filter_note}  " Count only those whose bounding box is AxB ..." or ""
    {index_note}            "Rows and columns are 0-indexed from the top-left corner."
    {unchanged_note}        "Leave every other cell unchanged."
    {placements_phrase}     task 3: "a color X cell at position (r, c)" / "color X cells at positions
                            (r, c), (r, c) and (r, c)" / "cells at the following positions: (r, c) in
                            color X, ..." depending on the example's mode
    {image_a} / {image_b}   task 9: "top image" / "bottom image" or "left image" / "right image"
    {rotation_phrase}       task 10: "90 degrees clockwise" or "180 degrees" (no direction at 180)
    {axis_phrase} / {flip_phrase}  task 11: "horizontal axis" / "up-down" or "vertical axis" / "left-right"
    {piece_rule} {anchor_rule} {fit_rule} {receive_rule} {leftover_rule} {piece_tail}
                            task 13: the piece definition (>= 2 cells, isolated cells), which pieces are
                            fixed (by colour or by touching the border), the flush-seating / jagged-seam
                            rule, how many loose pieces a fixed piece may receive (varies per example),
                            the leftover sentence (a piece that seats nowhere stays where it is, a cavity
                            nothing fits stays open) and the no-overlap / vacated-cells sentence
    {line_rule} {crossing_shown_rule} {distractor_note} {pairwise_count_rule}
                            tasks 15 / 15a / 16: the line definition, the shared-cell colour rule, the
                            distractor sentence and 15a's count-every-pair rule
    {largest_object_rule}   tasks 17/18/19: the object rule (connectivity-aware, >= 2 cells, isolated cells)
    {bg_ignore} {exact_object_rule} {cells_phrase}
                            tasks 20/21: "Ignore the background, whether it is a solid color or a pattern.",
                            the connectivity-aware component rule, and "1 cell" / "k cells"

Generic tokens (2026-09-16, for tasks 22-102) reach the named constants and helpers of the task's
own prompt module (the module that defines its canonical builder; ``prompts`` is the fallback), so
a variant can carry any policy sentence verbatim without a hand-written compound token:

    {const:_NAME}           a module-level string constant, e.g. {const:_LATTICE_BLOCKS}
    {fn:_name}              a helper called on the example: parameters named ``m`` / ``meta`` get
                            the metadata, ``conn`` gets ``_connectivity_phrase(meta)``, any other
                            parameter is read from the metadata key of the same name
    {fn:_name:a|b}          the same with explicit positional arguments after the metadata /
                            connectivity ones: a bare word is a metadata key (raw value), ``m:key``
                            the same value as text, a quoted word ('vertically') a literal
    {dict:_NAME:key}        ``_NAME[meta[key]]`` for a string-valued lookup table
    {dict:_NAME:key:i}      item ``i`` of a tuple-valued lookup table, e.g. {dict:_TRANSFORMS:transform:1}

Names must start with an underscore and exist in the module; anything else raises at render time.

Command line::

    python prompt_variants.py list [--task 4]
    python prompt_variants.py add --task 4 --template "..." [--note "..."] [--status draft]
    python prompt_variants.py validate [--tasks 4,5] [--seeds 0,1,2,3] [--include-drafts] [--coverage 400]

``add`` validates the template on real examples before saving.  ``validate`` renders every
stored template on real examples under a symbol palette and applies the review policies:
no unresolved placeholder, no raw colour digit, the background never named or defined, and
every policy sentence of the canonical prompt (isolated-cell rule, 0-indexed note, "leave every
other cell unchanged", the answer-format sentence) present in the variant too.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Iterable, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prompts  # noqa: E402
from prompts import ISOLATED_CELL_RULE, get_prompt, render_prompt_variant_template  # noqa: E402

DEFAULT_PATH = os.path.join(ROOT, "prompt_variants.json")
LETTER_PALETTE = list("ABCDEFGHIJ")
STATUSES = ("approved", "draft", "rejected")

INDEX_NOTE = "Rows and columns are 0-indexed from the top-left corner."
UNCHANGED_NOTE = "Leave every other cell unchanged."

# Sentences that, when the canonical prompt carries them, every variant must carry too.
_REQUIRED_IF_CANONICAL = (
    ISOLATED_CELL_RULE,
    # 2026-09-16, tasks 22-102 (proposed by the writers, one entry per load-bearing guarantee)
    "exactly one region", "without crossing a color", "strictly inside", "occurs more often than any other",  # 57 (26/35/52)
    "no tie needs to be resolved",                                                                       # 76
    "follow the grid diagonals", "top-corner to bottom-corner span", "combined filled area",             # 71/72
    "outside the diamonds", "counting both corner cells", "one-cell-thick", "infer the full span",
    "regardless of their endpoints", "parallel diagonal bands", "supporting rows, columns or diagonal bands",  # 73/74
    "at the grid edges",
    "which is drawn", "keeps its color", "marked start cell",                                             # 23/24
    "the cell where the trace stops is not a bounce", "change of direction at a wall cell", "background detail to ignore",
    "do not overlap", "start cell is not included",                                                      # 60-63
    "Copies may be cut off", "every block is filled with a single color", "small pattern", "width-by-height",  # 65/55/54/66
    "horizontal and vertical", "stays unchanged", "squares count", "uniform wall thickness",               # 26/35/49-52
    "one connected single-color object", "Do not count wall cells.", "outer height times",
    "at least 3 rows by 3 columns", "copied exactly",                                                     # 96
    "a cell that maps to itself", "is its own mirrored partner, so it is kept", "nowhere else",           # 99/101/38
    "a single cell that differs makes the answer no", "does not map the grid onto itself",               # 102
    "Use this exact order", "Do not add extra lines or extra text.",
    "including any isolated single cell", "Output the whole grid.",                                       # 5a
    "empty runs between partners do not cross or overlap.", "empty cells connect only up, down, left and right.",  # 37
    "the same horizontal, vertical or 45-degree diagonal", "whatever its color",                          # 37, 22/36/37
    # (27/28's translation-only clause, 34/40's texture clause and the colour-separation clause are
    #  stated in the writers' own words on every wording; they are checked by reading, not by substring)
    "in one simultaneous pass", "non-overlapping rectangles unchanged", "outside the stamped footprints unchanged",  # 47/67/78
    "the original input",                                                                                # 83
    "at least two same-color cells", "A single cell also counts as an object", "separate objects even when they touch",  # 68a/68b
    "still counts as a block", "not part of any block", "strictly between", "already neighbors",          # 87-89
    "every one of its cells is that color",                                                              # 89
    "does not span the full height or the full width",                                                   # 64
    "mirror image", "whose first cell comes earliest",                                                    # 34 / 81
    "[(top_row,left_column,bottom_row,right_column),...]", "inclusive",                                  # 94
    "smallest rectangle that contains all of", "outside the grid",                                        # 95
    "plain background, without any of the input's background texture", "the same dimensions as the input",  # 98
    # (56/58/59/84/85/93/97: the writer's proposed fragments are paraphrased on its own wordings; checked by reading)
    "0-indexed",
    UNCHANGED_NOTE,
    "at least two cells",
    "mirror images do not count",
    "0, 90, 180 or 270",   # the rotation set on 7/8: dropping the 0 would change the answer
    "at least two cells past",              # 15 / 15a / 16: rules out T-junctions and end-touches
    "exactly one object has",               # 17-19: the no-tie guarantee
    "count every pair",                     # 15a: pairwise counting at multi-line cells
    "always separate, even where they touch",  # 17-21: colour boundaries split objects
    "do not rotate or reflect",             # 13: translation only
    "stays exactly where it is",            # 13: a loose piece that seats nowhere is left alone
    "clockwise",                            # 12: the rotation direction
)
# Wordings the review policies forbid anywhere (the background is never named or defined).
_FORBIDDEN = (
    # "the background is X" / "the background colour is X" / "background colour X": naming or
    # defining the background.  A bare reference such as 49's "the plain background color and
    # clip ..." is allowed (the canonical prompts use it), so the pattern needs a defining verb
    # or a colour symbol after the noun.
    re.compile(r"background (is|colou?r is|colou?r,? which is|colou?r,? that is|colou?r [A-J0-9]\b|colou?r: )", re.I),
    re.compile(r"most (frequent|common) colou?r", re.I),
    re.compile(r"\bcolou?r \d\b(?!-)"),     # a raw digit under the symbol palette ("color 8-connected" is not one)
)
_UNRESOLVED = re.compile(r"\{[^{}]+\}")
_LB, _RB = "\u0000LB\u0000", "\u0000RB\u0000"
_ANSWER_SENTENCE = re.compile(r"(Answer as [^.]*\.|Answer with [^.]*\.|Format the answer [^.]*\.|"
                              r"Answer '[^']+'(?: or '[^']+')?\.|Return that offset as [^.]*\.|Format: [^\n]*)")


# ----------------------------------------------------------------------------- store
def load(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {"version": 1, "tasks": {}}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("version", 1)
    payload.setdefault("tasks", {})
    return payload


def save(payload: dict, path: str = DEFAULT_PATH) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def variants_for(task_id: str, payload: Optional[dict] = None, statuses: Iterable[str] = ("approved",)) -> list[dict]:
    payload = payload if payload is not None else load()
    wanted = set(statuses)
    return [v for v in payload["tasks"].get(str(task_id), []) if v.get("status", "approved") in wanted]


def templates_for(task_id: str, payload: Optional[dict] = None) -> list[str]:
    return [v["template"] for v in variants_for(task_id, payload)]


def next_id(task_id: str, payload: dict) -> str:
    n = len(payload["tasks"].get(str(task_id), [])) + 1
    return f"{task_id}_v{n:03d}"


def select_variant(task_id: str, rng, payload: Optional[dict] = None) -> Optional[str]:
    """One wording drawn uniformly over the canonical prompt (None) and the approved variants."""
    options: list[Optional[str]] = [None] + templates_for(task_id, payload)
    return options[int(rng.integers(0, len(options)))]


# ----------------------------------------------------------------------------- rendering
def _placements_phrase(meta: dict) -> str:
    """Task 3's placement list in the wording of the canonical prompt (mode-aware)."""
    rows, cols = meta.get("placement_rows"), meta.get("placement_cols")
    if not rows or len(rows) == 1:
        r = prompts._m(meta, "target_row") if "target_row" in meta else str(rows[0])
        c = prompts._m(meta, "target_col") if "target_col" in meta else str(cols[0])
        col = prompts._ci(meta, "target_color") if "target_color" in meta else prompts._ci_list(meta, "placement_colors")[0]
        return f"a color {col} cell at position ({r}, {c})"
    labels = prompts._ci_list(meta, "placement_colors")
    positions = [f"({r}, {c})" for r, c in zip(rows, cols)]
    if meta.get("colour_mode") == "single":
        return f"color {labels[0]} cells at positions {prompts._join_and(positions)}"
    return "cells at the following positions: " + ", ".join(f"{pos} in color {lab}" for pos, lab in zip(positions, labels))


def _rotation_phrase(meta: dict) -> str:
    angle = prompts._m(meta, "angle")
    direction = prompts._m(meta, "direction")
    return f"{angle} degrees {direction}" if direction else f"{angle} degrees"


def _compound_tokens(meta: dict) -> dict[str, str]:
    vertical = prompts._m(meta, "layout") == "vertical"
    axis = prompts._m(meta, "axis") if "axis" in meta else ""
    conn = str(prompts._connectivity_phrase(meta)) if "connectivity" in meta else ""
    return {
        # task 3 only: 87 also carries target_row / target_col (a block index) without any colour key
        "placements_phrase": _placements_phrase(meta) if ("placement_rows" in meta or (
            "target_row" in meta and ("target_color" in meta or "placement_colors" in meta))) else "",
        "image_a": "top image" if vertical else "left image",
        "image_b": "bottom image" if vertical else "right image",
        "rotation_phrase": _rotation_phrase(meta) if "angle" in meta else "",
        "shape_object_rule": prompts._SHAPE_OBJECT_RULE,
        "pair_object_rule": prompts._pair_object_rule(meta) if "connectivity" in meta else "",
        "isolated_cell_rule": ISOLATED_CELL_RULE,
        "shape_size_filter_note": prompts._shape_size_filter_note(meta),
        "index_note": INDEX_NOTE,
        "unchanged_note": UNCHANGED_NOTE,
        # task 11
        "axis_phrase": f"{axis} axis" if axis in ("horizontal", "vertical") else "",
        "flip_phrase": {"horizontal": "up-down", "vertical": "left-right"}.get(axis, ""),
        # task 13
        "piece_rule": prompts._P13_PIECE_RULE,
        "anchor_rule": prompts._p13_anchor_rule(meta) if "anchor_rule" in meta else "",
        "fit_rule": prompts._P13_FIT,
        "receive_rule": prompts._p13_receive_rule(meta) if "variant" in meta else "",
        "leftover_rule": prompts._P13_LEFTOVER,
        "piece_tail": prompts._P13_TAIL,
        # tasks 15 / 15a / 16
        "line_rule": prompts._LINE_RULE,
        "crossing_shown_rule": prompts._CROSSING_SHOWN_RULE,
        "distractor_note": prompts._DISTRACTOR_NOTE,
        "pairwise_count_rule": prompts._PAIRWISE_COUNT_RULE,
        # tasks 17-19 / 20-21 (connectivity comes from the example's metadata)
        "largest_object_rule": prompts._largest_object_rule(meta, conn) if conn else "",
        "bg_ignore": prompts._BG_IGNORE,
        "exact_object_rule": prompts._exact_object_rule(conn) if conn else "",
        "cells_phrase": prompts._cells(meta["n"]) if "n" in meta else "",
    }


_DSL_HELPERS = {"template_desc", "shape_query_phrase", "binary_layout_phrase", "connectivity_phrase",
                "symbol_color_order"}


def render(task_id: str, template: str, meta: Optional[dict] = None, color_palette=None) -> str:
    """Render a variant template for one example's metadata (compound tokens first).

    The palette is attached to the metadata before the compound tokens render, so colours
    inside them go through the palette too.  A bare ``{name}`` that is neither a token nor a
    metadata key raises instead of silently rendering as an empty string.
    """
    meta = dict(meta or {})
    if color_palette is not None:
        meta[prompts.COLOR_PALETTE_META_KEY] = color_palette
    compound = _compound_tokens(meta)
    for name in _UNRESOLVED.findall(template):
        token = name.strip("{}").strip()
        if ":" in token or token in compound or token in _DSL_HELPERS:
            continue
        if token not in meta:
            raise ValueError(f"unknown token or metadata key: {{{token}}}")
    text = template.replace("{{", _LB).replace("}}", _RB)
    for token, value in compound.items():
        text = text.replace("{" + token + "}", value)
    text = _GENERIC_TOKEN.sub(lambda mt: _render_generic_token(str(task_id), mt.group(1), mt.group(2), meta), text)
    # a constant may itself be a str.format template ({edge} in _P91_PACK_RULE): its bare fields
    # must be metadata keys too, or the DSL would render them as empty text without a word
    for name in _UNRESOLVED.findall(text):
        token = name.strip("{}").strip()
        if ":" in token or token in compound or token in _DSL_HELPERS:
            continue
        if token not in meta:
            raise ValueError(f"unknown token or metadata key: {{{token}}} (from an expanded constant?)")
    text = text.replace(_LB, "{{").replace(_RB, "}}")
    return render_prompt_variant_template(task_id, text, meta=meta, color_palette=color_palette)


# ----------------------------------------------------------------------------- generic tokens
_GENERIC_TOKEN = re.compile(r"\{(const|fn|dict):([^{}]+)\}")
_NAME_RE = re.compile(r"^_[A-Za-z][A-Za-z0-9_]*$")


def _prompt_module(task_id: str):
    """The module that defines the task's canonical prompt builder (``prompts`` as a fallback)."""
    import sys

    fn = prompts.PROMPT_TEMPLATES.get(str(task_id))
    mod = sys.modules.get(getattr(fn, "__module__", "") or "")
    return mod if mod is not None else prompts


def _lookup(task_id: str, name: str):
    if not _NAME_RE.match(name):
        raise ValueError(f"generic token names must be private module names, got {name!r}")
    for mod in (_prompt_module(task_id), prompts):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise ValueError(f"unknown name {name!r} for task {task_id} (looked in {_prompt_module(task_id).__name__} and prompts)")


def _parse_arg(arg: str, meta: dict):
    arg = arg.strip()
    if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in ("'", '"'):
        return arg[1:-1]
    if arg.startswith("m:"):
        return prompts._m(meta, arg[2:].strip())
    if arg not in meta:
        raise ValueError(f"argument {arg!r} is not a metadata key")
    return meta[arg]


def _render_generic_token(task_id: str, kind: str, body: str, meta: dict) -> str:
    import inspect

    parts = body.split(":")
    name = parts[0].strip()
    obj = _lookup(task_id, name)
    if kind == "const":
        if len(parts) != 1 or not isinstance(obj, str):
            raise ValueError(f"{{const:{body}}}: {name} is not a string constant")
        return obj
    if kind == "dict":
        if not isinstance(obj, dict) or len(parts) not in (2, 3):
            raise ValueError(f"{{dict:{body}}}: expected {{dict:_NAME:key[:index]}} on a dict")
        key = _parse_arg(parts[1], meta)
        if key not in obj:
            raise ValueError(f"{{dict:{body}}}: key {key!r} not in {name}")
        value = obj[key]
        if len(parts) == 3:
            value = value[int(parts[2])]
        if not isinstance(value, str):
            raise ValueError(f"{{dict:{body}}}: value is not a string (give an index?)")
        return value
    # fn
    if not callable(obj):
        raise ValueError(f"{{fn:{body}}}: {name} is not callable")
    rest = body.split(":", 1)[1] if ":" in body else ""
    explicit = [_parse_arg(a, meta) for a in rest.split("|")] if rest.strip() else []
    args: list = []
    for param in inspect.signature(obj).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            break
        if param.name in ("m", "meta"):
            args.append(meta)
        elif param.name == "conn":
            args.append(str(prompts._connectivity_phrase(meta)))
        elif explicit:
            args.append(explicit.pop(0))
        elif param.name in meta:
            args.append(meta[param.name])
        elif param.default is not param.empty:
            break
        else:
            raise ValueError(f"{{fn:{body}}}: no value for parameter {param.name!r}")
    if explicit:
        raise ValueError(f"{{fn:{body}}}: too many arguments")
    value = obj(*args)
    if not isinstance(value, str):
        raise ValueError(f"{{fn:{body}}}: helper returned {type(value).__name__}, not text")
    return value


# ----------------------------------------------------------------------------- validation
def check_template(task_id: str, template: str, seeds: Iterable[int] = (0, 1, 2, 3, 4, 5)) -> list[str]:
    """Problems with one template on real examples (empty list = passes)."""
    from registry import generate

    problems: list[str] = []
    for seed in seeds:
        ex = generate(str(task_id), int(seed))
        meta = ex.get("metadata", {})
        canonical = get_prompt(str(task_id), meta, color_palette=LETTER_PALETTE)
        try:
            text = render(str(task_id), template, meta, color_palette=LETTER_PALETTE)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"seed {seed}: render error: {type(exc).__name__}: {exc}")
            continue
        if not text.strip():
            problems.append(f"seed {seed}: empty prompt")
            continue
        if _UNRESOLVED.search(text):
            problems.append(f"seed {seed}: unresolved placeholder {_UNRESOLVED.search(text).group(0)!r}")
        for rx in _FORBIDDEN:
            hit = rx.search(text)
            if hit:
                problems.append(f"seed {seed}: forbidden wording {hit.group(0)!r}")
        for phrase in _REQUIRED_IF_CANONICAL:
            if phrase.lower() in canonical.lower() and phrase.lower() not in text.lower():
                problems.append(f"seed {seed}: canonical prompt carries {phrase[:40]!r}... and the variant does not")
        ans = _ANSWER_SENTENCE.search(canonical)
        if ans and ans.group(0) not in text:
            problems.append(f"seed {seed}: answer-format sentence {ans.group(0)!r} missing")
        if text.strip() == canonical.strip():
            problems.append(f"seed {seed}: identical to the canonical prompt")
    # de-duplicate while keeping order
    seen: set[str] = set()
    return [p for p in problems if not (p in seen or seen.add(p))]


_BUCKET_KEYS = ("connectivity", "object_connectivity", "angle", "axis", "layout", "variant", "anchor_rule",
                "max_sides", "colour_mode", "direction",
                # 2026-09-16 (writers' findings): branches that differ only by a number or that flip
                # the example without changing the wording
                "query_selection",     # 29/30: the named colour may be absent or the ground
                "ends_at_wall",        # 24: traces that stop away from a wall
                "border_thickness",    # 49: 1/2/3
                "gap",                 # 96: strip gap 1/2/3
                "tile_h", "tile_w",    # 32/33: periodic tile sizes
                "transform",           # 99/101
                "outline_thickness",   # 82: 1/2/3
                "orientation_rule", "query_mode", "contact_mode", "target_kind", "filter_kind", "criterion")  # 58/59/56/85/97


def coverage_seeds(task_id: str, sweep: int = 400, max_buckets: int = 200) -> list[int]:
    """One seed per distinct canonical wording of the task over ``sweep`` seeds.

    A task's canonical prompt varies with the metadata (task 4 has about a hundred wordings
    across categories, size filters and textures; task 3 has three placement modes; task 9
    two layouts; task 10 drops the direction word at 180).  Validating a template on one seed
    per wording proves it renders correctly for every form the canonical prompt takes.
    """
    from registry import generate

    seen: dict[str, int] = {}
    for seed in range(sweep):
        meta = generate(str(task_id), seed).get("metadata", {})
        key = get_prompt(str(task_id), meta, color_palette=LETTER_PALETTE)
        # strip the example-specific numbers / colour letters so only the wording buckets
        key = re.sub(r"\b[A-J]\b", "@", key)
        key = re.sub(r"\d+", "#", key)
        # digits are stripped above, so branches that differ only by a number (task 12's
        # connectivity and angle, 20/21's n = 1) are kept apart through the metadata itself
        key += "|" + "|".join(f"{k}={meta.get(k)}" for k in _BUCKET_KEYS if k in meta)
        key += f"|n1={meta.get('n') == 1}" if "n" in meta else ""
        key += f"|d1={meta.get('distance') == 1}" if "distance" in meta else ""   # 74: "1 rows" grammar
        if key not in seen:
            seen[key] = seed
            if len(seen) >= max_buckets:
                break
    return sorted(seen.values())


def validate(task_ids: Optional[Iterable[str]] = None, seeds: Iterable[int] = (0, 1, 2, 3, 4, 5),
             include_drafts: bool = False, payload: Optional[dict] = None,
             coverage: int = 0) -> dict[str, dict[str, list[str]]]:
    """{task: {variant id: [problems]}} for every stored variant (approved, plus drafts on request).

    With ``coverage`` > 0 the seeds are replaced by one seed per distinct canonical wording found
    in the first ``coverage`` seeds of the task (see ``coverage_seeds``).
    """
    payload = payload if payload is not None else load()
    statuses = ("approved", "draft") if include_drafts else ("approved",)
    tasks = [str(t) for t in task_ids] if task_ids else sorted(payload["tasks"], key=lambda t: (len(t), t))
    report: dict[str, dict[str, list[str]]] = {}
    for tid in tasks:
        rows = variants_for(tid, payload, statuses)
        if not rows:
            continue
        task_seeds = validation_seeds(tid, seeds, coverage)
        for v in rows:
            report.setdefault(tid, {})[v["id"]] = check_template(tid, v["template"], task_seeds)
    return report


def validation_seeds(task_id: str, seeds: Iterable[int], coverage: int) -> list[int]:
    return coverage_seeds(task_id, coverage) if coverage else list(seeds)


# ----------------------------------------------------------------------------- cli
def _cmd_list(args) -> int:
    payload = load(args.path)
    tasks = [args.task] if args.task else sorted(payload["tasks"], key=lambda t: (len(t), t))
    if not tasks:
        print("no variants stored (canonical prompts only)")
        return 0
    for tid in tasks:
        rows = payload["tasks"].get(tid, [])
        print(f"task {tid}: {len(rows)} variant(s)")
        for v in rows:
            print(f"  [{v.get('status', 'approved')}] {v['id']}: {v['template'][:110]}" + (f"  # {v['note']}" if v.get("note") else ""))
    return 0


def _cmd_add(args) -> int:
    payload = load(args.path)
    problems = check_template(args.task, args.template, seeds=_seeds(args.seeds))
    if problems and not args.force:
        print("not saved; fix these first (or pass --force to store as a draft):")
        for p in problems:
            print("  -", p)
        return 1
    rec = {"id": next_id(args.task, payload), "template": args.template, "note": args.note,
           "status": "draft" if (problems or args.status == "draft") else args.status}
    payload["tasks"].setdefault(args.task, []).append(rec)
    save(payload, args.path)
    print(f"saved {rec['id']} as {rec['status']}")
    return 0


def _cmd_validate(args) -> int:
    payload = load(args.path)
    report = validate(args.tasks.split(",") if args.tasks else None, seeds=_seeds(args.seeds),
                      include_drafts=args.include_drafts, payload=payload, coverage=args.coverage)
    bad = 0
    for tid, per in report.items():
        n_seeds = len(validation_seeds(tid, _seeds(args.seeds), args.coverage))
        print(f"task {tid}: {n_seeds} seed(s)" + (" = one per canonical wording" if args.coverage else ""))
        for vid, problems in per.items():
            status = "ok" if not problems else "FAIL"
            print(f"task {tid} {vid}: {status}")
            for p in problems:
                print("   -", p)
            bad += bool(problems)
    n = sum(len(per) for per in report.values())
    print(f"{n} variant(s) checked, {bad} failing")
    return 1 if bad else 0


def _seeds(value: str) -> list[int]:
    return [int(s) for s in value.split(",") if s.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--path", default=DEFAULT_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--task", default=None); p.set_defaults(fn=_cmd_list)
    p = sub.add_parser("add"); p.add_argument("--task", required=True); p.add_argument("--template", required=True)
    p.add_argument("--note", default=""); p.add_argument("--status", choices=STATUSES, default="approved")
    p.add_argument("--seeds", default="0,1,2,3,4,5"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=_cmd_add)
    p = sub.add_parser("validate"); p.add_argument("--tasks", default=""); p.add_argument("--seeds", default="0,1,2,3,4,5")
    p.add_argument("--include-drafts", action="store_true")
    p.add_argument("--coverage", type=int, default=0, help="sweep this many seeds and validate on one seed per distinct canonical wording")
    p.set_defaults(fn=_cmd_validate)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
