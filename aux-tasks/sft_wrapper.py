"""SFT wrapper: serializes task examples to text and bundles them into one prompt.

Supports same-task bundles (N seeds of one task via ``format_bundle``) and
mixed-task bundles (N different tasks via ``format_mixed_bundle``).
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from registry import generate
from prompts import get_prompt
from symbol_remap import (
    ITEM_SYMBOL_REMAP_PREAMBLE,
    SYMBOL_REMAP_PREAMBLE,
    normalize_color_palette,
    render_color_label,
    render_string_output,
    render_text_input,
)


# --- Instruction rendering ----------------------------------------------------
#
# Prompt variants are templates stored in the generator package's
# ``prompt_variants.json``. They may use *compound tokens* such as
# ``{shape_object_rule}`` or ``{placements_phrase}`` that only
# ``prompt_variants.render`` knows how to expand. The base renderer,
# ``prompts.render_prompt_variant_template``, silently renders any token it does
# not know as an empty string, so routing a variant through ``get_prompt`` alone
# drops whole policy sentences from the instruction ("Copy this grid and paint .").
# Every variant must therefore go through ``render_instruction`` below.
try:  # the generator package ships prompt_variants.py next to prompts.py
    from prompt_variants import render as _render_variant_template
except ImportError:  # pragma: no cover - legacy trees without the module
    _render_variant_template = None

# A bare ``{name}`` token: no ``:``/``|`` DSL syntax and not a ``{{ }}`` escape.
_BARE_TOKEN_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


def render_instruction(
    task_id: str,
    meta: Optional[dict],
    color_palette: Optional[Sequence[str]] = None,
    prompt_variant: Optional[str] = None,
) -> str:
    """Render the canonical instruction or a prompt-variant template for one example.

    ``prompt_variant=None`` is exactly ``get_prompt``. With a template, the generator
    package's ``prompt_variants.render`` is used (compound tokens expanded first; an
    unknown token raises). If that module is not importable, the base DSL renderer is
    used but any bare token that is neither a metadata key nor resolvable by the base
    DSL raises instead of rendering as empty text.
    """
    if prompt_variant is None:
        return get_prompt(task_id, meta, color_palette=color_palette)
    if _render_variant_template is not None:
        return _render_variant_template(
            task_id, prompt_variant, meta=meta, color_palette=color_palette,
        )
    meta_keys = set((meta or {}).keys())
    for name in sorted(set(_BARE_TOKEN_RE.findall(prompt_variant))):
        if name in meta_keys:
            continue
        probe = get_prompt(
            task_id, meta, color_palette=color_palette, prompt_variant="{" + name + "}",
        )
        if not probe.strip():
            raise ValueError(
                f"prompt variant for task {task_id} uses token {{{name}}}, which the base "
                f"renderer would silently drop, and no prompt_variants module is importable "
                f"to expand it. Ship prompt_variants.py with the generator package or fix "
                f"the template."
            )
    return get_prompt(task_id, meta, color_palette=color_palette, prompt_variant=prompt_variant)


# --- Grid rendering -----------------------------------------------------------

GRID_FORMAT_SPACED = "spaced"
GRID_FORMAT_COMPACT = "compact"
VALID_GRID_FORMATS = (GRID_FORMAT_SPACED, GRID_FORMAT_COMPACT)


def render_grid(
    grid: list[list[int]],
    fmt: str = GRID_FORMAT_SPACED,
    color_palette: Optional[Sequence[str]] = None,
) -> str:
    """Render a grid. spaced='1 2 3\\n4 5 6', compact='123\\n456'."""
    palette = normalize_color_palette(color_palette)
    rows = []
    for row in grid:
        cells = [render_color_label(int(c), palette) for c in row]
        rows.append("".join(cells) if fmt == GRID_FORMAT_COMPACT else " ".join(cells))
    return "\n".join(rows)


# --- Input rendering ----------------------------------------------------------

# Tasks with special (non-grid) input formats
_TEXT_INPUT_TASKS = {"14", "86"}
_DICT_INPUT_TASKS = {
    "41": ("before", "after"),
    "78": ("stamp", "canvas"),
}


def _looks_like_grid(value) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and isinstance(value[0], list)
        and (not value[0] or not isinstance(value[0][0], list))
    )


def _render_named_payload(
    value,
    fmt: str = GRID_FORMAT_SPACED,
    color_palette: Optional[Sequence[str]] = None,
) -> str:
    """Render structured dict/list inputs with stable labels."""
    if _looks_like_grid(value):
        return render_grid(value, fmt=fmt, color_palette=color_palette)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(
                f"{key}:\n{_render_named_payload(item, fmt=fmt, color_palette=color_palette)}"
            )
        return "\n\n".join(parts)
    if isinstance(value, list):
        parts = []
        for idx, item in enumerate(value, start=1):
            parts.append(
                f"item {idx}:\n{_render_named_payload(item, fmt=fmt, color_palette=color_palette)}"
            )
        return "\n\n".join(parts)
    return str(value)


def render_input(
    task_id: str,
    inp,
    fmt: str = GRID_FORMAT_SPACED,
    color_palette: Optional[Sequence[str]] = None,
) -> str:
    """Render an example's input to text, dispatching on type."""
    # Dispatch on the payload type as well as the id: task 86 was a text-input task
    # (lattice from text) until 2026-09-29, when the id was reused for a grid -> grid
    # crop, so an id-only check would print the new task's grid as a Python list.
    if task_id in _TEXT_INPUT_TASKS and isinstance(inp, str):
        return render_text_input(task_id, inp, color_palette=color_palette)

    if task_id in _DICT_INPUT_TASKS and isinstance(inp, dict):
        keys = _DICT_INPUT_TASKS[task_id]
        parts = []
        for key in keys:
            val = inp[key]
            if _looks_like_grid(val):
                parts.append(f"{key}:\n{render_grid(val, fmt=fmt, color_palette=color_palette)}")
            else:
                parts.append(f"{key}:\n{val}")
        return "\n\n".join(parts)

    if isinstance(inp, dict):
        return _render_named_payload(inp, fmt=fmt, color_palette=color_palette)

    # Default: plain grid
    if _looks_like_grid(inp):
        return render_grid(inp, fmt=fmt, color_palette=color_palette)

    # Fallback for unexpected formats
    return str(inp)


# --- Output rendering ---------------------------------------------------------

def render_output(
    output,
    fmt: str = GRID_FORMAT_SPACED,
    task_id: Optional[str] = None,
    color_palette: Optional[Sequence[str]] = None,
) -> str:
    """Render an example's output to text."""
    if isinstance(output, list) and output and isinstance(output[0], list):
        return render_grid(output, fmt=fmt, color_palette=color_palette)
    return render_string_output(task_id, output, color_palette=color_palette)


def _render_task_block(
    idx: int,
    task_id: str,
    seed: int,
    fmt: str,
    palette: Optional[Sequence[str]],
    prompt_variant: Optional[str],
) -> tuple[str, str]:
    """Render one numbered (task_block, answer_block) pair for a multi-item bundle.

    Shared by same-task (``format_bundle``) and mixed-task
    (``format_mixed_bundle``) formatting so their per-item layout stays
    byte-identical. ``palette`` is assumed already normalized.
    """
    ex = generate(task_id, seed)
    instruction = render_instruction(
        task_id,
        ex.get("metadata"),
        color_palette=palette,
        prompt_variant=prompt_variant,
    )
    inp_text = render_input(task_id, ex["input"], fmt=fmt, color_palette=palette)
    out_text = render_output(ex["output"], fmt=fmt, task_id=task_id, color_palette=palette)
    return (
        f"Task {idx}\nInstruction: {instruction}\nInput:\n{inp_text}",
        f"Answer {idx}:\n{out_text}",
    )


# --- Single example formatting ------------------------------------------------

def format_single(
    task_id: str,
    seed: int,
    fmt: str = GRID_FORMAT_SPACED,
    color_palette: Optional[Sequence[str]] = None,
    prompt_variant: Optional[str] = None,
) -> tuple[str, str]:
    """Format one example as (user_content, assistant_content)."""
    palette = normalize_color_palette(color_palette)
    ex = generate(task_id, seed)
    instruction = render_instruction(
        task_id,
        ex.get("metadata"),
        color_palette=palette,
        prompt_variant=prompt_variant,
    )
    inp_text = render_input(task_id, ex["input"], fmt=fmt, color_palette=palette)
    out_text = render_output(ex["output"], fmt=fmt, task_id=task_id, color_palette=palette)
    if palette is not None:
        instruction = f"{SYMBOL_REMAP_PREAMBLE}\n\n{instruction}"

    prompt = f"{instruction}\n\nInput:\n{inp_text}"
    return prompt, out_text


# --- Bundle formatting --------------------------------------------------------

def format_bundle(
    task_id: str,
    seeds: list[int],
    fmt: str = GRID_FORMAT_SPACED,
    color_palette: Optional[Sequence[str]] = None,
    item_palettes: Optional[list[Sequence[str]]] = None,
    prompt_variant: Optional[str] = None,
    item_prompt_variants: Optional[list[Optional[str]]] = None,
) -> tuple[str, str]:
    """Format 1-5 examples of the same task as a bundled (prompt, answer) pair.

    If len(seeds)==1, returns the same format as format_single (no header).
    """
    palette = normalize_color_palette(color_palette)
    if palette is not None and item_palettes is not None:
        raise ValueError("Pass either color_palette or item_palettes, not both")
    if prompt_variant is not None and item_prompt_variants is not None:
        raise ValueError("Pass either prompt_variant or item_prompt_variants, not both")
    normalized_item_palettes = None
    if item_palettes is not None:
        if len(item_palettes) != len(seeds):
            raise ValueError("item_palettes must have the same length as seeds")
        normalized_item_palettes = [normalize_color_palette(p) for p in item_palettes]
    if item_prompt_variants is not None and len(item_prompt_variants) != len(seeds):
        raise ValueError("item_prompt_variants must have the same length as seeds")

    if len(seeds) == 1:
        single_palette = normalized_item_palettes[0] if normalized_item_palettes is not None else palette
        single_prompt_variant = item_prompt_variants[0] if item_prompt_variants is not None else prompt_variant
        return format_single(
            task_id,
            seeds[0],
            fmt=fmt,
            color_palette=single_palette,
            prompt_variant=single_prompt_variant,
        )

    n = len(seeds)
    header = (
        f"You will get {n} instances of the same task. "
        f"Solve each one separately and return Answer 1 through Answer {n} in order."
    )
    if normalized_item_palettes is not None:
        header = f"{header}\n\n{ITEM_SYMBOL_REMAP_PREAMBLE}"
    elif palette is not None:
        header = f"{header}\n\n{SYMBOL_REMAP_PREAMBLE}"

    task_blocks = []
    answer_blocks = []
    for i, seed in enumerate(seeds, 1):
        item_palette = normalized_item_palettes[i - 1] if normalized_item_palettes is not None else palette
        item_prompt_variant = item_prompt_variants[i - 1] if item_prompt_variants is not None else prompt_variant
        task_block, answer_block = _render_task_block(
            i, task_id, seed, fmt, item_palette, item_prompt_variant,
        )
        task_blocks.append(task_block)
        answer_blocks.append(answer_block)

    prompt = header + "\n\n" + "\n\n".join(task_blocks)
    answer = "\n\n".join(answer_blocks)
    return prompt, answer


# --- Mixed-task bundle formatting ---------------------------------------------

def format_mixed_bundle(
    items: Sequence[dict],
    fmt: str = GRID_FORMAT_SPACED,
    color_palette: Optional[Sequence[str]] = None,
    item_palettes: Optional[list[Sequence[str]]] = None,
    item_prompt_variants: Optional[list[Optional[str]]] = None,
) -> tuple[str, str]:
    """Format N *different* tasks as one bundled (prompt, answer) pair.

    ``items`` is a list of ``{"task_id": str, "seed": int}`` dicts, one per
    bundle position. Unlike ``format_bundle`` (N seeds of the same task), each
    item carries its own task_id and therefore its own instruction; the header
    tells the model the tasks are independent.

    Palette scope mirrors ``format_bundle``: pass ``color_palette`` for one
    shared mapping across the whole prompt (bundle scope) or ``item_palettes``
    (one per item) for item scope -- not both. ``item_prompt_variants`` selects
    a prompt variant per item (mixed bundles sample variants independently).

    If ``len(items) == 1`` this delegates to ``format_single`` (no header).
    """
    palette = normalize_color_palette(color_palette)
    if palette is not None and item_palettes is not None:
        raise ValueError("Pass either color_palette or item_palettes, not both")
    normalized_item_palettes = None
    if item_palettes is not None:
        if len(item_palettes) != len(items):
            raise ValueError("item_palettes must have the same length as items")
        normalized_item_palettes = [normalize_color_palette(p) for p in item_palettes]
    if item_prompt_variants is not None and len(item_prompt_variants) != len(items):
        raise ValueError("item_prompt_variants must have the same length as items")

    if len(items) == 1:
        single_palette = (
            normalized_item_palettes[0] if normalized_item_palettes is not None else palette
        )
        single_variant = item_prompt_variants[0] if item_prompt_variants is not None else None
        return format_single(
            str(items[0]["task_id"]),
            int(items[0]["seed"]),
            fmt=fmt,
            color_palette=single_palette,
            prompt_variant=single_variant,
        )

    n = len(items)
    header = (
        f"You will get {n} independent tasks. "
        f"Solve each one separately and return Answer 1 through Answer {n} in order."
    )
    if normalized_item_palettes is not None:
        header = f"{header}\n\n{ITEM_SYMBOL_REMAP_PREAMBLE}"
    elif palette is not None:
        header = f"{header}\n\n{SYMBOL_REMAP_PREAMBLE}"

    task_blocks = []
    answer_blocks = []
    for i, item in enumerate(items, 1):
        item_palette = (
            normalized_item_palettes[i - 1] if normalized_item_palettes is not None else palette
        )
        item_variant = item_prompt_variants[i - 1] if item_prompt_variants is not None else None
        task_block, answer_block = _render_task_block(
            i, str(item["task_id"]), int(item["seed"]), fmt, item_palette, item_variant,
        )
        task_blocks.append(task_block)
        answer_blocks.append(answer_block)

    prompt = header + "\n\n" + "\n\n".join(task_blocks)
    answer = "\n\n".join(answer_blocks)
    return prompt, answer


# --- Size estimation ----------------------------------------------------------

# Token budget: examples exceeding this get truncated, so cap bundles to fit.
# With max_seq_length=3072, leave room for chat template overhead (~150 tokens).
_MAX_TOKENS = 2900


def estimate_bundle_size(task_id: str, max_bundle: int = 5, fmt: str = GRID_FORMAT_SPACED,
                         max_tokens: int = _MAX_TOKENS, n_seeds: int = 40) -> int:
    """Estimate max safe bundle size by sampling and counting chars.

    Uses a conservative chars-to-tokens ratio of 1.2 (grid data is mostly
    single digits + spaces, where each character ≈ 1 token). ``n_seeds``
    examples are sampled and the largest sets the size; generators with
    independent 5-30 sides need well over a handful of seeds before the
    largest grids show up (5 seeds mis-sized 86/93 tasks at a 32k budget).
    """
    # Sample several seeds and take the max to account for variance
    max_single_chars = 0
    for seed in range(n_seeds):
        try:
            ex = generate(task_id, seed)
        except Exception:
            continue
        inp_text = render_input(task_id, ex["input"], fmt=fmt)
        out_text = render_output(ex["output"], fmt=fmt)
        prompt_text = get_prompt(task_id, ex.get("metadata"))
        single_chars = len(inp_text) + len(out_text) + len(prompt_text) + 50
        max_single_chars = max(max_single_chars, single_chars)

    if max_single_chars == 0:
        return 1

    # Conservative: grid data is ~1 char/token (digits, spaces, newlines)
    single_tokens = max_single_chars / 1.2

    for n in range(max_bundle, 0, -1):
        total_tokens = 50 + n * single_tokens  # 50 tokens for header
        if total_tokens <= max_tokens:
            return n
    return 1
