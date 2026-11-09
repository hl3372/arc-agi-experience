"""The classic family: today's formats, byte-identical to the old builder (DESIGN.md 4, C6).

A copy of ``sft_wrapper.py``'s ``format_single`` / ``format_bundle`` / ``format_mixed_bundle`` for
the configuration the training runs use -- ``fmt="compact"`` and no symbol palette -- so v2 never
imports the old builder at runtime.  ``tests/test_classic.py`` compares the output with the old
builder byte for byte.  Every task the old builder formats is covered, including 14 (text input)
and 41 / 78 (two-part inputs), which stay in the classic formats.

    <instruction>\\n\\nInput:\\n<grid, compact digits>          ->  <bare answer>
    bundles: header + "Task i\\nInstruction: ...\\nInput:\\n..."  ->  "Answer i:\\n..." blocks

``prompt_variant`` is a stored template string (``prompt_variants.json``), as in the old builder;
``variant_template(task_id, index)`` looks one up by its index (0 = canonical).
"""
from __future__ import annotations

from typing import Optional, Sequence

from pv2.paths import ensure_generator_package

ensure_generator_package()
import prompt_variants as _pv  # noqa: E402
from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402


def variant_template(task_id: str, index: int) -> Optional[str]:
    """The template of the ``index``-th approved stored variant (1-based); None for 0 (canonical)."""
    if int(index) == 0:
        return None
    return _pv.variants_for(str(task_id), _store())[int(index) - 1]["template"]


_STORE = None


def _store() -> dict:
    global _STORE
    if _STORE is None:
        _STORE = _pv.load()
    return _STORE


# --- instruction rendering (sft_wrapper.render_instruction, no palette) ------------------------------

def render_instruction(task_id: str, meta: Optional[dict], prompt_variant: Optional[str] = None) -> str:
    """The canonical instruction, or a stored variant template rendered through
    ``prompt_variants.render`` (compound tokens expanded; an unknown token raises)."""
    if prompt_variant is None:
        return get_prompt(task_id, meta)
    return _pv.render(task_id, prompt_variant, meta=meta)


# --- grid / input / output rendering (compact, no palette) ------------------------------------------

def render_grid(grid) -> str:
    """Compact digit rows: '123\\n456'."""
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


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


def _render_named_payload(value) -> str:
    if _looks_like_grid(value):
        return render_grid(value)
    if isinstance(value, dict):
        return "\n\n".join(f"{key}:\n{_render_named_payload(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "\n\n".join(f"item {idx}:\n{_render_named_payload(item)}" for idx, item in enumerate(value, start=1))
    return str(value)


def render_input(task_id: str, inp) -> str:
    """An example's input as text (text inputs verbatim, two-part inputs labelled, grids compact)."""
    if task_id in _TEXT_INPUT_TASKS and isinstance(inp, str):
        return str(inp)
    if task_id in _DICT_INPUT_TASKS and isinstance(inp, dict):
        parts = []
        for key in _DICT_INPUT_TASKS[task_id]:
            val = inp[key]
            if _looks_like_grid(val):
                parts.append(f"{key}:\n{render_grid(val)}")
            else:
                parts.append(f"{key}:\n{val}")
        return "\n\n".join(parts)
    if isinstance(inp, dict):
        return _render_named_payload(inp)
    if _looks_like_grid(inp):
        return render_grid(inp)
    return str(inp)


def render_output(output) -> str:
    """Grid outputs compact; every other output ``str(output)``."""
    if isinstance(output, list) and output and isinstance(output[0], list):
        return render_grid(output)
    return str(output)


def _task_block(idx: int, task_id: str, seed: int, prompt_variant: Optional[str]) -> tuple[str, str]:
    ex = generate(task_id, seed)
    instruction = render_instruction(task_id, ex.get("metadata"), prompt_variant)
    inp_text = render_input(task_id, ex["input"])
    out_text = render_output(ex["output"])
    return f"Task {idx}\nInstruction: {instruction}\nInput:\n{inp_text}", f"Answer {idx}:\n{out_text}"


# --- the three formats ------------------------------------------------------------------------------

def classic_single(task_id: str, seed: int, prompt_variant: Optional[str] = None) -> tuple[str, str]:
    """One example as (user, assistant): ``<instruction>\\n\\nInput:\\n<grid>`` -> the bare answer."""
    task_id = str(task_id)
    ex = generate(task_id, int(seed))
    instruction = render_instruction(task_id, ex.get("metadata"), prompt_variant)
    inp_text = render_input(task_id, ex["input"])
    out_text = render_output(ex["output"])
    return f"{instruction}\n\nInput:\n{inp_text}", out_text


def classic_bundle(task_id: str, seeds: Sequence[int], prompt_variant: Optional[str] = None,
                   item_prompt_variants: Optional[Sequence[Optional[str]]] = None) -> tuple[str, str]:
    """1-5 examples of one task; a single seed is exactly ``classic_single`` (no header)."""
    task_id = str(task_id)
    seeds = [int(s) for s in seeds]
    if prompt_variant is not None and item_prompt_variants is not None:
        raise ValueError("Pass either prompt_variant or item_prompt_variants, not both")
    if item_prompt_variants is not None and len(item_prompt_variants) != len(seeds):
        raise ValueError("item_prompt_variants must have the same length as seeds")
    if len(seeds) == 1:
        single = item_prompt_variants[0] if item_prompt_variants is not None else prompt_variant
        return classic_single(task_id, seeds[0], single)
    n = len(seeds)
    header = (f"You will get {n} instances of the same task. "
              f"Solve each one separately and return Answer 1 through Answer {n} in order.")
    blocks, answers = [], []
    for i, seed in enumerate(seeds, 1):
        variant = item_prompt_variants[i - 1] if item_prompt_variants is not None else prompt_variant
        block, answer = _task_block(i, task_id, seed, variant)
        blocks.append(block)
        answers.append(answer)
    return header + "\n\n" + "\n\n".join(blocks), "\n\n".join(answers)


def classic_mixed(items: Sequence[dict],
                  item_prompt_variants: Optional[Sequence[Optional[str]]] = None) -> tuple[str, str]:
    """N different tasks (``[{"task_id", "seed"}, ...]``); a single item is ``classic_single``."""
    if item_prompt_variants is not None and len(item_prompt_variants) != len(items):
        raise ValueError("item_prompt_variants must have the same length as items")
    if len(items) == 1:
        variant = item_prompt_variants[0] if item_prompt_variants is not None else None
        return classic_single(str(items[0]["task_id"]), int(items[0]["seed"]), variant)
    n = len(items)
    header = (f"You will get {n} independent tasks. "
              f"Solve each one separately and return Answer 1 through Answer {n} in order.")
    blocks, answers = [], []
    for i, item in enumerate(items, 1):
        variant = item_prompt_variants[i - 1] if item_prompt_variants is not None else None
        block, answer = _task_block(i, str(item["task_id"]), int(item["seed"]), variant)
        blocks.append(block)
        answers.append(answer)
    return header + "\n\n" + "\n\n".join(blocks), "\n\n".join(answers)
