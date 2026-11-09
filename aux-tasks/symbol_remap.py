"""Color-symbol palette helpers for ARC auxiliary task rendering.

Generators continue to use integer color indices internally. This module only
controls how those color labels are rendered in prompts, grids, and selected
color-valued string outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


ColorPalette = tuple[str, ...]

COLOR_PALETTE_META_KEY = "__color_palette__"

SYMBOL_REMAP_PREAMBLE = (
    "Colors are represented by symbols. Treat identical symbols as the same color."
)

ITEM_SYMBOL_REMAP_PREAMBLE = (
    "Colors are represented by symbols. Within each task block, treat identical "
    "symbols as the same color; symbol meanings may differ between task blocks."
)

# Curated non-digit subset of the tokenizer-stable clique in
# tokenizer_atoms/atoms_summary.md. Keep every entry a single visible codepoint.
_SYMBOL_BANK_TEXT = (
    "µßàáâãåæçèéêëìíîïðñòôõøùúûýþ"
    "āăďēėěğģħĩīįķĺļľłńņňōőœŗśşťũūůűųźż"
    "βδζθκξπςφχψϊбгжзйхцчшщъђѓѕїјљњћќў"
    "ғҙқҡңұҳҷһәӣӯ"
    "□▪▲△▼▽◇◎●◯♡♥♦♪✓✔"
)

SYMBOL_BANK: ColorPalette = tuple(dict.fromkeys(_SYMBOL_BANK_TEXT))

# Name under which the built-in bank is exposed via load_bank_by_name().
BUILTIN_BANK_NAME = "latin_mixed_123"

# Registry of named banks beyond the built-in (loaded lazily).
_BANKS_REGISTRY_PATH = Path(__file__).parent / "tokenizer_atoms" / "symbol_banks.json"


def _validate_bank_chars(bank: Sequence[str], name: str = "bank") -> None:
    """Shared validation for any symbol bank (built-in or loaded)."""
    if len(bank) < 10:
        raise ValueError(f"{name} must contain at least 10 symbols, got {len(bank)}")
    for symbol in bank:
        if len(symbol) != 1:
            raise ValueError(f"{name} symbol must be one codepoint: {symbol!r}")
        if symbol.isdigit():
            raise ValueError(f"{name} must not contain digits: {symbol!r}")
        if symbol.isspace():
            raise ValueError(f"{name} must not contain whitespace")


_validate_bank_chars(SYMBOL_BANK, name="SYMBOL_BANK (built-in)")


# --- Module-level "active bank" (set once at run start by train/gen scripts) ---

_ACTIVE_BANK: ColorPalette = SYMBOL_BANK


def get_active_bank() -> ColorPalette:
    """Return the bank that helpers default to when no `bank=` is passed."""
    return _ACTIVE_BANK


def set_active_bank(bank: Sequence[str]) -> None:
    """Set the module-level active bank used by sample/palette helpers.

    Call this once at the start of a run (after parsing --symbol-bank), before
    any cache generation or eval-palette computation, so the entire pipeline
    sees the same bank.
    """
    global _ACTIVE_BANK
    _validate_bank_chars(bank, name="active bank")
    _ACTIVE_BANK = tuple(bank)


def load_bank_by_name(name: str) -> ColorPalette:
    """Resolve a bank name to its character tuple.

    Special-cases ``BUILTIN_BANK_NAME`` (the curated 123-char Latin/Greek/
    Cyrillic/geometric bank baked into this module). All other names are looked
    up in ``tokenizer_atoms/symbol_banks.json``.

    Raises ValueError on unknown name; FileNotFoundError if the registry is
    missing for a non-built-in lookup.
    """
    if name == BUILTIN_BANK_NAME:
        return SYMBOL_BANK
    if not _BANKS_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Symbol bank registry not found: {_BANKS_REGISTRY_PATH}"
        )
    with open(_BANKS_REGISTRY_PATH) as f:
        registry = json.load(f)
    if name not in registry:
        avail = sorted({BUILTIN_BANK_NAME, *registry.keys()})
        raise ValueError(f"Unknown symbol bank {name!r}. Available: {avail}")
    entry = registry[name]
    text = entry["text"]
    bank = tuple(dict.fromkeys(text))
    declared = entry.get("size")
    if declared is not None and len(bank) != declared:
        raise ValueError(
            f"Bank {name!r}: dedup size {len(bank)} != declared size {declared}"
        )
    _validate_bank_chars(bank, name=f"bank {name!r}")
    return bank


def normalize_color_palette(
    color_palette: Optional[Sequence[str]],
) -> Optional[ColorPalette]:
    """Validate and normalize a 10-symbol color palette."""
    if color_palette is None:
        return None
    palette = tuple(str(s) for s in color_palette)
    if len(palette) != 10:
        raise ValueError(f"Color palette must have 10 symbols, got {len(palette)}")
    if len(set(palette)) != 10:
        raise ValueError("Color palette symbols must be distinct")
    for symbol in palette:
        if not symbol or symbol.isspace():
            raise ValueError("Color palette symbols must be visible")
        if len(symbol) != 1:
            raise ValueError(f"Color palette symbol must be one codepoint: {symbol!r}")
        if symbol.isdigit():
            raise ValueError(f"Color palette must not contain digits: {symbol!r}")
    return palette


def coerce_color_index(value) -> Optional[int]:
    """Return an int color index for ints, digit strings, or color-name strings."""
    from common.grid import COLOR_INDEX

    if isinstance(value, int):
        return value if 0 <= value <= 9 else None
    if isinstance(value, str):
        if value in COLOR_INDEX:
            return COLOR_INDEX[value]
        if len(value) == 1 and value.isdigit():
            idx = int(value)
            return idx if 0 <= idx <= 9 else None
    return None


def render_color_label(value, color_palette: Optional[Sequence[str]] = None) -> str:
    """Render a color label through *color_palette* when present."""
    palette = normalize_color_palette(color_palette)
    idx = coerce_color_index(value)
    if palette is not None and idx is not None:
        return palette[idx]
    return str(value) if value is not None else ""


def render_pattern_grid(
    grid: list[list[int]],
    color_palette: Optional[Sequence[str]] = None,
    row_sep: str = "\n",
    compact: bool = True,
) -> str:
    """Render a small literal grid/pattern with optional color remapping."""
    palette = normalize_color_palette(color_palette)
    rows = []
    for row in grid:
        cells = [render_color_label(int(c), palette) for c in row]
        rows.append("".join(cells) if compact else " ".join(cells))
    return row_sep.join(rows)


def sample_color_palette(
    rng: np.random.Generator,
    bank: Optional[Sequence[str]] = None,
) -> ColorPalette:
    """Sample 10 distinct non-digit color symbols.

    If ``bank`` is None, uses the module-level active bank
    (see ``set_active_bank``).
    """
    if bank is None:
        bank = get_active_bank()
    if len(bank) < 10:
        raise ValueError("Need at least 10 symbols to sample a color palette")
    indices = rng.choice(len(bank), size=10, replace=False)
    return normalize_color_palette(tuple(bank[int(i)] for i in indices))  # type: ignore[return-value]


def sample_color_palettes(
    rng: np.random.Generator,
    n_items: int,
    scope: str = "bundle",
    bank: Optional[Sequence[str]] = None,
) -> list[ColorPalette]:
    """Sample palettes for a formatted row.

    `bundle` scope returns the same palette for all items. `item` scope returns
    one palette per item and samples disjoint symbols where the bank is large
    enough. Item scope is intentionally harder/noisier and should be called out
    in run docs.

    If ``bank`` is None, uses the module-level active bank.
    """
    if bank is None:
        bank = get_active_bank()
    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    if scope == "bundle":
        palette = sample_color_palette(rng, bank=bank)
        return [palette for _ in range(n_items)]
    if scope != "item":
        raise ValueError(f"Unknown symbol map scope: {scope!r}")

    if len(bank) >= n_items * 10:
        indices = rng.choice(len(bank), size=n_items * 10, replace=False)
        palettes = []
        for i in range(n_items):
            chunk = indices[i * 10:(i + 1) * 10]
            palettes.append(
                normalize_color_palette(tuple(bank[int(j)] for j in chunk))  # type: ignore[arg-type]
            )
        return palettes

    return [sample_color_palette(rng, bank=bank) for _ in range(n_items)]


def palette_for_example(task_id: str, seed: int, salt: int = 0,
                        bank: Optional[Sequence[str]] = None) -> ColorPalette:
    """Deterministically sample a palette for eval examples.

    Uses the module-level active bank when ``bank`` is None, so a single
    ``set_active_bank()`` call at run start makes eval palettes consistent
    with the training-cache palettes.
    """
    key = f"{task_id}:{seed}:{salt}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    palette_seed = int.from_bytes(digest, "little", signed=False)
    return sample_color_palette(np.random.default_rng(palette_seed), bank=bank)


def reverse_palette(color_palette: Sequence[str]) -> dict[str, int]:
    palette = normalize_color_palette(color_palette)
    assert palette is not None
    return {symbol: idx for idx, symbol in enumerate(palette)}


def render_text_input(
    task_id: str,
    text: str,
    color_palette: Optional[Sequence[str]] = None,
) -> str:
    """Remap color references inside special text-input tasks."""
    palette = normalize_color_palette(color_palette)
    if palette is None:
        return str(text)

    if task_id == "14":
        return re.sub(
            r"\bcolor ([0-9])\b",
            lambda m: f"color {palette[int(m.group(1))]}",
            str(text),
        )

    return str(text)


_COLOR_STRING_OUTPUT_TASKS = {"25", "57", "75"}


def render_string_output(
    task_id: Optional[str],
    output,
    color_palette: Optional[Sequence[str]] = None,
) -> str:
    """Render string outputs, remapping only color-valued strings."""
    text = str(output)
    palette = normalize_color_palette(color_palette)
    if palette is None or task_id is None:
        return text

    stripped = text.strip()
    if task_id in _COLOR_STRING_OUTPUT_TASKS:
        idx = coerce_color_index(stripped)
        if idx is not None:
            return palette[idx]
        return text

    if task_id == "76":
        parts = []
        for entry in text.split(", "):
            if ":" not in entry:
                return text
            color, count = entry.split(":", 1)
            idx = coerce_color_index(color.strip())
            if idx is None:
                return text
            parts.append(f"{palette[idx]}:{count.strip()}")
        return ", ".join(parts)

    return text
