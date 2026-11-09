"""Small helpers shared by the assembler modules: weighted picks, number and ordinal words."""
from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")

ORDINAL_WORDS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
                 "tenth", "eleventh", "twelfth")
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve")
WORD_TO_NUMBER = {w: i for i, w in enumerate(NUMBER_WORDS)}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def pick(rng, items: Sequence[T], weights: Sequence[float] | None = None) -> T:
    """One item, uniformly or by (unnormalised) weights; deterministic for a numpy Generator."""
    if not items:
        raise ValueError("nothing to pick from")
    if weights is None:
        return items[int(rng.integers(len(items)))]
    w = [float(x) for x in weights]
    total = sum(w)
    if total <= 0:
        raise ValueError("weights sum to zero")
    return items[int(rng.choice(len(items), p=[x / total for x in w]))]


def pick_weighted(rng, table: dict, allowed=None):
    """A key of ``table`` (key -> weight) among ``allowed`` keys (all when None) with weight > 0."""
    keys = [k for k, w in table.items() if w > 0 and (allowed is None or k in allowed)]
    if not keys:
        raise ValueError(f"no allowed option among {sorted(table)} (allowed={allowed})")
    return pick(rng, keys, [table[k] for k in keys])


def chance(rng, p: float) -> bool:
    return bool(rng.random() < p)


def count_phrase(n: int, noun: str, rng=None, words: bool = False) -> str:
    """"3 grids" / "one grid" / "four questions" (words only up to twelve)."""
    if rng is not None:
        words = words or chance(rng, 0.35)
    num = NUMBER_WORDS[n] if words and n <= 12 else str(n)
    return f"{num} {noun}" + ("" if n == 1 else "s")


def letter(k: int) -> str:
    if not 0 <= k < len(LETTERS):
        raise ValueError(f"no letter for index {k}")
    return LETTERS[k]


def join_tags(tags: list[str]) -> str:
    """"3", "3 and 4", "3, 4 and 5"."""
    if len(tags) == 1:
        return tags[0]
    return ", ".join(tags[:-1]) + " and " + tags[-1]
