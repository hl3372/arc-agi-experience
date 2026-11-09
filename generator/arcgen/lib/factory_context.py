"""Read-only rendered-prompt preflight; never truncate or enlarge CLI context.

The fallback is deliberately conservative for English prose, JSON and grids,
but is not a tokenizer and cannot establish a model/account's actual capacity.
Near-limit estimates are flagged for live transport confirmation; estimates
over the reserved budget stop before any call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from math import ceil, floor, isfinite
from string import punctuation
from typing import Any


ESTIMATOR = "utf8_over_3_with_dense_json_and_unicode_floor_v1"
ESTIMATOR_FORMULA = "max(ceil(utf8_bytes / 3), ascii_digits + ascii_punctuation + non_ascii_utf8_bytes)"
NEAR_LIMIT_FRACTION = 0.90


@lru_cache(maxsize=4)
def _tokenizer(model: str | None) -> tuple[Any, dict[str, Any]]:
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return None, {"tokenizer_available": False}
    try:
        encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("o200k_base")
        mapping = "tiktoken_model_mapping" if model else "unverified_model_fallback"
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
        mapping = "unverified_model_fallback"
    return encoding, {
        "tokenizer_available": True, "tokenizer_version": tiktoken.__version__,
        "encoding": encoding.name, "model_mapping": mapping,
    }


class FactoryContextError(RuntimeError):
    """The unmodified prompt needs an explicit context decision before a call."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__(
            f"factory context preflight {report['status']} for {report['purpose']}: "
            f"{report['characters']} characters, {report['estimated_prompt_tokens']} "
            f"estimated prompt tokens versus {report['prompt_token_budget']} budget; "
            "no provider call, truncation, compaction, or context enlargement was requested. "
            "Measure tokens and verify this CLI/account's capacity before proceeding."
        )


def estimate_prompt_tokens(prompt: str) -> tuple[int, int]:
    """Return a tagged heuristic and a byte-level upper bound, not exact tokens.

    The second floor avoids the common chars/3 underestimate for dense ARC
    grids, JSON punctuation, and non-ASCII text. Arbitrary ASCII identifiers
    can still tokenize differently; the report never claims an exact count.
    """
    if not isinstance(prompt, str):
        raise TypeError("rendered prompt must be text")
    size = len(prompt.encode("utf-8"))
    dense_floor = sum(
        len(char.encode("utf-8")) if not char.isascii()
        else int(char.isdigit() or char in punctuation)
        for char in prompt
    )
    return max(ceil(size / 3), dense_floor), size


@dataclass(frozen=True)
class FactoryContextGuard:
    context_limit: int = 272_000
    usable_fraction: float = 0.95
    response_headroom: int = 32_768
    model: str | None = None

    def __post_init__(self) -> None:
        if type(self.context_limit) is not int or self.context_limit < 1:
            raise ValueError("context_limit must be a positive integer")
        if type(self.response_headroom) is not int or self.response_headroom < 0:
            raise ValueError("response_headroom must be a nonnegative integer")
        if (type(self.usable_fraction) not in (int, float)
                or not isfinite(self.usable_fraction)
                or not 0 < self.usable_fraction <= 0.95):
            raise ValueError("usable_fraction must be positive and no more than 0.95")
        if self.prompt_token_budget < 1:
            raise ValueError("response_headroom leaves no usable prompt context")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FactoryContextGuard:
        return cls(**dict(config.get("context", {})), model=config.get("model_profile", {}).get("model"))

    @property
    def prompt_token_budget(self) -> int:
        return floor(self.context_limit * self.usable_fraction) - self.response_headroom

    def inspect(self, prompt: str, purpose: str) -> dict[str, Any]:
        heuristic, size = estimate_prompt_tokens(prompt)
        encoding, tokenizer = _tokenizer(self.model)
        estimate = len(encoding.encode(prompt, disallowed_special=())) if encoding is not None else heuristic
        budget = self.prompt_token_budget
        if estimate > budget:
            status = "blocked_estimated_overflow"
        elif estimate >= floor(budget * NEAR_LIMIT_FRACTION):
            status = "estimated_fit_near_limit"
        else:
            status = "estimated_fit"
        return {
            "purpose": purpose,
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "characters": len(prompt), "utf8_bytes": size,
            "estimated_prompt_tokens": estimate,
            "byte_level_token_upper_bound": size,
            "conservative_heuristic_tokens": heuristic,
            "estimator": f"tiktoken.{encoding.name}_prompt_only" if encoding is not None else ESTIMATOR,
            "estimator_formula": "len(encoding.encode(prompt, disallowed_special=()))" if encoding is not None else ESTIMATOR_FORMULA,
            "tokenizer": tokenizer, "model": self.model,
            "exact_token_count": False,
            "measurement_scope": "rendered prompt only; excludes transport instructions/schema, response and reasoning tokens",
            "context_limit": self.context_limit,
            "usable_fraction": self.usable_fraction,
            "response_headroom": self.response_headroom,
            "prompt_token_budget": budget,
            "estimated_remaining_prompt_tokens": budget - estimate,
            "near_limit_fraction": NEAR_LIMIT_FRACTION,
            "status": status, "requires_context_decision": estimate > budget,
            "near_limit": estimate >= floor(budget * NEAR_LIMIT_FRACTION),
            "requires_live_no_compaction_confirmation": estimate >= floor(budget * NEAR_LIMIT_FRACTION),
            "model_account_capacity_verified": False,
            "prompt_modified": False,
        }

    def __call__(self, prompt: str, purpose: str) -> None:
        report = self.inspect(prompt, purpose)
        if report["requires_context_decision"]:
            raise FactoryContextError(report)
