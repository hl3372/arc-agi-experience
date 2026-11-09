"""Real chat-templated token lengths, as the training run sees them.

Mirrors ``generate_dataset._real_templated_len`` of the old builder: the user + assistant turn
through the model's chat template, ids unwrapped from a BatchEncoding.  The tokenizer is
``--tokenizer`` (a path or hub id), else ``$PV2_TOKENIZER``, else ``Qwen/Qwen3.5-4B``.
"""
from __future__ import annotations

import functools
import os

DEFAULT_TOKENIZER = "Qwen/Qwen3.5-4B"   # the tokenizer our aux pool was measured with


@functools.lru_cache(maxsize=4)
def load_tokenizer(path: str | None = None):
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path or os.environ.get("PV2_TOKENIZER", DEFAULT_TOKENIZER))


def templated_len(tokenizer, prompt: str, answer: str) -> int:
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    try:
        out = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    except Exception:
        wrapped = ("<|im_start|>user\n" + prompt + "<|im_end|>\n"
                   "<|im_start|>assistant\n" + answer + "<|im_end|>\n")
        return len(tokenizer.encode(wrapped, add_special_tokens=False))
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return len(out)


def rough_len(prompt: str, answer: str) -> int:
    """Cheap upper-ish estimate for planning (digits are ~1 token each, words ~1.3)."""
    return int((len(prompt) + len(answer)) / 1.1) + 60
