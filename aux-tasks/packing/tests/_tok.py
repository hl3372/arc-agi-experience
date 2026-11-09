"""The tokenizer for the tests that build whole rows (real chat-templated lengths).

It is ``$PV2_TOKENIZER`` (a local folder) or the hub id in ``pv2.tokens``; when neither loads (offline,
no local copy) those tests skip instead of failing.
"""
import pytest


def tokenizer_or_skip():
    from pv2.tokens import load_tokenizer
    try:
        return load_tokenizer()
    except Exception as err:  # noqa: BLE001  (offline, missing files, no transformers)
        pytest.skip(f"tokenizer not available ({type(err).__name__}); set PV2_TOKENIZER to a local Qwen3.5-4B folder")
