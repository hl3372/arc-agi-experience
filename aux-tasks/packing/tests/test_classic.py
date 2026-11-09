"""The classic family is byte-identical to the old builder (sft_wrapper.py, fmt="compact", no palette).

The old builder is imported BY FILE PATH from the parent workspace (skipped when it is absent), with
``registry`` / ``prompts`` / ``prompt_variants`` already resolved to the generator package.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

import prompt_variants as pv
import registry
from pv2 import classic
from pv2.paths import DEFAULT_GENERATOR_ROOT, WORKSPACE_ROOT

OLD_PATH = os.path.join(WORKSPACE_ROOT, "sft_wrapper.py")


@pytest.fixture(scope="module")
def old():
    if not os.path.isfile(OLD_PATH):
        pytest.skip("the old builder (sft_wrapper.py) is not present")
    gen_root = os.path.realpath(DEFAULT_GENERATOR_ROOT)
    for name in ("registry", "prompts", "prompt_variants"):
        __import__(name)
        assert os.path.realpath(os.path.dirname(sys.modules[name].__file__)) == gen_root, name
    spec = importlib.util.spec_from_file_location("_old_sft_wrapper", OLD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.generate is registry.generate
    return module


def _all_tasks():
    import re
    return [t for t in registry.list_tasks() if int(re.match(r"\d+", t).group(0)) <= 102]


def _variant(rng, tid, payload):
    rows = pv.variants_for(tid, payload)
    k = int(rng.integers(0, len(rows) + 1))
    return None if k == 0 else rows[k - 1]["template"]


def test_classic_is_byte_identical(old):
    rng = np.random.default_rng(29)
    payload = pv.load()
    tasks = _all_tasks()
    assert {"14", "41", "78"} <= set(tasks)
    n = 0
    # singles, over every task at least once
    for i in range(190):
        tid = tasks[i % len(tasks)] if i < len(tasks) else tasks[int(rng.integers(len(tasks)))]
        seed = int(rng.integers(0, 2 ** 31))
        v = _variant(rng, tid, payload)
        assert classic.classic_single(tid, seed, v) == old.format_single(tid, seed, fmt="compact", prompt_variant=v)
        n += 1
    # same-task bundles (1-5 seeds; one shared variant or one per item)
    for _ in range(70):
        tid = tasks[int(rng.integers(len(tasks)))]
        seeds = [int(s) for s in rng.integers(0, 2 ** 31, size=int(rng.integers(1, 6)))]
        if rng.random() < 0.5:
            v = _variant(rng, tid, payload)
            got = classic.classic_bundle(tid, seeds, prompt_variant=v)
            want = old.format_bundle(tid, seeds, fmt="compact", prompt_variant=v)
        else:
            vs = [_variant(rng, tid, payload) for _ in seeds]
            got = classic.classic_bundle(tid, seeds, item_prompt_variants=vs)
            want = old.format_bundle(tid, seeds, fmt="compact", item_prompt_variants=vs)
        assert got == want
        n += 1
    # mixed bundles (1-5 different tasks, variants per item or none)
    for _ in range(70):
        k = int(rng.integers(1, 6))
        chosen = [tasks[int(i)] for i in rng.choice(len(tasks), size=k, replace=False)]
        items = [{"task_id": t, "seed": int(rng.integers(0, 2 ** 31))} for t in chosen]
        vs = [_variant(rng, t, payload) for t in chosen] if rng.random() < 0.7 else None
        got = classic.classic_mixed(items, item_prompt_variants=vs)
        want = old.format_mixed_bundle(items, fmt="compact", item_prompt_variants=vs)
        assert got == want
        n += 1
    assert n >= 300


def test_classic_shapes():
    prompt, answer = classic.classic_single("1", 0)
    assert prompt.startswith("What are the dimensions of this grid? Answer as HxW.\n\nInput:\n")
    assert answer == registry.generate("1", 0)["output"]
    prompt, answer = classic.classic_bundle("30", [1, 2, 3])
    assert prompt.startswith("You will get 3 instances of the same task.")
    assert answer.startswith("Answer 1:\n") and "\n\nAnswer 3:\n" in answer
    prompt, _ = classic.classic_mixed([{"task_id": "41", "seed": 0}, {"task_id": "14", "seed": 1}])
    assert "Task 1\nInstruction: " in prompt and "\nInput:\nbefore:\n" in prompt
    assert classic.variant_template("77", 0) is None
    assert classic.variant_template("77", 1) == pv.variants_for("77")[0]["template"]
    with pytest.raises(ValueError):
        classic.classic_bundle("30", [1, 2], prompt_variant="x", item_prompt_variants=[None, None])
