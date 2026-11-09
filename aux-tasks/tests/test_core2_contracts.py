"""Contract checks for second-batch core task additions."""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate, list_tasks
from sft_wrapper import format_single
from symbol_remap import SYMBOL_REMAP_PREAMBLE, palette_for_example


CORE2_TASK_IDS = [
    "93", "94", "95", "96", "118", "133", "134", "135", "136", "137",
    "138", "145", "146", "147", "153", "154", "157", "158", "162", "164",
    "97", "98", "105", "106", "108", "109", "110", "111", "112", "113",
    "114", "115", "116", "117", "119", "124", "125", "126", "131", "132",
    "139", "140", "141", "142", "143", "144", "149", "150", "151", "152",
    "155", "156", "159", "160", "161", "163", "165", "166", "167", "168",
    "169", "170", "171", "172", "173", "174", "177", "180", "181", "182",
    "99", "101", "102", "103", "104", "107", "120", "127", "128",
    "129", "130", "148", "175", "176", "178", "179", "183", "184", "185",
    "186", "187", "188", "189", "190", "191", "192", "194", "195", "196",
    "197", "198", "199", "200", "201", "202", "203", "227", "228", "229",
    "230", "231", "249", "250", "251", "252", "253", "254", "255", "256",
    "121", "122", "123", "193", "204", "205", "206", "207", "208", "209",
    "210", "211", "212", "213", "214", "215", "216", "217", "218", "219",
    "220", "221", "222", "223", "224", "225", "226", "232", "233", "234",
    "235", "236", "237", "238", "239", "240", "241", "242", "243", "244",
    "245", "246", "247", "248", "257", "258", "259", "281",
]

RAW_COLOR_RE = re.compile(r"\bcolor [0-9]\b(?!-)", re.I)
COLOR_INDEX_RE = re.compile(r"\bcolor index\b", re.I)


def test_core2_task_ids_registered():
    ids = set(list_tasks())
    missing = [task_id for task_id in CORE2_TASK_IDS if task_id not in ids]
    assert not missing


def test_core2_examples_are_json_serializable_and_prompted():
    errors = []
    for task_id in CORE2_TASK_IDS:
        for seed in range(20):
            try:
                ex = generate(task_id, seed)
                json.dumps(ex)
                prompt = get_prompt(task_id, ex.get("metadata"))
                if "[No prompt" in prompt:
                    raise AssertionError(prompt)
            except Exception as exc:
                errors.append(f"{task_id}/{seed}: {type(exc).__name__}: {exc}")
    assert not errors, "\n".join(errors[:100])


def test_core2_symbol_prompts_do_not_depend_on_hidden_color_indices():
    errors = []
    for task_id in CORE2_TASK_IDS:
        for seed in range(20):
            palette = palette_for_example(task_id, seed)
            ex = generate(task_id, seed)
            prompt = get_prompt(task_id, ex.get("metadata"), color_palette=palette)
            if RAW_COLOR_RE.search(prompt):
                errors.append(f"{task_id}/{seed}: raw color digit in prompt: {prompt!r}")
            if COLOR_INDEX_RE.search(prompt):
                errors.append(f"{task_id}/{seed}: hidden color-index wording in symbol prompt: {prompt!r}")
            rendered_prompt, _answer = format_single(task_id, seed, color_palette=palette)
            if SYMBOL_REMAP_PREAMBLE not in rendered_prompt:
                errors.append(f"{task_id}/{seed}: missing symbol-remap preamble")
    assert not errors, "\n".join(errors[:100])


if __name__ == "__main__":
    tests = [
        test_core2_task_ids_registered,
        test_core2_examples_are_json_serializable_and_prompted,
        test_core2_symbol_prompts_do_not_depend_on_hidden_color_indices,
    ]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
