"""The duplicate self-tiling task stays retired; its id now names the row/column reader.

Self-tiling (an n x n block repeated n times each way) was retired on 2026-09-14
because task 44 already covers it.  On 2026-09-29 the free id 77 was reused for a
new task (output one row or column), so 77 is active again, but no self-tiling
generator or prompt came back and 44 is unchanged.
"""
from prompts import PROMPT_TEMPLATES, get_prompt
from registry import RETIRED, generate, list_tasks


def test_self_tiling_stays_covered_by_44_and_77_is_the_row_column_task():
    assert "77" not in RETIRED
    assert "44" in list_tasks() and "44" in PROMPT_TEMPLATES
    prompt = get_prompt("77", generate("77", 0)["metadata"])
    assert prompt.startswith(("Output row ", "Output column "))
    assert "tile" not in prompt.lower() and "cop" not in prompt.lower()
