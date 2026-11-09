"""Merged counting IDs must not silently double the task sampling weight."""

import pytest

from prompts import PROMPT_TEMPLATES
from registry import RETIRED, generate, list_tasks


@pytest.mark.parametrize("retired,active", [("69a", "68a"), ("69b", "68b"), ("92b", "92")])
def test_merged_connectivity_tasks_have_one_active_registration(retired, active):
    assert RETIRED[retired] == active
    assert active in list_tasks() and active in PROMPT_TEMPLATES
    assert retired not in list_tasks() and retired not in PROMPT_TEMPLATES
    with pytest.raises(KeyError, match=f"merged into task {active}"):
        generate(retired, 0)
