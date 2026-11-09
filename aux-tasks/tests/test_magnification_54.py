"""Pinned local-transform siblings; task 54 is tested in test_rebuild_51_55."""
import hashlib
import json
import pytest
from registry import generate

# September 13 refresh changes only 39; 40/41 retain their September 12 pins.
SIBLING_HASHES = {
    "39": "bb0bbf25f8a6be79a29d31466a46f31cc5bef5f5",
    "40": "98f888d00cba2235de7a3f992f735bd86d29a860",
    "41": "66d229816a4cadf0576a70895408bfdc4122ba31",
}


@pytest.mark.parametrize("task_id", sorted(SIBLING_HASHES))
def test_sibling_tasks_unchanged(task_id):
    digest = hashlib.sha1()
    for seed in range(500):
        digest.update(json.dumps(generate(task_id, seed), sort_keys=True).encode())
    assert digest.hexdigest() == SIBLING_HASHES[task_id], task_id
