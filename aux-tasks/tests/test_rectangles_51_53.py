"""Pinned sibling outputs; current frame-measurement tests are in test_rebuild_51_55."""
import hashlib
import json
import os
from registry import generate
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "rectangles_siblings_49_50_67.json")


def test_siblings_49_50_67_byte_identical():
    with open(FIXTURE) as fh:
        expected = json.load(fh)
    assert set(expected) == {"49", "50", "67"}
    for tid, hashes in expected.items():
        assert len(hashes) == 300
        for seed, want in enumerate(hashes):
            got = hashlib.sha1(json.dumps(generate(tid, seed), sort_keys=True).encode()).hexdigest()
            assert got == want, f"task {tid} seed {seed} changed"
