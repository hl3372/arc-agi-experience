"""Regression checks for the topology family (tasks 36, 37; task 22 has tests/test_count_holes_22.py).

The oracle here is written independently of ``common.topology`` (plain BFS)
so a bug in the shared library cannot hide itself.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from registry import generate  # noqa: E402
from prompts import get_prompt  # noqa: E402

SEEDS = 300


def _reach_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable)
    dq: deque = deque()
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and passable[r, c]:
                seen[r, c] = True
                dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                dq.append((nr, nc))
    return seen


def _regions(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask)
    n = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                n += 1
                dq = deque([(r, c)])
                seen[r, c] = True
                while dq:
                    a, b = dq.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + dr, b + dc
                        if 0 <= na < h and 0 <= nb < w and mask[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True
                            dq.append((na, nb))
    return n


def _holes(grid: np.ndarray, obj: int) -> int:
    passable = grid != obj
    return _regions(passable & ~_reach_border(passable))


def _examples(task_id: str, n: int = SEEDS):
    for seed in range(n):
        ex = generate(task_id, seed)
        yield seed, ex, np.array(ex["input"], dtype=int), ex["metadata"]


def test_task36_answer_matches_independent_oracle_and_is_balanced():
    answers = Counter()
    cases = Counter()
    for seed, ex, grid, meta in _examples("36"):
        obj = meta["obj_color"]
        closed = _holes(grid, obj) > 0
        assert ex["output"] == ("closed" if closed else "open"), seed
        assert str(obj) in get_prompt("36", meta), seed
        answers[ex["output"]] += 1
        cases[meta["case"]] += 1
    assert 0.35 <= answers["closed"] / SEEDS <= 0.65, answers
    assert len(cases) >= 8, cases


def test_task37_repair_is_unique_fewest_cells_restoring_original_region():
    sizes = Counter()
    for seed, ex, grid, meta in _examples("37"):
        obj = meta["obj_color"]
        out = np.array(ex["output"], dtype=int)
        assert out.shape == grid.shape, seed
        assert grid.shape[0] * grid.shape[1] <= 900, seed
        diff = [tuple(int(v) for v in rc) for rc in np.argwhere(out != grid)]
        assert diff, seed
        assert all(out[r, c] == obj and grid[r, c] != obj for r, c in diff), seed
        assert sorted(diff) == sorted(tuple(c) for c in meta["removed_cells"]), seed
        # closed grid encloses something; opened grid lets that region reach the border
        passable_closed = out != obj
        enclosed = passable_closed & ~_reach_border(passable_closed)
        assert enclosed.any(), seed
        assert (_reach_border(grid != obj) & enclosed).any(), seed
        # uniqueness condition: every gap cell is 4-adjacent to the enclosed region
        h, w = grid.shape
        for r, c in diff:
            assert any(
                0 <= r + dr < h and 0 <= c + dc < w and enclosed[r + dr, c + dc]
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ), (seed, (r, c))
        assert str(obj) in get_prompt("37", meta), seed
        sizes[len(diff)] += 1
    assert min(sizes) == 1 and max(sizes) >= 3, sizes


def test_topology_tasks_are_deterministic():
    for tid in ("36", "37"):
        for seed in (0, 7, 123):
            a = generate(tid, seed)
            b = generate(tid, seed)
            assert a["input"] == b["input"] and a["output"] == b["output"], (tid, seed)
