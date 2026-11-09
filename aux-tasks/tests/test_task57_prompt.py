"""Focused checks for the task 57 prompt and its uniqueness contract.

The recompute is a 4-connected flood from the grid border blocked by the wall
colour (the definition the prompt states), not a bounding-box crop, so organic
enclosures and thick walls are handled exactly like the solver must.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate
from symbol_remap import palette_for_example


FORBIDDEN_ORDINAL_WORDING = re.compile(
    r"color number|\b0-9\b|lower color index|lowest color index|"
    r"color order|earlier color|later color|greater color",
    re.I,
)

FREQUENCY_RULE = "unique most frequent non-background color strictly inside the wall; runner-up at least 2 cells behind"


def _enclosed_cells(grid: np.ndarray, wall: int) -> np.ndarray:
    h, w = grid.shape
    passable = grid != wall
    seen = np.zeros((h, w), dtype=bool)
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
    return passable & ~seen


def test_task57_prompt_and_metadata_do_not_use_color_ordinal_rule():
    for seed in range(40):
        ex = generate("57", seed)
        meta = ex["metadata"]
        prompts = [
            get_prompt("57", meta),
            get_prompt("57", meta, color_palette=palette_for_example("57", seed)),
        ]
        assert meta["frequency_rule"] == FREQUENCY_RULE
        for prompt in prompts:
            assert "occurs more often than any other" in prompt
            assert "strictly inside" in prompt
            assert "ignoring the background" in prompt
            assert "ignoring cells of color" not in prompt
            assert "Generated examples" not in prompt
        assert f"color {meta['enclosure_color']}" in prompts[0]
        assert not FORBIDDEN_ORDINAL_WORDING.search("\n".join(prompts))
        assert not FORBIDDEN_ORDINAL_WORDING.search(json.dumps(meta))


def test_task57_generated_examples_have_unique_enclosed_mode():
    for seed in range(300):
        ex = generate("57", seed)
        grid = np.array(ex["input"], dtype=int)
        meta = ex["metadata"]
        enclosed = _enclosed_cells(grid, meta["enclosure_color"])
        assert enclosed.any(), seed

        counts = Counter(int(color) for color in grid[enclosed])
        if meta["ignore_background"]:
            counts.pop(meta["background_color"], None)
        else:
            assert meta["background_color"] not in counts, seed
        ranked = sorted(counts.values(), reverse=True)
        max_count = ranked[0]
        winners = [color for color, count in counts.items() if count == max_count]

        assert len(winners) == 1, seed
        assert ex["output"] == str(winners[0]), seed
        assert max_count - (ranked[1] if len(ranked) > 1 else 0) >= 2, seed
