"""Reference schemes: labels, ordinals, roles, qualification and the resolver."""
import importlib.util
import os

import numpy as np
import pytest

from pv2.assemble import refs as R
from pv2.assemble.row import _allowed_nouns
from pv2.model import LABEL_WORD_PREFIX, Answer, Part, Query, Slot
from pv2.paths import SOLVER_PROMPT_DIR


def _arc_render():
    path = os.path.join(SOLVER_PROMPT_DIR, "arc_render.py")
    if not os.path.isfile(path):
        pytest.skip("the solver renderer (../training/arcsolver/prompt/arc_render.py) is not present")
    spec = importlib.util.spec_from_file_location("arc_render_for_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ordinals_read_within_the_part_and_across_the_row():
    row = [[None, None, None], ["Map A", "Map B"]]
    assert R.resolve("the second grid", 0, row) == {(0, 1)}
    # with "Grid A", "the first grid" could also mean Grid A (the first letter): ambiguous
    assert R.resolve("the first grid", 0, [[None, None], ["Grid A"]]) == {(0, 0), (1, 0)}
    assert R.resolve("the second grid", 0, [[None, None], ["Grid 2"]]) == {(0, 1), (1, 0)}
    assert R.resolve("this part's first grid", 0, row) == {(0, 0)}
    # "the last grid" of part 0 is not the row's last grid
    assert R.resolve("the last grid", 0, row) == {(0, 2), (1, 1)}
    assert R.resolve("this part's last grid", 0, row) == {(0, 2)}
    # an ordinal in a later part differs from the row-wide count
    row2 = [["Image 1"], [None, None, None]]
    assert R.resolve("the second grid", 1, row2) == {(1, 1), (1, 0)}
    assert R.resolve("this section's second grid", 1, row2) == {(1, 1)}
    # a single-part row: both readings agree
    assert R.resolve("the last grid", 0, [[None] * 4]) == {(0, 3)}
    assert R.resolve("the second-to-last grid", 0, [[None] * 4]) == {(0, 2)}
    assert R.resolve("the fifth grid", 0, [[None] * 4]) == set()


def test_labels_resolve_by_print_only_and_must_be_unique():
    row = [["Training pair 1 input", "Training pair 1 output"], ["Training pair 1 input", "Test input 1"]]
    assert R.resolve("Training pair 1 input", 1, row) == {(0, 0), (1, 0)}
    assert R.resolve("this part's Training pair 1 input", 1, row) == {(1, 0)}
    assert R.resolve("Test input 1", 0, row) == {(1, 1)}
    assert R.resolve("Grid 3", 0, [["Grid 3"], [None, None, None]]) == {(0, 0)}
    assert R.resolve("the grid", 1, [[None, None], ["Map 1"]]) == {(1, 0)}


def test_label_book_never_repeats_a_label():
    rng = np.random.default_rng(0)
    book = R.LabelBook()
    seen = []
    for _ in range(40):
        noun, style, labels = book.assign(rng, int(rng.integers(1, 6)), ["Grid", "Map", "#"],
                                          {"Grid": 1, "Map": 1, "#": 1}, 0.5)
        seen += labels
    assert len(seen) == len(set(seen))
    assert all(lab.startswith(("Grid ", "Map ", "#")) for lab in seen)


def test_ordinal_phrases_are_words_only():
    rng = np.random.default_rng(1)
    for n in range(2, 15):
        for _ in range(20):
            for i, ph in enumerate(R.ordinal_phrases(rng, n)):
                if ph is None:
                    assert i >= 12 and i < n - 2
                    continue
                assert not any(ch.isdigit() for ch in ph)
                assert R.resolve(ph, 0, [[None] * n]) == {(0, i)}


def test_role_labels_are_the_stage2_names():
    ar = _arc_render()
    slots = [Slot([[1]], role="train_input", pair=0), Slot([[2]], role="train_output", pair=0),
             Slot([[3]], role="train_input", pair=1), Slot([[4]], role="train_output", pair=1),
             Slot([[5]], role="test_input", pair=0)]
    for base in (0, 1):
        labels = R.role_labels(slots, base, R.LabelBook())
        probe_names = [f"Training pair {k + base} {r}" for k in (0, 1) for r in ("input", "output")] + \
                      [f"Test input {base}"]
        assert labels == probe_names
        block = ar._indexed_block(labels[0], [[1, 2]])
        assert block.split("\n")[0] == R.header_text(labels[0], [[1, 2]]) + ":"


def test_label_mentions_respect_boundaries():
    text = "Compare Grid 1 with Grid 10 and #1 with #12; this part's Grid 1 matters."
    found = [(lab, q) for _, _, lab, q in R.find_label_mentions(text, ["Grid 1", "Grid 10", "#1", "#12"])]
    assert found == [("Grid 1", False), ("Grid 10", False), ("#1", False), ("#12", False), ("Grid 1", True)]


def test_label_nouns_avoid_the_rows_words():
    g = [[1]]
    q1 = Query("item", "item:30", (0,), "How many cells of this image are not color 0?", Answer("number", 1, "1"),
               flags=frozenset({LABEL_WORD_PREFIX + "image"}), lead_clause=True)
    q2 = Query("lib", "lib:x", (0,), "Is ⟪g0⟫ a map of the panel?", Answer("bool", True, "yes"))
    nouns = _allowed_nouns([Part("x", [Slot(g)], [q1]), Part("y", [Slot(g)], [q2])])
    assert "Image" not in nouns and "Map" not in nouns and "Panel" not in nouns
    assert "Grid" in nouns and "Figure" in nouns and "#" in nouns
    q3 = Query("item", "item:31", (0,), "Look at the top image.", Answer("number", 1, "1"),
               flags=frozenset({LABEL_WORD_PREFIX + "top image"}), lead_clause=True)
    assert "Image" not in _allowed_nouns([Part("z", [Slot(g)], [q3])])
