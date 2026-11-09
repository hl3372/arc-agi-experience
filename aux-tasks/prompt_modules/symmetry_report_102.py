"""Prompt template for task 102 (which symmetries does the grid have).

One wording for every example: the five labels, their coordinate rules, the
0-indexed origin, the shape rule that settles a diagonal on a rectangle, the
all-or-nothing rule that a single differing cell makes a symmetry absent, and
the output format.  No colour is ever named -- the question is about the
picture's geometry, not its palette.
"""
from __future__ import annotations

from prompts import register_prompt

INDEX_NOTE = "Rows and columns are 0-indexed from the top-left corner."


@register_prompt("102")
def _p102(m):
    labels = ", ".join(m.get("allowed_symmetries", []))
    return (
        "For each allowed symmetry below, decide whether it is present in the grid. "
        f"{INDEX_NOTE} horizontal_reflection means row r, column c matches "
        "row height - 1 - r, column c; vertical_reflection means row r, column c "
        "matches row r, column width - 1 - c; main_diagonal_reflection means a square "
        "grid where row r, column c matches row c, column r; anti_diagonal_reflection means "
        "a square grid where row r, column c matches row width - 1 - c, column height - 1 - r; "
        "rotational_180 means row r, column c matches row height - 1 - r, column width - 1 - c. "
        "A diagonal reflection is answered no whenever the grid is not square, because the "
        "reflection swaps the height and the width, so it does not map the grid onto itself. "
        "A symmetry is present only when every cell matches the cell it is paired with: a "
        "single cell that differs makes the answer no. "
        f"Use this exact order: {labels}. "
        "Return exactly one line for each label in that order, formatted as label: yes or label: no. "
        "Do not add extra lines or extra text."
    )
