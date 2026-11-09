"""Prompt template for task 101 (erase the cells that break one colour's symmetry).

One wording per transform, byte-identical across every example of that
transform: only the two colour labels change.  The coordinate rule is written
out for the transform the example drew, with the origin stated, because "the
horizontal axis" alone is read both ways (the task 11 lesson).  The ground and
any texture are never named; the target colour and the replacement colour are
per-example givens and part of the task statement.
"""
from __future__ import annotations

from prompts import _ci, _m, register_prompt

INDEX_NOTE = "Rows and columns are 0-indexed from the top-left corner."

# transform -> (what the cells of the target colour should be, the coordinate rule)
_TRANSFORMS = {
    "horizontal_mirror": (
        "mirror-symmetric across the horizontal axis, so that the top and the bottom swap",
        "row r, column c mirrors to row height - 1 - r, column c",
    ),
    "vertical_mirror": (
        "mirror-symmetric across the vertical axis, so that the left and the right swap",
        "row r, column c mirrors to row r, column width - 1 - c",
    ),
    "main_diagonal": (
        "mirror-symmetric across the main diagonal, the one running from the top-left "
        "corner to the bottom-right corner",
        "row r, column c mirrors to row c, column r",
    ),
    "anti_diagonal": (
        "mirror-symmetric across the anti-diagonal, the one running from the top-right "
        "corner to the bottom-left corner",
        "row r, column c mirrors to row width - 1 - c, column height - 1 - r",
    ),
    "rotation_180": (
        "symmetric under the half turn around the grid center, the mirroring that sends "
        "every cell through the center point to the opposite side",
        "row r, column c mirrors to row height - 1 - r, column width - 1 - c",
    ),
}


@register_prompt("101")
def _p101(m):
    goal, rule = _TRANSFORMS[_m(m, "transform")]
    target = _ci(m, "target_color")
    return (
        f"Look only at cells of color {target}. Those cells should be {goal}: {rule}. "
        f"{INDEX_NOTE} For each cell of that color, check its mirrored partner in the input. "
        f"If the mirrored partner is also color {target}, keep the cell. Otherwise recolor "
        f"that cell to color {_ci(m, 'replacement_color')}. A cell that this rule maps to "
        "itself is its own mirrored partner, so it is kept. Leave every other cell unchanged. "
        "Return the full grid."
    )
