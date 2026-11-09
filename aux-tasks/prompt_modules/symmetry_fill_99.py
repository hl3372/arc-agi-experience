"""Prompt template for task 99 (complete a symmetry; formerly 99 and 100).

One wording per transform, byte-identical across every example of that
transform: only the unknown colour's label changes.  The coordinate rule is
written out for the transform the example drew, with the origin stated, because
"the horizontal axis" alone is read both ways (the task 11 lesson).  The ground
and any texture are never named -- the unknown colour is the one per-example
given, and it is part of the task statement.
"""
from __future__ import annotations

from prompts import _ci, _m, register_prompt

INDEX_NOTE = "Rows and columns are 0-indexed from the top-left corner."
UNCHANGED_NOTE = "Leave every other cell unchanged."

# transform -> (what the grid must become, the coordinate rule, the group noun,
#               the partner noun)
_TRANSFORMS = {
    "horizontal_mirror": (
        "mirror-symmetric across the horizontal axis, so that the top and the bottom swap",
        "row r, column c maps to row height - 1 - r, column c",
        "each mapped pair",
        "its partner",
    ),
    "vertical_mirror": (
        "mirror-symmetric across the vertical axis, so that the left and the right swap",
        "row r, column c maps to row r, column width - 1 - c",
        "each mapped pair",
        "its partner",
    ),
    "main_diagonal": (
        "mirror-symmetric across the main diagonal, the one running from the top-left "
        "corner to the bottom-right corner",
        "row r, column c maps to row c, column r",
        "each mapped pair",
        "its partner",
    ),
    "anti_diagonal": (
        "mirror-symmetric across the anti-diagonal, the one running from the top-right "
        "corner to the bottom-left corner",
        "row r, column c maps to row width - 1 - c, column height - 1 - r",
        "each mapped pair",
        "its partner",
    ),
    "rotation_90": (
        "rotationally symmetric by 90 degrees around the grid center",
        "row r, column c maps to row c, column height - 1 - r",
        "each mapped group of four",
        "another cell of the group",
    ),
    "rotation_180": (
        "rotationally symmetric by 180 degrees around the grid center",
        "row r, column c maps to row height - 1 - r, column width - 1 - c",
        "each mapped pair",
        "its partner",
    ),
}


@register_prompt("99")
def _p99(m):
    goal, rule, group, partner = _TRANSFORMS[_m(m, "transform")]
    return (
        f"Only cells of unknown color {_ci(m, 'unknown_color')} may change. Make the whole "
        f"grid {goal}. {INDEX_NOTE} Use this coordinate rule: {rule}. Applying the rule "
        "again and again from any cell comes back to that cell, and every cell it visits "
        f"must end up with the same color. Copy a known color across {group} when "
        f"{partner} is unknown; a cell whose every mapped partner is also unknown, "
        f"including a cell that maps to itself, keeps color {_ci(m, 'unknown_color')}. "
        f"{UNCHANGED_NOTE} Return the completed grid."
    )
