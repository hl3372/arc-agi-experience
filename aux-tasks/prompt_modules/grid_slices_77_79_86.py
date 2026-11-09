"""Prompt templates for tasks 77, 79 and 86 (read rows and columns by number).

One wording per axis (77/79) and per singular/plural shape of the two ranges
(86), byte-identical across examples of that bucket: only the numbers change.
The numbers are 0-indexed from the top-left corner and every range includes
both ends; both facts are stated because the 1-indexed and the exclusive-end
readings give different answers on every example (the generator guarantees
it).  Nothing about the picture is named.
"""
from __future__ import annotations

from prompts import register_prompt

INDEX_NOTE = "Rows and columns are 0-indexed from the top-left corner."


def _axis(m) -> str:
    axis = m["axis"]
    if axis not in ("row", "column"):
        raise ValueError(f"tasks 77/79: unknown axis {axis!r}")
    return axis


def _span_phrase(noun: str, start: int, end: int) -> str:
    """'row 3' for a single line, 'rows 3 through 8' for a range."""
    return f"{noun} {start}" if start == end else f"{noun}s {start} through {end}"


@register_prompt("77")
def _p77(m):
    if _axis(m) == "row":
        return (f"Output row {m['index']} of the grid as a grid with one row, keeping its cells "
                f"in their left-to-right order. {INDEX_NOTE}")
    return (f"Output column {m['index']} of the grid as a grid with one column, keeping its cells "
            f"in their top-to-bottom order. {INDEX_NOTE}")


@register_prompt("79")
def _p79(m):
    if _axis(m) == "row":
        return (f"Output rows {m['start']} through {m['end']} of the grid, keeping them in their "
                f"original order with every column. The range includes both ends. {INDEX_NOTE}")
    return (f"Output columns {m['start']} through {m['end']} of the grid, keeping them in their "
            f"original order with every row. The range includes both ends. {INDEX_NOTE}")


def _rows_span(m) -> str:
    return _span_phrase("row", m["row_start"], m["row_end"])


def _cols_span(m) -> str:
    return _span_phrase("column", m["col_start"], m["col_end"])


@register_prompt("86")
def _p86(m):
    verb = "intersects" if m["row_start"] == m["row_end"] else "intersect"
    single = m["row_start"] == m["row_end"] or m["col_start"] == m["col_end"]
    ends = "The range includes both ends." if single else "Ranges include both ends."
    return (f"Output the rectangle where {_rows_span(m)} {verb} {_cols_span(m)}, keeping its cells in their "
            f"original arrangement. {ends} {INDEX_NOTE}")


# Axis words the stored wordings reach as {fn:...} tokens (prompt_variants.py), so one
# wording reads correctly on both the row and the column bucket.
def _lines_word(m) -> str:
    return {"row": "rows", "column": "columns"}[_axis(m)]


def _cross_line(m) -> str:
    return {"row": "column", "column": "row"}[_axis(m)]


def _reading_order(m) -> str:
    return {"row": "left to right", "column": "top to bottom"}[_axis(m)]


def _near_edge(m) -> str:
    return {"row": "top", "column": "left"}[_axis(m)]


def _extent(m) -> str:
    return {"row": "width", "column": "height"}[_axis(m)]
