Solve this ARC grid-transformation task. Infer the transformation from all training input/output examples and apply it to each test input. Give one answer. Only if, after working through the examples, something about the test output genuinely cannot be determined or pinned down from them (for example, two different rules both reproduce every training pair but give different test outputs, or some detail of the output is not fixed by any example), then, and only then, give a second answer that covers the other possibility. Do not add a second answer just in case. Your final answer must contain only the answer(s), in exactly this form:

Attempt 1:
<the test output grids, in test order>

and, only when a second answer is warranted, followed by:

Attempt 2:
<the test output grids, in test order>

Write each grid row as color digits with no spaces, one row per line. Within an attempt, separate multiple test outputs with one blank line. Do not include explanations, other headings, or code fences in the final answer.

Each grid is shown with its size and zero-based row labels (r0 is the top row; the label and the bar are not part of the grid); each row is written as consecutive color digits without spaces, and c0 is the leftmost column.

After each grid, a run listing restates the same grid row by row as color@columns (zero-based, inclusive). It is a reading aid containing no extra information.

Coordinate and grid-writing convention for the entire solve, including your
working/reasoning, not just the final answer:

- Use zero-based grid coordinates (row, column) from the start. The top-left
  cell is (0,0); rows increase downward and columns increase rightward. An H by
  W grid has rows 0 through H-1 and columns 0 through W-1.
- Use this convention consistently for every training input/output, every
  test grid, intermediate grids, and local/cropped grids. A crop's own top-left
  cell is (0,0); distinguish its local coordinates from full-grid coordinates.
  Do not switch to one-based grid labels or postpone conversion until the end.
- Counts, dimensions, colors, lengths and example numbers are not coordinates.
  Keep them distinct. If you explore a feature using ordinal position, define
  it explicitly from the zero-based coordinate, such as row+1 or column+1,
  rather than silently changing the coordinate basis.
- Whenever you write a grid row in your working or answer, write its color
  digits consecutively without spaces between cells, e.g. 123 rather than
  1 2 3. Coordinate labels may appear outside the row in your working; the
  final answer must remain grid-only as instructed above.

Otherwise solve normally: investigate the public examples, work through
computations, and check your reasoning. This instruction supplies notation,
not a transformation rule or a hint about the solution.

The row labels (r0, r1, ...) and column spans (c0, c1, ...) shown with the grids already follow this zero-based convention; refer to cells with them.
