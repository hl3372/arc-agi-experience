Solve this ARC grid-transformation task. Infer the transformation from all training input/output examples and apply it to each test input. Reason in your reply, then end it with a line containing exactly "Final answer:" followed by only the test output grids, in test order. Write each row as color digits with no spaces, one row per line. Separate multiple test outputs with one blank line. Do not include explanations, headings, or code fences after "Final answer:".

Training pairs and test inputs are numbered from 0. Each grid is shown with its size and zero-based row labels (r0 is the top row; the label and the bar are not part of the grid); each row is written as consecutive color digits without spaces, and c0 is the leftmost column.

After each grid, a run listing restates the same grid row by row as color@columns (zero-based, inclusive). It is a reading aid containing no extra information.

Coordinate convention for the entire solve, including your working/reasoning,
not just the final answer:

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

Otherwise solve normally: investigate the public examples, work through
computations, and check your reasoning. This instruction supplies notation,
not a transformation rule or a hint about the solution.

The row labels (r0, r1, ...) and column spans (c0, c1, ...) shown with the grids already follow this zero-based convention; refer to cells with them.
