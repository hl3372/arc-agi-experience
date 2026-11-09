<!-- prompt-version: weakness-v1-emit-repair-4 -->
This episode also records measurement data; it changes nothing in the task. Return weakness_outputs (null for an unrepairable response) with the fields below filled and every other field null (changed_pairs empty when it is not listed), built as carefully as the episode's own grids, in the same form as the realization's (points as id, kind, note, test, at, taught_by, and cells listing [row, col, colour] where the answer with the named reading applied at that point only differs from the true answer).
- changed_pairs: every training pair you add or change, by its number in the repaired episode, with the output {SHORTCUT} gives for its input.
- points: every declared point again, recomputed where your change alters it.
- overlay_points: the border points again ({OVERLAY_SHORTCUT}), recomputed where your change alters them.
- objects: the object ledger again, for every input grid of the repaired episode.
- assembly: the pieces record again.
- killer: the killer record again, with finished_sites one list per input grid of the repaired episode.
- alternatives: every structure record again, same ids, kinds and line: train with one entry per training pair of the repaired episode, in order, giving the record's grid for every pair you add or change and null for the others; test giving the record's grid for every test input.
- units: the unit labels for every input grid you add or change, named by its place in the repaired episode.
- surfaces: the surface record for every input grid you add or change, named by its place in the repaired episode.
