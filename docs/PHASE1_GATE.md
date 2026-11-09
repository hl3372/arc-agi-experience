# Phase 1 gate (2026-11-05)

> **Release note.** These are the design notes as written during development, kept for the record. Paths in them
> use the original layout: `construct/` and `discover/` are now `research_envs/construct/` and
> `research_envs/discover/`; the round scripts (`scripts/round1.py`, `round2.py`, `demo_a.py`, `demo_b.py`,
> `preview.py`, `run_with_resume.sh`) are now presets of the `research-envs` command; the other scripts are in
> `scripts/analysis/`. The `runs/...` folders they cite are the original run records, which are not part of this
> release (the 20 demo tasks are in `examples/`). "The brief" is the internal project brief these designs
> answered; it is not included. "This machine" is the development machine (Linux, 20 cores). The program
> sandbox planned as bubblewrap was built with Landlock instead (`research_envs/construct/sandbox.py`).

Two hand-built calibration instances through every mechanical check (now `scripts/analysis/gate_b.py`, results in
`runs/gate/phase1_gate.json`).

| Check | Ramsey R(4,4), n=17 (graph, D2) | Projective plane of order 3 as a packing (set system, D2) |
|---|---|---|
| Witness (sandboxed program) | valid (Paley graph) | valid, 13 blocks |
| Sibling (exhaustive) | n=7: 2,097,152 objects, 923,012 valid, no redundant constraint | m=6: degenerate (optimum 1); see bug 3 |
| Battery at 20k / 200k evals | every unrestricted method fails at both budgets; symmetric search succeeds | greedy, hill climbing and kick reach 13 at 20k |
| Two-sided D2 audit | **passes, and persists at 10×** | **fails**: the shortcut already succeeds, so the task is calibration |
| Probe adequacy (spec mutants killed) | 100% | 83% → 100% after fix 2 |
| Zoo | paley_17 | projective_plane_3 |
| Textbook archive match | missed → caught after fix 1 | none (not in the archive) |

**Bugs found and fixed:**
1. Fresh fingerprints (tuples) never equalled archived ones (lists), so the planted canary was missed. Values are
   now normalised, with a regression test.
2. The private search hint (4-sets only) hid the size constraint from the probes and from the redundancy check.
   Probes now include sub-families and set-level mutations that change set sizes, with a regression test.
3. Exhaustive fingerprints are degenerate for set systems with blocks of size 4 or more. Fingerprints now add a fixed
   behavioural probe battery (small mixed families at several sizes); a match needs both parts.

**Result:** the gate passes. The audits discriminate in both directions, the sandbox holds, every check catches its
planted defect after the fixes, and 94 tests pass.
