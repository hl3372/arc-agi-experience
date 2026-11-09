# Examples: the 20 demo tasks

The ten domain A worlds and ten domain B problems picked for the paper, as self-contained task files (format in
`research_envs/tasks.py`). Each file has three parts:
- `public`: what a solver sees (A: the device description, interface and the 30 queries; B: the statement and
  the authoritative formal spec);
- `private`: the hidden law and the answers (A: the world, the planner's named rival laws, its experiment plan,
  the queries with true answers; B: the planner's witness program and object, the reference value, the best
  known object, sibling size and search hint);
- `results`: what the pipeline recorded in the original run (class, pilot scores, audit summary).

Model call records, pilot transcripts and review texts are not included.

Check them offline (no model calls):

    research-envs validate examples/discover examples/construct

Validation recompiles each world or spec and re-derives what it can: the public interface and every query answer
(A), the witness, the witness program run in the sandbox, and the exhaustive sibling instance (B), plus the archive
novelty checks. `--full` also reruns the whole build and audit (slow).

## How they were picked

A model review panel (Opus 5.5, the stricter reviewer, which built simulators and rival-law checks; and
GPT-6.1-sol via Codex, with the rubric in `docs/REVIEW_RUBRIC.md`) reviewed the tasks that passed the pipeline's
mechanical gates. GPT accepted every task; the Opus verdict decided the ranking. **No human reviewed these tasks.**
Scores are coherence / interest / demonstration value (A) and coherence / interest (B), each 1-5.

## Domain A: discovery worlds

Final query sets: eight worlds from the run `demo_a2` (discriminating queries, answer cap 0.4) and two from
`demo_a3` (the same rules with answer cap 0.25). "Pilot" is GPT-6.1-sol (effort high) at a quarter of the
budget (10 experiments) and at the full budget (40), as accuracy on the 30 queries, all of them critical.
Pipeline class: `solved` when the full-budget pilot reached 0.9, `frontier` when it did not.

| # | id | title | lever | pipeline class | pilot at 10 | pilot at 40 | Opus | Opus scores | GPT |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `da_time_1` | Suspended Sediment | time | solved | 1.00 | 1.00 | accept | 5/4/5 | accept |
| 2 | `da_alias_2` | The Stretching Detour | alias | frontier | 0.33 | 0.63 | accept | 4/4/4 | accept |
| 3 | `da_time_3` | The Differential Drain Gate | time | solved | 0.00 | 1.00 | accept | 4/4/4 | accept |
| 4 | `da_alias_1` | The Lift-Off Stamp | alias | solved | 1.00 | 1.00 | accept | 4/4/4 | accept |
| 5 | `da_cycle_3` | The Hunting Crown | cycle | solved | 1.00 | 1.00 | accept | 4/3/4 | accept |
| 6 | `da_cycle_2` | The Granular Transfer Plate | cycle | solved | 0.93 | 1.00 | accept | 4/4/4 | accept |
| 7 | `a2_time_2` | Pair-Reversing Carousel | time | frontier | 0.23 | 0.30 | accept | 4/4/4 | accept |
| 8 | `da_cycle_1` | The Pair-Fed Take-Up Drum | cycle | solved | 1.00 | 1.00 | accept | 3/3/3 | accept |
| 9 | `da_coupled_2` | The Transmission Plating Cycle | coupled | solved | 0.27 | 1.00 | accept | 3/3/3 | accept |
| 10 | `da_coupled_1` | Cross-Cast Snap Stops | coupled | solved | 0.70 | 1.00 | revise | 4/3/2 | accept |

## Domain B: construction problems

Seven problems from the run `demo_b`, two from `round2_b` and one from `round1_b` (the last three before the
quality-bar prompts). "Reference" is the planner's own witness; "best known" is the best valid object found by
the reference improver (a code agent with more time than the pilots), where it ran. Pipeline classes:
`accepted`; `calibration` (sound, but easy for search: the baseline search battery came within the gap of the
reference at ten times the budget, or for a pinned task its named shortcut did); `closed` (the best known value
meets the planner's proven bound). The novelty column is the Opus reviewer's judgement against
the literature.

| # | id | title | object | demand | kind | pipeline class | reference | best known | Opus | Opus scores | novelty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `db_d4_g1` | Can equally situated vertices be excluded from every optimal pair of facilities? | graph | D4 | counterexample | accepted | - | - | accept | 5/3 | new |
| 2 | `db_ctl_s1` | Uniformly rare points with few local union failures | set system | control | optimize | accepted | 5 | 5 | accept | 4/3 | variant |
| 3 | `db_ctl_g2` | Dense graphs resisting local complementation and edge pivots | graph | control | optimize | calibration | 211 | 212 | accept | 4/3 | variant |
| 4 | `db_d2_s1` | Uniform abundance of association reversals under conditioning | set system | D2 | optimize | accepted | 20 | 22 | accept | 4/3 | new |
| 5 | `db_d4_g2` | A best meeting point missed by every pairwise shortest route | graph | D4 | counterexample | accepted | - | - | accept | 4/3 | new |
| 6 | `db_d2_s3` | Many smallest identifying samples through every point | set system | D2 | optimize | calibration | 495 | 610 | accept | 4/3 | new |
| 7 | `db_ctl_s2` | Many frozen two-colorings at equal incidence budget | set system | control | optimize | closed | 16 | 16 | accept | 4/3 | new |
| 8 | `r2_ctl_s2` | Disjoint blocks that cannot share an exact cover | set system | control | optimize | accepted | 72 | 72 | accept | 4/3 | not asked |
| 9 | `r1_ctl_g6` | Many unique three-way geodesic junctions | graph | control | optimize | accepted | 3230 | - | accept | 4/3 | not asked |
| 10 | `r2_d2_s2` | Reusing regions covered exactly once | set system | D2 | optimize | accepted | 444 | 531 | accept | 3/3 | not asked |

Construction problems turned out easy for frontier models: on all ten, a GPT-6.1-sol pilot (explicit object or
sandboxed program) matched or beat the planner's reference, and on four it beat it (each file's
`results.pilot_outcomes`). They demonstrate the method; they are not a hard benchmark.

