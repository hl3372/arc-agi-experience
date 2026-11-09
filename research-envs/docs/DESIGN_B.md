# Domain B design note: mathematical construction and counterexample discovery

> **Release note.** These are the design notes as written during development, kept for the record. Paths in them
> use the original layout: `construct/` and `discover/` are now `research_envs/construct/` and
> `research_envs/discover/`; the round scripts (`scripts/round1.py`, `round2.py`, `demo_a.py`, `demo_b.py`,
> `preview.py`, `run_with_resume.sh`) are now presets of the `research-envs` command; the other scripts are in
> `scripts/analysis/`. The `runs/...` folders they cite are the original run records, which are not part of this
> release (the 20 demo tasks are in `examples/`). "The brief" is the internal project brief these designs
> answered; it is not included. "This machine" is the development machine (Linux, 20 cores). The program
> sandbox planned as bubblewrap was built with Landlock instead (`research_envs/construct/sandbox.py`).

Status: Phase 0, revision 2 (2026-11-05), after one reviewer pass; the changes are listed in section 11. Scope: a demo
of 30–40 accepted tasks over two rounds. Domain A (discovery worlds) follows only if B works. Nothing here touches
the ARC generator, its task bank or production settings.

## 1. What the demo should show, and what it must not claim

**Claim to earn:** the ARC generation *method* transfers to a domain with nothing in common with ARC. That method is:
- property-pinned requests;
- novelty checked against an archive;
- plans that name the shortcuts the task must defeat;
- building by execution;
- blind checks, adjudication and repair;
- a meta-review loop with matched controls.

The method should yield finite construction problems that are sound, varied, open to solutions better than our own,
and consequential: the targeted demand separates good from naive solutions, and a demand-suited method succeeds where
the shortcut fails.

**Not claimed:**
- new mathematics;
- proofs about infinite families;
- an uncertified optimum;
- training value.

Round 1 is a screen, not a statistically powered test.

## 2. How each step of the method transfers

| ARC step | Domain B |
|---|---|
| Pin properties, never rules | Requests are pinned to an object type and a demand (section 4). Matched controls drop the demand. |
| Novel by construction | Ideas are compared with their nearest archive entries by **behaviour**: a fingerprint on a fixed probe battery, plus small-size optimum values. Syntax-tree shape is secondary. The archive is seeded with textbook problems and a zoo of standard objects (section 6). |
| Plan names the shortcuts to defeat | The shortcuts are **executable baselines**. Each demand pairs a shortcut that must fail with a demand-suited method that must succeed at the same budget (section 4). |
| Build by execution | The spec is written in a bounded expression language run by our own interpreter. The reference witness comes from a construction program written by the planner, run in a sandbox, then evaluated like any candidate. The planner states whether the witness is a closed-form construction or found by search. |
| Blind check for a unique answer | Replaced. Many valid solutions are the point. The blind checks now cover: **intent vs spec** (a blind model writes its own formal spec from the statement alone, and the two are compared on an adequate probe set and exhaustively on the small sibling); **openness** (structurally distinct alternatives); and **headroom** (a verified bound where possible). |
| Adjudicate and repair | Outcomes are classed **mechanically** wherever possible: valid, loophole, defect, too easy, broken. A solution can be a loophole only if the blind spec rejects it, or it falls in a degenerate class the planner declared in advance. Solutions that beat the reference are never repaired away. |
| Meta-review | The same loop. Audits are mechanical baseline separations, and no raters are used for interest. |

## 3. The problem model

**Object types:**
- **Graph:** simple undirected, vertices `0..n-1`.
- **Set system:** ground set `0..m-1` and a family of distinct subsets.

No third type in round 1. If the vocabulary blocks ideas, that is recorded.

**Problem kinds:**
- **optimize:** maximize or minimize an objective subject to constraints.
- **satisfy:** find any object meeting tight constraints.
- **counterexample:** given assumptions P and a plausible claim Q, find an object with P and not Q.

**Parametric specs.** Constraints and objectives may refer to the size parameters, so every task names a target size
and a **small sibling size** at which the same spec can be enumerated exhaustively (section 5).

**Spec language:** a whitelisted Python expression subset, parsed with `ast`, validated node by node, then compiled.
- There is no attribute access, no imports, no lambdas and no dunder names.
- It runs only with our functions as globals, under a per-evaluation wall-clock limit.
- Iteration helpers (`range`, `combinations`, `product`) are Python generators with a step budget, so data-dependent
  blow-ups are interruptible and bounded.
- **Domain functions:**
  - graphs: `n`, `V`, `E`, `adj`, `deg`, `nbrs`, `dist`, `connected`, `components`, `triangles`, `girth`, `diameter`,
    `clique_number`, `independence_number`, `is_bipartite`, edge and vertex connectivity;
  - set systems: `m`, `ground`, `F`, `size`.

Each domain function is cross-checked against brute force (`tests/test_evaluator.py`).

**Candidate format:** explicit JSON objects. Malformed input is invalid, never repaired.

**Public package:**
- the statement and the authoritative formal spec;
- the candidate format and objective direction;
- a budgeted check tool.

**Private package:**
- the witness and its program;
- the sibling instance and its exhaustive results;
- the baseline battery results;
- probes, the bound, and audit records.

## 4. Property axes, with two-sided audits

| Axis | Values (round 1) |
|---|---|
| **About** | graph; set system |
| **Demand** | D1–D4 below |
| **Control** | Same object type, with no demand instruction |

Kind (optimize, satisfy, counterexample) follows from the demand and is not crossed. Every audit is **two-sided** and
run at the same evaluation budget:

| Demand | The shortcut must fail | The demand-suited method must succeed |
|---|---|---|
| **D1 interacting constraints** | Relax-then-repair: optimize a subset of the constraints, then greedily repair the rest. It falls short of the witness, or ends invalid. | Joint search over all constraints reaches the witness. Also: no constraint is implied by the others (exhaustive check at sibling size). |
| **D2 exploitable structure** | Unrestricted annealing stays below the witness. | Search restricted to symmetric objects (circulants, orbit unions of base blocks under a cyclic group) reaches it. The witness must be a stated closed-form construction; a witness found by search is no evidence for D2. |
| **D3 escaping local optima** | Hill climbing finds no 1- or 2-move improvement below the witness. | Large-neighbourhood or kick search reaches the witness. |
| **D4 counterexample** | Annealing on a margin of violating the claim fails. Plausibility: the claim holds exhaustively at sizes up to the sibling, and on at least 200 sampled P-objects from a stated sampler. | The witness violates the claim. |

**Controls** have no named shortcut. Both arms are therefore measured against the **best result of the full baseline
battery**, and the comparison is the gap between that result and the witness.

## 5. Verification

1. **Evaluator trust:**
   - unit and brute-force tests (80 passing);
   - spec mutation tests: automatic AST mutations (a flipped comparison, ±1 on a constant, a dropped constraint, a
     dropped conjunct). Each task's probe set must tell at least 90% of its spec mutants apart from the true spec.
2. **Small sibling instance:**
   - exhaustive enumeration of the spec at the sibling size;
   - the certified optimum there;
   - a redundancy check: every constraint must be violated, alone, by some object;
   - an exhaustive intent-vs-spec comparison (point 6).
3. **Witness:** the planner's construction program runs in bubblewrap with no network, read-only mounts, the
   archive and private files hidden, a memory cap and a timeout. Its output is evaluated like any candidate.
4. **Baseline battery:** run under fixed evaluation budgets set before the round, with parallel restarts:
   - random;
   - greedy add and delete;
   - hill climbing;
   - kick and large-neighbourhood search;
   - annealing with **graded violations** (the count of failing instances inside each `all(...)`, or the distance
     from a comparison's bound);
   - symmetric-restricted search;
   - relax-then-repair;
   - margin annealing for counterexamples.

   A gap counts only if it persists when the budget rises tenfold. The battery is first calibrated on seed problems
   with published best values (e.g. Fano and covering designs, cap sets in F_3^n for small n, Sidon sets, and the
   Paley-graph colouring that shows R(4,4) > 17).
5. **Code-agent baseline:** one model with sandboxed Python (numpy, itertools, OR-Tools) and a fixed wall-clock
   tries to reach the witness. It's reported separately from the explicit-object pilots, and answers "is this just a
   search problem?".
6. **Intent vs spec:** a blind model writes its own formal spec from the statement alone. It is compared with ours
   on validity *and* objective values, over the probe set and exhaustively at the sibling size. The probe set
   contains:
   - the witness and baseline outputs;
   - their single-move mutants;
   - objects on constraint boundaries.

   Disagreement is reported **before** repair. After a repair, a fresh blind call re-checks.
7. **Headroom and openness:**
   - Every optimize task reports a verified upper bound where one exists: CP-SAT on linear-encodable specs, or the
     certified optimum at sibling size scaled where valid. A reference equal to its bound marks the task *closed*.
   - Openness needs a structurally distinct alternative: another construction, or one beyond a stated edit distance
     from the witness. Non-isomorphic one-edge variants don't count.
   - Isomorphism uses a WL hash only to bucket objects; equality is decided by an exact check.
8. **Scoring:**
   - validity is kept separate from objective value;
   - raw values are kept;
   - gaps are absolute and also relative, reported against the certified optimum or the reference;
   - nothing is clamped.

## 6. Novelty and families

- **Behavioural fingerprint:** validity and objective values over a fixed probe battery for the object type, plus
  certified small-size optimum values.
- **Family vs instance:** a new fingerprint is a new family. The same fingerprint up to size parameters is an
  instance.
- **Archive:** about 25 textbook problems, plus a **zoo of standard objects**: Paley, Kneser and Petersen graphs,
  small strongly regular and Cayley graphs, projective planes, Steiner systems. A witness isomorphic to a zoo object,
  or a task whose small-size optimum sequence matches a textbook sequence, is **calibration**, not novelty.
- **Canaries:** round 1 plants about 10 disguised textbook problems among the ideas, to measure whether the novelty
  check catches them.

## 7. Pipeline and acceptance

**Stages:**
1. request
2. ideas (with the nearest archive entries by fingerprint)
3. select
4. plan: a parametric spec, target and sibling sizes, the shortcut and demand-suited method, the witness program,
   declared degenerate classes
5. build, sibling, battery and checks
6. pilots: explicit-object solver (2 attempts, up to 3 check rounds each) and code agent
7. mechanical adjudication, plus repair with a fresh re-check; earlier pilot solutions are re-scored after any repair
8. record

**Acceptance classes:**
- **accepted:** sound, the demand's two-sided audit passes, and new.
- **calibration:** sound, but easy or textbook.
- **closed:** the reference meets the verified bound.
- **frontier:** sound, and no solver found a valid object.
- **rejected.**

Every class is reported, with per-task gaps.

## 8. Rounds, control and review

- **Round 1 (a screen):** 16 property-pinned requests (4 demands × 2 object types × 2) plus 16 controls, plus about
  10 planted canaries.
- **Review:** one named generator fix per failure, judged against the bars.
- **Round 2:** the same size with the fixes in.

**Bars**, with absolute thresholds, so a 5% gap on an objective of 12 can't count:
- **Soundness:** 100% of accepted tasks pass every mechanical check.
- **Consequence:**
  - the two-sided audit passes on at least 9 of 16 pinned tasks;
  - the battery gap to the witness is at least 2 units and at least 5%, and persists at 10× budget;
  - pinned tasks beat controls on the persistent-gap rate by at least 25 points.

  With 16 per arm this is still underpowered; it is reported as a screen, with every task's gap shown.
- **Novelty:** at least 8 of 10 canaries caught, and at least 80% of accepted tasks are new families.
- **Openness:** at least half of the accepted optimize tasks have a structurally distinct alternative, or are open
  against a verified bound.

**Falsifiers:**
- **Trivial:** the code agent reaches the witness on more than 60% of accepted tasks.
- **Cosmetic:** the fingerprint check collapses more than half the families.
- **Ambiguous:** blind-spec disagreement before repair on more than 30% of tasks.

## 9. Records, layout, cost

- **Records:** one dossier per task, as in the brief's minimal record.
- **Layout:**
  - `construct/`: `lang.py`, `domain.py`, `evaluate.py`, `search.py`, then `sandbox.py`, `sibling.py`, `probes.py`,
    `novelty.py`, `pipeline/`;
  - `tests/`;
  - `runs/` (not in git);
  - `docs/`.
- **Cost:**
  - **model calls:** about 300k tokens per accepted task including both pilot types, so about 15–20M for B;
  - **compute:** the battery at 10^5–10^6 evaluations per task, on 20 cores.

## 10. Open questions

- Whether the 10× persistence check fits the compute budget at medium sizes. If not, sizes shrink, rather than gaps
  being accepted on weak baselines.
- Whether a verified bound can be had for non-linear specs beyond sibling sizes. Otherwise, report "no bound".

## 11. Changes after review (revision 2)

- Two-sided demand audits; the automorphism-group test for D2 is dropped (empty and complete objects maximize it).
- Graded violations, fixed budgets, the 10× persistence check, and calibration on published best values.
- A code-agent baseline, and closed-form vs search witness recorded.
- Behavioural novelty fingerprints, the object zoo and planted canaries.
- Small sibling instances; probe adequacy via spec mutants; disagreement reported before repair; a fresh re-check
  after repair.
- Mechanical loophole rulings; earlier solutions re-scored after a repair.
- Structural openness, verified bounds, and the "closed" class.
- A bubblewrap sandbox, and interruptible, budgeted iteration helpers.
- Statistics: a 16/16 screen, absolute thresholds, controls measured against the full battery.
- Cut: the format cross, the given-data toggle, a third object type, and the interest raters.
- Not adopted: nauty. WL hashing only buckets objects; isomorphism is decided by an exact check.

## 12. Changes after the preview (revision 3, 2026-11-05)

The six-task preview (`runs/preview_b1`) exposed battery weaknesses, a few crash paths, and a naturalness problem.

- **Battery:**
  - structure-preserving moves: double-edge swaps for graphs, which keep every degree, and one-element
    replacement for set systems, which keeps set sizes;
  - sampled first-improvement local search in place of full swap enumeration;
  - each method's seeds form a start portfolio: even seeds start dense-random, odd seeds start empty or sparse;
  - universes up to 500k elements; set-system starts are a few times m sets, never a density of a huge universe;
  - families beyond the 512-set answer format are invalid inside search;
  - per-run wall-clock caps of 240 s at B and 1200 s at 10B; runs that hit a cap are recorded (`time_capped`).
- **Crash paths:**
  - a claim or mutant that raises makes the candidate invalid, or the mutant killed;
  - object-dependent size limits (pool, range, product) make the object invalid instead of raising a spec error;
  - the round runs each task end to end in its own thread, catches and records each failure, and never writes
    `summary.json` while any task has failed;
  - `scripts/run_with_resume.sh` reruns the round until it completes. Every model call resumes from its record,
    and builds are cached by a hash of spec, hint, sibling and witness program.
- **Graded violation:** `all(...)` now grades as the sum of each instance's own grade, so `all(deg(v) == d ...)` is
  the total distance of the degrees from d. Before, it was the number of failing vertices: no single edge improved
  it, and local search never found a regular graph. That made preview p5 look like a control only symmetric search
  could solve.
- **Threads and timeouts:** the evaluator's timeouts use SIGALRM, which works only in a main thread, and every
  preview pilot failed on it. Each task now runs in its own spawned process.
- **Compute slots:** compute jobs and sandboxed programs hold machine-wide file-lock slots, two compute and three
  program. Overlapping rounds therefore never oversubscribe the cores, which would quietly weaken a battery that has
  wall-clock caps. Tests use their own slots.
- **Probes:** structure-preserving chains of the witness (degree-preserving swaps, element replacements) keep
  regularity-type assumptions, so mutants of claims and objectives are compared on objects that satisfy them.
  Preview p4 adequacy rose from 0.56 to 0.94.
- **Records:** a resumed call keeps the prompt it was made with, and a different prompt on resume is saved
  separately. Each dossier records a hash of the pipeline code.
- **Hints:** when a set-system task gives no set sizes and the full universe is too large, the battery uses the set
  sizes in the witness (recorded as `hint_inferred_from_witness`).
- **Audit mode** (reported next to the class):
  - `demand_verified`;
  - `hard_unattributed`: every battery method falls short of the reference, so the task is hard but not shown to be
    hard in the named way;
  - `easy_for_shortcut`.
- **Naturalness** (soft, measured and reported, never a gate):
  - the prompts ask for one clean objective and at most about three constraints, each needed by the question, and
    forbid constraints or tuned constants added only to defeat search. D1 is reframed as natural requirements that
    conflict.
  - The selector prefers the more natural of two candidates.
  - A blind rater call scores each statement from 1 to 5, gives a one-line "about", and lists the parts that look
    arbitrary.
  - Mechanical signals are recorded: constraint count, constants other than 0, 1 and 2, spec length, statement
    length.
- **Canaries:** `scripts/canaries.py` runs twelve disguised textbook specs (renamed, restated, with a redundant
  constraint added) through the fingerprint check. All twelve are recognised. Complements (for example
  min edges with no independent 3-set versus Mantel) are not recognised yet.
- **Cost:** idea calls at effort high instead of xhigh. The preview's xhigh idea calls took about 1.5 h.
- **Pilot isolation (checked 2026-11-05 23:5x):** pilots are Codex calls in read-only mode. On this machine AppArmor
  blocks the user namespaces Codex's bubblewrap sandbox needs, so every shell command a pilot tries fails with
  `bwrap: loopback: Failed RTM_NEWADDR`. The only exception is `pwd`, which Codex answers itself. Pilots therefore
  cannot read files, including the private dossiers under `runs/`, or run code. Their answers come from the model
  alone; code-pilot programs run in our Landlock sandbox. This isolation fails closed only because bubblewrap is
  blocked. If it were enabled, read-only mode would allow reading the whole disk, and pilots would need an explicit
  filesystem restriction.

### Fixes during round 1 (2026-11-06, applied to every task by `scripts/recheck.py` at the end of the round)

- **Oversized set universes:** a hint such as set sizes [3, 12] at m = 24 (2.7M candidate sets) crashed the battery.
  Sizes are now restricted to those the witness uses. If that is still above 500k, the battery runs on a fixed
  random sample of 200k sets plus the witness's sets, recorded as `universe_sampled`.
- **Targeted probes for probe adequacy:** tight constraints (block sizes, regularity) left many mutants alive,
  because random and one-move probes rarely violate one tight constraint while keeping the others. For each
  surviving mutant, a short search now looks for an object on which the mutant and the true spec disagree. Survivors
  that agree with the true spec on every object at the sibling size are counted as equivalent and leave the
  denominator. On r1_d2_s1, adequacy went from 0.47 to 1.0: 31 mutants killed, 3 equivalent.
- **Literal blind specs:** the statement gives the target size, so a blind spec may use literals ("degree 10") where
  ours is parametric ("(n-1)//3"). The sibling-size comparison is now made only when the blind spec is parametric;
  otherwise the target-size probe comparison decides. On r1_d2_g1, 465 sibling "disagreements" were this artefact.

## 13. Round 1 meta-review (2026-11-06, 05:15)

**Run:** 32 requests (16 pinned, 16 controls), 22:49–04:55 after the restart, then a repair pass for 5 build errors.
The numbers below are as judged at run time. The run-time code predates the mid-round fixes, because all 32 task
processes started at 22:49 and imported the code then. `scripts/recheck.py` re-judges every task with the fixed rules.

**Against the pre-registered bars** (27 final at run time):
- **Two-sided audit:** passes on 5 of 15 pinned tasks (bar: 9 of 16). **Missed.**
- **Persistent gap to the best battery result at 10×B:** pinned 53%, controls 55% (bar: pinned ahead by 25 points).
  **Missed.** Controls are just as search-hard: the generator makes search-hard problems by default, so this bar
  measures general hardness, not the demand.
- **Like-for-like demand audits** (the same audit computed on controls, a measurement fix):
  - D2, structure: pinned 2/4 vs controls 0/11;
  - D3, local optima: 0/4 vs 0/11;
  - D4, counterexamples: 3/4 verified, with no comparable controls.

  So pinning works for D2 and D4 and not for D3. D3 tasks are hard for kick search too.
- **Falsifier "trivial": tripped.** The code agent reaches the witness on 6 of 6 accepted tasks. Some pilot reaches
  or beats the reference on 26 of 26 tasks with a witness, and beats it on 9. References are what the planner
  easily reaches.
- **Falsifier "ambiguous": fine,** 2 of 26.
- **Novelty:** no textbook or zoo matches. But ideas converge across parallel requests: 4 groups share a
  behavioural fingerprint (for example, three graph controls on cyclic majorities in a distance metric).
- **Naturalness:** median 4, but every task got 4, so the rater carries no information as used.

**Defects found in our checks, not in the tasks:** they are fixed and applied by the recheck.
- Probe adequacy on tight constraints was too weak (targeted probes).
- Literal blind specs were compared at the sibling size.
- Oversized set universes crashed the battery (sampling, plus orbit handling for sampled universes).

**Round-2 fixes** (one per failure pattern, in `scripts/round2.py`):
1. **Convergent ideas → a round registry.** A task takes its highest-ranked candidate whose fingerprint matches no
   family already taken in the round, with one idea retry. Round 1's titles join the avoid list.
2. **Trivial references → a reference improver.** A code agent with 600 s, more than the pilots' 240 s, pushes the
   reference. Pilots are judged against both the planner's witness and the best known value, so headroom is
   measured, not assumed.
3. **Measurement:** demand audits on controls are reported like-for-like.

**Not changed:** the D3 audit and the naturalness rater. These are next candidates, depending on round 2.
