# Domain A design note: experimental discovery in synthetic worlds

> **Release note.** These are the design notes as written during development, kept for the record. Paths in them
> use the original layout: `construct/` and `discover/` are now `research_envs/construct/` and
> `research_envs/discover/`; the round scripts (`scripts/round1.py`, `round2.py`, `demo_a.py`, `demo_b.py`,
> `preview.py`, `run_with_resume.sh`) are now presets of the `research-envs` command; the other scripts are in
> `scripts/analysis/`. The `runs/...` folders they cite are the original run records, which are not part of this
> release (the 20 demo tasks are in `examples/`). "The brief" is the internal project brief these designs
> answered; it is not included. "This machine" is the development machine (Linux, 20 cores). The program
> sandbox planned as bubblewrap was built with Landlock instead (`research_envs/construct/sandbox.py`).

Status: Phase 0, revision 2 (2026-11-05), after one reviewer pass; the changes are listed in section 11. Domain A starts only if B
works. Nothing here touches the ARC generator, its task bank, production settings or domain B's code paths (A gets its own
package, `discover/`, and copies or imports B's tested pieces where they fit).

## 1. What the demo should show, and what it must not claim

**Claim to earn:** the ARC generation *method* transfers to a second domain with nothing in common with ARC: no
grids, no visual induction, no ARC conventions. Generated worlds should be:
- sound: the world, the interface and the evaluator are correct;
- fair: what the solver must predict is decided by experiments it can afford;
- consequential: the targeted demand separates good experimenters from naive ones;
- new: a new mechanism, not a renamed one.

The tasks are tilted toward finding the law, not just control: the main artifact is a **predictive model** of the
world, scored on withheld interventions. This makes "experimental discovery" an honest name: the solver must form
hypotheses, design experiments that separate them, and generalize to conditions it never tested.

**Not claimed:**
- scientific discovery about the real world. The laws are planted by us, the worlds are small and noise-free, and
  nobody learns anything new about reality;
- transfer to laboratory science;
- fairness against every conceivable hypothesis. Fairness is computed relative to a stated rival set (section 5),
  which is an Occam-style assumption and is labelled as one;
- training value.

**Name:** "experimental discovery in synthetic worlds", never "scientific discovery" unqualified.

**Precedents to cite and differ from:** DiscoveryWorld (interactive discovery tasks), Alchemy (hidden-chemistry
meta-RL), BoxingGym (experimental design). The contribution is property-directed *generation and auditing* of
worlds, not simulated discovery itself.

## 2. How each step of the method transfers

| ARC step | Domain A |
|---|---|
| Pin properties, never rules | Requests are pinned to a system type and a demand (section 4). Matched controls drop the demand. No example worlds are shown to the generator. |
| Novel by construction | Ideas are compared with their nearest archive entries by **behaviour**: observation traces on a canonical probe battery, compared up to relabelling of actions and sensors, plus the mechanism's primitive structure (section 6). The archive is seeded with textbook mechanisms (thermostat, counter with overflow, two-tank transfer, combination lock, delay line, flip-flop, vending machine). |
| Plan names the shortcuts to defeat | Shortcuts are **executable baseline experimenters and model classes**: no experiments, random experiments, a systematic sweep, a lookup table, memoryless and order-blind models. Each demand pairs a shortcut that must fail with an experimenter that must succeed (section 4). |
| Build by execution | The world is written in a bounded world language run by our own interpreter (section 3). The planner supplies the world, the rival laws it is designed around, the query set, and a reference experiment plan. Code checks all of them. |
| Blind check for a unique answer | Inverted. The question is no longer "do the examples fix the answer?" but "**can affordable experiments separate every rival law that would change an answer?**" (section 5). Plus intent vs spec: a blind model restates the public description, and the restatement must not contradict the world. |
| Adjudicate and repair | Pilot outcomes are classed mechanically: correct, wrong but the query was unfair (a defect), wrong and fair, or an interface exploit. Correct answers from unexpected experiments are never repaired away. |
| Meta-review | The same loop: between rounds, measured funnel, shortcut-failure rates, pilots, one named generator fix per failure pattern. |

## 3. The world model

**A world** is a small deterministic machine with integer state:
- **State:** 2–5 integer variables, each with a declared range and boundary rule (clamp or wrap). Some are
  *settable* at reset (the experimenter's controllable initial conditions); the rest start at fixed values the
  solver is not told.
- **Actions:** 2–6 named actions, some with a small integer parameter (e.g. `turn(k)` for k in 0..5).
- **Effects:** each action is a list of guarded simultaneous assignments, `if <guard>: var := <expr>, ...`, evaluated
  in order with the first matching guard winning, or all applied, as declared. An optional `tick` applies after every
  action, which expresses drift, delays and coupling.
- **Sensors:** 1–4 named readings, each an integer or boolean expression of the state. A sensor can be direct (`x`),
  coarse (`x // 3`), relational (`x > y`), or modular (`(x + 2*y) % 5`). Hidden variables have no direct sensor.

Expressions use B's validated language (`construct/lang.py`: integer arithmetic, comparisons, if-else, min, max,
abs; no attribute access, imports or lambdas), with names = state variables + action parameters. One interpreter
serves every world; the mechanism primitives below are patterns written in it, not special cases in code:
- **thresholds and hysteresis:** a guard on a level, with a mode variable remembering which side it was last on;
- **counters and memory:** a hidden count of presses that changes what an action does;
- **delays:** a hidden pipeline variable that releases an effect k ticks later;
- **conserved transfers:** an action that moves amounts between variables, with the total conserved;
- **mode switches:** a hidden mode that selects between effect lists;
- **saturation and cycles:** clamp and wrap boundaries;
- **coupling:** a tick that moves one variable in response to another.

**Names are neutral and fictional** ("the brass lever", "chamber 2", "the amber lamp"). Real-world labels such as
"acid" or "thermostat" would let a model use prior knowledge (a shortcut) or mislead it (unfair). Misleading priors
may become a property axis later. Some tasks include an **irrelevant control**: part of discovery is learning what
does not matter.

**Size bounds** keep every check exact or exhaustively sampled: at most about 10^4 states, sequences of at most 8
actions per experiment, experiment budgets of 20–60 experiments.

**Public package:** the description of the device (what each action and sensor is, in neutral words; settable
variables and their ranges; that there is hidden internal state, without its form), the experiment interface and
budget, and the artifact format.

**Private package:** the world, the rival laws, the query set with true answers, the reference experiment plan,
baseline results and audit records.

## 4. The interface, the artifact, and the property axes

**Experiment interface.** One experiment is `{"start": {settable values}, "actions": [[name, param], ...]}` with at
most L actions. It returns the sensor readings after reset and after every action. Reset restores the stated initial
condition in the *same* world; it never samples a new law. The budget counts experiments. Experiments are submitted
in batches (up to 8 per turn), so a pilot can adapt between turns.

**Artifact.** After the experiment phase closes, the solver receives the **query set**: 20–40 withheld
interventions, each a start and an action sequence, for which it must predict the final sensor readings. It also
writes its law in words (recorded, not scored). Optional per task: a **procedure**, a small observation-conditioned
policy in a fixed format, scored on fresh starts against a target condition.

Queries mix three regimes, scored separately:
- **interpolation:** starts and lengths like those the solver could test;
- **composition:** combinations of actions in orders the solver may not have tried;
- **extrapolation:** longer sequences than the experiment limit L, or values in a range no experiment can directly
  start from. These test the law, not a lookup table; their fairness is checked like every other query.

**Scoring:** exact-match accuracy per regime; validity separate from accuracy; experiment cost recorded. Answers
that are behaviourally right count, whatever law the solver states.

**Property axes:**

| Axis | Values (round 1) |
|---|---|
| **About (system type)** | device (levers, dials, lamps, gauges); substances and vessels (amounts, mixing, transfers) |
| **Demand** | A1–A4 below |
| **Format** | experiment budget; number of controls; whether an irrelevant control is present |
| **Control** | same system type, no demand instruction |

Every demand audit is **two-sided** at the same experiment budget:

| Demand | The shortcut must fail | The demand-suited experimenter must succeed |
|---|---|---|
| **A1 critical experiment** | The default behaviour fits at least two laws that disagree on queries. Passive and random experimenters (best consistent rival afterwards) fall short on query accuracy. | The hypothesis-eliminating experimenter (section 5) reaches every query within budget, using a separating experiment the planner names. |
| **A2 hidden variable** | The best *memoryless* model (next readings as a function of current readings and action) fits the experiments worse and misses queries. | The eliminator reaches every query. The planner names the probing experiment that reveals the hidden variable. |
| **A3 order and history** | The best *order-blind* model (outcome as a function of the multiset of actions) misses queries. | The eliminator reaches every query. |
| **A4 quantitative law** | A lookup-table and nearest-neighbour model, trained on everything the budget allows, misses the extrapolation queries. | The eliminator reaches every query, extrapolation included. |

**Controls** have no named shortcut. Both arms are measured against the full baseline suite, and the comparison is
the gap between the best baseline and the eliminator.

## 5. Fairness and verification

1. **Evaluator trust:** unit tests for the interpreter; brute-force cross-checks of traces; mutation tests (each
   world's probe experiments must distinguish at least 90% of its single-change mutants).
2. **The rival set R.** The rival laws are:
   - every single-change mutant of the true world: a constant ±1, a flipped comparison, a swapped variable, a
     dropped guard, a dropped assignment, a changed operator, a changed boundary rule;
   - the planner's named rivals: the misleading laws the task is designed around;
   - laws in each shortcut's model class (memoryless, order-blind) fitted to the data, where relevant.

   R is the hypothesis space against which fairness is computed. It is an assumption: laws far from the true one
   are not considered. That is stated in every dossier.
3. **Query determinacy.** For each query, every rival that agrees with the true world on *all affordable
   experiments* must give the same answer. Affordable means within the interface's length limit. For tiny worlds,
   all affordable experiments are enumerated; otherwise, an adequate sample plus the eliminator's own plans. An
   undetermined query is dropped before the task is published, and the dropped count is reported.
4. **Accessibility.** The hypothesis-eliminating experimenter knows R but not which law is true. Each step it picks
   the experiment that splits the consistent rivals most evenly, from a candidate pool, and stops at the budget. It
   must reach full query accuracy within the stated budget. This shows that useful information is obtainable within
   the interface and budget, which is the brief's second fact. A controller that knows the law would prove only the
   first fact.
5. **Consequence:** the demand's shortcut fails at the same budget (section 4), and still fails at twice the budget.
   With twice the budget the passive or random baseline must still fall short for A1. Otherwise the task is
   calibration.
6. **Intent vs spec:** a blind model reads the public description and lists what it says about each action and
   sensor. Code checks every stated fact against the world, for example "the brass lever never changes the gauge"
   against traces. A contradiction is a statement defect.
7. **Robustness of fairness:** at the end of the round, R is widened to two-change mutants. The fraction of
   accepted queries that become undetermined is reported. That number measures how much the fairness claim leans on
   the Occam assumption.
8. **Cross-world transfer is out of scope** for the demo. Held-out mechanism families are named in section 10.

## 6. Novelty and families

- **Behavioural fingerprint:** the sensor traces of the world on a fixed canonical battery of experiments (all
  single actions, all pairs, a few long repeats), canonicalized over permutations of action and sensor names and
  over affine relabellings of sensor values. Small action counts make the permutation search cheap.
- **Structure signature:** the multiset of primitives the world uses and how they are wired (which actions write
  which variables, which variables feed which sensors).
- **Family vs instance:** a new fingerprint and structure is a new family. The same structure with different
  constants is an instance.
- **Archive:** about 15 textbook mechanisms written in the world language. A world whose fingerprint matches one is
  **calibration**.
- **Canaries:** disguised textbook mechanisms (renamed, re-scaled, with an irrelevant control added) run through the
  novelty check. The bar is 8 of 10 caught.

## 7. Pipeline and acceptance

**Stages** (one process per task, as in B):
1. request (system type, demand or control, format values);
2. ideas: 3 candidates, written against the archive;
3. mechanical prechecks: the world compiles; its sizes and bounds; fingerprint and structure against the archive;
4. select;
5. plan: the world, the public description, the named rivals, the query set, the reference experiment plan;
6. build: the rival set; query determinacy (dropping undetermined queries); the eliminator; the baselines and the
   two-sided audit; probes and mutants; novelty;
7. blind intent check;
8. pilots: an interactive experimenter (turns of experiment batches, then queries) and a no-experiment pilot
   given only the public description. The latter measures how much prior knowledge or naming leaks;
9. mechanical adjudication and class.

**Classes:**
- **accepted:** sound, fair, the demand's two-sided audit passes, and new;
- **calibration:** sound and fair, but easy (a shortcut reaches the queries) or textbook;
- **frontier:** accepted, but no pilot reached the accuracy bar;
- **rejected:** unsound or unfair, with reasons.

## 8. Rounds, bars and falsifiers

**Round 1:** 16 pinned requests (4 demands × 2 system types × 2) plus 16 controls, plus about 10 canaries.
**Review:** one named generator fix per failure pattern. **Round 2:** the same size with the fixes.

**Bars:**
- **Soundness:** every accepted world passes every mechanical check; zero undetermined queries in published tasks.
- **Consequence:** the two-sided audit passes on at least 9 of 16 pinned tasks; pinned beat controls on the
  shortcut-failure rate by at least 25 points.
- **Novelty:** at least 8 of 10 canaries caught; at least 80% of accepted worlds are new families.
- **Accessibility:** the eliminator reaches every published query within budget, by construction.

**Falsifiers (pre-registered):**
- **Trivial:** the no-experiment pilot reaches at least 70% query accuracy on more than half the accepted tasks. That
  means names or priors leak the law.
- **Guessing game:** interactive pilots succeed only on interpolation queries and fail extrapolation on more than
  half the accepted tasks while the eliminator succeeds. That means the worlds reward memorization, not law-finding.
- **Narrow fairness:** widening R to two-change mutants makes more than 20% of accepted queries undetermined.
- **Cosmetic:** the fingerprint check collapses more than half the families.

## 9. Records, layout, cost

- **Records:** one dossier per task, as in the brief's minimal record and B's layout.
- **Layout:** `discover/` with `world.py` (language and interpreter), `harness.py` (experiments, budgets,
  queries), `rivals.py` (mutants and the rival set), `fairness.py` (determinacy and the eliminator), `baselines.py`,
  `novelty.py`, `prompts.py`, `pipeline.py`. Reused from B: `construct/lang.py`, `construct/llm.py`, the
  process-per-task round runner, file-lock slots, resume and recheck scripts.
- **Cost:** pilots are interactive, about 6 calls each at effort high, so roughly 10–12 calls per task in total.
  Compute is light: worlds are tiny, and determinacy is exhaustive or sampled.

## 10. Open questions

- Whether to give the solver the world language as a modelling formalism. It would make hypothesis elimination
  feasible for the solver but would tell it the shape of the hypothesis space. Default: no. The solver predicts in
  plain answers and states its law in words.
- Noise and replication, and held-out world families for cross-world transfer, are deferred past the demo.
- Whether the procedure artifact (control) adds enough to justify its cost in round 1. Default: only on a few
  device tasks.

## 11. Changes after review (revision 2)

These supersede the sections above where they conflict.

1. **The fairness claim is local identifiability, not discoverability.**
   - Interpolation and composition queries are themselves affordable experiments, so determinacy can only bite on
     extrapolation.
   - The rival set R is centred on the true law, so an eliminator that knows R knows how many hidden variables there
     are and the mechanism's form.
   - The R-aware eliminator is therefore renamed the **local identifiability check**. It certifies that, among laws
     near the truth, the published queries are fixed by affordable experiments. Nothing more.
   - **Discoverability** gets its own evidence from **public-information baselines** that never see the true law:
     - active automaton learning (an L*-style learner over the input alphabet, one experiment per reset, prefixes
       free because readings come after every action, equivalence approximated by budgeted random testing);
     - an enumerative law search over a public grammar (B's expression language to a fixed depth, per sensor and per
       action).
   - Evidence from these and from blind pilots is labelled **empirical**.
   - Dossier wording: "Each published query is fixed by every law in R that matches the experiments an R-aware
     eliminator can run within budget. R is centred on the true law, so this certifies local identifiability, not
     discoverability."
2. **Needle check, against guessing games.**
   - For each named rival, measure the fraction of experiments from the public random sampler and from the
     systematic sweep that separate it from the truth.
   - If that fraction is under about 1/B and no public-information experimenter finds the separation, the task is
     **needle** and is rejected. The passive data must contain an observable clue.
   - R gains **structural rivals**: one primitive swapped for another with the same interface (counter↔delay,
     clamp↔wrap↔none, hysteresis↔plain threshold, mode↔no mode, tick removed or sign-flipped).
   - R also gains **inserted-guard rivals** beyond the tested range. They make out-of-range queries undetermined,
     which states the Occam assumption honestly.
   - The two-change widening runs **per task at build time**.
3. **Audits are split into presence and consequence.**
   - **Presence:** the restricted model class fails. The classes are strengthened:
     - A2: an observation-history model with window w ≤ L, not memoryless;
     - A3: a bigram model over actions, not order-blind;
     - A4: linear or periodic extrapolation plus small symbolic regression, not nearest-neighbour.
   - **Consequence:** the same public learner, given the planner's demand-suited experiments versus the sweep or
     random experiments at **equal budget**, succeeds versus fails.
   - **Thresholds and persistence:** absolute thresholds (at least 5 critical queries missed and at least 25 points
     below), persisting at 4× budget.
   - **The systematic sweep is in every audit, controls included.** It is defined as: one factor at a time, all
     action pairs, single actions repeated to L, and space-filling starts. It is paired with every model class, and
     the best result is reported.
   - **Coverage class:** if a full sweep of the settable start space fits in 10× budget, the task is classed
     **coverage**, not accepted.
   - "Best consistent rival" means the average over consistent rivals, never a tie-break that could leak the truth.
   - The eliminator's candidate pool is never seeded from the planner's reference plan.
4. **Queries are sampled by code, not chosen by the planner.**
   - Queries are drawn per regime from declared distributions.
   - **Critical queries** are those on which named or structural rivals disagree with the truth. They must make up
     at least a third of the set.
   - Scores are reported per query, per **facet** (the queries resolving the same rivals) and per sensor.
   - **Extrapolation** is a separate, parsimony-conditional diagnostic, never in the accept bars. A query is
     admitted only if every guard branch on its trajectory has been exercised, both firing and not firing, by some
     affordable experiment.
5. **Pilots and difficulty.**
   - Pilots run at budgets 0, B/4 and B, and the dose-response curve is reported.
   - Scores are chance-corrected against a public persistence or majority baseline.
   - Pilots are judged on critical queries.
   - **New falsifier (easy):** interactive pilots reach at least 90% on every regime using at most half the budget,
     on more than 60% of accepted tasks. This is B's lesson.
   - **Difficulty levers:**
     - a settable start space at least 20× the budget;
     - a minimal observable automaton larger than what the automaton learner can learn within budget;
     - coarse or relational sensors.
   - Accepted tasks get two pilot draws.
   - **Isolation:** checked for B. Codex's sandbox fails closed on this machine, so pilots have no shell and cannot
     read files. Pilots are therefore pure reasoning plus the experiment interface. The brief's "one model, fixed
     interface" holds.
6. **Novelty compares minimized behaviour.**
   - Drop irrelevant inputs and constant sensors.
   - Minimize the observable Mealy machine.
   - Compare up to permutations of inputs and bijective relabelling of output symbols. This covers affine recodings
     and more.
   - Detect product decompositions and compare their components.
   - Half the canaries are written by a blind model prompted to evade detection.
7. **The brief's requirements.**
   - **Deviation, stated:** the brief makes valid-policy rate, target achievement and worst case primary, with
     prediction as a diagnostic. This design makes prediction primary, to keep the discovery claim honest.
   - A small **procedure subset** (about 4 tasks, device-like, exhaustively scored on fresh starts) covers the
     brief's "indirect measurement" direction, which has no demand of its own here.
   - Evidence of more than one valid strategy is recorded: distinct successful experiment plans.
   - Whole families are reserved before any iteration against solvers.
   - The eliminator's logs are never used as teacher trajectories.
8. **Scope, names, compute and naturalness.**
   - The device-vs-substances axis is cut. It is mostly naming. Round 1 is 4 demands × 4 pinned plus 16 controls.
   - **Names are abstract** (actions `A`, `B`, `C(k)`; readouts `r1`, `r2`). Even "dial" or "lamp" carries priors
     (wrap, boolean). The no-experiment pilot measures any remaining leak.
   - Worlds and rivals are **compiled to explicit transition tables** over the whole state grid with numpy, so
     determinacy, equivalence and minimization are exact table operations (product-machine search).
   - Pilot cost: budget 40 at 8 experiments per turn is 5 turns plus a query call. With three budget levels, that
     is about 10 calls per pilot draw.
   - **Naturalness guards, ported from B's lesson:**
     - a short law;
     - few constants;
     - threshold guards rather than equalities;
     - a one-sentence mechanism story;
     - every component changes some query answer.
   - Mutation adequacy is computed after grouping mutants by exact equivalence, so mutants that hit unreachable
     guards or untouched clamps don't count against it.
   - Shortcut-class laws are fitted on a fixed sweep, so R doesn't depend on a pilot's data.
   - The blind intent check targets interface mechanics (tick timing, readings per action), with every stated fact
     quoting the description.
   - Side channels are tested: errors on out-of-range starts, and on hidden-variable overflow.

## 12. Phase 1 build and gate (2026-11-06, overnight)

**Built** (`discover/`, 19 tests plus a process-mode dry run of a whole round):
- `world.py`: the world language, compiled to explicit transition tables;
- `rivals.py`: single-change and structural rivals, with same-law rivals merged under all their labels;
- `fairness.py`: exact shortest separating experiments (product-machine search), the public sampler, needle
  fractions, the R-aware eliminator, query sampling;
- `baselines.py`: the systematic sweep, random experiments, and learners. The learners are persistence, windowed
  models, passive state merging, a symbolic law search, the law search with one hypothesized hidden counter
  (cost-based choice between explanations, plus a committee), and an adaptive query-by-committee experimenter;
- `audit.py`: the build and the two-sided audit;
- `novelty.py`: a minimized-machine fingerprint invariant to renaming, irrelevant controls, constant readouts and
  readout recoding, plus a structure signature for families. The archive holds 10 textbook mechanisms;
- `prompts.py`, `harness.py`, `pipeline.py`: generation, the pilot interface (abstract public names generated by code
  from the world), and the round runner.

**What the gate taught (fixes already in):**
- **Tabular learners are useless as shortcuts.** On a fully visible 64-state world they reached 11% of critical
  queries, because they never generalize "this action adds 1" across values, so every world would look hard. The
  public learners are now symbolic law searches, which reach 100% on fully visible worlds from 20 experiments. The
  memoryless law search is the presence check for hidden variables.
- **Memoryless laws can overfit with threshold splits.** The choice between explanations is by cost (minimum
  description), not by whether the memoryless one fits.
- **Random queries rarely test what a task is about.** Critical queries are now those where a named or structural
  rival disagrees, drawn from a pool that includes structured probes (a run of one action, then another),
  stratified by rival.
- **Accessibility needs an adaptive public experimenter.** Query-by-committee over the law search's near-cheapest
  explanations. The needle check counts a separation the adaptive experimenter finds as a followed clue, not a
  needle.

**Hand-built results:**
- A fully visible world is calibration (every learner reaches 100%).
- Two hidden-latch worlds are calibration or near misses: random experiments reach 83–100% of critical queries at
  B or 4B. For example, latch5: adaptive 1.0, sweep and random 0.83, a gap of 17 points against a bar of 25, plus
  one needle rival.

So no hand-built world yet passes the full two-sided audit. The gate shows the audit refuses worlds that random
experimentation can crack. Generated worlds must hide their law better while keeping a clue: the generator's job, and
the round's measurement.

## 13. Revision 3: the goal and metrics, restructured (2026-11-06, before round 1 runs)

Domain B showed the old goal was wrong. It required each task to prove its property against our hand-built search
baselines, and pinned tasks to beat controls. Those are proxies. They are not what makes an environment worth having,
and they are not what made the ARC generator work. In ARC the goal is a task a careful solver can crack from the
evidence, that frontier models fail, and that is new. Properties are levers for producing such tasks, not acceptance
criteria. Domain A adopts that goal. These definitions are fixed now, before any round-1 result is seen.

**Gates (all must hold):**
- **sound:** the world compiles; the evaluator's tests pass;
- **fair:** no needle rivals, and every published query is decided by the local identifiability check within budget;
- **new:** no behavioural match with the archive;
- **not trivial:** a full sweep of starts and sequences does not fit in 10× budget.

**Primary outcome:** frontier difficulty, the full-budget interactive pilot's accuracy on critical queries.
- **frontier:** fair, and the pilot misses critical queries (accuracy under 0.9). This is the target.
- **solved:** fair, and the pilot reaches them. Still a sound environment, usable for training, and reported.
- **calibration:** textbook (an archive match) or coverage-trivial.
- **rejected:** fails a gate, with reasons.

**Diagnostics (reported, never gates):**
- the demand audits and shortcut baselines (sweep, random, symbolic learners);
- the budget curve (no experiments, a quarter, the full budget);
- `public_discoverable`: the planner's plan or the adaptive public experimenter reaches the critical queries. This is
  the strong evidence that a frontier world is solvable, not just fair.

**Falsifiers:**
- **frontier-solved:** pilots solve more than 70% of fair tasks, so the generator does not produce frontier-hard
  worlds.
- **leak:** the no-experiment pilot reaches at least 70% of critical queries on more than 30% of fair tasks, so
  descriptions or names give the law away.
- **unfair-looking frontier:** fewer than a third of frontier tasks are `public_discoverable`. Their difficulty may
  then be unfairness that the local checks miss, and they need a human look.
- **cosmetic:** fingerprints collapse more than half the families.

**Levers:** the four demands plus controls stay as request arms. The comparison of interest is the frontier yield per
arm, not each task proving its demand.
