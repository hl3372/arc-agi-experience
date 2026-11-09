# Review rubric: generated research tasks

You are reviewing tasks produced by an automatic generator, as an expert stand-in for a human reviewer. Judge each
task on its merits. Be demanding: the question is whether this task deserves a place in a benchmark or a training set
of research-style tasks.

There are two domains.
- **Domain A, experimental discovery.** A solver probes a hidden deterministic device through a fixed interface
  (actions, readouts, a 40-experiment budget). It must then predict the readouts of withheld experiments. The packet
  gives you the solver's view, the hidden law, sample queries with true answers, and a strong AI solver's attempt.
- **Domain B, mathematical construction.** A solver must construct a finite graph or set system: optimizing an
  objective under constraints, or a counterexample to a claim. The packet gives you the statement, the formal spec
  and the known results.

## Judge each task on

1. **coherence** (1–5). Is the hidden law, or the problem, one coherent idea a scientist or mathematician could state
   in a sentence or two? A 1 is an arbitrary pile of special cases or a lookup table. A 5 is a clean, natural
   mechanism or question.
2. **fairness** (`fair` / `unfair` / `unsure`).
   - **A:** could a careful human scientist, with only the solver's interface and 40 experiments, pin down the
     answers to queries like the samples? A task is unfair if an answer depends on something no reasonable experiment
     could reveal, if the queries are underdetermined, or if the interface misleads.
   - **B:** is the problem precisely posed, does the statement say what the formal spec says, and is it solvable as
     stated?
   - Give the reason in one or two sentences.
3. **interest** (1–5). How worthwhile is it as a research-style task: would solving it exercise real experimental
   design and law-finding (A), or real mathematical insight (B)?
4. **known_as.** If it closely resembles a known textbook mechanism, puzzle, benchmark task or literature problem,
   name it. Otherwise give an empty string.
5. **description_ok** (true / false). A: does the public description avoid leaking or misstating the law? B: does the
   statement match the spec?
6. **solver_failure** (A only: `task` / `solver` / `n/a`). If the AI solver missed queries, was that the task's fault
   (unfair, underdetermined) or the solver's (it could have found out)? Use `n/a` if it did not miss, and for B.
7. **verdict** (`accept` / `revise` / `reject`), and **main_issue**: the single most important problem, or an empty
   string.

## Output

One JSON object per task:
{"id": "...", "coherence": 1-5, "fairness": "fair|unfair|unsure", "fairness_reason": "...", "interest": 1-5,
 "known_as": "...", "description_ok": true|false, "solver_failure": "task|solver|n/a", "verdict": "accept|revise|reject",
 "main_issue": "..."}
