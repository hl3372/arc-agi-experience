<!-- prompt-version: weakness-v1-selector-weakness-2 -->
Assess every candidate in this frozen window. Rank all intrinsically plausible
candidates, including qualifying alternatives in a redundant group. An external
allocator handles true redundancy and scheduling. There is no acceptance quota,
retained-rank cutoff or numerical procedure-family limit. If none qualify, rank
none; if all qualify, rank all. Available development capacity does not change
the quality threshold.

Candidates and their reference links:
{{CANDIDATES_WITH_REFERENCES}}

Shared comparison records (accepted generated tasks and permitted real tasks):
{{SHARED_REFERENCES}}

Frozen in-flight reservations:
{{IN_FLIGHT_RESERVATIONS}}

Tiered rejection/failure memory:
{{REJECTED_MEMORY_DIGEST}}

Assess conceptual potential, coherence, fair exact realizability and a credible
path to sufficiently supported answers. A rule that depends on tough or esoteric
maths (solving equations, non-obvious weighted formulas, fitted constants, a named
algorithm or specialist method) is a quality defect; step-by-step procedures and
simple counting, comparison, parity, logic or small arithmetic are not. Both concise concepts and rich functional
interactions can qualify. No automatic advantage follows from additional steps,
scopes, decoding, longer execution, larger grids or greater apparent difficulty.
A simpler useful task can be the stronger contribution. Do not penalize demanding
tasks merely for being demanding. Do not infer arm, source or invitation quotas.

Novelty standard: someone who has already mastered the existing tasks should still
need a fresh insight to solve the candidate. An insight is what a solver must figure
out: which features matter, how parts relate, what the rule is, or how to apply it in
a situation the examples did not literally show. For each close reference, ask whether
a solver who has mastered it would need anything new; if that reference's rule, applied
as its own examples teach it, already produces the candidate's answers, the candidate
needs no fresh insight. Judge the reasoning, not the story, theme or wording.
- A candidate that needs no fresh insight relative to some reference is redundant with
  it and ineligible, even when it changes the teaching, presentation, story, scale,
  number of examples, difficulty of the instances or a tie-break. Record the reference
  as redundant.
- A candidate related to a reference but still needing a fresh insight (a different
  feature decides the outcome, a new condition changes outputs, the rule runs in
  reverse, elements interact in a new way) is eligible: record that reference as a
  useful_relative and state the fresh insight in useful_contribution.
- Familiar laws are not redundancy by themselves: when the answers depend on how those
  laws meet in a way no reference's examples teach, that is a fresh insight; when a
  reference's rule run on a larger or busier scene already produces the answers, it is not.
- Many candidates in this run are built to carry a task property by design (a familiar reading that
  one example visibly refutes, a test that meets a case no example shows, small marks that set
  the rule's numbers, a worked instance inside each grid, many exact local decisions, a
  construction forced step by step, a long trace whose every step depends on the one before, an
  end state of a visible process, a relation to other things that look-alikes do not share, a
  rule that is simple only in another frame, or two readings that both fit every example while
  the test itself, or a preference careful people share, settles which holds; such a fork is the
  property, not a defect, when the settling evidence is visible): sharing the property with
  references is not redundancy;
  but a candidate whose answers a reference's rule, as its examples teach it, already produces
  stays redundant, however differently it teaches. Judge what the solver must read and decide in
  the candidate's scenes: a reference whose rule would answer differently on the candidate's
  examples or test does not make it redundant, and a familiar reading together with a visible
  condition that changes its answers is a fresh insight when no reference's rule produces those
  answers; the same rule on a bigger, busier, recoloured or re-storied scene is not. A condition
  whose cause the pictures show is not a convention the solver must be told.
- A candidate unlike every reference is eligible: record the closest ones as distinct.
Easier tasks are welcome when they need a fresh insight. Exclude concrete quality
defects and infeasibility. Require a concrete useful_contribution naming the fresh
insight for every plausible candidate; do not manufacture novelty through extra apparatus.
When a candidate's fresh insight is a convention the solver must be told rather than
something the grids show, such as a colour code, a certificate, a processing order, a
tie-break or a framing rule, treat it as a quality defect, not as novelty. In `develop`,
where the kernel leaves a choice open, ask for scenes in which the picture settles the
choice or it never arises, never for a convention to establish or teach.

For WITHIN-WINDOW redundancy, use redundant_with when one candidate needs no fresh
insight relative to another in this window. Assess and rank all otherwise
qualifying alternatives; selection of a different representative is not intrinsic
ineligibility. Use same_procedure_family_with for broader procedural relatives,
including useful distinct variants. The allocator will develop at most one
representative of true redundancy and retain every substantively useful relative.
Ranks determine processing order, not eligibility. Explain their differences;
being the third or later member of a family is not a reason for exclusion.

Identifier contract:
- redundant_with and same_procedure_family_with name ONLY other candidate IDs
  in this request. Never put a bank, real-task or memory reference in these arrays.
- bank_comparisons names ONLY a reference_id in SHARED_REFERENCES. Its relationship
  is redundant, useful_relative, distinct or uncertain, with a concrete reason.
- Describe reservation and rejection-memory concerns in reason, preserving their
  tier. Soft failure warnings are not permanent exclusions of a territory.

Retrieved proximity is a request for comparison, never a rejection threshold.
The reference set is a bounded retrieval from the complete searchable corpus;
absence of a comparator does not prove global novelty. The exact-text precheck
is not a semantic equivalence test. Compare the actual procedural details.
References whose ID begins with batch: are ideas already selected earlier in this
batch and now being developed; judge redundancy with them exactly as with existing tasks, recording
them in bank_comparisons, never in redundant_with or same_procedure_family_with.

Use plausible for a credible worthwhile fair episode; it need not be standout.
Use uncertain for a structurally valid but unresolved proposal, and ineligible
for a concrete defect, infeasibility or substantive redundancy with accepted work.
Low rank alone is not ineligibility. Rank all and only plausible candidates 1..N.
At most two ranked briefs may have an optional suggestion; zero is valid.
Development briefs must stand on their own: never include neighboring tasks,
reference IDs, comparator-specific explanations or the study's private labels.

Selector ID: {{SELECTOR_ID}}. Return exactly the schema object.
