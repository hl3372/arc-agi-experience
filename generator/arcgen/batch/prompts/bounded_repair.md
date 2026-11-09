<!-- prompt-version: astra-bounded-repair-v1-20260916 -->
# ARC V4 one-shot targeted repair

Make at most one local repair. Do not redesign the task.

Candidate ID: {{CANDIDATE_ID}}
Repair type: {{REPAIR_TYPE}}
Concrete instruction:
{{REPAIR_INSTRUCTION}}

Frozen plan and authored realization:
{{AUTHOR_SPECIFICATION}}

Evaluator evidence:
{{EVALUATOR_EVIDENCE}}

The outer response uses contract_version astra-bounded-repair-v1. Nested plans
and realizations retain astra-authoring-v3. Use these exact reference strings;
code copies the referenced fields from this request without regenerating them:
- Mechanical plan_update: "reuse_frozen_plan". Do not return an edited plan.
- Ambiguity plan_update.public_contract.test_inputs: "reuse_frozen_tests".
- Ambiguity plan_update.private_execution.test_outputs: "reuse_frozen_answers".
- Every repaired_realization.authoring_metadata.expected_test_outputs:
  "reuse_frozen_answers". Never emit or change hidden answer arrays.
- repaired_realization.task.test: "reuse_frozen_tests", except the explicitly
  enabled mechanical restoration described below. Never include public outputs.
- Ambiguity plan_update.public_contract.invariants and every
  repaired_realization.authoring_metadata.plan_fidelity.preserved_frozen_decisions:
  "reuse_frozen_invariants". This copies the original public invariant list exactly,
  including order. Do not paraphrase, add, remove or strengthen a frozen commitment.

For ambiguity only, training_support_updates may contain {pair_number,
training_support} rows describing how the changed training teaches each test.
Use [] when no update is needed and for every mechanical repair. Only the
training_support field can change this way; other test-plan fields stay frozen.
The episode_overview.intended_difficulty_and_inferability narrative may reflect
the repaired demonstration. Other overview fields stay fixed except train count.

Mechanical test-input restoration enabled: {{ALLOW_MECHANICAL_TEST_INPUT_RESTORATION}}
When true, a mechanical repair may supply test input arrays of the same count
and dimensions only to restore the frozen plan, with unchanged intended rule
and hidden answers. Otherwise use "reuse_frozen_tests". The supplied training
and optional restored test grids still undergo all normal fidelity checks.

For a mechanical repair, correct arrays while retaining the plan. For an
ambiguity repair, revise and realize at most one added or replaced training
pair while preserving the rule, test input, hidden output, composition, and
test purpose. Recompute outputs for the corrected training pair.
Any stricter demonstration count in the concrete instruction takes precedence:
when the count is frozen, replace one existing pair instead of adding a pair.
Check every row width before returning a grid. Runtime validation rejects ragged
arrays; it does not pad, crop or otherwise repair model-supplied grids.

For an ambiguity repair, use the frozen references above. Code copies the
original test arrays, plan constructions and hidden answers exactly, preserving
dimensions, cells, order and test count. Do not resize, simplify or
substitute a test to make a demonstration repair work. If resolving the issue
requires changing a test or its answer, return unrepairable. Before returning,
compare each repaired test with the frozen original, not just the rule prose.
You may update training-support and training-comparison descriptions to reflect
the repaired demonstration; those updates do not permit changing a test.

Do not repair weak concepts, artificial canvas inflation, fundamental
ambiguity, or major plan failure. Return `unrepairable` with a concrete reason, null plan_update and
repaired_realization, and [] training_support_updates when a local correction
is not enough. Public test pairs must never include outputs.

Return exactly the schema object.

Authoring contract `astra-authoring-v3` (required):
- `public_contract` is the sole authority for the general rule, functional color
  roles, invariants, training constructions/evidence, and test INPUT constructions.
  Move the existing semantic_core, episode_overview, training_pairs and allowed
  freedom here. Put general frozen decisions in its required `invariants` list.
- `test_inputs` contains input dimensions/construction and the variation and
  training support only. General commitments must be executable from public
  training evidence. They must not enumerate worked test outputs or intermediate
  test results, even inside a mixed list or summary.
- `private_execution` contains `test_outputs` (output dimensions, placements,
  expected construction and worked derivations), the cross-pair concrete audit,
  full canvas plan, deterministic realization procedure and axis assessment.
  These are executions of the public commitments, not a place to add decisions.
  Keep any test-specific coordinates/results out of global public narratives.
  The technical verifier will never receive private_execution.
- On abort use a concrete reason and null public/private sections.

Color references (required throughout the plan and later metadata):
Use one named binding per used color in functional_color_roles: a short stable
`role` alias such as `background`, `signal`, `barrier`, its integer `color` 0..9,
and its functional meaning. In ALL prose and serialized grid descriptions,
refer to color values only with `@background`, `@signal`, etc. For example a
row may be `@background @signal @background`; a numeric grid must be emitted
only by the realizer in actual task arrays. Do not embed color names or bare
numeric color literals in prose. Counts, coordinates, dimensions and time steps
remain ordinary numbers: "move @signal by 2 columns". All @references resolve
against this one legend, including private derivations, summaries and traces.
The palette_contract declares `mode: symbolic` and the background role name
(without @) when colors are arbitrary symbols with one background. Use
`literal_numeric` and explain when actual numeric palette indices are part of
the rule; use `unspecified` with a reason if no complete reference contract is
possible. Use null background_role if no single background. These cases retain
the original palette; do not reshape a concept to force palette eligibility.

The schema accepts at most ten strongest_simple_shortcuts. Select the strongest
up to ten; this is an existing schema limit, not a request for ten shortcuts.
Do not weaken the rule/evidence to fit a summary or add new planning objectives.
Return outer contract_version astra-bounded-repair-v1. Nested repaired plans
and realizations use astra-authoring-v3, with the reference fields above. Preserve the public/private boundary and named references.
Copy the authoritative public central_rule and invariants exactly into realizer
metadata, as in normal realization. The top-level realization.deviations and
its authoring_metadata.plan_fidelity.deviations must be exactly equal in content
and order. A disagreement is rejected under repair-agreement-v2, with no runtime
normalization. Only report a deviation that actually remains; do not fabricate
an empty ledger for a remaining violation. Repair is followed by normal checks
and independent final-version technical/blind reverification.
