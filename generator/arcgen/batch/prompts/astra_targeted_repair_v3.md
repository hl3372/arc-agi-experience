<!-- prompt-version: astra-contract-authoring-v3-r8-repair-identity -->
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

For a mechanical repair, correct arrays while retaining the plan. For an
ambiguity repair, revise and realize at most one added or replaced training
pair while preserving the rule, test input, hidden output, composition, and
test purpose. Recompute outputs for the corrected training pair.

For an ambiguity repair, copy the original test input arrays and their plan
constructions verbatim, preserving dimensions, cells, order and test count.
Preserve the original hidden test answers exactly. Do not resize, simplify or
substitute a test to make a demonstration repair work. If resolving the issue
requires changing a test or its answer, return unrepairable. Before returning,
compare each repaired test with the frozen original, not just the rule prose.
You may update training-support and training-comparison descriptions to reflect
the repaired demonstration; those updates do not permit changing a test.

Do not repair weak concepts, artificial canvas inflation, fundamental
ambiguity, or major plan failure. Return `unrepairable` when a local correction
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
Return contract_version astra-authoring-v3. Repaired plans and realizations use
this same version. Preserve the public/private boundary and named references.
Copy the authoritative public central_rule and invariants exactly into realizer
metadata, as in normal realization. The top-level realization.deviations and
its authoring_metadata.plan_fidelity.deviations must be exactly equal in content
and order. A disagreement is rejected under repair-agreement-v2, with no runtime
normalization. Only report a deviation that actually remains; do not fabricate
an empty ledger for a remaining violation. Repair is followed by normal checks
and independent final-version technical/blind reverification.
