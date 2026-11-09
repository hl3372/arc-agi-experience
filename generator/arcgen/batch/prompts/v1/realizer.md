<!-- prompt-version: diversity-v1-realizer-1 -->
# Realization: compile a frozen concrete storyboard

You are a grid compiler, not a second task designer. Translate the complete
storyboard below into exact ARC arrays.

Frozen concrete storyboard:
{{CONCRETE_PLAN}}

You must preserve its pair count, pair purposes, core dimensions, main layout,
functional color roles, relevant quantities and relations, variation plan,
test semantics, and shortcut-defeating evidence. Do not improve, simplify,
expand, or reinterpret those decisions.

Use only the explicitly allowed presentational freedom. Validate every
coordinate, dimension, and color; independently recompute each training output
and hidden test output. If the plan cannot be executed as written, return
`refused` rather than improvising.

Public test pairs contain inputs only. Hidden answers belong only in
`authoring_metadata.expected_test_outputs`. Give exact `train_1`, `train_2`, …
and `test_1`, `test_2`, … trace identifiers. Any departure from the plan, even a
presentational one, fails fidelity: use only the listed realizer freedom, and if
the plan cannot be followed exactly, refuse and give the reason.

Classify canvas use honestly as compact and appropriate, large and
functionally used, large but partly underused, or artificially inflated.

Candidate ID is `{{CANDIDATE_ID}}`. Return exactly the schema object.

Return `contract_version: astra-authoring-v3`. Read public_contract as the
single semantic authority and private_execution as its worked execution.
Independently recompute that execution. If it conflicts with the public rule,
invariants or training evidence, refuse; do not choose a new semantic decision.
Copy public_contract.semantic_core.central_rule EXACTLY into intended_rule.
Copy public_contract.invariants EXACTLY into preserved_frozen_decisions, in order.
These copies are checked by code; a statement of consistency is not sufficient.
Some plans let one role use different colour aliases in different pairs; follow
each pair's stated aliases exactly and do not unify them. Use the named colour
references from the supplied legend in all metadata prose,
including construction_witness, every trace and both deviation ledgers. Actual
ARC task arrays and expected_test_outputs use the bound integer colors. Do not
write numeric row strings or color names in prose. Preserve ordinary numeric
counts/coordinates. Both deviation lists must agree exactly; never reconcile
conflicting declarations by erasing a defect.

Before submitting, check every serialized row against its grid's declared width
and every grid against the frozen dimensions. Construct and validate the arrays
consistently; do not fix a width error by padding or deleting unspecified cells.
For `status: refused`, return a concrete `refusal_reason` and set BOTH `task`
and `authoring_metadata` to JSON null. Put the obstruction in the reason, not in
partial metadata. A valid refusal records a plan obstruction; it is not a pass.
