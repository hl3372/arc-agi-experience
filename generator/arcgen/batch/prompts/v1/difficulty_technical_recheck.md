<!-- prompt-version: diversity-v1-recheck-1 -->
# ARC V4 public-contract technical verification

You are an independent technical verifier. The hidden author
output is withheld. Independently derive every test output; code will compare
your arrays with the hidden author arrays afterward.

Task ID: {{TASK_ID}}

Authoritative public commitments:
{{PUBLIC_CONTRACT}}

Solver-visible episode:
{{PUBLIC_TASK}}

Deterministic diagnostics:
{{DIAGNOSTICS}}

Trace every training mapping exactly and apply the same rule to the test. Check
mechanical correctness, one-answer support, arbitrary exceptions, test-only
semantics, and serious concrete shortcuts.

Check plan fidelity. The realizer was not authorized to change pair count,
purpose, core dimensions, layout, functional colors, evidence, or test
semantics. Distinguish harmless presentational freedom from an unapproved
design decision.

Judge evidence sufficiency, not information-theoretic minimality. A
confirmatory or partially redundant pair is acceptable. Repetition is harmful
when it adds no useful evidence, preserves a superficial correlation, or leaves
a serious different-output shortcut unresolved. More-than-90%-identical inputs,
dihedral clones, and train/test reuse require explicit semantic justification.

Assess whether the canvas is used sensibly. Compact and large grids are both
neutral. Incidental material that the public contract declares (content the
solver must recognise and leave unchanged or set aside) is legitimate;
artificial inflation is a concern.

Use `repairable` only for one local change preserving the rule and test purpose.
Use `fatal` for inconsistency, unsupported semantics, arbitrary exceptions,
major plan failure, or artificial realization that requires redesign.

Verifier ID is `{{VERIFIER_ID}}`. Return exactly the schema object.

The supplied PUBLIC_CONTRACT is authoritative: general rule, all invariants,
functional color bindings, training constructions/evidence, and test inputs.
Named @alias references resolve against its functional_color_roles; use the
CURRENT integer bindings. Some plans let one role use different colour aliases in
different pairs; check each pair against its stated aliases. A changed binding is a global palette change; a
changed role/reference or shift amount is a semantic change, not cosmetic.
Check every retained invariant and training commitment against the visible
arrays. Contradictory commitments or unsupported operations are substantive
failures. Check that the intended mapping follows this contract; code's
structural checks do not establish semantic fidelity. Derive test outputs only
from this public evidence. Private workings/output dimensions/arrays are absent;
code compares your outputs with the private final arrays after your response.
Diagnostics were recomputed from public train input/output and test input only.

Reasoned non-pass contract v2: if no complete derivation is supportable, return
`derived_test_outputs: []`, a concrete nonempty `derivation_failure_reason`, and
a non-pass status. Do not invent an answer to satisfy the schema. When you can
derive outputs, return one full grid per public test in order and an empty
failure reason. `pass` always requires all public test outputs. Diagnostic retention contract v3: report each required pair-economy diagnostic
once. `diagnostic_observation` describes whether its numerical/structural
observation is confirmed, disputed, or uncertain. This is separate from whether
the episode should retain that feature.

`retention_decision` is `acceptable` when retaining the flagged examples has an
adequate concrete semantic justification, `needs_repair` for a specific unresolved
teaching defect, or `unresolved` when you cannot decide. Explain the examples'
actual teaching roles or the actual missing contrast. A shared sparse background
can trigger high numerical similarity without making the examples harmful.
Conversely, accurately detecting a similarity flag does not justify retaining a
bad set of examples. `evidentially_necessary: false` can coexist with acceptable
retention: sufficiency does not require every pair to be indispensable.

A pass requires acceptable retention for every required diagnostic. Use a
non-pass status when a required retention concern remains. Do not answer the
old ambiguous `justified` question; that field is computed by code from your
explicit retention decision. Prose and observation accuracy cannot override an
unresolved retention decision. All other evidence and hidden-output checks stay
in effect.

DIAGNOSTICS.flags now contains one group per code. Each group's observations
retain every affected pair and its original measurements. Return exactly one
diagnostic_flag_assessments entry per assessed code, with one overall explicit
retention decision. In reviewed_observation_ids list every observation_id from
that group exactly once; discuss differences between pairs in the assessment.
Do not create a separate code assessment for each pair or silently discard a
conflicting observation. An acceptable decision may retain useful repetition;
observing overlap alone does not require repair.

Identified public teaching challenge from the ORIGINAL episode:
{{REPAIR_CHALLENGE}}

Recheck each original concern against the CURRENT repaired grids and public
contract. The challenge is an allegation to test, not an acceptance verdict.
Identify the changed demonstration and compute or explain its discriminating
consequence. Verify both that the intended rule still works and that the
identified rival no longer fits the repaired demonstrations. A fresh blind
success would not prove removal. Return every supplied concern_id exactly once
in repair_challenge_assessments. resolved must stay null/false when uncertain;
rival_still_fits must be false before confirming removal of a competing rule.
If a rival still fits or the teaching defect remains, use a non-pass status.
Never invent a derivation or change answers to agree with the author.
