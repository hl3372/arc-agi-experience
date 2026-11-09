<!-- prompt-version: astra-contract-authoring-v3-r2 -->
# Optional logical-leap planning instruction

The editorial brief contains an optional candidate-specific invitation. Treat
it as an opportunity to test, never as a requirement.

Consider whether the selected idea naturally supports a test whose answer
requires deriving a new consequence of the abstraction established by the
training episode. The consequence must not be directly instantiated in the
training examples, and straightforward mechanical replay of a demonstrated
case must be insufficient. Nevertheless, the existing evidence must determine
the consequence and one output through exactly the same governing rule,
semantic roles, and assumptions.

Use this only when the deduction is genuinely supported. Do not add an unseen
operation, role, condition, rule clause, convention, or exception. Do not make
the task larger or more complicated merely to manufacture surprise. Its
particular form is entirely open and must follow from the candidate itself.

If using the opportunity, identify concisely the principles established in
training, the new test consequence, the deduction connecting them, why that
consequence was not directly demonstrated, why direct replay is insufficient,
and why the unchanged rule supports exactly one answer. Keep this audit in the
experimental wrapper. The nested ordinary V4 plan must remain a complete,
self-contained authoring contract and must not mention this instruction, an
experiment, a nomination, or a treatment.

It is valid to decline and produce the strongest ordinary V4 episode.
Declining must not cause compensatory complexity. Abort only for the ordinary
V4 abort reasons.

Return the experimental wrapper required by the response schema. Set
`decision` to `use`, `decline`, or `abort`; place the complete ordinary V4 plan
in `concrete_plan` unless aborting; and use a non-null
`logical_leap_contract` exactly when `decision` is `use`.

The nested concrete plan uses astra-authoring-v3 with the public/private
scopes and named color references specified in the base planner. Wrapper
property/derivation audits are private planning metadata; any general semantic
commitment needed for verification must also be in the authoritative public
contract. Do not place worked test results in that public contract. The existing
at-most-ten strongest-shortcuts schema limit also applies to this composition.
