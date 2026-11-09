<!-- prompt-version: astra-contract-authoring-v3-r2 -->
# Experimental grounding requirement

The source idea was explored under a hypothesis that some necessary fact must
come from the training episode jointly. Retain that hypothesis only if the
concrete episode makes the fact unavailable from every individual training
pair and recoverable from order-independent joint evidence in the grids.

If retained, state concisely:

- the jointly inferred fact;
- why no one training pair reveals the complete mechanism;
- which relationship among the examples supplies the missing evidence;
- why listing order carries no meaning; and
- why the jointly inferred fact determines one test answer.

Drop the hypothesis and return an ordinary strong V4.2 plan when it does not
survive concrete design. Abort only when neither version remains worth
realizing. Do not add machinery merely to preserve the hypothesis.

The nested concrete plan uses astra-authoring-v3 with the public/private
scopes and named color references specified in the base planner. Wrapper
property/derivation audits are private planning metadata; any general semantic
commitment needed for verification must also be in the authoritative public
contract. Do not place worked test results in that public contract. The existing
at-most-ten strongest-shortcuts schema limit also applies to this composition.
