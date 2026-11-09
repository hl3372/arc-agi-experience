<!-- prompt-version: arcgen-v4.2-logical-leap-auditor-exp-1 -->
# Arm-blind logical-leap audit

Evaluate each completed ARC episode on its own. You are not told how the task
was routed, whether an intervention was attempted, or whether an author
received an optional suggestion. This audit is descriptive and non-gating.

Determine whether solving the test requires deriving a genuinely new logical
consequence of an abstraction supported by training. The required consequence
must not already be directly instantiated by a training example, and direct
mechanical replay must be insufficient. The deduction must still use the same
rule and semantics, be grounded in visible training evidence, and support one
best output.

Use exactly one status:

- `genuine_logical_leap`: training establishes premises or an abstraction from
  which the solver must derive a new test consequence through a nontrivial but
  well-supported inference;
- `structural_transfer_only`: the test changes configuration and requires
  transferring the same known procedure, but no new consequence has to be
  deduced;
- `parametric_or_executive_extension`: the test mainly changes a parameter or
  increases execution burden while following an already demonstrated
  procedure;
- `absent`: the test is essentially direct application or repetition;
- `unsupported_new_semantics`: the apparent leap depends on an unseen role,
  operation, condition, convention, exception, or otherwise unsupported
  assumption.

Do not reward mere difficulty, size, busyness, visual novelty, or extra work.
A task may be excellent while receiving any non-invalid descriptive label;
these labels do not alter ordinary task quality. Be precise about the training
premises, the required deduction, and how it determines the output.

Audit ID: {{AUDIT_ID}}

Task packages:
{{TASK_PACKAGES}}

Return only the required structured object and confirm that the audit is
non-gating.
