<!-- prompt-version: arcgen-v4.1-structural-transfer-auditor-exp-1 -->
# Arm-blind structural-transfer audit

Evaluate each completed ARC episode on its own. You are not told whether an
intervention was attempted or whether an author received an optional
suggestion. This audit is descriptive and non-gating.

Determine whether the episode contains a conspicuous change in configuration,
preferably in the test, that requires transferring the same underlying rule
instead of merely repeating the demonstrations. The same semantic roles,
operations, and governing relation must remain active, and the training
episode must still support one best answer.

Use exactly one status:

- `substantial_structural_transfer`: a meaningful configuration change defeats
  a credible shallow repetition rule while the established abstraction still
  determines one answer;
- `soft_or_parametric_variation`: the later case varies size, count, position,
  orientation, or another parameter but does not substantially change the
  configuration a solver must understand;
- `cosmetic`: the apparent difference is surface presentation or clutter;
- `absent`: no meaningful departure from the demonstrated configuration;
- `unsupported`: the changed case requires a new semantic clause, role,
  operation, exception, or evidence unavailable in training.

Do not reward a task merely for being harder, larger, busier, or more
laborious. A compact configuration change may be substantial. A task may be
excellent while receiving `absent` or `soft_or_parametric_variation`; those
labels do not alter ordinary task quality.

Audit ID: {{AUDIT_ID}}

Task packages:
{{TASK_PACKAGES}}

Return only the required structured object and confirm that the audit is
non-gating.
