<!-- prompt-version: arcgen-v4.0.0-control -->
# ARC V4 fully blind solve and ambiguity probe

You are a fresh blind solver. You receive only training inputs/outputs and test
inputs. Infer the task without access to author intent.

Task ID: {{TASK_ID}}

Solver-facing ARC episode:
{{PUBLIC_TASK}}

Return a primary inferred rule and exact predicted output for every test input.
Then report only the strongest plausible alternatives. Distinguish whether each
fits all training pairs, is comparably simple, is genuinely well supported, and
changes a test output.

Do not manufacture alternatives merely because finite examples permit
infinitely many descriptions. A difficult task or your own failure is not
itself a task defect. The important ambiguity is a comparably simple,
well-supported rule fitting all training evidence and predicting a different
test output.

Set `serious_different_output_alternative` true exactly when at least one
reported alternative meets that standard. State whether unsupported guessing
is required and assess fairness concisely.

Solver ID is `{{SOLVER_ID}}`. Return exactly the schema object.

