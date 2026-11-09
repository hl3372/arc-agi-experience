<!-- prompt-version: astra-evidence-runtime-r4-adjudicator-20260911 -->
# ARC V4 conditional ambiguity adjudication

You are invoked because the saved public blind evidence leaves an unresolved
answer mismatch or alternative. Decide whether one well-supported answer remains.

Task ID: {{TASK_ID}}

Intended rule and concrete plan:
{{AUTHOR_SPECIFICATION}}

Complete authored task:
{{COMPLETE_TASK}}

Blind results:
{{BLIND_RESULTS}}

Determine whether a comparably simple and well-supported alternative fits all
training pairs and predicts a different test output. Solver difficulty, a
contrived finite-data hypothesis, or a weakly supported alternative is not
fatal ambiguity.

Choose `clear`, `repairable`, or `serious_ambiguity`. A repair may add or replace
one discriminating training pair while preserving the rule, test input, hidden
answer, core composition, and test purpose.

Return exactly the schema object.

Inspect each solver's PRIMARY inferred rule as well as its listed alternatives,
even when its alternative flag is false. Distinguish an array copying/execution
mistake from a different procedure. Check the competing procedures against all
training pairs; do not resolve a concrete rival by a majority vote or merely by
the author's held-out answer. A clear decision must explain the relevant evidence
and why the mismatch does not leave an equally supported answer. Contrived
rules and difficulty alone do not justify rejection. For a one-demo episode,
any proposed repair must replace that demonstration rather than add another.
