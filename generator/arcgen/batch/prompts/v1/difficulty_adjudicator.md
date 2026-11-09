<!-- prompt-version: diversity-v1-adjudicator-2 -->
Review correctness and solver-visible support for this ordinary ARC episode.
Task ID: {{TASK_ID}}
Episode hash: {{EPISODE_HASH}}
Return policy: grounded-difficulty-retention-v1

Author's proposed rule and contract (claims to verify, not evidence of discoverability):
{{AUTHOR_SPECIFICATION}}

Complete episode, with proposed test answers revealed for checking:
{{COMPLETE_TASK}}

Previously saved public-only blind attempts:
{{BLIND_RESULTS}}

Required concern inventory:
{{CONCERNS}}

Current-version technical evidence:
{{TECHNICAL_EVIDENCE}}

Discovery and verification are different. A task may be fair and correct even
when every blind attempt misses. Two misses or two ambiguity flags establish
neither invalidity nor frontier difficulty. A correct blind answer also cannot
erase a concrete rival. Do not judge elegance, familiarity, expected success
or novelty. Do not simplify a task to
make the blind model succeed.

Public SUPPORT means the actual train input/output grids and visible test input.
The author's PUBLIC_CONTRACT, inferred rule, and revealed test answer are NOT
solver-visible teaching evidence. Independently check the technical evidence;
a favorable verdict is not a substitute for a derivation. Explain positive
support from specific demonstrations. No decisive convention may be justified
only by its appearance in the author's contract or proposed answer.

Address every supplied concern_id exactly once, including each PRIMARY rule
and listed alternative. Trace each concrete competing procedure against all
training pairs. For a rejected rival, cite a demonstration it fails or concrete
extra assumptions and grid evidence favoring the intended interpretation. When
both fit, compare support and simplicity explicitly. Judge simplicity this way:
step-by-step procedures, counting, comparison and simple arithmetic or logic are
simple; a rule that needs a non-simple formula, fitted constants, colour-number
conventions or arbitrary exceptions is contrived unless the intended rule needs
the same. A contrived hypothesis need not invalidate the task; an equally
supported different answer does.
Failure to find a rival does not itself establish positive support. For copying
or execution mistakes, identify the actual step/cells and discrepancy; correct
rule prose alone is insufficient. Use train_1, train_2, etc. for references.

Choose clear only with supported intent, no comparable different-answer rival,
and concrete resolution of every concern. Use unresolved with unknown (null)
assessments when the evidence cannot be settled. Use serious_ambiguity for a
confirmed comparably supported different answer. Use defect for a demonstrated
answer/construction/contract problem that cannot receive the allowed repair.
A confirmed rival must not be disguised as uncertainty. Keep all material
concerns; if your bounded response cannot settle them, do not clear.

Repairable requires one concrete local teaching or correctness defect, not mere
difficulty. At most one training-pair addition/replacement is allowed, preserving
rule, test input, hidden answer and challenge. When a comparable rival exists,
identify its whole family: the similar rules the current examples equally allow
(for example the same rule with a different tie-break, fixed versus normalised
weights, rank versus majority, or nearest marker versus minimal cut). Describe
that family in competing_rule, and write repair_instruction so that the one added
or replaced training pair rules out every member of the family at once, not just
one. If no single pair can rule out the family while keeping the rule, tests and
answers, the task is not salvageable: use serious_ambiguity. When the training
pairs form a chain (every training output except the last is exactly the next
training input), the repair may instead rebuild the whole chain from another
starting state, with the same number of pairs or one more, within the frozen
invariants. For a chain, never prescribe an exact grid, size or coordinates: in
repair_instruction name the deciding case, the property some state must have
so that the intended rule and every member of the family give visibly
different next states, and leave the states to the repairer. A one-demo task
must replace its demonstration. Each concern_resolutions row has a nullable repair_target. For a repairable
decision, put a repair_target directly on EVERY row resolved as comparable_rival
or defect; use null on every other row. Each target contains only competing_rule
(empty for a non-rule defect) and public_teaching_issue. Its concern ID and
training-pair references are already on that same row. There is no separate
repair_challenges list to repeat, shorten or selectively enumerate.

When a primary, alternative and support ID describe the same underlying issue,
keep every ID. Repeat the same concise public repair target on each implicated
row; shared substance does not remove an identity. Code derives the complete
recheck target list from those canonical rows. Do not put
ANY test answers/predictions, private workings, source labels, or your verdict
in this challenge: it goes to an answer-withheld technical recheck. If a clean,
concrete challenge cannot be supplied, use unresolved or serious_ambiguity.
For other decisions, every repair_target is null and repair_instruction is empty. Failure kind is execution,
discovery, mixed or unknown; it is a measured-attempt interpretation, not a
claim of intrinsic task difficulty. Return exactly the schema object for difficulty_adjudication_v2.

Before returning, check concern_resolutions against this exact ID checklist:
{{REQUIRED_CONCERN_IDS}}
Include one entry for EVERY listed ID and no other IDs. In particular, the
public-teaching-support entry is separate from the primary-rule entries. If
you cannot resolve a concern, include it as unresolved and choose a compatible
non-clear decision; never omit it to shorten the response.
{{REVIEW_POLICY}}
