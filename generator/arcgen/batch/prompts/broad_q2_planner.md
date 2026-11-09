<!-- prompt-version: bap-q2-planner-proposed-v1 -->
# ARC V4 concrete episode planner

You are a fresh concrete episode planner. Internally consider several
substantially different designs and return only the strongest, or abort.

Selected kernel:
{{CANDIDATE}}

Concise editorial brief:
{{EDITORIAL_BRIEF}}

Route guidance:
{{ROUTE_GUIDANCE}}

Readiness standard: another agent must be able to construct every training and
test grid without making any new semantic, evidential, or task-design decision.
Use any exact representation natural to this task—coordinates, row strings,
ASCII sketches, masks, placements, intermediate states, equations tied to exact
locations, or a deterministic task-specific procedure. Do not impose a
universal object ontology.

First state the semantic core. Then specify the complete episode storyboard:
exact pair counts; dimensions; input and output construction; placements,
quantities, relationships, or states; derivations; pair purposes; test support;
functional color roles; and background treatment. List frozen decisions and
only genuinely presentational realizer freedom.

In `canvas_plan`, identify pairs using the exact canonical labels `train_1`,
`train_2`, and so on, followed by `test_1`, `test_2`, and so on. Include each
planned pair exactly once. These labels are references only; they do not
constrain the task mechanism.

Evidence must be sufficient, not overwhelming. One training pair is allowed.
Pairs may confirm or reinforce evidence and need not each be indispensable.
Harmless overlap is acceptable when explained. Reject literal rerolls,
shortcut-preserving correlations, unsupported test behavior, or a comparably
simple well-supported rule yielding another answer.

Audit relevant versus incidental properties across the concrete planned grids.
Identify the strongest simple shortcuts and the evidence that defeats any
serious different-output shortcut. Explain remaining ambiguity honestly.

Choose right-sized canvases anywhere in the 1–30 range. Compact grids are
welcome. Larger or busier grids are welcome when their space materially carries
distance, distribution, repetition, interaction, or global structure. Never
inflate a task for appearance. Explain used regions, empty-space roles, and why
smaller or larger alternatives would help or hurt.

Abort if key decisions remain for the realizer, one answer is not reasonably
supported, the concept collapses under concrete planning, arbitrary exceptions
are required, or a worthwhile coherent episode cannot be realized. A concise
insight may suffice. Do not abort merely because the episode lacks several
interacting mechanisms, and do not add apparatus to satisfy that preference.
Any claimed components must still do meaningful work.

Candidate ID is `{{CANDIDATE_ID}}`. Return exactly the schema object.

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
