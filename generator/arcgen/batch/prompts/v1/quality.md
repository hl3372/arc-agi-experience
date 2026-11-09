<!-- prompt-version: diversity-v1-embodiment-3 -->
# Embodiment guidance

Size the episode to the kernel: add nothing the kernel does not call for, and
remove nothing it does. When the kernel is a simple relation, a small clean
episode is the right answer; do not add apparatus, distractors or area to make it
look substantial.

Build the grids for a human looking at them, not for an AI model reading the
arrays. A human with only the grids, pencil and paper, in one sitting, should be
able to see what the rule depends on: shapes, positions, relations, and colours
whose roles are obvious at a glance, even when colours change between pairs.
Keep the work of producing the answer within what a careful human can do by
hand. Do not let the main difficulty be reading long runs of symbols as data,
keeping exact tallies of many items, long step-by-step bookkeeping, or
differences too small to see. When a layout, size or colour choice makes the
rule harder to see without making the idea any deeper, choose the clearer one.
Every grid is at most 30 by 30 cells; a scene that needs more must be redesigned to fit.

The grids should give the human solver a reasonable entry point and an intelligible
click without the author's prose. Every element the plan claims matters must
either determine output cells or be something the human solver has to recognise and
set aside. Incidental material (distractor objects, noise, context outside the
rule's scope, unchanged content) is legitimate when it arises from the kernel's
scene and the training pairs show it is irrelevant; declare it in
public_contract (for example as an invariant that it is copied unchanged) and
list it under incidental properties. Applying one rule to several independent
instances is legitimate. A worthwhile single relation is eligible. A latent code,
key or convention is acceptable when the grids ground it rather than assert it.
Canvas size follows what the space carries (distance, distribution, repetition,
interaction, global structure or incidental content); do not choose size for
appearance. Explain used regions and empty-space roles in the canvas plan.

Return `abort` when the grids are substantially thinner than the explanation, a
comparably simple well-supported rule gives a different answer, the plan claims
something matters that neither determines output nor must be set aside, the
episode bundles unrelated rules that could be posed as separate tasks, or one
answer needs an arbitrary convention. If a concrete comparably simple grounded account reproduces the whole episode
while bypassing a claimed essential construction, describe the task at the
level supported by the grids and do not claim that richer construction is
necessary. Preserve worthwhile representation and explanation; an alternative
algorithm alone does not erase their conceptual value.

Complete the active planning schema exactly.
