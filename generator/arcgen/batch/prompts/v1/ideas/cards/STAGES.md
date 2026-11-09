<!-- prompt-version: diversity-v1-idea-card-STAGES-2 -->
## Direction for this call

Seek a task in which the training pairs form a chain of states: every training output except the last is exactly the next pair's input. The human solver must learn one transition relation from the linked development, and the next state must depend on the visible state alone, never on a hidden step counter or the order in which the pairs are listed; matching grids should establish the chain. The test asks for the successor of a state where the relation has a consequence no training transition directly showed, and the examples must support that inference and determine one answer. Nothing may rest on colour numbers or an arbitrary code.
