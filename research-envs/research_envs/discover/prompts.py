import os
V2 = os.environ.get("RE_A_V2") == "1"   # round 2: world-named interface, clearer query wording
QUALITY = os.environ.get("RE_A_QUALITY") == "1"   # demo round: the reviewers' quality bar in the prompts
QUALITY_SELECT = (" Weigh elegance and interest heavily: prefer the candidate a demanding reviewer would call one clean, "
                  "satisfying mechanism with informative readouts, over a more convoluted one. silent_fraction in the checks "
                  "is the share of random experiments whose readouts end unchanged from the start; high values are bad.")
"""Prompts for domain A. No example worlds are shown to the generator (no menus): the language reference gives
syntax only, and ideas come from the demand, the archive to avoid, and the naturalness guards."""
import json

LANGUAGE = """\
## The world format

A world is a small deterministic machine with integer state, hidden from the solver. The solver runs experiments:
choose a start (values for the settable variables), then a sequence of at most max_len actions; it sees every readout
after the start and after each action. Reset always restores the same world.

A world is JSON:
  {"state": {"<var>": {"range": [lo, hi], "boundary": "clamp" | "wrap", "init": <int>, "settable": true|false}, ...},
   "actions": {"<Name>": {"rules": [{"if": "<bool expr>", "set": {"<var>": "<int expr>", ...}}, ...]},
               "<Name>": {"param": [lo, hi], "rules": [...]}},
   "tick": [{"if": ..., "set": {...}}],
   "readouts": {"<name>": "<int or bool expr>", ...},
   "max_len": 6}

Semantics: an action applies its FIRST rule whose "if" holds (a rule without "if" always holds). Its assignments are
simultaneous (all right-hand sides read the old state). Each assigned variable is then clamped, or wrapped modulo its
range size, as declared. Then the optional tick applies the same way. A parameterised action C with "param": [0, 3]
is four distinct actions C0..C3, and its rules can use the name k. Non-settable variables start at their init.
Limits: 1-6 variables, each range at most 64 values, at most 2000 states in total; 1-8 actions, at most 12 action
symbols counting parameters; 1-4 readouts; max_len 1-10.

Expressions are a small subset of Python over the state variable names (and k): integer arithmetic (+ - * // % and
** with a constant exponent), comparisons, and/or/not, if-else expressions, min, max, abs. No attribute access,
strings, lambdas or imports.

Naming: actions and readouts get abstract names (A, B, C ...; r1, r2 ...) in the public interface; your own names are
for you. Names must not carry meaning the solver could use as a prior.
"""

DEMANDS = {
    'A1': ('a critical experiment',
           'Passive or typical use of the device fits at least two laws that disagree about some outcomes; a specific, '
           'affordable experiment tells them apart. A solver must design that experiment.',
           'Your named rival laws must fit typical experiments but disagree with the true law on some outcomes. Code '
           'checks that random and systematic experiments at the budget fail to pin the outcomes, that an adaptive '
           'experimenter (and your plan) succeeds, and that the separating experiment is not a needle in a haystack: '
           'typical behaviour must contain a clue that something is left to discover.'),
    'A2': ('a hidden variable',
           'A quantity the solver cannot read directly changes what actions do; it must be inferred through '
           'experiments that probe it.',
           'Code checks that the best memoryless explanation (readouts as a function of current readouts and the '
           'action) fails on the outcomes that matter, that a learner allowed one hidden counter succeeds with your '
           'plan or an adaptive experimenter, and that random and systematic experiments at the budget do not suffice. '
           'Some readout must give a clue that hidden state exists.'),
    'A3': ('order and history',
           'Outcomes depend on the order of actions, not just which ones occur; the solver must discover the order '
           'dependence and its rule.',
           'Code checks that order-blind and short-window explanations fail on the outcomes that matter while an '
           'adaptive experimenter (and your plan) succeeds, with random and systematic experiments falling short.'),
    'A4': ('a quantitative law',
           'A readout follows a numerical relation (a rate, a conserved total, a periodic cycle, a threshold that '
           'scales) that must be found and applied to new starts and sequences.',
           'Code checks that tabular and nearest-neighbour explanations fail on outcomes at starts and sequences the '
           'solver is unlikely to have tried, while a law-finding learner with your plan succeeds.'),
}

LEVERS = {
    'time': ('time-driven dynamics',
             'The device changes on its own as steps pass (drift, decay, delayed effects, oscillation), not only in '
             'direct response to actions: part of the law is how behaviour unfolds over time. Use the tick.'),
    'cycle': ('cyclic state',
              'Some quantity, visible or hidden, is cyclic: it wraps around, rotates or has a phase, and outcomes depend '
              'on where in its cycle the device is.'),
    'alias': ('aliased sensing',
              'The readouts are coarse or relational, so many distinct internal states look the same; the solver must '
              'design experiments that tell them apart.'),
    'coupled': ('interacting subsystems',
                'The device has two or more parts with their own state, and what the solver must work out is how they '
                'interact (coupling, transfer, interference), not just how each behaves alone.'),
}

QUALITY_BAR = """
## Quality bar (this round)

Reviewers will judge each world as a benchmark task. Aim for worlds they would call elegant and interesting:
- one mechanism a scientist would find satisfying to discover and could state in a sentence once found;
- no arbitrary formulas, no piles of special-case guard rules, no formula-fitting dressed up as physics;
- readouts that are informative in most experiments (not a meter that reads zero or never changes most of the time);
- the decisive behaviour appears in a fair share of sensible experiments, but understanding it takes insight.
"""

NATURALNESS = """\
## Naturalness

The world should feel like one coherent mechanism that a scientist could describe in a sentence or two once found:
- a short law, few constants, threshold guards rather than equality tests on exact values;
- every component changes some outcome the solver will be asked about (no decoration);
- no arbitrary lookup tables, no special cases added only to defeat search;
- the difficulty comes from what is hidden and what typical behaviour suggests, not from size.
"""


def archive_lines(arch):
    return '\n'.join(f"- {a['title']}" for a in arch)


def ideas_prompt(demand, arch_lines, budget, count=3, lever=None, naming=None):
    lev = ''
    if lever:
        t, what = LEVERS[lever]
        lev = (f"## Required property: {t}\n\n{what}\n\nEvery candidate must have this property in a way that matters "
               f"for what the solver will be asked; code checks that it is really present.\n")
    if demand:
        title, what, audit = DEMANDS[demand]
        dem = (f"## The demand: {title}\n\n{what}\n\nHow it will be checked (mechanically, against your world): {audit}\n\n"
               f"A world that does not meet this check is rejected, however interesting.\n")
    else:
        dem = ("## The demand\n\nNo particular demand: design the best new world you can, one whose law a careful "
               "experimenter can find within the budget but random experimentation cannot.\n")
    return f"""You are designing new worlds for a benchmark of experimental discovery. Each world is a small fictional
device whose law is hidden; a solver runs experiments within a budget of {budget} experiments, then must predict
the readouts of withheld experiments. The worlds must be:
- new: not a textbook mechanism in disguise (renaming, rescaling, or adding an unused control does not make it new);
- fair: a careful experimenter can find out everything the predictions depend on, within the budget;
- hard for strong AI experimenters: the benchmark exists to find worlds that frontier AI systems fail to work out
  while a careful scientist with the same budget could. Difficulty must come from what is hidden and from what typical
  behaviour suggests, never from unfairness: there must be clues, and the decisive experiments must be ones a
  thoughtful experimenter would think to run;
- natural (see below).

{dem}
{lev}
{QUALITY_BAR if QUALITY else ''}
{NATURALNESS}
## Known mechanisms to avoid (do not restate, rescale or combine trivially)

{arch_lines}

{LANGUAGE}
{naming or ''}
## Your task

Propose {count} distinct candidate worlds. For each: a title (for us), a one-sentence mechanism story (for us), the
world as a JSON string, a neutral public description of the device (one short paragraph; it may say what kind of
thing the device is in abstract terms but must not reveal or hint at the law), and why the demand holds. (Rival laws
and an experiment plan come later, for the chosen world only.)
"""


IDEAS_SCHEMA = {'type': 'object', 'properties': {'candidates': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'title': {'type': 'string'}, 'story': {'type': 'string'}, 'world_json': {'type': 'string'},
    'public_description': {'type': 'string'}, 'demand_rationale': {'type': 'string'}}}}}}


def select_prompt(demand, listing, lever=None):
    dem = f"the demand '{DEMANDS[demand][0]}'" if demand else (
        (f"the required property '{LEVERS[lever][0]}' (pinned_lever_present in the checks is a rough mechanical "
         f"indication; judge the property from the world and its story)" if lever == 'coupled' else
         f"the required property '{LEVERS[lever][0]}' (pinned_lever_present in the checks says whether it is really there)")
        if lever else 'no particular demand (a matched control)')
    return f"""You are selecting one world for a benchmark of experimental discovery ({dem}). Below are candidates with
our mechanical checks so far: whether the world compiles, its size, whether a full sweep of experiments would trivially
cover it (trivially_sweepable: false is good), and matches against known mechanisms (empty lists are good). Rival laws,
an experiment plan and the full fairness checks come later, for the chosen world only, so do not reject a candidate
for lacking them.

Choose the single best candidate: new, fair, natural, likely to be hard for strong AI experimenters, and likely to
pass the checks.{QUALITY_SELECT if QUALITY else ''} Choose none (chosen_index -1) only if no candidate is acceptable on these grounds. Explain briefly.

{listing}
"""


SELECT_SCHEMA = {'type': 'object', 'properties': {'chosen_index': {'type': 'integer'}, 'reason': {'type': 'string'}}}


def plan_prompt(demand, cand_json, checks, budget, lever=None, naming=None):
    if demand:
        title, what, audit = DEMANDS[demand]
        dem = f"Demand: {title}. {what}\nCheck: {audit}"
    elif lever:
        dem = f"Required property: {LEVERS[lever][0]}. {LEVERS[lever][1]} Keep it, and make it matter for the queries."
    else:
        dem = 'No particular demand (matched control).'
    dem += naming or ''
    return f"""You are finalizing a world for a benchmark of experimental discovery (budget {budget} experiments).

{dem}

The chosen candidate and our mechanical checks:
{cand_json}

{checks}

{NATURALNESS}
{LANGUAGE}

Revise the world if the checks show a problem (a world a full sweep covers, no clue in typical behaviour, an unused
component, a world too large for the limits). Return the final world JSON; the neutral public description; named rival
laws (2-4, each a full world JSON string with a label: plausible wrong laws a careful solver might settle on after
typical experiments, each differing from the true law somewhere a thoughtful experiment can reach); an experiment plan
that would reveal the law (at most {budget} experiments, each {{"start": {{...}}, "inputs": [["A"], ["C", 2], ...]}}
using your own action and variable names); and notes.
"""


PLAN_SCHEMA = {'type': 'object', 'properties': {
    'world_json': {'type': 'string'}, 'public_description': {'type': 'string'},
    'rivals': {'type': 'array', 'items': {'type': 'object', 'properties': {'label': {'type': 'string'},
                                                                        'world_json': {'type': 'string'}}}},
    'plan_json': {'type': 'string'}, 'notes': {'type': 'string'}}}


# ------------------------------------------------------------------ the public interface and the pilot

def public_interface(world, description, budget):
    """The solver's view, generated by code from the world: abstract action and readout names, parameter ranges,
    settable variables and ranges, budget and length limit. The law is never shown."""
    acts = []
    for i, (name, d) in enumerate(world.raw['actions'].items()):
        label = chr(ord('A') + i)
        acts.append(f"{label}" + (f"(k) with k in {d['param'][0]}..{d['param'][1]}" if d.get('param') else ''))
    rds = [f"r{j + 1}" for j in range(len(world.readout_names))]
    sett = [f"{v if V2 else f's{j + 1}'} in {world.lo[v]}..{world.hi[v]}" for j, v in enumerate(world.settable)]
    return {'actions': acts, 'readouts': rds, 'settable': sett, 'budget': budget, 'max_len': world.max_len,
            'description': description}


def interface_text(pub):
    return (f"{pub['description']}\n\nActions: {', '.join(pub['actions'])}.\n"
            f"Readouts (integers, shown after the start and after every action): {', '.join(pub['readouts'])}.\n"
            f"Settable start values: {', '.join(pub['settable']) if pub['settable'] else 'none (the device always starts the same way)'}.\n"
            f"An experiment: a start and a sequence of at most {pub['max_len']} actions. Budget: {pub['budget']} experiments in total.\n"
            f"The device is deterministic, and every experiment starts from a fresh reset of the same device.")


EXPERIMENT_FORMAT = (('Each experiment is {"start": {"x": <int>, ...}, "actions": ["A", "C2", ...]}: start values by the '
                      'settable names above' if V2 else 'Each experiment is {"start": {"s1": <int>, ...}, "actions": ["A", "C2", ...]}') +
                     ': actions by name, a parameterised action written with its value appended (C2 is C with k = 2).')


def pilot_turn_prompt(pub, history, remaining, per_turn):
    past = '\n'.join(f"Experiment {i + 1}: start {h['start']}, actions {h['actions']} -> readouts {h['trace']}"
                     for i, h in enumerate(history)) or '(none yet)'
    return f"""You are investigating an unfamiliar device by experiment. Later you will be asked to predict the readouts
of experiments you have not run, so find out how it works.

{interface_text(pub)}

{EXPERIMENT_FORMAT}

Your experiments so far and what you observed (readouts listed in order r1, r2, ... after the start and after each
action):
{past}

You have {remaining} experiments left. Propose up to {per_turn} new experiments now (you will see their results before
proposing more), or none if you are confident you understand the device. Also say briefly what you currently believe.
"""


PILOT_TURN_SCHEMA = {'type': 'object', 'properties': {
    'experiments_json': {'type': 'string'}, 'belief': {'type': 'string'}, 'done': {'type': 'boolean'}}}


def pilot_query_prompt(pub, history, queries):
    past = '\n'.join(f"Experiment {i + 1}: start {h['start']}, actions {h['actions']} -> readouts {h['trace']}"
                     for i, h in enumerate(history)) or '(no experiments were run)'
    qs = '\n'.join(f"Q{i + 1}: start {q['start']}, actions {q['actions']}" for i, q in enumerate(queries))
    return f"""You investigated an unfamiliar device. Now predict, for each query below, the readouts after the LAST
action (as a list [r1, r2, ...]). You cannot run more experiments.{' ' + QUERY_WORDING_V2 if V2 else ''}

{interface_text(pub)}

Your experiments and observations:
{past}

Queries:
{qs}

Answer with a JSON object mapping "Q1", "Q2", ... to readout lists, and state the law you believe in, briefly.
Also write the law as a Python 3 program: it reads ./spec.json, a JSON object {{"queries": [{{"start": {{...}},
"actions": [...]}}, ...]}} in the same format as the queries above, and prints on its last line a JSON object mapping
"Q1", "Q2", ... to readout lists. If both are given, the program's answers are used when it runs.
"""


PILOT_QUERY_SCHEMA = {'type': 'object', 'properties': {'answers_json': {'type': 'string'}, 'law': {'type': 'string'},
                                                        'predictor_program': {'type': 'string'}}}


def pilot_program_prompt(pub, history, queries):
    """Fallback when a solver cannot produce the full query answer: only the law as a program."""
    past = '\n'.join(f"Experiment {i + 1}: start {h['start']}, actions {h['actions']} -> readouts {h['trace']}"
                     for i, h in enumerate(history)) or '(no experiments were run)'
    qs = '\n'.join(f"Q{i + 1}: start {q['start']}, actions {q['actions']}" for i, q in enumerate(queries[:3]))
    return f"""You investigated an unfamiliar device. Write the law you believe in as a Python 3 program; we will run it
on withheld queries. You cannot run more experiments.

{interface_text(pub)}

Your experiments and observations:
{past}

The program reads ./spec.json, a JSON object {{"queries": [{{"start": {{...}}, "actions": [...]}}, ...]}}, and prints
on its last line a JSON object mapping "Q1", "Q2", ... to the list of readouts [r1, r2, ...] after each query's LAST
action. The queries look like these:
{qs}

Answer with the program and the law in one or two sentences.
"""


PILOT_PROGRAM_SCHEMA = {'type': 'object', 'properties': {'predictor_program': {'type': 'string'}, 'law': {'type': 'string'}}}


NAMING_V2 = """
Naming (this round): name the actions A, B, C, ... in order and the readouts r1, r2, ... in order. Give settable
variables short neutral names (x, y, ...): the interface shows them exactly as named. The public description may refer
to settable variables, actions and readouts by these names, and must not mention hidden variables at all.
"""

QUERY_WORDING_V2 = ('For each query give only the readouts after its LAST action: one list with exactly one number per '
                    'readout, in the order r1, r2, ... (not the sequence of readouts over time).')
