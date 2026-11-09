"""Prompts for the model-driven stages of domain B. No example problems are given anywhere (no menus): the language
reference shows syntax only, and ideas must come from the demand, the object type and the archive to avoid."""

import os
QUALITY_B = os.environ.get('RE_B_QUALITY') == '1'   # demo round: the reviewers' quality bar
QUALITY_BAR_B = """## Quality bar (this round)

Reviewers will judge each problem as a research-style benchmark task. Aim for problems they would call natural and
interesting:
- one natural question a combinatorialist would want answered, not an arbitrary statistic and not two ideas glued
  together;
- the obvious construction is not optimal: the best objects need an idea (a structure, a symmetry, a clever
  decomposition);
- not already settled: avoid problems where a short counting argument gives a bound that a simple construction meets.

"""

LANGUAGE = """\
## The problem format

A problem asks for one finite object. Object types:
- graph: a simple undirected graph on vertices 0..n-1. Answer format: {"n": N, "edges": [[u, v], ...]}
- set_system: a family of distinct subsets of the ground set 0..m-1. Answer format: {"m": M, "sets": [[...], ...]}

Problem kinds:
- optimize: maximize or minimize an integer objective subject to constraints.
- satisfy: find any object meeting all constraints (only interesting when such objects are hard to find).
- counterexample: given assumptions (constraints) and a claim, find an object meeting the assumptions where the
  claim is false. The claim must be plausible: true for every object meeting the assumptions at small sizes.

A spec is JSON:
  {"object": "graph" | "set_system", "kind": "optimize" | "satisfy" | "counterexample",
   "params": {"n": 17}  or  {"m": 13},
   "constraints": [{"name": "snake_case_name", "expr": "<boolean expression>"}, ...],
   "objective": {"expr": "<integer expression>", "direction": "max" | "min"},   (optimize only)
   "claim": {"expr": "<boolean expression>"}}                                    (counterexample only)

Expressions are a small subset of Python: integer and boolean literals, arithmetic (+ - * // % and ** with a constant
exponent), comparisons, and/or/not, if-else expressions, set operations (& | ^ -, <= for subset), tuples, generator
expressions and comprehensions (with if), indexing with an integer. No attribute access (no x.method()), no strings,
no lambdas, no imports, no slicing. Builtins: all, any, sum, len, min, max, abs, range, combinations(iterable, k)
with k <= 6, product(...), pow, set, frozenset, tuple, sorted.

Graph functions: n, V (the vertices), E (sorted edge tuples (u, v) with u < v), adj(u, v) (bool), deg(v) (int),
nbrs(v) (a frozenset of vertices), dist(u, v) (int; INF if unreachable). Whole-graph functions return numbers or
booleans, never lists: connected() (bool), components() (the number of connected components), triangles() (the
number of triangles in the graph; to count the triangles at v, write
sum(1 for a, b in combinations(nbrs(v), 2) if adj(a, b))), girth() (int; INF for a forest), diameter() (int; INF if
disconnected), clique_number(), independence_number(), edge_connectivity(), vertex_connectivity() (ints),
is_bipartite() (bool), INF (a large integer).
Set-system functions: m, ground (the elements 0..m-1, a tuple), F (the family: a tuple of distinct frozensets),
size(S) (the same as len(S)).

Syntax only (not problems): `all(deg(v) <= 4 for v in V)`, `sum(1 for A, B in combinations(F, 2) if len(A & B) == 2)`,
`max(len(S) for S in F) if F else 0`.

Parametric specs: write constraints and objectives in terms of n (or m) wherever the problem is naturally parametric,
so the same spec makes sense at a small size. Every problem names a target size (in params) and a small sibling size
at which every object can be enumerated: graphs n <= 7; set systems with few candidate sets (the sibling's candidate
sets, given the search hint below, must number at most 20).

Search hint (private, for our baselines only): for set systems, the set sizes worth searching, e.g. {"set_sizes": [4]}.
Constraints, not the hint, define the problem: if sets must have size 4, say so in a constraint.

Evaluation limits: target sizes up to n = 40 or m = 24; each check must take well under a second, so prefer built-in
functions to deep nested quantifiers (combinations of more than 3 over large families are too slow).
"""

DEMANDS = {
    'D1': ('interacting constraints',
           'Two or three natural requirements that conflict: each alone is easy to satisfy well, but objects that are '
           'good for one violate another, so they must be handled jointly. The conflict should come from what the '
           'requirements mean, not from a pile of side conditions.',
           'We drop each constraint in turn, optimize the rest, then repair greedily; this must fall clearly short of '
           'your reference (or end invalid), while a joint search over all constraints reaches it. No constraint may be '
           'implied by the others (checked exhaustively at the sibling size).'),
    'D2': ('exploitable structure',
           'The best objects are highly structured (cyclic, algebraic, product or symmetric constructions) and an '
           'unstructured search falls short; seeing the structure is the insight.',
           'Unrestricted simulated annealing and kick search must fall clearly short of your reference at a fixed budget, '
           'even at ten times the budget, while a search over cyclically symmetric objects only (circulant graphs, or '
           'set systems made of whole orbits under i -> i+1 mod m) reaches it. Your reference must be an explicit '
           'construction, not the output of a search.'),
    'D3': ('escaping local optima',
           'The obvious construction is a local optimum: improving on it needs a coordinated change of many elements at '
           'once, not a sequence of small improvements.',
           'Restarted 1- and 2-move local search must stall clearly below your reference, while iterated local search '
           'with large random kicks reaches it.'),
    'D4': ('counterexample',
           'A natural-sounding claim that holds for every small object and for typical objects, but fails for some '
           'carefully structured object at the target size.',
           'The claim must hold for every object meeting the assumptions at the sibling size (checked exhaustively); '
           'annealing on the margin of falsifying the claim must fail at a fixed budget; your counterexample must be '
           'valid.'),
}


def ideas_prompt(obj, demand, archive_lines, count=3):
    if demand:
        title, what, audit = DEMANDS[demand]
        dem = (f"## The demand: {title}\n\n{what}\n\nHow it will be checked (mechanically, by code, against your own "
               f"reference): {audit}\n\nA problem that does not meet this check is rejected, however interesting.\n")
        kind_note = 'Use kind "counterexample".' if demand == 'D4' else 'Use kind "optimize" or "satisfy".'
    else:
        dem = ("## The demand\n\nNo particular demand: design the best new problem you can, nontrivial for simple "
               "search to solve well.\n")
        kind_note = 'Use any kind.'
    return f"""You are designing new research-style construction problems in finite combinatorics for a benchmark.
Each problem asks a solver to construct one finite {obj.replace('_', ' ')}. It must be:
- new: not a known textbook or literature problem in disguise (restating, dualizing, renaming or adding a redundant
  constraint does not make a problem new);
- precise: the plain-language statement and the formal spec say exactly the same thing;
- feasible: a valid object exists, and you can construct one explicitly;
- open: other valid constructions may exist, and a solver may beat your reference; that is welcome;
- natural: it reads like a question a combinatorialist would ask for its own sake (see below).

{dem}
{QUALITY_BAR_B if QUALITY_B else ''}## Naturalness

A good problem has one clean objective or property and at most three constraints, each of which the question itself
needs. Never add a constraint, threshold or constant only to make search harder or to block a shortcut: if the
problem is too easy without it, look for a different problem. Constants should be forced by the structure (n, m,
small numbers with a reason), not tuned. A reader should be able to say in one sentence what the problem is about.

## Known problems to avoid (do not restate, dualize or combine trivially)

{archive_lines}

{LANGUAGE}

## Your task

Propose {count} distinct candidate problems for object type `{obj}`. {kind_note} Each needs a plain-language statement
a mathematician would accept as a problem statement, the formal spec as a JSON string, a target size in the spec's
params, a sibling size, a search hint (JSON string, "{{}}" for graphs), why the demand above holds (or, with no demand,
why simple search will not solve it well), the construction idea behind your reference object, and why it is not a
known problem.
"""


IDEAS_SCHEMA = {'type': 'object', 'properties': {'candidates': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'title': {'type': 'string'}, 'statement': {'type': 'string'}, 'spec_json': {'type': 'string'},
    'sibling_params_json': {'type': 'string'}, 'search_hint_json': {'type': 'string'},
    'demand_rationale': {'type': 'string'}, 'construction_idea': {'type': 'string'}, 'novelty_note': {'type': 'string'}}}}}}


def select_prompt(obj, demand, candidates_with_checks):
    dem = f"the demand '{DEMANDS[demand][0]}'" if demand else 'no particular demand (a matched control)'
    return f"""You are selecting one problem for a benchmark of new construction problems in finite combinatorics
(object type `{obj}`, {dem}). Below are the candidates with the results of our mechanical checks: whether the spec
compiles, exhaustive results at small sizes, redundant constraints, and whether its behaviour matches a known problem.

Choose the single best candidate: new (no match to a known problem), precise, feasible, natural (a question worth
asking for its own sake, with few constraints, none of them there only to defeat search), and likely to pass the
demand's check.{' Weigh interest heavily: prefer one natural question whose best constructions need an idea and whose optimum is not settled by a short counting argument; avoid arbitrary statistics.' if QUALITY_B else ''} Prefer the more natural of two otherwise similar candidates. If none is acceptable, choose none
(chosen_index -1). Also rank every acceptable candidate from best to worst (ranking, a list of indices; leave out
unacceptable ones). Explain briefly.

{candidates_with_checks}
"""


SELECT_SCHEMA = {'type': 'object', 'properties': {'chosen_index': {'type': 'integer'}, 'reason': {'type': 'string'},
                                                   'ranking': {'type': 'array', 'items': {'type': 'integer'}}}}


def plan_prompt(obj, demand, idea_json, checks):
    if demand:
        title, what, audit = DEMANDS[demand]
        dem = f"Demand: {title}. {what}\nCheck: {audit}"
    else:
        dem = 'No particular demand (matched control). It must still be nontrivial for simple search.'
    return f"""You are finalizing a new construction problem (object type `{obj}`) for a benchmark, and writing the
program that builds your reference object.

{dem}

The chosen idea, and our mechanical checks on it so far:
{idea_json}

{checks}

{LANGUAGE}

## Your task

1. Final spec (JSON string), revised if the checks show a problem (a redundant constraint, a vacuous or degenerate
   case, an expression that is too slow). Keep it parametric, and keep it natural: do not add constraints or tune
   constants only to defeat search.
2. Final plain-language statement: precise, and saying exactly what the spec says.
3. Witness program: Python 3 that prints the reference object as JSON on its last line of stdout. It runs in a sandbox
   with numpy, networkx, itertools and ortools available, no network, a 120 s limit, and the public spec in
   ./spec.json. It must be deterministic. For demand "exploitable structure" it must be an explicit construction, not
   a search.
4. witness_is_construction: true if the program builds the object by an explicit rule rather than by search.
5. Degenerate classes: boolean expressions in the spec language describing valid but degenerate answers that would
   mean the spec has a loophole (for example an empty or near-empty object if the problem is meant to need
   substance). Use an empty list if there are none.
6. Sibling params and search hint (JSON strings).
7. For an optimization problem: the best bound on the optimum you can prove (upper bound for max, lower bound for min),
   as an integer in upper_bound ("none" if you cannot prove one), and the argument in bound_argument. Be honest: if your
   reference meets the bound, say so.
"""


PLAN_SCHEMA = {'type': 'object', 'properties': {
    'spec_json': {'type': 'string'}, 'statement': {'type': 'string'}, 'witness_program': {'type': 'string'},
    'witness_is_construction': {'type': 'boolean'}, 'degenerate_exprs': {'type': 'array', 'items': {'type': 'string'}},
    'sibling_params_json': {'type': 'string'}, 'search_hint_json': {'type': 'string'}, 'notes': {'type': 'string'},
    'upper_bound': {'type': 'string'}, 'bound_argument': {'type': 'string'}}}


def restate_prompt(obj, statement, params, error=None):
    feedback = (f"\nYour previous translation could not be evaluated: {error}\nUse only the functions above, with the "
                f"types given.\n" if error else '')
    return f"""Translate this problem statement into a formal spec, exactly as stated, adding nothing and leaving
nothing out. Object type: {obj}. Size params: {params}.

Statement:
{statement}

{LANGUAGE}

Write only the formal spec (JSON string) and note any ambiguity you had to resolve.
{feedback}"""


RESTATE_SCHEMA = {'type': 'object', 'properties': {'spec_json': {'type': 'string'}, 'ambiguities': {'type': 'string'}}}


def naturalness_prompt(statement):
    return f"""Here is a construction problem in finite combinatorics, written for a benchmark. Rate how natural it is as a
mathematical question, on this scale:
5 - a question a researcher would pose for its own sake; every condition is essential to what it is about.
4 - natural, with a condition or constant that is a little arbitrary.
3 - a reasonable exercise, but some conditions look chosen to make it hard rather than because they matter.
2 - mostly an assembly of conditions; hard to say in one sentence what it is about.
1 - an arbitrary pile of conditions.

Problem:
{statement}

Give the score, a one-sentence summary of what the problem is about (if you can), and list any conditions or
constants that look added only to make the problem harder.
"""


NATURALNESS_SCHEMA = {'type': 'object', 'properties': {'score': {'type': 'integer'}, 'about': {'type': 'string'},
                                                        'arbitrary_parts': {'type': 'array', 'items': {'type': 'string'}},
                                                        'reasons': {'type': 'string'}}}


def solve_object_prompt(public, history):
    past = ''
    if history:
        past = '\n## Your previous attempts and the checker\'s verdicts\n\n' + '\n\n'.join(
            f"Attempt {i + 1}: {h['object_json'][:3000]}\nVerdict: {h['verdict']}" for i, h in enumerate(history))
    return f"""Solve this construction problem. Give the object explicitly as JSON. If it is an optimization problem,
make the objective as good as you can.

{public}
{past}
Answer with the object as a JSON string in the required format, and a short note on your construction.
"""


SOLVE_OBJECT_SCHEMA = {'type': 'object', 'properties': {'object_json': {'type': 'string'}, 'note': {'type': 'string'}}}


def solve_code_prompt(public, history, limit):
    past = ''
    if history:
        past = '\n## Your previous programs and the results\n\n' + '\n\n'.join(
            f"Program {i + 1} result: {h['verdict']}" for i, h in enumerate(history))
    return f"""Solve this construction problem by writing a Python 3 program. The program runs in a sandbox with numpy,
networkx, itertools and ortools (CP-SAT) available, no network, about {limit} s of wall-clock time, and the public
spec in ./spec.json. It must print the object as JSON on its last line of stdout. If it is an optimization problem,
make the objective as good as you can within the time limit.

{public}
{past}
Answer with the program and a short note on its approach.
"""


SOLVE_CODE_SCHEMA = {'type': 'object', 'properties': {'program': {'type': 'string'}, 'note': {'type': 'string'}}}


def public_package(statement, raw):
    fmt = '{"n": N, "edges": [[u, v], ...]}' if raw['object'] == 'graph' else '{"m": M, "sets": [[...], ...]}'
    spec = {k: v for k, v in raw.items() if k in ('object', 'kind', 'params', 'constraints', 'objective', 'claim')}
    import json
    return (f"## Problem\n\n{statement}\n\n## Formal spec (authoritative)\n\n```json\n{json.dumps(spec, indent=1)}\n```\n\n"
            f"Answer format: {fmt}\n\nExpressions in the spec use this language:\n\n{LANGUAGE}")


def improve_prompt(public, objective, direction, obj_json, limit):
    return f"""A construction problem and the best object known for it are given below. Write a Python 3 program that
finds a strictly better valid object if one exists ({'larger' if direction == 'max' else 'smaller'} objective than
{objective}), using any method: search, algebraic construction, exact solvers. The program runs in a sandbox with numpy,
networkx, itertools and ortools (CP-SAT), no network, about {limit} s of wall-clock time, and the public spec in
./spec.json. Print the best valid object you find as JSON on the last line of stdout (print the known object if you
cannot improve on it).

{public}

## Best known object (objective {objective})

{obj_json[:20000]}

Answer with the program and a short note on its approach.
"""
