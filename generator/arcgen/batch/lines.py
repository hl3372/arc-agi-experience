"""Weakness lines (2026-10-20).

Each line is a property that admits many rules and trains against one shortcut the 4B takes. The departure package's
pattern, generalised to every line and driven by data (weakness_lines.json; a new line is its prompt files plus one entry):
- an idea modifier (prompts/v1/ideas/modifiers/<LINE>.md), pinned per spec row, with the scoped-novelty steps;
- a binding planner line composed last (prompts/v1/lines/<LINE>.planner.md), naming the default clauses it overrides, and
  for an idea in a relaxed fence, that relaxation's addendum after it;
- a protected public invariant code finds by prefix (repairs cannot change frozen invariants);
- fixed-prefix plan fields checked here (plan_issues), with one re-plan sharing the rung re-plan (runtime._plan_with_colour);
- a review policy and a repair note, keyed on the invariant (gate.py, runtime.call_review);
- what the realizer emits in-call (REALIZER_STAGE) and the repairer re-emits (REPAIR_STAGE): one alternative per declared
  point (DESIGN §6.1: the shortcut applied at that point only, as the cells where it differs from the truth), the
  shortcut on every training pair for SETTLE and COINCIDE, and RELATION's object ledger, FORCED's pieces and COINCIDE's
  killer; PROCESS's conserved quantity and the growth of DEPART's tests are computed from the grids.
Code checks all of it on every episode version and records the result; nothing here rejects a task: a task that loses its
property keeps going and only loses the line's tag. The measurement records are validated apart from the realization and
the repair, leniently (a malformed or oversized record is recorded as a problem, never fails the call), and FORCED's
solver runs once per emitted assembly, outside the runtime lock, its result stored with the emission. Overlays (EDGE, ROLES) are per-idea planner draws
(admission._weakness_draws) with their own planner lines; EDGE also declares a point ("the cut thing ignored"). The
matched control has a planner line for its episode shape and declares its plan's top shortcut as its point.
"""
from copy import deepcopy
from collections import Counter
import json
import re
from arcgen.lib import astra_authoring_schemas_v3 as schemas
from arcgen.lib.schemas import parse_json_object, SchemaError
from arcgen.lib.util import load_prompt
from arcgen.lib.v4_schemas import GRID, TEXT, _array, _object, _transport_schema
from arcgen.astra_evidence_runtime_revision_20260911.structure import check
from .common import HERE, IntegrityError
from .config import LINE_REGISTRY, LINE_CONTROL, SCIENCE
from . import repair_v1

V1 = HERE / 'prompts/v1'
LINE_DIR = V1 / 'lines'
REALIZER_STAGE = 'weakness_v1_task_realizer'
REPAIR_STAGE = 'weakness_v1_bounded_repair'
POLICY_VERSION = 'weakness-v1-policies-4'  # 4: RECOMBINE's policy for laws acting in turn (2026-10-23)
LINES, OVERLAYS = LINE_REGISTRY['lines'], LINE_REGISTRY['overlays']
STRUCTS = LINE_REGISTRY.get('structures') or {}  # structure lines (2026-10-22, STRUCTURE_SPEC.md): a second axis
DEFAULT_STAGE = 'weakness_v1_default_reading'  # the recorded default-reading call (STRUCTURE_SPEC §D)
EMIT_FIELDS = {'pairs': 'changed_pairs', 'points': 'points', 'overlay_points': 'overlay_points', 'objects': 'objects',
               'assembly': 'assembly', 'killer': 'killer', 'alternatives': 'alternatives', 'units': 'units',
               'surfaces': 'surfaces'}
LEDGERS = ('points', 'overlay_points', 'objects', 'assembly', 'killer')  # emitted whole; the latest emission stands
# Structure records (STRUCTURE_SPEC amendment v2, S1): kept per input grid, so a repair re-writes only the pairs it adds or
# changes; alternatives are whole rival grids of any shape, per record, never mixed into the content line's points
KEYED = ('alternatives', 'units', 'surfaces')
CAPTURE = 0.8  # DESIGN §6.1: a sample is on a point when it equals the alternative on this share of the point's cells
# R4's compressibility floor (v2 §6.3 and §12): the test output's row run-length count is no higher than the 90th
# percentile of the public eval set's test outputs of 600 cells or more (53 outputs: median 178; the 47th of 53, 351, as
# v2 §12 gives it; re-measured 2026-10-20). Computed fields with a one-law per-cell value declared in the plan ("Regular
# field:") are exempt; tedium is guarded by the rater bar (a kill on pooled R4). A calibration number, never in a prompt.
R4_ROW_RUNS = 351
R4_TRAIN_CELLS, R4_COMPUTED, R4_COPIED = 200, 0.30, 0.70  # training pair at least min(half the test, 200) cells
NO_FORM = {'label': None, 'text': None, 'basis': None}


def body(name):
    """A policy or note file's text after its version header."""
    return (V1 / name).read_text().split('-->', 1)[1].strip()


def component(name):
    return load_prompt(V1 / name)


def spec(name, fence=None):
    """A line's or overlay's registry entry; with a relaxed fence, its plan and grid checks take the relaxation's
    overrides (the relaxation's own flags are kept under 'relaxation')."""
    s = (LINES.get(name) or OVERLAYS.get(name) or STRUCTS.get(name)) if name else None
    relax = ((s or {}).get('relax') or {}).get(fence) if fence else None
    if not relax:
        return s
    return {**s, 'plan': {**(s.get('plan') or {}), **(relax.get('plan') or {})},
            'grids': {**(s.get('grids') or {}), **(relax.get('grids') or {})}, 'relaxation': relax}


# ------------------------------------------------------------------ study


def line_of(study):
    return (study or {}).get('line')


def overlay_of(study):
    """The overlay drawn for an idea (study.weak_overlays: {OVERLAY: drawn | control} for each overlay it was eligible for)."""
    return next((k for k, v in ((study or {}).get('weak_overlays') or {}).items() if v == 'drawn'), None)


def fence_of(study):
    """The relaxed fence an idea is in, or None (strict, or a line without relaxations)."""
    fence = (study or {}).get('fence')
    return fence if fence and fence != 'strict' else None


def structure_of(study):
    """The structure line an idea carries (study.structure, set at admission from the call's pin), or None."""
    return (study or {}).get('structure')


def part_of(name):
    """Which part of an idea a registry name is: its content line, its overlay, or its structure line."""
    return 'line' if name in LINES else 'overlay' if name in OVERLAYS else 'structure' if name in STRUCTS else None


def _specs(study):
    """[(name, spec)] of the idea's line (with its fence's overrides), its drawn overlay and its structure line."""
    return [(n, spec(n, f)) for n, f in ((line_of(study), fence_of(study)), (overlay_of(study), None), (structure_of(study), None))
            if spec(n)]


def emits(study):
    """What the realizer records for this idea (empty: the plain realizer and repair stages)."""
    return tuple(k for _, s in _specs(study) for k in s.get('emits', ()))


def tagged(study):
    return bool(_specs(study))


# ------------------------------------------------------------------ plan fields


def _sections(plan):
    if not isinstance(plan, dict):
        return None, None
    plan = plan if 'public_contract' in plan else plan.get('concrete_plan') or {}
    if plan.get('status', 'ready') != 'ready':
        return None, None
    return plan.get('public_contract') or None, plan.get('private_execution') or None


def _starts(text, prefix):
    return isinstance(text, str) and isinstance(prefix, str) and text.strip().startswith(prefix)


def invariant(plan, prefix):
    public, _ = _sections(plan)
    return next((x.strip() for x in (public or {}).get('invariants') or [] if _starts(x, prefix)), None) if prefix else None


def shortcuts(plan):
    _, private = _sections(plan)
    return ((private or {}).get('cross_pair_concrete_audit') or {}).get('strongest_simple_shortcuts') or []


def declared_shortcut(name, plan):
    """The rule of the shortcut a line or overlay names: the entry beginning its prefix, or the plan's top entry."""
    prefix = (spec(name) or {}).get('shortcut')
    rows = shortcuts(plan)
    if prefix == 'top':
        return rows[0]['rule'].strip() if rows and isinstance(rows[0].get('rule'), str) else None
    return next((r['rule'].strip() for r in rows if _starts(r.get('rule'), prefix)), None) if prefix else None


def _value(text, prefix):
    """The text after a fixed prefix, up to its first sentence end, normalised for comparison."""
    rest = text.strip()[len(prefix):].strip()
    rest = re.split(r'(?<=[^\d])[.;](?:\s|$)', rest, maxsplit=1)[0]
    return re.sub(r'\s+', ' ', rest).strip().lower()


def _area(d):
    return d['height'] * d['width']


def declared(name, plan, fence=None):
    """The fixed-prefix fields a plan carries for one line or overlay (claims: they tell code what to protect and where to
    look, never evidence of the property)."""
    public, private = _sections(plan)
    s = spec(name, fence)
    if not public or not private or not s:
        return {'plan': 'none'}
    rules = s.get('plan') or {}
    pairs = public.get('training_pairs') or []
    tests = public.get('test_inputs') or []
    notes = [p.get('what_it_helps_communicate') or '' for p in pairs]
    test_notes = [t.get('what_is_new_relative_to_training') or '' for t in tests]
    first_test = test_notes[0] if test_notes else ''
    risk = (private.get('cross_pair_concrete_audit') or {}).get('remaining_ambiguity_risk') or ''
    inv = invariant(plan, s.get('invariant'))
    opt = s.get('opt_out') or {}
    if opt.get('field') == 'invariants':  # a structure line's opt-out is its own invariant ("Structure: none — why")
        opted = bool(invariant(plan, opt.get('prefix')))
    else:
        where = {'risk': risk, 'test': first_test, 'invariant': inv or ''}.get(opt.get('field'), '')
        opted = _starts(where, opt.get('prefix'))
    out = {'plan': 'ready', 'name': name, 'fence': fence, 'training_pairs': len(pairs), 'test_inputs': len(tests),
           'invariant': inv, 'shortcut': declared_shortcut(name, plan), 'shortcuts_listed': len(shortcuts(plan)),
           'opted_out': opted}
    if rules.get('shortcut_count'):  # structure lines: how many strongest_simple_shortcuts entries begin each prefix
        rules_text = [r.get('rule') for r in shortcuts(plan)]
        out['shortcut_counts'] = {c['prefix']: sum(_starts(t, c['prefix']) for t in rules_text) for c in rules['shortcut_count']}
    if s.get('invariants_extra'):
        out['invariants_extra'] = {p: invariant(plan, p) for p in s['invariants_extra']}
    train_areas = [_area(p['input_dimensions']) for p in pairs if p.get('input_dimensions')]
    test_areas = [_area(t['input_dimensions']) for t in tests if t.get('input_dimensions')]
    if train_areas and test_areas:
        out['test_growth'] = round(max(test_areas) / max(train_areas), 3)
    labels = rules.get('pair_labels')
    if labels:
        out['marked_pairs'] = [i for i, t in enumerate(notes, 1) if _starts(t, labels['marked'])]
        out['other_pairs'] = [i for i, t in enumerate(notes, 1) if _starts(t, labels['other'])]
    values = rules.get('pair_values')
    if values:
        # string keys: the receipt is frozen as JSON, and a resume compares it again (review S1)
        out['pair_values'] = {str(i): _value(t, values['prefix']) for i, t in enumerate(notes, 1) if _starts(t, values['prefix'])}
    test_rule = rules.get('test_field')
    if test_rule:
        out['test_field'] = first_test.strip() if _starts(first_test, test_rule['prefix']) else None
        out['test_value'] = _value(first_test, test_rule['prefix']) if out['test_field'] else None
        out['later_tests_with_field'] = [i for i, t in enumerate(test_notes[1:], 2) if _starts(t, test_rule['prefix'])]
    private_rule = rules.get('private_field')
    if private_rule:
        why = ((private.get('test_outputs') or [{}])[0].get('why_one_output_is_best_supported') or '').strip()
        rest = why[len(private_rule['prefix']):].strip().lower() if _starts(why, private_rule['prefix']) else None
        out['private_field'] = why if rest is not None else None
        out['private_value'] = next((v for k, v in private_rule['values'].items() if rest and rest.startswith(k)), None)
        if rest and not out['private_value']:  # a named value off the list (JUDGE: a dropped, supporting or excluded prior)
            out['private_rejected'] = next(([k, v] for k, v in (private_rule.get('rejected') or {}).items() if k in rest), None)
        out['private_missing'] = [x for x in private_rule.get('requires') or () if x.lower() not in why.lower()]
    if rules.get('pair_mention'):
        out['mentioning_pairs'] = [i for i, t in enumerate(notes, 1) if rules['pair_mention'] in t]
    if rules.get('role_meaning'):
        roles = (public.get('episode_overview') or {}).get('functional_color_roles') or []
        out['role_meanings'] = [r['role'] for r in roles if _starts(r.get('functional_meaning'), rules['role_meaning'])]
    out['regular_field'] = invariant(plan, 'Regular field:')  # R4: a declared regular computed field is exempt from the floor
    return out


def _issues(name, plan, fence=None):
    s = spec(name, fence)
    rules = s.get('plan') or {}
    d = declared(name, plan, fence)
    if d['plan'] != 'ready' or d['opted_out'] or rules.get('record_only'):
        return []
    out, n = [], d['training_pairs']
    if s.get('invariant') and not d['invariant']:
        out.append(f'it had no invariant beginning "{s["invariant"]}"')
    for prefix, found in (d.get('invariants_extra') or {}).items():
        if not found:
            out.append(f'it had no invariant beginning "{prefix}"')
    if s.get('shortcut') == 'top' and not d['shortcuts_listed']:
        out.append('it listed no strongest_simple_shortcuts')
    elif s.get('shortcut') not in (None, 'top') and not d['shortcut']:
        out.append(f'no strongest_simple_shortcuts entry began "{s["shortcut"]}"')
    if rules.get('pairs') and not rules['pairs'][0] <= n <= rules['pairs'][1]:
        a, b = rules['pairs']
        out.append(f'it had {n} training pairs, not {a}' + (f' to {b}' if b != a else ''))
    labels = rules.get('pair_labels')
    if labels:
        k = len(d['marked_pairs'])
        if not (labels['min'] <= k <= labels['max'] and (not labels.get('minority') or 2 * k < n)):
            out.append(f'{k} pairs began "{labels["marked"]}", where {labels["min"]} to {labels["max"]}'
                       + (' (fewer than half the pairs)' if labels.get('minority') else '') + ' should')
        if k + len(d['other_pairs']) != n:
            out.append(f'not every other pair began "{labels["other"]}"')
    values = rules.get('pair_values')
    if values:
        got = d['pair_values']
        distinct = len(set(got.values()))
        if len(got) != n:
            out.append(f'not every training pair began "{values["prefix"]}"')
        elif values.get('all_distinct') and distinct != n:
            out.append(f'two training pairs gave the same value after "{values["prefix"]}"')
        need, relaxed = values.get('distinct_min'), values.get('relaxed')
        if relaxed and d['invariant'] and relaxed['phrase'] in d['invariant']:
            need = relaxed['distinct_min']
        if need and distinct < need:
            out.append(f'the pairs showed {distinct} distinct values after "{values["prefix"]}", where {need} are needed'
                       + (f' (fewer only when the invariant says "{relaxed["phrase"]}")' if relaxed else ''))
    test_rule = rules.get('test_field')
    if test_rule:
        if not d['test_field']:
            out.append(f'test_1\'s what_is_new_relative_to_training did not begin "{test_rule["prefix"]}"')
        elif test_rule.get('not_among_pair_values') and d['test_value'] in set((d.get('pair_values') or {}).values()):
            out.append(f'the test\'s value after "{test_rule["prefix"]}" equals a training pair\'s')
    private_rule = rules.get('private_field')
    if private_rule and not d['private_value']:
        named = d.get('private_rejected')
        out.append((f'test_1\'s why_one_output_is_best_supported named "{named[0]}" after "{private_rule["prefix"]}", which is '
                    f'{named[1]}; it must begin "{private_rule["prefix"]}" and one of: ' if named else
                    f'test_1\'s why_one_output_is_best_supported did not begin "{private_rule["prefix"]}" and one of: ')
                   + ', '.join(private_rule['values']))
    for phrase in (d.get('private_missing') or []) if private_rule else []:
        out.append(f'test_1\'s why_one_output_is_best_supported did not state "{phrase}"'
                   + (f' ({private_rule["requires_note"]})' if private_rule.get('requires_note') else ''))
    if rules.get('growth_max') and d.get('test_growth', 0) > rules['growth_max']:
        out.append(f'its test input was {d["test_growth"]:g} times the largest training input, larger than allowed '
                   f'({rules["growth_max"]:g})')
    if rules.get('pair_mention') and not d['mentioning_pairs']:
        out.append(f'no training pair\'s what_it_helps_communicate named "{rules["pair_mention"]}"')
    for c in rules.get('shortcut_count') or ():
        k = (d.get('shortcut_counts') or {}).get(c['prefix'], 0)
        if not c['min'] <= k <= c['max']:
            out.append(f'{k} strongest_simple_shortcuts entries began "{c["prefix"]}", where {c["min"]}'
                       + (f' to {c["max"]}' if c['max'] != c['min'] else '') + ' should')
    if rules.get('pairs_per_rival'):  # ELIMINATE: at least one more training pair than rivals
        k = (d.get('shortcut_counts') or {}).get(rules['pairs_per_rival'], 0)
        if k and n < k + 1:
            out.append(f'it had {n} training pairs for {k} rivals, where at least {k + 1} are needed')
    return out


def plan_issues(line, overlay, plan, fence=None, structure=None):
    """What a ready plan lacks for its line (under its relaxed fence, if any), overlay or structure line, each as a short
    note for the one re-plan. An explicit opt-out (a structure line's "Structure: none") is honest, not missing: it is
    recorded and never re-planned; a record_only overlay is never re-planned."""
    return [i for n, f in ((line, fence), (overlay, None), (structure, None)) if spec(n) for i in _issues(n, plan, f)]


def replan_note(issues):
    """The one note a re-plan receives (appended after any scale note), naming what the line check found."""
    return ('Line check on an earlier plan for this idea: ' + '; '.join(issues) + '. Plan the idea again so that it meets '
            'the binding lines above and records them in exactly the stated forms. Keep what makes the idea work. If the '
            'idea cannot carry them, say so in the stated way; the earlier plan is used if the new one aborts.')


def form(name, d):
    """A measurement-only form label (DESIGN §3 "form tagging", v2 §2), never shown to a model: a declared variant or
    value, or a coarse keyword reading of the plan's own declaration for the mid-run form count (the main session
    classifies the recorded texts properly). The relaxed fence, if any, is its own stratum (d['fence'])."""
    rule = (spec(name) or {}).get('form')
    if d.get('plan') != 'ready':
        return dict(NO_FORM)
    if not rule:
        return {**NO_FORM, 'text': d.get('invariant')}
    if rule['field'] in ('killer', 'private'):  # COINCIDE's killer, JUDGE's prior
        return {'label': d.get('private_value'), 'text': d.get('private_field'), 'basis': 'declared'}
    text = d.get('test_field') if rule['field'] == 'test' else d.get('invariant')
    if not text:
        return dict(NO_FORM)
    if 'variants' in rule:
        return {'label': next((label for label, prefix in rule['variants'] if text.startswith(prefix)), None), 'text': text,
                'basis': 'declared'}
    low = text.lower()
    return {'label': next((label for label, pattern in rule['keywords'] if re.search(pattern, low)), None), 'text': text,
            'basis': 'keyword'}


def plan_check(line, overlay, first, second=None, replanned=False, used='first', fence=None, structure=None):
    """The plan-stage receipt (run_dir/line_check/<task>.json and record['weakness']['plan_check']). A structure line's
    keys are present only when the idea carries one (a receipt without one is unchanged)."""
    chosen = second if used == 'second' else first
    value = {'line': line, 'overlay': overlay, 'fence': fence,
             'first_issues': plan_issues(line, overlay, first, fence, structure),
             'replanned': replanned, 'used': used, 'final_issues': plan_issues(line, overlay, chosen, fence, structure)}
    if line:
        value['declared'] = declared(line, chosen, fence)
    if overlay:
        value['overlay_declared'] = declared(overlay, chosen)
    value['form'] = form(line, value['declared']) if line else dict(NO_FORM)
    if structure:
        value['structure'] = structure
        value['structure_declared'] = d = declared(structure, chosen)
        value['structure_form'] = form(structure, d)
    return value


# ------------------------------------------------------------------ realizer and repair outputs


NULL = {'type': 'null'}
nullable = lambda x: {'anyOf': [deepcopy(x), NULL]}
INT = lambda lo, hi: {'type': 'integer', 'minimum': lo, 'maximum': hi}
CELL = {'type': 'array', 'items': INT(0, 29), 'minItems': 2, 'maxItems': 2}
CELLS = _array(CELL, minimum=1, maximum=900)
PATCH = _array({'type': 'array', 'items': INT(0, 29), 'minItems': 3, 'maxItems': 3}, minimum=1, maximum=900)
# One declared point: the cells where the answer with the shortcut applied at this point only differs from the truth,
# as [row, col, colour the shortcut puts there]; `at` is the cell where it is decided (TRACE, PRECISE), and `taught_by`
# the pairs where its kind acts (PRECISE).
LTEXT = {'type': 'string'}  # measurement text may be empty (review B2)
POINT = _object({'id': LTEXT, 'kind': LTEXT, 'note': LTEXT, 'test': INT(1, 5), 'at': nullable(CELL),
                 'taught_by': _array(LTEXT, maximum=30), 'cells': PATCH})
CAPS = {'points': 200, 'overlay_points': 200, 'objects': 1000, 'alternatives': 12, 'units': 16, 'surfaces': 16}
# list ledgers beyond these are truncated and recorded
POINTS = _array(POINT, minimum=1, maximum=CAPS['points'])
OBJECT = _object({'grid': LTEXT, 'object': LTEXT, 'colour': INT(0, 9), 'size': INT(0, 900), 'shape': LTEXT, 'region': LTEXT,
                  'relation': LTEXT, 'outcome': LTEXT})
PIECE = _object({'id': LTEXT, 'cells': CELLS, 'colour': INT(0, 9)})
ASSEMBLY = _object({'constraint': {'type': 'string', 'enum': ['cover_region', 'connectors', 'edges', 'other']},
                    'transforms': {'type': 'string', 'enum': ['none', 'turns', 'turns_and_mirrors']},
                    'target': CELLS, 'pieces': _array(PIECE, minimum=1, maximum=200),
                    'anchor': _object({'piece': LTEXT, 'cells': CELLS})})
KILLER = _object({'family': {'type': 'string', 'enum': ['no_single_answer', 'broken_regularity', 'finished_instance']},
                  'invariant': nullable(LTEXT), 'finished_sites': nullable(_array(_array(CELL, maximum=900), minimum=1, maximum=13))})
CHANGED = _object({'pair_number': INT(1, 10), 'shortcut_output': nullable(GRID)})
# A structure record: one rival (or single law, order, replay, naive parse, surface reading) as whole grids, one per
# training pair (null where not given) and one per test input
ALT = _object({'id': LTEXT, 'line': LTEXT, 'kind': LTEXT, 'note': LTEXT, 'train': _array(nullable(GRID), maximum=10),
               'test': _array(nullable(GRID), maximum=5)})
LABELS = {'type': 'array', 'minItems': 1, 'maxItems': 30,
          'items': {'type': 'array', 'minItems': 1, 'maxItems': 30, 'items': INT(0, 999)}}
UNIT = _object({'grid': LTEXT, 'labels': LABELS})  # PARSE: each cell's unit under the intended parse (0: the ground)
SURFACE = _object({'grid': LTEXT, 'roles': _array(_object({'role': LTEXT, 'colour': INT(0, 9)}), maximum=20),
                   'key_box': nullable(_array(INT(0, 29), minimum=4, maximum=4)), 'orientation': nullable(LTEXT),
                   'scale': nullable(LTEXT)})  # VARY: one input grid's surface
LEDGER_FIELDS = {'points': nullable(POINTS), 'overlay_points': nullable(POINTS),
                 'objects': nullable(_array(OBJECT, minimum=1, maximum=CAPS['objects'])), 'assembly': nullable(ASSEMBLY),
                 'killer': nullable(KILLER), 'alternatives': nullable(_array(ALT, minimum=1, maximum=CAPS['alternatives'])),
                 'units': nullable(_array(UNIT, minimum=1, maximum=CAPS['units'])),
                 'surfaces': nullable(_array(SURFACE, minimum=1, maximum=CAPS['surfaces']))}
REALIZER_OUTPUTS = _object({'training_outputs': nullable(_array(GRID, minimum=1, maximum=10)), **LEDGER_FIELDS})
REPAIR_OUTPUTS = _object({'changed_pairs': _array(CHANGED, maximum=10), **LEDGER_FIELDS})
ITEMS = {'points': POINT, 'overlay_points': POINT, 'objects': OBJECT, 'changed_pairs': CHANGED, 'alternatives': ALT,
         'units': UNIT, 'surfaces': SURFACE}
WHOLE = {'training_outputs': nullable(_array(GRID, minimum=1, maximum=10)), 'assembly': nullable(ASSEMBLY),
         'killer': nullable(KILLER)}


def _with_outputs(schema, outputs):
    value = deepcopy(schema)
    value['properties']['weakness_outputs'] = nullable(outputs)
    value['required'] = list(value['required']) + ['weakness_outputs']
    return value


REALIZER_STRICT = _with_outputs(schemas.REALIZER, REALIZER_OUTPUTS)  # the transport shape the model sees
REPAIR_STRICT = _with_outputs(repair_v1.STRICT_SCHEMA, REPAIR_OUTPUTS)
REALIZER_SCHEMA, REPAIR_SCHEMA = _transport_schema(REALIZER_STRICT), _transport_schema(REPAIR_STRICT)
from arcgen.lib import e18_calls
# a failed repair's retry sees the validation error, as the ordinary repair's does; the weakness realizer retries as the
# ordinary realizer does (its measurement record never fails the call)
e18_calls.SEMANTIC_FEEDBACK_STAGES.add(REPAIR_STAGE)


def split(value):
    """(the ordinary response, its weakness_outputs)."""
    base = deepcopy(value)
    extra = base.pop('weakness_outputs', None) if isinstance(base, dict) else None
    return base, extra


def _error(exc):
    return str(exc).strip().splitlines()[0][:300] if str(exc).strip() else type(exc).__name__


def lenient(extra, first):
    """weakness_outputs validated apart from the realization or repair (review B2): a malformed field is dropped, the
    bad items of a list are dropped, a list beyond its cap is truncated, and each is recorded under '_problems'; the call
    itself never fails on its measurement record. first: 'training_outputs' (realizer) or 'changed_pairs' (repair)."""
    if extra is None:
        return None
    if not isinstance(extra, dict):
        return {'_problems': ['weakness_outputs is not an object']}
    out, problems = {}, []
    for field in (first, *LEDGERS, *KEYED):
        value = extra.get(field)
        if value is None:
            out[field] = [] if field == 'changed_pairs' else None
            continue
        if field in ITEMS:
            if not isinstance(value, list):
                out[field] = [] if field == 'changed_pairs' else None
                problems.append(f'{field}: not a list')
                continue
            kept, errors = [], []
            for item in value:
                try:
                    check(item, ITEMS[field])
                    kept.append(item)
                except Exception as exc:  # a malformed item is dropped and recorded
                    errors.append(_error(exc))
            if errors:
                problems.append(f'{field}: dropped {len(errors)} of {len(value)} ({errors[0]})')
            cap = CAPS.get(field)
            if cap and len(kept) > cap:
                problems.append(f'{field}: kept the first {cap} of {len(kept)}')
                kept = kept[:cap]
            out[field] = kept if kept or field == 'changed_pairs' else None
        else:
            try:
                check(value, WHOLE[field])
                out[field] = value
            except Exception as exc:
                out[field] = None
                problems.append(f'{field}: dropped ({_error(exc)})')
    if problems:
        out['_problems'] = problems
    return out


def parse_realizer(data):
    """The ordinary realization is validated strictly, exactly as the ordinary realizer's; the measurement record
    leniently (lenient)."""
    from arcgen.lib.astra_authoring_contract_v3 import parse_stage
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    if not isinstance(value, dict):
        raise SchemaError('$', 'realization must be an object')
    base, extra = split(value)
    check(base, schemas.REALIZER)
    parse_stage(schemas.REALIZER_STAGE, base)
    return {**base, 'weakness_outputs': lenient(extra, 'training_outputs')}


def parse_repair(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    if not isinstance(value, dict):
        raise SchemaError('$', 'repair must be an object')
    base, extra = split(value)
    check(base, repair_v1.STRICT_SCHEMA)
    repair_v1.parse(base)
    return {**base, 'weakness_outputs': lenient(extra, 'changed_pairs')}


def stage_bindings():
    from arcgen.lib.factory_runtime import StageBinding
    return {REALIZER_STAGE: StageBinding(REALIZER_SCHEMA, parse_realizer, ('realization',)),
            REPAIR_STAGE: StageBinding(REPAIR_SCHEMA, parse_repair, ('targeted_repair',))}


def default_binding():
    """When a run records default readings (science.weakness.default_reading): the default-reading stage, the blind
    solver's schema and parser under its own stage name, a conditional blind call. It is an additional (review-like) stage,
    never an authoring contract stage: the factory keeps blind calls out of candidate contracts, so the batch runtime
    validates its exact request as it does the adjudicator's (runtime.invoke_rendered). Empty otherwise."""
    if not (SCIENCE.get('weakness') or {}).get('default_reading'):
        return {}
    from functools import partial
    from arcgen.lib.factory_runtime import StageBinding
    from arcgen.lib.v4_schemas import OUTPUT_SCHEMAS, parse_stage_output
    return {DEFAULT_STAGE: StageBinding(deepcopy(OUTPUT_SCHEMAS['v4_blind_solver']), partial(parse_stage_output, 'v4_blind_solver'),
                                        ('conditional_blind',))}


DEFAULT_FIELDS = {'TASK_ID', 'PUBLIC_TASK', 'SOLVER_ID'}


def default_templates():
    """The default-reading call's prompt (STRUCTURE_SPEC D): the unchanged blind solver prompt, then one sentence asking
    for the most straightforward answer. No plan and no hints."""
    from .common import HERE as BATCH
    return [load_prompt(BATCH / 'prompts/v4_blind_solver.md'), component('lines/default_reading.md')]


def realizer_components(study):
    """The realizer's addenda for an idea, after the unchanged realizer template (empty: the plain realizer)."""
    names = [s['realizer'] for _, s in _specs(study) if s.get('emits') and s.get('realizer')]
    return ['lines/outputs.realizer.md'] + names if names else []


def emission_problems(study, realization, extra):
    """Shape problems of a realizer's weakness_outputs (recorded, never fatal: the task goes on without them)."""
    if realization.get('status') != 'realized':
        return []
    if extra is None:
        return ['weakness_outputs missing']
    kinds, task, out = emits(study), realization['task'], []
    if 'pairs' in kinds and (extra.get('training_outputs') is None or len(extra['training_outputs']) != len(task['train'])):
        out.append('training_outputs missing or not one per training pair')
    out += [EMIT_FIELDS[k] + ' missing' for k in kinds if k in LEDGERS + KEYED and extra.get(EMIT_FIELDS[k]) is None]
    return out


def _record_emission(runtime, tid, entry):
    """Save one emission on the record (replacing an earlier one from the same call, so a resume is idempotent) and as a
    frozen receipt."""
    from .common import frozen
    frozen(runtime.run_dir / 'weakness' / 'emissions' / (entry['call_id'] + '.json'), entry)
    state = getattr(runtime, 'state', None)
    if state is None:
        return
    with runtime.lock:
        weak = state['semantic_records'][tid].setdefault('weakness', {})
        weak['emissions'] = [e for e in weak.get('emissions', []) if e['call_id'] != entry['call_id']] + [deepcopy(entry)]
        runtime.save()


def _entry(source, call_id, palette, **extra):
    return {'source': source, 'call_id': call_id, 'palette': palette, 'problems': [], 'train': None,
            **{k: None for k in LEDGERS}, **extra}


ASSEMBLY_NODES = 3000  # review B1: a puzzle needing more nodes under most-constrained-first search is not forced


def _node_limit(study):
    s = spec(line_of(study), fence_of(study)) or {}
    return ((s.get('grids') or {}).get('assembly') or {}).get('node_limit', ASSEMBLY_NODES)


def _solve_emitted(entry, study):
    """FORCED's solver, once per emitted assembly and outside the runtime lock (review B1): the result is stored with
    the emission (in the emission's palette), so the checks read it instead of searching again."""
    if entry.get('assembly'):
        entry['assembly_solution'] = solve_assembly(entry['assembly'], _node_limit(study))


def _keyed(entry, extra, task):
    """A structure record is stored with the inputs it refers to (the realization's or the repaired episode's, in the
    emission's palette), so that the tables key each grid by its input; only when one is emitted (other entries keep
    their shape)."""
    if any(extra.get(k) is not None for k in KEYED):
        entry.update({k: deepcopy(extra.get(k)) for k in KEYED})
        entry['inputs'] = {'train': [deepcopy(p['input']) for p in task['train']],
                           'test': [deepcopy(t['input']) for t in task['test']]}


def realizer_emission(runtime, tid, call_id, study, base, extra):
    """The realizer's measurement record (palette original), with the realizer's own episode, so that the first-version
    checks run on what the realizer wrote whatever happened before the first review (review S6)."""
    task = base.get('task') if base.get('status') == 'realized' else None
    entry = _entry('realizer', call_id, 'original')
    if task:
        entry['problems'] = emission_problems(study, base, extra) + list((extra or {}).get('_problems') or [])
        extra = extra or {}
        if extra.get('training_outputs') is not None:
            entry['train'] = [[p['input'], o] for p, o in zip(task['train'], extra['training_outputs'])]
        entry.update({k: deepcopy(extra.get(k)) for k in LEDGERS})
        expected = (base.get('authoring_metadata') or {}).get('expected_test_outputs') or []
        entry['episode'] = {'train': deepcopy(task['train']),
                            'test': [{'input': deepcopy(t['input']), 'output': deepcopy(o)} for t, o in zip(task['test'], expected)]}
        _keyed(entry, extra, task)
        _solve_emitted(entry, study)
    _record_emission(runtime, tid, entry)
    return entry


def repair_emission(runtime, tid, call_id, source_hash, expanded, extra):
    """A repair's re-emitted outputs, in the final palette (the repairer sees the materialised episode): the shortcut on
    every training pair it adds or changes, and the points and ledgers again, whole."""
    entry = _entry('repair', call_id, 'final', source_episode_hash=source_hash)
    realized = (expanded or {}).get('repaired_realization') or {}
    task = realized.get('task') if expanded and expanded.get('status') == 'repaired' else None
    if task and extra:
        entry['problems'] += list(extra.get('_problems') or [])
        train, rows = task['train'], []
        for row in extra.get('changed_pairs') or []:
            k = row['pair_number']
            if not 1 <= k <= len(train):
                entry['problems'].append(f'changed pair {k} is not in the repaired episode')
            elif row.get('shortcut_output') is not None:
                rows.append([train[k - 1]['input'], row['shortcut_output']])
        entry['train'] = rows or None
        entry.update({k: deepcopy(extra.get(k)) for k in LEDGERS})
        _keyed(entry, extra, task)
        state = getattr(runtime, 'state', None)
        study = deepcopy(((state or {}).get('semantic_records') or {}).get(tid, {}).get('study') or {})
        _solve_emitted(entry, study)
    elif task:
        entry['problems'].append('weakness_outputs missing')
    _record_emission(runtime, tid, entry)
    return entry


# ------------------------------------------------------------------ the emitted outputs, in the final palette


def _key(grid):
    return json.dumps(grid, separators=(',', ':'))


def _mapping(record):
    raw = (record.get('palette_treatment') or {}).get('mapping') or {}
    return {int(k): int(v) for k, v in raw.items()}


def _remap(grid, mapping):
    if not mapping or not isinstance(grid, list):
        return grid
    return [[mapping.get(c, c) if isinstance(c, int) else c for c in row] if isinstance(row, list) else row for row in grid]


def _remap_ledger(kind, value, mapping):
    """Colours inside a realizer's ledgers are in the original palette too (point cells, objects' and pieces' colours)."""
    if not mapping or value is None:
        return value
    value = deepcopy(value)
    if kind in ('points', 'overlay_points'):
        for point in value:
            point['cells'] = [[r, c, mapping.get(x, x)] for r, c, x in point['cells']]
    elif kind == 'objects':
        for row in value:
            row['colour'] = mapping.get(row['colour'], row['colour'])
    elif kind == 'assembly':
        for piece in value['pieces']:
            piece['colour'] = mapping.get(piece['colour'], piece['colour'])
    return value


def _remap_solution(solution, mapping):
    if not mapping or not solution or not solution.get('rendering'):
        return solution
    return {**solution, 'rendering': [[cell, mapping.get(colour, colour)] for cell, colour in solution['rendering']]}


def _grid_name(name, inputs):
    """The input a record names (train_2, test_1), or None."""
    m = re.fullmatch(r'\s*(train|test)_(\d+)\s*', name or '')
    rows = (inputs or {}).get(m.group(1)) or [] if m else []
    return rows[int(m.group(2)) - 1] if m and 1 <= int(m.group(2)) <= len(rows) else None


def _merge_keyed(out, e, m):
    """Structure records merged per record and per input (STRUCTURE_SPEC S1): a later emission replaces the grids it gives
    and keeps the others, so a repair re-writes only the pairs it adds or changes."""
    inputs = e.get('inputs') or {}
    for a in e.get('alternatives') or []:
        rec = out['alternatives'].setdefault(a['id'].strip(), {'id': a['id'].strip(), 'train': {}, 'test': {}})
        rec.update(line=a['line'], kind=a['kind'], note=a['note'])
        for part in ('train', 'test'):
            for inp, grid in zip(inputs.get(part) or [], a.get(part) or []):
                if grid is not None:
                    rec[part][_key(_remap(inp, m))] = _remap(grid, m)
    for u in e.get('units') or []:
        inp = _grid_name(u['grid'], inputs)
        if inp is not None:
            out['units'][_key(_remap(inp, m))] = u['labels']
    for v in e.get('surfaces') or []:
        inp = _grid_name(v['grid'], inputs)
        if inp is not None:
            out['surfaces'][_key(_remap(inp, m))] = {**deepcopy(v), 'roles': [
                {'role': r['role'], 'colour': m.get(r['colour'], r['colour'])} for r in v['roles']]}


def tables(record, first=False):
    """The emitted outputs known for this record, in the final palette: training shortcut outputs keyed by input grid (a
    shortcut depends on the input only; later emissions override earlier ones), the latest of each ledger, FORCED's
    stored solver result for that assembly, and the structure records keyed by input (_merge_keyed). first: the
    realizer's emission alone (the first-version checks, review S6)."""
    mapping = _mapping(record)
    out = {'train': {}, **{k: None for k in LEDGERS}, 'assembly_solution': None, **{k: {} for k in KEYED}}
    for e in (record.get('weakness') or {}).get('emissions') or []:
        if first and e.get('source') != 'realizer':
            continue
        m = mapping if e.get('palette') == 'original' else {}
        for inp, grid in e.get('train') or []:
            out['train'][_key(_remap(inp, m))] = _remap(grid, m)
        for kind in LEDGERS:
            if e.get(kind) is not None:
                out[kind] = _remap_ledger(kind, e[kind], m)
        if e.get('assembly') is not None:
            out['assembly_solution'] = _remap_solution(e.get('assembly_solution'), m)
        _merge_keyed(out, e, m)
    return out


def realizer_episode(record):
    """The realizer's own episode in the final palette, or None (records from before it was stored)."""
    entry = next((e for e in (record.get('weakness') or {}).get('emissions') or [] if e.get('source') == 'realizer'
                  and e.get('episode')), None)
    if not entry:
        return None
    m = _mapping(record)
    return {part: [{k: _remap(v, m) for k, v in p.items()} for p in entry['episode'][part]] for part in ('train', 'test')}


def _shape(grid):
    if not isinstance(grid, list) or not grid or not all(isinstance(r, list) for r in grid):
        return None
    widths = {len(r) for r in grid}
    return (len(grid), widths.pop()) if len(widths) == 1 else None


def _groups(cells):
    seen, out = set(), []
    for cell in cells:
        if cell in seen:
            continue
        stack, group = [cell], set()
        seen.add(cell)
        while stack:
            r, c = stack.pop(); group.add((r, c))
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells and n not in seen:
                    seen.add(n); stack.append(n)
        out.append(group)
    return out


def difference(a, b):
    """Cells, 4-connected regions and the largest region where two grids differ (same_shape False otherwise)."""
    if _shape(a) is None or _shape(a) != _shape(b):
        return {'same_shape': False}
    cells = {(r, c) for r, row in enumerate(a) for c, x in enumerate(row) if x != b[r][c]}
    regions = [len(g) for g in _groups(cells)]
    return {'same_shape': True, 'cells': len(cells), 'regions': len(regions), 'largest': max(regions, default=0)}


def point_cells(point, truth):
    """The point's cells where its alternative differs from the truth, as [row, col, alternative colour] (cells outside
    the answer or with an impossible colour are dropped)."""
    if not _shape(truth):
        return []
    return [[r, c, x] for r, c, x in point['cells'] if r < len(truth) and c < len(truth[0]) and 0 <= x <= 9 and truth[r][c] != x]


def alternative(point, truth):
    """The answer with the shortcut applied at this point only."""
    grid = deepcopy(truth)
    for r, c, x in point_cells(point, truth):
        grid[r][c] = x
    return grid


# ------------------------------------------------------------------ checks


def _open_ground(pair, grid):
    """A small refutation on open ground (v2 §5.1): within two cells of each differing cell, nothing else changes between
    the pair's input and output."""
    inp, out = pair['input'], pair['output']
    if _shape(inp) != _shape(out) or _shape(grid) != _shape(out):
        return None
    diff = {(r, c) for r, row in enumerate(out) for c, x in enumerate(row) if x != grid[r][c]}
    near = {(r + i, c + j) for r, c in diff for i in range(-2, 3) for j in range(-2, 3)} - diff
    return all(inp[r][c] == out[r][c] for r, c in near if 0 <= r < len(out) and 0 <= c < len(out[0]))


def _changed(pair):
    """The cells the true rule changes in a pair (all output cells when input and output differ in size)."""
    inp, out = pair['input'], pair['output']
    if _shape(inp) != _shape(out):
        return len(out) * len(out[0])
    return sum(a != b for ra, rb in zip(inp, out) for a, b in zip(ra, rb))


def _pairs(episode, table, rule):
    """The shortcut on the training pairs: SETTLE's default reproduces all but its allowed number of outputs (one or two,
    fewer than half; the two-pair form: exactly one); COINCIDE's rival reproduces every output. The in-pair-minority form
    (draft 4 §6.4) has its own check: in every pair the default misses, it still gets most of what the rule changes.
    With open_ground (SETTLE), a refuting difference of three cells or fewer must sit on open ground."""
    missing, differing = [], []
    for i, p in enumerate(episode['train'], 1):
        grid = table.get(_key(p['input']))
        if grid is None:
            missing.append(i)
        elif grid != p['output']:
            d = difference(grid, p['output'])
            if d.get('same_shape') and d['cells'] <= 3:
                d['open_ground'] = _open_ground(p, grid)
            if rule.get('in_pair_minority'):
                d['minority_of_changes'] = d.get('same_shape') and 2 * d['cells'] < max(1, _changed(p))
            differing.append({'pair': i, **d})
    k, n = len(differing), len(episode['train'])
    ok = rule['min'] <= k <= rule['max'] and (not rule.get('minority') or 2 * k < n)
    if rule.get('open_ground'):  # SETTLE (DESIGN §3, v2 §5.1): a refutation of three cells or fewer only on open ground
        ok = ok and not any(d.get('open_ground') is False for d in differing)
    if rule.get('in_pair_minority'):
        ok = ok and all(d.get('minority_of_changes') for d in differing)
    return {'missing': missing, 'differing_pairs': differing, 'status': 'incomplete' if missing else 'pass' if ok else 'fail'}


def _scaled(rung):
    return bool(rung) and rung != 'R1'


def _points(episode, points, rung, rule):
    """The declared points (DESIGN §6.1): each inside its test's answer, with at least one cell where its alternative
    differs from the truth, within the line's counts (SETTLE at most two settled points; DEPART at most one situation, two
    in its relaxed form, in test 1 only; COINCIDE B's one or two completions). PRECISE's decision points and TRACE's
    junctions also meet their ledger counts."""
    if not points:
        return {'status': 'incomplete'}
    tests = episode['test']
    rows, per_test = [], Counter()
    for p in points:
        truth = tests[p['test'] - 1]['output'] if 1 <= p['test'] <= len(tests) else None
        diff = point_cells(p, truth) if truth else []
        rows.append({'id': p['id'], 'kind': p['kind'], 'test': p['test'], 'differing_cells': len(diff),
                     'listed_cells': len(p['cells'])})
        per_test[p['test']] += 1
    value = {'points': rows, 'per_test': {str(k): v for k, v in sorted(per_test.items())}}  # JSON-stable keys
    ok = all(r['differing_cells'] for r in rows)
    if 'tests' in rule:
        ok = ok and set(per_test) <= set(rule['tests'])
    if 'max_per_test' in rule:
        ok = ok and max(per_test.values()) <= rule['max_per_test']
    if 'max_kinds_per_test' in rule:  # SETTLE (screen hole 1): settled points are kinds; a point has one entry per place
        kinds_per_test = Counter()
        for test, kind in {(p['test'], (p.get('kind') or '').strip().lower()) for p in points}:
            kinds_per_test[test] += 1
        value['kinds_per_test'] = {str(k): v for k, v in sorted(kinds_per_test.items())}
        ok = ok and max(kinds_per_test.values()) <= rule['max_kinds_per_test']
    if 'min_places' in rule:  # SETTLE's single-cell form (v2 §5.1): the test needs the settled point in two or more places
        places = {}
        for p in points:
            if 1 <= p['test'] <= len(tests):
                places.setdefault(p['test'], set()).update((r, c) for r, c, _ in point_cells(p, tests[p['test'] - 1]['output']))
        value['places'] = {str(k): len(_groups(v)) for k, v in places.items()}
        ok = ok and any(len(_groups(v)) >= rule['min_places'] for v in places.values())
    if 'max_total' in rule:
        ok = ok and len(points) <= rule['max_total']
    if 'decisions' in rule:  # PRECISE: counts, kinds and the pairs that teach each kind
        r = rule['decisions']
        train = len(episode['train'])
        kinds = {}
        for p in points:
            kinds.setdefault(p['kind'].strip().lower(), set()).update(
                t for t in p['taught_by'] if re.fullmatch(r'train_(\d+)', t) and int(t[6:]) <= train)
        floor = r['points_min_scaled'] if _scaled(rung) else r['points_min']
        value.update(kinds={k: sorted(v) for k, v in kinds.items()}, points_min=floor)
        ok = ok and min(per_test.get(k, 0) for k in range(1, len(tests) + 1)) >= floor and r['kinds'][0] <= len(kinds) <= r['kinds'][1] \
            and all(len(v) >= r['taught_min'] for v in kinds.values())
    if 'junctions' in rule:  # TRACE: at least four junction decisions; at most about 60 at R1-R2 (local-cue points apart)
        r = rule['junctions']
        junctions = sum('junction' in (p.get('kind') or '').lower() for p in points)
        value['junctions'] = junctions
        ok = ok and junctions >= r['min'] and (rung in ('R3', 'R4') or junctions <= r['max_unscaled'])
    value['status'] = 'pass' if ok else 'fail'
    return value


OWN = ('colour', 'size', 'shape', 'region')


def _objects(rows, rule=None):
    """RELATION's object ledger (v2 §3.3): no simple own-feature predictor (one own feature, or a pair of them) maps every
    training object's features to its outcome and also gives every test object's outcome, and some training grid holds a
    minimal contrast (two look-alikes with the same colour, size and shape and different outcomes). In the across-grids
    stratum (v2 §12, rule contrast "across") the minimal contrast may lie between two training grids."""
    if not rows:
        return {'status': 'incomplete'}
    norm = lambda v: v.strip().lower() if isinstance(v, str) else v
    train = [r for r in rows if r['grid'].startswith('train')]
    test = [r for r in rows if r['grid'].startswith('test')]
    leaks = []
    for combo in [(f,) for f in OWN] + [(a, b) for i, a in enumerate(OWN) for b in OWN[i + 1:]]:
        key = lambda r: tuple(norm(r[f]) for f in combo)
        table, consistent = {}, True
        for r in train:
            if table.setdefault(key(r), norm(r['outcome'])) != norm(r['outcome']):
                consistent = False; break
        if consistent and test and all(table.get(key(r)) == norm(r['outcome']) for r in test):
            leaks.append('+'.join(combo))
    contrast, contrast_grids = False, 0
    for grid in {r['grid'] for r in train}:
        seen, here = {}, False
        for r in (x for x in train if x['grid'] == grid):
            k = tuple(norm(r[f]) for f in ('colour', 'size', 'shape'))
            here = here or (k in seen and seen[k] != norm(r['outcome']))
            seen.setdefault(k, norm(r['outcome']))
        contrast, contrast_grids = contrast or here, contrast_grids + here
    outcomes = {}
    for r in train:
        outcomes.setdefault(tuple(norm(r[f]) for f in ('colour', 'size', 'shape')), set()).add(norm(r['outcome']))
    across = any(len(v) > 1 for v in outcomes.values())
    allowed = contrast or ((rule or {}).get('contrast') == 'across' and across)
    value = {'objects': len(rows), 'test_objects': len(test), 'own_feature_predictors': leaks, 'minimal_contrast': contrast,
             'across_grids_contrast': across}
    if (rule or {}).get('contrast_grids'):  # RELEVANCE (S8): the minimal contrast inside at least this many training grids
        value.update(contrast_grids=contrast_grids, contrast_grids_min=rule['contrast_grids'])
        allowed = contrast_grids >= rule['contrast_grids']
    return {**value, 'status': 'pass' if not leaks and allowed and test else 'fail'}


class _Limit(Exception):
    pass


def _orientations(cells, transforms):
    def norm(cs):
        r0, c0 = min(r for r, _ in cs), min(c for _, c in cs)
        return tuple(sorted((r - r0, c - c0) for r, c in cs))
    ops = [lambda r, c: (r, c)]
    if transforms in ('turns', 'turns_and_mirrors'):
        ops = [lambda r, c: (r, c), lambda r, c: (c, -r), lambda r, c: (-r, -c), lambda r, c: (-c, r)]
    if transforms == 'turns_and_mirrors':
        ops = ops + [lambda r, c, f=f: f(r, -c) for f in ops]
    return {norm([op(r, c) for r, c in cells]) for op in ops}


def solve_assembly(a, limit):
    """Exact cover of the target region by the pieces (each once, under the allowed transforms), the anchor's placement
    fixed first (it may be turned or reflected whatever the other pieces may do): the number of distinct coloured
    solutions (up to 2) and, for the first, the branch points on its path when the most constrained cell or piece is
    filled first. Knuth's Algorithm X on sets (review B1: each node costs about the rows it touches, not a scan of every
    placement), identical pieces interchanged only once. None for constraint types this checker does not support; past
    `limit` nodes, solutions None (recorded as unchecked)."""
    if a['constraint'] != 'cover_region':
        return None
    target = {tuple(c) for c in a['target']}
    anchor_piece = next((p for p in a['pieces'] if p['id'] == a['anchor']['piece']), None)
    anchor = frozenset(tuple(c) for c in a['anchor']['cells'])
    if anchor_piece is None:
        return {'solutions': 0, 'branch_points': None, 'reason': 'anchor names no piece'}
    if sum(len(p['cells']) for p in a['pieces']) != len(target):
        return {'solutions': 0, 'branch_points': None, 'reason': 'piece cells and target differ in size'}
    norm = lambda cs: tuple(sorted((r - min(x for x, _ in cs), c - min(y for _, y in cs)) for r, c in cs))
    if not anchor <= target or norm(anchor) not in _orientations(anchor_piece['cells'], 'turns_and_mirrors'):
        return {'solutions': 0, 'branch_points': None, 'reason': 'anchor is not its piece, turned or reflected, in the target'}
    free = target - anchor
    pieces = [p for p in a['pieces'] if p is not anchor_piece]
    kinds, kind_of = {}, []
    for p in pieces:
        shapes = frozenset(_orientations(p['cells'], a['transforms']))
        kind_of.append(kinds.setdefault((shapes, p['colour']), len(kinds)))
    columns = {('cell', c): set() for c in sorted(free)}
    columns.update({('piece', i): set() for i in range(len(pieces))})
    rows = {}
    for i, p in enumerate(pieces):
        for shape in sorted(_orientations(p['cells'], a['transforms'])):
            r0, c0 = shape[0]
            for tr, tc in sorted(free):
                placed = frozenset((tr - r0 + r, tc - c0 + c) for r, c in shape)
                if placed <= free and (i, placed) not in rows:
                    rows[(i, placed)] = [('piece', i)] + [('cell', x) for x in sorted(placed)]
    for row, cols in rows.items():
        for col in cols:
            columns[col].add(row)
    order = {row: k for k, row in enumerate(rows)}  # a fixed order, so the search and its counts are reproducible

    def select(row):
        removed = []
        for col in rows[row]:
            for other in columns[col]:
                for k in rows[other]:
                    if k != col:
                        columns[k].discard(other)
            removed.append(columns.pop(col))
        return removed

    def deselect(row, removed):
        for col in reversed(rows[row]):
            columns[col] = removed.pop()
            for other in columns[col]:
                for k in rows[other]:
                    if k != col:
                        columns[k].add(other)

    solutions, nodes, first_branches, chosen = [], [0], [None], []
    base = frozenset((cell, anchor_piece['colour']) for cell in anchor)

    def search(branches):
        nodes[0] += 1
        if nodes[0] > limit:
            raise _Limit()
        if not columns:
            render = base | frozenset((cell, pieces[i]['colour']) for i, placed in chosen for cell in placed)
            if render not in solutions:
                solutions.append(render)
                if first_branches[0] is None:
                    first_branches[0] = branches
            return
        col = min(columns, key=lambda c: (len(columns[c]), c))
        options, seen = [], set()
        for row in sorted(columns[col], key=order.get):
            key = (kind_of[row[0]], row[1])  # identical pieces in the same place are one option
            if key not in seen:
                seen.add(key); options.append(row)
        for row in options:
            removed = select(row); chosen.append(row)
            search(branches + (len(options) > 1))
            chosen.pop(); deselect(row, removed)
            if len(solutions) >= 2:
                return
    try:
        search(0)
    except _Limit:
        return {'solutions': None, 'branch_points': None, 'reason': f'node limit {limit}'}
    return {'solutions': len(solutions), 'branch_points': first_branches[0], 'nodes': nodes[0],
            'rendering': sorted([list(cell), colour] for cell, colour in solutions[0]) if solutions else None}


def _assembly(episode, a, rule, solution=None):
    """FORCED's checks (v2 §3.4): exactly one solution, reached with at most two branch points, equal to the test's
    answer on the target cells; types the checker does not support, and searches past the node limit, are recorded as
    unchecked. The solver result stored with the emission is used when present (review B1)."""
    if not a:
        return {'status': 'incomplete'}
    result = solution if solution is not None else solve_assembly(a, rule['node_limit'])
    if result is None or result.get('solutions') is None:
        return {'constraint': a['constraint'], **(result or {}), 'status': 'unchecked'}
    value = {'constraint': a['constraint'], 'pieces': len(a['pieces']), **{k: v for k, v in result.items() if k != 'rendering'}}
    answer = episode['test'][0]['output'] if episode['test'] else None
    if result.get('rendering') and _shape(answer):
        inside = [(r, c) for (r, c), _ in result['rendering'] if r < len(answer) and c < len(answer[0])]
        target_colours = {answer[r][c] for r, c in inside}
        if len(inside) < len(result['rendering']):
            value['matches_answer'] = False  # the solution leaves the answer's grid
        elif target_colours <= {p['colour'] for p in a['pieces']}:
            value['matches_answer'] = all(answer[r][c] == colour for (r, c), colour in result['rendering'])
        else:  # screen hole 2: the ledger gives each piece one colour, but many pieces carry several; compare only cover
            value['matches_answer'] = None
            value['colours_compared'] = False
    value['status'] = ('pass' if result['solutions'] == 1 and result['branch_points'] <= rule['branch_max']
                       and value.get('matches_answer') is not False else 'fail')
    return value


def _background(plan, grid):
    public, _ = _sections(plan)
    overview = (public or {}).get('episode_overview') or {}
    role = (overview.get('palette_contract') or {}).get('background_role')
    colour = next((r['color'] for r in overview.get('functional_color_roles') or [] if r['role'] == role), None)
    if colour is not None:
        return colour
    counts = Counter(c for row in grid for c in row)
    return min(counts, key=lambda c: (-counts[c], c))


def _alias(plan, name):
    public, _ = _sections(plan)
    return next((r['color'] for r in ((public or {}).get('episode_overview') or {}).get('functional_color_roles') or []
                 if r['role'] == name), None)


def _things(grid, ground):
    return len(_groups({(r, c) for r, row in enumerate(grid) for c, x in enumerate(row) if x != ground}))


def regularity(text, plan):
    """A checkable regularity stated in words (PROCESS's conserved quantity, COINCIDE's broken regularity): a function of
    (input, output) -> bool, or None for a statement this checker does not read. The forms are the planner texts' own."""
    t = (text or '').lower()
    if re.search(r'leaves? the grid|clipped|movers', t):  # cells past the border, or movers, are not visible in the grids
        return None
    m = re.search(r'number of @([a-z][a-z0-9_]*) cells', t)
    if m:
        colour = _alias(plan, m[1])
        if colour is None:
            return None
        count = lambda g: sum(x == colour for row in g for x in row)
        return lambda i, o: count(i) == count(o)
    if 'number of non-background cells' in t:
        return lambda i, o: (sum(x != _background(plan, i) for row in i for x in row)
                             == sum(x != _background(plan, i) for row in o for x in row))
    if 'number of things' in t:
        return lambda i, o: _things(i, _background(plan, i)) == _things(o, _background(plan, i))
    m = re.search(r'@([a-z][a-z0-9_]*) cells stay where they are', t)
    if m:
        colour = _alias(plan, m[1])
        if colour is None:
            return None
        where = lambda g: {(r, c) for r, row in enumerate(g) for c, x in enumerate(row) if x == colour}
        return lambda i, o: where(i) == where(o)
    if 'mirror-symmetric left to right' in t:
        return lambda i, o: all(row == row[::-1] for row in o)
    if 'mirror-symmetric top to bottom' in t:
        return lambda i, o: o == o[::-1]
    if "the input's size" in t:
        return lambda i, o: _shape(i) == _shape(o)
    return None


def _conservation(record, episode):
    """PROCESS's conserved quantity (v2 §3.2), stated in the plan's "Conserved:" invariant, holds between input and output
    in every pair and the test; 'unchecked' for frames over time or a statement the checker does not read."""
    plan = record.get('concrete_plan')
    text = invariant(plan, 'Conserved:')
    if not text:
        return {'status': 'incomplete'}
    holds = regularity(text, plan)
    if holds is None:
        return {'statement': text, 'status': 'unchecked'}
    pairs = [holds(p['input'], p['output']) for p in episode['train']]
    tests = [holds(p['input'], p['output']) for p in episode['test']]
    return {'statement': text, 'pairs': pairs, 'tests': tests, 'status': 'pass' if all(pairs + tests) else 'fail'}


def conservation_break(record, sample, test=1):
    """Whether a wrong sample breaks PROCESS's code-checked conserved quantity (DESIGN §6.1, the line's bite), or None."""
    episode, plan = record.get('complete_task'), record.get('concrete_plan')
    holds = regularity(invariant(plan, 'Conserved:'), plan) if episode else None
    if holds is None or not 1 <= test <= len(episode['test']) or _shape(sample) is None:
        return None
    return not holds(episode['test'][test - 1]['input'], sample)


def _parts(grid):
    """Parts in a grid, measured as 8-connected components of cells other than its most common colour (a proxy: touching
    things count once, and a multi-coloured thing stays one part)."""
    ground = Counter(c for row in grid for c in row).most_common(1)[0][0]
    seen, parts = set(), 0
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour == ground or (r, c) in seen:
                continue
            parts += 1
            stack = [(r, c)]; seen.add((r, c))
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < len(grid) and 0 <= nx < len(grid[ny]) and (ny, nx) not in seen and grid[ny][nx] != ground:
                            seen.add((ny, nx)); stack.append((ny, nx))
    return parts


def _growth(episode, rule):
    """DEPART's growth rule on the final grids (draft 4: one rule everywhere, as the plan check and the texts state it):
    every test input at most area_max times the area of the largest training input and, when parts_max is set, at most
    parts_max times its most parts (the parts are a component count, so that half is a measured proxy)."""
    largest = max(len(p['input']) * len(p['input'][0]) for p in episode['train'])
    ratio = max(len(p['input']) * len(p['input'][0]) for p in episode['test']) / largest
    value = {'ratio': round(ratio, 3), 'area_max': rule['area_max']}
    ok = ratio <= rule['area_max'] + 1e-9
    if rule.get('parts_max'):
        most = max(1, max(_parts(p['input']) for p in episode['train']))
        parts = max(_parts(p['input']) for p in episode['test']) / most
        value.update(parts_ratio=round(parts, 3), parts_max=rule['parts_max'])
        ok = ok and parts <= rule['parts_max'] + 1e-9
    return {**value, 'status': 'pass' if ok else 'fail'}


def _killer(record, episode, k, points):
    """COINCIDE's coincidence killer at test 1 (v2 §4; draft 4: no second-test killer, the eval shows one test input per
    prompt). no_single_answer: B's two completions (points) disagree and both differ from the answer; broken_regularity:
    the stated regularity holds on every pair and on the answer, and fails on B's answer; finished_instance: every input
    holds a finished site, the answer keeps the test's as drawn, and B's answer differs from the answer."""
    if not k:
        return {'status': 'incomplete'}
    test = episode['test'][0]
    rivals = [alternative(p, test['output']) for p in points or [] if p['test'] == 1]
    value, ok = {'family': k['family'], 'completions': len(rivals)}, None
    if k['family'] == 'no_single_answer':
        value['completions_disagree'] = len(rivals) == 2 and rivals[0] != rivals[1]
        value['completions_differ_from_answer'] = bool(rivals) and all(r != test['output'] for r in rivals)
        ok = value['completions_disagree'] and value['completions_differ_from_answer']
    elif k['family'] == 'broken_regularity':
        holds = regularity(k.get('invariant'), record.get('concrete_plan'))
        if holds is None or not rivals:
            value.update(regularity='unread' if holds is None else 'no rival answer')
        else:
            value['regularity_on_pairs'] = all(holds(p['input'], p['output']) for p in episode['train'])
            value['regularity_on_answer'] = holds(test['input'], test['output'])
            value['regularity_broken_by_rival'] = not holds(test['input'], rivals[0])
            ok = value['regularity_on_pairs'] and value['regularity_on_answer'] and value['regularity_broken_by_rival']
    elif k['family'] == 'finished_instance':
        # screen hole 2: the finished instance is drawn in the test input and both readings keep it, so the check is that
        # every input holds one, that the answer leaves the test's as drawn (it is finished), and that the rival answers
        # differently elsewhere; whether the drawn instance contradicts the rival is for the adjudicator and the raters
        sites = k.get('finished_sites') or []
        n = len(episode['train']) + len(episode['test'])
        value['sites_in_every_input'] = len(sites) == n and all(sites)
        site = [tuple(c) for c in sites[len(episode['train'])]] if len(sites) > len(episode['train']) else []
        drawn, answer = test['input'], test['output']
        inside = [(r, c) for r, c in site if _shape(drawn) == _shape(answer) and r < len(drawn) and c < len(drawn[0])]
        value['site_kept_by_answer'] = bool(inside) and all(drawn[r][c] == answer[r][c] for r, c in inside)
        value['rival_differs'] = bool(rivals) and all(x != answer for x in rivals)
        value['contradiction'] = 'not machine-checked'
        ok = value['sites_in_every_input'] and value['site_kept_by_answer'] and value['rival_differs']
    value['status'] = 'incomplete' if ok is None and not rivals else 'pass' if ok else 'fail'
    return value


def size_check(record, episode):
    """R4's checks (v2 §6, as corrected 2026-10-20): the test output at R4's cell floor (science.rungs.R4), a training pair
    of at least min(half the test output, 200) cells,
    computed cells at least 30% of the test output, the output equal to the input in at most 70% of cells, and the
    compressibility floor (row run-length count at most about the eval's 90th percentile), from which a declared regular
    computed field is exempt. Recorded only."""
    if (record.get('study') or {}).get('rung') != 'R4' or not episode:
        return None
    out = max((p['output'] for p in episode['test']), key=lambda g: len(g) * len(g[0]))
    inp = next(p['input'] for p in episode['test'] if p['output'] is out)
    cells = len(out) * len(out[0])
    same = sum(a == b for ra, rb in zip(inp, out) for a, b in zip(ra, rb)) if _shape(inp) == _shape(out) else 0
    runs = sum(1 + sum(1 for a, b in zip(row, row[1:]) if a != b) for row in out)
    train = max(max(len(p['input']) * len(p['input'][0]), len(p['output']) * len(p['output'][0])) for p in episode['train'])
    regular = bool(invariant(record.get('concrete_plan'), 'Regular field:'))
    need = min(cells / 2, R4_TRAIN_CELLS)
    floor = ((SCIENCE.get('rungs') or {}).get('R4') or {}).get('output_cells', 800)
    value = {'test_output_cells': cells, 'output_cells_min': floor, 'row_runs': runs, 'row_runs_max': R4_ROW_RUNS,
             'regular_field_exempt': regular, 'computed_share': round(1 - same / cells, 3), 'largest_training_cells': train,
             'training_cells_needed': need}
    value['status'] = ('pass' if cells >= floor and (runs <= R4_ROW_RUNS or regular) and value['computed_share'] >= R4_COMPUTED
                       and same / cells <= R4_COPIED and train >= need else 'fail')
    return value


def episode_check(record, episode, table=None):
    """The mechanical checks of one episode version (DESIGN §5.1, §6.1; v2 §2-§6), per line (under its relaxed fence) and
    overlay, as the registry says. Missing outputs give 'incomplete', an unsupported statement 'unchecked'; never a
    failure of the task."""
    from . import structures
    study = record.get('study') or {}
    table = table or tables(record)
    out = {}
    for part, name, fence in (('line', line_of(study), fence_of(study)), ('overlay', overlay_of(study), None),
                              ('structure', structure_of(study), None)):
        s = spec(name, fence) or {}
        kinds = tuple(s.get('emits') or ()) + tuple(s.get('computed') or ())
        if not kinds:
            continue
        value, grids = {'name': name, 'fence': fence}, s.get('grids') or {}
        if 'pairs' in kinds:
            value['pairs'] = _pairs(episode, table['train'], grids['pairs'])
        for kind in ('points', 'overlay_points'):
            if kind in kinds:
                value['points'] = _points(episode, table[kind], study.get('rung'), grids.get(kind) or {})
        if 'objects' in kinds:
            value['objects'] = _objects(table['objects'], grids.get('objects'))
        if 'assembly' in kinds:
            value['assembly'] = _assembly(episode, table['assembly'], grids['assembly'], table.get('assembly_solution'))
        if 'killer' in kinds:
            value['killer'] = _killer(record, episode, table['killer'], table['points'])
        if 'conservation' in kinds:
            value['conservation'] = _conservation(record, episode)
        if 'growth' in kinds:
            value['growth'] = _growth(episode, grids['growth'])
        if 'alternatives' in kinds:  # a structure line's own record (S1), checked per record
            d = declared(name, record.get('concrete_plan'))
            declared_ = (d.get('shortcut_counts') or {}).get(s.get('shortcut'))
            value['alternatives'] = structures.alternatives(name, grids['alternatives'], episode, table, declared_,
                                                            form=form(name, d)['label'])
        if 'frame' in kinds:  # SHAPE: computed from the grids alone
            value['frame'] = structures.frame(episode, grids['frame'])
        parts = [value[k]['status'] for k in ('pairs', 'points', 'objects', 'assembly', 'killer', 'conservation', 'growth',
                                              'alternatives', 'frame') if k in value]
        value['status'] = ('incomplete' if 'incomplete' in parts else 'fail' if 'fail' in parts
                           else 'unchecked' if 'unchecked' in parts else 'pass')
        out[part] = value
    return out


# ------------------------------------------------------------------ capture


def point_capture(points, truth, sample):
    """DESIGN §6.1, per point of one test: eligible when the sample (right shape, wrong) differs from the truth on any of
    the point's cells; captured when it equals the point's alternative on at least 80% of them."""
    out = []
    if _shape(sample) is None or _shape(sample) != _shape(truth) or sample == truth:
        return out
    for p in points:
        cells = point_cells(p, truth)
        if not cells:
            continue
        eligible = any(sample[r][c] != truth[r][c] for r, c, _ in cells)
        share = sum(sample[r][c] == x for r, c, x in cells) / len(cells)
        out.append({'id': p['id'], 'eligible': eligible, 'captured': eligible and share >= CAPTURE, 'share': round(share, 3)})
    return out


PARTS = ('line', 'overlay', 'structure')


def on_points(record, response, episode=None, table=None):
    """The parts ('line', 'overlay') whose declared points a blind answer lands on (any point of any test captured, or the
    answer equal to a point's alternative): the routing signal (DESIGN §5.2)."""
    return sorted({r['part'] for r in point_results(record, response, episode, table) if r['captured'] or r['equals_alternative']},
                  key=PARTS.index)


def point_results(record, response, episode=None, table=None):
    """One blind answer against every declared point (review S11): per point its part, test, whether the answer errs on
    the point's cells (eligible), the share of them equal to the alternative and capture at 80% (DESIGN §6.1), and
    whether the answer equals that point's alternative exactly (the definition the §6 clean-task and Sol criteria use)."""
    episode = episode or record.get('complete_task')
    if not episode or not isinstance(response, dict):
        return []
    table = table or tables(record)
    answers = response.get('predicted_test_outputs') or []
    out = []
    for part, kind in (('line', 'points'), ('overlay', 'overlay_points')):
        for k, pair in enumerate(episode['test'], 1):
            mine = [p for p in table[kind] or [] if p['test'] == k]
            if k > len(answers) or not mine:
                continue
            captured = {r['id']: r for r in point_capture(mine, pair['output'], answers[k - 1])}
            for p in mine:
                row = captured.get(p['id'], {'eligible': False, 'captured': False, 'share': None})
                out.append({'part': part, 'id': p['id'], 'test': k, 'eligible': row['eligible'], 'captured': row['captured'],
                            'share': row['share'], 'equals_alternative': answers[k - 1] == alternative(p, pair['output'])})
    structure = structure_of(record.get('study'))
    if structure:  # the structure line's whole-grid records (S1, S9), per record; SHAPE against its naive frames
        from . import structures
        out += structures.results(structure, episode, table, answers)
    return out


def capture_route(record, blind):
    """The part both of the first two current blind calls land on (DESIGN §5.2: routed to a clarifying repair rather than
    clearance), or None. A part without a clarifying note (JUDGE: both readings fit every pair by design, so two answers
    on the rival are a consensus signal, not a teaching defect) takes the normal adjudication instead."""
    if len(blind) < 2 or not tagged(record.get('study')) or is_decoy(record):
        return None
    study = record.get('study')
    named = {'line': line_of(study), 'overlay': overlay_of(study), 'structure': structure_of(study)}
    first, second = (on_points(record, row['response']) for row in blind[:2])
    return next((part for part in first if part in second and spec(named[part]).get('clarify')), None)


def clarify(record, part, blind):
    """(instruction, challenges, evidence) of a clarifying repair for a task whose two blind calls took a shortcut. The
    line's and overlay's repair notes are appended later, as for every repair of the task (runtime.call_review)."""
    study = record.get('study') or {}
    name = {'line': line_of, 'overlay': overlay_of, 'structure': structure_of}[part](study)
    rule = declared_shortcut(name, record.get('concrete_plan')) or ''
    pairs = []
    if part == 'line' and 'pairs' in spec(name).get('emits', ()):
        found = (episode_check(record, record['complete_task']).get('line') or {}).get('pairs') or {}
        pairs = ['train_' + str(p['pair']) for p in found.get('differing_pairs') or []]
    if part == 'structure':  # the record both blind calls landed on: its rule, and for ELIMINATE its killer pair
        rule, pairs = _structure_shortcut(record, blind, name, rule)
    instruction = body(spec(name).get('clarify') or 'lines/clarify_note.md').replace(
        '{SHORTCUT}', (rule.strip().rstrip('.') or 'the declared shortcut') + '.')
    challenge = {'concern_id': 'shortcut-capture:' + part, 'competing_rule': '',
                 'public_teaching_issue': ('Both blind solvers answered the test with the shortcut the plan names ("'
                                           + rule[:1500] + '"); the evidence against it must be plain to a person '
                                           'looking at the grids.'),
                 'training_pair_ids': pairs}
    evidence = {'route': 'shortcut_capture', 'shortcut': part, 'name': name, 'shortcut_rule': rule,
                'blind_rules': [row['response'].get('primary_inferred_rule') for row in blind[:2]]}
    return instruction, [challenge], evidence


def _structure_shortcut(record, blind, name, rule):
    """(rule text, killer pairs) for a structure capture: the record both calls landed on (the first by id), named by its
    note (the rule in a few words); ELIMINATE adds that rival's killer pair; SHAPE names the naive frames."""
    hits = [{r['id'] for r in point_results(record, row['response']) if r['part'] == 'structure'
             and (r['captured'] or r['equals_alternative'])} for row in blind[:2]]
    common = sorted(hits[0] & hits[1]) if len(hits) == 2 else []
    if name == 'SHAPE':
        return 'an output frame of a naive size: the test input\'s size, a training output\'s size or the largest object\'s box', []
    alt = (tables(record)['alternatives'] or {}).get(common[0]) if common else None
    if not alt:
        return rule, []
    text = (alt.get('note') or '').strip() or (alt.get('kind') or '').strip()  # the record's rule in a few words
    check = ((episode_check(record, record['complete_task']).get('structure') or {}).get('alternatives') or {})
    killer = next((r.get('killer') for r in check.get('records') or [] if r.get('id') == alt['id']), None)
    return text or rule, (['train_' + str(killer)] if killer else [])


# ------------------------------------------------------------------ review policies and repair notes


def _present(s, record):
    """Whether a line's, overlay's or structure line's invariant is in the plan (and is not its opt-out; a structure
    line's opt-out is its own invariant, "Structure: none")."""
    plan = record.get('concrete_plan')
    inv = invariant(plan, (s or {}).get('invariant'))
    opt = (s or {}).get('opt_out') or {}
    if opt.get('field') == 'invariants' and invariant(plan, opt.get('prefix')):
        return False
    return bool(inv) and not (opt.get('field') == 'invariant' and _starts(inv, opt.get('prefix')))


def review(record, blind):
    """Which review policies apply to this episode version: the line's and the overlay's, each when its invariant is in the
    plan. policy_needs_right_answer (DEPART) keeps the departure package's condition: at least one current blind call gave
    the intended answer. policy_needs_check (COINCIDE) applies the policy only when the line's checks pass on this version
    (its killer verified): otherwise a rival that fits every pair stays comparable. Returns (names, reason); (None, None)
    for an idea without any policy."""
    specs = [(n, s) for n, s in _specs(record.get('study')) if s.get('policy')]
    if not specs or is_decoy(record):  # a JUDGE decoy is adjudicated as an ordinary episode (review S3)
        return None, None
    if not record.get('complete_task'):
        return [], 'no_episode'
    names = [n for n, s in specs if _present(s, record)]
    if not names:
        return [], 'no_invariant'
    expected = [p['output'] for p in record['complete_task']['test']]
    reasons = []
    if any(spec(n).get('policy_needs_right_answer') for n in names) and not any(
            row['response']['predicted_test_outputs'] == expected for row in blind):
        names = [n for n in names if not spec(n).get('policy_needs_right_answer')]
        reasons.append('no_right_answer')
    if any(spec(n).get('policy_needs_check') for n in names):
        checks = episode_check(record, record['complete_task'])
        failed = [n for n in names if spec(n).get('policy_needs_check')
                  and (checks.get(part_of(n)) or {}).get('status') != 'pass']
        names = [n for n in names if n not in failed]
        reasons += ['check_not_passed'] if failed else []
    return (names, None) if names else ([], reasons[0] if reasons else 'no_invariant')


def withholds(names):
    """Whether the adjudicator judges these policies without the plan's arguments (the departure package's withholding)."""
    return any(spec(n).get('withhold_arguments') for n in names or ())


def policy_text(names):
    return '\n\n'.join(body(spec(n)['policy']) for n in names)


def _shortcut_words(name):
    prefix = (spec(name) or {}).get('shortcut')
    return ('the first strongest_simple_shortcuts entry' if prefix == 'top' else
            f'the strongest_simple_shortcuts entry whose rule begins "{prefix}"' if prefix else 'the shortcut the line names')


def emit_note(study):
    """Which outputs a repair re-emits: the note's opening, then only the bullets for this idea's fields."""
    fields = {EMIT_FIELDS[k] for k in emits(study)}
    if not fields:
        return None
    kept = [x for x in body('lines/emit.repair_note.md').splitlines() if not x.startswith('- ') or x[2:].split(':', 1)[0] in fields]
    return ('\n'.join(kept).replace('{SHORTCUT}', _shortcut_words(line_of(study)))
            .replace('{OVERLAY_SHORTCUT}', _shortcut_words(overlay_of(study))))


def is_decoy(record):
    return bool((record.get('study') or {}).get('judge_decoy'))


def repair_notes(record):
    """Notes appended to every repair instruction of a line or overlay idea: its repair note (when its invariant is in the
    plan, or always for one without an invariant) and, when the realizer recorded outputs, which outputs to re-emit. A
    JUDGE decoy is never repaired (gate.py), so it carries none."""
    if is_decoy(record):
        return []
    notes = [body(s['note']) for _, s in _specs(record.get('study'))
             if s.get('note') and (not s.get('invariant') or _present(s, record))]
    emit = emit_note(record.get('study'))
    return notes + ([emit] if emit else [])


# ------------------------------------------------------------------ the record that reaches the bank entry


def _versions(record):
    return list((record.get('difficulty_versions') or {}).values())


def final_points(record):
    """Every declared point on the final episode, in the final palette, with the cells where its alternative and the truth
    differ (DESIGN §6.1, for the 4B probe's point capture)."""
    episode = record.get('complete_task')
    if not episode:
        return []
    table, out = tables(record), []
    for part, kind in (('line', 'points'), ('overlay', 'overlay_points')):
        for p in table[kind] or []:
            truth = episode['test'][p['test'] - 1]['output'] if 1 <= p['test'] <= len(episode['test']) else None
            out.append({'part': part, 'id': p['id'], 'kind': p['kind'], 'note': p['note'], 'test': p['test'], 'at': p['at'],
                        'cells': point_cells(p, truth) if truth else []})
    return out


def fields(tid, record, calls):
    """blind_performance['weakness'] (reaches the bank entry; DESIGN §6 draft 4): the line, its stratum (relaxed fence or
    strict), overlay, size arm and rung, the plan check and form, the checks on the first version (the realizer's own
    episode and records, review S6) and on the final version (repair breakage when a check passing on the first fails on
    the final; a record the repairer left out is counted apart), the final points with their differing cells, which pairs
    each repair touched, a retag when the property turned into another line's, the R4 size check, Sol's answers against
    each declared point, clarifying repairs, and whether the task keeps its line's tag. Cheap: FORCED's solver result is
    stored with each emission (review B1), so this runs under the runtime lock without a search."""
    study = record.get('study') or {}
    if not tagged(study):
        return {}
    from .gate import current_blind
    weak = record.get('weakness') or {}
    s = spec(line_of(study), fence_of(study)) or {}
    value = {'line': line_of(study), 'stratum': fence_of(study) or ('strict' if (s.get('relax') or s.get('relaxation')) else None),
             'judge_decoy': bool(study.get('judge_decoy')),
             'fence': study.get('fence'), 'overlay': overlay_of(study), 'overlay_draws': deepcopy(study.get('weak_overlays')),
             'size_arm': study.get('size_arm'), 'rung': study.get('rung'), 'plan_check': deepcopy(weak.get('plan_check')),
             'emission_problems': [p for e in weak.get('emissions') or [] for p in e.get('problems') or []]}
    if structure_of(study) or study.get('structure_call') or study.get('structure_control'):  # the second axis; absent keys stay absent
        value.update(structure=structure_of(study), structure_call=study.get('structure_call'),
                     structure_excluded=study.get('structure_excluded'), structure_counted_as=study.get('structure_counted_as'),
                     structure_control=bool(study.get('structure_control')))
    if weak.get('default_readings') is not None:  # the recorded default-reading calls (SETTLE, ELIMINATE, PARSE)
        value['default_readings'] = deepcopy(weak['default_readings'])
    episode = record.get('complete_task')
    if not episode:
        value['line_tag'] = _tag(value)
        if 'structure' in value:
            value['structure_tag'] = _structure_tag(value)
        return {'weakness': value}
    table, first_table = tables(record), tables(record, first=True)
    versions = _versions(record)
    realized = realizer_episode(record)
    first_episode = realized or (versions[0]['episode'] if versions else episode)
    from arcgen.lib.util import content_hash
    value['first_version_basis'] = 'realizer' if realized else 'first_review' if versions else 'final'
    value['changed_before_review'] = bool(realized and versions and realized != versions[0]['episode'])
    value['first_version'] = episode_check(record, first_episode, first_table)
    value['final_version'] = episode_check(record, episode, table)
    final = lambda part: (value['final_version'].get(part) or {}).get('status')
    value['repair_breakage'] = {part: v['status'] == 'pass' and final(part) == 'fail' for part, v in value['first_version'].items()}
    value['repair_missing'] = {part: v['status'] == 'pass' and final(part) == 'incomplete' for part, v in value['first_version'].items()}
    value['first_episode_hash'] = content_hash(first_episode)
    value['points'] = final_points(record)
    value['repairs_touched'] = [['train_' + str(i + 1) for i, p in enumerate(after['episode']['train'])
                                 if i >= len(before['episode']['train']) or before['episode']['train'][i] != p]
                                for before, after in zip(versions, versions[1:])]
    retag = s.get('retag_on_break')
    first_pairs = ((value['first_version'].get('line') or {}).get('pairs') or {})
    final_pairs = ((value['final_version'].get('line') or {}).get('pairs') or {})
    if retag and len(versions) > 1 and not first_pairs.get('differing_pairs') and 1 <= len(final_pairs.get('differing_pairs') or []) <= 2:
        value['retagged'] = retag  # a repair broke the coincidence in a training pair: the task is a settling-pair task
    if s.get('retag_if_refuted') and 1 <= len(final_pairs.get('differing_pairs') or []) <= 2:
        value['retagged'] = s['retag_if_refuted']  # JUDGE: a rival a pair refutes makes a settling-pair task, not a judgement
    if s.get('record_alternatives'):  # JUDGE: the rival's test grids (first and final version) and the named prior
        grids = lambda ep, points: [{'id': p['id'], 'test': p['test'], 'grid': alternative(p, ep['test'][p['test'] - 1]['output'])}
                                    for p in points or [] if 1 <= p['test'] <= len(ep['test'])]
        value['prior'] = ((value.get('plan_check') or {}).get('declared') or {}).get('private_value')
        value['alternatives'] = grids(episode, table['points'])
        value['first_alternatives'] = grids(first_episode, first_table['points'])
    value['size_check'] = size_check(record, episode)
    first = (record.get('blind_results') or [{}])[0].get('response')
    review_episode = versions[0]['episode'] if versions else episode  # the episode the first blind call answered
    results = point_results(record, first, review_episode, first_table) if first else []
    value['first_blind'] = {'on_points': sorted({r['part'] for r in results if r['captured'] or r['equals_alternative']}) if first else None,
                            'right': bool(first) and first.get('predicted_test_outputs') == [p['output'] for p in review_episode['test']],
                            'points': results, 'equals_alternative': any(r['equals_alternative'] for r in results)}
    current = [point_results(record, row['response']) for row in current_blind(tid, record, calls)]
    value['current_blind_on_points'] = [sorted({r['part'] for r in rows if r['captured'] or r['equals_alternative']}) for rows in current]
    value['current_blind_points'] = current
    value['capture_routes'] = [v.get('shortcut_capture') for v in versions if v.get('shortcut_capture')]
    value['line_tag'] = _tag(value)
    if 'structure' in value:
        value['structure_tag'] = _structure_tag(value)
        # the final records' test grids, in the final palette (for answer-level capture offline, S3)
        value['structure_alternatives'] = [{'id': a['id'], 'kind': a.get('kind'), 'note': a.get('note'),
                                            'test': [a['test'].get(_key(p['input'])) for p in episode['test']]}
                                           for a in (table['alternatives'] or {}).values()]
    return {'weakness': value}


def _structure_tag(value):
    """Whether the task keeps its structure line's tag: 'opted_out' (the plan said "Structure: none", recorded apart for
    the per-host report), 'lost' (the invariant is missing or a check fails), 'kept' (the final version's checks pass),
    'unknown' (records missing or unchecked), 'no_episode'."""
    name = value.get('structure')
    if not name:
        return None
    if not value.get('final_version') and value.get('first_version') is None and 'first_blind' not in value:
        return 'no_episode'
    d = (value.get('plan_check') or {}).get('structure_declared') or {}
    if d.get('opted_out'):
        return 'opted_out'
    if spec(name).get('invariant') and d.get('plan') == 'ready' and not d.get('invariant'):
        return 'lost'
    status = ((value.get('final_version') or {}).get('structure') or {}).get('status')
    return {'pass': 'kept', 'fail': 'lost'}.get(status, 'unknown')


def _tag(value):
    """Whether the task keeps its line's tag (DESIGN §4: a task that loses its property is not killed, it loses the tag):
    'kept' when the final version's checks pass (or, for a line without checks, its invariant is declared); 'lost' when
    the plan opted out, the invariant is missing, a check fails, a repair retagged it, or a relaxation that needs Sol's
    first answer right (SETTLE's in-pair minority) did not get it; 'unknown' when outputs are missing or unchecked."""
    if not value['line']:
        return None
    if value.get('judge_decoy'):  # a planted negative for JUDGE's consensus procedure: never in the line's bite
        return 'decoy'
    if not value.get('final_version') and value.get('first_version') is None and 'first_blind' not in value:
        return 'no_episode'  # no realized episode: nothing to keep (review N2)
    declared_ = (value.get('plan_check') or {}).get('declared') or {}
    relaxation = (spec(value['line'], value.get('stratum') if value.get('stratum') != 'strict' else None) or {}).get('relaxation') or {}
    if (declared_.get('opted_out') or value.get('retagged')
            or (spec(value['line']).get('invariant') and declared_.get('plan') == 'ready' and not declared_.get('invariant'))
            or (relaxation.get('needs_right_first_answer') and (value.get('first_blind') or {}).get('right') is False)):
        return 'lost'
    status = ((value.get('final_version') or {}).get('line') or {}).get('status')
    tag = {'pass': 'kept', 'fail': 'lost', 'incomplete': 'unknown', 'unchecked': 'unknown'}.get(
        status, 'kept' if declared_.get('plan') == 'ready' else 'unknown')
    # JUDGE: the tag waits for the offline consensus reads (DESIGN §6.9)
    return 'pending_consensus' if tag == 'kept' and spec(value['line']).get('needs_consensus') else tag


def strawman_task(record):
    """SETTLE's strawman probe input (DESIGN §5.3, screen only, run offline by the main session): the consistent pairs
    and the test inputs (with two-pair tasks, the one consistent pair). None without a consistent pair (the in-pair
    minority form) or a passing pair check. The settling pairs are the ones the shortcut does not reproduce."""
    episode = record.get('complete_task')
    pairs = ((episode_check(record, episode).get('line') or {}).get('pairs') or {}) if episode else {}
    if pairs.get('status') != 'pass' or line_of(record.get('study')) != 'SETTLE' or fence_of(record.get('study')) == 'in_pair_minority':
        return None
    settling = {p['pair'] for p in pairs['differing_pairs']}
    return {'train': [deepcopy(p) for i, p in enumerate(episode['train'], 1) if i not in settling],
            'test': [{'input': deepcopy(p['input'])} for p in episode['test']]}


KNOWN = {'kind', 'invariant', 'invariants_extra', 'opt_out', 'shortcut', 'emits', 'computed', 'realizer', 'policy',
         'policy_needs_right_answer', 'policy_needs_check', 'withhold_arguments', 'note', 'clarify', 'retag_on_break',
         'explore', 'step3', 'r4', 'plan', 'grids', 'relax', 'form', 'colour_vary', 'no_margin_canvas', 'needs_consensus',
         'retag_if_refuted', 'record_alternatives', 'default_reading'}
# a structure line's entry (STRUCTURE_SPEC): its own fields, and the line fields it shares
STRUCT_KNOWN = {'kind', 'invariant', 'invariants_extra', 'opt_out', 'shortcut', 'emits', 'computed', 'realizer', 'policy',
                'policy_needs_check', 'note', 'clarify', 'plan', 'grids', 'form', 'not_on', 'counted_as', 'rungs', 'idea',
                'replaces_overlay', 'colour_vary', 'default_reading'}
PLAN_CHECKS = {'pairs', 'pair_labels', 'pair_values', 'test_field', 'private_field', 'growth_max', 'pair_mention',
               'role_meaning', 'record_only', 'shortcut_count', 'pairs_per_rival'}
GRID_RULES = {'pairs': {'min', 'max', 'minority', 'in_pair_minority', 'open_ground'}, 'growth': {'area_max', 'parts_max'},
              'points': {'tests', 'max_per_test', 'max_kinds_per_test', 'min_places', 'max_total', 'decisions', 'junctions'},
              'overlay_points': {'tests', 'max_per_test', 'max_total'}, 'assembly': {'branch_max', 'node_limit'},
              'objects': {'contrast', 'contrast_grids'}, 'alternatives': {'rule', 'rivals', 'dimensions_min'},
              'frame': {'sizes_min'}}
RELAX_KNOWN = {'text', 'plan', 'grids', 'needs_right_first_answer', 'decoy', 'rungs'}


def verify_registry():
    """Every registry entry uses known fields, kinds, emissions, checks and relaxations (a new line's entry is checked as
    the others are)."""
    if not LINES and not OVERLAYS:  # a deployment without the registry file: no weakness run is possible (config.py)
        return True
    for name, s in {**LINES, **OVERLAYS}.items():
        kinds = set(s.get('emits', ())) | set(s.get('computed') or ())
        variants = [s] + [spec(name, f) for f in s.get('relax') or {}]
        if (set(s) - KNOWN or s.get('kind') not in ('line', 'control', 'overlay') or set(s.get('emits', ())) - set(EMIT_FIELDS)
                or set(s.get('computed') or ()) - {'conservation', 'growth'}
                or any(set(v.get('plan') or {}) - PLAN_CHECKS for v in variants)
                or any(k not in GRID_RULES or set(r) - GRID_RULES[k] for v in variants for k, r in (v.get('grids') or {}).items())
                or any(k in kinds and k not in (s.get('grids') or {}) for k in ('pairs', 'points', 'overlay_points', 'assembly', 'growth'))
                or (s.get('opt_out') and s['opt_out'].get('field') not in ('risk', 'test', 'invariant'))
                or (kinds & {'pairs', 'points', 'overlay_points'} - {'points'} and not s.get('shortcut'))
                or (s['kind'] == 'overlay') != (name in OVERLAYS) or (('overlay_points' in kinds) != (s['kind'] == 'overlay' and bool(s.get('emits'))))
                or (s.get('emits') and not s.get('realizer'))
                or (s.get('relax') and (s['kind'] != 'line' or any(set(r) - RELAX_KNOWN or 'text' not in r
                                                                  or set(r.get('rungs') or ()) - {'R1', 'R2', 'R3', 'R4'}
                                                                  for r in s['relax'].values())))
                or s.get('default_reading') not in (None, 'settling')
                or any(s.get(k) and s[k] not in LINES for k in ('retag_on_break', 'retag_if_refuted'))):
            raise IntegrityError('weakness line registry entry is malformed: ' + name)
    if [n for n, s in LINES.items() if s['kind'] == 'control'] != [LINE_CONTROL]:
        raise IntegrityError('the weakness line registry needs exactly one control line')
    from .structures import CHECKS
    for name, s in STRUCTS.items():
        kinds = set(s.get('emits', ())) | set(s.get('computed') or ())
        if (set(s) - STRUCT_KNOWN or s.get('kind') != 'structure' or name in LINES or name in OVERLAYS
                or set(s.get('emits', ())) - {'alternatives', 'units', 'surfaces', 'objects'}
                or set(s.get('computed') or ()) - {'frame'} or not s.get('invariant')
                or (s.get('opt_out') or {}).get('field') != 'invariants' or not (s.get('opt_out') or {}).get('prefix')
                or set(s.get('plan') or {}) - PLAN_CHECKS
                or any(k not in GRID_RULES or set(r) - GRID_RULES[k] for k, r in (s.get('grids') or {}).items())
                or any(k in kinds and k not in (s.get('grids') or {}) for k in ('alternatives', 'frame'))
                or ('alternatives' in kinds and s['grids']['alternatives'].get('rule') not in CHECKS)
                or (s.get('emits') and not s.get('realizer'))
                or set(s.get('not_on') or ()) - set(LINES) or set(s.get('counted_as') or {}) - set(LINES)
                or not s.get('rungs') or set(s['rungs']) - {'R1', 'R2', 'R3', 'R4'}
                or (s.get('replaces_overlay') and s['replaces_overlay'] not in OVERLAYS)
                or s.get('default_reading') not in (None, 'killers', 'fitting_pairs')
                or any(c.get('prefix') is None or not 0 <= c['min'] <= c['max'] for c in (s.get('plan') or {}).get('shortcut_count') or ())):
            raise IntegrityError('weakness structure registry entry is malformed: ' + name)
    return True


def verify_names():
    """Every file the registry names exists: each line's idea text, every planner line, relaxation, policy, note and
    addendum, and the R4 scale line."""
    names = {'lines/outputs.realizer.md', 'lines/emit.repair_note.md', 'lines/clarify_note.md', 'rungs/ANY.R4.planner.md'}
    for name, s in {**LINES, **OVERLAYS}.items():
        names |= {s[k] for k in ('policy', 'note', 'clarify', 'realizer') if s.get(k)}
        names |= {f'ideas/{s[k]}' for k in ('explore', 'step3') if s.get(k)}
        names |= {r['text'] for r in (s.get('relax') or {}).values()}
        names.add(f'lines/{name}.planner.md')
        if s['kind'] == 'line':
            names.add(f'ideas/modifiers/{name}.md')
    for name, s in STRUCTS.items():
        names |= {s[k] for k in ('policy', 'note', 'clarify', 'realizer', 'idea') if s.get(k)}
        names |= {f'lines/{name}.planner.md', 'lines/default_reading.md'}
    missing = sorted(n for n in names if not (V1 / n).is_file())
    if missing:
        raise IntegrityError('weakness line texts missing: ' + ', '.join(missing))
    return sorted(names)


verify_registry()
