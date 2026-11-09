"""Batch configuration: what a run is (science) and how it is executed (operations).

A run freezes its profile at `<run root>/config.json` during `prepare`. The launch authorization binds the code hash and
the *science* hash. Some operational settings (workers, pool, development windows, timeout, usage credits, concurrent
activity, ramp-up, slate barriers) may be amended on a prepared run by recorded amendment files in
`<run root>/config_amendments/` (`entry.py amend`, `bin/arcgen workers`), without a code change. The Codex home and
executable are bound at launch.

Selection order (resolved by `entry.py` before any pipeline module is imported):
  1. ARCGEN_CONFIG environment variable (a frozen run config or a profile path)
  2. `profiles/v1.json`
Amendments are applied when ARCGEN_RUN_ROOT is set.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
from . import slots as _slots

HERE = Path(__file__).resolve().parent
from arcgen.paths import WORKSPACE as ROOT, PACKAGE  # noqa: E402  (ROOT: the data root, where runs and the bank live)
PROFILES = HERE / 'profiles'
# One Codex account. Internally the pipeline keeps an account dimension of size one (run folders are
# accounts/<ACCOUNT>/...), so every ledger, pool and route record names it.
ACCOUNT = 'main'
# Operations: how a run executes. `workers` is the number of concurrent Codex calls; the idea/development pool split
# and the development windows follow from it unless given (split()). The Codex home is `codex_home`, else $CODEX_HOME,
# else ~/.codex; the executable is `codex_executable` (a name on PATH or a path), optionally pinned by `codex_sha256`.
OPERATIONAL_KEYS = frozenset({'timeout_seconds', 'workers', 'group_size'})
# Settings a prepared run may change by recorded amendment. The Codex home and executable are bound by the launch
# authorization; group size is baked into the frozen design.
AMENDABLE_KEYS = frozenset({'timeout_seconds', 'workers', 'pool', 'development_windows', 'slate_barriers', 'usage_credits',
                            'allow_concurrent_activity', 'ramp_up'})
# Optional operational keys (absent means the default). allow_concurrent_activity=true lets authorize/run proceed
# while other Codex activity is present on the account (the snapshot is still recorded); default false = idle.
OPTIONAL_OPERATIONAL_KEYS = frozenset({'codex_home', 'codex_executable', 'codex_sha256', 'pool', 'development_windows',
                                       'slate_barriers', 'ramp_up', 'allow_concurrent_activity', 'continuous_supply',
                                       'supply_windows', 'idle_supply_windows', 'usage_credits'})
# usage_credits=true: the account has usage credits, so a used-up weekly limit does not pause a run (Codex keeps working);
# a real usage refusal from the provider still pauses.
# continuous_supply / supply_windows shape the scheduler policy, which is frozen when a run first starts:
# set them before launch (bin/arcgen new), never by amendment. idle_supply_windows (continuous supply only):
# idea windows open at once while there is no development to run (absent: no extra windows).


def split(total, group_size=1):
    """Idea and development workers for `total` concurrent Codex calls, and the development windows that keep the
    development workers busy: a task holds one call at a time, so about as many tasks in flight as workers, plus two.
    Once the idea work has ended, the scheduler adds the idea workers to development (continuous supply)."""
    up = max(1, round(total * 0.27)); down = max(1, total - up)
    return {'pool': {'downstream_workers': down, 'upstream_workers': up},
            'development_windows': max(4, -(-(down + 2) // group_size))}


def codex_home(ops):
    """The Codex home a run uses: operations.codex_home, else $CODEX_HOME, else ~/.codex (resolved, not required to exist)."""
    value = ops.get('codex_home') or os.environ.get('CODEX_HOME') or '~/.codex'
    return str(Path(value).expanduser().resolve())


def codex_executable(ops):
    """{'path', 'sha256'} of the Codex executable: the resolved path and its bytes' hash (sha256 None when not found).
    A configured codex_sha256 must match; the launch manifest binds the pair, so a changed executable stops a resume."""
    import shutil
    name = ops.get('codex_executable') or 'codex'
    found = shutil.which(name) if not Path(name).expanduser().is_absolute() else str(Path(name).expanduser())
    path = Path(found).resolve() if found else None
    actual = sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
    return {'path': str(path) if path else name, 'sha256': actual, 'pinned_sha256': ops.get('codex_sha256')}


def expand_operations(ops):
    """The internal operations view: user settings plus the derived pool and windows, the single account and its home."""
    out = deepcopy(ops)
    shape = split(ops['workers'], ops['group_size'])
    out.setdefault('pool', shape['pool'])
    out.setdefault('development_windows', shape['development_windows'])
    out.setdefault('slate_barriers', False)
    out['accounts'] = {ACCOUNT: codex_home(ops)}
    out['account_assignment'] = 'dynamic'  # development groups float; with one account they all land on it
    # internal names used by the scheduler and design (one account: per account = per run)
    out['development_windows_per_account'] = out['development_windows']
    if 'supply_windows' in out:
        out['supply_windows_per_account'] = out['supply_windows']
    if 'idle_supply_windows' in out:
        out['idle_supply_windows_per_account'] = out['idle_supply_windows']
    return out


PROMPT_SETS = ('v1',)
# Skill lines (2026-10-18): each centres an idea on one common reasoning skill. With science.skill_novelty, calls that
# drew one use the skill variants of explore and step 3, and the run's selector judges novelty on what the solver reads
# and decides in the scene, not on sharing the skill (prompts/v1/selector_skill.md).
SKILL_MODIFIERS = ('QUANTITY', 'SYMMETRY', 'MATCH', 'CUED', 'EMIT', 'GROWTH', 'NOISY',
                   # 2026-10-25 (composite batch c1): new territory, built
                   # exactly as the lines above (idea text, binding planner line, skill-scoped novelty): self-similar
                   # structure, an iterated local rule, an order of a visible quantity, a drawn network
                   'RECURSE', 'AUTOMATON', 'RANK', 'GRAPH')
# Weakness lines (2026-10-20): each is a property that admits many rules and
# trains against one shortcut the 4B takes. A line is an idea modifier pinned per spec row, as the skill lines are, with
# skill-scoped novelty; it also carries a binding planner line composed last, a protected invariant, fixed-prefix plan
# fields checked in code, a review policy and repair note, and shortcut grids the realizer emits (lines.py). The lines are
# data: weakness_lines.json names each line's files and checks, so a new line is its prompt files plus one entry there.
# INGRID is the EXEMPLAR v2 line (EXEMPLAR is an arm's name). The control line (kind control) is the matched control: no
# idea modifier, only its planner line and its plan's top shortcut; a spec row marks it with {"control": true}. Overlays
# (EDGE, ROLES) are per-idea planner draws in a weakness run, at most one per idea.
# A deployment without the file (one that ships only *.py) still runs every other profile; a weakness profile refuses.
_REGISTRY_PATH = HERE / 'weakness_lines.json'
LINE_REGISTRY = json.loads(_REGISTRY_PATH.read_text()) if _REGISTRY_PATH.is_file() else {'lines': {}, 'overlays': {}}
LINE_MODIFIERS = tuple(n for n, v in LINE_REGISTRY['lines'].items() if v['kind'] == 'line')
LINE_CONTROL = next((n for n, v in LINE_REGISTRY['lines'].items() if v['kind'] == 'control'), None)
LINE_NAMES = LINE_MODIFIERS + ((LINE_CONTROL,) if LINE_CONTROL else ())
WEAKNESS_OVERLAYS = tuple(LINE_REGISTRY['overlays'])
# structure lines (2026-10-22): a second axis drawn per call on top of
# a content line or an arm (a spec row's {"structure": NAME}), never an idea modifier
STRUCTURES = tuple(LINE_REGISTRY.get('structures') or ())
NOVELTY_LINES = SKILL_MODIFIERS + LINE_MODIFIERS  # modifiers whose calls use the scoped-novelty variants
# v1 idea-call property values (prompts/v1/ideas/modifiers/). The first four are the original modifiers; then the
# 2026-10-07 property slots (output: DIGEST, EXTENT; scene: PACKED; evidence: VACANCY; relevance: IGNORE; colour: PAYLOAD,
# MONO; demand: LIGHT) and the 2026-10-09 ones (episode: KEY, CHAINED, MINIMAL, DISTRIBUTED; test: DERIVED; content:
# MEASURE). Most have a planner line in prompts/v1/slots/.
MODIFIER_NAMES = ('CONSTRUCT', 'EASIER', 'COLOURFUL', 'BUSY', 'DIGEST', 'EXTENT', 'PACKED', 'VACANCY', 'IGNORE', 'PAYLOAD',
                  'MONO', 'LIGHT', 'KEY', 'CHAINED', 'MINIMAL', 'DISTRIBUTED', 'DERIVED', 'MEASURE',
                  # 2026-10-12: an output that is a different scene
                  'SCENE',
                  # 2026-10-14 (eval-style study F-05/F-08): two or three plain rules acting together on one scene
                  'JOINED',
                  # 2026-10-16 (round-2 coverage): an ordered symbolic
                  # instruction applied step by step, its effects reaching across the grid
                  'INSTRUCT',
                  # 2026-10-18 (blind-spot batch): important skills
                  # that human tasks centre on and ours rarely do, or use only buried in conventions
                  *SKILL_MODIFIERS,
                  # 2026-10-20 (weakness batch): one line per call, pinned per spec row
                  *LINE_MODIFIERS,
                  # 2026-10-23: a single demonstration pair
                  # whose instances teach the whole rule (slots.py)
                  _slots.ONE)
# Targeted families and scale rungs (2026-10-12 plan): call attributes set in the allocation spec, carried into the call
# identity only when present (prompts/v1/families/, prompts/v1/rungs/). A rung's floors come from science.rungs; a
# general run may instead draw one per idea at admission (science.rung_draw: R1 at a rate, or a ladder of shares).
FAMILIES = ('SIM', 'FIT', 'ENCL', 'LEGEND', 'SYMSEQ')
RUNGS = ('R1', 'R2', 'R3')
# R4 (2026-10-20, weakness batch): the full-scale rung, about 30x30 test outputs, for weakness
# line ideas only (prompts/v1/rungs/ANY.R4.planner.md; no family texts, no general rung draw). Floors for it are accepted
# only with science.weakness, so every other profile, family call and rung draw keeps R1-R3.
FULL_SCALE = 'R4'
ALL_RUNGS = RUNGS + (FULL_SCALE,)
SIZE_ARMS = {'small': ALL_RUNGS[:1], 'large': ALL_RUNGS[1:]}  # a weakness run's size arms and the rungs each may draw
RELAX_CAP = 0.30  # per-idea relaxed-fence draws: at most this share of a line's ideas (DESIGN draft 4 §6.4, the full run);
# the screen fixes fences per call instead (a spec row's {"fence": ...}), at most 40% of a line's calls (v2 §5)
RUNG_FLOOR_KEYS = frozenset({'output_cells', 'computed_cells', 'escalation'})
OVERLAY_KINDS = frozenset({'scale', 'COMPACT', 'MANY', 'XFER', 'RICH', 'SECOND'})  # v1 presentation overlays  # downstream prompt templates: r23 as launched, or diversity-v1 (batch/prompts/v1)
ROLE_PURPOSES = frozenset({'full_window_selection', 'planning', 'realization', 'technical_verification', 'primary_blind',
                           'conditional_blind', 'ambiguity_adjudication', 'targeted_repair', 'mechanism_audit'})


class ConfigError(ValueError):
    pass


def rung_shares(draw):
    """A rung draw's (rung, share) pairs in rung order: one rung at a rate, or a ladder of shares (2026-10-15 size
    weight for general runs). Unknown rungs sort last."""
    if 'ladder' in draw:
        ladder = draw['ladder'] if isinstance(draw['ladder'], dict) else {}
        return sorted(ladder.items(), key=lambda kv: ALL_RUNGS.index(kv[0]) if kv[0] in ALL_RUNGS else len(ALL_RUNGS))
    return [(draw.get('rung'), draw.get('rate'))]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def _share(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1


def _ladder_ok(ladder, rungs, allowed=ALL_RUNGS):
    return (isinstance(ladder, dict) and set(ladder) <= set(allowed) and set(ladder) <= set(rungs or {})
            and all(_share(x) for x in ladder.values()) and sum(ladder.values()) <= 1 + 1e-9)


def _validate_weakness(sci):
    """science.weakness (2026-10-20):
      lines: {LINE or CONTROL: {rungs: {rung: share}, relaxed: {relaxation: share}}}, every line of the run. rungs is the
             line's size mix (the remainder takes no size line); an idea whose call fixes its size arm draws inside that
             arm of the mix (admission._weakness_draws). A line the registry marks r4 false never draws R4. relaxed gives
             each of the line's registry relaxations (relaxed fences) a per-idea share, at most RELAX_CAP together (a decoy
             form, JUDGE's planted negatives, is never drawn: a spec row's {"decoy": true} fixes it);
      overlays: {OVERLAY: {share, excluded: [arms, modifiers or lines that take no draw]}}, per-idea shares of the planner
             overlays (one uniform draw in registry order, so at most one per idea);
      structures: {STRUCTURE: {}}, the structure lines a spec row may pin on a call ({"structure": NAME});
      default_reading: true runs the recorded default-reading call (one low-effort Sol call per rival, binding
             call_overrides.default_reading) on verified SETTLE, ELIMINATE and PARSE tasks.
    Absent, nothing in a run changes."""
    w = sci.get('weakness')
    if w is None:
        return
    if not LINE_CONTROL:
        raise ConfigError('weakness: the line registry (batch/weakness_lines.json) is missing')
    mods = sci.get('idea_modifiers') or {}
    rungs = sci.get('rungs')
    if (not isinstance(w, dict) or set(w) - {'lines', 'overlays', 'structures', 'default_reading'}
            or 'lines' not in w or sci.get('prompt_set') != 'v1' or sci.get('idea_stage') != 'v1_loop'
            or sci.get('modifier_assignment') != 'exact'):
        raise ConfigError('weakness: {lines, overlays, structures, default_reading} with prompt_set v1, the v1 idea loop and '
                          'exact modifier assignment')
    structures = w.get('structures', {})
    if (not isinstance(structures, dict) or set(structures) - set(STRUCTURES)
            or any(v != {} for v in structures.values())):
        raise ConfigError('weakness.structures: {' + '|'.join(STRUCTURES) + ': {}}, the structure lines a run may pin per call')
    if any(LINE_REGISTRY['structures'][k].get('colour_vary') for k in structures) and sci.get('colour_roles') != 'mix':
        raise ConfigError('weakness.structures: a structure line that varies colour roles per idea (VARY) needs colour_roles mix')
    if type(w.get('default_reading', False)) is not bool or (w.get('default_reading') and 'default_reading' not in sci.get('call_overrides', {})):
        raise ConfigError('weakness.default_reading: true needs call_overrides.default_reading (the low-effort Sol binding)')
    lines = w['lines']
    if (not isinstance(lines, dict) or not lines or set(lines) - set(LINE_NAMES)
            or any(not isinstance(v, dict) or set(v) - {'relaxed'} != {'rungs'} or not _ladder_ok(v['rungs'], rungs)
                   for v in lines.values())):
        raise ConfigError('weakness.lines: {LINE or CONTROL: {rungs: {rung of science.rungs: share}, relaxed: {...}}}, shares '
                          'summing to at most 1')
    for name, v in lines.items():
        entry = LINE_REGISTRY['lines'][name]
        if entry.get('r4') is False and v['rungs'].get('R4'):
            raise ConfigError(f'weakness: {name} never draws the full-scale rung R4')
        relaxed = v.get('relaxed', {})
        drawable = {k for k, r in (entry.get('relax') or {}).items() if not r.get('decoy')}  # decoys come only by spec row
        if (not isinstance(relaxed, dict) or set(relaxed) - drawable
                or not all(_share(x) for x in relaxed.values()) or sum(relaxed.values()) > RELAX_CAP + 1e-9):
            raise ConfigError(f'weakness.lines.{name}.relaxed: {{relaxation of the registry: share}}, at most {RELAX_CAP} together')
    used = {m for m in LINE_MODIFIERS if mods.get(m)}
    if used != set(lines) - {LINE_CONTROL}:
        raise ConfigError('weakness.lines must list exactly the weakness lines among idea_modifiers')
    over = w.get('overlays', {})
    if (not isinstance(over, dict) or set(over) - set(WEAKNESS_OVERLAYS)
            or any(not isinstance(v, dict) or set(v) - {'share', 'excluded'} or not _share(v.get('share'))
                   or not isinstance(v.get('excluded', []), list) or not all(isinstance(x, str) for x in v.get('excluded', []))
                   for v in over.values())
            or sum(v['share'] for v in over.values()) > 1 + 1e-9):
        raise ConfigError('weakness.overlays: {' + '|'.join(WEAKNESS_OVERLAYS) + ': {share, excluded: [names]}}, shares '
                          'summing to at most 1')
    if any(v['share'] and LINE_REGISTRY['overlays'][k].get('colour_vary') for k, v in over.items()) and sci.get('colour_roles') != 'mix':
        raise ConfigError('weakness: an overlay that varies colour roles per idea needs colour_roles mix')
    if 'MARKS' in used:  # busy, packed and noisy scenes make small marks indistinguishable from noise
        excl = sci.get('modifier_exclusions') or {}
        for other in ('BUSY', 'PACKED', 'NOISY'):
            if mods.get(other) and other not in excl.get('MARKS', ()) and 'MARKS' not in excl.get(other, ()):
                raise ConfigError(f'weakness: MARKS must exclude {other}')


def _validate_one(sci):
    """ONE (2026-10-23, slots.py): a profile that draws or pins the single-pair slot keeps it, through modifier_exclusions
    (or a shared slot), off every arm, family and slot value whose text needs two or more pairs or variation between them,
    and off every weakness line it has. Admission enforces the rest (no structure line, teaching or departure draw, weakness
    overlay or MANY overlay)."""
    missing = _slots.exclusion_problems(sci, LINE_MODIFIERS)
    if missing:
        raise ConfigError('idea_modifiers.ONE: modifier_exclusions.ONE (or a shared slot) must keep the single-pair slot off '
                          + ', '.join(missing))


def validate(value):
    try:
        sci, ops = value['science'], value['operations']
        models = sci['models']
        if not {'astra', 'sol'} <= set(models): raise ConfigError('models must name astra and sol')
        if sci.get('idea_stage') != 'v1_loop' or sci.get('prompt_set') != 'v1':
            raise ConfigError('this pipeline runs the v1 design: science.idea_stage "v1_loop" and science.prompt_set "v1"')
        if set(sci['roles']) != ROLE_PURPOSES: raise ConfigError('roles must bind every production purpose')
        for key, (model, effort) in list(sci['roles'].items()) + list(sci['call_overrides'].items()) + list(sci['planners'].items()):
            if model not in models or effort not in ('low', 'medium', 'high', 'xhigh', 'max'):
                raise ConfigError(f'bad binding for {key}')
        pa = sci['planner_assignment']
        if pa['default_planner'] not in sci['planners'] or not set(pa['multi_planners']) <= set(sci['planners']):
            raise ConfigError('planner assignment names an unknown planner')
        if pa['multi_planners'] and pa['multi_planners'][0] != pa['default_planner']:
            raise ConfigError('the first multi planner must be the default planner (it keeps the native task identity)')
        if not 0 <= pa['multi_planner_fraction'] <= 1: raise ConfigError('multi_planner_fraction must be in [0,1]')
        if sci.get('prompt_set', 'r23') not in PROMPT_SETS: raise ConfigError('unknown prompt_set')
        if sci.get('colour_roles', 'v1') not in ('v1', 'choice', 'vary', 'mix'): raise ConfigError('unknown colour_roles')
        mods = sci.get('idea_modifiers')
        if mods is not None and (sci.get('idea_stage') != 'v1_loop' or not isinstance(mods, dict)
                                 or set(mods) - set(MODIFIER_NAMES) or any(not 0 <= r <= 1 for r in mods.values())):
            raise ConfigError('idea_modifiers: {' + '|'.join(MODIFIER_NAMES) + ': rate in [0,1]} with the v1 idea loop')
        if sci.get('modifier_assignment', 'per_call') not in ('per_call', 'exact'):
            raise ConfigError('modifier_assignment is per_call (independent draws) or exact (exact share of calls)')
        # Property slots: values in one slot exclude each other, a call holds at most modifier_cap slots, and
        # modifier_exclusions keeps a value off calls with the listed arms or alongside the listed values.
        if ({'modifier_slots', 'modifier_cap', 'modifier_exclusions', 'modifier_arm_caps', 'modifier_bare'} & set(sci)
                and (sci.get('modifier_assignment') != 'exact' or not mods)):
            raise ConfigError('modifier slots, cap and exclusions need exact assignment of idea_modifiers')
        slots = sci.get('modifier_slots')
        if slots is not None:
            members = [m for values in slots.values() for m in values] if isinstance(slots, dict) else []
            if (not members or len(members) != len(set(members)) or set(members) != set(mods)
                    or any(sum(mods[m] for m in values) > 1 + 1e-9 for values in slots.values())):
                raise ConfigError('modifier_slots must partition idea_modifiers, with rates in each slot summing to at most 1')
        if 'modifier_cap' in sci and (type(sci['modifier_cap']) is not int or sci['modifier_cap'] < 1):
            raise ConfigError('modifier_cap must be a positive integer')
        excl = sci.get('modifier_exclusions')
        if excl is not None and (not isinstance(excl, dict) or set(excl) - set(mods or {})
                                 or any(not isinstance(v, list) or not all(isinstance(x, str) for x in v) for v in excl.values())):
            raise ConfigError('modifier_exclusions: {modifier: [arm or modifier names]}')
        # modifier_arm_caps: a lower cap for some arms; modifier_bare: calls kept free of modifiers, per arm ('*': any arm)
        for key in ('modifier_arm_caps', 'modifier_bare'):
            counts = sci.get(key)
            if counts is not None and (not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values())):
                raise ConfigError(key + ': {arm: non-negative integer}')
        sel = sci['selector']
        if 'minimum_accepted_neighbors' in sel and (type(sel['minimum_accepted_neighbors']) is not int
                                                    or not 1 <= sel['minimum_accepted_neighbors'] <= sel['accepted_neighbors_per_candidate']):
            raise ConfigError('selector.minimum_accepted_neighbors must be an integer from 1 to accepted_neighbors_per_candidate')
        # comparator_depth_cap (2026-10-23): the deepest the selector follows a drifting comparator (qualification.py)
        if 'comparator_depth_cap' in sel and (type(sel['comparator_depth_cap']) is not int
                                              or sel['comparator_depth_cap'] < sel['accepted_neighbors_per_candidate']):
            raise ConfigError('selector.comparator_depth_cap must be an integer of at least accepted_neighbors_per_candidate')
        # batch_references: idea windows of a later wave see the kernels selected in earlier waves of the same batch
        # (frozen when the wave opens); needs the v1 idea loop and continuous supply.
        if type(sci.get('batch_references', False)) is not bool:
            raise ConfigError('batch_references must be boolean')
        if sci.get('batch_references') and (sci.get('idea_stage') != 'v1_loop' or not ops.get('continuous_supply')):
            raise ConfigError('batch_references needs the v1 idea loop and continuous_supply')
        if sci.get('pose_rate') is not None and (sci.get('prompt_set') != 'v1' or not 0 <= sci['pose_rate'] <= 1):
            raise ConfigError('pose_rate in [0,1] needs prompt_set v1')
        # Scale rungs: floors checked on the plan (test output cells, test input over mean training input) and measured
        # on the grids (computed cells). rung_draw gives a rung to a seeded share of a general run's ideas at admission
        # (one rung at a rate, or a ladder of shares; prompts/v1/rungs/ANY.<rung>.planner.md), never to ideas carrying
        # an excluded slot value or overlay. targeted switches the run's selector to the family sentence
        # (prompts/v1/selector_targeted.md).
        rungs = sci.get('rungs')
        if rungs is not None and (sci.get('prompt_set') != 'v1' or not isinstance(rungs, dict)
                                  or set(rungs) - set(ALL_RUNGS if sci.get('weakness') else RUNGS)
                                  or any(not isinstance(v, dict) or set(v) != RUNG_FLOOR_KEYS
                                         or any(not isinstance(x, (int, float)) or x < 0 for x in v.values()) for v in rungs.values())):
            raise ConfigError('rungs: {R1|R2|R3: {output_cells, computed_cells, escalation}} with prompt_set v1')
        draw = sci.get('rung_draw')
        if draw is not None and (not rungs or not isinstance(draw, dict)
                                 or set(draw) - {'max_rung'} not in ({'rung', 'rate', 'excluded'}, {'ladder', 'excluded'})
                                 or not isinstance(draw['excluded'], list)
                                 or set(draw['excluded']) - set(MODIFIER_NAMES) - OVERLAY_KINDS
                                 or not isinstance(draw.get('max_rung', {}), dict)
                                 or set(draw.get('max_rung', {})) - set(MODIFIER_NAMES) - OVERLAY_KINDS
                                 or set(draw.get('max_rung', {})) & set(draw['excluded'])
                                 or any(r not in rungs or r not in RUNGS for r in draw.get('max_rung', {}).values())
                                 or ('ladder' in draw and (not isinstance(draw['ladder'], dict) or not draw['ladder']))
                                 or any(r not in rungs or r not in RUNGS or not isinstance(x, (int, float)) or not 0 <= x <= 1
                                        for r, x in rung_shares(draw))
                                 or sum(x for _, x in rung_shares(draw)) > 1 + 1e-9):
            raise ConfigError('rung_draw: {rung, rate in [0,1]} or {ladder: {rung: share}, shares summing to at most 1}, '
                              'with excluded: [slot values or overlays], optional max_rung: {slot value or overlay not '
                              'excluded: rung}, and rungs')
        if type(sci.get('targeted', False)) is not bool or (sci.get('targeted') and not rungs):
            raise ConfigError('targeted must be boolean and needs rungs')
        if type(sci.get('skill_novelty', False)) is not bool or (sci.get('skill_novelty') and (
                sci.get('idea_stage') != 'v1_loop' or sci.get('targeted')
                or not set(NOVELTY_LINES) & set(sci.get('idea_modifiers') or {}))):
            raise ConfigError('skill_novelty must be boolean and needs the v1 idea loop, skill or weakness lines among '
                              'idea_modifiers, and no targeted selector (its selector already carries the targeted sentences)')
        _validate_weakness(sci)
        _validate_one(sci)
        if sci.get('pose_off_families') is not None and (not isinstance(sci['pose_off_families'], list)
                                                         or set(sci['pose_off_families']) - set(FAMILIES)):
            raise ConfigError('pose_off_families: [family names]')
        if (sci.get('colour_roles', 'v1') == 'mix') != ('colour_vary_rate' in sci) or not 0 <= sci.get('colour_vary_rate', 0) <= 1:
            raise ConfigError('colour_roles mix needs colour_vary_rate in [0,1] (and only mix uses it)')
        if sci.get('colour_roles', 'v1') != 'v1' and sci.get('prompt_set', 'r23') != 'v1':
            raise ConfigError('colour_roles choice/vary/mix need prompt_set v1')
        canvas = sci.get('canvas_draw')
        if canvas is not None and (not isinstance(canvas, dict) or set(canvas) - {'square', 'margin', 'both', 'margin_excluded'}
                                   or any(not 0 <= canvas.get(k, 0) <= 1 for k in ('square', 'margin', 'both'))
                                   or sum(canvas.get(k, 0) for k in ('square', 'margin', 'both')) > 1
                                   or not isinstance(canvas.get('margin_excluded', []), list)
                                   or sci.get('prompt_set', 'r23') != 'v1'):
            raise ConfigError('canvas_draw: {square, margin, both: shares summing to at most 1, margin_excluded: [rungs or slots]}, v1 only')
        # b17 review (2026-10-15): a seeded share of ideas gets the teaching line
        # (prompts/v1/teach.plain.planner.md); the rest are the control. pairs 'three' gives round 1's sentence on the
        # pair count (teach.three.planner.md) instead of round 2's 'fewest'.
        teach = sci.get('teach_draw')
        if teach is not None and (not isinstance(teach, dict) or set(teach) - {'excluded', 'pairs'} != {'plain'}
                                  or teach.get('pairs', 'fewest') not in ('fewest', 'three')
                                  or isinstance(teach['plain'], bool) or not isinstance(teach['plain'], (int, float))
                                  or not 0 <= teach['plain'] <= 1 or not isinstance(teach.get('excluded', []), list)
                                  or not all(isinstance(x, str) for x in teach.get('excluded', []))
                                  or sci.get('prompt_set', 'r23') != 'v1'):
            raise ConfigError('teach_draw: {plain: share in [0,1], excluded: [arms, slots or routes], pairs: fewest or three}, '
                              'v1 only')
        # departure package (2026-10-16): a seeded share of ideas whose
        # first test meets one visible situation no training pair shows; excluded names take no draw.
        depart = sci.get('depart_draw')
        if depart is not None and (not isinstance(depart, dict) or set(depart) - {'excluded'} != {'test'}
                                   or isinstance(depart['test'], bool) or not isinstance(depart['test'], (int, float))
                                   or not 0 <= depart['test'] <= 1 or not isinstance(depart.get('excluded', []), list)
                                   or not all(isinstance(x, str) for x in depart.get('excluded', []))
                                   or sci.get('prompt_set', 'r23') != 'v1'):
            raise ConfigError('depart_draw: {test: share in [0,1], excluded: [arms, slots, rungs, families or routes]}, v1 only')
        # gate study (2026-10-15): a seeded share of ideas whose
        # flagged but right blind answer clears when the second blind call is right too; the rest keep the old gate.
        gate_draw = sci.get('gate_draw')
        if gate_draw is not None and (not isinstance(gate_draw, dict) or set(gate_draw) != {'right_twice'}
                                      or not isinstance(gate_draw['right_twice'], (int, float))
                                      or not 0 <= gate_draw['right_twice'] <= 1 or sci.get('prompt_set', 'r23') != 'v1'):
            raise ConfigError('gate_draw: {right_twice: share in [0,1]}, v1 only')
        routes = sci['routes']['scale_overlay']
        if routes['assignment'] not in ('per_group_top', 'per_idea_seeded', 'none'): raise ConfigError('unknown scale overlay assignment')
        if not 0 <= routes.get('rate', 0) <= 1: raise ConfigError('scale overlay rate must be in [0,1]')
        kinds = routes.get('kinds')
        if kinds is not None and (not kinds or set(kinds) - OVERLAY_KINDS or sci.get('prompt_set', 'r23') != 'v1'
                                  or routes['assignment'] != 'per_idea_seeded'):
            raise ConfigError('overlay kinds need prompt_set v1, per-idea seeded routes and known kinds')
        if routes['assignment'] == 'per_group_top' and ops['group_size'] != 6:
            raise ConfigError('per_group_top scale routes are defined for six-task groups; use per_idea_seeded')
        if set(ops) - OPERATIONAL_KEYS - OPTIONAL_OPERATIONAL_KEYS or OPERATIONAL_KEYS - set(ops):
            raise ConfigError('operations keys differ from the schema: required ' + ', '.join(sorted(OPERATIONAL_KEYS))
                              + '; optional ' + ', '.join(sorted(OPTIONAL_OPERATIONAL_KEYS)))
        for key in ('codex_home', 'codex_executable', 'codex_sha256'):
            if ops.get(key) is not None and (not isinstance(ops[key], str) or not ops[key].strip()):
                raise ConfigError(key + ' must be a non-empty string or null')
        if ops.get('codex_sha256') is not None and (len(ops['codex_sha256']) != 64 or set(ops['codex_sha256']) - set('0123456789abcdef')):
            raise ConfigError('codex_sha256 must be a lowercase SHA-256 hex digest')
        if type(ops['workers']) is not int or ops['workers'] < 2:
            raise ConfigError('workers (concurrent Codex calls) must be an integer of at least 2')
        if type(ops['timeout_seconds']) is not int or ops['timeout_seconds'] < 1:
            raise ConfigError('timeout_seconds must be a positive integer')
        if type(ops.get('slate_barriers', False)) is not bool: raise ConfigError('slate_barriers must be boolean')
        if type(ops.get('allow_concurrent_activity', False)) is not bool: raise ConfigError('allow_concurrent_activity must be boolean')
        if type(ops.get('usage_credits', False)) is not bool: raise ConfigError('usage_credits must be boolean')
        if type(ops.get('continuous_supply', False)) is not bool: raise ConfigError('continuous_supply must be boolean')
        if 'supply_windows' in ops and (type(ops['supply_windows']) is not int or ops['supply_windows'] < 1):
            raise ConfigError('supply_windows must be a positive integer')
        if 'idle_supply_windows' in ops and (type(ops['idle_supply_windows']) is not int
                or ops['idle_supply_windows'] < ops.get('supply_windows', 2) or not ops.get('continuous_supply')):
            raise ConfigError('idle_supply_windows needs continuous supply and at least supply_windows')
        ramp = ops.get('ramp_up')  # start-up pacing of the worker pool: one acquisition per gap_seconds for window_seconds
        if ramp is not None and (not isinstance(ramp, dict) or set(ramp) != {'gap_seconds', 'window_seconds'}
                                 or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in ramp.values())
                                 or not 0 <= ramp['gap_seconds'] <= 600 or not 0 <= ramp['window_seconds'] <= 7200):
            raise ConfigError('ramp_up: {gap_seconds in [0,600], window_seconds in [0,7200]}')
        pool = ops.get('pool')
        if pool is not None and (not isinstance(pool, dict) or set(pool) != {'downstream_workers', 'upstream_workers'}
                                 or any(type(v) is not int or v < 1 for v in pool.values())):
            raise ConfigError('pool: {downstream_workers, upstream_workers}, positive integers (default: derived from workers)')
        if not 1 <= ops['group_size'] <= 6: raise ConfigError('group_size must be 1..6')
        if 'development_windows' in ops and (type(ops['development_windows']) is not int or ops['development_windows'] < 1):
            raise ConfigError('development_windows must be a positive integer')
    except KeyError as exc:
        raise ConfigError(f'missing config field: {exc}') from None
    return value


def science_hash(value):
    return digest({k: v for k, v in value.items() if k != 'operations' and not k.startswith('_')})


def apply_amendments(value, run_root):
    value = deepcopy(value); applied = []
    for path in sorted(Path(run_root, 'config_amendments').glob('*.json')):
        amendment = json.loads(path.read_text())
        ops = amendment['operations']
        if set(ops) - AMENDABLE_KEYS: raise ConfigError('amendments may only change ' + ', '.join(sorted(AMENDABLE_KEYS)) + ': ' + path.name)
        value['operations'].update(deepcopy(ops)); applied.append({'file': path.name, 'sha256': sha256(path.read_bytes()).hexdigest()})
    value['_amendments'] = applied
    return validate(value)


def load(path=None, run_root=None):
    path = Path(path or os.environ.get('ARCGEN_CONFIG') or PROFILES / 'v1.json')
    value = validate(json.loads(path.read_text()))
    run_root = run_root or os.environ.get('ARCGEN_RUN_ROOT')
    if run_root and Path(run_root, 'config_amendments').is_dir():
        value = apply_amendments(value, run_root)
    value['_source'] = str(path)
    value['operations'] = expand_operations(value['operations'])
    return value


CONFIG = load()
SCIENCE = CONFIG['science']
OPERATIONS = CONFIG['operations']


def binding(pair):
    model, effort = pair
    return {'model': SCIENCE['models'][model], 'reasoning_effort': effort}
