"""Native admission (r10) with planner-aware task identities and configured planning routes."""
from copy import deepcopy
from arcgen.lib.factory_authoring_screen import initial_record
from arcgen.astra_targeted_followup_20260911.admission import assign_routes
from .common import frozen, digest, IntegrityError
from .pairing import task_id
from .config import CONFIG, SCIENCE, RUNGS, ALL_RUNGS, rung_shares, LINE_MODIFIERS, LINE_CONTROL, LINE_REGISTRY, SIZE_ARMS
from . import slots

SPECIALIST_ROUTES = ('logical_leap', 'episode_joint_inference')
# Arms whose ideas keep one colouring and no pose instruction: STAGES chains each training output into the next input,
# which per-pair colour variation or pose would break; CONTRAST compares pairs that differ only in what matters, which
# varying colours or pose between pairs would blur.
FIXED_PRESENTATION_ARMS = frozenset({'STAGES', 'CONTRAST'})
# The same holds for the episode-slot versions of those properties on other arms (2026-10-09).
FIXED_PRESENTATION_SLOTS = frozenset({'CHAINED', 'MINIMAL'})
# COLOURFUL gives each colour job one fixed colour across the pairs and the test, which per-pair colour variation breaks
# (v1-b15 screen: 15 of 20 COLOURFUL tasks drew 'vary' and their role colours rotated per pair). Pose is unaffected.
FIXED_COLOUR_SLOTS = frozenset({'COLOURFUL'})
# Canvas lines (prompts/v1/canvas.<kind>.planner.md), drawn per idea by science.canvas_draw (eval-style study F-04).
CANVAS_KINDS = ('square', 'margin', 'both')
# Ideas that already spread their evidence over many pairs never get the MANY overlay (more pairs): only XFER.
MANY_PAIR_ARMS, MANY_PAIR_SLOTS = frozenset({'SPREAD'}), frozenset({'DISTRIBUTED'})


def fixed_presentation(study):
    # ONE (2026-10-23) and its matched control keep one colouring and no pose too: with one training pair, a role colour or
    # pose first seen at test is a kind of case the pair does not show (slots.py)
    return (study.get('primary_arm') in FIXED_PRESENTATION_ARMS or bool(FIXED_PRESENTATION_SLOTS & set(study.get('idea_modifiers') or ()))
            or slots.bound(study))


def many_pair(study):
    return study.get('primary_arm') in MANY_PAIR_ARMS or bool(MANY_PAIR_SLOTS & set(study.get('idea_modifiers') or ()))


def idea_line(study):
    """A weakness run's line for an idea (2026-10-20): its pinned line modifier, or the control line for a matched-control
    call (weakness_lines.json)."""
    found = [m for m in study.get('idea_modifiers') or () if m in LINE_MODIFIERS]
    if len(found) > 1:
        raise IntegrityError('an idea carries more than one weakness line')
    return found[0] if found else LINE_CONTROL if study.get('line_control') else None


def _uniform(policy, row):
    return int(digest({'policy': policy, 'seed': CONFIG['seed'], 'source': row['source_candidate_id']})[:12], 16) / 16 ** 12


def _structure(study, line, weakness):
    """Structure lines (2026-10-22, STRUCTURE_SPEC.md amendment v2 S5): the call's structure line (study.structure_call) on
    top of the idea's content line or arm, with the host matrix enforced here: a host the structure is never drawn on keeps
    its content line only and records why (study.structure_excluded); a host on which the structure is the host's own
    form is counted as the host (study.structure_counted_as). The matched control draws none. Returns the registry entry
    of the structure drawn, or None."""
    name = study.get('structure_call')
    if not name:
        return None
    entry = (LINE_REGISTRY.get('structures') or {}).get(name)
    if entry is None or name not in (weakness.get('structures') or {}):
        raise IntegrityError(f'a structure call needs a structure line of science.weakness.structures, not {name}')
    if study.get('line_control') or line == LINE_CONTROL:
        raise IntegrityError('the matched control draws no structure line')
    if line in (entry.get('not_on') or ()):
        study['structure_excluded'] = {'structure': name, 'reason': 'host', 'host': line}
        return None
    study['structure'] = name
    if line in (entry.get('counted_as') or {}):
        study['structure_counted_as'] = entry['counted_as'][line]
    return entry


STRUCTURE_RUNGS = frozenset({'R1', 'R2'})  # the structure screen's sizes (S5), for an arm control's fixed size


def _allowed(entry, relaxation):
    """The size levels an idea may take: its structure line's and its relaxed fence's (when they name any), intersected."""
    allowed = set(ALL_RUNGS)
    for source in (entry, relaxation):
        if source and source.get('rungs'):
            allowed &= set(source['rungs'])
    return allowed


def _weakness_draws(row, weakness):
    """Weakness run (2026-10-20): the idea's line, its planner overlay and,
    for a line idea, its size arm and rung. Overlays (science.weakness.overlays, in registry order) share one seeded uniform,
    so an idea draws at most one; an eligible idea that does not draw one is that overlay's control (study.weak_overlays).
    Excluded names, and fixed colours for an overlay that varies colour roles, take no draw. A call that fixes a rung takes
    it; a call that fixes its size arm draws the rung inside the arm from the line's mix (science.weakness.lines.<LINE>.rungs)
    renormalised; otherwise the line's whole mix decides and the arm follows from the rung. A call's fence is a relaxed
    form, "strict", or "relaxed" (each idea draws one of the line's relaxed forms at their relative shares); a call that
    names none draws per idea at the run's shares."""
    study = row['study']
    line = idea_line(study)
    if slots.bound(study) and (line or study.get('structure_call') or study.get('structure_control') or study.get('line_control')):
        # ONE (2026-10-23): every weakness and structure line, and both matched controls, need evidence across pairs
        raise IntegrityError('a ONE idea or its matched control carries no weakness line, structure line or line control')
    if line:
        study['line'] = line
    structure = _structure(study, line, weakness)
    names = {study.get('primary_arm'), row.get('private_route'), *(study.get('idea_modifiers') or ()), line} - {None}
    pick, edge, draws = _uniform('weakness-overlay-per-idea-v1', row), 0.0, {}
    for name, entry in LINE_REGISTRY['overlays'].items():
        share = (weakness.get('overlays') or {}).get(name)
        if not share:
            continue
        low, edge = edge, edge + share['share']
        fixed = entry.get('colour_vary') and (fixed_presentation(study) or FIXED_COLOUR_SLOTS & set(study.get('idea_modifiers') or ()))
        if names & set(share.get('excluded', ())) or fixed or slots.bound(study):  # EDGE needs cut things in two pairs
            continue
        if structure and structure.get('replaces_overlay') == name:  # VARY is drawn as one with ROLES (S5): no ROLES draw
            continue
        draws[name] = 'drawn' if low <= pick < edge else 'control'
        if draws[name] == 'drawn' and entry.get('colour_vary'):  # its binding line rides on the colour-variation planner
            study['colour_roles_before_overlay'] = study.get('colour_roles')  # the mix's own draw, for the comparison (N6)
            study['colour_roles'] = 'vary'
    if draws:
        study['weak_overlays'] = draws
    fixed_colours = fixed_presentation(study) or FIXED_COLOUR_SLOTS & set(study.get('idea_modifiers') or ())
    if structure and structure.get('colour_vary') and not fixed_colours and study.get('colour_roles') != 'vary':
        study['colour_roles_before_overlay'] = study.get('colour_roles')  # VARY varies role colours, as ROLES does
        study['colour_roles'] = 'vary'
    if not line:
        if structure:  # an arm idea with a structure line: the call's size, within the levels the structure allows
            _structure_size(study, _allowed(structure, None))
        elif study.get('structure_control'):  # its matched control on the same arm takes the same size
            _structure_size(study, STRUCTURE_RUNGS)
        elif slots.bound(study):  # ONE (2026-10-23) and its matched control: the call's size, R1-R3 (slots.ONE_RUNGS)
            _structure_size(study, slots.ONE_RUNGS)
        elif study.get('size_arm') or study.get('size_rung'):
            # any other arm idea whose call fixes a size (2026-10-25, composite batch c1: skill lines and their matched
            # controls, {"slot_control": SKILL}) takes it as ONE does, R1-R3 (R4 stays a weakness line's rung); a call that
            # fixes none keeps the run's general size draw
            _structure_size(study, frozenset(RUNGS))
        return
    entry, cfg = LINE_REGISTRY['lines'][line], weakness['lines'][line]
    decoys = {k for k, r in (entry.get('relax') or {}).items() if r.get('decoy')}
    # a planted negative (DESIGN §6.9, JUDGE's decoy calls): fixed by the spec row, recorded as study.judge_decoy
    if study.get('judge_decoy'):
        if not decoys or study.get('fence_call') not in (None, *decoys):
            raise IntegrityError(f'a decoy call needs a line with a decoy form, not {line}')
        study['fence_call'] = sorted(decoys)[0]
    elif study.get('fence_call') in decoys:
        raise IntegrityError('a decoy form is set by the spec row\'s decoy flag only')
    # the relaxed fence (v2 §5, draft 4 §6.4): fixed by the spec row, or drawn per idea at the run's shares; the rest strict
    if entry.get('relax'):
        fence, shares = study.get('fence_call'), dict(cfg.get('relaxed') or {})
        if fence == 'relaxed':  # a relaxed call: every idea takes one relaxed form, at the forms' relative shares
            if not shares:
                raise IntegrityError(f'a relaxed {line} call needs science.weakness.lines.{line}.relaxed')
            total = sum(shares.values())
            shares = {k: v / total for k, v in shares.items()}
        if fence in (None, 'relaxed') and shares:
            pick, edge, fence = _uniform('weakness-fence-per-idea-v1', row), 0.0, 'strict'
            ordered = sorted(shares.items())
            for name, share in ordered:
                edge += share
                if pick < edge:
                    fence = name
                    break
            if study.get('fence_call') == 'relaxed' and fence == 'strict':  # rounding at the top of the unit interval
                fence = ordered[-1][0]
        if fence not in (None, 'strict', *entry['relax']):
            raise IntegrityError(f'{line} has no relaxed fence {fence}')
        if fence:
            study['fence'] = fence
    elif study.get('fence_call') not in (None, 'strict'):
        raise IntegrityError(f'{line} has no relaxed fences')
    relaxation = (entry.get('relax') or {}).get(study.get('fence')) if study.get('fence') not in (None, 'strict') else None
    allowed = _allowed(structure, relaxation)
    if 'rung' in study:
        if study['rung'] not in allowed:
            raise IntegrityError(f'{study["rung"]} is outside the levels this idea allows')
        return
    rung, mix = study.get('size_rung'), cfg['rungs']
    if rung is not None and rung not in allowed:
        raise IntegrityError(f'{rung} is outside the levels this idea allows ({", ".join(sorted(allowed))})')
    if allowed != set(ALL_RUNGS):  # S5: the content line's size mix, limited to the levels the structure (or fence) allows
        mix = {k: v for k, v in mix.items() if k in allowed}
    if rung is None:  # within the call's size arm, the line's mix renormalised; with no arm, the whole mix
        arm = study.get('size_arm')
        if arm not in (None, *SIZE_ARMS):
            raise IntegrityError('a call fixes a size arm the weakness batch does not know: ' + str(arm))
        none = max(0.0, 1 - sum(cfg['rungs'].values()))
        if arm == 'small':
            total = mix.get('R1', 0) + none
            shares = {'R1': mix.get('R1', 0) / total} if total else {}
        elif arm == 'large':
            total = sum(v for k, v in mix.items() if k in SIZE_ARMS['large'])
            shares = {k: v / total for k, v in mix.items() if k in SIZE_ARMS['large']} if total else {}
        else:
            shares = dict(mix)
            if allowed != set(ALL_RUNGS):  # the dropped levels' share goes to the allowed ones, as within an arm
                total = sum(shares.values()) + none
                shares = {k: v / total for k, v in shares.items()} if total else {}
        pick, edge = _uniform('weakness-rung-per-idea-v1', row), 0.0
        for r, share in sorted(shares.items(), key=lambda kv: ALL_RUNGS.index(kv[0])):
            edge += share
            if pick < edge:
                rung = r
                break
        if arm == 'large' and rung is None:
            raise IntegrityError(f'{line} has no large rung in its size mix')
    if rung == 'R4' and entry.get('r4') is False:
        raise IntegrityError(f'{line} never takes the full-scale rung R4')
    if rung:
        study['rung'] = rung
    study.setdefault('size_arm', 'large' if rung in SIZE_ARMS['large'] else 'small')


def _structure_size(study, allowed):
    """An arm idea carrying a structure line: a fixed rung or size rung must be allowed; a small arm takes R1 and a large
    arm the largest allowed level of the large arm; with neither, the run's general size draw applies, capped at the
    largest allowed level (configured_routes)."""
    rung = study.get('rung') or study.get('size_rung')
    arm = study.get('size_arm')
    if rung is None and arm == 'small':
        rung = ALL_RUNGS[0]
    elif rung is None and arm == 'large':
        rung = max((r for r in SIZE_ARMS['large'] if r in allowed), key=ALL_RUNGS.index, default=None)
    if rung is None:
        return
    if rung not in allowed:
        raise IntegrityError(f'{rung} is outside the levels this structure line allows ({", ".join(sorted(allowed))})')
    study['rung'] = rung
    study.setdefault('size_arm', 'large' if rung in SIZE_ARMS['large'] else 'small')


def configured_routes(config, rows):
    """Scale-overlay routes per the batch config.

    per_group_top: r23 behavior, the top-ranked ordinary row of each group is invited.
    per_idea_seeded: a seeded draw per kernel at the configured rate; every planner version of an
    idea shares its route, independent of grouping. none: no scale invitations.
    J/T specialist routes always follow their source arm.
    """
    policy = SCIENCE['routes']['scale_overlay']
    if policy['assignment'] == 'per_group_top':
        return assign_routes(config, rows)
    rows = deepcopy(rows)
    for row in rows:
        source = row['private_source']
        route = source if source in SPECIALIST_ROUTES else 'ordinary_current'
        if route == 'ordinary_current' and policy['assignment'] == 'per_idea_seeded':
            draw = int(digest({'policy': 'scale-overlay-per-idea-v1', 'seed': CONFIG['seed'],
                               'source': row['source_candidate_id']})[:12], 16) / 16 ** 12
            if draw < policy['rate']:
                route = 'evidence_preserving_scale'
                kinds = policy.get('kinds')
                if kinds and SCIENCE.get('weakness') and (idea_line(row['study']) or row['study'].get('structure_call')
                                                          or row['study'].get('structure_control')):
                    # weakness lines and the matched control (review S9) take no presentation overlay: MANY's five to eight
                    # pairs would swamp their pair counts, and XFER's extra tests would multiply full-scale outputs and
                    # the declared points per task; the screen keeps one test per task and comparable sizes
                    kinds, route = None, 'ordinary_current'
                elif kinds and (many_pair(row['study']) or slots.bound(row['study'])):
                    # ONE (2026-10-23) and its matched control fix their own pair count: no MANY (five to eight pairs)
                    kinds = [k for k in kinds if k != 'MANY'] or None
                    if kinds is None:  # nothing left to invite: an ordinary plan
                        route = 'ordinary_current'
                if kinds:  # v1: one presentation overlay per idea, drawn independently of the rate draw
                    pick = int(digest({'policy': 'presentation-overlay-kind-v1', 'seed': CONFIG['seed'],
                                       'source': row['source_candidate_id']})[:12], 16) % len(kinds)
                    row['study']['overlay'] = sorted(kinds)[pick]
        row['private_route'] = route
        if SCIENCE.get('colour_roles') == 'mix':  # per-idea colour instruction, independent of the route draws
            pick = int(digest({'policy': 'colour-vary-per-idea-v1', 'seed': CONFIG['seed'],
                               'source': row['source_candidate_id']})[:12], 16) / 16 ** 12
            row['study']['colour_roles'] = 'vary' if pick < SCIENCE['colour_vary_rate'] else 'v1'
        weakness = SCIENCE.get('weakness')
        if weakness:  # weakness run: line, overlay and size arm before the other draws read them
            _weakness_draws(row, weakness)
        if SCIENCE.get('pose_rate') is not None:  # per-idea pose instruction for the planner
            pick = int(digest({'policy': 'pose-per-idea-v1', 'seed': CONFIG['seed'],
                               'source': row['source_candidate_id']})[:12], 16) / 16 ** 12
            row['study']['pose'] = pick < SCIENCE['pose_rate']
            # Turning, mirroring or resizing breaks the fit of pieces, the axes of a structure and the laws of a simulation.
            if row['study'].get('family') in SCIENCE.get('pose_off_families', ()):
                row['study']['pose'] = False
        draw = SCIENCE.get('rung_draw')
        # a general run's binding scale line, drawn per idea as pose is; a weakness line idea took its own size draw above
        if draw and 'rung' not in row['study'] and not row['study'].get('line'):
            excluded = set(draw['excluded'])
            names = set(row['study'].get('idea_modifiers') or ()) | {row['study'].get('overlay')}
            if not excluded & names:
                pick = int(digest({'policy': 'rung-per-idea-v1', 'seed': CONFIG['seed'],
                                   'source': row['source_candidate_id']})[:12], 16) / 16 ** 12
                edge = 0.0
                # round 3 (2026-10-20): an idea carrying a capped name keeps a draw at or below its cap and takes no rung
                # above it, so the other ideas' draws are unchanged
                cap = min((RUNGS.index(r) for n, r in (draw.get('max_rung') or {}).items() if n in names), default=len(RUNGS))
                structure = (LINE_REGISTRY.get('structures') or {}).get(row['study'].get('structure'))
                if structure:  # an arm idea with a structure line takes no level above the structure's (S5)
                    cap = min(cap, max(RUNGS.index(r) for r in structure['rungs'] if r in RUNGS))
                for rung, share in rung_shares(draw):  # one rung at a rate keeps its draws; a ladder walks the shares
                    edge += share
                    if pick < edge:
                        if RUNGS.index(rung) <= cap:
                            row['study']['rung'] = rung
                        break
        canvas = SCIENCE.get('canvas_draw')
        if canvas:  # eval-style study F-04: a per-idea canvas line for the planner, seeded as pose is; undrawn ideas are the control
            pick = int(digest({'policy': 'canvas-per-idea-v1', 'seed': CONFIG['seed'],
                               'source': row['source_candidate_id']})[:12], 16) / 16 ** 12
            choice, edge = None, 0.0
            for kind in CANVAS_KINDS:
                edge += canvas.get(kind, 0)
                if pick < edge:
                    choice = kind; break
            # open margins would inflate canvas-measured scale floors and fight dense scenes: those ideas get square only
            if choice in ('margin', 'both') and (set(canvas.get('margin_excluded', ())) & (
                    {row['study'].get('rung')} | set(row['study'].get('idea_modifiers') or ()))
                    or any(LINE_REGISTRY['overlays'][k].get('no_margin_canvas') for k, v in (row['study'].get('weak_overlays') or {}).items()
                           if v == 'drawn')):  # weakness run: an overlay with content at the border (EDGE)
                choice = 'square' if choice == 'both' else None
            if choice:
                row['study']['canvas'] = choice
        teach = SCIENCE.get('teach_draw')
        # b17 review 2026-10-15: the teaching line for the planner, seeded per idea; the control keeps the plan as before.
        # Arms, slots and routes that fix their own pair structure (joint inference, chains of states) take no draw.
        names = {row['study'].get('primary_arm'), row.get('private_route'), *(row['study'].get('idea_modifiers') or ())}
        # weakness lines (2026-10-20) fix their own pair structure: never the teaching line
        # structure lines (2026-10-22) lay out the evidence themselves: never the teaching line either
        # ONE (2026-10-23) and its matched control: the teaching line asks for every law in two pairs, never with ONE
        if (teach and not names & set(teach.get('excluded', ())) and not row['study'].get('line')
                and not row['study'].get('structure') and not row['study'].get('structure_control') and not slots.bound(row['study'])):
            pick = int(digest({'policy': 'teach-per-idea-v1', 'seed': CONFIG['seed'],
                               'source': row['source_candidate_id']})[:12], 16) / 16 ** 12
            row['study']['teach'] = 'plain' if pick < teach['plain'] else 'control'
        depart = SCIENCE.get('depart_draw')
        # departure package (2026-10-16): the first test meets one visible situation no pair shows, for a seeded share of
        # ideas; arms, slots, rungs, families or routes that fix their own test novelty or pair structure take no draw.
        names = {row['study'].get('primary_arm'), row.get('private_route'), row['study'].get('rung'), row['study'].get('family'),
                 *(row['study'].get('idea_modifiers') or ())}
        if (depart and not names & set(depart.get('excluded', ())) and not row['study'].get('line')  # one line per idea
                and not row['study'].get('structure') and not row['study'].get('structure_control')
                and not slots.bound(row['study'])):  # ONE: the test meets no case the one pair does not show
            pick = int(digest({'policy': 'depart-per-idea-v1', 'seed': CONFIG['seed'],
                               'source': row['source_candidate_id']})[:12], 16) / 16 ** 12
            row['study']['depart'] = 'test' if pick < depart['test'] else 'control'
        gate_draw = SCIENCE.get('gate_draw')
        if gate_draw:  # gate study 2026-10-15: the clear rule for flagged but right blind answers; undrawn ideas are the control
            pick = int(digest({'policy': 'gate-per-idea-v1', 'seed': CONFIG['seed'],
                               'source': row['source_candidate_id']})[:12], 16) / 16 ** 12
            row['study']['gate'] = 'right_twice' if pick < gate_draw['right_twice'] else 'control'
        if fixed_presentation(row['study']):
            if 'colour_roles' in row['study']:
                row['study']['colour_roles'] = 'v1'
            if 'pose' in row['study']:
                row['study']['pose'] = False
        elif FIXED_COLOUR_SLOTS & set(row['study'].get('idea_modifiers') or ()) and 'colour_roles' in row['study']:
            row['study']['colour_roles'] = 'v1'
    return rows


def prepare(runtime, window, allocation):
    rows = deepcopy(allocation['rows'])
    routed = configured_routes(runtime.factory_config, rows) if rows else []
    result = []
    for row in routed:
        row['slate_index'] = 1
        tid = task_id(window['id'], row)
        initial = initial_record({'task_id': tid, 'source_candidate_id': row['source_candidate_id'],
            'lineage_id': row['lineage_id'], 'arm': 'single', 'private_source': row['private_source'],
            'private_conditioned': row['private_conditioned'], 'private_source_call_id': row['private_source_call_id'],
            'private_idea_effort': row['private_effort'], 'private_route': row['private_route'],
            'slate_index': 1, 'source_development_priority': row['study']['dispatch_ordinal'],
            'search_mode': row['search_mode'], 'kernel': row['kernel'],
            'editorial_brief': row['editorial_brief'], 'novelty': row['novelty']})
        initial['study'] = deepcopy(row['study'])
        if SCIENCE.get('prompt_set', 'r23') == 'v1':
            initial['selection_note'] = {'useful_contribution': (row.get('selector_assessment') or {}).get('useful_contribution')}
        runtime.budget.reserve_development(tid, runtime.phase, window['id'], digest(initial), window['maximum_developments'])
        runtime.palette_allocator.register(tid, window['account'], window['id'], digest(initial))
        result.append({'task_id': tid, 'initial': initial, 'selected': row})
    # None is an explicit deferred target, not a request to skip the new hook.
    manifest = frozen(runtime.run_dir/'admission/manifest.private.json', {
        'window': window, 'rows': result, 'assignments': {r['task_id']: None for r in result},
        'palette_allocation': 'batch-background-mixture-v1/deferred',
        'palette_context_hash': runtime.palette_allocator.context_hash,
        'allocation_hash': digest(allocation), 'source_manifest_hash': digest(runtime.source_manifest)})
    if set(runtime.state['semantic_records']) - {r['task_id'] for r in result}:
        raise IntegrityError('foreign candidate in window runtime')
    for row in result:
        previous = runtime.state['semantic_records'].get(row['task_id'])
        if previous:
            for key in ('task_id', 'source_candidate_id', 'private_source', 'private_source_call_id', 'study'):
                if previous[key] != row['initial'][key]:
                    raise IntegrityError('resumed candidate attribution changed')
        else:
            runtime.state['semantic_records'][row['task_id']] = deepcopy(row['initial'])
    runtime.save()
    return manifest
