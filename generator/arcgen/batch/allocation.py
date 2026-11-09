"""Batch design from the configured allocation table: supply windows, development groups and ceilings.

Ceilings are derived, not hand-written. Each supply window can yield at most its kernel count of
eligible ideas, plus extra planner versions for the multi-planner fraction; groups hold
`group_size` developments. With static accounts every group belongs to its supply window's
account. With dynamic accounts a group is assigned to whichever account has capacity when it
starts, so each account's development ceilings cover the whole batch and the global ceilings bind.
"""
from collections import Counter
from math import ceil
from .common import ROOT, REVISION, HOMES, SETTINGS, read, IntegrityError
from .config import CONFIG, SCIENCE, OPERATIONS, FAMILIES, RUNGS, ALL_RUNGS
from . import supply

DEFAULT_RUN = ROOT / CONFIG['default_run']
TURNS = {'unconditioned': 1, 'bank_aware': 1, 'bank_inspired': 1, 'loop': 2}  # logical idea calls per author call
PHASE = CONFIG['phase']
SEED = CONFIG['seed']
GROUP_SIZE = OPERATIONS['group_size']
DYNAMIC = OPERATIONS['account_assignment'] == 'dynamic'
from .common import IDEA_MODELS
PROBES_PER_ACCOUNT = 3 * len(IDEA_MODELS)  # every idea model x high/xhigh/max route probes
# With scale rungs a plan below its rung's floors gets one re-plan (runtime.py), a fifth primary call per task; a weakness
# run's plan-field check shares that one re-plan.
# ONE (2026-10-23, slots.py) shares that re-plan: a plan with more than one training pair is planned once more.
PRIMARY_PER_TASK = 5 if SCIENCE.get('rungs') or SCIENCE.get('weakness') or (SCIENCE.get('idea_modifiers') or {}).get('ONE') else 4


def development_capacity(kernels):
    pa = SCIENCE['planner_assignment']
    paired = ceil(kernels * pa['multi_planner_fraction']) if len(pa['multi_planners']) > 1 else 0
    return kernels + paired * max(0, len(pa['multi_planners']) - 1)


def assign_modifiers(calls, rates, slots=None, cap=None, exclusions=None, arm_caps=None, bare=None):
    """Exact shares of the idea-call modifiers (property slots), written to each call's `idea_modifiers`.

    Each modifier goes to round(rate x calls) calls, taken in the order of a per-call hash: still random, but with no
    count variance (v1-b10 drew BUSY on 14 of 112 calls against a 25% rate, and its answers came out smaller). Values
    sharing a slot exclude each other, a call holds at most `cap` slots, and a modifier skips calls whose arm or other
    modifiers it excludes, or that exclude it. The rarest values are placed first so the cap never starves them.
    `arm_caps` lowers the cap for some arms; `bare` first reserves calls with no modifiers at all ({arm: n}, '*' for
    any arm), so a new arm is also seen without them. Shares stay fractions of all calls, bare ones included.
    Exclusions may also name a call's family or rung (targeted runs), which act like its arm.
    Calls with `pinned_modifiers` (a spec row's skill) hold them for certain: they are placed first, count toward their
    modifier's share and the call's cap, and must not break a slot, an exclusion or a reservation.
    Without slots, cap, exclusions or reservations this is the plain per-modifier exact draw. Returns {modifier: calls short}."""
    from .common import digest
    slot_of = {m: s for s, values in (slots or {m: [m] for m in rates}).items() for m in values}
    known = set(rates) | {c['arm'] for c in calls} | set(supply.idea_v1.ARMS) | set(FAMILIES) | set(ALL_RUNGS)
    excl = {m: set(v) for m, v in (exclusions or {}).items()}
    if set(slot_of) != set(rates) or any(set(v) - known for v in excl.values()):
        raise IntegrityError('modifier slots or exclusions name unknown modifiers or arms')
    if any(a != '*' and a not in known for a in list(arm_caps or {}) + list(bare or {})):
        raise IntegrityError('modifier arm caps or reservations name unknown arms')
    limit = {c['opaque_id']: min(cap if cap is not None else len(rates), (arm_caps or {}).get(c['arm'], len(rates))) for c in calls}
    for arm, n in sorted((bare or {}).items(), key=lambda kv: kv[0] == '*'):  # named arms first, then any arm
        pool = [c for c in calls if (arm == '*' or c['arm'] == arm) and limit[c['opaque_id']] > 0
                and not c.get('pinned_modifiers')]  # a call with a pinned skill is never kept bare
        pool.sort(key=lambda c: digest({'policy': 'idea-modifier-bare-v1', 'call': c['opaque_id']}))
        if arm != '*' and not any(c['arm'] == arm for c in calls):
            continue  # an arm this allocation does not use needs no reservation
        if len(pool) < n:
            raise IntegrityError(f'cannot keep {n} calls free of modifiers for {arm}')
        for c in pool[:n]:
            limit[c['opaque_id']] = 0
    chosen = {c['opaque_id']: [] for c in calls}
    for c in calls:
        for m in c.get('pinned_modifiers') or ():
            have = chosen[c['opaque_id']]
            if m not in rates:
                raise IntegrityError(f"pinned modifier {m} has no share in this profile (call {c['opaque_id']})")
            if (any(slot_of[v] == slot_of[m] for v in have) or len(have) >= limit[c['opaque_id']]
                    or {c['arm'], c.get('family'), c.get('rung')} & excl.get(m, set())
                    or any(v in excl.get(m, ()) or m in excl.get(v, ()) for v in have)):
                raise IntegrityError(f"pinned modifier {m} does not fit call {c['opaque_id']} ({c['arm']})")
            have.append(m)
    shortfall = {}
    for m in sorted(rates, key=lambda m: (rates[m], m)):
        target = round(rates[m] * len(calls))
        placed = sum(m in chosen[c['opaque_id']] for c in calls)
        if placed > target:
            raise IntegrityError(f'more calls pin {m} ({placed}) than its share gives ({target})')
        for c in sorted(calls, key=lambda c: digest({'policy': 'idea-modifier-exact-v1', 'modifier': m, 'call': c['opaque_id']})):
            if placed == target:
                break
            if c.get('slot_control') == m:  # a slot's matched control (2026-10-23) never draws that slot
                continue
            have = chosen[c['opaque_id']]
            if (any(slot_of[v] == slot_of[m] for v in have) or len(have) >= limit[c['opaque_id']]
                    or {c['arm'], c.get('family'), c.get('rung')} & excl.get(m, set())
                    or any(v in excl.get(m, ()) or m in excl.get(v, ()) for v in have)):
                continue
            have.append(m); placed += 1
        if placed < target:
            shortfall[m] = target - placed
    for c in calls:
        c['idea_modifiers'] = sorted(chosen[c['opaque_id']])
    return shortfall


SELECTOR_KINDS = ('fresh_supply', 'saved_ideas')  # windows that run a selector


def design():
    from arcgen.paths import resolve
    proposal = read(resolve(SCIENCE['allocation']))
    if proposal['revision'] != REVISION:
        raise IntegrityError('allocation table belongs to another revision')
    windows = []
    for pairing_index, (wid, spec) in enumerate(proposal['windows'].items()):
        cells = [c for c in proposal['calls'] if c['window'] == wid]
        if not cells: raise IntegrityError('window has no author calls: ' + wid)
        if spec['account'] not in HOMES: raise IntegrityError('window account is not configured: ' + wid)
        first = cells[0]
        w = dict(id=wid, phase=PHASE, wave=spec['wave'], account=spec['account'],
                 context=spec['context'], seed=first['seed'], kind='fresh_supply',
                 supply_only=True, cohort=REVISION, editorial_mode='bypass',
                 maximum_developments=0, minimum_selector_eligible_kernels=1,
                 maximum_retained=sum(len(c['candidate_ids']) for c in cells), protected=False, protected_targets=[], calls=[],
                 pairing_index=pairing_index)
        if 'reference_wave' in spec:  # in-batch references (waves.py); only with science.batch_references
            if not SCIENCE.get('batch_references'):
                raise IntegrityError('reference waves need science.batch_references')
            w['reference_wave'] = spec['reference_wave']
        for j, cell in enumerate(cells, 1):
            if (cell['account'], cell['wave'], cell['context']) != (w['account'], w['wave'], w['context']):
                raise IntegrityError('call and window bindings disagree')
            c = supply.make_call(arm=cell['arm'], invitation=cell['invitation'],
                bank_setting=cell['bank_setting'], effort=cell['effort'], idea_model=cell['model'],
                context=w['context'], seed=w['seed'], account=w['account'], window=wid, replicate=j,
                family=cell.get('family'), rung=cell.get('rung'))
            if cell.get('commit') in ('kept', 'explore'):
                c['commit'] = cell['commit']
            if cell.get('skill'):  # pinned idea modifier (skill runs, 2026-10-18): placed before the drawn shares
                c['pinned_modifiers'] = [cell['skill']] if isinstance(cell['skill'], str) else list(cell['skill'])
            if cell.get('control'):  # weakness run (2026-10-20): a matched-control call, no line (study.line CONTROL)
                c['line_control'] = True
            if cell.get('size'):  # weakness run: the call's size arm or rung (study.size_arm, study.size_rung)
                c['size_arm'] = cell['size']
            if cell.get('fence'):  # weakness run: the call's relaxed fence, or strict (study.fence)
                c['fence'] = cell['fence']
            if cell.get('decoy'):  # weakness run: a planted negative for JUDGE's consensus procedure (study.judge_decoy)
                c['judge_decoy'] = True
            if cell.get('structure'):  # structure lines (2026-10-22): the call's structure line (study.structure_call)
                c['structure'] = cell['structure']
            if cell.get('structure_control'):  # the structure screen's matched control: the same host, no structure line
                c['structure_control'] = True
            if cell.get('slot_control'):  # a slot's matched control (2026-10-23, ONE; 2026-10-25, a skill line): the same
                # arm and size, no slot
                c['slot_control'] = cell['slot_control']
            c.update(private_id=cell['private_id'], opaque_kernel_ids=cell['candidate_ids'],
                     wave=w['wave'], allocation_cell=f'{wid}-{j}', pilot_revision=REVISION)
            w['calls'].append(c)
        windows.append(w)
    # a saved-only design (rescue r1, 2026-10-25) has no idea call to give a modifier, nor to keep bare (modifier_bare)
    if SCIENCE.get('modifier_assignment') == 'exact' and SCIENCE.get('idea_modifiers') and any(w['calls'] for w in windows):
        short = assign_modifiers([c for w in windows for c in w['calls']], SCIENCE['idea_modifiers'], SCIENCE.get('modifier_slots'),
                                 SCIENCE.get('modifier_cap'), SCIENCE.get('modifier_exclusions'),
                                 SCIENCE.get('modifier_arm_caps'), SCIENCE.get('modifier_bare'))
        if short:  # the slots, cap or exclusions leave too few eligible calls: change the rates or the allocation
            raise IntegrityError(f'modifier shares cannot be met (calls short: {short})')
    from hashlib import sha256
    queued_ids = []  # saved-queue ideas, for the identity check below
    for wid, spec in proposal.get('saved_windows', {}).items():
        if spec['account'] not in HOMES: raise IntegrityError('saved window account is not configured: ' + wid)
        if sha256((ROOT / spec['queue']).read_bytes()).hexdigest() != spec['queue_sha256']:
            raise IntegrityError('saved queue differs from its bound hash: ' + wid)
        queued_ids += [r['source_candidate_id'] for r in read(ROOT / spec['queue'])['rows']]
        rows = len(read(ROOT / spec['queue'])['rows'])
        windows.append(dict(id=wid, phase=PHASE, wave=spec['wave'], account=spec['account'], context=spec['context'],
                            seed=spec['seed'], kind='saved_queue', supply_only=True, cohort=REVISION, editorial_mode='bypass',
                            maximum_developments=0, minimum_selector_eligible_kernels=1, maximum_retained=rows,
                            protected=False, protected_targets=[], calls=[], pairing_index=len(windows),
                            saved_queue=spec['queue'], saved_queue_sha256=spec['queue_sha256'], saved_rows=rows,
                            **({'reference_wave': spec['reference_wave']} if 'reference_wave' in spec else {})))
    for wid, spec in proposal.get('saved_idea_windows', {}).items():
        # ideas a stopped run generated but never selected: no idea calls, this batch's selector (saved_ideas.py)
        if spec['account'] not in HOMES: raise IntegrityError('saved ideas account is not configured: ' + wid)
        if sha256((ROOT / spec['ideas']).read_bytes()).hexdigest() != spec['ideas_sha256']:
            raise IntegrityError('saved ideas differ from their bound hash: ' + wid)
        ids = sorted(read(ROOT / spec['ideas'])['source_rows'])
        if not 1 <= len(ids) <= SETTINGS['selector']['maximum_candidates']:
            raise IntegrityError('saved ideas window size outside the selector contract: ' + wid)
        windows.append(dict(id=wid, phase=PHASE, wave=spec['wave'], account=spec['account'], context=spec['context'],
                            seed=spec['seed'], kind='saved_ideas', supply_only=True, cohort=REVISION, editorial_mode='bypass',
                            maximum_developments=0, minimum_selector_eligible_kernels=1, maximum_retained=len(ids),
                            protected=False, protected_targets=[], calls=[], pairing_index=len(windows),
                            saved_ideas=spec['ideas'], saved_ideas_sha256=spec['ideas_sha256'], saved_kernel_ids=ids,
                            **({'reference_wave': spec['reference_wave']} if 'reference_wave' in spec else {})))
    if not windows: raise IntegrityError('allocation has neither supply nor saved windows')
    supply_windows = [w for w in windows if w['kind'] == 'fresh_supply']
    saved = [w for w in windows if w['kind'] != 'fresh_supply']
    if any('reference_wave' in w for w in windows) and any(w['kind'] == 'saved_ideas' and 'reference_wave' not in w for w in windows):
        raise IntegrityError('with reference waves, every saved ideas window needs one')
    if any('reference_wave' in w for w in saved) and (
            not SCIENCE.get('batch_references') or not any('reference_wave' in w for w in supply_windows)
            or not all(isinstance(w['reference_wave'], str) and len(w['reference_wave']) == 1 and w['reference_wave'].isupper()
                       for w in saved if 'reference_wave' in w)):
        raise IntegrityError('a saved window\'s reference wave needs batch references, idea windows with waves and a capital letter')
    if any('reference_wave' in w for w in supply_windows):
        # Reference waves (waves.py): every idea window has one, as a single capital letter, all windows run in one
        # scheduler tranche (spec wave A), and each account lists its windows in wave order.
        if (not all(isinstance(w.get('reference_wave'), str) and len(w['reference_wave']) == 1 and w['reference_wave'].isupper()
                    for w in supply_windows) or any(w['wave'] != 'A' for w in windows)):
            raise IntegrityError('reference waves need a single capital letter on every idea window and spec wave A throughout')
        for account in HOMES:
            order = [w['reference_wave'] for w in supply_windows if w['account'] == account]
            if order != sorted(order):
                raise IntegrityError(f'idea windows of {account} are not listed in reference-wave order')
    calls = [c for w in windows for c in w['calls']]
    kernel_ids = ([k for c in calls for k in c['opaque_kernel_ids']] + [k for w in windows for k in w.get('saved_kernel_ids', [])]
                  + queued_ids)
    if (len(kernel_ids) != len(set(kernel_ids)) or len({c['private_id'] for c in calls}) != len(calls)
            or any(c['bank_setting'] not in supply.BANK_SETTINGS for c in calls)):
        raise IntegrityError('allocation table has repeated identities or unknown bank settings')
    groups, dev_by_account, groups_by_account = [], Counter(), Counter()
    for w in windows:
        capacity = development_capacity(w['saved_rows'] if w['kind'] == 'saved_queue'
                                        else len(w['saved_kernel_ids']) if w['kind'] == 'saved_ideas'
                                        else sum(len(c['opaque_kernel_ids']) for c in w['calls']))
        count = ceil(capacity / GROUP_SIZE)
        dev_by_account[w['account']] += capacity; groups_by_account[w['account']] += count
        groups.extend(dict(id=f"{w['id']}-develop-{n+1:02d}", phase=PHASE, wave=w['wave'],
            account=None if DYNAMIC else w['account'], source_account=w['account'], context=w['context'],
            seed=w['seed']+n+1, kind='eligible_group', cohort=REVISION, calls=[], editorial_mode='bypass',
            maximum_developments=GROUP_SIZE, source_window=w['id'], group_ordinal=n,
            research_calls_per_account={a: 0 for a in HOMES}, research_call_allowance=0,
            **({'primary_call_allowance': PRIMARY_PER_TASK * GROUP_SIZE} if PRIMARY_PER_TASK != 4 else {})) for n in range(count))
    # Interleave groups across windows by ordinal so early groups from every window start first.
    groups.sort(key=lambda g: (g['group_ordinal'], [w['id'] for w in windows].index(g['source_window'])))
    total_dev, total_groups = sum(dev_by_account.values()), sum(groups_by_account.values())
    review = SCIENCE['review_calls_per_task']

    def categories(account):
        dev = total_dev if DYNAMIC else dev_by_account[account]
        return dict(ideas=sum(TURNS[c['bank_setting']] + (c.get('commit') == 'explore') for c in calls if c['account'] == account),
                    selectors=sum(w['account'] == account and w['kind'] in SELECTOR_KINDS for w in windows),
                    primary_downstream=PRIMARY_PER_TASK * dev, conditional_evidence_repair=int(review * dev),
                    transport_preflight=PROBES_PER_ACCOUNT, completed_research=0, research_disagreement=0, actual_boards=0)

    per = {a: categories(a) for a in HOMES}
    cats = dict(ideas=sum(TURNS[c['bank_setting']] + (c.get('commit') == 'explore') for c in calls), selectors=sum(w['kind'] in SELECTOR_KINDS for w in windows), primary_downstream=PRIMARY_PER_TASK * total_dev,
                conditional_evidence_repair=int(review * total_dev), transport_preflight=PROBES_PER_ACCOUNT * len(HOMES),
                completed_research=0, research_disagreement=0, actual_boards=0)
    total = sum(cats.values())
    limits = dict(logical_cap=total, physical_attempt_cap=2*total, maximum_developments=total_dev,
        maximum_execution_groups=total_groups, phases=[dict(phase=PHASE, logical_cap=total,
        physical_attempt_cap=2*total, maximum_developments=total_dev, astra_logical_cap=total,
        fable_logical_cap=0, categories=cats)],
        accounts={a: dict(logical_cap=min(total, sum(per[a].values())), physical_attempt_cap=2*min(total, sum(per[a].values())),
                          maximum_developments=total_dev if DYNAMIC else dev_by_account[a],
                          maximum_execution_groups=total_groups if DYNAMIC else groups_by_account[a], categories=per[a])
                  for a in HOMES})
    return dict(revision=REVISION, phase=PHASE, comparison_blocks=[], windows=windows,
        potential_groups=groups, limits=limits,
        generation_counts=dict(Counter(c['effort'] for c in calls)),
        generation_model_effort_counts=dict(Counter(c['idea_model']+'/'+c['effort'] for c in calls)),
        generation_account_model_effort_counts=dict(Counter(c['account']+'/'+c['idea_model']+'/'+c['effort'] for c in calls)),
        selection_policy='All useful quality-eligible candidates retained; no rank/family eligibility quota.',
        group_policy=(f'Groups of {GROUP_SIZE}; ' + ('dynamic account assignment at group start' if DYNAMIC
                      else 'every call uses the window owner account') + '.'),
        planner_comparison=proposal.get('planner_comparison'),
        research_policy='disabled_by_design', research_enabled=False,
        coverage={**proposal['coverage'], 'full_factorial': False})
