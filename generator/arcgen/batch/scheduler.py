"""Bounded window overlap, using the existing account pools and native journals."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from pathlib import Path
import json
import threading
import time
from .common import (REVISION, IntegrityError, isolated_root, frozen, atomic,
                     read, digest, ref, code_identity, HOMES, account_pool)

from .config import OPERATIONS
_POOL = OPERATIONS['pool']
POLICY = {'supply_per_account': 2, 'downstream_per_account': OPERATIONS['development_windows_per_account'],
          'ready_per_account': 2, 'supply_plus_ready_per_account': 3,
          'research_groups': 4,
          'provider_pool': f"native_{_POOL['downstream_workers']}_downstream_{_POOL['upstream_workers']}_upstream_{_POOL['downstream_workers'] + _POOL['upstream_workers']}_total",
          'account_assignment': OPERATIONS['account_assignment'],
          'scientific_order': 'frozen; never inferred from completion order'}
# Continuous supply (v1 batches): idea windows and development overlap. Development groups get their own setup slots
# and start as soon as their source window's selection is frozen and development capacity is free, instead of
# waiting behind every pending idea window. Absent the flag, the policy above is unchanged (r23/default behaviour).
CONTINUOUS = bool(OPERATIONS.get('continuous_supply', False))
if CONTINUOUS:
    POLICY = {**POLICY, 'continuous_supply': True,
              'supply_per_account': int(OPERATIONS.get('supply_windows_per_account', 2)),
              'group_preps_per_account': 4,
              'ready_per_account': POLICY['downstream_per_account'],
              'supply_plus_ready_per_account': int(OPERATIONS.get('supply_windows_per_account', 2)) + POLICY['downstream_per_account'],
              # since 2026-10-11 (v1-b14 study): saved windows open first, and development takes an account's idle idea
              # slots once its idea and selector calls have all ended
              'saved_windows_first': True, 'development_takes_idea_slots_after_supply': True}
    if 'idle_supply_windows_per_account' in OPERATIONS:
        # More idea windows while an account has no development to run (the start of a batch): idea calls borrow the
        # idle development slots, and the pool stops new loans the moment development is ready. Once it is, no new
        # window opens until the account is back under supply_per_account.
        POLICY['idle_supply_windows_per_account'] = int(OPERATIONS['idle_supply_windows_per_account'])
SAVED_KINDS = ('saved_queue', 'saved_ideas')  # source windows without idea calls


def assignments_path(root):
    return Path(root) / 'design/execution/assignments.private.json'


def load_assignments(root):
    path = assignments_path(root)
    return read(path) if path.exists() else {}


def record_assignment(root, wid, account):
    """Append-only durable group->account map; an existing entry can never change."""
    value = load_assignments(root)
    if wid in value:
        if value[wid] != account: raise IntegrityError('dynamic group assignment changed on resume: ' + wid)
        return value
    value[wid] = account
    atomic(assignments_path(root), value)
    return value


class DownstreamPriority:
    """One set per shared pool, including allocations waiting for a DAG slot."""
    def __init__(self, pool):
        self.pool, self.windows, self.lock = pool, set(), threading.RLock()

    @classmethod
    def for_pool(cls, pool):
        with pool.condition:
            if not hasattr(pool, '_arm_bank_downstream_priority'):
                pool._arm_bank_downstream_priority = cls(pool)
            return pool._arm_bank_downstream_priority

    def add(self, window):
        with self.lock:
            self.windows.add(window)
            self.pool.set_downstream_ready(True, reason='selected-or-research-work-ready-or-active')

    def remove(self, window):
        with self.lock:
            self.windows.discard(window)
            self.pool.set_downstream_ready(bool(self.windows), reason='selected-or-research-work-ready-or-active')


class StageEvents:
    """Coarse observations only; native call and budget ledgers remain authoritative."""
    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()

    def record(self, kind, **fields):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as stream:
                stream.write(json.dumps({'kind': kind, 'at_epoch_seconds': time.time(), **fields}, sort_keys=True) + '\n')

    @contextmanager
    def span(self, stage, **fields):
        start = time.time()
        self.record('stage_started', stage=stage, **fields)
        try:
            yield
        except BaseException:
            self.record('stage_finished', stage=stage, started_at_epoch_seconds=start, outcome='failed', **fields)
            raise
        else:
            self.record('stage_finished', stage=stage, started_at_epoch_seconds=start, outcome='returned', **fields)


# Policy entries that follow the worker pool. A run may resume with a different pool by recorded amendment, so these
# are compared without their values; everything else in the scheduler spec stays frozen.
ADJUSTABLE_POLICY = ('downstream_per_account', 'ready_per_account', 'supply_plus_ready_per_account', 'provider_pool')


def _freeze_spec(path, spec):
    if not path.exists():
        return frozen(path, spec)
    saved = read(path)
    strip = lambda value: {**value, 'policy': {k: v for k, v in value['policy'].items() if k not in ADJUSTABLE_POLICY}}
    if strip(saved) != strip(spec):
        raise IntegrityError(f'frozen artifact changed: {path}')
    history = path.with_name(path.name.replace('.scheduler.private.json', '.policy_history.jsonl'))
    lines = history.read_text().splitlines() if history.exists() else []
    last = json.loads(lines[-1])['policy'] if lines else saved['policy']
    if last != spec['policy']:  # one line per change of pool or windows
        with history.open('a') as stream:
            stream.write(json.dumps({'at_epoch_seconds': time.time(), 'policy': spec['policy']}, sort_keys=True) + '\n')
    return saved


def run_windows(root, *args, **kwargs):
    """Own the run lock before constructing pools; reuse explicitly shared pools."""
    from arcgen.lib.factory import run_lock
    from arcgen.lib.e18_pre_e20_pool import DedicatedCallPool
    root = isolated_root(root)
    with run_lock(root):
        managed = {}
        try:
            if kwargs.get('pools') is None:
                if (root / 'STOP_DISPATCH.json').exists(): raise IntegrityError('revision dispatch paused')
                for account in HOMES:
                    managed[account] = account_pool(root / 'accounts' / account / 'worker_pool.json')
                kwargs['pools'] = managed
            return _run_windows(root, *args, **kwargs)
        finally:
            for pool in managed.values(): pool.finish()


def _run_windows(root, phase, windows, requests, inputs, *, budget, pools, production,
                manifest, index=None, research_jobs=(), research_plan=None, mock=False, invoke_call=None, scheduler_id=None, before_development=None):
    """Overlap frozen windows; ready research must already have frozen identities.

    Per account: supply<=2, ready<=2, supply+ready<=3, downstream<=2.
    A completed supply future can remain held until a ready slot opens. Thus
    no more than the predecessor's five windows are outstanding per account.
    """
    from . import pipeline, waves
    from .retrieval import make_run_index
    from .research_dispatch import validate_jobs, run_job, frozen_anchors, CanonicalPanels
    root = isolated_root(root)
    if (root / 'STOP_DISPATCH.json').exists(): raise IntegrityError('revision dispatch paused')
    if research_jobs or research_plan is not None:
        raise IntegrityError('r21 separate research is disabled by design')
    ids = [w['id'] for w in windows]
    if (Path(budget.root).resolve() != root or not pools or len(ids) != len(set(ids)) or set(ids) != set(requests)
            or any(w['phase'] != phase or (w['account'] not in pools and not (w['account'] is None and w.get('kind') == 'eligible_group'))
                   for w in windows)
            or set(pools) - set(HOMES)):
        raise IntegrityError('coordinator window/account membership changed')
    for account, pool in pools.items():
        if (Path(pool.path).resolve() != root / 'accounts' / account / 'worker_pool.json'
                or pool.downstream_workers != _POOL['downstream_workers'] or pool.upstream_workers != _POOL['upstream_workers']
                or pool.workers != _POOL['downstream_workers'] + _POOL['upstream_workers']):
            raise IntegrityError('one existing native pool per account is required')
    if research_jobs and research_plan is not None:
        raise IntegrityError('supply either concrete research jobs or a canonical assembly plan')
    research_jobs = list(research_jobs)
    anchors = frozen_anchors(root) if research_jobs or research_plan is not None else None
    assembler = CanonicalPanels(research_plan, windows, pools, phase,
        budget.phase(phase)['categories'].get('completed_research', 0), anchors) if research_plan is not None else None
    validate_jobs(research_jobs, windows, pools, phase, anchors=anchors)
    if len(research_jobs)*4 > budget.phase(phase)['categories'].get('completed_research', 0):
        raise IntegrityError('frozen research jobs exceed their existing category allowance')
    spec = {'revision': REVISION, 'phase': phase, 'policy': POLICY, 'windows': windows,
            'request_hashes': {k: digest({cid: {f: v for f, v in r.items() if f != 'response_validator'}
                for cid, r in requests[k].items()}) for k in ids},
            'research_jobs': list(research_jobs), 'research_plan': research_plan, 'code_hash': code_identity(),
            'budget_hash': digest(budget.limits)}
    file_phase = scheduler_id or phase
    spec_path = root / 'phases' / (file_phase + '.scheduler.private.json')
    _freeze_spec(spec_path, spec)
    terminal_path = root / 'phases' / (file_phase + '.terminal.private.json')
    prior_terminal = read(terminal_path) if terminal_path.exists() else None
    events = StageEvents(root / 'timing' / (file_phase + '.events.jsonl'))
    events.record('session_started', phase=phase, terminal_verification=prior_terminal is not None)
    # A fresh status object cannot carry a stale previous-phase DNS/error reason.
    atomic(root / 'STATUS.json', {'revision': REVISION, 'phase': phase, 'status': 'verifying_terminal' if prior_terminal else 'running',
                                'updated_at_epoch': time.time(), 'automatic_next_phase': False})
    by_id = {w['id']: w for w in windows}; order = {wid: i for i, wid in enumerate(ids)}
    pending = {a: deque(w['id'] for w in windows if w['account'] == a and not (CONTINUOUS and w.get('kind') == 'eligible_group'))
               for a in pools}
    groups = {a: deque(w['id'] for w in windows if w['account'] == a and CONTINUOUS and w.get('kind') == 'eligible_group')
              for a in pools}  # continuous mode: development groups waiting for a setup slot
    # Dynamic development groups: previously assigned ones return to their account on resume.
    assigned = load_assignments(root)
    floating = deque()
    for w in windows:
        if w['account'] is not None: continue
        if w['id'] in assigned:
            by_id[w['id']] = {**w, 'account': assigned[w['id']]}
            if not budget.reserve_group(by_id[w['id']]): raise IntegrityError('assigned group lost its reservation')
            (groups if CONTINUOUS else pending)[assigned[w['id']]].append(w['id'])
        else:
            floating.append(w['id'])
    for account in pending: pending[account] = deque(sorted(pending[account], key=order.get))
    for account in groups: groups[account] = deque(sorted(groups[account], key=order.get))
    ready = {a: [] for a in pools}
    priority = {a: DownstreamPriority.for_pool(p) for a, p in pools.items()}
    supply, development, research = {}, {}, {}
    closed = set()  # accounts whose pool lent its idea slots to development this session
    complete, findings, research_started = {}, {}, set()
    prepared, last_counts = {}, {}
    failure_lock, first_failure = threading.Lock(), []
    common = dict(budget=budget, production=production, manifest=manifest, mock=mock)
    if invoke_call is not None: common['invoke_call'] = invoke_call

    supply_windows = [w for w in windows if w.get('kind') in waves.SOURCE_KINDS]
    wave_indexes = {}

    def index_for(w):
        """The run's reference index, extended with earlier waves' selected kernels for a later-wave idea window."""
        if waves.wave(w) is None or not waves.earlier(supply_windows, w):
            return index
        if waves.wave(w) not in wave_indexes:
            with events.span('wave_references', wave=waves.wave(w)):
                wave_indexes[waves.wave(w)] = waves.index(root, supply_windows, w, index)
        return wave_indexes[waves.wave(w)]

    def counts(account):
        return (sum(by_id[wid]['account'] == account and not (CONTINUOUS and by_id[wid].get('kind') == 'eligible_group')
                    for wid in supply.values()),
                len(ready[account]), sum(by_id[wid]['account'] == account for wid in development.values()))

    def group_preps(account):
        return sum(by_id[wid]['account'] == account and by_id[wid].get('kind') == 'eligible_group' for wid in supply.values())

    def developing(account):
        """Development groups of this account waiting for setup, in setup, ready or running."""
        return bool(groups[account] or group_preps(account)
                    or any(by_id[wid]['account'] == account and by_id[wid].get('kind') == 'eligible_group'
                           for wid in [*ready[account], *development.values()]))

    def supply_cap(account):
        idle = POLICY.get('idle_supply_windows_per_account', 0)
        return max(idle, POLICY['supply_per_account']) if idle and not developing(account) else POLICY['supply_per_account']

    def supply_done(account):  # continuous mode: this account has no idea or selector call left to make
        return CONTINUOUS and not pending[account] and counts(account)[0] == 0

    def development_cap(account):
        if not (POLICY.get('development_takes_idea_slots_after_supply') and supply_done(account)):
            return POLICY['downstream_per_account']
        if account not in closed:
            closed.add(account); pools[account].close_upstream(reason='idea and selector calls ended for this account')
            events.record('idea_slots_to_development', account=account)
        return POLICY['downstream_per_account'] + pools[account].upstream_workers

    def develop(wid):
        w = by_id[wid]
        try:
            return pipeline.develop_window(root, w, prepared[wid], pool=pools[w['account']],
                                           events=events, on_failure=report_failure, **common)
        finally:
            priority[w['account']].remove(wid)

    def review(job, dependencies):
        try:
            return run_job(root, job, dependencies, pools=pools, events=events, on_failure=report_failure, **common)
        finally:
            for account in {job['primary_account'], job['secondary_account']}:
                priority[account].remove('research:' + job['id'])

    def report_failure(exc):
        from .quota import UsagePaused
        with failure_lock:
            if first_failure: return
            first_failure.append(exc)
            if isinstance(exc, UsagePaused):
                atomic(root/'STATUS.json', {'status':'paused_usage', 'reason':str(exc), 'phase':phase})
                events.record('usage_paused', phase=phase, reason=str(exc))
                return
            if not (root / 'STOP_DISPATCH.json').exists():
                atomic(root / 'STOP_DISPATCH.json', {'reason': str(exc), 'component': 'bounded_coordinator',
                                                   'phase': phase, 'at_epoch': time.time()})
            atomic(root / 'STATUS.json', {'revision': REVISION, 'phase': phase, 'status': 'stopped',
                'reason': str(exc), 'updated_at_epoch': time.time(), 'automatic_resume_authorized': False})
            events.record('admission_stopped', phase=phase, reason=str(exc))

    def stop(exc):
        report_failure(exc)
        for future in [*supply, *development, *research]: future.cancel()

    try:
        if index is None:
            try:
                with events.span('reference_index', phase=phase):
                    index = make_run_index(root, inputs, budget)
            except BaseException as exc:
                stop(exc)
                raise
        upstream_slots = (max(POLICY['supply_per_account'], POLICY.get('idle_supply_windows_per_account', 0))
                          + POLICY['group_preps_per_account']) if CONTINUOUS else 2
        downstream_slots = POLICY['downstream_per_account'] + (_POOL['upstream_workers'] if POLICY.get('development_takes_idea_slots_after_supply') else 0)
        with ThreadPoolExecutor(max_workers=upstream_slots*len(pools)) as upstream, \
             ThreadPoolExecutor(max_workers=downstream_slots*len(pools)) as downstream, \
             ThreadPoolExecutor(max_workers=POLICY['research_groups']) as readers:
            try:
                while True:
                    if first_failure: raise first_failure[0]
                    if (root / 'STOP_DISPATCH.json').exists(): raise IntegrityError('revision dispatch paused')
                    # Inspect ALL completed futures for failure before admitting replacements.
                    for future in [*supply, *development, *research]:
                        if future.done(): future.result()
                    for future, wid in list(development.items()):
                        if future.done():
                            complete[wid] = future.result(); del development[future]
                            budget.release_stage(wid, 'development')
                            events.record('window_terminal', window=wid, account=by_id[wid]['account'])
                    for future, jid in list(research.items()):
                        if future.done():
                            findings[jid] = future.result(); del research[future]
                            job = next(j for j in research_jobs if j['id'] == jid)
                            budget.release_stage(job['authorization_window'], 'research')
                    if assembler:
                        for job in assembler.advance(complete):
                            validate_jobs([job], windows, pools, phase, anchors=anchors)
                            frozen(root / 'research' / phase / 'jobs' / (job['id'] + '.private.json'), job)
                            research_jobs.append(job)
                    for job in sorted(research_jobs, key=lambda j: j['group']['ordinal']):
                        if (job['id'] not in research_started and set(job['depends_on']) <= set(complete)
                                and len(research) < POLICY['research_groups']):
                            if first_failure: raise first_failure[0]
                            research_started.add(job['id'])
                            for account in {job['primary_account'], job['secondary_account']}:
                                priority[account].add('research:' + job['id'])
                            events.record('research_submitted', group=job['id'], ordinal=job['group']['ordinal'])
                            dependencies = {wid: complete[wid] for wid in job['depends_on']}
                            research[readers.submit(review, job, dependencies)] = job['id']
                    # Drain ready allocations before refilling supply. A full ready list
                    # leaves completed supply in its original bounded future slot.
                    for account in sorted(pools, key=lambda a: (counts(a)[2] + counts(a)[1], a)):
                        # Idle capacity takes the next floating group whose eligible queue is frozen.
                        def room(account):
                            if CONTINUOUS:  # development capacity, independent of idea windows still pending
                                return (counts(account)[2] + counts(account)[1] + len(groups[account]) + group_preps(account)
                                        < development_cap(account) + POLICY['ready_per_account'])
                            return (not pending[account] and counts(account)[0] < POLICY['supply_per_account']
                                    and sum(counts(account)) < POLICY['downstream_per_account'] + POLICY['ready_per_account'])
                        while floating and room(account):
                            # the earliest floating group whose window's selection is frozen; groups of windows still
                            # selecting do not block ready ones behind them (frozen order is kept among ready groups)
                            wid = next((f for f in floating if (root/'accounts'/by_id[f]['source_account']/by_id[f]['source_window']
                                                                /'eligible_queue.private.json').exists()), None)
                            if wid is None:
                                break
                            floating.remove(wid); w = {**by_id[wid], 'account': account}
                            record_assignment(root, wid, account)
                            if not budget.reserve_group(w): raise IntegrityError('dynamic group reservation failed: ' + wid)
                            by_id[wid] = w; (groups if CONTINUOUS else pending)[account].append(wid)
                            events.record('group_assigned', window=wid, account=account)
                        for future, wid in sorted(list(supply.items()), key=lambda x: order[x[1]]):
                            if by_id[wid]['account'] == account and future.done() and len(ready[account]) < POLICY['ready_per_account']:
                                prepared[wid] = future.result(); ready[account].append(wid); del supply[future]
                        ready[account].sort(key=order.get)
                        while CONTINUOUS and groups[account] and group_preps(account) < POLICY['group_preps_per_account']:
                            if first_failure: raise first_failure[0]
                            wid = groups[account].popleft(); w = by_id[wid]
                            events.record('supply_submitted', window=wid, account=account)
                            supply[upstream.submit(pipeline.prepare_window, root, w, requests[wid], inputs, index,
                                pool=pools[account], on_ready=priority[account].add, on_failure=report_failure,
                                events=events, **common)] = wid
                        while ready[account] and counts(account)[2] < development_cap(account):
                            if first_failure: raise first_failure[0]
                            wid = ready[account][0]
                            if before_development and by_id[wid].get('kind') == 'eligible_group': before_development(by_id[wid])
                            ready[account].pop(0)
                            events.record('development_submitted', window=wid, account=account)
                            development[downstream.submit(develop, wid)] = wid
                        while (pending[account] and counts(account)[0] < supply_cap(account)
                               and sum(counts(account)[:2]) < POLICY['supply_plus_ready_per_account'] + supply_cap(account) - POLICY['supply_per_account']):
                            if first_failure: raise first_failure[0]
                            # A later reference wave waits for every earlier idea window; any startable window behind
                            # it goes first (frozen order among startable ones), so a mixed order cannot deadlock.
                            unblocked = lambda x: not (by_id[x].get('kind') in waves.SOURCE_KINDS and waves.blocked(root, supply_windows, by_id[x]))
                            # Continuous mode: windows without idea calls (saved queues and saved ideas) go before idea
                            # windows; in v1-b14 they were listed last and opened 1.5-2 h late while development idled.
                            startable = next((x for x in pending[account] if by_id[x].get('kind') in SAVED_KINDS and unblocked(x)),
                                             None) if POLICY.get('saved_windows_first') else None
                            startable = startable or next((x for x in pending[account] if unblocked(x)), None)
                            if startable is None:
                                break
                            if startable != pending[account][0]:
                                pending[account].remove(startable); pending[account].appendleft(startable)
                            next_window = by_id[pending[account][0]]
                            if (next_window.get('kind') == 'eligible_group'
                                    and not (root/'accounts'/(next_window.get('source_account') or account)/next_window['source_window']/'eligible_queue.private.json').exists()):
                                if not supply and not development and not any(ready.values()):
                                    raise IntegrityError('eligible queue dependency cannot make progress')
                                break
                            wid = pending[account].popleft(); w = by_id[wid]
                            events.record('supply_submitted', window=wid, account=account)
                            supply[upstream.submit(pipeline.prepare_window, root, w, requests[wid], inputs, index_for(w),
                                pool=pools[account], on_ready=priority[account].add, on_failure=report_failure,
                                events=events, **common)] = wid
                        s, r, d = counts(account)
                        burst = max(0, POLICY.get('idle_supply_windows_per_account', 0) - POLICY['supply_per_account'])
                        assert (s <= POLICY['supply_per_account'] + burst and r <= POLICY['ready_per_account']
                                and s + r <= POLICY['supply_plus_ready_per_account'] + burst and d <= development_cap(account))
                        if last_counts.get(account) != (s, r, d):
                            events.record('window_counts', account=account, supply=s, ready=r, downstream=d)
                            last_counts[account] = (s, r, d)
                    if (not supply and not development and not research and not any(pending.values()) and not any(ready.values())
                            and not any(groups.values())):
                        if floating:
                            raise IntegrityError('floating development groups cannot make progress')
                        break
                    unfinished = [f for f in [*supply, *development, *research] if not f.done()]
                    if unfinished:
                        wait(unfinished, timeout=.25, return_when=FIRST_COMPLETED)
                    elif not any(ready.values()) and not any(groups.values()) and any(pending.values()):
                        # Nothing runs and nothing waits for a slot: only idea windows blocked by a reference wave
                        # whose earlier windows can never finish. Stop with a reason rather than spin.
                        raise IntegrityError('reference wave cannot make progress: '
                                             + ', '.join(w for q in pending.values() for w in q))
                    else:
                        time.sleep(.25)
            except BaseException as exc:
                stop(exc)
                raise  # Context managers drain running work; cancellation never kills a provider.
    finally:
        for w in by_id.values():
            if w['account'] is not None: priority[w['account']].remove(w['id'])
        for job in research_jobs:
            for account in {job['primary_account'], job['secondary_account']}:
                priority[account].remove('research:' + job['id'])
        events.record('session_finished', phase=phase)
    for wid in complete:
        if not any(wid in j['depends_on'] for j in research_jobs): budget.release_stage(wid, 'research')
    if len(findings) != len(research_jobs): raise IntegrityError('unresolved frozen research dependencies')
    terminal = {'status': 'complete_pending_semantic_synthesis', 'phase': phase, 'scheduler': ref(spec_path),
                'windows': [complete[wid]['result'] for wid in ids],
                'research': [findings[j['id']] for j in sorted(research_jobs, key=lambda j: j['group']['ordinal'])],
                'unreviewed_context_overflow': assembler.lost if assembler else [],
                'completed_at_epoch': prior_terminal['completed_at_epoch'] if prior_terminal else time.time(),
                'automatic_next_phase': False, 'next_action': 'analyst synthesis / explicit next experiment decision'}
    frozen(terminal_path, terminal)
    atomic(root / 'STATUS.json', {**terminal, 'revision': REVISION, 'updated_at_epoch': time.time(),
                                'terminal': ref(terminal_path)})
    return terminal
