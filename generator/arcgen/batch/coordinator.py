"""The batch run: idea windows and development under one continuous scheduler, wave by wave."""
import time
from .common import read, ref, frozen, atomic, IntegrityError, HOMES, account_pool
from .allocation import PHASE, SEED
from . import scheduler


def freeze_execution(root, budget):
    groups=read(root/'design/batch.private.json')['potential_groups']
    value=frozen(root/'design/execution/all.private.json',dict(groups=groups,deferred=[],
        policy='Capacity for all eligible supply. Reservations are ceilings, not fabricated task attempts.'))
    for w in value['groups']:
        if w['account'] is None: continue  # dynamic: reserved when the scheduler assigns an account
        if not budget.reserve_group(w): raise IntegrityError('full-eligibility capacity reservation failed')
    return value


def continuing_windows(plan, execution):
    # Admit B supply alongside A groups, then drain all groups without a repair
    # or research barrier between tranches. B groups wait for their own queue.
    groups=execution['groups']
    first=[w for w in groups if w['wave']=='A' and w['group_ordinal']==0]
    bw=[w for w in plan['windows'] if w['wave']=='B']
    prefix=[]
    for i in range(max(len(first),len(bw))):
        if i<len(bw): prefix.append(bw[i])
        if i<len(first): prefix.append(first[i])
    return prefix+[w for w in groups if w not in first]


def run(root, *, budget, manifest, inputs, mock=False):
    from arcgen.lib.factory import run_lock
    from arcgen.lib.e18_pre_e20_pool import DedicatedCallPool
    from .recent import freeze_snapshot, for_wave
    from .retrieval import make_run_index
    from .next_batch import prepare_wave
    plan=read(root/'design/batch.private.json');pools={}
    terminal_path=root/'phases'/(PHASE+'.terminal.private.json')
    prior=read(terminal_path) if terminal_path.exists() else None;before=budget.summary()
    with run_lock(root):
        try:
            for a in HOMES: pools[a]=account_pool(root/'accounts'/a/'worker_pool.json')
            execution=freeze_execution(root,budget)
            index=make_run_index(root,inputs,budget)
            def tranche(name,windows,wave,research=False):
                requests={w['id']:read(root/'design/windows'/w['id']/'ideas.private.json') if w['calls'] else {} for w in windows}
                rp=dict(phase=PHASE,mode='per_window',seed=SEED,window_order=[w['id'] for w in windows],
                    maximum_groups=sum(w['kind']=='eligible_group' for w in windows),ordinal_offset=0) if research else None
                return scheduler._run_windows(root,PHASE,windows,requests,for_wave(root,wave,inputs),
                    budget=budget,pools=pools,production=inputs['production'],manifest=manifest,index=index,
                    research_plan=rp,scheduler_id=name,mock=mock)
            aw=[w for w in plan['windows'] if w['wave']=='A']
            if scheduler.CONTINUOUS and not any(w['wave']=='B' for w in plan['windows']):
                # One wave: idea windows and development run in one continuous tranche (no supply barrier).
                tranches=[tranche('continuous-supply-development',aw+execution['groups'],'A')]
            else:
                a=tranche('A-supply',aw,'A')
                freeze_snapshot(root,'B',inputs,current_windows=aw)
                prepare_wave(root,'B',inputs,index=index)
                tranches=[a,tranche('continuous-development-B-supply',continuing_windows(plan,execution),'B',False)]
            value=dict(status='complete_pending_semantic_synthesis',phase=PHASE,tranches=tranches,
                execution_manifests=[ref(root/'design/execution/all.private.json')],resource_deferrals=[],
                completed_at_epoch=prior['completed_at_epoch'] if prior else time.time(),
                automatic_admission=False,automatic_extension=False,research_status='disabled_by_design')
            frozen(terminal_path,value)
            if prior:
                after=budget.summary()
                if any(before[k]!=after[k] for k in ('logical_calls','physical_attempts','developments')):
                    raise IntegrityError('terminal resume spent new calls')
            atomic(root/'STATUS.json',dict(status=value['status'],terminal=ref(terminal_path)))
            return value
        finally:
            for pool in pools.values():pool.finish()
