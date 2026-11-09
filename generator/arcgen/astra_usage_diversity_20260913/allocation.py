"""192 frozen kernels; development capacity never changes selector eligibility."""
from collections import Counter
from .common import ROOT, HERE, REVISION, read, ref, digest, IntegrityError
from . import supply
DEFAULT_RUN=ROOT/'runs/astra-usage-diversity-both-20260913'
PHASE='usage_diversity'
SEED=2026091348
COMPARISONS=[]
CATEGORY_NAMES={'transport_probes':'transport_preflight','completed_task_research':'completed_research','editorial_boards':'actual_boards'}


def design():
    spec=read(HERE/'allocation_spec.private.json');windows=[]
    for wi,draft in enumerate(spec['windows']):
        wid='supply-'+draft['draft_window_id'];seed=SEED+wi*101
        w={'id':wid,'phase':PHASE,'wave':draft['wave'],'account':draft['account'],'context':draft['context'],
            'seed':seed,'kind':'fresh_supply','supply_only':True,'cohort':'usage_diversity',
            'editorial_mode':'bypass','maximum_developments':0,'minimum_selector_eligible_kernels':1,
            'maximum_retained':24,'protected':False,'protected_targets':[],'calls':[]}
        for ci,c in enumerate(draft['calls']):
            call=supply.make_call(arm=c['arm'],invitation=c['invitation'],bank_setting=c['bank_setting'],
                context=w['context'],seed=seed,account=w['account'],window=wid,replicate=ci+1,effort=c['effort'],idea_model=c['model'])
            call.update(wave=w['wave'],allocation_cell=c['cell_id']);w['calls'].append(call)
        windows.append(w)
    calls=[c for w in windows for c in w['calls']]
    if len(calls)!=48 or len(windows)!=8 or {c['arm'] for c in calls}!=set(supply.ARMS) or len({c['invitation'] for c in calls if c['invitation']})!=11:raise IntegrityError('allocation coverage changed')
    potential=[]
    for ordinal in range(4):
        for w in windows:
            potential.append({'id':f"{w['id']}-develop-{ordinal+1:02d}",'phase':PHASE,'wave':w['wave'],
                'account':w['account'],'context':w['context'],'seed':w['seed']+ordinal+1,'kind':'eligible_group',
                'cohort':'usage_diversity','calls':[],'editorial_mode':'bypass','maximum_developments':6,
                'source_window':w['id'],'group_ordinal':ordinal})
    def cats(v):return {CATEGORY_NAMES.get(k,k):n for k,n in v.items()}
    b=spec['proposed_budget'];limits={'logical_cap':596,'physical_attempt_cap':660,'maximum_developments':96,
        'maximum_execution_groups':16,'phases':[{'phase':PHASE,'logical_cap':596,'physical_attempt_cap':660,
        'maximum_developments':96,'astra_logical_cap':596,'fable_logical_cap':0,'categories':cats(b['categories'])}],
        'accounts':{a:{**v,'categories':cats(v['categories'])} for a,v in b['accounts'].items()}}
    return {'revision':REVISION,'phase':PHASE,'comparison_blocks':[],'windows':windows,'potential_groups':potential,'limits':limits,'allocation_source':ref(HERE/'allocation_spec.private.json'),
        'selection_policy':'all quality-eligible retained; no rank/family cutoff','group_policy':'first group of each window, then stable round robin, at most eight groups and 48 developments per account',
        'research_policy':'one bounded panel per executed group, two fresh public then two fresh reveal calls, one reader on each account',
        'generation_counts':dict(Counter(c['idea_model']+'/'+c['effort'] for c in calls))}


def group_candidates(root, *, initial=False):
    """Independent of wall-clock completion order. First A groups overlap B supply."""
    plan=design();rows=[]
    for w in plan['potential_groups']:
        if initial and (w['wave']!='A' or w['group_ordinal']!=0):continue
        if not initial and w['wave']=='A' and w['group_ordinal']==0:continue
        q=root/'accounts'/w['account']/w['source_window']/'eligible_queue.private.json'
        if not q.exists():continue
        count=len(read(q).get('rows',[])[6*w['group_ordinal']:6*(w['group_ordinal']+1)])
        if count:rows.append({**w,'maximum_developments':count})
    return rows
