"""Native palette scheduling with already validated stationary input measures.

Reuse the pinned input measures, then run the native scheduling algorithm on
new source IDs and seeds. No old assignments or outcomes are used.
"""
from pathlib import Path
from copy import deepcopy
from arcgen.lib.v41_palette import schedule_backgrounds
from arcgen.lib.e18_candidate_runtime import _task_id
from .common import read,frozen,atomic,IntegrityError

def assignments(config,runtime,selected,*,slate):
    context=read(runtime.budget.root/'inputs/palette_context.private.json')
    path=runtime.run_dir/'palette_assignments.dynamic.json'
    initial={k:deepcopy(context[k]) for k in ('target_distribution','real_reference','committed_distribution')}
    value=read(path) if path.exists() else {'schema_version':1,'assignments':{},'by_source_candidate':{},'slates':{},**initial}
    if any(value[k]!=v for k,v in initial.items()):raise IntegrityError('frozen palette inputs changed')
    source_ids=sorted(r['source_candidate_id'] for r in selected);existing=value['by_source_candidate']
    missing=[k for k in source_ids if k not in existing]
    if missing:
        counts={int(k):int(v) for k,v in context['committed_distribution']['counts'].items() if str(k).isdigit() and 0<=int(k)<=9}
        for color in existing.values():
            if color is not None:counts[int(color)]=counts.get(int(color),0)+1
        existing.update(schedule_backgrounds(missing,seed=int(config['seed'])+slate*180271,current_counts=counts,
            target_distribution={int(k):v for k,v in context['target_distribution'].items()},eligible={k:True for k in missing}))
    assigned={_task_id(config,k):existing[k] for k in source_ids}
    for k,v in assigned.items():
        if k in value['assignments'] and value['assignments'][k]!=v:raise IntegrityError('resumed palette assignment changed')
        value['assignments'][k]=v
    value['slates'][str(slate)]={'source_candidate_ids':source_ids,'task_ids':sorted(assigned)}
    atomic(path,value);return dict(value['assignments'])
