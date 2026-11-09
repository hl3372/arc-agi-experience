"""Read native ledgers; keep model spend distinct from the legacy transport label."""
from collections import Counter,defaultdict
from types import SimpleNamespace
from .common import read,ref,frozen,atomic,IntegrityError
from arcgen.lib.factory import call_accounting as native_accounting


def call_accounting(runtimes):
    """Include both journals once while preserving the shared task ledger."""
    projected=[]
    for runtime in runtimes:
        value={**runtime.invoker.value,'calls':dict(runtime.invoker.value.get('calls',{}))}
        path=runtime.run_dir/'flex_attempt_journal.json'
        if path.exists():
            other=read(path)['calls']
            if set(other)&set(value['calls']):raise IntegrityError('planner appears in both attempt journals')
            value['calls'].update(other)
        projected.append(SimpleNamespace(state=runtime.state,run_dir=runtime.run_dir,
            calls_dir=runtime.calls_dir,invoker=SimpleNamespace(value=value)))
    return native_accounting(projected)




def collect(root, *, terminal=False):
    runtimes=[];models=defaultdict(lambda:Counter());sources=[]
    for path in sorted((root/'accounts').rglob('state.json')):
        state=read(path)
        if 'call_ledger' not in state:continue
        journal=path.parent/'attempt_journal.json'
        if not journal.exists():continue
        sources.extend([ref(path),ref(journal)])
        runtimes.append(SimpleNamespace(state=state,run_dir=path.parent,calls_dir=path.parent/'model_calls',
            invoker=SimpleNamespace(value=read(journal))))
        for call in state['call_ledger'].values():
            key=(call['account'],call['model'],call['reasoning_effort'],call['purpose'])
            models[key].update(logical_calls=1,physical_attempts=call.get('attempts',0),**{call['status']:1})
    native=call_accounting(runtimes)
    reused={}
    for path in sorted((root/'inputs/saved').glob('*/manifest.private.json')):
        for row in read(path)['reused_calls']:
            key=(row['account'],row['category'],row['call_id'])
            if key in reused and reused[key]!=row: raise IntegrityError('reused call accounting changed')
            reused[key]=row
    value={'scope':'calls of this run on the configured Codex account','by_model_role':[dict(account=k[0],model=k[1],effort=k[2],purpose=k[3],**v) for k,v in sorted(models.items())],
        'native':native,'sources':sources,
        'interpretation':['The inherited budget provider name astra denotes native Codex transport; model identity is reported explicitly here.',
            'Native model_execution_seconds are observed transport spans. Host suspension, provider wait and queueing must not be described as active inference.',
            'Subscription billing is unavailable; no API calls are made by this revision.'],
        'reused_costs':{'calls':list(reused.values()), 'logical_calls':len(reused),
            'physical_attempts':sum(r['physical_attempts'] for r in reused.values()),
            'charged_to_new_budget':False,'retrieval_cache_reuse':ref(root/'retrieval_cache_reuse.private.json'),
            'qualification':ref(root/'selector_qualification.private.json'),'research':'disabled_by_design'}}
    path=root/'analysis'/('call_timing_accounting.private.json' if terminal else 'call_timing_status.private.json')
    (frozen if terminal else atomic)(path,value)
    return {'artifact':ref(path),'by_model_role':value['by_model_role'],
        'native_logical_calls':native['logical_calls'],'native_physical_attempts':native['physical_attempts'],
        'model_span_seconds':native['active_model_seconds'],'queue_wait_seconds':native['queue_wait_seconds'],
        'billing':'subscription only; no API dollars'}
