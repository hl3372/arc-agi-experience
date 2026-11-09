"""Frozen private admission into the unchanged candidate evidence DAG."""
from copy import deepcopy
from pathlib import Path
from arcgen.lib.e18_candidate_runtime import assign_routes,run_slate,_task_id
from .palette import assignments as palette_assignments
from arcgen.lib.factory_authoring_screen import initial_record
from arcgen.lib.factory_evidence import apply_final_evidence,recover_terminal_evidence_metadata
from arcgen.lib.e18_evidence_gate import evidence_is_clear
from arcgen.lib.e18_artifacts import materialize_terminal_artifacts,verify_terminal_artifact_bundle
from arcgen.lib.factory import call_accounting
from arcgen.lib.util import content_hash
from .common import frozen,read,atomic,IntegrityError
from . import reviews

def prepare(runtime,window,allocation):
    rows=deepcopy(allocation['rows']);root=runtime.run_dir/'admission'
    routed=assign_routes(runtime.factory_config,rows) if rows else []
    result=[]
    for r in routed:
        r['slate_index']=1
        task_id='d'+content_hash({'window':window['id'],'source':r['source_candidate_id']})[:19]
        initial=initial_record({'task_id':task_id,'source_candidate_id':r['source_candidate_id'],'lineage_id':r['lineage_id'],
            'arm':'single','private_source':r['private_source'],'private_conditioned':r['private_conditioned'],
            'private_source_call_id':r['private_source_call_id'],'private_idea_effort':r['private_effort'],
            'private_route':r['private_route'],'slate_index':1,'source_development_priority':r['study']['dispatch_ordinal'],
            'search_mode':r['search_mode'],'kernel':r['kernel'],'editorial_brief':r['editorial_brief'],'novelty':r['novelty']})
        initial['study']=deepcopy(r['study'])
        runtime.budget.reserve_development(task_id,runtime.phase,window['id'],content_hash(initial),window['maximum_developments'])
        result.append({'task_id':task_id,'initial':initial,'selected':r})
    native=palette_assignments(runtime.factory_config,runtime,routed,slate=1) if routed else {}
    assignments={r['task_id']:native[_task_id(runtime.factory_config,r['selected']['source_candidate_id'])] for r in result}
    manifest=frozen(root/'manifest.private.json',{'window':window,'rows':result,'assignments':assignments,
        'allocation_hash':content_hash(allocation),'source_manifest_hash':content_hash(runtime.source_manifest)})
    if set(runtime.state['semantic_records'])-set(r['task_id'] for r in result): raise IntegrityError('foreign candidate in window runtime')
    for r in result:
        previous=runtime.state['semantic_records'].get(r['task_id'])
        if previous:
            for key in ('task_id','source_candidate_id','private_source','private_source_call_id','study'):
                if previous[key]!=r['initial'][key]: raise IntegrityError('resumed candidate attribution changed')
        else: runtime.state['semantic_records'][r['task_id']]=deepcopy(r['initial'])
    runtime.save();return manifest

def run(runtime,window,allocation,portfolio,invoke):
    manifest=prepare(runtime,window,allocation)
    path=runtime.run_dir/'window_terminal.private.json'
    recover_terminal_evidence_metadata(runtime)
    if path.exists():
        saved=read(path);verify(runtime,saved);return saved
    ids=[r['task_id'] for r in manifest['rows']]
    if ids: run_slate(runtime.factory_config,runtime,slate=1,task_ids=ids,assignments=manifest['assignments'])
    for task_id in ids: apply_final_evidence(runtime,task_id)
    reviews.run(runtime,portfolio,invoke)
    for task_id,r in runtime.state['semantic_records'].items():
        usable=bool(evidence_is_clear(r) and r.get('q2_review_binding') and
            r.get('intrinsic_quality') in ('core_quality','reserve_quality') and r.get('portfolio_placement')=='include_now')
        r['q2_usable_before_union_audit']=usable
        r['curriculum_status']='q2_'+r['intrinsic_quality'] if usable else 'q2_not_usable'
    runtime.save()
    artifacts=materialize_terminal_artifacts(runtime.factory_config,runtime)
    accounting=call_accounting([runtime]);atomic(runtime.run_dir/'call_timing_ledger.json',accounting)
    result={'status':'complete','window':window['id'],'runtime_directory':str(runtime.run_dir),
        'manifest_hash':content_hash(manifest),'records_hash':content_hash(runtime.state['semantic_records']),
        'call_ledger_hash':content_hash(runtime.state['call_ledger']),'artifacts':artifacts,
        'conditional_reservations':deepcopy(runtime.state.get('e2_escalation_budgets',{})),
        'tasks_committed':0,'new_provider_calls_on_terminal_resume':0}
    frozen(path,result);verify(runtime,result);return result

def verify(runtime,result):
    if result['records_hash']!=content_hash(runtime.state['semantic_records']) or result['call_ledger_hash']!=content_hash(runtime.state['call_ledger']): raise IntegrityError('terminal window changed')
    if result['manifest_hash']!=content_hash(read(runtime.run_dir/'admission/manifest.private.json')): raise IntegrityError('terminal admission changed')
    for row in result['artifacts']['rows']:
        record=runtime.state['semantic_records'][row['task_id']]
        verify_terminal_artifact_bundle(runtime.factory_config,row,record)
        if record.get('q2_review_binding'): reviews.verify(runtime,record)
    for value in result['conditional_reservations'].values():
        if value['limit']!=5 or value['used']>5: raise IntegrityError('conditional reservations exceeded five')
    return {'status':'verified','window':result['window']}
