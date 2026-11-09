"""One actual q2 board, complete payload splitting and a matching file verifier."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from jsonschema import Draft202012Validator
from arcgen.lib.e18_evidence_gate import evidence_is_clear
from arcgen.lib.e18_pre_e20_deferred import freeze_board_packages
from arcgen.lib.factory_runtime import StageBinding
from arcgen.lib.factory_review_policy import evidence_hash,one_board_grade
from arcgen.lib.schemas import parse_json_object
from arcgen.lib.v4_schemas import _transport_schema
from arcgen.lib.util import content_hash,load_prompt,render_prompt
from .common import CARDS,HOMES,read,ref,frozen,IntegrityError,canonical

STAGE='broad_q2_editorial_v1'
SCHEMA=read(CARDS/'protocol/editorial_board_schema.proposed.json')
POLICY={'id':'broad-arm-q2-v1','stage':STAGE,'model':'gpt-6-astra','effort':'high',
    'schema_hash':content_hash(SCHEMA),'template':ref(CARDS/'prompts/policy/editorial_board.proposed.md')}

def parse(data):
    v=parse_json_object(data) if isinstance(data,(str,bytes,bytearray)) else deepcopy(data)
    Draft202012Validator(SCHEMA).validate(v)
    ids=[r['task_id'] for r in v['judgments']]
    if len(ids)!=len(set(ids)): raise ValueError('board repeats an episode')
    return v

def stage_bindings(): return {STAGE:StageBinding(_transport_schema(SCHEMA),parse,('editorial_board',))}

def grade(j):
    if j['novel_concept']=='weak' or j['conceptual_coherence']=='insufficient' or j['overall_quality']=='below_real_bar' or j['q2_intrinsic_recommendation']=='rejected_quality':
        return 'rejected_quality'
    if any(j[k]=='standout' for k in ('novel_concept','functional_composition','conceptual_coherence')) and j['q2_intrinsic_recommendation']=='core_quality': return 'core_quality'
    return 'reserve_quality'

def disposition(j):
    q=grade(j)
    placement=('same_procedure_duplicate' if j['same_minimal_procedure'] or j['portfolio_recommendation']=='same_procedure_duplicate' else
        'hold_crowded' if j['addition_is_excessive'] or j['portfolio_recommendation']=='hold_crowded' else
        'not_applicable' if q=='rejected_quality' else 'include_now')
    return {'intrinsic_quality':q,'portfolio_placement':placement,'historical_projected_grade':one_board_grade(j)}

def render_spec(spec):
    text,version,h=load_prompt(CARDS/'prompts/policy/editorial_board.proposed.md')
    return {'purpose':'editorial_board','stage':STAGE,'call_id':spec['call_id'],
        'prompt':render_prompt(text,spec['values']),'prompt_template_version':version,
        'prompt_template_hash':h,'parent_ids':spec['parent_ids']}

def complete_fit(runtime,request):
    report=runtime.context_guard.inspect(request['prompt']+'\n'+canonical(runtime.schemas[request['stage']])+'\n'+(' system'*2048),'editorial_board')
    return not report['requires_context_decision']

def prepare(runtime,portfolio):
    path=runtime.run_dir/'q2_reviews/prepared.private.json'
    records=runtime.state['semantic_records']
    eligible=sorted(k for k,r in records.items() if r.get('status') in ('blind_pass','editorial_complete') and evidence_is_clear(r))
    identity=content_hash({k:evidence_hash(records[k]) for k in eligible})
    if path.exists():
        old=read(path)
        if old['evidence_identity']!=identity or old['policy']!=POLICY or old['portfolio_hash']!=content_hash(portfolio): raise IntegrityError('frozen q2 cohort changed')
        return old
    groups=[]; overflow=[]
    def pack(ids,label):
        view=SimpleNamespace(config=runtime.config,run_dir=runtime.run_dir/'q2_reviews/packing'/label,
            state={'semantic_records':{k:deepcopy(records[k]) for k in ids}})
        p=freeze_board_packages(view,portfolio_digest=portfolio,board_indices=(1,))
        group,=p['groups'];spec,=group['calls']
        spec['call_id']='b'+content_hash({'namespace':runtime.prefix,'ids':ids})[:18]
        spec['values']['BOARD_ID']=spec['call_id']
        if complete_fit(runtime,render_spec(spec)):
            if len(groups)<3: groups.append(spec)
            else: overflow.extend(ids)
        elif len(ids)>1:
            n=len(ids)//2;pack(ids[:n],label+'a');pack(ids[n:],label+'b')
        else: overflow.extend(ids)
    # Canonical mixing by stable hash rather than arrival time or arm labels.
    ordered=sorted(eligible,key=lambda k:content_hash({'window':runtime.window,'board_mix':k}))
    for i in range(0,len(ordered),6): pack(ordered[i:i+6],str(i//6))
    return frozen(path,{'policy':POLICY,'evidence_identity':identity,'portfolio_hash':content_hash(portfolio),
        'groups':groups,'unboarded_context_overflow':overflow,'records':{k:deepcopy(records[k]) for k in eligible}})

def bind(runtime,spec,response,record):
    alias,= [a for a,k in spec['mapping'].items() if k=='generated:'+record['task_id']]
    judgment,=[r for r in response['judgments'] if r['task_id']==alias]
    call,=[c for c in runtime.state['call_ledger'].values() if c['stage']==STAGE and c['call_id']==spec['call_id']]
    identity={k:call.get(k) for k in ('purpose','stage','call_id','model','reasoning_effort','account','codex_home','request_fingerprint','response_content_hash','parent_ids','candidate_episode_hashes')}
    value={'policy':POLICY,'spec_hash':content_hash(spec),'episode_hash':content_hash(record['complete_task']),
        'evidence_hash':evidence_hash(record),'response_hash':content_hash(response),'judgment_hash':content_hash(judgment),
        'alias':alias,'call_identity':identity}
    value['binding_hash']=content_hash(value)
    return {**disposition(judgment),'editorial_judgments':[{**deepcopy(judgment),'task_id':record['task_id']}],
        'q2_review_binding':value}

def run(runtime,portfolio,invoke):
    prepared=prepare(runtime,portfolio)
    for spec in prepared['groups']:
        request=render_spec(spec)
        def validate(v):
            parse(v)
            if v['board_id']!=spec['call_id'] or {j['task_id'] for j in v['judgments']}!=set(spec['mapping']): raise ValueError('q2 board membership changed')
        response=invoke(runtime,{**request,'response_validator':validate})
        if response is None: continue
        for task_id in spec['parent_ids']:
            record=runtime.state['semantic_records'][task_id]
            record.update(bind(runtime,spec,response,record),status='editorial_complete')
        runtime.save()
    for task_id in prepared['unboarded_context_overflow']:
        runtime.state['semantic_records'][task_id]['q2_unboarded_reason']='complete_payload_context_overflow'
    runtime.save()
    for record in runtime.state['semantic_records'].values():
        if record.get('q2_review_binding'): verify(runtime,record)
    return prepared

def verify(runtime,record):
    b=record['q2_review_binding'];cid=b['call_identity']['call_id']
    prepared=read(runtime.run_dir/'q2_reviews/prepared.private.json')
    spec,=[s for s in prepared['groups'] if s['call_id']==cid]
    logs=runtime.calls_dir/STAGE/cid
    request,response,marker=(read(logs/n) for n in ('request.json','parsed_response.json','result.json'))
    raw=(logs/'raw_response.txt').read_text();prompt=(logs/'prompt.md').read_text()
    expected_request=render_spec(spec)
    expected=bind(runtime,spec,response,record)
    identity=b['call_identity']
    settings=runtime.factory_config['role_profile']['editorial_board']
    fingerprint=content_hash({'purpose':'editorial_board','stage':STAGE,'call_id':cid,'prompt':prompt,
        'template_version':expected_request['prompt_template_version'],'template_hash':expected_request['prompt_template_hash'],
        'schema':runtime.schemas[STAGE],'settings':settings,'parent_ids':spec['parent_ids']})
    episode_hashes={k:content_hash(prepared['records'][k]['complete_task']) for k in spec['parent_ids']}
    if (any(record.get(k)!=v for k,v in expected.items()) or parse(raw)!=response
        or prompt!=expected_request['prompt'] or request['prompt_hash']!=content_hash(prompt)
        or request['schema_hash']!=content_hash(runtime.schemas[STAGE])
        or request['prompt_template_hash']!=expected_request['prompt_template_hash']
        or request['prompt_template_version']!=expected_request['prompt_template_version']
        or request['parent_ids']!=spec['parent_ids']
        or request['model']['name']!=POLICY['model'] or request['model']['reasoning_effort']!=POLICY['effort']
        or identity['account']!=runtime.factory_config['run']['account']
        or identity['codex_home']!=HOMES[runtime.factory_config['run']['account']]
        or identity['request_fingerprint']!=fingerprint or identity['parent_ids']!=spec['parent_ids']
        or identity['candidate_episode_hashes']!=episode_hashes or identity['response_content_hash']!=content_hash(response)
        or identity['model']!=POLICY['model'] or identity['reasoning_effort']!=POLICY['effort']
        or request['stage']!=STAGE or request['call_id']!=cid or marker['cache_key']!=request['cache_key']
        or marker['raw_response_hash']!=content_hash(raw) or marker['parsed_response_hash']!=content_hash(response)
        or evidence_hash(prepared['records'][record['task_id']])!=evidence_hash(record)):
        raise IntegrityError('q2 actual-board binding or saved evidence changed')
    return {'status':'verified_q2_actual_board','call_id':cid,'binding_hash':b['binding_hash']}
