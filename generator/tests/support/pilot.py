"""One development window through the native downstream DAG with scripted model responses (mock transport)."""
from copy import deepcopy
import json
import pytest
from arcgen.lib.factory_runtime import FactoryModelInvoker
from arcgen.lib.e18_pre_e20_pool import DedicatedCallPool
from arcgen.lib.util import content_hash
from arcgen.lib import astra_authoring_schemas_v3 as s
from arcgen.astra_evidence_runtime_revision_20260911 import evidence
from arcgen.batch import runtime
from arcgen.astra_usage_diversity_20260913 import adjudication, recheck
from arcgen.batch import gate, repair
from arcgen.batch.common import config_for, IntegrityError
from arcgen.batch.budget import Budget
from arcgen.astra_repetition_quality_20260912 import selection
from arcgen.astra_usage_diversity_20260913.palette import freeze_inputs
from support.fixtures import fixture, restrict, episode, technical, blind
from arcgen.sol_routine_astra_escalation_20260914 import technical as technical_recheck
from arcgen.batch.quota import Quota, UsagePaused


def limits(account='main', primary=4):
    value={'logical_cap':20,'physical_attempt_cap':40,'maximum_developments':1,'phases':[{
        'phase':'fixture','logical_cap':20,'physical_attempt_cap':40,'maximum_developments':1,
        'astra_logical_cap':20,'fable_logical_cap':0,'categories':{
            'primary_downstream':primary,'conditional_evidence_repair':5,'actual_boards':0}}]}
    cats=value['phases'][0]['categories'];cats.update(ideas=0,selectors=0,completed_research=0,transport_preflight=0)
    value.update(maximum_execution_groups=1,accounts={
        account:dict(logical_cap=20,physical_attempt_cap=40,maximum_developments=1,maximum_execution_groups=1,categories=dict(cats))})
    return value


def make_adjudication(record, tid, calls, case):
    current = gate.current_blind(tid, record, calls)
    decision = 'clear'
    if case in ('rival','flagged_rival'): decision='serious_ambiguity'
    if case=='unresolved': decision='unresolved'
    if case.startswith('repair') and record['repair_count'] < (2 if case=='repair_twice' else 1): decision='repairable'
    resolutions=[dict(concern_id=c['concern_id'],resolution='consistent_intended_rule',training_pair_ids=['train_1'],
                      explanation='The visible examples support the fixed displacement in this synthetic fixture.', repair_target=None) for c in adjudication.concerns(current)]
    if decision in ('serious_ambiguity','repairable'):resolutions[0]['resolution']='comparable_rival'
    if case=='discovery':
        for row in resolutions:
            if row['concern_id'].endswith(':primary'):
                row.update(resolution='contradicted_by_training',explanation='In train_1 the mark moves from column 0 to column 2, contradicting a fixed destination of column 4.')
    if decision=='unresolved':
        for r in resolutions:r['resolution']='unresolved'
    value=dict(task_id=tid,episode_hash=content_hash(record['complete_task']),policy=adjudication.POLICY,decision=decision,
        intended_rule_reasonably_supported=None if decision=='unresolved' else True,
        comparably_simple_different_output_rule=None if decision=='unresolved' else decision in ('serious_ambiguity','repairable'),
        failure_kind='discovery' if case=='discovery' else 'execution',
        public_support=[dict(training_pair_id='train_1',evidence='The output retains the anchor and shifts the marked cell.')],
        concern_resolutions=resolutions,reason='Specific synthetic adjudication outcome.',repair_instruction='')
    if decision=='repairable':
        value['repair_instruction']='Replace train_1 with the same displacement applied at a new location.'
        resolutions[0]['repair_target']=dict(competing_rule='A rule restricted to the original training location.',
            public_teaching_issue='Demonstrate the displacement at a different location.')
    return value


def run_case(tmp_path,monkeypatch,case,expected_calls,usable,crash_point=None,arm='H_DISC',account='main',palette_enabled=False,palette_prefill=0,planner='astra',
             study=None,primary=4):
    conf=config_for(tmp_path/'accounts'/account/'w',account=account,mock=True)
    w,source,response=fixture();key=next(k for k,r in source.items() if r['private_source']=='ordinary_unconditioned')
    source[key]['study']['primary_arm'] = arm
    source[key]['study'].update(bank_setting='unconditioned',generation_model='gpt-6-astra',**(study or {}))
    restrict(response,[key]);w.update(id='r5-'+case,phase='fixture',account=account,seed=777,maximum_developments=1,editorial_mode='bypass',protected=False)
    w.update(research_calls_per_account={account:0})
    if primary!=4: w['primary_call_allowance']=primary  # a scale rung's re-plan is a fifth primary call
    allocation=selection.allocate(w,source,response,conf['selection'],protected=False)
    assert len(allocation['rows'])==1
    from arcgen.batch import pairing
    allocation['rows']=[pairing._mark(deepcopy(allocation['rows'][0]),planner,paired=planner!='astra',
        pair_id=allocation['rows'][0]['source_candidate_id'] if planner!='astra' else None,version=dict(pairing.ORDER)[planner])]
    bank = tmp_path/'fixture_bank'; bank.mkdir(exist_ok=True)
    task_path = bank/'task.json'
    task_path.write_text(json.dumps({'train':[{'input':[[0,0,1]],'output':[[0,1,0]]}], 'test':[{'input':[[0,0,1]],'output':[[0,1,0]]}]}))
    from arcgen.astra_usage_diversity_20260913.common import sha
    registry_path = bank/'registry.json'
    registry_path.write_text(json.dumps({'entries':[{'registry_task_id':'fixture', 'task_path':'task.json', 'arc_sha256':sha(task_path)}]}))
    freeze_inputs(tmp_path, registry_path=registry_path)
    budget=Budget(tmp_path,limits(account,primary) if primary!=4 else limits(account));assert budget.reserve_group(w);pool=DedicatedCallPool(tmp_path/'pool.json')
    if crash_point=='usage_after_realization': budget.quota=Quota(tmp_path,'unused',mock=True,account=account)
    if palette_prefill:
        from arcgen.astra_usage_diversity_20260913.palette import Allocator
        from arcgen.lib.astra_authoring_contract_v3 import materialize_palette
        allocator = Allocator(tmp_path)
        for i in range(palette_prefill):
            tid = f'palette-prefill-{i}'
            p, r = episode(tid)
            p['public_contract']['episode_overview']['palette_contract']['opt_out_reason'] = 'Retain this symbolic palette for the fixture.'
            allocator.register(tid, 'main', 'palette-prefill', content_hash(p))
            allocator.materialize(materialize_palette, p, r, None, {})
    captured=[];calls=[];original=runtime.make_runtime
    def capture(*a,**kw):
        result=original(*a,**kw);captured.append(result);return result
    monkeypatch.setattr(runtime,'make_runtime',capture)
    def answer(invoker,stage,cid):
        calls.append((stage,cid));rt=captured[-1]
        request=json.loads((invoker.calls_dir/stage/cid/'request.json').read_text())
        prompt=(invoker.calls_dir/stage/cid/'prompt.md').read_text()
        tid=request['parent_ids'][0];plan,realized=episode(tid,no_ground=not palette_enabled)
        plan['private_execution']['cross_pair_concrete_audit']['relevant_and_incidental_variation'] = 'AUTHOR_PROFILE_NOTE_9ea28: intended demand, support in train_1, and test_1 application; synthetic integration fixture only.'
        if stage==s.PLAN_STAGE:
            from arcgen.batch.routing import PLANNERS
            assert request['model']['name']==PLANNERS[planner]['model'] and request['model']['reasoning_effort']==PLANNERS[planner]['reasoning_effort']
            return json.dumps(plan)
        if stage==s.REALIZER_STAGE:
            assert 'AUTHOR_PROFILE_NOTE_9ea28' in prompt
            assert rt.role_profile['realization']['model']=='gpt-6.1-sol' and rt.role_profile['realization']['reasoning_effort']=='xhigh'
            if case=='refusal':realized.update(status='refused',refusal_reason='Frozen plan obstruction.',task=None,authoring_metadata=None)
            if case=='format_retry' and sum(st==stage for st,_ in calls)==1:realized['task']['train'][0]['input'][0].pop()
            return json.dumps(realized)
        record=rt.state['semantic_records'][tid];outputs=deepcopy(record['realization']['authoring_metadata']['expected_test_outputs'])
        if stage in (evidence.STAGE,recheck.STAGE,technical_recheck.STAGE):
            value=technical(tid);suffix=cid[len(tid)+1:]
            value.update(verifier_id='verifier-'+content_hash({'task':tid,'suffix':suffix})[:12],derived_test_outputs=outputs,derivation_failure_reason='')
            groups=evidence.grouped_diagnostics(record['technical_diagnostics'])['flags']
            value['diagnostic_flag_assessments']=[dict(flag_code=g['code'],diagnostic_observation='confirmed',retention_decision='acceptable',
                evidentially_necessary=False,assessment='All observations checked in this fixture.',reviewed_observation_ids=[o['observation_id'] for o in g['observations']]) for g in groups]
            if case in ('false_fatal','confirmed_fatal') and cid.endswith('-technical-primary'):
                value.update(status='fatal',mechanically_correct=False)
            if stage==technical_recheck.STAGE:
                context=json.loads((rt.run_dir/'review_context'/stage/(cid+'.json')).read_text())['values']
                value['prior_concern_resolutions']=[dict(concern_id=c['concern_id'],resolution='resolved',evidence='Public construction independently traced.') for c in context['TECHNICAL_CONCERNS']]
                if case=='confirmed_fatal':
                    value.update(status='fatal',mechanically_correct=False)
                    value['prior_concern_resolutions'][0]['resolution']='confirmed_defect'
            if stage==recheck.STAGE:
                challenge=gate._repair_values(record)
                value['repair_challenge_assessments']=[dict(concern_id=c['concern_id'],resolved=case!='repair_unremoved',
                    rival_still_fits=case=='repair_unremoved',changed_training_pairs=challenge['changed_training_pairs'],
                    discriminating_consequence='The changed demonstration disproves the old location restriction.') for c in challenge['concerns']]
                if case=='repair_unremoved':value['status']='uncertain'
                assert 'AUTHOR_PROFILE_NOTE_9ea28' not in prompt
            assert 'ANSWER_CANARY_727893b1' not in prompt
            assert 'AUTHOR_PROFILE_NOTE_9ea28' not in prompt
            assert 'ANSWER_CANARY_727893b1' not in prompt
            return json.dumps(value)
        if stage=='v4_blind_solver':
            if (case=='blind_unavailable' or case=='flagged_unavailable' and cid.endswith('-blind-second')
                    or case=='repair_blind_unavailable' and record['repair_count']):
                raise __import__('subprocess').TimeoutExpired('fixture',1)
            suffix=cid[len(tid)+1:];value=blind(tid,rt.factory_config['run']['id'])
            miss=case not in ('clear','format_retry','false_fatal','confirmed_fatal','flagged_right') and (record['repair_count'] < (2 if case=='repair_twice' else 1) or case in ('repair_mismatch','repair_flagged'))
            if miss:outputs[0][0][0]=(outputs[0][0][0]+1)%10
            flagged=case.startswith('flagged') or case=='repair_flagged' and record['repair_count']>0
            value.update(task_id='blind-'+content_hash({'seed':777+18031,'task':tid})[:12],solver_id='solver-'+content_hash({'task':tid,'suffix':suffix})[:12],
                predicted_test_outputs=outputs,serious_different_output_alternative=flagged,test_requires_unsupported_guessing=flagged,
                strongest_plausible_alternatives=[])
            if case=='discovery':value['primary_inferred_rule']='Place the moving mark in column 4 regardless of its input column.'
            assert 'AUTHOR_PROFILE_NOTE_9ea28' not in prompt
            assert 'ANSWER_CANARY_727893b1' not in prompt and 'AUTHOR_SPECIFICATION' not in prompt
            return json.dumps(value)
        if stage==adjudication.STAGE:
            assert 'AUTHOR_PROFILE_NOTE_9ea28' not in prompt
            if case=='adjudication_unavailable':raise __import__('subprocess').TimeoutExpired('fixture',1)
            assert 'NOT\nsolver-visible teaching evidence' in prompt
            return json.dumps(make_adjudication(record,tid,list(rt.state['call_ledger'].values()),case))
        if stage==repair.STAGE:
            p=deepcopy(record['concrete_plan']);r=deepcopy(record['realization'])
            # A concrete public change, preserving the fixture's fixed displacement.
            pair=record['repair_count'] if case=='repair_twice' else 0  # a second repair changes the other demonstration
            for name in ('input', 'output'):
                grid=r['task']['train'][pair][name]
                grid[0],grid[1]=grid[1],grid[0]
            p['public_contract']['training_pairs'][pair]['exact_input_construction']='Place the marked cell in row 1 and preserve the anchor.'
            p['public_contract']['test_inputs']=repair.REUSE_TESTS
            p['private_execution']['test_outputs']=repair.REUSE_ANSWERS
            r['task']['test']=repair.REUSE_TESTS
            r['authoring_metadata']['expected_test_outputs']=repair.REUSE_ANSWERS
            value = dict(contract_version=repair.VERSION,training_support_updates=[],candidate_id=tid,repair_type='ambiguity',status='repaired',unrepairable_reason='',
                plan_update=p,repaired_realization=r,changes=['Replaced one public demonstration.'],
                why_rule_and_test_purpose_are_preserved='The rule, count, test input and answer remain unchanged.')
            if case=='repair_bad_contract' or case=='repair_retry' and sum(st==stage for st,_ in calls)==1:
                value['repaired_realization']['authoring_metadata']['expected_test_outputs']='wrong_reference'
            return json.dumps(value)
        raise AssertionError(stage)
    monkeypatch.setattr(FactoryModelInvoker,'_mock_response',answer)
    fired=[]
    if crash_point:
        save=runtime.Runtime.save
        def interrupted_save(rt):
            save(rt)
            for r in rt.state['semantic_records'].values():
                versions=list(r.get('difficulty_versions',{}).values())
                current=versions[-1] if versions else {}
                match={
                    'adjudication_saved':bool(current.get('adjudication')) and not r.get('ambiguity_adjudication'),
                    'repair_stored':r.get('repair_count')==1 and r.get('status')=='realized',
                    'technical_recheck_saved':current.get('technical_recheck_saved') is True,
                    'blind_recheck_saved':r.get('repair_count')==1 and len(r.get('blind_results',[]))==2,
                    'outcome_saved':bool(current.get('outcome')),
                    'terminal_status_saved':r.get('status') in ('blind_pass','ambiguous') and r.get('answer_evidence',{}).get('status')=='review_required',
                    'terminal_metadata_saved':bool(r.get('answer_evidence',{}).get('difficulty_outcome')),
                }.get(crash_point,False)
                if match and not fired:
                    fired.append(crash_point);raise KeyboardInterrupt(crash_point)
        monkeypatch.setattr(runtime.Runtime,'save',interrupted_save)
        if crash_point == 'palette_saved':
            from arcgen.astra_usage_diversity_20260913.palette import Allocator
            materialize = Allocator.materialize
            def interrupted_palette(allocator, *args, **kwargs):
                value = materialize(allocator, *args, **kwargs)
                if not fired:
                    fired.append(crash_point)
                    raise KeyboardInterrupt(crash_point)
                return value
            monkeypatch.setattr(Allocator, 'materialize', interrupted_palette)
        if crash_point in ('response_saved', 'repair_response_saved','usage_after_realization'):
            invoke=runtime.Runtime.invoke_rendered
            def interrupted_response(rt,**kw):
                value=invoke(rt,**kw)
                if kw['stage']==(s.REALIZER_STAGE if crash_point=='usage_after_realization' else repair.STAGE if crash_point=='repair_response_saved' else adjudication.STAGE) and not fired:
                    fired.append(crash_point)
                    if crash_point=='usage_after_realization': budget.quota.pause('provider_usage_limit')
                    raise KeyboardInterrupt(crash_point)
                return value
            monkeypatch.setattr(runtime.Runtime,'invoke_rendered',interrupted_response)
        if crash_point=='expansion_saved':
            from arcgen.batch import common
            freeze=common.frozen
            def interrupted_expansion(path,value):
                result=freeze(path,value)
                if path.parent.name=='repair_expansions' and not fired:
                    fired.append(crash_point);raise KeyboardInterrupt(crash_point)
                return result
            monkeypatch.setattr(common,'frozen',interrupted_expansion)
        if crash_point=='reservation_saved':
            from arcgen.lib.e2_live_streaming import IdempotentSlateBudget
            reserve=IdempotentSlateBudget.reserve_once
            def interrupted_reserve(b,key,count):
                value=reserve(b,key,count)
                if ':difficulty-review:' in key and not fired:
                    fired.append(crash_point)
                    if crash_point=='usage_after_realization': budget.quota.pause('provider_usage_limit')
                    raise KeyboardInterrupt(crash_point)
                return value
            monkeypatch.setattr(IdempotentSlateBudget,'reserve_once',interrupted_reserve)
    kw=dict(pool=pool,budget=budget,production={'portfolio_digest':{}},manifest={'fixture':'r5-dag'},mock=True)
    try:
        if crash_point:
            with pytest.raises((KeyboardInterrupt,UsagePaused)):runtime.develop_window(tmp_path,w,allocation,**kw)
            assert fired==[crash_point]
            if crash_point=='usage_after_realization':
                assert not (tmp_path/'STOP_DISPATCH.json').exists()
                budget.quota.start()
        first=runtime.develop_window(tmp_path,w,allocation,**kw);record=next(iter(first['runtime'].state['semantic_records'].values()))
        assert len(calls)==expected_calls, (calls,record.get('answer_evidence'),record.get('blind_error'),record.get('escalation_error'))
        assert record['pipeline_usable_before_union_audit'] is usable, record.get('final_episode_evidence')
        if case in ('copy_error','discovery','flagged_copy','repair_mismatch'):
            assert record['blind_performance']['label']=='verified; unsolved at tested budget'
        if case=='repair_clear':
            assert record['blind_performance']['any_all_test_success'] is True and len(record['difficulty_versions'])==2
        if case=='repair_unremoved':assert len(record['technical_results'])==2
        if case=='repair_flagged':assert record['answer_evidence']['hold_kind']=='budget'
        assert not any('board' in st for st,_ in calls)
        before=deepcopy(first['runtime'].state['call_ledger']);count=budget.summary()['logical_calls']
        for _ in range(2):
            again=runtime.develop_window(tmp_path,w,allocation,**kw)
            assert again['result']==first['result'] and again['runtime'].state['call_ledger']==before
        assert len(calls)==expected_calls and budget.summary()['logical_calls']==count
        return first, calls, kw, w, allocation
    finally:pool.finish()
