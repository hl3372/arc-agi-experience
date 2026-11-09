"""Native Astra transport, with program admission and exact local q2 planning."""
from copy import deepcopy
from pathlib import Path
import re, shlex
from arcgen.lib.factory_runtime import FactoryRuntime, StageBinding
from arcgen.lib.factory_context import FactoryContextError
from arcgen.lib.util import content_hash, render_prompt
from .common import IntegrityError, HOMES, ref, read, atomic, halt, log
from .contracts import Q2Bindings, bindings

PRIMARY = {'planning':'concrete-plan','realization':'realization',
    'technical_verification':'technical-primary','primary_blind':'blind-primary'}
CONDITIONAL = {'technical_verification':{'technical-second','technical-after-repair','technical-after-ambiguity-repair'},
    'conditional_blind':{'blind-second','blind-after-ambiguity-repair','default-reading'},
    'ambiguity_adjudication':{'ambiguity-adjudication'},'targeted_repair':{'mechanical-repair','ambiguity-repair'}}

def category_for(purpose, call_id):
    if purpose.startswith('idea_'): return 'ideas'
    if purpose=='full_window_selection': return 'selectors'
    if purpose in PRIMARY and call_id.endswith('-'+PRIMARY[purpose]): return 'primary_downstream'
    if any(call_id.endswith('-'+s) for s in CONDITIONAL.get(purpose,())): return 'conditional_evidence_repair'
    if purpose=='editorial_board': return 'actual_boards'
    if purpose=='mechanism_audit': return 'completed_research'
    raise IntegrityError('unbound logical purpose: '+purpose+'/'+call_id)

class Runtime(FactoryRuntime):
    def __init__(self,*args,budget,phase,window='',category=None,**kwargs):
        self.budget,self.phase,self.window,self.fixed_category=budget,phase,window,category
        self.q2=Q2Bindings()
        kwargs.setdefault('contract_bindings',bindings())
        super().__init__(*args,**kwargs)
        self.prefix=str(self.run_dir.relative_to(self.budget.root))
        if not self.factory_config['run'].get('mock'):
            original=self.transport._codex_response
            def physical(prompt,schema_path,**settings):
                self.check_launcher()
                stage,cid=self.transport._transport_context.identity
                key=f'{self.prefix}/{stage}/{cid}'
                ordinal=self.budget.start_attempt(key)
                try:
                    response=original(prompt,schema_path,**settings)
                    self.check_observations(stage,cid,settings)
                except BaseException as exc:
                    self.budget.end_attempt(key,ordinal,'failed',error_type=type(exc).__name__)
                    raise
                self.budget.end_attempt(key,ordinal,'returned')
                return response
            self.transport._codex_response=physical

    def check_launcher(self):
        if (self.budget.root/'STOP_DISPATCH.json').exists(): raise IntegrityError('dispatch halted')
        account=self.factory_config['run']['account']; home=HOMES[account]
        path=self.run_dir/'transport/codex-account'; manifest=read(self.run_dir/'transport/manifest.json')
        wanted=['exec','/usr/bin/env','CODEX_HOME='+home,manifest['resolved_executable'],'$@']
        if (shlex.split(path.read_text(),comments=True)!=wanted or manifest['account']!=account
            or manifest['codex_home']!=home or ref(path)['sha256']!=manifest['launcher_sha256']
            or self.factory_config['transport']['codex_home']!=home):
            halt(self.budget.root,'account launcher identity changed',namespace=self.prefix)
        prepared=self.budget.root/'transport_preflight.json'
        if prepared.exists() and not self.factory_config['run'].get('mock'):
            if ref(manifest['resolved_executable'])!=read(prepared)['astra'][account]['executable']:
                halt(self.budget.root,'Astra executable changed after launch freeze',namespace=self.prefix)

    def check_observations(self,stage,cid,settings):
        directory=self.calls_dir/stage/cid
        rows=[]
        for p in sorted((directory/'transport_observations').glob('response-*.stderr.log')):
            header=p.read_text(errors='replace')[:5000]
            fields={key:(m.group(1).strip() if (m:=re.search('^'+re.escape(label)+r'\s*(.+?)\s*$',header,re.M)) else None)
                for key,label in [('model','model:'),('effort','reasoning effort:'),('provider','provider:'),('workdir','workdir:')]}
            rows.append({**ref(p),**fields})
        ok=bool(rows) and all(r['model']==settings['model_name'] and r['effort']==settings['reasoning_effort']
            and r['provider']=='openai' and Path(str(r['workdir'])).name.startswith('arcgen-model-') for r in rows)
        receipt={'expected':settings,'account':self.factory_config['run']['account'],'observations':rows,'verified':ok}
        atomic(directory/'program_transport_identity.json',receipt)
        if not ok: halt(self.budget.root,'returned model/effort/provider/workdir identity changed',receipt=receipt)

    def invoke_rendered(self,**kw):
        if (self.budget.root/'STOP_DISPATCH.json').exists(): raise IntegrityError('dispatch halted')
        stage,cid,purpose=kw['stage'],kw['call_id'],kw['purpose']
        # Native guard covers prompt-only. Add the full strict schema and a
        # conservative transport/system allowance before any budget admission.
        from .common import canonical
        full=kw['prompt']+'\n'+canonical(self.schemas[stage])+'\n'+(' system' * 2048)
        report=self.context_guard.inspect(full,purpose)
        atomic(self.run_dir/'complete_context_preflight'/stage/(cid+'.json'),report)
        if report['requires_context_decision']: raise FactoryContextError(report)
        key=f'{self.prefix}/{stage}/{cid}'
        identity=content_hash({'request':{k:v for k,v in kw.items() if k!='response_validator'},
            'schema':self.schemas[stage],'settings':self.role_profile[purpose]})
        self.budget.admit(key,phase=self.phase,category=self.fixed_category or category_for(purpose,cid),
            provider='astra',account=self.factory_config['run']['account'],window=self.window,fingerprint=identity)
        log('call',state='started',account=self.factory_config['run']['account'],phase=self.phase,window=self.window,purpose=purpose,call_id=cid)
        try: result=super().invoke_rendered(**kw)
        except BaseException as exc:
            # Log an end for every call, so progress never counts a failed or interrupted call as still running.
            if isinstance(exc,Exception): self.budget.outcome(key,'failed',error_type=type(exc).__name__)
            log('call',state='failed' if isinstance(exc,Exception) else 'interrupted',account=self.factory_config['run']['account'],
                phase=self.phase,window=self.window,purpose=purpose,call_id=cid,error_type=type(exc).__name__)
            raise
        self.budget.outcome(key,'complete',response_hash=content_hash(result.value),native_attempts=result.attempts)
        log('call',state='complete',account=self.factory_config['run']['account'],phase=self.phase,window=self.window,purpose=purpose,call_id=cid)
        return result

    def call(self,**kw):
        if kw['prompt_name']!='concrete_planner': return super().call(**kw)
        task_id,=kw['parent_ids']; record=self.state['semantic_records'][task_id]
        primary=record['study']['primary_arm']
        names=tuple(kw.get('composed_with',()))
        if not names or names[0]!='v44_quality_plan': raise IntegrityError('native planning composition changed')
        routes=names[1:]
        rows=self.q2.planning_templates(primary,routes)
        stage=self.q2.prompt_overrides[routes[-1] if routes else 'concrete_planner'][1]
        values,context=self.q2.prepare_values('concrete_planner',deepcopy(kw['values']))
        context.update(q2_primary=primary,q2_template_hashes=[r[2] for r in rows])
        template='\n\n'.join(r[0] for r in rows);prompt=render_prompt(template,values)
        self.q2.validate_request(stage,prompt,context)
        def validate(value):
            for k,v in (kw.get('expected') or {}).items():
                if value.get(k)!=v: raise IntegrityError('planner returned wrong identity')
            if kw.get('validator'): kw['validator'](value)
        return self.invoke_rendered(purpose='planning',stage=stage,call_id=kw['call_id'],prompt=prompt,
            prompt_template_version='+'.join(r[1] for r in rows),
            prompt_template_hash=content_hash({'templates':[r[2] for r in rows],'combined':template}),
            parent_ids=kw['parent_ids'],response_validator=validate,arm_id=kw['arm_id'],role=kw['role'],request_context=context).value
