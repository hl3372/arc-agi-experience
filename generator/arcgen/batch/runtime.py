"""Reuse native transport/contracts with immutable per-call routing views."""
from collections import Counter
from copy import copy, deepcopy
from pathlib import Path
from arcgen.lib.factory_runtime import StageBinding
from arcgen.lib.schemas import SchemaError
from arcgen.lib.util import load_prompt, render_prompt
from arcgen.astra_broad_arm_program_20260908.runtime import Runtime as NativeRuntime
from arcgen.astra_usage_diversity_20260913.runtime import Runtime as R10Runtime
from .contracts import Bindings, bindings
from arcgen.astra_usage_diversity_20260913 import adjudication, recheck
from . import repair, repair_v1, lines
from . import slots as _slots  # ONE (2026-10-23); _plan_with_colour has a local 'slots'
from .common import HOMES, HERE, REVISION, IntegrityError, frozen, digest, assert_launch
from .routing import POLICY, resolve
from arcgen.sol_routine_astra_escalation_20260914 import technical


class Runtime(R10Runtime):
    def __init__(self, config, *args, quota=None, palette_allocator=None, **kwargs):
        if (config['run']['account'] not in HOMES or config['_experiment_profile']['id']!=REVISION
                or config['_experiment_profile']['runtime_policy']!=POLICY):
            raise IntegrityError('the runtime requires this batch\'s account and routing configuration')
        if quota is None:
            quota=getattr(kwargs['budget'],'quotas',{}).get(config['run']['account'],getattr(kwargs['budget'],'quota',None))
        if palette_allocator is None:
            from arcgen.astra_usage_diversity_20260913.palette import Allocator
            palette_allocator=Allocator(kwargs['budget'].root)
        self.quota=quota;self.palette_allocator=palette_allocator
        q=Bindings()
        kwargs['contract_bindings']=bindings(palette_allocator=palette_allocator)
        kwargs['stage_bindings']={**kwargs.get('stage_bindings',{}),
            **{s:q.stage_bindings[s] for s in (adjudication.STAGE,recheck.STAGE)},technical.STAGE:technical.stage_binding(),
            **lines.default_binding()}  # structure lines (2026-10-22): the default-reading stage, when the run records it
        NativeRuntime.__init__(self,config,*args,**kwargs)
        self.q2=q
        if quota is not None:
            self.invoker.before_attempt=quota.guard
            self.invoker.after_failed_attempt=quota.failed

    def invoke_rendered(self, **kw):
        from arcgen.astra_targeted_followup_20260911.pipeline import DispatchPaused
        if (self.budget.root/'STOP_DISPATCH.json').exists():
            raise DispatchPaused('revision dispatch paused')
        review_context=kw.pop('_review_context',None)
        stage=kw['stage']
        repair_type=None
        if stage in (adjudication.STAGE,recheck.STAGE,repair.STAGE,repair_v1.STAGE,lines.REPAIR_STAGE,technical.STAGE,lines.DEFAULT_STAGE):
            if review_context is None: raise IntegrityError('review requires its exact context')
            validator=kw.get('response_validator')
            if stage==technical.STAGE:
                values=review_context['values']
                expected=technical_prompt(values)[0]
                if kw['prompt']!=expected: raise IntegrityError('technical recheck prompt changed')
                def validate(value):
                    technical.validate_response(value,values)
                    if validator: validator(value)
            else:
                self.q2.validate_request(stage,kw['prompt'],review_context)
                def validate(value):
                    self.q2.validate_response(stage,value,review_context)
                    if validator: validator(value)
                if stage in (repair.STAGE,repair_v1.STAGE,lines.REPAIR_STAGE):
                    kw['request_context']=review_context
                    repair_type=review_context['values']['REPAIR_TYPE']
            kw['response_validator']=validate
            frozen(self.run_dir/'review_context'/stage/(kw['call_id']+'.json'),review_context)
        elif review_context is not None:
            raise IntegrityError('review context supplied to an ordinary stage')
        launch = assert_launch(self.budget.root) if not self.factory_config['run'].get('mock') else None
        planner=None
        if kw['purpose']=='planning':
            planner=frozen_planner(self,kw['call_id'])
        binding=resolve(kw['purpose'],kw['call_id'],repair_type=repair_type,default=self.role_profile[kw['purpose']],planner=planner)
        account=self.factory_config['run']['account']
        frozen(self.run_dir/'routing'/stage/(kw['call_id']+'.json'),{
            'policy':POLICY['id'],'account':account,'purpose':kw['purpose'],'call_id':kw['call_id'],
            'repair_type':repair_type,'planner':planner,**binding})
        # The view shares thread-safe state, call locks and transport, while its
        # role dictionary is private to this call. Parallel stages cannot leak bindings.
        view=copy(self)
        view.role_profile=deepcopy(self.role_profile)
        view.role_profile[kw['purpose']].update(binding)
        try:
            return NativeRuntime.invoke_rendered(view,**kw)
        except IntegrityError as exc:
            if str(exc)=='dispatch halted' and (self.budget.root/'STOP_DISPATCH.json').exists():
                raise DispatchPaused('revision dispatch paused') from exc
            raise

    def episode_defect(self, task):
        """v1: a realized episode with a degenerate or reused test pair fails at realization, before technical and blind
        verification (e2_live_streaming._realize_one), and a repair may not produce one (_store_repaired_realization).
        Repair cannot help: every repair keeps the test answers and the planner chose these test pairs. r23: never."""
        if self.q2.prompt_set!='v1':
            return ''
        found = degenerate_tests(task)
        if found:
            return 'degenerate test: ' + '; '.join(found)
        reused = reused_tests(task)
        return ('reused test: ' + '; '.join(reused)) if reused else ''

    def call(self, **kw):
        if kw['prompt_name']=='concrete_planner' and self.q2.prompt_set=='v1':
            from .contracts import v1_planner_values
            record=self.state['semantic_records'][kw['parent_ids'][0]]
            kw={**kw,'values':v1_planner_values(record,kw['values'])}
            if (record['study'].get('colour_roles') is not None or 'pose' in record['study'] or record['study'].get('idea_modifiers')
                    or record['study'].get('rung') or record['study'].get('teach') == 'plain'
                    or record['study'].get('depart') == 'test' or lines.tagged(record['study'])):
                return self._plan_with_colour(kw,record,record['study'].get('colour_roles'),record['study'].get('pose',False))
        if kw['prompt_name']=='task_realizer' and self.q2.prompt_set=='v1':
            record=self.state['semantic_records'][kw['parent_ids'][0]]
            if lines.emits(record['study']):  # weakness run: the realizer also records the shortcut grids or ledger
                return self._realize_with_outputs(kw,record)
        if (kw['prompt_name']=='technical_verifier' and not kw['call_id'].endswith('-technical-primary')):
            values,_=self.q2.prepare_values('technical_verifier',deepcopy(kw['values']))
            tid,=kw['parent_ids'];record=self.state['semantic_records'][tid]
            if not record.get('technical_results'): raise IntegrityError('technical escalation lacks a prior response')
            values['TECHNICAL_CONCERNS']=technical.concerns(record['technical_results'][0])
            prompt,version,identity=technical_prompt(values)
            return self.invoke_rendered(purpose='technical_verification',stage=technical.STAGE,
                call_id=kw['call_id'],prompt=prompt,prompt_template_version=version,prompt_template_hash=identity,
                parent_ids=kw['parent_ids'],arm_id=kw['arm_id'],role=kw['role'],
                _review_context={'values':values}).value
        return self.call_review(**kw)


    def _plan_with_colour(self, kw, record, colour, pose=False):
        """The native planner composition, with the per-idea colour and pose instructions recorded in the request context.

        An idea with a scale rung also gets its family and rung lines; a ready plan below the rung's plan-stage floors
        is planned once more with a one-line note (rungs.py). Both calls are ordinary resumable calls: on resume the
        first returns its saved response, the check repeats exactly, and the re-plan is found under its own ID."""
        from arcgen.lib.util import content_hash
        from . import rungs
        study=record['study']
        primary=study['primary_arm']
        names=tuple(kw.get('composed_with',()))
        if not names or names[0]!='v44_quality_plan': raise IntegrityError('native planning composition changed')
        routes=names[1:]
        from .contracts import planner_slots
        slots=planner_slots(study)
        family,rung,canvas,teach,depart=study.get('family'),study.get('rung'),study.get('canvas'),study.get('teach'),study.get('depart')
        line,overlay,fence=lines.line_of(study),lines.overlay_of(study),lines.fence_of(study)
        structure=lines.structure_of(study)  # structure lines (2026-10-22): composed after the content line
        rows=self.q2.planning_templates(primary,routes,colour=colour,pose=pose,slots=slots,family=family,rung=rung,canvas=canvas,
                                        teach=teach,depart=depart,line=line,overlay=overlay,fence=fence,structure=structure)
        stage=self.q2.prompt_overrides[routes[-1] if routes else 'concrete_planner'][1]
        values,context=self.q2.prepare_values('concrete_planner',deepcopy(kw['values']))
        context.update(q2_primary=primary,q2_template_hashes=[r[2] for r in rows],q2_colour_roles=colour,q2_pose=pose)
        if slots: context['q2_slots']=list(slots)
        if family: context['q2_family']=family
        if rung: context['q2_rung']=rung
        if canvas: context['q2_canvas']=canvas
        if teach: context['q2_teach']=teach
        if depart: context['q2_depart']=depart
        if line: context['q2_line']=line
        if overlay: context['q2_overlay']=overlay
        if fence: context['q2_fence']=fence
        if structure: context['q2_structure']=structure
        template='\n\n'.join(r[0] for r in rows);prompt=render_prompt(template,values)
        self.q2.validate_request(stage,prompt,context)
        def validate(value):
            for k,v in (kw.get('expected') or {}).items():
                if value.get(k)!=v: raise IntegrityError('planner returned wrong identity')
            if kw.get('validator'): kw['validator'](value)
        version,identity='+'.join(r[1] for r in rows),content_hash({'templates':[r[2] for r in rows],'combined':template})
        value=self.invoke_rendered(purpose='planning',stage=stage,call_id=kw['call_id'],prompt=prompt,
            prompt_template_version=version,prompt_template_hash=identity,
            parent_ids=kw['parent_ids'],response_validator=validate,arm_id=kw['arm_id'],role=kw['role'],request_context=context).value
        checked=_slots.checked(study)  # ONE (2026-10-23): the planned pair count, sharing the same one re-plan
        if not rung and not line and not overlay and not structure and not checked:
            return value
        task_id,=kw['parent_ids']
        plan=rungs.plan_of(value)
        first=rungs.measures(plan) if plan and rung else None
        receipt={'task_id':task_id,'rung':rung,'family':family,'floors':rungs.floors(rung) if rung else None,'first':first,
                 'first_reached':rungs.reached(first) if first else None,'replanned':False,'second':None,'used':'first'}
        # weakness run (2026-10-20): the line's fixed-prefix fields, checked in code; a miss shares the rung's one re-plan
        issues=lines.plan_issues(line,overlay,plan,fence,structure) if (line or overlay or structure) and plan else []
        slot_issues=_slots.plan_issues(study,plan) if checked and plan else []
        notes=(([rungs.feedback(rung,first)] if first and rungs.shortfalls(rung,first) else [])+([lines.replan_note(issues)] if issues else [])
               +([_slots.replan_note(slot_issues)] if slot_issues else []))
        second_plan=None
        if notes:
            note='\n\n'.join(notes)
            again={**context,'q2_replan':note}
            replan_prompt=prompt+'\n\n'+note
            self.q2.validate_request(stage,replan_prompt,again)
            suffix=(('+rung-replan-1' if first and rungs.shortfalls(rung,first) else '')+('+line-replan-1' if issues else '')
                    +('+slot-replan-1' if slot_issues else ''))
            second_value=self.invoke_rendered(purpose='planning',stage=stage,
                call_id=task_id+rungs.REPLAN_SUFFIX+'-concrete-plan',prompt=replan_prompt,
                prompt_template_version=version+suffix,prompt_template_hash=content_hash({'base':identity,'replan':note}),
                parent_ids=kw['parent_ids'],response_validator=validate,arm_id=kw['arm_id'],role=kw['role'],request_context=again).value
            second_plan=rungs.plan_of(second_value)
            second=rungs.measures(second_plan) if second_plan and rung else None
            receipt.update(replanned=True,note=note,second=second,second_reached=rungs.reached(second) if second else None)
            if second_plan:  # a re-plan that aborts keeps the first plan
                value=second_value; receipt['used']='second'
        if line or overlay or structure:
            check=lines.plan_check(line,overlay,plan,second_plan,replanned=bool(notes),used=receipt['used'],fence=fence,
                                   structure=structure)
            frozen(self.run_dir/'line_check'/(task_id+'.json'),{'task_id':task_id,**check})
            if getattr(self,'state',None) is not None:
                with self.lock:
                    self.state['semantic_records'][task_id].setdefault('weakness',{})['plan_check']=check
                    self.save()
        if checked:  # ONE: the receipt of the planned pair count (blind_performance.one_pair reads it at the end)
            slot_check=_slots.plan_check(study,plan,second_plan,replanned=bool(notes),used=receipt['used'])
            frozen(self.run_dir/'slot_check'/(task_id+'.json'),{'task_id':task_id,**slot_check})
            if getattr(self,'state',None) is not None:
                with self.lock:
                    self.state['semantic_records'][task_id]['slot_check']=slot_check
                    self.save()
        if not rung:
            return value
        used=receipt['second'] if receipt['used']=='second' else first
        receipt['reached']=rungs.reached(used) if used else None
        frozen(self.run_dir/'rung_check'/(task_id+'.json'),receipt)
        return value

    def _realize_with_outputs(self, kw, record):
        """Weakness run (2026-10-20, DESIGN §5.1): the unchanged realizer template plus addenda asking for the declared
        shortcut's grids (SETTLE: every training pair and the test; DEPART, MARKS, INGRID, the matched control and EDGE: the
        test) or PRECISE's decision ledger, under a stage whose schema adds weakness_outputs. The ordinary realization is
        validated and returned exactly as always; the outputs are recorded on the task (lines.realizer_emission)."""
        from arcgen.lib.util import content_hash
        study=record['study']
        values,context=self.q2.prepare_values('task_realizer',deepcopy(kw['values']))
        rows=self.q2.realizer_templates(study)
        template='\n\n'.join(r[0] for r in rows);prompt=render_prompt(template,values)
        leaks=[f for f in kw.get('forbidden_prompt_fragments',()) if f.casefold() in prompt.casefold()]
        if leaks: raise IntegrityError(f'downstream prompt leaked private label(s): {leaks}')
        context.update(q2_line=lines.line_of(study),q2_overlay=lines.overlay_of(study),q2_template_hashes=[r[2] for r in rows])
        if lines.structure_of(study): context['q2_structure']=lines.structure_of(study)
        self.q2.validate_request(lines.REALIZER_STAGE,prompt,context)
        def validate(value):
            for k,v in (kw.get('expected') or {}).items():
                if value.get(k)!=v: raise IntegrityError('realizer returned wrong identity')
            if kw.get('validator'): kw['validator'](lines.split(value)[0])
        value=self.invoke_rendered(purpose='realization',stage=lines.REALIZER_STAGE,call_id=kw['call_id'],prompt=prompt,
            prompt_template_version='+'.join(r[1] for r in rows),
            prompt_template_hash=content_hash({'templates':[r[2] for r in rows],'combined':template}),
            parent_ids=kw['parent_ids'],response_validator=validate,arm_id=kw['arm_id'],role=kw['role'],request_context=context).value
        base,extra=lines.split(value)
        lines.realizer_emission(self,kw['parent_ids'][0],kw['call_id'],study,base,extra)
        return base

    def call_review(self, **kwargs):
        if kwargs['prompt_name'] in ('ambiguity_adjudicator', 'difficulty_technical_recheck', 'targeted_repair'):
            from arcgen.lib.util import load_prompt, render_prompt
            from . import repair
            name = kwargs['prompt_name']
            if name == 'targeted_repair':
                kwargs = deepcopy(kwargs)
                tid, = kwargs['parent_ids']
                record = deepcopy(self.state['semantic_records'][tid])
                specification = kwargs['values']['AUTHOR_SPECIFICATION']
                if (specification['concrete_plan'] != record['concrete_plan']
                        or specification['realization'] != record['realization']):
                    raise IntegrityError('repair request does not match the current source episode')
                kwargs['values']['ALLOW_MECHANICAL_TEST_INPUT_RESTORATION'] = self.config.get('repair_policy', {}).get('allow_mechanical_test_input_restoration') is True
                if record.get('study', {}).get('primary_arm') == 'D1':
                    count = len(record['complete_task']['train'])
                    kwargs['values']['REPAIR_INSTRUCTION'] += (
                        f'\nD1 demonstration count is frozen at {count}. Do not add or remove a pair. '
                        'An ambiguity repair may replace at most one existing training pair; '
                        'update that pair in the plan and realization consistently. '
                        'If this cannot resolve the concern while preserving the rule and tests, return unrepairable.')
                from arcgen.lib.astra_contract_bindings_v3 import chained_episode
                study = record.get('study', {})
                if (chained_episode(record['realization']['task']['train']) and study.get('primary_arm') != 'D1'
                        and (study.get('primary_arm') == 'STAGES' or 'CHAINED' in (study.get('idea_modifiers') or ()))):
                    kwargs['values']['REPAIR_INSTRUCTION'] += (
                        '\nThis episode is a chain: each training output is exactly the next training input, and it must stay '
                        'one. A new pair can join only at an end of the chain, so to show a deciding case you may instead '
                        'rebuild the whole chain from a different starting state chosen so that the chain passes through that '
                        'case, keeping the same number of training pairs or adding one. Every rebuilt pair must follow the frozen '
                        'rule exactly, every training output except the last must equal the next training input, and the tests '
                        'and their answers stay unchanged. Give every rebuilt pair in full in the plan. If the concern cannot be '
                        'resolved that way, return unrepairable.')
                from .gate import departure_invariant, DEPART_REPAIR_NOTE
                # departure package: any repair of a drawn plan that names its reserved situation keeps it unshown, on
                # every episode version (the invariant is frozen, so a repair that showed it would break it anyway)
                if record.get('study', {}).get('depart') == 'test' and departure_invariant(record):
                    kwargs['values']['REPAIR_INSTRUCTION'] += '\n' + DEPART_REPAIR_NOTE
                # weakness run (2026-10-20): the line's and overlay's repair notes, and the outputs to re-emit
                for note in lines.repair_notes(record):
                    kwargs['values']['REPAIR_INSTRUCTION'] += '\n' + note
                # ONE (2026-10-23): keep the one demonstration pair
                for note in _slots.repair_notes(record):
                    kwargs['values']['REPAIR_INSTRUCTION'] += '\n' + note
            filename, stage = self.q2.prompt_overrides[name]
            weak_repair = name == 'targeted_repair' and bool(lines.emits(record.get('study', {})))
            if weak_repair:  # the repair schema adds weakness_outputs; the repair itself is validated as always
                stage = lines.REPAIR_STAGE
            repair_module = repair_v1 if stage in (repair_v1.STAGE, lines.REPAIR_STAGE) else repair
            template, version, identity = load_prompt(self.q2.prompt_root / filename)
            values, context = self.q2.prepare_values(name, kwargs['values'])
            prompt = render_prompt(template, values)
            self.q2.validate_request(stage, prompt, context)
            def validate(value):
                for key, expected in kwargs.get('expected', {}).items():
                    if value.get(key) != expected:
                        raise SchemaError('$', 'review response identity changed')
                evaluated = repair_module.expand(lines.split(value)[0] if weak_repair else value, values) if name == 'targeted_repair' else value
                if name == 'targeted_repair':
                    validate_repair_identity(evaluated, record)
                if kwargs.get('validator'): kwargs['validator'](evaluated)
            result = self.invoke_rendered(stage=stage, call_id=kwargs['call_id'], prompt=prompt,
                purpose={'ambiguity_adjudicator': 'ambiguity_adjudication', 'difficulty_technical_recheck': 'technical_verification', 'targeted_repair': 'targeted_repair'}[name],
                prompt_template_version=version, prompt_template_hash=identity,
                parent_ids=kwargs['parent_ids'], arm_id=kwargs['arm_id'], role=kwargs['role'],
                response_validator=validate, _review_context=context).value
            if name == 'targeted_repair':
                raw, extra = lines.split(result) if weak_repair else (result, None)
                expanded = repair_module.expand(raw, values)
                from .common import frozen
                frozen(self.run_dir / 'repair_expansions' / (kwargs['call_id'] + '.json'), repair_module.receipt(result, expanded, values))
                if weak_repair:
                    from arcgen.lib.util import content_hash
                    lines.repair_emission(self, tid, kwargs['call_id'], content_hash(record['complete_task']) if record.get('complete_task') else None,
                                          expanded, extra)
                return expanded
            return result
        return NativeRuntime.call(self,**kwargs)


def frozen_planner(runtime, call_id):
    """The planner is fixed per task before any planning call; never inferred later."""
    from .rungs import task_of_plan_call
    task_id=task_of_plan_call(call_id)  # a rung re-plan shares its task's planner
    study=runtime.state['semantic_records'][task_id]['study']
    planner=study.get('planner')
    if planner not in POLICY['planners'] or study.get('planner_binding')!=POLICY['planners'][planner]:
        raise IntegrityError('task lacks its frozen planner assignment')
    return planner


def degenerate_tests(task):
    """Test pairs that ask nothing (2026-10-11, after v1-b14 verified some): the answer equals its test input while no
    training pair leaves its input unchanged (only a demonstrated no-op may recur in a test), or the answer is a cleared
    canvas: the test input's size with every cell its most common colour (ties: the lowest). A smaller one-colour answer
    stays legal, whatever its colour (v1-b14's da93ab30 answers a 3x3 square in its dense input's commonest colour)."""
    same_shown = any(p['input'] == p['output'] for p in task['train'])
    found = []
    for n, p in enumerate(task['test'], 1):
        counts = Counter(c for row in p['input'] for c in row)
        ground = min(counts, key=lambda c: (-counts[c], c))
        if p['output'] == p['input'] and not same_shown:
            found.append(f'test {n} answer equals its input and no training pair leaves its input unchanged')
        elif (len(p['output']), len(p['output'][0])) == (len(p['input']), len(p['input'][0])) and all(c == ground for row in p['output'] for c in row):
            found.append(f'test {n} answer is blank (the input cleared to {ground}, its most common colour)')
    return found


REUSE_MIN_FIGURE = 30  # copies smaller than this, off their own most common colour, are exempt (symbols, small digests)
REUSE_COVER = 0.8  # a training answer inside the test answer counts when it covers this share of the answer's area


def _d4(grid):
    g = [tuple(r) for r in grid]; out = []
    for _ in range(4):
        out += [g, [tuple(reversed(r)) for r in g]]
        g = [tuple(r) for r in zip(*g[::-1])]
    return out


def _inside(small, big):
    """True when `small` equals some window of `big` under one colour bijection."""
    h, w, H, W = len(small), len(small[0]), len(big), len(big[0])
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            fwd, back, ok = {}, {}, True
            for i in range(h):
                row, brow = small[i], big[r0 + i]
                for j in range(w):
                    a, b = row[j], brow[c0 + j]
                    if fwd.setdefault(a, b) != b or back.setdefault(b, a) != a:
                        ok = False; break
                if not ok:
                    break
            if ok:
                return True
    return False


def _figure(grid):
    """(colours, cells off the grid's own most common colour)."""
    counts = Counter(c for row in grid for c in row)
    ground = min(counts, key=lambda c: (-counts[c], c))
    return len(counts), sum(v for c, v in counts.items() if c != ground)


def reused_tests(task):
    """Test answers a solver could copy (2026-10-13; the v1-b16 screen's tasks 121 and 123): the whole answer is one window
    of a training output, or a training output covering most of the answer's area is one window of the answer, in either
    case turned or reflected and with its colours consistently renamed. A copied grid with fewer than REUSE_MIN_FIGURE cells
    off its own most common colour, or a single colour, is exempt. Calibrated on the bank: 73 of 7,565 tasks (about 1%)."""
    found = []
    for n, p in enumerate(task['test'], 1):
        out = p.get('output')
        if not out:
            continue
        answer, area = [tuple(r) for r in out], len(out) * len(out[0])
        colours, figure = _figure(out)
        for j, pair in enumerate(task['train'], 1):
            shown = pair['output']
            if colours >= 2 and figure >= REUSE_MIN_FIGURE and any(
                    len(t) >= len(answer) and len(t[0]) >= len(answer[0]) and _inside(answer, t) for t in _d4(shown)):
                found.append(f'test {n} answer is a part of training output {j}, turned, reflected or recoloured'); break
            k, f = _figure(shown)
            if k >= 2 and f >= REUSE_MIN_FIGURE and len(shown) * len(shown[0]) >= REUSE_COVER * area and any(
                    len(t) <= len(answer) and len(t[0]) <= len(answer[0]) and _inside(t, answer) for t in _d4(shown)):
                found.append(f'training output {j}, turned, reflected or recoloured, makes up most of test {n} answer'); break
    return found


def technical_prompt(values):
    q=Bindings();name=q.prompt_overrides['technical_verifier'][0]
    base=load_prompt(q.prompt_root/name)
    extra=load_prompt(Path(technical.__file__).resolve().parent/'prompts/technical_recheck_addendum.md')
    template=base[0]+'\n\n'+extra[0]
    return render_prompt(template,values),base[1]+'+'+extra[1],digest({'templates':[base[2],extra[2]],'combined':template})


from arcgen.astra_usage_diversity_20260913.runtime import validate_repair_identity


def idea_config(config, purpose, model, effort, namespace):
    """r10's binding with this revision's idea-model set (Astra and GPT-6.1 Sol)."""
    from arcgen.lib.factory_effort import runtime_profile
    from arcgen.lib.factory_supply import IDEA_PURPOSES
    from .common import IDEA_MODELS, SETTINGS
    if purpose not in IDEA_PURPOSES.values() or model not in IDEA_MODELS or effort not in SETTINGS['idea_efforts']:
        raise IntegrityError('unsupported frozen idea binding')
    config = runtime_profile(config, {'purpose': purpose, 'effort': effort,
        'account': config['run']['account'], 'namespace': namespace})
    config = deepcopy(config)
    config['role_profile'][purpose].update(model=model, reasoning_effort=effort)
    config['_idea_binding'] = {'purpose': purpose, 'model': model, 'effort': effort}
    config['_profile_hash'] = digest(config['role_profile'])
    config['_config_sha256'] = digest({'base': config['_config_sha256'], 'idea_binding': config['_idea_binding'], 'roles': config['role_profile']})
    config['_run_identity']['config_sha256'] = config['_config_sha256']
    return config
from arcgen.astra_broad_arm_program_20260908 import reviews
from .common import config_for, HOMES

def make_runtime(config, manifest, pool, budget, phase, window, namespace='downstream', lane='downstream', *, stage_bindings=None, packed=None, effort=None, category=None, extra_bindings=None, idea_model=None):
    from arcgen.lib.factory_window import window_bindings
    from arcgen.lib.factory_effort import runtime_profile
    from arcgen.astra_usage_diversity_20260913 import selection
    if idea_model:
        if not effort: raise IntegrityError('idea model requires an explicit effort binding')
        config = idea_config(config, effort[0], idea_model, effort[1], namespace)
    elif effort:
        purpose, level = effort
        config = runtime_profile(config, {'purpose': purpose, 'effort': level,
            'account': config['run']['account'], 'namespace': namespace})
    return Runtime(config, manifest, pool, budget=budget, phase=phase, window=window,
                   namespace=namespace, lane=lane, category=category,
                   stage_bindings={**window_bindings(), **reviews.stage_bindings(),
                       **selection.stage_bindings(packed), **(stage_bindings or {}), **(extra_bindings or {})})


from . import downstream
from arcgen.astra_broad_arm_program_20260908.program import invoke
def develop_window(root, window, allocation, *, pool, budget, production, manifest, mock=False):
    """Integration seam for a separately frozen future allocation, with no selector changes."""
    if window.get('editorial_mode') != 'bypass' or window['account'] not in HOMES:
        raise IntegrityError('r10 development must be boardless on a frozen account')
    conf = config_for(root / 'accounts' / window['account'] / window['id'], window['account'], seed=window['seed'], mock=mock)
    if 'paths' in production:
        conf['_paths'].update(active_registry=root / 'inputs/production/committed_tasks/active_registry.json')
    elif not mock:
        raise IntegrityError('development requires its frozen production snapshot')
    runtime = make_runtime(conf, manifest, pool, budget, window['phase'], window['id'])
    return {'runtime': runtime, 'result': downstream.run(runtime, window, allocation,
                                                        production['portfolio_digest'], invoke)}
