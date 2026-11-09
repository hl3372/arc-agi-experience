"""Two-account adapter using the existing candidate DAG, pools, and hard budgets."""
from pathlib import Path
from copy import deepcopy
from arcgen.lib.e18_evidence_gate import EVIDENCE_POLICY
from arcgen.lib.e18_calls import ATTEMPT_POLICY
from arcgen.lib.schemas import SchemaError
from arcgen.astra_broad_arm_program_20260908.runtime import Runtime as NativeRuntime
from arcgen.astra_broad_arm_program_20260908.program import invoke
from arcgen.astra_broad_arm_program_20260908 import reviews
from arcgen.astra_targeted_followup_20260911.pipeline import DispatchPaused
from . import downstream
from .common import HERE, REVISION, POLICY, IntegrityError, config_for, require_launch
from .contracts import Bindings, bindings
from .common import SETTINGS, HOMES, IDEA_MODELS, digest
from arcgen.lib.factory_supply import IDEA_PURPOSES


class Runtime(NativeRuntime):
    def __init__(self, config, *args, **kwargs):
        if (config['run']['account'] not in HOMES or config['_experiment_profile']['id'] != REVISION
                or config['_experiment_profile'].get('runtime_policy') != POLICY):
            raise IntegrityError('runtime requires an isolated r10 account configuration')
        if EVIDENCE_POLICY != 'grounded-difficulty-retention-v1' or ATTEMPT_POLICY != 'two-attempt-own-structure-feedback-v1':
            raise IntegrityError('fresh-process r10 overlay is required')
        for purpose, expected in SETTINGS['role_overrides'].items():
            if any(config['role_profile'][purpose].get(k) != v for k, v in expected.items()):
                raise IntegrityError('requested Sol role binding changed')
        idea_binding = config.get('_idea_binding')
        if idea_binding and (idea_binding['purpose'] not in IDEA_PURPOSES.values()
                or idea_binding['model'] not in IDEA_MODELS or idea_binding['effort'] not in SETTINGS['idea_efforts']
                or config['role_profile'][idea_binding['purpose']]['model'] != idea_binding['model']):
            raise IntegrityError('unbound idea-model override')
        for purpose, binding in config['role_profile'].items():
            if (purpose not in SETTINGS['role_overrides'] and purpose != 'transport_probe'
                    and not (idea_binding and purpose == idea_binding['purpose']) and binding['model'] != 'gpt-6-astra'):
                raise IntegrityError('other roles must remain on Astra')
        from .palette import Allocator
        self.palette_allocator = Allocator(kwargs['budget'].root)
        kwargs['contract_bindings'] = bindings(palette_allocator=self.palette_allocator)
        from . import adjudication, recheck, repair
        review_bindings = Bindings().stage_bindings
        kwargs['stage_bindings'] = {**kwargs.get('stage_bindings', {}),
            **{stage: review_bindings[stage] for stage in (adjudication.STAGE, recheck.STAGE)}}
        super().__init__(config, *args, **kwargs)
        self.q2 = Bindings()

    def check_launcher(self):
        if (self.budget.root / 'STOP_DISPATCH.json').exists():
            raise DispatchPaused('revision dispatch paused')
        try:
            return super().check_launcher()
        except IntegrityError as exc:
            if str(exc) == 'dispatch halted' and (self.budget.root / 'STOP_DISPATCH.json').exists():
                raise DispatchPaused('revision dispatch paused') from exc
            raise

    def invoke_rendered(self, **kwargs):
        from arcgen.lib.model import ModelInvocationError
        try:
            from . import adjudication, recheck, repair
            review_context = kwargs.pop('_review_context', None)
            if kwargs.get('stage') in (adjudication.STAGE, recheck.STAGE, repair.STAGE):
                stage = kwargs['stage']
                self.q2.validate_request(stage, kwargs['prompt'], review_context)
                from .common import frozen
                frozen(self.run_dir / 'review_context' / stage / (kwargs['call_id'] + '.json'), review_context)
                supplied = kwargs.get('response_validator')
                def validate_review(value):
                    self.q2.validate_response(stage, value, review_context)
                    if supplied: supplied(value)
                kwargs['response_validator'] = validate_review
                if stage == repair.STAGE:
                    kwargs['request_context'] = review_context
            elif review_context is not None:
                raise IntegrityError('review context supplied to a different stage')
            if (self.budget.root / 'STOP_DISPATCH.json').exists():
                raise DispatchPaused('revision dispatch paused')
            if kwargs['purpose'] == 'editorial_board':
                raise IntegrityError('editorial boards remain bypassed')
            if not self.factory_config['run'].get('mock'):
                require_launch(self)
            return super().invoke_rendered(**kwargs)
        except (DispatchPaused, ModelInvocationError):
            raise
        except BaseException as exc:
            if getattr(self, 'on_failure', None): self.on_failure(exc)
            raise

    def call(self, **kwargs):
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
            filename, stage = self.q2.prompt_overrides[name]
            template, version, identity = load_prompt(HERE / 'prompts' / filename)
            values, context = self.q2.prepare_values(name, kwargs['values'])
            prompt = render_prompt(template, values)
            self.q2.validate_request(stage, prompt, context)
            def validate(value):
                for key, expected in kwargs.get('expected', {}).items():
                    if value.get(key) != expected:
                        raise SchemaError('$', 'review response identity changed')
                evaluated = repair.expand(value, values) if name == 'targeted_repair' else value
                if name == 'targeted_repair':
                    validate_repair_identity(evaluated, record)
                if kwargs.get('validator'): kwargs['validator'](evaluated)
            result = self.invoke_rendered(stage=stage, call_id=kwargs['call_id'], prompt=prompt,
                purpose={'ambiguity_adjudicator': 'ambiguity_adjudication', 'difficulty_technical_recheck': 'technical_verification', 'targeted_repair': 'targeted_repair'}[name],
                prompt_template_version=version, prompt_template_hash=identity,
                parent_ids=kwargs['parent_ids'], arm_id=kwargs['arm_id'], role=kwargs['role'],
                response_validator=validate, _review_context=context).value
            if name == 'targeted_repair':
                expanded = repair.expand(result, values)
                from .common import frozen
                frozen(self.run_dir / 'repair_expansions' / (kwargs['call_id'] + '.json'), repair.receipt(result, expanded, values))
                return expanded
            return result
        return super().call(**kwargs)


def idea_config(config, purpose, model, effort, namespace):
    from arcgen.lib.factory_effort import runtime_profile
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


def develop_window(root, window, allocation, *, pool, budget, production, manifest, mock=False):
    """Integration seam for a separately frozen future allocation, with no selector changes."""
    if window.get('editorial_mode') != 'bypass' or window['account'] not in HOMES:
        raise IntegrityError('r10 development must be boardless on a frozen account')
    conf = config_for(root / 'accounts' / window['account'] / window['id'], window['account'], seed=window['seed'], mock=mock)
    if 'paths' in production:
        p = root / 'inputs/production'
        conf['_paths'].update(active_registry=p / 'committed_tasks/active_registry.json',
                              tasks=p / 'evaluation', fixed_anchor_manifest=p / 'config/lean_authoring_v1_anchors.json')
        annotation = next((str(Path(k).parent) for k in production['paths']
                           if Path(k).parent.name not in ('evaluation', 'config', 'committed_tasks')
                           and Path(k).suffix == '.json'), None)
        if annotation:
            conf['_paths']['annotations'] = p / annotation
    elif not mock:
        raise IntegrityError('development requires its frozen production snapshot')
    runtime = make_runtime(conf, manifest, pool, budget, window['phase'], window['id'])
    return {'runtime': runtime, 'result': downstream.run(runtime, window, allocation,
                                                        production['portfolio_digest'], invoke)}


def validate_repair_identity(value, record):
    """Reject frozen-input edits before accepting a physical repair response."""
    if value['status'] != 'repaired':
        return
    plan = value['repaired_plan']
    original = record['concrete_plan']
    if record.get('study', {}).get('primary_arm') == 'D1':
        if len(plan['public_contract']['training_pairs']) != len(original['public_contract']['training_pairs']):
            raise SchemaError('$.repaired_plan.public_contract.training_pairs', 'D1 repair must preserve its demonstration count')
    if value['repair_type'] == 'ambiguity':
        frozen_input_fields = ('pair_number', 'input_dimensions', 'exact_input_construction')
        inputs = lambda p: [{k: t[k] for k in frozen_input_fields} for t in p['public_contract']['test_inputs']]
        if inputs(plan) != inputs(original):
            raise SchemaError('$.repaired_plan.public_contract.test_inputs', 'ambiguity repair must copy the frozen test input constructions exactly')
        if value['repaired_realization']['task']['test'] != record['realization']['task']['test']:
            raise SchemaError('$.repaired_realization.task.test', 'ambiguity repair must copy every frozen test input exactly')
        if value['repaired_realization']['authoring_metadata']['expected_test_outputs'] != record['realization']['authoring_metadata']['expected_test_outputs']:
            raise SchemaError('$.repaired_realization.authoring_metadata.expected_test_outputs', 'ambiguity repair must preserve every frozen hidden test answer')
