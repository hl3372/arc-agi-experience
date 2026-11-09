"""Explicit per-instance candidate bindings; no global installation or transport.

The integration owner supplies this object before constructing/freezing runtime
manifests. Both ordinary and direct rendered invocations must use its validators.
"""
from copy import deepcopy
from functools import partial
from pathlib import Path
from arcgen.lib.schemas import SchemaError
from arcgen.lib.util import content_hash, load_prompt, render_prompt
from . import astra_authoring_schemas_v3 as schemas
from .astra_authoring_contract_v3 import (parse_stage, validate_plan, validate_realization,
    technical_values, materialize_palette, legacy_plan_view, structural_fidelity,
    validate_public_realization)
from .factory_technical_contract_v2 import (TECHNICAL_STAGE, C4_SCHEMA,
    parse_negative_technical, validate_test_count)

PROFILE_ID = 'astra_authoring_contract_r2'
PROMPT_OVERRIDES = {
    'concrete_planner': ('astra_concrete_planner_v3.md', schemas.PLAN_STAGE),
    'task_realizer': ('astra_task_realizer_v3.md', schemas.REALIZER_STAGE),
    'targeted_repair': ('astra_targeted_repair_v3.md', schemas.REPAIR_STAGE),
    'technical_verifier': ('astra_technical_verifier_v2.md', TECHNICAL_STAGE),
    'v44_quality_plan': ('astra_quality_planning_addendum_v3.md', schemas.PLAN_STAGE),
    'logical_leap_planner_addendum': ('astra_logical_leap_planner_addendum_v3.md', schemas.LOGICAL_STAGE),
    'v43_episode_joint_plan': ('astra_episode_joint_planning_addendum_v3.md', schemas.JOINT_STAGE),
    'evidence_scale_addendum': ('astra_evidence_scale_addendum_v3.md', schemas.SCALE_STAGE),
}

def stage_bindings():
    from arcgen.lib.factory_runtime import StageBinding
    result = {}
    for stage, schema in schemas.SCHEMAS.items():
        purpose = ('realization' if stage == schemas.REALIZER_STAGE else
                   'targeted_repair' if stage == schemas.REPAIR_STAGE else 'planning')
        result[stage] = StageBinding(deepcopy(schema), partial(parse_stage, stage), (purpose,))
    result[TECHNICAL_STAGE] = StageBinding(deepcopy(C4_SCHEMA), parse_negative_technical,
                                         ('technical_verification',))
    return result

class AstraContractBindings:
    profile_id = PROFILE_ID
    prompt_overrides = PROMPT_OVERRIDES
    materialize_palette = staticmethod(materialize_palette)
    legacy_plan_view = staticmethod(legacy_plan_view)
    structural_fidelity = staticmethod(structural_fidelity)

    def __init__(self, prompt_root):
        self.prompt_root = Path(prompt_root)
        self.stage_bindings = stage_bindings()

    def manifest(self):
        module_root = Path(__file__).resolve().parent
        files = {p.name: content_hash(p.read_bytes()) for p in sorted(module_root.glob('*_v[23].py'))}
        return {'profile_id': self.profile_id, 'contract_revision': 'r2-02', 'authoring_version': schemas.VERSION,
                'repair_agreement_identity': 'v4_targeted_repair_v2:exact-ledgers/no-normalization',
                'technical_identity': TECHNICAL_STAGE,
                'prompt_overrides': {key: {'file': name, 'stage': stage,
                    'sha256': content_hash((self.prompt_root / name).read_bytes())}
                    for key,(name,stage) in self.prompt_overrides.items()}, 'modules': files}

    def prepare_values(self, prompt_name, values):
        if prompt_name == 'technical_verifier':
            prepared = technical_values(values)
        elif prompt_name == 'blind_solver':
            # Same legacy template and fields, but a strict public-only shape.
            from .astra_authoring_contract_v3 import public_diagnostics
            public_diagnostics(values['PUBLIC_TASK'])
            if set(values) != {'TASK_ID', 'PUBLIC_TASK', 'SOLVER_ID'}:
                raise SchemaError('$', 'blind solver receives only its unchanged public fields')
            prepared = deepcopy(dict(values))
        else:
            prepared = deepcopy(dict(values))
            if prompt_name == 'task_realizer': validate_plan(prepared['CONCRETE_PLAN'])
        context = {'profile_id': self.profile_id, 'prompt_name': prompt_name,
                   'values': deepcopy(prepared)}
        return prepared, context

    def bind_request(self, stage, rendered_prompt, request_context):
        context = deepcopy(request_context)
        context.update(stage=stage, rendered_prompt_hash=content_hash(rendered_prompt))
        self.validate_request(stage, rendered_prompt, context)
        return context

    def validate_request(self, stage, rendered_prompt, request_context):
        if stage not in self.stage_bindings:
            return
        c = request_context
        if not isinstance(c, dict) or c.get('profile_id') != self.profile_id or 'values' not in c:
            raise SchemaError('$', 'versioned invocation requires its bound request context')
        if 'rendered_prompt_hash' in c and c['rendered_prompt_hash'] != content_hash(rendered_prompt):
            raise SchemaError('$', 'request context differs from rendered invocation')
        compositions = {
            schemas.PLAN_STAGE: [('concrete_planner', 'v44_quality_plan'), ('concrete_planner',)],
            schemas.LOGICAL_STAGE: [('concrete_planner', 'v44_quality_plan', 'logical_leap_planner_addendum')],
            schemas.JOINT_STAGE: [('concrete_planner', 'v44_quality_plan', 'v43_episode_joint_plan')],
            schemas.SCALE_STAGE: [('concrete_planner', 'v44_quality_plan', 'evidence_scale_addendum')],
            schemas.REALIZER_STAGE: [('task_realizer',)], schemas.REPAIR_STAGE: [('targeted_repair',)],
            TECHNICAL_STAGE: [('technical_verifier',)],
        }
        renders = []
        for names in compositions[stage]:
            if c.get('prompt_name') != names[0]: continue
            texts = [load_prompt(self.prompt_root / self.prompt_overrides[n][0])[0] for n in names]
            renders.append(render_prompt('\n\n'.join(texts), c['values']))
        if rendered_prompt not in renders:
            raise SchemaError('$', 'rendered request differs from versioned templates and bound values')
        if stage == TECHNICAL_STAGE:
            values = c['values']
            if c['prompt_name'] != 'technical_verifier' or set(values) != {'TASK_ID','VERIFIER_ID','PUBLIC_CONTRACT','PUBLIC_TASK','DIAGNOSTICS'}:
                raise SchemaError('$', 'technical request context has the wrong boundary')
            from .astra_authoring_contract_v3 import _validate, public_diagnostics
            _validate(schemas.PUBLIC_CONTRACT, values['PUBLIC_CONTRACT'])
            validate_public_realization(values['PUBLIC_CONTRACT'], values['PUBLIC_TASK'])
            if values['DIAGNOSTICS'] != public_diagnostics(values['PUBLIC_TASK']):
                raise SchemaError('$.DIAGNOSTICS', 'must be recomputed solely from public grids')
            text, _, _ = load_prompt(self.prompt_root / self.prompt_overrides['technical_verifier'][0])
            if render_prompt(text, values) != rendered_prompt:
                raise SchemaError('$', 'technical rendered request must equal the versioned public-only template')
        elif stage == schemas.REALIZER_STAGE:
            validate_plan(c['values']['CONCRETE_PLAN'])

    def validate_response(self, stage, parsed_response, request_context):
        if stage not in self.stage_bindings: return
        if request_context is None or request_context.get('profile_id') != self.profile_id:
            raise SchemaError('$', 'response validation requires the original request context')
        values = request_context['values']
        if stage == TECHNICAL_STAGE:
            validate_test_count(parsed_response, len(values['PUBLIC_TASK']['test']))
            if parsed_response['task_id'] != values['TASK_ID'] or parsed_response['verifier_id'] != values['VERIFIER_ID']:
                raise SchemaError('$', 'technical response identity differs from request')
        elif stage == schemas.REALIZER_STAGE:
            validate_realization(values['CONCRETE_PLAN'], parsed_response)
        elif stage == schemas.REPAIR_STAGE:
            if parsed_response['candidate_id'] != values['CANDIDATE_ID'] or parsed_response['repair_type'] != values['REPAIR_TYPE']:
                raise SchemaError('$', 'repair identity/type differs from request')
            if parsed_response['status'] == 'repaired':
                validate_repair_scope(values['AUTHOR_SPECIFICATION'], parsed_response)
        else:
            if parsed_response['candidate_id'] != values['CANDIDATE_ID']:
                raise SchemaError('$.candidate_id', 'planner identity differs from request')
            nested = parsed_response.get('concrete_plan')
            if nested and nested['candidate_id'] != values['CANDIDATE_ID']:
                raise SchemaError('$.concrete_plan.candidate_id', 'nested planner identity differs from request')

def chained_episode(train):
    """True when the training pairs form one chain of states: every output except the last is the next input."""
    return len(train) >= 2 and all(train[i]['output'] == train[i + 1]['input'] for i in range(len(train) - 1))


def chain_rebuild(prior_train, current_train):
    """An ambiguity repair of a chained episode may rebuild the whole chain (2026-10-09): a fresh pair can only join
    a chain at its ends, where the rule fixes what it shows, so a deciding case usually needs a new starting state.
    The rebuilt episode must still be one chain, with the same number of pairs or one more."""
    return (chained_episode(prior_train) and chained_episode(current_train)
            and len(current_train) in (len(prior_train), len(prior_train) + 1))


def validate_repair_scope(original, response):
    """Assert the existing one-shot repair bounds against the actual input contract."""
    before, after = original['concrete_plan'], response['repaired_plan']
    prior, current = original['realization'], response['repaired_realization']
    if response['repair_type'] == 'mechanical':
        if before != after:
            raise SchemaError('$.repaired_plan', 'mechanical repair must retain the exact authoritative plan')
    else:
        p, q = before['public_contract'], after['public_contract']
        for key in ('semantic_core', 'test_inputs', 'invariants', 'allowed_realizer_freedom'):
            if p[key] != q[key]:
                raise SchemaError('$.repaired_plan.public_contract.'+key, 'ambiguity repair changed a frozen public commitment')
        a, b = deepcopy(p['episode_overview']), deepcopy(q['episode_overview'])
        a.pop('training_pair_count'); b.pop('training_pair_count')
        if a != b or before['private_execution']['test_outputs'] != after['private_execution']['test_outputs']:
            raise SchemaError('$.repaired_plan', 'ambiguity repair changed palette, test workings or episode commitments')
        clean = lambda rows: [{k:v for k,v in row.items() if k != 'pair_number'} for row in rows]
        a, b = clean(p['training_pairs']), clean(q['training_pairs'])
        one_replacement = len(a)==len(b) and sum(x!=y for x,y in zip(a,b))<=1
        one_addition = len(b)==len(a)+1 and any(b[:i]+b[i+1:]==a for i in range(len(b)))
        rebuilt = chain_rebuild(prior['task']['train'], current['task']['train']) and len(b)==len(current['task']['train'])
        if not (one_replacement or one_addition or rebuilt):
            raise SchemaError('$.repaired_plan.public_contract.training_pairs', 'ambiguity repair may add or replace at most one training pair')
        if prior['task']['test'] != current['task']['test']:
            raise SchemaError('$.repaired_realization.task.test', 'ambiguity repair changed a frozen test input')
    if prior['authoring_metadata']['expected_test_outputs'] != current['authoring_metadata']['expected_test_outputs']:
        raise SchemaError('$.repaired_realization.authoring_metadata.expected_test_outputs', 'repair changed frozen hidden outputs')

def build_contract_bindings():
    """The seven-member constructor seam agreed with the integration owner."""
    binding = AstraContractBindings(Path(__file__).resolve().parents[1] / 'prompts')
    def palette(plan, realization, target_background, palette_config):
        result = materialize_palette(plan, realization, target_background=target_background,
                                     palette_config=palette_config)
        return result['plan'], result['realization'], result
    return {'profile_id': binding.profile_id, 'prompt_overrides': deepcopy(binding.prompt_overrides),
            'stage_bindings': binding.stage_bindings, 'prepare_values': binding.prepare_values,
            'validate_request': binding.validate_request, 'validate_response': binding.validate_response,
            'materialize_palette': palette}
