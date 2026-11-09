"""Per-instance authoring bindings; only the technical retention contract changes."""
from copy import deepcopy
from arcgen.lib.factory_runtime import StageBinding
from arcgen.lib.factory_technical_contract_v2 import TECHNICAL_STAGE
from arcgen.lib.astra_authoring_contract_v3 import (
    public_diagnostics, validate_public_realization, _validate,
)
from arcgen.lib import astra_authoring_schemas_v3 as schemas
from arcgen.lib.schemas import SchemaError
from arcgen.lib.util import content_hash, load_prompt, render_prompt
from arcgen.astra_broad_arm_program_20260908.contracts import Q2Bindings
from .common import HERE, CARDS, read, component
from . import evidence

ARMS = {r['id']: r for r in read(CARDS / 'catalog.proposed.json')['arms']}


class RevisedBindings(Q2Bindings):
    def __init__(self):
        super().__init__()
        self.prompt_root = HERE / 'prompts'
        self.prompt_overrides['technical_verifier'] = ('astra_technical_retention_v3.md', evidence.STAGE)
        self.stage_bindings.pop(TECHNICAL_STAGE)
        self.stage_bindings[evidence.STAGE] = StageBinding(evidence.SCHEMA, evidence.parse,
                                                         ('technical_verification',))

    def planning_templates(self, primary, route_names):
        if primary not in ARMS:
            raise SchemaError('$', 'unknown frozen planning bridge')
        rows = [load_prompt(self.prompt_root / self.prompt_overrides[n][0])
                for n in ('concrete_planner', 'v44_quality_plan')]
        bridge = ARMS[primary].get('planner_bridge')
        if bridge:
            rows.append(component(CARDS / bridge))
        return rows + [load_prompt(self.prompt_root / self.prompt_overrides[n][0]) for n in route_names]

    def validate_request(self, stage, prompt, context):
        if stage != evidence.STAGE:
            return super().validate_request(stage, prompt, context)
        if (not isinstance(context, dict) or context.get('profile_id') != self.profile_id
                or context.get('prompt_name') != 'technical_verifier'):
            raise SchemaError('$', 'technical verification requires bound public context')
        if context.get('rendered_prompt_hash', content_hash(prompt)) != content_hash(prompt):
            raise SchemaError('$', 'technical prompt hash differs from bound context')
        values = context['values']
        if set(values) != {'TASK_ID', 'VERIFIER_ID', 'PUBLIC_TASK', 'PUBLIC_CONTRACT', 'DIAGNOSTICS'}:
            raise SchemaError('$', 'technical request crossed the public-only boundary')
        _validate(schemas.PUBLIC_CONTRACT, values['PUBLIC_CONTRACT'])
        validate_public_realization(values['PUBLIC_CONTRACT'], values['PUBLIC_TASK'])
        if public_diagnostics(values['PUBLIC_TASK']) != values['DIAGNOSTICS']:
            raise SchemaError('$.DIAGNOSTICS', 'diagnostics must be recomputed from public grids')
        template = load_prompt(self.prompt_root / self.prompt_overrides['technical_verifier'][0])[0]
        if render_prompt(template, values) != prompt:
            raise SchemaError('$', 'technical prompt differs from the exact versioned template')

    def validate_response(self, stage, value, context):
        if stage != evidence.STAGE:
            return super().validate_response(stage, value, context)
        if not context or context.get('profile_id') != self.profile_id:
            raise SchemaError('$', 'technical response requires the original request context')
        evidence.validate_response(value, context['values'])


def bindings():
    binding = RevisedBindings()
    def palette(plan, realization, target_background, palette_config):
        result = binding.materialize_palette(plan, realization, target_background=target_background,
                                             palette_config=palette_config)
        return result['plan'], result['realization'], result
    return {'profile_id': binding.profile_id, 'prompt_overrides': deepcopy(binding.prompt_overrides),
            'stage_bindings': binding.stage_bindings, 'prepare_values': binding.prepare_values,
            'validate_request': binding.validate_request, 'validate_response': binding.validate_response,
            'materialize_palette': palette}
