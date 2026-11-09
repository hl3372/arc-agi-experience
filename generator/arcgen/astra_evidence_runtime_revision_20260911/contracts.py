"""Local authoring bindings over the qualified r3 plans and public boundary."""
from copy import deepcopy
from functools import partial
from arcgen.lib import astra_authoring_schemas_v3 as schemas
from arcgen.lib.astra_authoring_contract_v3 import parse_stage, public_diagnostics, validate_public_realization
from arcgen.lib.factory_runtime import StageBinding
from arcgen.lib.schemas import SchemaError
from arcgen.lib.util import content_hash, load_prompt, render_prompt
from arcgen.astra_targeted_followup_20260911.contracts import RevisedBindings as PreviousBindings
from arcgen.astra_targeted_followup_20260911 import evidence as previous_evidence
from .common import HERE
from . import evidence, structure


def parse_author(stage, data):
    from arcgen.lib.schemas import parse_json_object
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    structure.check(value, schemas.STRICT_SCHEMAS[stage])
    return parse_stage(stage, value)


class Bindings(PreviousBindings):
    def __init__(self):
        super().__init__()
        self.prompt_root = HERE / 'prompts'
        self.prompt_overrides = deepcopy(self.prompt_overrides)
        self.prompt_overrides['technical_verifier'] = ('technical.md', evidence.STAGE)
        self.prompt_overrides['task_realizer'] = ('realizer.md', schemas.REALIZER_STAGE)
        self.stage_bindings.pop(previous_evidence.STAGE)
        self.stage_bindings[evidence.STAGE] = StageBinding(evidence.SCHEMA, evidence.parse, ('technical_verification',))
        for stage in schemas.SCHEMAS:
            previous = self.stage_bindings[stage]
            self.stage_bindings[stage] = StageBinding(previous.schema, partial(parse_author, stage), previous.purposes)

    def planning_templates(self, primary, route_names):
        return super().planning_templates(primary, route_names) + [load_prompt(HERE / 'prompts/plan_preflight.md')]

    def prepare_values(self, prompt_name, values):
        prepared, context = super().prepare_values(prompt_name, values)
        if prompt_name == 'technical_verifier':
            prepared['DIAGNOSTICS'] = evidence.grouped_diagnostics(prepared['DIAGNOSTICS'])
            context['values'] = deepcopy(prepared)
        return prepared, context

    def validate_request(self, stage, prompt, context):
        if stage != evidence.STAGE:
            return super().validate_request(stage, prompt, context)
        if not isinstance(context, dict) or context.get('profile_id') != self.profile_id or context.get('prompt_name') != 'technical_verifier':
            raise SchemaError('$', 'technical verification requires its original public-only context')
        values = context['values']
        if set(values) != {'TASK_ID', 'VERIFIER_ID', 'PUBLIC_TASK', 'PUBLIC_CONTRACT', 'DIAGNOSTICS'}:
            raise SchemaError('$', 'technical request crossed its public-only boundary')
        validate_public_realization(values['PUBLIC_CONTRACT'], values['PUBLIC_TASK'])
        if values['DIAGNOSTICS'] != evidence.grouped_diagnostics(public_diagnostics(values['PUBLIC_TASK'])):
            raise SchemaError('$.DIAGNOSTICS', 'grouped diagnostics must preserve the exact public observations')
        # Resolved through the active override (byte-identical to this layer's file for the r23 prompt set).
        template = load_prompt(self.prompt_root / self.prompt_overrides['technical_verifier'][0])[0]
        if prompt != render_prompt(template, values) or context.get('rendered_prompt_hash', content_hash(prompt)) != content_hash(prompt):
            raise SchemaError('$', 'technical prompt differs from its versioned public context')

    def validate_response(self, stage, value, context):
        if stage != evidence.STAGE:
            return super().validate_response(stage, value, context)
        if not context or context.get('profile_id') != self.profile_id:
            raise SchemaError('$', 'technical response needs its original request context')
        evidence.validate_response(value, context['values'])


def bindings():
    binding = Bindings()
    def palette(plan, realization, target_background, palette_config):
        result = binding.materialize_palette(plan, realization, target_background=target_background, palette_config=palette_config)
        return result['plan'], result['realization'], result
    return {'profile_id': binding.profile_id, 'prompt_overrides': binding.prompt_overrides,
            'stage_bindings': binding.stage_bindings, 'prepare_values': binding.prepare_values,
            'validate_request': binding.validate_request, 'validate_response': binding.validate_response,
            'materialize_palette': palette}
