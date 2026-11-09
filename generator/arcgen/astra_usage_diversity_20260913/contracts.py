"""Reuse r4 authoring; add only informed adjudication and a public repair challenge."""
from copy import deepcopy
from arcgen.lib.factory_runtime import StageBinding
from arcgen.lib.schemas import SchemaError
from arcgen.lib.astra_authoring_contract_v3 import public_diagnostics, validate_public_realization
from arcgen.lib.util import content_hash, load_prompt, render_prompt
from arcgen.astra_evidence_runtime_revision_20260911.contracts import Bindings as R4Bindings
from arcgen.astra_evidence_runtime_revision_20260911 import evidence
from .common import HERE, CARDS, component
from arcgen.astra_usage_diversity_20260913.supply import ARMS
from arcgen.lib import astra_authoring_schemas_v3 as schemas
from . import adjudication, recheck, repair

ADJ_FIELDS = {'TASK_ID', 'EPISODE_HASH', 'AUTHOR_SPECIFICATION', 'COMPLETE_TASK',
              'BLIND_RESULTS', 'CONCERNS', 'TECHNICAL_EVIDENCE'}
ADJ_PREPARED_FIELDS = ADJ_FIELDS | {'REQUIRED_CONCERN_IDS'}
TECH_FIELDS = {'TASK_ID', 'VERIFIER_ID', 'PUBLIC_TASK', 'PUBLIC_CONTRACT', 'DIAGNOSTICS', 'REPAIR_CHALLENGE'}
REPAIR_FIELDS = {'CANDIDATE_ID', 'REPAIR_TYPE', 'REPAIR_INSTRUCTION', 'AUTHOR_SPECIFICATION',
                 'EVALUATOR_EVIDENCE', 'ALLOW_MECHANICAL_TEST_INPUT_RESTORATION'}


class Bindings(R4Bindings):
    def __init__(self):
        super().__init__()
        self.prompt_root = HERE / 'prompts'
        self.prompt_overrides['ambiguity_adjudicator'] = ('difficulty_adjudicator.md', adjudication.STAGE)
        self.prompt_overrides['difficulty_technical_recheck'] = ('difficulty_technical_recheck.md', recheck.STAGE)
        self.prompt_overrides['targeted_repair'] = ('bounded_repair.md', repair.STAGE)
        self.stage_bindings.pop(schemas.REPAIR_STAGE)
        self.stage_bindings[repair.STAGE] = StageBinding(repair.SCHEMA, repair.parse, ('targeted_repair',))
        self.stage_bindings[adjudication.STAGE] = StageBinding(adjudication.SCHEMA, adjudication.parse, ('ambiguity_adjudication',))
        self.stage_bindings[recheck.STAGE] = StageBinding(recheck.SCHEMA, recheck.parse, ('technical_verification',))

    def planning_templates(self, primary, route_names):
        if primary not in ARMS:
            raise SchemaError('$', 'unknown local planning bridge')
        rows = [load_prompt(self.prompt_root / self.prompt_overrides[name][0])
                for name in ('concrete_planner', 'v44_quality_plan')]
        bridge = ARMS[primary].get('planner_bridge')
        if bridge:
            rows.append(component(CARDS / bridge))
        if primary != 'FORM': rows.append(component(CARDS / 'prompts/ordinary_territory.planner.md'))
        return rows + [load_prompt(self.prompt_root / self.prompt_overrides[name][0]) for name in route_names]

    def prepare_values(self, prompt_name, values):
        if prompt_name == 'targeted_repair':
            if set(values) != REPAIR_FIELDS or type(values['ALLOW_MECHANICAL_TEST_INPUT_RESTORATION']) is not bool:
                raise SchemaError('$', 'repair received unapproved context')
            prepared = deepcopy(values)
            return prepared, {'profile_id': self.profile_id, 'prompt_name': prompt_name, 'values': prepared}
        if prompt_name == 'ambiguity_adjudicator':
            if set(values) != ADJ_FIELDS:
                raise SchemaError('$', 'adjudication received unapproved context')
            prepared = deepcopy(values)
            prepared['REQUIRED_CONCERN_IDS'] = [r['concern_id'] for r in prepared['CONCERNS']]
        elif prompt_name == 'difficulty_technical_recheck':
            original = deepcopy(values); challenge = original.pop('REPAIR_CHALLENGE')
            prepared, _ = super().prepare_values('technical_verifier', original)
            prepared['REPAIR_CHALLENGE'] = deepcopy(challenge)
        else:
            return super().prepare_values(prompt_name, values)
        return prepared, {'profile_id': self.profile_id, 'prompt_name': prompt_name, 'values': deepcopy(prepared)}

    def validate_request(self, stage, prompt, context):
        if stage == repair.STAGE:
            if (not isinstance(context, dict) or context.get('profile_id') != self.profile_id
                    or context.get('prompt_name') != 'targeted_repair' or set(context['values']) != REPAIR_FIELDS):
                raise SchemaError('$', 'bounded repair requires its original context')
            if type(context['values']['ALLOW_MECHANICAL_TEST_INPUT_RESTORATION']) is not bool:
                raise SchemaError('$', 'unbound mechanical restoration policy')
            template = load_prompt(HERE / 'prompts/bounded_repair.md')[0]
            if prompt != render_prompt(template, context['values']):
                raise SchemaError('$', 'repair prompt differs from its exact context')
            return
        routes = {schemas.PLAN_STAGE: (), schemas.LOGICAL_STAGE: ('logical_leap_planner_addendum',),
                  schemas.JOINT_STAGE: ('v43_episode_joint_plan',),
                  schemas.SCALE_STAGE: ('evidence_scale_addendum',)}
        if stage in routes:
            if (not isinstance(context, dict) or context.get('profile_id') != self.profile_id
                    or context.get('prompt_name') != 'concrete_planner'
                    or context.get('q2_primary') not in ARMS):
                raise SchemaError('$', 'planning requires its explicit local bridge and canonical context')
            primary = context['q2_primary']
            specialist = {schemas.LOGICAL_STAGE: 'T', schemas.JOINT_STAGE: 'J'}
            if (stage in specialist and primary != specialist[stage]
                    or primary in ('J', 'T') and stage not in specialist):
                raise SchemaError('$', 'specialist source and planning route differ')
            rows = self.planning_templates(primary, routes[stage])
            if (context.get('q2_template_hashes') != [row[2] for row in rows]
                    or context.get('rendered_prompt_hash', content_hash(prompt)) != content_hash(prompt)
                    or prompt != render_prompt('\n\n'.join(row[0] for row in rows), context['values'])):
                raise SchemaError('$', 'planning prompt differs from its exact local components')
            return
        if stage not in (adjudication.STAGE, recheck.STAGE):
            return super().validate_request(stage, prompt, context)
        name = 'ambiguity_adjudicator' if stage == adjudication.STAGE else 'difficulty_technical_recheck'
        if not context or context.get('profile_id') != self.profile_id or context.get('prompt_name') != name:
            raise SchemaError('$', 'new review requires its original versioned request context')
        values = context['values']
        if stage == adjudication.STAGE:
            if set(values) != ADJ_PREPARED_FIELDS or content_hash(values['COMPLETE_TASK']) != values['EPISODE_HASH']:
                raise SchemaError('$', 'adjudicator context or episode identity changed')
            if values['CONCERNS'] != adjudication.concerns(values['BLIND_RESULTS']):
                raise SchemaError('$', 'material concerns were removed or changed')
            if values['REQUIRED_CONCERN_IDS'] != [r['concern_id'] for r in values['CONCERNS']]:
                raise SchemaError('$', 'required concern checklist changed')
        else:
            if set(values) != TECH_FIELDS:
                raise SchemaError('$', 'technical recheck crossed its public boundary')
            validate_public_realization(values['PUBLIC_CONTRACT'], values['PUBLIC_TASK'])
            if values['DIAGNOSTICS'] != evidence.grouped_diagnostics(public_diagnostics(values['PUBLIC_TASK'])):
                raise SchemaError('$', 'technical diagnostics must be derived from public grids')
            challenge = values['REPAIR_CHALLENGE']
            if set(challenge) != {'policy', 'original_episode_hash', 'changed_training_pairs', 'concerns'} or challenge['policy'] != adjudication.POLICY:
                raise SchemaError('$', 'repair challenge includes unapproved fields')
            if not challenge['concerns'] or not challenge['changed_training_pairs']:
                raise SchemaError('$', 'repair has no identified public teaching change')
            ids = []
            for row in challenge['concerns']:
                from arcgen.astra_evidence_runtime_revision_20260911.structure import check
                check(row, adjudication.CHALLENGE)
                ids.append(row['concern_id'])
            if len(ids) != len(set(ids)):
                raise SchemaError('$', 'duplicate repair target')
            valid = {'train_' + str(i + 1) for i in range(len(values['PUBLIC_TASK']['train']))}
            if not set(challenge['changed_training_pairs']) <= valid:
                raise SchemaError('$', 'repair cites unknown changed demonstrations')
        template = load_prompt(HERE / 'prompts' / self.prompt_overrides[name][0])[0]
        if prompt != render_prompt(template, values) or context.get('rendered_prompt_hash', content_hash(prompt)) != content_hash(prompt):
            raise SchemaError('$', 'review prompt differs from its exact versioned context')

    def validate_response(self, stage, value, context):
        if stage == repair.STAGE:
            if not context or context.get('profile_id') != self.profile_id:
                raise SchemaError('$', 'repair response lacks its original context')
            repair.expand(value, context['values'])
            return
        if stage not in (adjudication.STAGE, recheck.STAGE):
            return super().validate_response(stage, value, context)
        if not context or context.get('profile_id') != self.profile_id:
            raise SchemaError('$', 'review response lacks its original context')
        (adjudication if stage == adjudication.STAGE else recheck).validate_response(value, context['values'])


def bindings(*, palette_allocator=None):
    b = Bindings()
    def palette(plan, realization, target_background, palette_config):
        result = (palette_allocator.materialize(b.materialize_palette, plan, realization, target_background, palette_config)
                  if palette_allocator is not None else
                  b.materialize_palette(plan, realization, target_background=target_background, palette_config=palette_config))
        return result['plan'], result['realization'], result
    # New review stages use the adapter's explicit render path; the frozen
    # factory's authoring-only prompt whitelist remains unchanged.
    overrides = {k: v for k, v in b.prompt_overrides.items()
                 if k not in ('ambiguity_adjudicator', 'difficulty_technical_recheck')}
    return {'profile_id': b.profile_id, 'prompt_overrides': overrides,
            'stage_bindings': {k: v for k, v in b.stage_bindings.items()
                               if k not in (adjudication.STAGE, recheck.STAGE)}, 'prepare_values': b.prepare_values,
            'validate_request': b.validate_request, 'validate_response': b.validate_response,
            'materialize_palette': palette}
