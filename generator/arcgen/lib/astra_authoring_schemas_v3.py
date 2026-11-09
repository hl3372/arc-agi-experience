"""Opt-in authoring schema: public commitments, private execution, named colors.

No legacy dictionaries or parser registrations are modified by importing this file.
"""
from copy import deepcopy
from arcgen.lib import v4_schemas as v4
from arcgen.lib.v42_logical_leap import LOGICAL_PLAN_SCHEMA
from arcgen.lib.v43_episode_joint import STRICT_EXTENSION_SCHEMAS as JOINT_SCHEMAS
from arcgen.lib.v44_functional_scale import SCALE_PLAN_SCHEMA

VERSION = 'astra-authoring-v3'
PLAN_STAGE = 'astra_concrete_plan_v3'
REALIZER_STAGE = 'astra_task_realizer_v3'
REPAIR_STAGE = 'astra_targeted_repair_v3'
LOGICAL_STAGE = 'astra_logical_leap_plan_v3'
JOINT_STAGE = 'astra_episode_joint_plan_v3'
SCALE_STAGE = 'astra_functional_scale_plan_v3'
obj, arr = v4._object, v4._array
nullable = lambda value: {'anyOf': [deepcopy(value), {'type': 'null'}]}

overview = deepcopy(v4.EPISODE_OVERVIEW['properties'])
del overview['background_color']
overview['functional_color_roles']['items']['properties']['role'] = {
    'type': 'string', 'pattern': '^[a-z][a-z0-9_]{0,23}$'}
overview['palette_contract'] = obj({
    'mode': {'type': 'string', 'enum': ['symbolic', 'literal_numeric', 'unspecified']},
    'background_role': nullable(v4.TEXT),
    'opt_out_reason': v4.MAYBE_TEXT,
})
test_props = v4.TEST_STORYBOARD['properties']
PUBLIC_TEST_INPUT = obj({key: test_props[key] for key in (
    'pair_number', 'input_dimensions', 'exact_input_construction',
    'what_is_new_relative_to_training', 'training_support')})
PRIVATE_TEST_OUTPUT = obj({key: test_props[key] for key in (
    'pair_number', 'output_dimensions', 'exact_expected_output_construction',
    'precise_placements_quantities_relations_or_states', 'output_derivation',
    'test_only_semantics_audit', 'why_one_output_is_best_supported')})
PUBLIC_CONTRACT = obj({
    'semantic_core': v4.SEMANTIC_CORE,
    'episode_overview': obj(overview),
    'training_pairs': arr(v4.TRAINING_STORYBOARD, minimum=1, maximum=10),
    'test_inputs': arr(PUBLIC_TEST_INPUT, minimum=1, maximum=3),
    'invariants': arr(v4.TEXT, minimum=1, maximum=30),
    'allowed_realizer_freedom': arr(v4.TEXT, maximum=20),
})
PRIVATE_EXECUTION = obj({
    'test_outputs': arr(PRIVATE_TEST_OUTPUT, minimum=1, maximum=3),
    'cross_pair_concrete_audit': v4.CROSS_PAIR_AUDIT,
    'canvas_plan': deepcopy(v4.CONCRETE_PLAN['properties']['canvas_plan']),
    'deterministic_realization_procedure': v4.TEXT,
    'axis_self_assessment': deepcopy(v4.CONCRETE_PLAN['properties']['axis_self_assessment']['anyOf'][0]),
})
PLAN = obj({
    'contract_version': {'type': 'string', 'const': VERSION},
    'candidate_id': v4.TEXT,
    'status': {'type': 'string', 'enum': ['ready', 'abort']},
    'abort_reason': v4.MAYBE_TEXT,
    'public_contract': nullable(PUBLIC_CONTRACT),
    'private_execution': nullable(PRIVATE_EXECUTION),
})
REALIZER = deepcopy(v4.STRICT_OUTPUT_SCHEMAS['v4_task_realizer'])
REALIZER['properties']['contract_version'] = {'type': 'string', 'const': VERSION}
REALIZER['required'].append('contract_version')
REPAIR = deepcopy(v4.STRICT_OUTPUT_SCHEMAS['v4_targeted_repair'])
REPAIR['properties']['repaired_plan'] = nullable(PLAN)
REPAIR['properties']['repaired_realization'] = nullable(REALIZER)
REPAIR['properties']['contract_version'] = {'type': 'string', 'const': VERSION}
REPAIR['required'].append('contract_version')

STRICT_SCHEMAS = {PLAN_STAGE: PLAN, REALIZER_STAGE: REALIZER, REPAIR_STAGE: REPAIR}
for stage, old in ((LOGICAL_STAGE, LOGICAL_PLAN_SCHEMA),
                   (JOINT_STAGE, JOINT_SCHEMAS['v43_episode_joint_plan']),
                   (SCALE_STAGE, SCALE_PLAN_SCHEMA)):
    wrapper = deepcopy(old)
    wrapper['properties']['concrete_plan'] = nullable(PLAN)
    STRICT_SCHEMAS[stage] = wrapper
SCHEMAS = {key: v4._transport_schema(value) for key, value in STRICT_SCHEMAS.items()}
