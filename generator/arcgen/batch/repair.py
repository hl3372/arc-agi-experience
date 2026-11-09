"""A bounded repair response with explicit references to immutable source data.

Raw responses and deterministic expansions have separate persisted identities.
The expansion supplies only fields the model is forbidden to change; normal
authoring, scope, technical and blind validation still apply afterward.
"""
from copy import deepcopy

from arcgen.lib import astra_authoring_schemas_v3 as schemas
from arcgen.lib.astra_authoring_contract_v3 import parse_stage
from arcgen.lib.astra_contract_bindings_v3 import validate_repair_scope
from arcgen.lib.schemas import SchemaError, parse_json_object
from arcgen.lib.util import content_hash
from arcgen.lib.v4_schemas import _transport_schema
from arcgen.astra_evidence_runtime_revision_20260911.structure import check

VERSION = 'astra-bounded-repair-v1'
STAGE = 'astra_bounded_repair_v1'
REUSE_PLAN = 'reuse_frozen_plan'
REUSE_TESTS = 'reuse_frozen_tests'
REUSE_ANSWERS = 'reuse_frozen_answers'
REUSE_INVARIANTS = 'reuse_frozen_invariants'


def reference(name):
    return {'type': 'string', 'const': name}


def object_branch(spec):
    return next(x for x in spec['anyOf'] if x.get('type') == 'object')


PLAN = deepcopy(schemas.PLAN)
PUBLIC = object_branch(PLAN['properties']['public_contract'])
PUBLIC['properties']['invariants'] = {'anyOf': [PUBLIC['properties']['invariants'], reference(REUSE_INVARIANTS)]}
object_branch(PLAN['properties']['public_contract'])['properties']['test_inputs'] = reference(REUSE_TESTS)
object_branch(PLAN['properties']['private_execution'])['properties']['test_outputs'] = reference(REUSE_ANSWERS)
REALIZATION = deepcopy(schemas.REALIZER)
FIDELITY = object_branch(REALIZATION['properties']['authoring_metadata'])['properties']['plan_fidelity']
FIDELITY['properties']['preserved_frozen_decisions'] = {'anyOf': [FIDELITY['properties']['preserved_frozen_decisions'], reference(REUSE_INVARIANTS)]}
TASK = object_branch(REALIZATION['properties']['task'])
TASK['properties']['test'] = {'anyOf': [TASK['properties']['test'], reference(REUSE_TESTS)]}
object_branch(REALIZATION['properties']['authoring_metadata'])['properties']['expected_test_outputs'] = reference(REUSE_ANSWERS)
STRICT_SCHEMA = deepcopy(schemas.REPAIR)
STRICT_SCHEMA['properties']['contract_version'] = reference(VERSION)
STRICT_SCHEMA['properties'].pop('repaired_plan')
STRICT_SCHEMA['required'].remove('repaired_plan')
STRICT_SCHEMA['required'].append('plan_update')
STRICT_SCHEMA['properties']['plan_update'] = {'anyOf': [PLAN, reference(REUSE_PLAN), {'type': 'null'}]}
STRICT_SCHEMA['properties']['repaired_realization'] = {'anyOf': [REALIZATION, {'type': 'null'}]}
STRICT_SCHEMA['properties']['training_support_updates'] = {
    'type': 'array', 'maxItems': 3, 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['pair_number', 'training_support'], 'properties': {
            'pair_number': {'type': 'integer', 'minimum': 1, 'maximum': 3},
            'training_support': deepcopy(schemas.PUBLIC_TEST_INPUT['properties']['training_support'])}}}
STRICT_SCHEMA['required'].append('training_support_updates')
SCHEMA = _transport_schema(STRICT_SCHEMA)


def parse(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    check(value, STRICT_SCHEMA)
    if value['status'] == 'unrepairable':
        if (not value['unrepairable_reason'].strip() or value['plan_update'] is not None
                or value['repaired_realization'] is not None or value['training_support_updates']):
            raise SchemaError('$', 'unrepairable requires a reason, null repair bodies and no support updates')
        return value
    if value['unrepairable_reason'].strip() or value['repaired_realization'] is None:
        raise SchemaError('$', 'repaired requires a realization and no refusal reason')
    if value['repair_type'] == 'mechanical':
        if value['plan_update'] != REUSE_PLAN or value['training_support_updates']:
            raise SchemaError('$.plan_update', 'mechanical repair must reference the frozen plan without teaching edits')
    elif not isinstance(value['plan_update'], dict):
        raise SchemaError('$.plan_update', 'ambiguity repair requires its revised training plan')
    r = value['repaired_realization']
    if r['status'] != 'realized' or r['task'] is None or r['authoring_metadata'] is None:
        raise SchemaError('$', 'repaired requires a realized episode')
    if value['repair_type'] == 'ambiguity' and r['task']['test'] != REUSE_TESTS:
        raise SchemaError('$.repaired_realization.task.test', 'ambiguity repair must reference frozen test inputs')
    return value


def expand(value, values):
    """Expand documented frozen references; never edit a supplied grid or verdict."""
    value = parse(value)
    if value['candidate_id'] != values['CANDIDATE_ID'] or value['repair_type'] != values['REPAIR_TYPE']:
        raise SchemaError('$', 'repair identity/type differs from its request')
    original = values['AUTHOR_SPECIFICATION']
    result = deepcopy(value)
    updates = result.pop('training_support_updates')
    result['repaired_plan'] = result.pop('plan_update')
    result['contract_version'] = schemas.VERSION
    if value['status'] == 'unrepairable':
        return parse_stage(schemas.REPAIR_STAGE, result)
    before, prior = original['concrete_plan'], original['realization']
    if value['repair_type'] == 'mechanical':
        result['repaired_plan'] = deepcopy(before)
    else:
        after = result['repaired_plan']
        if after['status'] != 'ready' or after['public_contract'] is None or after['private_execution'] is None:
            raise SchemaError('$.repaired_plan', 'repair must contain a ready plan')
        after['public_contract']['test_inputs'] = deepcopy(before['public_contract']['test_inputs'])
        after['private_execution']['test_outputs'] = deepcopy(before['private_execution']['test_outputs'])
        if after['public_contract']['invariants'] == REUSE_INVARIANTS:
            after['public_contract']['invariants'] = deepcopy(before['public_contract']['invariants'])
        seen = set()
        for update in updates:
            ordinal = update['pair_number']
            if ordinal in seen or ordinal > len(after['public_contract']['test_inputs']):
                raise SchemaError('$.training_support_updates', 'unknown or repeated test pair')
            seen.add(ordinal)
            after['public_contract']['test_inputs'][ordinal - 1]['training_support'] = deepcopy(update['training_support'])
    realization = result['repaired_realization']
    if realization['task']['test'] == REUSE_TESTS:
        realization['task']['test'] = deepcopy(prior['task']['test'])
    elif realization['task']['test'] != prior['task']['test']:
        if not values['ALLOW_MECHANICAL_TEST_INPUT_RESTORATION']:
            raise SchemaError('$.repaired_realization.task.test', 'mechanical test-input restoration is not enabled')
        if len(realization['task']['test']) != len(prior['task']['test']):
            raise SchemaError('$.repaired_realization.task.test', 'mechanical repair changed the test count')
        for old, new in zip(prior['task']['test'], realization['task']['test']):
            if (len(old['input']), len(old['input'][0])) != (len(new['input']), len(new['input'][0])):
                raise SchemaError('$.repaired_realization.task.test', 'mechanical repair changed test dimensions')
    realization['authoring_metadata']['expected_test_outputs'] = deepcopy(prior['authoring_metadata']['expected_test_outputs'])
    fidelity = realization['authoring_metadata']['plan_fidelity']
    if fidelity['preserved_frozen_decisions'] == REUSE_INVARIANTS:
        fidelity['preserved_frozen_decisions'] = deepcopy(before['public_contract']['invariants'])
    check(result, schemas.REPAIR)
    result = parse_stage(schemas.REPAIR_STAGE, result)
    validate_scope(original, result)
    return result


def validate_scope(original, response):
    # The r8 prompt allowed teaching-description updates but inherited validation
    # rejected them. Compare the frozen behavioral fields unchanged, allowing
    # only these two narrative fields to reflect the changed demonstration.
    comparison = deepcopy(response)
    if response['repair_type'] == 'ambiguity':
        before = original['concrete_plan']['public_contract']
        after = comparison['repaired_plan']['public_contract']
        for old, new in zip(before['test_inputs'], after['test_inputs']):
            new['training_support'] = deepcopy(old['training_support'])
        after['episode_overview']['intended_difficulty_and_inferability'] = before['episode_overview']['intended_difficulty_and_inferability']
        old = original['realization']['task']['train']
        new = response['repaired_realization']['task']['train']
        replacement = len(old) == len(new) and sum(a != b for a, b in zip(old, new)) <= 1
        addition = len(new) == len(old) + 1 and any(new[:i] + new[i+1:] == old for i in range(len(new)))
        from arcgen.lib.astra_contract_bindings_v3 import chain_rebuild
        if not (replacement or addition or chain_rebuild(old, new)):
            raise SchemaError('$.repaired_realization.task.train', 'ambiguity repair may add or replace at most one actual training pair')
        # Descriptions may refer to the new teaching pair. Exclude only these
        # three explanatory fields from the inherited plan-pair change count.
        # Construction, dimensions, outputs and all other semantic fields remain
        # subject to the exact one-pair constraint, independently of grid checks.
        narrative = ('deliberate_overlap_with_other_pairs', 'why_overlap_is_harmless',
                     'what_it_helps_communicate')
        old_pairs = before['training_pairs']
        new_pairs = after['training_pairs']
        def substantive(pair):return {k:v for k,v in pair.items() if k not in narrative}
        for new_pair in new_pairs:
            matches = [p for p in old_pairs if substantive(p) == substantive(new_pair)]
            if len(matches) == 1:
                for field in narrative:new_pair[field] = deepcopy(matches[0][field])
    validate_repair_scope(original, comparison)


def receipt(raw, expanded, values):
    return {'policy': VERSION, 'raw_response_hash': content_hash(raw),
            'source_specification_hash': content_hash(values['AUTHOR_SPECIFICATION']),
            'expanded_response_hash': content_hash(expanded),
            'derived_not_provider_response': True, 'expanded_response': deepcopy(expanded)}
