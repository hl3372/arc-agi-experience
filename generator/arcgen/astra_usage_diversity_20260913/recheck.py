"""Narrow public teaching challenges for the existing post-repair verifier call."""
from copy import deepcopy
from arcgen.lib.schemas import SchemaError, parse_json_object
from arcgen.lib.v4_schemas import _transport_schema
from arcgen.astra_evidence_runtime_revision_20260911 import evidence as previous
from arcgen.astra_evidence_runtime_revision_20260911.structure import check
from .adjudication import obj, SHORT, TEXT, PAIRS, BOOL

STAGE = 'difficulty_technical_recheck_v1'
STRICT_SCHEMA = deepcopy(previous.STRICT_SCHEMA)
STRICT_SCHEMA['properties']['repair_challenge_assessments'] = {'type': 'array', 'items': obj({
    'concern_id': SHORT, 'resolved': BOOL, 'rival_still_fits': BOOL,
    'changed_training_pairs': PAIRS, 'discriminating_consequence': TEXT})}
STRICT_SCHEMA['required'].append('repair_challenge_assessments')
SCHEMA = _transport_schema(STRICT_SCHEMA)


def parse(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    strict = deepcopy(value)
    if isinstance(strict, dict):
        for row in strict.get('diagnostic_flag_assessments', []):
            if isinstance(row, dict): row.pop('justified', None)
    check(strict, STRICT_SCHEMA)
    assessments = value.pop('repair_challenge_assessments')
    if len({r['concern_id'] for r in assessments}) != len(assessments):
        raise SchemaError('$', 'duplicate repair-challenge assessment')
    if any(r['resolved'] is True and r['rival_still_fits'] is True for r in assessments):
        raise SchemaError('$', 'a still-fitting rival was not removed')
    result = previous.parse(value)
    result['repair_challenge_assessments'] = assessments
    if result['status'] == 'pass' and any(r['resolved'] is not True for r in assessments):
        raise SchemaError('$', 'unresolved repair challenge requires a non-pass')
    return result


def validate_response(value, values):
    value = parse(value); plain = deepcopy(value)
    rows = plain.pop('repair_challenge_assessments')
    previous.validate_response(plain, values)
    challenge = values['REPAIR_CHALLENGE']
    wanted = {r['concern_id']: r for r in challenge['concerns']}
    if {r['concern_id'] for r in rows} != set(wanted):
        raise SchemaError('$', 'all original repair concerns require rechecking')
    changed = set(challenge['changed_training_pairs'])
    for row in rows:
        if not set(row['changed_training_pairs']) <= changed:
            raise SchemaError('$', 'recheck cited an unchanged or nonexistent demonstration')
        if row['resolved'] is True:
            if not row['changed_training_pairs']:
                raise SchemaError('$', 'successful repair requires a concrete changed demonstration')
            if wanted[row['concern_id']]['competing_rule'] and row['rival_still_fits'] is not False:
                raise SchemaError('$', 'removal of the identified rival must be established')


def cleared(response, values):
    try:
        validate_response(response, values)
        rows = response['repair_challenge_assessments']
        return bool(rows) and all(r['resolved'] is True for r in rows)
    except (SchemaError, ValueError, KeyError, TypeError):
        return False
