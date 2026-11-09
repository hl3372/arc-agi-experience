"""Explicit diagnostic retention judgments, projected into the unchanged gate."""
from copy import deepcopy
from jsonschema import Draft202012Validator
from arcgen.lib.schemas import SchemaError, parse_json_object
from arcgen.lib.factory_technical_contract_v2 import (
    C4_STRICT_SCHEMA, parse_negative_technical, validate_test_count,
)
from arcgen.lib.v4_schemas import _transport_schema

STAGE = 'astra_technical_retention_v3'
STRICT_SCHEMA = deepcopy(C4_STRICT_SCHEMA)
ASSESSMENT = STRICT_SCHEMA['properties']['diagnostic_flag_assessments']['items']
ASSESSMENT['properties'].pop('justified')
ASSESSMENT['required'].remove('justified')
ASSESSMENT['properties'].update(
    diagnostic_observation={'type': 'string', 'enum': ['confirmed', 'disputed', 'uncertain']},
    retention_decision={'type': 'string', 'enum': ['acceptable', 'needs_repair', 'unresolved']},
)
ASSESSMENT['required'] += ['diagnostic_observation', 'retention_decision']
SCHEMA = _transport_schema(STRICT_SCHEMA)


def _legacy(value):
    result = deepcopy(value)
    for row in result['diagnostic_flag_assessments']:
        row['justified'] = row.pop('retention_decision') == 'acceptable'
        row.pop('diagnostic_observation')
    return result


def parse(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    if not isinstance(value, dict):
        raise SchemaError('$', 'technical response must be an object')
    # Native caches persist the parsed canonical response, including this
    # computed compatibility field. Validate it on replay; never reinterpret it.
    computed = []
    for row in value.get('diagnostic_flag_assessments', []):
        computed.append(('justified' in row, row.pop('justified', None)))
    errors = list(Draft202012Validator(STRICT_SCHEMA).iter_errors(value))
    if errors:
        raise SchemaError('$', errors[0].message)
    rows = value['diagnostic_flag_assessments']
    codes = [r['flag_code'] for r in rows]
    if len(codes) != len(set(codes)):
        raise SchemaError('$.diagnostic_flag_assessments', 'duplicate diagnostic code')
    parse_negative_technical(_legacy(value))
    if any(not r['assessment'].strip() for r in rows):
        raise SchemaError('$.diagnostic_flag_assessments', 'each decision requires a concrete explanation')
    if value['status'] == 'pass' and any(r['retention_decision'] != 'acceptable' for r in rows):
        raise SchemaError('$.status', 'pass conflicts with a repair/unresolved retention decision')
    for row, (present, prior) in zip(rows, computed, strict=True):
        derived = row['retention_decision'] == 'acceptable'
        if present and (type(prior) is not bool or prior is not derived):
            raise SchemaError('$.diagnostic_flag_assessments', 'computed retention boolean changed')
        row['justified'] = derived
    return value


def validate_response(value, values):
    value = parse(value)
    validate_test_count(value, len(values['PUBLIC_TASK']['test']))
    if value['task_id'] != values['TASK_ID'] or value['verifier_id'] != values['VERIFIER_ID']:
        raise SchemaError('$', 'technical response identity differs from its request')
    diagnostics = values['DIAGNOSTICS']
    allowed = {r['code'] for r in diagnostics.get('flags', [])}
    required = set(diagnostics.get('pair_economy_review_codes', []))
    allowed |= required
    assessed = {r['flag_code'] for r in value['diagnostic_flag_assessments']}
    if not assessed <= allowed:
        raise SchemaError('$.diagnostic_flag_assessments', 'assessment names an unsupplied diagnostic')
    if value['status'] == 'pass' and not required <= assessed:
        raise SchemaError('$.diagnostic_flag_assessments', 'pass must resolve every required diagnostic')


def native_view(value):
    """Verify the canonical projection already bound into the native call ledger.

    The old consumer still rejects absent or false flags and independently
    compares hidden arrays. Observation accuracy and prose never override that
    consumer. Explicit source fields remain in the episode's evidence record.
    """
    return parse(value)
