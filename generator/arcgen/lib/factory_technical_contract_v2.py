"""Reasoned non-pass outputs; complete pass semantics stay unchanged."""
from copy import deepcopy
from jsonschema import Draft202012Validator
from arcgen.lib.schemas import SchemaError, parse_grid, parse_json_object
from arcgen.lib.v4_schemas import STRICT_OUTPUT_SCHEMAS, _transport_schema

TECHNICAL_STAGE = 'v4_technical_verifier_v2'
C4_STRICT_SCHEMA = deepcopy(STRICT_OUTPUT_SCHEMAS['v4_technical_verifier'])
C4_STRICT_SCHEMA['properties']['derived_test_outputs']['minItems'] = 0
C4_STRICT_SCHEMA['properties']['derivation_failure_reason'] = {'type': 'string'}
C4_STRICT_SCHEMA['required'].append('derivation_failure_reason')
C4_SCHEMA = _transport_schema(C4_STRICT_SCHEMA)

def parse_negative_technical(data):
    """C4: only reasoned non-pass may have no independently derived output."""
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(dict(data))
    errors = list(Draft202012Validator(C4_STRICT_SCHEMA).iter_errors(value))
    if errors:
        error = errors[0]
        path = '$' + ''.join(f'[{x}]' if isinstance(x, int) else f'.{x}' for x in error.absolute_path)
        raise SchemaError(path, error.message)
    outputs = value['derived_test_outputs']
    reason = value['derivation_failure_reason'].strip()
    if not outputs:
        if value['status'] == 'pass':
            raise SchemaError('$.derived_test_outputs', 'pass requires independently derived outputs')
        if not reason:
            raise SchemaError('$.derivation_failure_reason', 'an unavailable derivation requires a concrete reason')
    elif reason:
        raise SchemaError('$.derivation_failure_reason', 'complete derivations require an empty failure reason')
    for i, grid in enumerate(outputs):
        parse_grid(grid, f'$.derived_test_outputs[{i}]')
    return value


def validate_test_count(value, expected):
    if type(expected) is not int or not 1 <= expected <= 3:
        raise SchemaError('$.test', 'expected test count must be in 1..3')
    count = len(value['derived_test_outputs'])
    if count and count != expected:
        raise SchemaError('$.derived_test_outputs', 'a nonempty derivation must cover every public test')
    if value['status'] == 'pass' and count != expected:
        raise SchemaError('$.derived_test_outputs', 'pass must cover every public test')

