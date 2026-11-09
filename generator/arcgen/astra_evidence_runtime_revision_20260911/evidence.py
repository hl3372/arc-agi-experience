"""One explicit decision per diagnostic, preserving every public observation."""
from copy import deepcopy
from arcgen.lib.schemas import SchemaError, parse_json_object
from arcgen.lib.v4_schemas import _transport_schema
from arcgen.astra_targeted_followup_20260911 import evidence as previous
from .structure import check

STAGE = 'astra_technical_retention_v4'
STRICT_SCHEMA = deepcopy(previous.STRICT_SCHEMA)
item = STRICT_SCHEMA['properties']['diagnostic_flag_assessments']['items']
item['properties']['reviewed_observation_ids'] = {
    'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True}
item['required'].append('reviewed_observation_ids')
SCHEMA = _transport_schema(STRICT_SCHEMA)


def grouped_diagnostics(diagnostics):
    value = deepcopy(diagnostics)
    groups = {}
    for i, flag in enumerate(value.get('flags', []), 1):
        code = flag['code']
        observation = {k: v for k, v in flag.items() if k != 'code'}
        groups.setdefault(code, {'code': code, 'observations': []})['observations'].append(
            {'observation_id': f'observation-{i:04}', **observation})
    for code in value.get('pair_economy_review_codes', []):
        groups.setdefault(code, {'code': code, 'observations': []})
    value['flags'] = [groups[code] for code in sorted(groups)]
    value['diagnostic_grouping'] = 'one_code_all_observations_v1'
    return value


def parse(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    strict = deepcopy(value)
    if isinstance(strict, dict):
        for row in strict.get('diagnostic_flag_assessments', []):
            if isinstance(row, dict):
                row.pop('justified', None)
    check(strict, STRICT_SCHEMA)
    ids = [row.pop('reviewed_observation_ids') for row in value['diagnostic_flag_assessments']]
    result = previous.parse(value)
    for row, observations in zip(result['diagnostic_flag_assessments'], ids, strict=True):
        row['reviewed_observation_ids'] = observations
    return result


def validate_response(value, values):
    value = parse(value)
    plain = deepcopy(value)
    for row in plain['diagnostic_flag_assessments']:
        row.pop('reviewed_observation_ids')
    previous.validate_response(plain, values)
    groups = {row['code']: {x['observation_id'] for x in row['observations']}
              for row in values['DIAGNOSTICS']['flags']}
    for row in value['diagnostic_flag_assessments']:
        if set(row['reviewed_observation_ids']) != groups.get(row['flag_code']):
            raise SchemaError('$.diagnostic_flag_assessments', 'each code must cover exactly its supplied observation IDs')
