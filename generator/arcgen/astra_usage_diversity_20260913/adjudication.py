"""Concrete, version-bound adjudication; schema checks are not semantic proof."""
from copy import deepcopy
from arcgen.lib.schemas import SchemaError, parse_json_object
from arcgen.lib.util import content_hash
from arcgen.lib.v4_schemas import _transport_schema
from arcgen.astra_evidence_runtime_revision_20260911.structure import check

POLICY = 'grounded-difficulty-retention-v1'
STAGE = 'difficulty_adjudication_v2'
TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 6000}
SHORT = {'type': 'string', 'minLength': 1, 'maxLength': 200}
BOOL = {'type': ['boolean', 'null']}
PAIRS = {'type': 'array', 'items': SHORT, 'uniqueItems': True}


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


PUBLIC_TARGET = obj({'competing_rule': {'type': 'string', 'maxLength': 6000},
                     'public_teaching_issue': TEXT})
RESOLUTION = obj({'concern_id': SHORT,
    'resolution': {'type': 'string', 'enum': ['consistent_intended_rule', 'execution_error', 'contradicted_by_training',
                            'less_supported', 'comparable_rival', 'defect', 'unresolved']},
    'training_pair_ids': PAIRS, 'explanation': TEXT,
    'repair_target': {'anyOf': [PUBLIC_TARGET, {'type': 'null'}]}})
CHALLENGE = obj({'concern_id': SHORT, 'competing_rule': {'type': 'string', 'maxLength': 6000},
                 'public_teaching_issue': TEXT, 'training_pair_ids': PAIRS})
STRICT_SCHEMA = obj({'task_id': SHORT, 'episode_hash': SHORT, 'policy': {'type': 'string', 'const': POLICY},
    'decision': {'type': 'string', 'enum': ['clear', 'repairable', 'serious_ambiguity', 'unresolved', 'defect']},
    'intended_rule_reasonably_supported': BOOL, 'comparably_simple_different_output_rule': BOOL,
    'failure_kind': {'type': 'string', 'enum': ['execution', 'discovery', 'mixed', 'unknown']},
    'public_support': {'type': 'array', 'items': obj({'training_pair_id': SHORT, 'evidence': TEXT})},
    'concern_resolutions': {'type': 'array', 'items': RESOLUTION},
    'reason': TEXT, 'repair_instruction': {'type': 'string', 'maxLength': 6000}})
SCHEMA = _transport_schema(STRICT_SCHEMA)
CLEAR_RESOLUTIONS = {'consistent_intended_rule', 'execution_error', 'contradicted_by_training', 'less_supported'}


def parse(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    check(value, STRICT_SCHEMA)
    ids = [r['concern_id'] for r in value['concern_resolutions']]
    if len(ids) != len(set(ids)):
        raise SchemaError('$', 'concern identities must be unique')
    decision, rival = value['decision'], value['comparably_simple_different_output_rule']
    resolutions = {r['resolution'] for r in value['concern_resolutions']}
    if decision == 'clear' and (value['intended_rule_reasonably_supported'] is not True
            or rival is not False or not value['public_support'] or not ids
            or not resolutions <= CLEAR_RESOLUTIONS):
        raise SchemaError('$', 'clear requires positive public support and every concrete concern resolved')
    if decision == 'serious_ambiguity' and rival is not True:
        raise SchemaError('$', 'confirmed ambiguity requires a confirmed comparable rival')
    if rival is True and decision not in ('serious_ambiguity', 'repairable'):
        raise SchemaError('$', 'a confirmed rival cannot be erased by clear, defect or uncertainty')
    if 'comparable_rival' in resolutions and rival is not True:
        raise SchemaError('$', 'comparable-rival resolution contradicts the aggregate assessment')
    if rival is True and 'comparable_rival' not in resolutions:
        raise SchemaError('$', 'a confirmed rival requires its concrete concern resolution')
    if decision == 'repairable':
        if 'unresolved' in resolutions:
            raise SchemaError('$', 'unsettled concerns cannot be forgotten during repair')
        required = [r for r in value['concern_resolutions'] if r['resolution'] in ('comparable_rival', 'defect')]
        if not value['repair_instruction'].strip() or not required:
            raise SchemaError('$', 'repair requires identified concerns and a concrete bounded instruction')
    elif value['repair_instruction'].strip():
        raise SchemaError('$', 'only a repairable decision may prescribe repair')
    for row in value['concern_resolutions']:
        needs_target = decision == 'repairable' and row['resolution'] in ('comparable_rival', 'defect')
        if needs_target != (row['repair_target'] is not None):
            raise SchemaError('$.concern_resolutions', 'repair_target must appear on each established repair concern and nowhere else: ' + row['concern_id'])
        if needs_target and not row['repair_target']['public_teaching_issue'].strip():
            raise SchemaError('$.concern_resolutions', 'repair target must state a concrete public teaching issue')
    return value


def repair_challenges(value):
    """Public recheck payload derived from the one canonical concern list.

    Never infer targets from private explanations or merge identities. The raw
    provider response stays unchanged; this projection carries every target.
    """
    value = parse(value)
    return [dict(concern_id=row['concern_id'], **deepcopy(row['repair_target']),
                 training_pair_ids=deepcopy(row['training_pair_ids']))
            for row in value['concern_resolutions'] if row['repair_target'] is not None]


def concerns(blind_rows):
    """Build the input inventory, not a favorable subset chosen by the judge."""
    out = [{'concern_id': 'public-teaching-support', 'kind': 'public_support',
            'procedure': 'Establish positive grid evidence and address any further concrete rival found during this review.'}]
    for wrapped in blind_rows:
        response = wrapped['response']
        identity = content_hash(response)[:16]
        out.append({'concern_id': identity + ':primary', 'kind': 'primary_rule',
                    'procedure': response['primary_inferred_rule']})
        for i, alternate in enumerate(response['strongest_plausible_alternatives']):
            out.append({'concern_id': identity + ':alternative:' + str(i),
                        'kind': 'alternative_rule', 'procedure': alternate['rule']})
        if response['test_requires_unsupported_guessing']:
            out.append({'concern_id': identity + ':support', 'kind': 'unsupported_guessing',
                        'procedure': 'Check which required convention lacks solver-visible support.'})
    return out


def validate_response(value, values):
    value = parse(value)
    if value['task_id'] != values['TASK_ID'] or value['episode_hash'] != values['EPISODE_HASH']:
        raise SchemaError('$', 'adjudication reviewed a different episode')
    wanted = {r['concern_id'] for r in values['CONCERNS']}
    supplied = {r['concern_id'] for r in value['concern_resolutions']}
    if supplied != wanted:
        raise SchemaError('$.concern_resolutions', 'every supplied concern requires exactly one resolution; '
                          + 'missing=' + repr(sorted(wanted - supplied))
                          + '; unexpected=' + repr(sorted(supplied - wanted)))
    training = {'train_' + str(i + 1) for i in range(len(values['COMPLETE_TASK']['train']))}
    for row in value['concern_resolutions']:
        if not set(row['training_pair_ids']) <= training:
            raise SchemaError('$', 'unknown training-pair evidence reference')
        if row.get('resolution') in ('contradicted_by_training', 'less_supported') and not row['training_pair_ids']:
            raise SchemaError('$', 'rebuttal must cite concrete solver-visible teaching evidence')
    if any(row['training_pair_id'] not in training for row in value['public_support']):
        raise SchemaError('$', 'unknown positive-support training pair')


def is_clear(value):
    try:
        return parse(value)['decision'] == 'clear'
    except (SchemaError, TypeError, ValueError, KeyError):
        return False


def confirmed_rival(value):
    return isinstance(value, dict) and value.get('comparably_simple_different_output_rule') is True
