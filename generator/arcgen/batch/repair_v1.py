"""Diversity-v1 bounded repair: copy-by-reference tokens for every frozen plan field.

r23's repair made the model re-type the whole plan; small rewordings of frozen text ("either order" -> "any order",
"Four demonstrations" -> "Five") then failed exact scope checks and lost valid tasks. Here the repair may reference
the frozen rule core, realizer freedom, episode overview (the pair count is updated by code) and every unchanged
training pair by token. Tokens are expanded first; the unchanged r23 expansion, scope and fidelity checks then apply.
"""
from copy import deepcopy
from arcgen.lib.schemas import SchemaError, parse_json_object
from arcgen.lib.v4_schemas import _transport_schema
from arcgen.astra_evidence_runtime_revision_20260911.structure import check
from . import repair

STAGE = 'diversity_v1_bounded_repair'
REUSE_CORE = 'reuse_frozen_semantic_core'
REUSE_FREEDOM = 'reuse_frozen_realizer_freedom'
REUSE_OVERVIEW = 'reuse_frozen_episode_overview'
PAIR_TOKENS = [f'reuse_frozen_training_pair_{k}' for k in range(1, 11)]

STRICT_SCHEMA = deepcopy(repair.STRICT_SCHEMA)
_plan = next(x for x in STRICT_SCHEMA['properties']['plan_update']['anyOf'] if x.get('type') == 'object')
_public = repair.object_branch(_plan['properties']['public_contract'])['properties']
_public['semantic_core'] = {'anyOf': [_public['semantic_core'], repair.reference(REUSE_CORE)]}
_public['allowed_realizer_freedom'] = {'anyOf': [_public['allowed_realizer_freedom'], repair.reference(REUSE_FREEDOM)]}
_public['episode_overview'] = {'anyOf': [_public['episode_overview'], repair.reference(REUSE_OVERVIEW)]}
_public['training_pairs'] = deepcopy(_public['training_pairs'])
_public['training_pairs']['items'] = {'anyOf': [_public['training_pairs']['items'], {'type': 'string', 'enum': PAIR_TOKENS}]}
SCHEMA = _transport_schema(STRICT_SCHEMA)
from arcgen.lib import e18_calls
e18_calls.SEMANTIC_FEEDBACK_STAGES.add(STAGE)  # a failed repair's retry sees the exact validation error


def parse(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    check(value, STRICT_SCHEMA)
    return value


def substitute(value, values):
    """Replace v1 reference tokens with the frozen source fields; the result is an ordinary r23 repair response."""
    value = parse(value)
    plan = value.get('plan_update')
    if not isinstance(plan, dict) or plan.get('public_contract') is None:
        return value
    before = values['AUTHOR_SPECIFICATION']['concrete_plan']['public_contract']
    public = plan['public_contract']
    if public['semantic_core'] == REUSE_CORE:
        public['semantic_core'] = deepcopy(before['semantic_core'])
    if public['allowed_realizer_freedom'] == REUSE_FREEDOM:
        public['allowed_realizer_freedom'] = deepcopy(before['allowed_realizer_freedom'])
    pairs, used = [], set()
    for ordinal, entry in enumerate(public['training_pairs'], 1):
        if isinstance(entry, str):
            k = PAIR_TOKENS.index(entry) + 1
            if k > len(before['training_pairs']) or k in used:
                raise SchemaError('$.plan_update.public_contract.training_pairs', 'unknown or repeated frozen pair token: ' + entry)
            used.add(k)
            entry = deepcopy(before['training_pairs'][k - 1])
        entry = deepcopy(entry); entry['pair_number'] = ordinal
        pairs.append(entry)
    public['training_pairs'] = pairs
    if public['episode_overview'] == REUSE_OVERVIEW:
        public['episode_overview'] = deepcopy(before['episode_overview'])
        public['episode_overview']['training_pair_count'] = len(pairs)
    return value


def expand(value, values):
    return repair.expand(substitute(value, values), values)


def receipt(raw, expanded, values):
    out = repair.receipt(raw, expanded, values)
    out['policy'] = STAGE
    return out
