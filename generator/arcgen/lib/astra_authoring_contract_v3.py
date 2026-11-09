"""Candidate-only scoped authoring and automatic palette materialization.

Prose is reviewed semantically by the technical verifier. Code validates shape,
exact repeated commitments, grids, palette references and identity, not execution
of arbitrary natural language. Historical plans are never upgraded here.
"""
from copy import deepcopy
from collections.abc import Mapping
import re
from jsonschema import Draft202012Validator
from arcgen.lib.schemas import SchemaError, parse_json_object
from arcgen.lib import v4_schemas as v4
from arcgen.lib.util import content_hash
from arcgen.lib.v4_diagnostics import check_plan_fidelity, complete_task
from arcgen.lib.v41_palette import (infer_episode_background, palette_permutation,
    remap_complete_task, palette_equivalent_hash)
from .astra_authoring_schemas_v3 import (VERSION, PLAN_STAGE, REALIZER_STAGE,
    REPAIR_STAGE, LOGICAL_STAGE, JOINT_STAGE, SCALE_STAGE, STRICT_SCHEMAS)
from .astra_public_literals_v3 import public_literal_checks, PREFIX as ROW_LITERAL_PREFIX

REF = re.compile(r'@([a-z][a-z0-9_]*)')
COLOR_WORD = re.compile(r'\b(?:black|blue|red|green|yellow|gray|grey|magenta|fuchsia|orange|azure|cyan|maroon|brown|white|pink|purple)\b', re.I)
NUMERIC_COLOR = re.compile(r'\b(?:colou?r|palette(?:\s+index)?|cell\s+value)\s*[:=]?\s*[0-9]\b|\b[0-9]\s*[- ]colou?red\b', re.I)
BARE_ROWS = re.compile(r'(?<![\w@])[0-9]{3,}(?:\s*[/|;\n]\s*[0-9]{3,})+(?!\w)')
WORKED_TEST = re.compile(r'\b(?:test|query|heldout)[\s_\[-]*\d+\b|\b(?:first|second|third)\s+test\b', re.I)
NUMBERED_TEST = re.compile(r'\b(?:test|query|heldout)[\s_\[-]*\d+\b', re.I)
ORDINAL_TEST = re.compile(r'(?:first|second|third)\s+test', re.I)
TEST_OPERATION = re.compile(
    r'\b(?:choose|select|pick|perform|run|conduct|administer|apply|evaluate|compare)\s+'
    r'(?:(?:the|a|unique|next|each|every|possible)\s+)*$|'
    r'\b(?:minimizing|maximizing|optimal)\s+$', re.I)
TEST_COST = re.compile(r'\b(?:cost|value)\s+of\s+(?:(?:the|a)\s+)?$', re.I)
TEST_PREDICATE = re.compile(r'\s+(?:asks|probes|measures|checks|partitions|splits|distinguishes|separates)\b', re.I)
DIAGNOSTIC_CONTEXT = re.compile(r'\b(?:diagnos\w*|binary|identif\w*|identit\w*|hypothes\w*|sensor\w*|assay\w*|measurement\w*)\b', re.I)
DOMAIN_TEST_DEFINITION = re.compile(r'\b(?:binary|diagnostic)\s+tests?\b', re.I)
DEICTIC_TEST = re.compile(
    r'\b(?:for|in|on)\s+(?:the\s+)?(?:first|second|third)\s+test\b|'
    r'\b(?:first|second|third)\s+test\s+has\s+\d+\b', re.I)
EPISODE_TEST_CONTEXT = re.compile(
    r'\b(?:ARC|held[\s_-]*out|private_execution)\b|'
    r'\b(?:test|query)\s+(?:pair|case|example|input|output|grid)\b|'
    r'\b(?:expected|private|final)\s+(?:answer|output|grid|shifts|rows|cells)\b|'
    r'\b(?:answer|output|result)\s*(?:is\b|are\b|=|:)', re.I)


def pair_specific_test_narrative(text, *, domain_test_rule=False):
    """Keep explicit pair references private; recognize a diagnostic operation.

    An ordinal alone also names an operation *inside* a puzzle. Permit that
    reading for an explicit diagnostic operation or a symbolic test-cost
    expression. A rule defining binary/diagnostic tests also supplies that
    meaning to its general invariants; they need not repeat the definition.
    Explicit episode/answer cues in the same commitment still fail closed.
    Other clauses and numbered
    references remain checked independently. This is a narrow syntactic guard,
    not a replacement for semantic fidelity/privacy review.
    """
    # Check indexed pairs separately, including overlapping "first test 1".
    if NUMBERED_TEST.search(text):
        return True
    for clause in re.split(r'[.;\n]', text):
        for match in WORKED_TEST.finditer(clause):
            before, after = clause[:match.start()], clause[match.end():]
            symbolic_cost = TEST_COST.search(before) and re.match(r'\s+[a-z]\b', after, re.I)
            operation = (TEST_OPERATION.search(before) or TEST_PREDICATE.match(after)
                         or symbolic_cost)
            if (not ORDINAL_TEST.fullmatch(match[0])
                    or EPISODE_TEST_CONTEXT.search(text)
                    or DEICTIC_TEST.search(text)
                    or (not domain_test_rule and
                        (not (DIAGNOSTIC_CONTEXT.search(text) or symbolic_cost) or not operation))):
                return True
    return False

def _validate(schema, value):
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        e = errors[0]
        path = '$' + ''.join(f'[{x}]' if isinstance(x, int) else f'.{x}' for x in e.absolute_path)
        raise SchemaError(path, e.message)

def strings(value, path='$'):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from strings(item, f'{path}[{i}]')
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(item, f'{path}.{key}')

def legacy_plan_view(plan):
    """One-way derived view for existing structural/dossier helpers, never input trust.

    Canonical author form must be retained separately. This is private: it contains
    worked tests and must never be projected using the historical public helper.
    """
    if plan.get('contract_version') != VERSION:
        raise SchemaError('$.contract_version', 'legacy plans require their own contract; no automatic upgrade')
    if plan['status'] != 'ready':
        return dict(candidate_id=plan['candidate_id'], status='abort', abort_reason=plan['abort_reason'],
                    semantic_core=None, episode_overview=None, training_pairs=[], test_pairs=[],
                    cross_pair_concrete_audit=None, canvas_plan=[], frozen_decisions=[],
                    allowed_realizer_freedom=[], deterministic_realization_procedure='', axis_self_assessment=None)
    public, private = plan['public_contract'], plan['private_execution']
    overview = deepcopy(public['episode_overview'])
    palette = overview.pop('palette_contract')
    roles = {r['role']: r['color'] for r in overview['functional_color_roles']}
    overview['background_color'] = roles.get(palette['background_role'])
    outputs = {x['pair_number']: x for x in private['test_outputs']}
    return dict(candidate_id=plan['candidate_id'], status=plan['status'], abort_reason=plan['abort_reason'],
                semantic_core=deepcopy(public['semantic_core']), episode_overview=overview,
                training_pairs=deepcopy(public['training_pairs']),
                test_pairs=[{**deepcopy(x), **deepcopy(outputs[x['pair_number']])} for x in public['test_inputs']],
                frozen_decisions=deepcopy(public['invariants']),
                allowed_realizer_freedom=deepcopy(public['allowed_realizer_freedom']),
                **{key: deepcopy(private[key]) for key in ('cross_pair_concrete_audit', 'canvas_plan',
                    'deterministic_realization_procedure', 'axis_self_assessment')})

def validate_plan(plan):
    _validate(STRICT_SCHEMAS[PLAN_STAGE], plan)
    if plan['status'] == 'abort':
        if not plan['abort_reason'].strip() or plan['public_contract'] is not None or plan['private_execution'] is not None:
            raise SchemaError('$', 'abort requires a reason and null contract sections')
        return
    if plan['abort_reason'].strip() or plan['public_contract'] is None or plan['private_execution'] is None:
        raise SchemaError('$', 'ready plan requires both scoped sections and no abort reason')
    p, q = plan['public_contract'], plan['private_execution']
    if [x['pair_number'] for x in q['test_outputs']] != [x['pair_number'] for x in p['test_inputs']]:
        raise SchemaError('$.private_execution.test_outputs', 'must cover public test inputs in the same order')
    validate_public_commitments(p)
    roles = p['episode_overview']['functional_color_roles']
    if len({r['role'] for r in roles}) != len(roles) or len({r['color'] for r in roles}) != len(roles):
        raise SchemaError('$.public_contract.episode_overview.functional_color_roles', 'use one unique named binding per color')
    v4._validate_plan(legacy_plan_view(plan))

def validate_public_commitments(public):
    """Same required public commitments for ordinary and direct technical calls."""
    from .astra_authoring_schemas_v3 import PUBLIC_CONTRACT
    _validate(PUBLIC_CONTRACT, public)
    domain_test_rule = bool(DOMAIN_TEST_DEFINITION.search(public['semantic_core']['central_rule']))
    for path, text in strings({k: public[k] for k in ('semantic_core', 'invariants', 'allowed_realizer_freedom')}):
        if not text.strip():
            raise SchemaError(path, 'public commitments cannot be blank')
        if pair_specific_test_narrative(text, domain_test_rule=domain_test_rule):
            raise SchemaError(path, 'pair-specific test narrative belongs in private_execution')
    overview = public['episode_overview']
    roles = overview['functional_color_roles']
    if len({r['role'] for r in roles}) != len(roles) or len({r['color'] for r in roles}) != len(roles):
        raise SchemaError('$.episode_overview.functional_color_roles', 'use one unique named binding per color')
    if overview['training_pair_count'] != len(public['training_pairs']) or overview['test_pair_count'] != len(public['test_inputs']):
        raise SchemaError('$.episode_overview', 'public pair counts disagree')
    for key in ('training_pairs', 'test_inputs'):
        if [p['pair_number'] for p in public[key]] != list(range(1, len(public[key])+1)):
            raise SchemaError('$.'+key, 'public pair numbers must be contiguous')

def validate_public_realization(public, task):
    """Counts, dimensions and recognized complete literals only; prose still reviewed."""
    validate_public_commitments(public)
    _validate(v4.PUBLIC_TASK, task)
    from arcgen.lib.schemas import parse_grid
    for split, key in (('train', 'training_pairs'), ('test', 'test_inputs')):
        if len(public[key]) != len(task[split]):
            raise SchemaError('$.PUBLIC_TASK.'+split, 'actual pair count differs from public commitment')
        for i, (story, pair) in enumerate(zip(public[key], task[split], strict=True)):
            for side in ('input', 'output') if split == 'train' else ('input',):
                grid = pair[side]
                parse_grid(grid, f'$.PUBLIC_TASK.{split}[{i}].{side}')
                if story[side+'_dimensions'] != {'height':len(grid), 'width':len(grid[0])}:
                    raise SchemaError(f'$.PUBLIC_TASK.{split}[{i}].{side}', 'dimensions differ from public commitment')
    checks = public_literal_checks(public, task)
    failures = [c for c in checks if not c['matched']]
    if failures:
        import json
        raise SchemaError('$.PUBLIC_TASK', 'complete public grid literal contradicts actual grids: '+json.dumps(failures, sort_keys=True))
    return checks

def validate_realization(plan, value):
    """Exact copies are checked, not accepted on an author consistency assertion."""
    validate_plan(plan)
    _validate(STRICT_SCHEMAS[REALIZER_STAGE], value)
    v4._validate_realizer(value)
    if value['candidate_id'] != plan['candidate_id']:
        raise SchemaError('$.candidate_id', 'realization must identify the requested plan')
    if value['status'] == 'refused':
        return
    if plan['status'] != 'ready':
        raise SchemaError('$', 'an aborted plan cannot be realized')
    p, metadata = plan['public_contract'], value['authoring_metadata']
    validate_public_realization(p, value['task'])
    if metadata['intended_rule'] != p['semantic_core']['central_rule']:
        raise SchemaError('$.authoring_metadata.intended_rule', 'must exactly copy authoritative public central_rule')
    if metadata['plan_fidelity']['preserved_frozen_decisions'] != p['invariants']:
        raise SchemaError('$.authoring_metadata.plan_fidelity.preserved_frozen_decisions', 'must exactly copy public invariants')
    if value['deviations'] != metadata['plan_fidelity']['deviations']:
        raise SchemaError('$.deviations', 'repair-agreement-v2: exact ledger agreement required; no normalization')

def parse_stage(stage, data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(dict(data))
    _validate(STRICT_SCHEMAS[stage], value)
    if stage == PLAN_STAGE:
        validate_plan(value)
    elif stage == REALIZER_STAGE:
        v4._validate_realizer(value)
    elif stage == REPAIR_STAGE:
        if value['status'] == 'unrepairable':
            if not value['unrepairable_reason'].strip() or value['repaired_plan'] is not None or value['repaired_realization'] is not None:
                raise SchemaError('$', 'unrepairable requires only a concrete reason')
        else:
            if value['unrepairable_reason'].strip() or value['repaired_plan'] is None or value['repaired_realization'] is None:
                raise SchemaError('$', 'repaired requires both new contract and realization')
            validate_realization(value['repaired_plan'], value['repaired_realization'])
            if value['repaired_plan']['status'] != 'ready' or value['repaired_realization']['status'] != 'realized':
                raise SchemaError('$', 'a repaired response must contain a ready plan and realized episode')
            if value['candidate_id'] != value['repaired_plan']['candidate_id']:
                raise SchemaError('$.candidate_id', 'repair IDs disagree')
    else:
        plan = value['concrete_plan']
        old = deepcopy(value)
        if plan is not None:
            validate_plan(plan)
            old['concrete_plan'] = legacy_plan_view(plan)
        if stage == LOGICAL_STAGE:
            from arcgen.lib.v42_logical_leap import parse_logical_extension
            parse_logical_extension('v42_logical_leap_plan', old)
        elif stage == JOINT_STAGE:
            from arcgen.lib.v43_episode_joint import parse_v43_extension
            parse_v43_extension('v43_episode_joint_plan', old)
        elif stage == SCALE_STAGE:
            from arcgen.lib.v44_functional_scale import parse_scale_plan
            parse_scale_plan(old)
        else:
            raise SchemaError('$.stage', 'unsupported authoring stage')
    return value

def public_diagnostics(task):
    """Recomputed solely from public grids; no saved full-task diagnostic accepted."""
    _validate(v4.PUBLIC_TASK, task)
    from arcgen.lib.v4_diagnostics import diagnose_public_arc_task_v4
    # Retain original C6 flags and strict consumer semantics. This existing
    # helper receives no hidden outputs and excludes surrogate output statistics.
    return diagnose_public_arc_task_v4(task)

def resolve_color_references(value, plan):
    """Derived display/legacy view; resolve only explicit references, never numbers.

    Canonical author form remains symbolic. Consumers that show a rule without
    its legend (notably the fixed-anchor board) must use this final-palette view.
    """
    roles = {r['role']: r['color'] for r in plan['public_contract']['episode_overview']['functional_color_roles']}
    def visit(x):
        if isinstance(x, str):
            return REF.sub(lambda m: str(roles[m[1]]) if m[1] in roles else m[0], x)
        if isinstance(x, list): return [visit(v) for v in x]
        if isinstance(x, dict): return {k: visit(v) for k,v in x.items()}
        return deepcopy(x)
    return visit(value)

def technical_values(values):
    plan = values['CONCRETE_PLAN']
    validate_plan(plan)
    if plan['status'] != 'ready':
        raise SchemaError('$.CONCRETE_PLAN', 'technical verification requires a ready authoring contract')
    task = deepcopy(values['PUBLIC_TASK'])
    public = deepcopy(plan['public_contract'])
    validate_public_realization(public, task)
    # Allowlist at the entire request boundary, not only a nested plan projector.
    return {'TASK_ID': values['TASK_ID'], 'VERIFIER_ID': values['VERIFIER_ID'], 'PUBLIC_TASK': task,
            'PUBLIC_CONTRACT': public, 'DIAGNOSTICS': public_diagnostics(task)}

def palette_eligibility(plan, realization):
    validate_realization(plan, realization)
    if realization['status'] != 'realized':
        return False, ['no realized episode']
    p = plan['public_contract']['episode_overview']
    palette, roles = p['palette_contract'], {r['role']: r['color'] for r in p['functional_color_roles']}
    reasons = []
    if palette['mode'] != 'symbolic': reasons.append('palette semantics: ' + palette['mode'])
    if palette['opt_out_reason'].strip(): reasons.append(palette['opt_out_reason'])
    source = roles.get(palette['background_role'])
    task = complete_task(realization)
    if source is None: reasons.append('no single named background')
    elif infer_episode_background(task) != source: reasons.append('declared background disagrees with unchanged input-based inference')
    used = {cell for split in task.values() for pair in split for grid in pair.values() for row in grid for cell in row}
    if not used <= set(roles.values()): reasons.append('grid color lacks a named binding')
    # Traverse all consumed plan/metadata prose, including private execution and
    # serialized rows. Exact named references need no numeric-prose substitution.
    for root in (plan, realization):
        for path, text in strings(root):
            if any(ref not in roles for ref in REF.findall(text)):
                reasons.append(path + ': unbound color reference')
            stripped = REF.sub('', text)
            if COLOR_WORD.search(stripped) or NUMERIC_COLOR.search(stripped) or BARE_ROWS.search(stripped):
                reasons.append(path + ': untyped color literal/serialized rows')
            # Complete numeric row literals are colors even at width/height one.
            # They remain useful fidelity evidence on the original palette, but
            # only named row literals can safely survive a changed binding.
            literal = ROW_LITERAL_PREFIX.match(text.strip())
            if literal:
                body = text.strip()[literal.end():].strip().removesuffix('.').strip()
                if any(re.fullmatch(r'[0-9\s]+', row.strip()) for row in body.split('/')):
                    reasons.append(path + ': numeric cells in an explicit row literal')
    return not reasons, list(dict.fromkeys(reasons))

def materialize_palette(plan, realization, *, target_background, palette_config, mapping=None):
    """Apply a single bijection to grids and the authoritative numeric bindings.

    Named references in every public/private string stay symbolic and resolve
    against the final bindings. No non-color number is rewritten. Opt-outs retain
    the exact original episode. The unchanged scheduler supplies the target.
    """
    eligible, reasons = palette_eligibility(plan, realization)
    original_plan, original_realization = deepcopy(plan), deepcopy(realization)
    original_task = complete_task(realization) if realization['status'] == 'realized' else None
    allowed = palette_config.get('allowed_background_colors', list(range(10)))
    if target_background is None:
        eligible = False
        reasons.append('unchanged scheduler opted out')
    elif type(target_background) is not int or target_background not in allowed:
        raise ValueError('target must come from the unchanged scheduler palette')
    roles = plan['public_contract']['episode_overview']['functional_color_roles']
    palette = plan['public_contract']['episode_overview']['palette_contract']
    source = next((r['color'] for r in roles if r['role'] == palette['background_role']), None)
    permutation = {i: i for i in range(10)}
    if eligible:
        permutation = palette_permutation(source, target_background) if mapping is None else dict(mapping)
        if set(permutation) != set(range(10)) or set(permutation.values()) != set(range(10)) or any(type(k) is not int or type(v) is not int for k, v in permutation.items()):
            raise ValueError('mapping must be one total bijection on integer colors 0..9')
        if permutation[source] != target_background:
            raise ValueError('mapping must realize the scheduled background')
    final_plan, final_realization = deepcopy(plan), deepcopy(realization)
    if eligible:
        for role in final_plan['public_contract']['episode_overview']['functional_color_roles']:
            role['color'] = permutation[role['color']]
        final_task = remap_complete_task(original_task, permutation)
        final_realization['task'] = {'train': deepcopy(final_task['train']), 'test': [{'input': deepcopy(p['input'])} for p in final_task['test']]}
        final_realization['authoring_metadata']['expected_test_outputs'] = [deepcopy(p['output']) for p in final_task['test']]
    else:
        final_task = original_task
    inverse = {v: k for k, v in permutation.items()}
    if original_task is not None:
        assert remap_complete_task(final_task, inverse) == original_task
        assert palette_equivalent_hash(original_task) == palette_equivalent_hash(final_task)
    validate_realization(final_plan, final_realization)
    return {'contract_version': VERSION, 'eligible': eligible, 'opt_out_reasons': reasons,
            'source_background': source, 'target_background': target_background,
            'mapping': permutation, 'inverse': inverse,
            'original_plan': original_plan, 'original_realization': original_realization,
            'plan': final_plan, 'realization': final_realization, 'complete_task': final_task,
            'legacy_plan_view': legacy_plan_view(final_plan),
            'resolved_legacy_plan_view': resolve_color_references(legacy_plan_view(final_plan), final_plan),
            'resolved_metadata_view': resolve_color_references(final_realization.get('authoring_metadata'), final_plan),
            'original_episode_hash': content_hash(original_task), 'final_episode_hash': content_hash(final_task),
            'palette_equivalent_hash': palette_equivalent_hash(final_task) if final_task else None,
            'inverse_round_trip': True, 'semantic_review_required': True}

def structural_fidelity(plan, realization):
    validate_realization(plan, realization)
    return check_plan_fidelity(legacy_plan_view(plan), realization)
