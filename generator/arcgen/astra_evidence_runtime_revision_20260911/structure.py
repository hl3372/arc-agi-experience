"""Bounded diagnostics about a role's own response; no answer comparisons."""
import ast
import json
import re
from jsonschema import Draft202012Validator
from arcgen.lib.schemas import SchemaError

MAX_ISSUES = 8
ROW_PREFIX = re.compile(r'^Rows top-to-bottom(?:, with slashes separating rows)?:\s*', re.I)
ROW_LIST = re.compile(r'\b([A-Za-z][A-Za-z0-9_]*) rows are\s*(\[[^\]]*\])')
TOKENS = re.compile(r'(?:@[a-z][a-z0-9_]*\s*)+')


def path_text(path):
    # JSON pointers contain field names/indices, never values or author answers.
    return '$' + ''.join(f'[{x}]' if isinstance(x, int) else '.' + str(x) for x in path)


def issue(path, code, **details):
    return {'path': path_text(path), 'code': code, **details}


def schema_issues(value, schema):
    def leaves(error):
        if error.context:
            for child in error.context:
                yield from leaves(child)
        else:
            yield error
    result = []
    for top in Draft202012Validator(schema).iter_errors(value):
        for e in leaves(top):
            # Ignore the null arm when a nullable object is present.
            if e.validator == 'type' and e.validator_value == 'null':
                continue
            details = {}
            if e.validator in {'minItems', 'maxItems', 'minLength', 'maxLength'}:
                details = {'limit': e.validator_value, 'actual': len(e.instance)}
            elif e.validator == 'required' and isinstance(e.instance, dict):
                details = {'missing': [k for k in e.validator_value if k not in e.instance]}
            elif e.validator in {'type', 'enum', 'const'}:
                details = {'expected': e.validator_value}
            # Never copy e.message: it may include the entire private payload.
            result.append(issue(list(e.absolute_path), 'schema_' + str(e.validator), **details))
            if len(result) >= MAX_ISSUES:
                return result
    return result


def _row_cells(row):
    if not isinstance(row, str):
        return None
    row = row.strip()
    if TOKENS.fullmatch(row):
        return row.split()
    if re.fullmatch(r'[0-9\s]+', row):
        return [x for x in row if x.isdigit()]
    return None


def plan_issues(plan, prefix=()):
    """Only complete row literals, including explicit named layer row lists."""
    if not isinstance(plan, dict) or plan.get('status') != 'ready':
        return []
    pub, private = plan.get('public_contract'), plan.get('private_execution')
    if not isinstance(pub, dict) or not isinstance(private, dict):
        return []
    result = []
    sections = [(pub.get('training_pairs', []), 'public_contract', 'training_pairs',
                 [('exact_input_construction', 'input_dimensions'), ('exact_output_construction', 'output_dimensions')]),
                (pub.get('test_inputs', []), 'public_contract', 'test_inputs', [('exact_input_construction', 'input_dimensions')]),
                (private.get('test_outputs', []), 'private_execution', 'test_outputs', [('exact_expected_output_construction', 'output_dimensions')])]
    for rows, scope, name, fields in sections:
        for i, pair in enumerate(rows):
            for text_field, dimensions_field in fields:
                text = pair.get(text_field, '')
                if not isinstance(text, str):
                    continue
                base = (*prefix, scope, name, i, text_field)
                full = ROW_PREFIX.match(text.strip())
                constructions = []
                if full:
                    body = text.strip()[full.end():].strip().removesuffix('.').strip()
                    constructions.append((body.split('/'), base, pair.get(dimensions_field)))
                for j, match in enumerate(ROW_LIST.finditer(text)):
                    try:
                        literal = ast.literal_eval(match[2])
                    except (SyntaxError, ValueError):
                        continue  # Natural-language prose is not a programming language.
                    if isinstance(literal, list):
                        constructions.append((literal, (*base, 'embedded_rows', j), None))
                for literals, path, dimensions in constructions:
                    parsed = [_row_cells(row) for row in literals]
                    if not parsed or any(row is None for row in parsed):
                        continue
                    width = dimensions.get('width') if isinstance(dimensions, dict) else len(parsed[0])
                    height = dimensions.get('height') if isinstance(dimensions, dict) else len(parsed)
                    if len(parsed) != height:
                        result.append(issue(path, 'literal_height', expected=height, actual=len(parsed)))
                    for row_index, cells in enumerate(parsed):
                        if len(cells) != width or not 1 <= len(cells) <= 30:
                            result.append(issue((*path, row_index), 'literal_width', expected=width, actual=len(cells)))
    return result[:MAX_ISSUES]


def response_issues(value, schema):
    result = schema_issues(value, schema)
    if result or not isinstance(value, dict):
        return result
    for prefix, item in [((), value), (('concrete_plan',), value.get('concrete_plan')),
                         (('repaired_plan',), value.get('repaired_plan'))]:
        if isinstance(item, dict):
            result.extend(plan_issues(item, prefix))
    for prefix, item in [((), value), (('repaired_realization',), value.get('repaired_realization'))]:
        if not isinstance(item, dict):
            continue
        if item.get('status') == 'refused':
            for key in ('task', 'authoring_metadata'):
                if item.get(key) is not None:
                    result.append(issue((*prefix, key), 'refusal_requires_null'))
        task = item.get('task')
        grids = []
        if isinstance(task, dict):
            for split in ('train', 'test'):
                for i, pair in enumerate(task.get(split, [])):
                    if isinstance(pair, dict):
                        for side in ('input', 'output'):
                            if side in pair:
                                grids.append(((*prefix, 'task', split, i, side), pair[side]))
        metadata = item.get('authoring_metadata')
        if isinstance(metadata, dict):
            grids.extend(((*prefix, 'authoring_metadata', 'expected_test_outputs', i), grid)
                         for i, grid in enumerate(metadata.get('expected_test_outputs', [])))
        for field in ('predicted_test_outputs', 'derived_test_outputs'):
            if isinstance(item.get(field), list):
                grids.extend(((*prefix, field, i), grid) for i, grid in enumerate(item[field]))
        for path, grid in grids:
            if isinstance(grid, list) and grid and all(isinstance(row, list) for row in grid):
                width = len(grid[0])
                for i, row in enumerate(grid):
                    if len(row) != width:
                        result.append(issue((*path, i), 'nonrectangular_grid', expected=width, actual=len(row)))
    codes = [row.get('flag_code') for row in value.get('diagnostic_flag_assessments', []) if isinstance(row, dict)]
    if len(codes) != len(set(codes)):
        result.append(issue(('diagnostic_flag_assessments',), 'one_assessment_per_code'))
    return result[:MAX_ISSUES]


def check(value, schema):
    problems = response_issues(value, schema)
    if problems:
        raise SchemaError('$', json.dumps(problems, sort_keys=True))


def retry_feedback(call_dir, schema):
    """An absent response/timeout has no invented correction or extra allowance."""
    path = call_dir / 'attempt-1.raw_response.txt'
    if not path.is_file():
        # An interruption may occur after archiving attempt one but before
        # persisting the second dispatch intent. Preserve the same feedback.
        path = call_dir / 'physical_attempts/attempt-01/attempt-1.raw_response.txt'
        if not path.is_file():
            return None
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        problems = [issue((), 'invalid_json', line=exc.lineno, column=exc.colno)]
    else:
        problems = response_issues(value, schema)
    if not problems:
        return None
    return {'policy': 'own_response_structure_v1', 'issues': problems,
            'instruction': 'Return a complete corrected schema response to the original request. Correct the listed structure using the original specification. Do not pad/truncate a grid, change the rule or frozen plan, or invent new evidence. If the frozen plan cannot be executed, use its refusal contract.'}
