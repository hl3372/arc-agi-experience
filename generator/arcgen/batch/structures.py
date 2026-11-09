"""Structure lines' checks, capture and default-reading requests (2026-10-22).

A structure line is a second slot axis drawn on top of a content line or an arm (weakness_lines.json "structures"): it
decides how the evidence is laid out, never the mechanism. Its rivals live in their own record, `alternatives` (S1):
whole grids of any shape per rival, for every training pair and every test, keyed here by input grid (lines.tables), and
never mixed into the content line's points, whose checks stay untouched. Everything here records; nothing rejects a task.
Missing records give 'incomplete' (the screen counts a blank or missing rival as failing, S3).

- ELIMINATE: each rival reproduces every training pair but one, its killer, derived from the grids (S9); the killer
  differs at object level (a region of two or more cells, or one cell on open ground as SETTLE's single-cell form);
  killers are distinct; every rival's test answer differs from the truth; at least one more pair than rivals.
- RECOMBINE (S6, loosened 2026-10-23 after the screen's 19 of 21 opt-outs): for an order-free combination (the plan's
  "Combination:" invariant says order-free, or commute, or names no form), the test answer under both orders of the laws is
  the same and is the truth; for laws acting in turn (the invariant says "in turn"), the order records are not asked for
  and only recorded when present. Each law alone reproduces at least one training output (its pair without the other law)
  and differs from the truth on the test, and the single laws' test answers differ from each other (one law in two words
  is caught); the closest-pair replay differs from the truth. Pairs no single law reproduces (both laws at work apart)
  are recorded.
- PARSE (S2): the naive parse reproduces at least one training output exactly (a plausible competing parse), differs on
  at least one pair and on the test; the unit label grids are compared with same-colour 4-connected components (recorded).
- VARY (S7): at least two answer-relevant dimensions: dimension records whose nearest-pair-surface test answer differs
  from the truth; the per-grid surface record's variation is recorded.
- SHAPE (S9): fully mechanical: at least three distinct training output sizes, and no test output size equal to a naive
  frame (the test input's size, each training output size, the largest object's box in the test input); the per-pair
  quantity is recorded, outside the pass rate.
- RELEVANCE (S8): both the use-everything and the ignore-all readings differ from the truth on a training pair and on the
  test; RELATION's object check with the minimal contrast in at least two training grids (lines._objects).
"""
from collections import Counter
import re
from . import lines as L

KINDS = {'rival': r'r\d+', 'order': r'o\d+', 'single law': r'l\d+', 'replay': r'p\d+', 'naive parse': r'n\d+',
         'use everything': r'u\d+', 'ignore all': r'i\d+'}


def _norm(text):
    return re.sub(r'\s+', ' ', (text or '').strip().lower())


def records(table, name, kind=None):
    """The structure records of one line (by its line field, or every record when none names a line), optionally of one
    kind (by kind, or by id pattern when the kind is missing)."""
    rows = list((table.get('alternatives') or {}).values())
    mine = [a for a in rows if _norm(a.get('line')) == name.lower()] or rows
    if kind is None:
        return mine
    return [a for a in mine if _norm(a.get('kind')) == kind
            or (_norm(a.get('kind')) not in KINDS and re.fullmatch(KINDS[kind], a['id'].strip().lower()))]


def _train(a, episode):
    """[(pair number, pair, the record's grid or None)]."""
    return [(i, p, a['train'].get(L._key(p['input']))) for i, p in enumerate(episode['train'], 1)]


def _test(a, episode):
    return [(k, p, a['test'].get(L._key(p['input']))) for k, p in enumerate(episode['test'], 1)]


def _object_level(pair, grid):
    """A killer difference visible at object level (A1): a differing region of two or more cells, a different shape, or
    a single cell on open ground (SETTLE's single-cell form)."""
    d = L.difference(grid, pair['output'])
    return not d['same_shape'] or d['largest'] >= 2 or bool(L._open_ground(pair, grid))


def _tests_differ(a, episode):
    rows = _test(a, episode)
    given = [(k, p, g) for k, p, g in rows if g is not None]
    return {'tests_given': len(given), 'tests_missing': [k for k, _, g in rows if g is None],
            'tests_differ': bool(given) and all(g != p['output'] for _, p, g in given)}


def _status(parts):
    return 'incomplete' if 'incomplete' in parts else 'fail' if 'fail' in parts else 'pass'


# ------------------------------------------------------------------ per line


def eliminate(rule, episode, table, declared=None):
    rivals = records(table, 'ELIMINATE', 'rival')
    if not rivals:
        return {'status': 'incomplete', 'rivals': 0}
    rows = []
    for a in rivals:
        train = _train(a, episode)
        missing = [i for i, _, g in train if g is None]
        differing = [i for i, p, g in train if g is not None and g != p['output']]
        killer = differing[0] if len(differing) == 1 else None
        visible = killer is not None and _object_level(episode['train'][killer - 1], train[killer - 1][2])
        t = _tests_differ(a, episode)
        ok = killer is not None and visible and t['tests_differ']
        rows.append({'id': a['id'], 'note': a.get('note'), 'missing_pairs': missing, 'differing_pairs': differing,
                     'killer': killer, 'killer_visible': visible, **t,
                     'status': 'incomplete' if missing or t['tests_missing'] else 'pass' if ok else 'fail'})
    lo, hi = rule.get('rivals') or (2, 3)
    killers = [r['killer'] for r in rows if r['killer'] is not None]
    n = len(episode['train'])
    value = {'rivals': len(rows), 'declared_rivals': declared, 'records': rows, 'killers': killers,
             'distinct_killers': len(killers) == len(set(killers)), 'rival_count_ok': lo <= len(rows) <= hi,
             'pairs': n, 'pairs_enough': n >= len(rows) + 1}
    parts = [r['status'] for r in rows]
    parts.append('incomplete' if declared and len(rows) < declared else 'pass')  # a declared rival left out is missing
    parts.append('pass' if value['distinct_killers'] and value['rival_count_ok'] and value['pairs_enough'] else 'fail')
    value['status'] = _status(parts)
    return value


def recombine(rule, episode, table, declared=None, form=None):
    orders, laws, replay = (records(table, 'RECOMBINE', k) for k in ('order', 'single law', 'replay'))
    if not orders and not laws and not replay:
        return {'status': 'incomplete'}
    value, parts = {'form': form}, []
    # S6: the test answer under both orders, equal to each other and to the truth (order independence); laws acting in
    # turn (2026-10-23) are settled by the scene's own order, so their order records are only recorded
    tests = [[g for _, _, g in _test(a, episode)] for a in orders]
    truth = [p['output'] for p in episode['test']]
    complete = len(orders) >= 2 and all(all(g is not None for g in t) for t in tests)
    required = form != 'in turn'
    value['orders'] = {'records': len(orders), 'complete': complete, 'required': required,
                       'orders_agree': complete and all(t == tests[0] for t in tests),
                       'orders_equal_truth': complete and all(t == truth for t in tests)}
    if required:
        parts.append('incomplete' if not complete else 'pass' if value['orders']['orders_equal_truth'] else 'fail')
    rows = []
    for a in laws:
        train = _train(a, episode)
        given = [(i, p, g) for i, p, g in train if g is not None]
        t = _tests_differ(a, episode)
        reproduces = [i for i, p, g in given if g == p['output']]
        rows.append({'id': a['id'], 'note': a.get('note'), 'pairs_given': len(given), 'reproduces': reproduces, **t,
                     'status': 'incomplete' if not given or t['tests_missing'] else
                     'pass' if reproduces and t['tests_differ'] else 'fail'})
    answers = [L._key([g for _, _, g in _test(a, episode)]) for a in laws]
    value['laws'] = {'records': rows, 'count': len(rows), 'distinct_test_answers': len(set(answers)) == len(answers),
                     # recorded: pairs no single law reproduces (a pair showing laws together, or something else)
                     'pairs_no_single_law': [i for i, p in enumerate(episode['train'], 1)
                                             if not any(g == p['output'] for a in laws for j, _, g in _train(a, episode) if j == i)]}
    parts += [r['status'] for r in rows] or ['incomplete']
    parts.append('pass' if len(rows) >= 2 and value['laws']['distinct_test_answers'] else 'fail' if rows else 'incomplete')
    if declared and len(rows) < declared:
        parts.append('incomplete')
    value['replay'] = [{'id': a['id'], **_tests_differ(a, episode)} for a in replay]
    parts.append('incomplete' if not replay or any(r['tests_missing'] for r in value['replay'])
                 else 'pass' if all(r['tests_differ'] for r in value['replay']) else 'fail')
    value['status'] = _status(parts)
    return value


def components(grid):
    """The naive parse: same-colour 4-connected components of cells other than the grid's most common colour."""
    ground = Counter(c for row in grid for c in row).most_common(1)[0][0]
    cells = {(r, c) for r, row in enumerate(grid) for c, x in enumerate(row) if x != ground}
    out, seen = [], set()
    for cell in sorted(cells):
        if cell in seen:
            continue
        colour, stack, group = grid[cell[0]][cell[1]], [cell], {cell}
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells and n not in seen and grid[n[0]][n[1]] == colour:
                    seen.add(n); group.add(n); stack.append(n)
        out.append(frozenset(group))
    return out


def _units(labels):
    groups = {}
    for r, row in enumerate(labels):
        for c, x in enumerate(row):
            if x:
                groups.setdefault(x, set()).add((r, c))
    return [frozenset(g) for g in groups.values()]


def unit_record(episode, table):
    """PARSE's unit labels against the naive components, per input grid (S2: recorded, not in the pass rate)."""
    grids = [('train_%d' % i, p['input']) for i, p in enumerate(episode['train'], 1)] + \
            [('test_%d' % k, p['input']) for k, p in enumerate(episode['test'], 1)]
    rows, missing = [], []
    for name, grid in grids:
        labels = (table.get('units') or {}).get(L._key(grid))
        if labels is None or L._shape(labels) != L._shape(grid):
            missing.append(name)
            continue
        units, naive = _units(labels), components(grid)
        rows.append({'grid': name, 'units': len(units), 'naive_components': len(naive),
                     'same_partition': set(units) == set(naive)})
    return {'grids': rows, 'missing': missing, 'complete': not missing,
            'differs_from_naive': any(not r['same_partition'] for r in rows)}


def parse(rule, episode, table, declared=None):
    naive = records(table, 'PARSE', 'naive parse')
    value = {'units': unit_record(episode, table)}
    if not naive:
        return {**value, 'status': 'incomplete'}
    a = naive[0]
    train = _train(a, episode)
    given = [(i, p, g) for i, p, g in train if g is not None]
    t = _tests_differ(a, episode)
    fits = [i for i, p, g in given if g == p['output']]
    differs = [i for i, p, g in given if g != p['output']]
    value.update(id=a['id'], note=a.get('note'), missing_pairs=[i for i, _, g in train if g is None], fitting_pairs=fits,
                 differing_pairs=differs, two_pairs_differ=len(differs) >= 2, **t)
    ok = fits and differs and t['tests_differ']
    value['status'] = 'incomplete' if value['missing_pairs'] or t['tests_missing'] else 'pass' if ok else 'fail'
    return value


def surface_record(episode, table):
    """VARY's per-grid surface record (S7): complete when every input has one; which surfaces vary across the training
    grids (role colours, key box position or shape, orientation, scale). Recorded."""
    names = [('train_%d' % i, p['input']) for i, p in enumerate(episode['train'], 1)] + \
            [('test_%d' % k, p['input']) for k, p in enumerate(episode['test'], 1)]
    found = {n: (table.get('surfaces') or {}).get(L._key(g)) for n, g in names}
    train = [v for n, v in found.items() if n.startswith('train') and v]
    colours = {}
    for v in train:
        for r in v['roles']:
            colours.setdefault(_norm(r['role']), set()).add(r['colour'])
    boxes = {tuple(v['key_box']) for v in train if v.get('key_box')}
    shapes = {(b[2] - b[0], b[3] - b[1]) for b in boxes}
    varied = {'role_colours': any(len(c) > 1 for c in colours.values()), 'key_box': len(boxes) > 1,
              'key_box_shape': len(shapes) > 1,
              'orientation': len({_norm(v.get('orientation')) for v in train if v.get('orientation')}) > 1,
              'scale': len({_norm(v.get('scale')) for v in train if v.get('scale')}) > 1}
    return {'missing': [n for n, v in found.items() if not v], 'complete': all(found.values()),
            'varied': varied, 'varied_count': sum(varied.values()) - (varied['key_box_shape'] and varied['key_box'])}


def vary(rule, episode, table, declared=None):
    dims = records(table, 'VARY')
    value = {'surfaces': surface_record(episode, table)}
    if not dims:
        return {**value, 'status': 'incomplete'}
    rows = [{'id': a['id'], 'dimension': _norm(a.get('kind')), 'note': a.get('note'), **_tests_differ(a, episode)} for a in dims]
    relevant = sorted({r['dimension'] or r['id'] for r in rows if r['tests_differ']})
    need = rule.get('dimensions_min', 2)
    value.update(records=rows, relevant_dimensions=relevant, dimensions_min=need)
    missing = any(r['tests_missing'] for r in rows)
    value['status'] = 'pass' if len(relevant) >= need else 'incomplete' if missing else 'fail'
    return value


def relevance(rule, episode, table, declared=None):
    rows = []
    for kind in ('use everything', 'ignore all'):
        found = records(table, 'RELEVANCE', kind)
        if not found:
            rows.append({'kind': kind, 'status': 'incomplete'})
            continue
        a = found[0]
        train = _train(a, episode)
        t = _tests_differ(a, episode)
        differs = [i for i, p, g in train if g is not None and g != p['output']]
        rows.append({'kind': kind, 'id': a['id'], 'missing_pairs': [i for i, _, g in train if g is None],
                     'differing_pairs': differs, **t,
                     'status': 'incomplete' if (not differs and any(g is None for _, _, g in train)) or t['tests_missing']
                     else 'pass' if differs and t['tests_differ'] else 'fail'})
    return {'records': rows, 'status': _status([r['status'] for r in rows])}


CHECKS = {'eliminate': eliminate, 'recombine': recombine, 'parse': parse, 'vary': vary, 'relevance': relevance}


def alternatives(name, rule, episode, table, declared=None, form=None):
    """A structure line's record check; `form` is the plan's declared form label (RECOMBINE: order-free or in turn)."""
    if rule['rule'] == 'recombine':
        return recombine(rule, episode, table, declared, form=form)
    return CHECKS[rule['rule']](rule, episode, table, declared)


# ------------------------------------------------------------------ SHAPE (computed from the grids alone)


def largest_box(grid):
    """The bounding box (height, width) of the grid's largest 8-connected part other than its most common colour."""
    ground = Counter(c for row in grid for c in row).most_common(1)[0][0]
    cells = {(r, c) for r, row in enumerate(grid) for c, x in enumerate(row) if x != ground}
    best, seen = None, set()
    for cell in sorted(cells):
        if cell in seen:
            continue
        stack, group = [cell], {cell}
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (r + dr, c + dc)
                    if n in cells and n not in seen:
                        seen.add(n); group.add(n); stack.append(n)
        if best is None or len(group) > len(best):
            best = group
    if not best:
        return None
    rows, cols = [r for r, _ in best], [c for _, c in best]
    return (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)


def naive_frames(episode, k):
    """The naive frames for test k (S9, A5): its input's size, every training output size and the largest object's box
    in its input (the most common training output size is among the training output sizes)."""
    test = episode['test'][k - 1]['input']
    frames = {L._shape(test): 'test input size'}
    for i, p in enumerate(episode['train'], 1):
        frames.setdefault(L._shape(p['output']), f'train_{i} output size')
    box = largest_box(test)
    if box:
        frames.setdefault(box, 'largest object box')
    return frames


def frame(episode, rule):
    sizes = [L._shape(p['output']) for p in episode['train']]
    common = Counter(sizes).most_common(1)[0][0] if sizes else None
    tests = []
    for k, p in enumerate(episode['test'], 1):
        frames = naive_frames(episode, k)
        size = L._shape(p['output'])
        tests.append({'test': k, 'size': list(size) if size else None, 'naive': frames.get(size)})
    need = rule.get('sizes_min', 3)
    value = {'training_sizes': [list(s) if s else None for s in sizes], 'distinct_training_sizes': len(set(sizes)),
             'sizes_min': need, 'most_common_training_size': list(common) if common else None, 'tests': tests,
             # S9: the declared quantity against the output size is recorded, outside the pass rate (free text)
             'quantity_check': 'not machine-read'}
    value['status'] = 'pass' if len(set(sizes)) >= need and all(t['naive'] is None for t in tests) else 'fail'
    return value


# ------------------------------------------------------------------ capture (S9: the 80% rule)


def grid_capture(alt, truth, sample):
    """One blind answer against one whole rival grid: eligible when the answer is wrong; with the rival in the answer's
    frame, captured when the answer takes the rival's colour on at least 80% of the cells where the rival differs from the
    truth (and errs on one of them); with the rival in another frame, when the answer has that frame and equals the
    rival on at least 80% of its cells."""
    if L._shape(sample) is None or L._shape(alt) is None or sample == truth or alt == truth:
        return None
    if L._shape(alt) == L._shape(truth):
        cells = [(r, c) for r, row in enumerate(alt) for c, x in enumerate(row) if x != truth[r][c]]
        if L._shape(sample) != L._shape(truth):
            return {'eligible': True, 'captured': False, 'share': 0.0}
        eligible = any(sample[r][c] != truth[r][c] for r, c in cells)
        share = sum(sample[r][c] == alt[r][c] for r, c in cells) / len(cells)
    else:
        eligible = True
        if L._shape(sample) != L._shape(alt):
            return {'eligible': True, 'captured': False, 'share': 0.0}
        total = len(alt) * len(alt[0])
        share = sum(a == b for ra, rb in zip(alt, sample) for a, b in zip(ra, rb)) / total
    return {'eligible': eligible, 'captured': eligible and share >= L.CAPTURE, 'share': round(share, 3)}


ROUTED = ('rival', 'single law', 'replay', 'naive parse', 'use everything', 'ignore all')


def capturable(name, table):
    """The records a blind answer can land on (the two order records are the truth by design and are left out)."""
    rows = records(table, name)
    return [a for a in rows if _norm(a.get('kind')) != 'order'
            and not (_norm(a.get('kind')) not in KINDS and re.fullmatch(KINDS['order'], a['id'].strip().lower()))]


def results(name, episode, table, answers):
    """One blind answer against every structure record of the idea's structure line, per test: per record its id, kind,
    whether the answer errs, the share and capture at 80%, and whether it equals the record's grid exactly. SHAPE: a wrong
    answer whose frame is a naive frame is captured (S9), and that is also its exact-alternative test."""
    out = []
    for k, pair in enumerate(episode['test'], 1):
        if k > len(answers):
            continue
        sample = answers[k - 1]
        if name == 'SHAPE':
            size = L._shape(sample)
            frames = naive_frames(episode, k)
            wrong = sample != pair['output']
            hit = wrong and size is not None and size != L._shape(pair['output']) and size in frames
            out.append({'part': 'structure', 'id': 'frame', 'kind': frames.get(size) if hit else 'naive frame', 'test': k,
                        'eligible': wrong, 'captured': hit, 'share': None, 'equals_alternative': hit})
            continue
        for a in capturable(name, table):
            alt = a['test'].get(L._key(pair['input']))
            if alt is None:
                continue
            row = grid_capture(alt, pair['output'], sample) or {'eligible': False, 'captured': False, 'share': None}
            out.append({'part': 'structure', 'id': a['id'], 'kind': a.get('kind'), 'test': k, **row,
                        'equals_alternative': sample == alt and alt != pair['output']})
    return out


# ------------------------------------------------------------------ the default-reading call (D)


def _shortcut_answer(points, truth):
    """The answer with every declared point of one test applied (SETTLE's default reading)."""
    grid = [row[:] for row in truth]
    for p in points:
        for r, c, x in L.point_cells(p, truth):
            grid[r][c] = x
    return grid


def default_requests(record, check=None):
    """The default-reading calls a verified task gets (STRUCTURE_SPEC D, S2, S9): [{key, subset, train (pair numbers),
    expected (the rival's test answers), basis}]. SETTLE: without its settling pairs, against the shortcut answer;
    ELIMINATE: one per rival, without that rival's killer pair, against the rival's test answers; PARSE: on the pairs the
    naive parse fits, against its test answers. At most three per task; none without a checked record or with no pair
    left."""
    study = record.get('study') or {}
    episode = record.get('complete_task')
    if not episode:
        return []
    check = check if check is not None else L.episode_check(record, episode)
    table = L.tables(record)
    out, n = [], len(episode['train'])
    keep = lambda drop: [i for i in range(1, n + 1) if i not in drop]
    line, structure = L.line_of(study), L.structure_of(study)
    if (L.spec(line) or {}).get('default_reading') == 'settling':
        pairs = ((check.get('line') or {}).get('pairs') or {})
        settling = {p['pair'] for p in pairs.get('differing_pairs') or []}
        if pairs.get('status') == 'pass' and settling and keep(settling) and L.fence_of(study) != 'in_pair_minority':
            expected = [_shortcut_answer([p for p in table['points'] or [] if p['test'] == k], t['output'])
                        for k, t in enumerate(episode['test'], 1)]
            out.append({'key': 'settle', 'basis': 'SETTLE', 'train': keep(settling), 'expected': expected})
    rule = (L.spec(structure) or {}).get('default_reading')
    value = (check.get('structure') or {}).get('alternatives') or {}
    if rule == 'killers':
        for r in value.get('records') or []:
            a = (table['alternatives'] or {}).get(r['id'])
            if r.get('killer') and a and keep({r['killer']}):
                expected = [a['test'].get(L._key(t['input'])) for t in episode['test']]
                if all(g is not None for g in expected):
                    out.append({'key': 'rival-' + re.sub(r'[^a-z0-9]', '', r['id'].lower()), 'basis': 'ELIMINATE',
                                'rival': r['id'], 'train': keep({r['killer']}), 'expected': expected})
    elif rule == 'fitting_pairs' and value.get('fitting_pairs') and value.get('id'):
        a = (table['alternatives'] or {}).get(value['id'])
        expected = [a['test'].get(L._key(t['input'])) for t in episode['test']] if a else []
        if expected and all(g is not None for g in expected):
            out.append({'key': 'naive', 'basis': 'PARSE', 'rival': value['id'], 'train': list(value['fitting_pairs']),
                        'expected': expected})
    return out[:3]


def default_task(record, request):
    """The public task a default-reading call sees: the chosen training pairs and the test inputs (no plan, no hints)."""
    episode = record['complete_task']
    return {'train': [{'input': p['input'], 'output': p['output']} for i, p in enumerate(episode['train'], 1)
                      if i in request['train']],
            'test': [{'input': p['input']} for p in episode['test']]}


def default_result(record, request, response):
    """A default-reading answer against the declared rival (the 80% rule, S9) and the truth, per test; captured on test 1
    is the summary."""
    episode = record['complete_task']
    answers = (response or {}).get('predicted_test_outputs') or []
    rows = []
    for k, (pair, alt) in enumerate(zip(episode['test'], request['expected']), 1):
        sample = answers[k - 1] if k <= len(answers) else None
        cap = grid_capture(alt, pair['output'], sample) if sample is not None else None
        rows.append({'test': k, 'equals_rival': sample == alt, 'equals_truth': sample == pair['output'],
                     'captured': bool(cap and cap['captured']) or (sample == alt and alt != pair['output']),
                     'share': cap['share'] if cap else None})
    first = rows[0] if rows else {}
    return {'key': request['key'], 'basis': request['basis'], 'rival': request.get('rival'), 'train': request['train'],
            'tests': rows, 'captured': bool(first.get('captured')), 'equals_rival': bool(first.get('equals_rival')),
            'equals_truth': bool(first.get('equals_truth'))}
