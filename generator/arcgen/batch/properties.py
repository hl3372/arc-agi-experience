"""Achieved properties of verified tasks, read from the grids alone (2026-10-07).

The property slots and arms say what an idea call was asked for; these checks say what the finished task shows, for
the properties a format-level test can see. Each check is a heuristic, kept deliberately simple:
- PACKED: no colour dominates the inputs as a ground (each input's own most common colour covers under 50% of its
  cells on average; v1-b11's median is 45%);
- DIGEST: every output is small (at most 50 cells and at most a quarter of its input);
- EXTENT: output sizes differ across the training pairs, and some output differs in size from its input;
- MONO: every input holds at most one colour besides its own most common colour;
- STAGES and the CHAINED slot: each training output is exactly the next training input;
- CONTRAST and the MINIMAL slot: two same-size training inputs differ in under 10% of their cells;
- SPREAD and the DISTRIBUTED slot: six or more training pairs;
- ONE (2026-10-23, slots.py): exactly one training pair.
The other values and arms (ACROSS, VACANCY, IGNORE, PAYLOAD, LIGHT, CONSTRUCT, ...) need reading, not grids, and are
only counted: a size-spread proxy for ACROSS passed 44 of v1-b11's 153 tasks, none of which shows different kinds of
picture.
Since 2026-10-11 (v1-b14 audit) PACKED and MONO read each grid's
own ground: against one task-wide ground, all 18 MONO "failures" passed per grid and 41 PACKED passes had a 30%+ ground
that changed colour across pairs. DIGEST takes up to 50 cells (DIGEST-5 bans swatch charts and guessable single cells,
and 16 of its 35 failures were 26-49-cell recovered objects).
Since 2026-10-12 :
- SCENE: the output is a different scene: every test output has more than 30 cells, the training outputs are not all
  identical, and no pair's output is a sub-grid of its input or an edit of it (same size with half its cells unchanged);
- scale (measured, not a pass check): on the largest test pair, the computed cells (changed cells for a same-size
  output, else the output's cells off its own ground), the 8-connected regions they form, cells per region, the test's
  size over the training pairs' (escalation) and computed cells per object in the test input (amplification), and the
  highest rung (science.rungs) the grids reach.
Since 2026-10-13: a resized output's computed cells are its cells off the
test input's ground, read on the input's border (until then, off the output's own most common colour, which read a solid FIT assembly of 541 cells
as 83); and repeated training outputs are counted (two training outputs equal under a turn or reflection and a colour
renaming, each with 30 or more figure cells): a measure for readers, not a gate, since restoration designs repeat a
picture under different erasures on purpose.
"""
from collections import Counter
from statistics import median

GRID_CHECKS = ('PACKED', 'DIGEST', 'EXTENT', 'MONO', 'STAGES', 'CONTRAST', 'SPREAD', 'SCENE', 'ONE')
ALIASES = {'CHAINED': 'STAGES', 'MINIMAL': 'CONTRAST', 'DISTRIBUTED': 'SPREAD'}  # slot versions of the property arms
FORMAT_CHECKS = GRID_CHECKS + tuple(ALIASES)


def _cells(grid):
    return [c for row in grid for c in row]


def _background(task):
    return Counter(c for pair in task['train'] + task['test'] for c in _cells(pair['input'])).most_common(1)[0][0]


def _ground(grid):
    """The grid's own most common colour (ties: the lowest colour)."""
    counts = Counter(_cells(grid))
    return min(counts, key=lambda c: (-counts[c], c))


def _border_ground(grid):
    """The most common colour on the grid's outer ring (ties: the lowest): the ground, even where pieces outnumber it."""
    ring = list(grid[0]) + list(grid[-1]) + [row[0] for row in grid[1:-1]] + [row[-1] for row in grid[1:-1]]
    counts = Counter(ring)
    return min(counts, key=lambda c: (-counts[c], c))


def _minimal_pair(train):
    ins = [p['input'] for p in train]
    for a in range(len(ins)):
        for b in range(a + 1, len(ins)):
            if (len(ins[a]), len(ins[a][0])) == (len(ins[b]), len(ins[b][0])):
                cells = _cells(ins[a]); other = _cells(ins[b])
                if 0 < sum(x != y for x, y in zip(cells, other)) < 0.1 * len(cells):
                    return True
    return False


def _sub_grid(small, big):
    h, w = len(small), len(small[0])
    return any(all(big[r + i][c:c + w] == small[i] for i in range(h))
               for r in range(len(big) - h + 1) for c in range(len(big[0]) - w + 1))


def _edit_or_crop(pair):
    a, b = pair['input'], pair['output']
    if (len(a), len(a[0])) == (len(b), len(b[0])):
        cells = _cells(b)
        return sum(x == y for x, y in zip(_cells(a), cells)) * 2 >= len(cells)
    return len(b) <= len(a) and len(b[0]) <= len(a[0]) and _sub_grid(b, a)


def scene(task):
    """The output is a different scene (see the module note)."""
    pairs = [p for p in task['train'] + task['test'] if 'output' in p]
    outs = [p['output'] for p in task['train']]
    return (all(len(_cells(p['output'])) > 30 for p in task['test'] if 'output' in p)
            and len({str(o) for o in outs}) > 1 and not any(_edit_or_crop(p) for p in pairs))


def _components(cells):
    """8-connected components of a set of (row, column) cells."""
    seen, out = set(), []
    for start in cells:
        if start in seen:
            continue
        stack, size = [start], 0
        seen.add(start)
        while stack:
            r, c = stack.pop(); size += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (r + dr, c + dc)
                    if n in cells and n not in seen:
                        seen.add(n); stack.append(n)
        out.append(size)
    return out


def scale(task, rungs=None):
    """Scale measures of the largest test pair (see the module note), against `rungs` (default science.rungs)."""
    area = lambda g: len(g) * len(g[0])
    test = max((p for p in task['test'] if 'output' in p), key=lambda p: area(p['output']))
    a, b = test['input'], test['output']
    if (len(a), len(a[0])) == (len(b), len(b[0])):
        computed = {(r, c) for r in range(len(b)) for c in range(len(b[0])) if a[r][c] != b[r][c]}
    else:  # the output's cells off the input's ground, read on its border (all of them when the output has none of it)
        g = _border_ground(a)
        computed = {(r, c) for r in range(len(b)) for c in range(len(b[0])) if b[r][c] != g}
    regions = _components(computed)
    g = _ground(a)
    objects = len(_components({(r, c) for r in range(len(a)) for c in range(len(a[0])) if a[r][c] != g}))
    train = task['train']
    escalation = max(area(a) / (sum(area(p['input']) for p in train) / len(train)),
                     area(b) / (sum(area(p['output']) for p in train) / len(train)))
    value = {'test_output_cells': area(b), 'computed_cells': len(computed), 'regions': len(regions),
             'cells_per_region': round(len(computed) / len(regions), 1) if regions else 0,
             'largest_region': max(regions, default=0), 'escalation': round(escalation, 2),
             'amplification': round(len(computed) / objects, 2) if objects else None}
    value['rung'] = grid_rung(value, rungs)
    return value


def grid_rung(value, rungs=None):
    """The highest rung whose floors the finished grids meet, or None."""
    if rungs is None:
        from .config import SCIENCE
        rungs = SCIENCE.get('rungs') or {}
    best = None
    for name in sorted(rungs):
        f = rungs[name]
        if (value['test_output_cells'] >= f['output_cells'] and value['computed_cells'] >= f['computed_cells']
                and (not f['escalation'] or value['escalation'] >= f['escalation'])):
            best = name
    return best


def checks(task):
    """{check: bool} for one complete task ({'train': [...], 'test': [...]} with test outputs)."""
    pairs = task['train'] + task['test']
    inputs = [p['input'] for p in pairs]
    outputs = [p['output'] for p in pairs if 'output' in p]
    shape = lambda g: (len(g), len(g[0]))
    grounds = [_ground(g) for g in inputs]  # each input's own ground: ground and figure colours may change across pairs
    bg_share = sum(_cells(g).count(bg) / len(_cells(g)) for g, bg in zip(inputs, grounds)) / len(inputs)
    train = task['train']
    value = {
        'PACKED': bg_share < 0.5,
        'DIGEST': all(len(_cells(p['output'])) <= 50 and len(_cells(p['output'])) * 4 <= len(_cells(p['input']))
                      for p in pairs if 'output' in p),
        'EXTENT': len({shape(p['output']) for p in train}) > 1 and any(shape(p['output']) != shape(p['input']) for p in pairs if 'output' in p),
        'MONO': all(len(set(_cells(g)) - {bg}) <= 1 for g, bg in zip(inputs, grounds)),
        'STAGES': len(train) >= 2 and all(train[i]['output'] == train[i + 1]['input'] for i in range(len(train) - 1)),
        'CONTRAST': _minimal_pair(train),
        'SPREAD': len(train) >= 6,
        'SCENE': scene(task),
        'ONE': len(train) == 1,
        'answer_cells': median(len(_cells(g)) for g in outputs) if outputs else 0,
        'same_size': all(shape(p['output']) == shape(p['input']) for p in pairs if 'output' in p),
    }
    return {**value, **{alias: value[name] for alias, name in ALIASES.items()}}


def _d4(grid):
    g = [tuple(r) for r in grid]; out = []
    for _ in range(4):
        out += [g, [tuple(reversed(r)) for r in g]]
        g = [tuple(r) for r in zip(*g[::-1])]
    return out


def _renamed(a, b):
    """True when same-shape grids a and b differ only by one colour bijection."""
    fwd, back = {}, {}
    return all(fwd.setdefault(x, y) == y and back.setdefault(y, x) == x for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def repeated_outputs(task):
    """Two training outputs equal under a turn or reflection and a colour renaming, each with 30 or more cells off its own
    ground and two or more colours."""
    outs = [p['output'] for p in task['train']
            if len(set(_cells(p['output']))) > 1 and sum(c != _ground(p['output']) for c in _cells(p['output'])) >= 30]
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            b = [tuple(r) for r in outs[j]]
            if any(len(t) == len(b) and len(t[0]) == len(b[0]) and _renamed(t, b) for t in _d4(outs[i])):
                return True
    return False


def report(records, rungs=None):
    """Per drawn modifier and per property arm: verified tasks, and how many pass the matching format check. `rungs` are
    the run's own floors: a summary runs under whatever profile is loaded, which need not be the run's."""
    rows, overall = {}, []
    for r in records:
        task = r.get('complete_task')
        if not task:
            continue
        c = checks(task); overall.append(c)
        drawn = list(r['study'].get('idea_modifiers') or []) + [r['study'].get('primary_arm')]
        for name in drawn:
            row = rows.setdefault(name, {'verified': 0, 'pass': 0})
            row['verified'] += 1
            if name in FORMAT_CHECKS:
                row['pass'] += int(c[name])
    for name, row in rows.items():
        if name not in FORMAT_CHECKS:
            row.pop('pass')
    by_rung = {}  # targeted family (or none) and planned rung: does the finished task reach it?
    for r in records:
        task, rung = r.get('complete_task'), r['study'].get('rung')
        if not task or not rung:
            continue
        s = scale(task, rungs)
        row = by_rung.setdefault(f"{r['study'].get('family') or 'general'} {rung}", {'verified': 0, 'reached': 0, 'measures': []})
        row['verified'] += 1
        row['reached'] += int(s['rung'] is not None and s['rung'] >= rung)
        row['measures'].append(s)
    for row in by_rung.values():
        ms = row.pop('measures')
        row.update({k: median(m[k] for m in ms) for k in ('test_output_cells', 'computed_cells', 'regions', 'cells_per_region')})
    base = {k: sum(c[k] for c in overall) for k in GRID_CHECKS}
    return {'by_property': dict(sorted(rows.items(), key=lambda kv: str(kv[0]))), 'all_verified': len(overall),
            'format_checks_all_verified': base,
            'by_rung': dict(sorted(by_rung.items())),
            'median_answer_cells': median(c['answer_cells'] for c in overall) if overall else None,
            'repeated_training_outputs': sum(repeated_outputs(r['complete_task']) for r in records if r.get('complete_task')),
            'same_size_outputs': sum(c['same_size'] for c in overall)}


def lines(value):
    """Markdown lines for SUMMARY.md."""
    out = [f"- Median answer size: {value['median_answer_cells']} cells; same-size outputs in "
           f"{value['same_size_outputs']} of {value['all_verified']} tasks; two training outputs repeat (turned or "
           f"recoloured) in {value.get('repeated_training_outputs', 0)}"]
    parts = []
    for name, row in value['by_property'].items():
        if name is None:
            continue
        parts.append(f"{name} {row['verified']}" + (f" ({row['pass']} pass the grid check)" if 'pass' in row else ''))
    out.append('- Verified tasks by drawn property or arm: ' + '; '.join(parts))
    out.append('- Grid checks over all verified tasks: ' + ', '.join(f'{k} {v}' for k, v in value['format_checks_all_verified'].items()))
    if value.get('by_rung'):
        out.append('- Scale rungs (family and planned rung: verified, reaching the rung on the grids; medians of test output '
                   'cells, computed cells, changed regions, cells per region): ' + '; '.join(
                       f"{k} {v['verified']}, {v['reached']} reach it; {v['test_output_cells']:g}, {v['computed_cells']:g}, "
                       f"{v['regions']:g}, {v['cells_per_region']:g}" for k, v in value['by_rung'].items()))
    return out
