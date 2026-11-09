"""Frozen multi-planner assignment from the batch config.

A configured fraction of each window's eligible ideas is planned by every configured planner; the
rest by the default planner. The assignment is a seeded hash over the frozen eligible rows, computed
once when a window's queue is frozen and before any planning call. The default planner's version
keeps the native task identity; each other planner's version is a separate development of the
same kernel, placed directly after it in the queue.
"""
from copy import deepcopy
from .common import IntegrityError, digest
from math import ceil, floor
from .routing import PLANNERS
from .config import SCIENCE

_PA = SCIENCE['planner_assignment']
_LETTERS = {'astra': 'A', 'sol': 'S', 'sol_max': 'M'}
ORDER = tuple((p, _LETTERS.get(p, p[:1].upper())) for p in (_PA['multi_planners'] or [_PA['default_planner']]))
DEFAULT = _PA['default_planner']
FRACTION = _PA['multi_planner_fraction'] if len(ORDER) > 1 else 0.0
POLICY = {'id': _PA['policy_id'], 'planners': PLANNERS, 'order': [p for p, _ in ORDER], 'unpaired': DEFAULT,
          'fraction': FRACTION, 'odd_rounding': 'floor on even window index, ceil on odd',
          'queue': 'default planner version then its other planner versions'}


def paired_ids(window_id, window_index, source_ids, seed):
    ids = list(source_ids)
    if len(ids) != len(set(ids)): raise IntegrityError('eligible queue repeats a kernel before pairing')
    count = floor(len(ids) * FRACTION) if window_index % 2 == 0 else ceil(len(ids) * FRACTION)
    ranked = sorted(ids, key=lambda sid: digest({'policy': POLICY['id'], 'seed': seed, 'window': window_id, 'source': sid}))
    return set(ranked[:count])


def _mark(row, planner, *, paired, pair_id, version):
    row['study'].update(planner=planner, planner_binding=dict(PLANNERS[planner]), planner_paired=paired,
                        planner_pair_id=pair_id, planner_version=version, planner_policy=POLICY['id'])
    return row


def expand(window, window_index, allocation, *, seed):
    """Return the eligible queue with Sol xhigh/max twins inserted directly after their Astra version."""
    rows = allocation['rows']
    if 'planner_pairing' in allocation or any('planner' in r.get('study', {}) for r in rows):
        raise IntegrityError('queue already carries planner assignments')
    chosen = paired_ids(window['id'], window_index, [r['source_candidate_id'] for r in rows], seed)
    out = []
    for row in rows:
        sid = row['source_candidate_id']
        paired = sid in chosen
        out.append(_mark(deepcopy(row), DEFAULT, paired=paired, pair_id=sid if paired else None, version=ORDER[0][1]))
        if paired:
            for planner, version in ORDER[1:]:
                twin = _mark(deepcopy(row), planner, paired=True, pair_id=sid, version=version)
                twin['pair_lineage_id'] = row['lineage_id']
                twin['lineage_id'] = digest({'lineage': row['lineage_id'], 'planner': planner})
                out.append(twin)
    for ordinal, row in enumerate(out, 1):
        row['study']['selection_dispatch_ordinal'] = row['study'].get('dispatch_ordinal')
        row['study']['dispatch_ordinal'] = ordinal
    if len(out) != len(rows) + (len(ORDER) - 1) * len(chosen): raise IntegrityError('planner expansion count changed')
    receipt = {'policy': POLICY, 'window': window['id'], 'window_index': window_index, 'seed': seed,
               'eligible': len(rows), 'paired': sorted(chosen), 'developments': len(out)}
    return {**deepcopy(allocation), 'rows': out,
            'retained_ids': [r['source_candidate_id'] for r in out], 'planner_pairing': receipt}


def task_id(window_id, row):
    """Astra versions keep the native identity; each Sol twin gets a distinct, stable identity."""
    base = {'window': window_id, 'source': row['source_candidate_id']}
    planner = row['study'].get('planner')
    if planner == DEFAULT:
        return 'd' + digest(base)[:19]
    if planner in PLANNERS:
        return 'd' + digest({**base, 'planner': planner})[:19]
    raise IntegrityError('row lacks a frozen planner assignment')
