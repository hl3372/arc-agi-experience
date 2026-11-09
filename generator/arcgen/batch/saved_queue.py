"""Saved-queue windows: develop a frozen list of already-selected ideas, with no idea or selector calls.

A saved queue file (under `arcgen/batch/allocations/queues/`) holds selector-shaped rows (kernel, editorial brief,
novelty, selector assessment, study) and the raw source rows they came from, plus provenance. The allocation table
names it with its sha256 under `saved_windows`; `allocation.design` turns each entry into a window of kind
`saved_queue` plus the usual development groups. At preparation the window materializes exactly what a supply
window would have frozen (eligible queue, planner pairing, source rows), so development groups are unchanged.
Planner assignments carried over from the source run are removed and re-drawn under the current profile.
"""
from copy import deepcopy
from hashlib import sha256
from .common import ROOT, IntegrityError, read, frozen, ref
from . import pairing

PLANNER_FIELDS = ('planner', 'planner_binding', 'planner_paired', 'planner_pair_id', 'planner_version',
                  'planner_policy', 'selection_dispatch_ordinal')


def load(window):
    path = ROOT / window['saved_queue']
    if sha256(path.read_bytes()).hexdigest() != window['saved_queue_sha256']:
        raise IntegrityError('saved queue changed after it was bound: ' + window['saved_queue'])
    value = read(path)
    rows = value['rows']
    ids = [r['source_candidate_id'] for r in rows]
    if not rows or len(ids) != len(set(ids)) or set(ids) != set(value['source_rows']) or len(rows) != window['saved_rows']:
        raise IntegrityError('saved queue rows and source rows disagree')
    # one hop: every idea was generated and selected in the queue's source run, where commit finds its calls
    if any('saved_queue_origin' in r['study'] or r['study'].get('window') != r['study'].get('original_selector_window') for r in rows):
        raise IntegrityError('saved queue holds ideas carried from an earlier run; carry from the run that generated them')
    return value


def fresh_row(row, ordinal):
    out = deepcopy(row)
    out.pop('pair_lineage_id', None)
    for key in PLANNER_FIELDS:
        out['study'].pop(key, None)
    out['study']['dispatch_ordinal'] = ordinal
    out['study']['saved_queue_origin'] = {'window': row['study'].get('window'),
                                          'original_selector_window': row['study'].get('original_selector_window')}
    out['source_development_priority'] = ordinal
    return out


def materialize(window, directory):
    value = load(window)
    rows = [fresh_row(r, i) for i, r in enumerate(value['rows'], 1)]
    ids = [r['source_candidate_id'] for r in rows]
    allocation = {'rows': rows, 'retained_ids': ids, 'natural_ids': ids, 'targets': [],
                  'real_ranks': {k: i for i, k in enumerate(ids, 1)}, 'redundancy_deferrals': [], 'uncertain_ids': [],
                  'ineligible_ids': [], 'capacity_deferrals': [], 'family_quota': None, 'rank_cutoff': None,
                  'maximum_developments': len(rows), 'no_replacement_generation': True,
                  'policy': 'saved_queue:' + value['provenance']['id']}
    allocation = pairing.expand(window, window['pairing_index'], allocation, seed=window['seed'])
    frozen(directory / 'planner_pairing.private.json', allocation['planner_pairing'])
    frozen(directory / 'eligible_queue.private.json', allocation)
    frozen(directory / 'source_rows.private.json', deepcopy(value['source_rows']))
    frozen(directory / 'saved_queue_receipt.private.json', {'queue': window['saved_queue'],
           'sha256': window['saved_queue_sha256'], 'provenance': value['provenance'], 'rows': len(rows)})
    out = {**allocation, 'rows': [], 'source_queue': ref(directory / 'eligible_queue.private.json'),
           'supply_only': True, 'maximum_developments': 0}
    frozen(directory / 'allocation.private.json', out)
    return out
