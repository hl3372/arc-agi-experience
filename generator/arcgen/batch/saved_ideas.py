"""Saved-ideas windows: send ideas a stopped run generated but never selected through this batch's selector.

A saved ideas file (under `arcgen/batch/allocations/queues/`) holds one source window's rows exactly as the source run
built them (supply.source_rows from its frozen idea-loop commits), plus provenance. The allocation table names it with
its sha256 under `saved_idea_windows`; `allocation.design` turns each entry into a window of kind `saved_ideas` with no
idea calls, and preparation runs the ordinary selector path on the saved rows (pipeline._select). Unlike a saved
queue, nothing here was selected, so every idea meets this batch's bank, memory and selector. The idea calls stay in
the source run; commit finds them there through `provenance.source_run`.
"""
from hashlib import sha256
from .common import ROOT, IntegrityError, read


def load(window):
    path = ROOT / window['saved_ideas']
    if sha256(path.read_bytes()).hexdigest() != window['saved_ideas_sha256']:
        raise IntegrityError('saved ideas changed after they were bound: ' + window['saved_ideas'])
    value = read(path)
    rows = value['source_rows']
    if (not rows or sorted(rows) != sorted(window['saved_kernel_ids'])
            or any(r['candidate_id'] != k or r['kernel']['candidate_id'] != k for k, r in rows.items())
            or not value['provenance'].get('source_run') or not value['provenance'].get('source_window')):
        raise IntegrityError('saved ideas and their window disagree')
    # one hop: the ideas were generated in the named source window, where commit finds their idea calls
    if any(r['study']['window'] != value['provenance']['source_window'] for r in rows.values()):
        raise IntegrityError('saved ideas were generated elsewhere; carry from the run that generated them')
    return value
