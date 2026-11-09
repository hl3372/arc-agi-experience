"""Selector memory per reference wave: what this batch has already selected (in flight) or seen fail.

Wave A freezes an empty snapshot. A later wave freezes, from this run's own windows of earlier waves, every selected
idea that is not yet a bank task: still in development or realized ones become in-flight reservations, held or failed
ones become failure memory. Each context's selector memory for the wave is the prepared memory plus these rows, frozen
under design/waves/<wave>/ so a resumed run sees the same memory. No other run is read.
"""
from copy import deepcopy
from .common import read, ref, sha, frozen, verify_refs


def classify(record, terminal, held_ids=()):
    if record and record.get('candidate_id') in held_ids: return 'failed_or_held'
    if record is None: return 'selected_queued'
    if record.get('boardless_disposition') == 'verified_candidate': return 'evidence_clear_unadmitted'
    if terminal or record.get('status') in ('planning_rejected', 'realization_failed', 'technical_fail', 'blind_fail'):
        return 'failed_or_held'
    if record.get('realization'): return 'realized'
    if record.get('concrete_plan'): return 'planning'
    return 'selected_queued'


_ORDER = {'failed_or_held': 0, 'selected_queued': 1, 'planning': 2, 'realized': 3, 'evidence_clear_unadmitted': 4}


def _rank(item):
    return _ORDER[classify(item[0], item[1])]


def freeze_snapshot(root, wave, inputs, *, current_windows=()):
    path = root / 'design/recent' / f'{wave}.private.json'
    if path.exists():
        saved = read(path)
        # Run states progress after the cutoff; the snapshot stores the projected rows and observed hashes.
        verify_refs(saved['immutable_sources']); return saved
    metadata = read(inputs['author_metadata'])['metadata']
    accepted = sorted(metadata, key=lambda k: metadata[k]['admission_ordinal'], reverse=True)[:300]
    observations, records = [], {}
    if current_windows:
        for file in sorted((root / 'accounts').glob('*/*/downstream/state.json')):
            state = read(file)
            observations.append({'observed_path': str(file), 'observed_sha256': sha(file)})
            terminal = (file.parent / 'window_terminal.private.json').exists()
            for tid, r in state.get('semantic_records', {}).items():
                sid = r.get('source_candidate_id')
                if not sid: continue
                # Planner versions share one kernel: keep the most advanced record so a failed twin never hides a verified one.
                candidate = ({**r, 'candidate_id': tid}, terminal)
                if sid not in records or _rank(candidate) > _rank(records[sid]): records[sid] = candidate
    rows, sources, ordinal = {}, [], 0
    for directory in [root / 'accounts' / w['account'] / w['id'] for w in current_windows]:
        queue = directory / 'eligible_queue.private.json'
        if not queue.exists(): queue = directory / 'allocation.private.json'
        if not queue.exists(): continue
        allocation = read(queue); sources.append(ref(queue))
        for row in allocation.get('rows', []):
            sid = row['source_candidate_id']; ordinal += 1
            record, terminal = records.get(sid, (None, False))
            status = classify(record, terminal); kernel = row['kernel']; study = row.get('study', {})
            rows[sid] = {'provisional_id': 'pending:' + sid, 'status': status,
                         'concept_summary': kernel['central_relation'], 'rule_summary': kernel['central_relation'],
                         'private': {'source_candidate_id': sid, 'recency_ordinal': ordinal, 'source': ref(queue),
                                     'source_labels': {'arm': study.get('primary_arm'), 'invitation': study.get('invitation'),
                                                       'evidence': ref(queue)},
                                     'stage_observation': record.get('status') if record else None,
                                     'failure_reason': '; '.join(record.get('boardless_reasons', [])) if record else None}}
    provisional = [r for r in rows.values() if r['status'] != 'failed_or_held']
    failures = [r for r in rows.values() if r['status'] == 'failed_or_held']
    # A provisional idea never masquerades as a bank entry: its text is the saved kernel only, never a plan,
    # private answer or evaluator response.
    return frozen(path, {'wave': wave, 'accepted_ids': accepted, 'provisional': provisional, 'failure_memory': failures,
        'immutable_sources': sources + [ref(inputs['author_bank']), ref(inputs['author_metadata'])],
        'state_cutoff_observations': observations, 'admitted_kernel_ids': [],
        'history_roots': [str(root)] if current_windows else [],
        'ordering': 'frozen window order of this run', 'new_text_calls': 0})


def for_wave(root, wave, inputs):
    """The wave's inputs: each context's selector memory plus the snapshot's in-flight and failed rows."""
    path = root / 'design/waves' / wave / 'inputs.private.json'
    if path.exists():
        value = read(path); verify_refs(value); return value
    recent = read(root / 'design/recent' / f'{wave}.private.json')
    value = deepcopy(inputs)
    for context, row in value['contexts'].items():
        memory = read(row['selector_memory']); payload = memory['payload']
        existing = {r['reservation_id'] for r in payload['in_flight_reservations']}
        for r in recent['provisional']:
            rid = 'recent:' + r['provisional_id']
            if rid not in existing:
                payload['in_flight_reservations'].append({'reservation_id': rid, 'mechanism_summary': r['concept_summary']}); existing.add(rid)
        for r in recent['failure_memory']:
            payload['rejected_memory_digest']['failure_mode'].append({'mechanism_summary': r['concept_summary'],
                'reason': r['private'].get('failure_reason') or 'Selected concept was held or failed at the frozen cutoff; this implementation is not positive inspiration. Do not infer a permanent ban on its territory.'})
        dest = root / 'design/waves' / wave / context / 'selector_memory.private.json'; frozen(dest, memory)
        row['selector_memory'] = str(dest); value['files'].append(ref(dest))
    value['recent_snapshot'] = ref(root / 'design/recent' / f'{wave}.private.json')
    return frozen(path, value)
