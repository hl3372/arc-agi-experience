"""Optional editorial bypass around the preserved native candidate evidence DAG."""
from copy import deepcopy
import time
from arcgen.lib.e18_candidate_runtime import run_slate
from arcgen.lib.e18_artifacts import (boardless_disposition, materialize_terminal_artifacts,
                                    verify_terminal_artifact_bundle)
from arcgen.lib.factory_evidence import apply_final_evidence, recover_terminal_evidence_metadata
from arcgen.lib.factory_review_policy import evidence_hash
from arcgen.lib.factory import call_accounting
from arcgen.astra_broad_arm_program_20260908 import downstream as native, reviews
from .common import IntegrityError, digest, read, frozen, atomic
from arcgen.astra_targeted_followup_20260911.admission import run as run_with_board
from .admission import prepare

POLICY = {'id': 'astra-editorial-bypass-v1', 'editorial_mode': 'bypass',
          'evidence_policy': 'grounded-difficulty-retention-v1',
          'quality_grade': 'unassigned', 'automatic_commit': False}


def _decision(record):
    return {**boardless_disposition(record), 'evidence_hash': evidence_hash(record)}


def apply_bypass(runtime):
    records = runtime.state['semantic_records']
    if (any(c.get('purpose') == 'editorial_board' for c in runtime.state['call_ledger'].values())
            or any(r.get('q2_review_binding') or r.get('editorial_judgments')
                   or r.get('intrinsic_quality') in ('core_quality', 'reserve_quality', 'rejected_quality')
                   for r in records.values())):
        raise IntegrityError('boardless window contains editorial calls or judgments')
    decisions = {tid: _decision(r) for tid, r in sorted(records.items())}
    path = runtime.run_dir / 'editorial_bypass.private.json'
    if path.exists():
        value = read(path)
        if value['policy'] != POLICY or value['decisions'] != decisions:
            raise IntegrityError('frozen boardless evidence changed')
    else:
        value = frozen(path, {'policy': POLICY, 'decisions': decisions,
                              'completed_at_epoch': time.time()})
    # Reapply a durable local decision after interruption between this file and
    # state.save; never rerun or reinterpret a completed evidence response.
    for tid, decision in decisions.items():
        record = records[tid]
        usable = decision['boardless_disposition'] == 'verified_candidate'
        record.update(editorial_mode='bypass', boardless_disposition=decision['boardless_disposition'],
                      boardless_reasons=deepcopy(decision['reasons']),
                      pipeline_usable_before_union_audit=usable,
                      curriculum_status='boardless_verified_candidate' if usable else 'boardless_not_verified',
                      editorial_bypass_binding={'policy': POLICY, 'decision_hash': digest(decision),
                          'receipt_hash': digest(value), 'completed_at_epoch': value['completed_at_epoch']})
    runtime.save()
    return value


def verify_bypass(runtime):
    value = read(runtime.run_dir / 'editorial_bypass.private.json')
    records = runtime.state['semantic_records']
    if value['policy'] != POLICY or set(value['decisions']) != set(records):
        raise IntegrityError('boardless terminal membership or policy changed')
    if any(c.get('purpose') == 'editorial_board' for c in runtime.state['call_ledger'].values()):
        raise IntegrityError('editorial call in bypassed window')
    for tid, record in records.items():
        decision = _decision(record)
        usable = decision['boardless_disposition'] == 'verified_candidate'
        binding = {'policy': POLICY, 'decision_hash': digest(decision),
                   'receipt_hash': digest(value), 'completed_at_epoch': value['completed_at_epoch']}
        if (value['decisions'][tid] != decision or record.get('editorial_bypass_binding') != binding
                or record.get('boardless_disposition') != decision['boardless_disposition']
                or record.get('boardless_reasons') != decision['reasons']
                or record.get('pipeline_usable_before_union_audit') is not usable
                or record.get('editorial_mode') != 'bypass' or record.get('q2_review_binding')
                or record.get('editorial_judgments')
                or record.get('intrinsic_quality') in ('core_quality', 'reserve_quality', 'rejected_quality')):
            raise IntegrityError('boardless disposition or evidence binding changed')
    return value


def verify(runtime, result):
    native.verify(runtime, result)
    if result.get('editorial_mode') != 'bypass':
        raise IntegrityError('terminal window has another editorial mode')
    receipt = verify_bypass(runtime)
    if result['editorial_bypass_hash'] != digest(receipt):
        raise IntegrityError('terminal bypass receipt changed')
    palette = runtime.palette_allocator.summary(window=runtime.window)
    if result.get('palette_accounting_hash') != digest(palette):
        raise IntegrityError('terminal palette accounting changed')
    if read(runtime.run_dir/'palette_accounting.private.json') != palette:
        raise IntegrityError('saved palette accounting differs from the shared ledger')
    return {'status': 'verified', 'window': result['window'], 'editorial_mode': 'bypass'}


def run(runtime, window, allocation, portfolio, invoke):
    mode = window.get('editorial_mode', 'enabled')
    configured = runtime.factory_config['_experiment_profile'].get('editorial_mode', 'enabled')
    if mode != configured or mode not in ('enabled', 'bypass'):
        raise IntegrityError('window/runtime editorial mode differs')
    if mode == 'enabled':
        return run_with_board(runtime, window, allocation, portfolio, invoke)
    manifest = prepare(runtime, window, allocation)
    path = runtime.run_dir / 'window_terminal.private.json'
    from .gate import recover_progress
    recover_progress(runtime)
    recover_terminal_evidence_metadata(runtime)
    if path.exists():
        result = read(path); verify(runtime, result)
        runtime.palette_allocator.write_report()
        return result
    ids = [r['task_id'] for r in manifest['rows']]
    if ids:
        run_slate(runtime.factory_config, runtime, slate=1, task_ids=ids, assignments=manifest['assignments'])
    for tid in ids:
        apply_final_evidence(runtime, tid)
    receipt = apply_bypass(runtime)
    artifacts = materialize_terminal_artifacts(runtime.factory_config, runtime)
    atomic(runtime.run_dir / 'call_timing_ledger.json', call_accounting([runtime]))
    palette = runtime.palette_allocator.summary(window=runtime.window)
    frozen(runtime.run_dir/'palette_accounting.private.json', palette)
    runtime.palette_allocator.write_report()
    result = {'status': 'complete', 'window': window['id'], 'runtime_directory': str(runtime.run_dir),
              'editorial_mode': mode, 'editorial_bypass_hash': digest(receipt),
              'palette_accounting_hash': digest(palette),
              'manifest_hash': digest(manifest), 'records_hash': digest(runtime.state['semantic_records']),
              'call_ledger_hash': digest(runtime.state['call_ledger']), 'artifacts': artifacts,
              'conditional_reservations': deepcopy(runtime.state.get('e2_escalation_budgets', {})),
              'tasks_committed': 0, 'new_provider_calls_on_terminal_resume': 0}
    frozen(path, result); verify(runtime, result)
    return result
