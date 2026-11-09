"""Saved-queue windows: develop frozen selected ideas with no idea or selector calls; resume and replay are stable."""
from hashlib import sha256
from math import ceil
from pathlib import Path
import json
import threading
import pytest
from arcgen.lib.factory_runtime import FactoryModelInvoker
from arcgen.batch import pipeline, scheduler
from arcgen.batch.allocation import design
from arcgen.batch.budget import Budget
from arcgen.batch.common import ROOT, account_pool
from arcgen.astra_usage_diversity_20260913.common import read, digest
from support.bank import prepare
from arcgen.batch.retrieval import ReferenceIndex
from support.fixtures import Vectors
from support.fixtures import episode, s

QUEUE = 'batches/queues/synthetic-queue.json'  # workspace-relative, as a carry script writes it


def saved_windows(inputs):
    from support.carry import saved_queue
    if not (ROOT / QUEUE).exists():
        saved_queue(ROOT / QUEUE, inputs)
    rows = len(read(ROOT / QUEUE)['rows'])
    wid = 'one-saved-01'
    saved = dict(id=wid, phase='screen', wave='A', account='main', context='c01', seed=4242, kind='saved_queue',
                 supply_only=True, editorial_mode='bypass', maximum_developments=0, minimum_selector_eligible_kernels=1,
                 maximum_retained=rows, protected=False, protected_targets=[], calls=[], pairing_index=0,
                 saved_queue=QUEUE, saved_queue_sha256=sha256((ROOT / QUEUE).read_bytes()).hexdigest(), saved_rows=rows)
    groups = [dict(id=f'{wid}-develop-{n + 1:02d}', phase='screen', wave='A', account=None, source_account='main',
                   context='c01', seed=4242 + n + 1, kind='eligible_group', calls=[], editorial_mode='bypass',
                   maximum_developments=3, source_window=wid, group_ordinal=n,
                   research_calls_per_account={'main': 0}) for n in range(ceil(rows / 3))]
    return [saved] + groups, rows


def test_saved_queue_rejects_a_changed_file(tmp_path):
    from arcgen.batch import saved_queue
    windows, _ = saved_windows(prepare(tmp_path))
    bad = dict(windows[0], saved_queue_sha256='0' * 64)
    with pytest.raises(Exception, match='saved queue changed'):
        saved_queue.load(bad)


def test_saved_queue_develops_every_row_without_idea_or_selector_calls(tmp_path, monkeypatch):
    inputs = prepare(tmp_path)
    windows, rows = saved_windows(inputs)
    requests = {w['id']: {} for w in windows}
    cap = design()['limits']; cap['phases'][0]['phase'] = 'screen'
    budget = Budget(tmp_path, cap)
    index = ReferenceIndex(read(inputs['selector_context'])['references'], Vectors(), provider_identity={'synthetic': True})
    seen, lock = [], threading.Lock()

    def fake(invoker, stage, cid):
        with lock: seen.append((stage, cid))
        if stage not in (s.PLAN_STAGE, s.SCALE_STAGE):
            raise AssertionError('saved queue made a non-planning call: ' + stage)
        req = read(invoker.calls_dir / stage / cid / 'request.json')
        tid = req['parent_ids'][0]; plan, _ = episode(tid)
        plan.update(status='abort', abort_reason='Concrete unresolved semantic defect in this test.', public_contract=None, private_execution=None)
        if stage == s.PLAN_STAGE: return json.dumps(plan)
        return json.dumps({'candidate_id': tid, 'scale_decision': 'abort', 'scale_reason': plan['abort_reason'], 'concrete_plan': None})

    monkeypatch.setattr(FactoryModelInvoker, '_mock_response', fake)

    def run():
        pools = {a: account_pool(tmp_path / 'accounts' / a / 'worker_pool.json') for a in ('main',)}
        try:
            return scheduler.run_windows(tmp_path, 'screen', windows, requests, inputs, index=index, pools=pools, budget=budget,
                                         production=inputs['production'], manifest={'fixture': 'saved'}, mock=True)
        finally:
            for pool in pools.values(): pool.finish()

    terminal = run()
    queue = read(tmp_path / 'accounts/main/one-saved-01/eligible_queue.private.json')
    assert len(queue['rows']) == rows == budget.summary()['developments']
    assert all(r['study']['planner'] == 'sol' and r['study']['saved_queue_origin']['window'] for r in queue['rows'])
    assert not any(stage in ('e16_idea_search',) or 'selector' in stage for stage, _ in seen)
    baseline = (len(seen), budget.summary()['logical_calls'], digest(terminal))
    assert run() == terminal and (len(seen), budget.summary()['logical_calls'], digest(terminal)) == baseline
