"""The palette ledger's read (summary) takes no write lock: a final audit with thousands of terminal window checks used to
stop on 'database is locked' when each check opened BEGIN IMMEDIATE just to read. Under WAL a plain SELECT reads the
committed snapshot while a writer holds the lock."""
import json
import sqlite3
import time
from copy import deepcopy

from arcgen.astra_usage_diversity_20260913 import palette


def _allocator(tmp_path):
    (tmp_path / 'inputs').mkdir()
    (tmp_path / 'inputs/palette_context.private.json').write_text(
        json.dumps({'schema_version': 2, 'policy': deepcopy(palette.POLICY), 'seed': 1}))
    return palette.Allocator(tmp_path)


def test_summary_reads_while_a_writer_holds_the_lock(tmp_path):
    a = _allocator(tmp_path)
    a.register('t1', 'main', 'w1', 'h1')
    other = sqlite3.connect(a.path)
    other.execute('BEGIN IMMEDIATE')
    other.execute("INSERT INTO candidates(task_id,account,window,admission_hash) VALUES('t2','main','w2','h2')")
    try:
        start = time.monotonic()
        value = a.summary()
        assert time.monotonic() - start < 5
        assert value['registered'] == 1 and value['unmaterialized_task_ids'] == ['t1']  # the committed snapshot only
        assert a.summary(window='w1')['registered'] == 1 and a.summary(window='w2')['registered'] == 0
    finally:
        other.commit(); other.close()
    assert a.summary()['registered'] == 2 and a.summary(window='w2')['unmaterialized_task_ids'] == ['t2']


def test_writers_still_serialize_and_wait_their_turn(tmp_path):
    a = _allocator(tmp_path)
    assert palette.BUSY_SECONDS >= 300
    with a.transaction() as db:
        db.execute("INSERT INTO candidates(task_id,account,window,admission_hash) VALUES('t1','main','w1','h1')")
    assert a.write_report()['registered'] == 1
    assert json.loads((tmp_path / 'palette_report.private.json').read_text())['registered'] == 1
