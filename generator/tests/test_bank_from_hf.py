"""The bank converter (arcgen/batch/bank_from_hf.py) on ten real rows of the published dataset (support/hf_bank_sample.json).

A converted bank verifies as a committed bank with the original task bytes, a run prepares on it and a commit adds to
it; the filters, the overwrite refusal, the grid and hash checks and the Parquet-grid fallback hold. The sample is written
as Parquet when pyarrow is installed; without it, the Parquet reader is replaced by one serving the same rows.
"""
import json
import zipfile
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from arcgen.batch import bank_from_hf as hf, bank_inputs, commit
from arcgen.lib.committed_tasks import load_committed_task_registry
from support import commit_fixtures as fx

SAMPLE = Path(__file__).resolve().parent / 'support/hf_bank_sample.json'
NEAR = {'e18-production__d93da3a0a42542f1ecf8', 'e18-production__d8e736dbeb6abbf0db86'}  # flagged twin and near
EXCLUDED = 'e18-production__d5a85687df4e8293c1b2'  # excluded_from_training
NO_SOURCE_ID = 'v43-joint-r2__kdc6555fb0df10'  # its published registry entry has no source_candidate_id


def quiet(message):
    pass


def arc_file(row):
    return json.dumps({s: [{'input': p['input'], 'output': p['output']} for p in row[s]] for s in ('train', 'test')},
                      separators=(',', ':'))


def original_sha(lifecycle_row):
    return json.loads(lifecycle_row['json'])['registry_entry']['arc_sha256']


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """build(tasks=None, lifecycle=None, arc=None) -> a dataset folder holding the sample rows (or the given ones)."""
    served = {}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pa = pq = None

        def read_rows(path, columns=None):
            for row in served[str(Path(path))]:
                yield {k: row[k] for k in columns} if columns else dict(row)
        monkeypatch.setattr(hf, 'read_rows', read_rows)
    counter = iter(range(100))

    def build(tasks=None, lifecycle=None, arc=None):
        sample = json.loads(SAMPLE.read_text())
        tasks = sample['tasks'] if tasks is None else tasks
        lifecycle = sample['lifecycle'] if lifecycle is None else lifecycle
        folder = tmp_path / f'dataset-{next(counter)}'
        for rows, path in ((tasks, folder / 'data/tasks/tasks-00000.parquet'),
                           (lifecycle, folder / 'data/records/lifecycle-00000.parquet')):
            path.parent.mkdir(parents=True)
            if pq:
                pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=4)  # several row groups
            else:
                path.write_bytes(b''); served[str(path)] = rows
        (folder / 'arc').mkdir()
        with zipfile.ZipFile(folder / 'arc/arc_tasks.zip', 'w') as archive:
            for tid, text in (arc if arc is not None else {r['task_id']: arc_file(r) for r in tasks}).items():
                archive.writestr(tid + '.json', text)
        (folder / 'LICENSE').write_text('CC0 1.0 Universal (test copy)\n')
        return folder
    return build


def test_a_converted_bank_verifies_with_the_original_task_bytes(dataset, tmp_path):
    folder, bank = dataset(), tmp_path / 'bank'
    sample = json.loads(SAMPLE.read_text())
    summary = hf.convert(folder, bank, log=quiet)
    assert summary['tasks'] == 10 and summary['arc_files'] == {'from_zip': 10, 'from_parquet_grids': 0}
    registry = load_committed_task_registry(bank / 'registry.json')  # the commit's own loader: every hash and dossier
    entries = {e['registry_task_id']: e for e in registry['entries']}
    assert set(entries) == {r['task_id'] for r in sample['tasks']}
    for row in sample['lifecycle']:  # the original bank's bytes, reproduced
        tid = row['record_id'][len('lifecycle-'):]
        assert sha256((bank / 'tasks' / f'{tid}.json').read_bytes()).hexdigest() == original_sha(row) == entries[tid]['arc_sha256']
    for row in sample['tasks']:
        entry = entries[row['task_id']]
        assert entry['search_mode'] == row['search_mode'] and entry['intrinsic_status'] == row['bank_status']
        assert entry['eval_near_twin'] == row['eval_near_twin'] and entry['excluded_from_training'] == row['excluded_from_training']
        if row['bank_status'] == 'boardless_verified':
            assert (entry['boardless_disposition'], entry['editorial_mode'], entry['quality_grade']) == ('verified_candidate', 'bypass', 'unassigned')
        review = json.loads((bank / 'dossiers' / row['task_id'] / 'review.json').read_text())
        assert review['complete_task'] == json.loads((bank / entry['task_path']).read_text())
        assert review['dataset_tasks_row']['call_record_ids'] == row['call_record_ids'] and 'train' not in review['dataset_tasks_row']
    assert entries[NO_SOURCE_ID]['source_candidate_id'] == 'kdc6555fb0df10'
    assert {e['registry_task_id'] for e in entries.values() if e['eval_near_twin'] in ('twin', 'near')} == NEAR
    active = json.loads((bank / 'active_registry.json').read_text())
    assert [e['registry_task_id'] for e in active['entries']] == [e['registry_task_id'] for e in registry['entries']]
    dates = [s['committed_on'] for s in registry['commit_sets']]
    assert dates == sorted(dates) and sum(s['entry_count'] for s in registry['commit_sets']) == 10
    assert registry['source_dataset']['dataset'] == hf.DATASET and (bank / 'LICENSE').is_file()
    assert not list(tmp_path.glob('.bank.converting-*'))
    inputs = bank_inputs.prepare(tmp_path / 'runs/r1', bank=bank)  # a run's frozen view of the bank
    assert inputs['active_tasks'] == 10


def test_filters_limit_and_lifecycle_dossiers(dataset, tmp_path):
    folder = dataset()
    order = [r['task_id'] for r in json.loads(SAMPLE.read_text())['tasks']]  # the dataset's order: commit date, then ID
    summary = hf.convert(folder, tmp_path / 'a', exclude=['eval_near_twin', 'excluded_from_training'], log=quiet)
    kept = [e['registry_task_id'] for e in json.loads((tmp_path / 'a/registry.json').read_text())['entries']]
    assert summary['excluded'] == {'eval_near_twin': 2, 'excluded_from_training': 1}
    assert set(kept) == set(order) - NEAR - {EXCLUDED}
    hf.convert(folder, tmp_path / 'b', limit=3, with_lifecycle=True, log=quiet)
    registry = load_committed_task_registry(tmp_path / 'b/registry.json')
    assert {e['registry_task_id'] for e in registry['entries']} == {order[0], order[3], order[6]}  # spread evenly
    lifecycle = {r['record_id']: r['json'] for r in json.loads(SAMPLE.read_text())['lifecycle']}
    for entry in registry['entries']:
        folder_ = tmp_path / 'b/dossiers' / entry['registry_task_id']
        assert (folder_ / 'lifecycle.json').read_text() == lifecycle[entry['dataset_record_id']] + '\n'
        assert 'lifecycle.json' in {f['path'] for f in json.loads((folder_ / 'manifest.json').read_text())['files']}


def test_refuses_an_existing_bank_and_inconsistent_data(dataset, tmp_path):
    sample = json.loads(SAMPLE.read_text())
    folder, bank = dataset(), tmp_path / 'bank'
    bank.mkdir(); (bank / 'registry.json').write_text('{}')
    with pytest.raises(hf.ConversionError, match='refusing to overwrite'):
        hf.convert(folder, bank, log=quiet)
    assert [p.name for p in bank.iterdir()] == ['registry.json']
    empty = tmp_path / 'empty'; empty.mkdir()
    assert hf.convert(folder, empty, limit=2, log=quiet)['tasks'] == 2  # an empty folder is fine
    first = sample['tasks'][0]['task_id']
    arc = {r['task_id']: arc_file(r) for r in sample['tasks']}
    changed = json.loads(arc[first]); changed['test'][0]['output'][0][0] = (changed['test'][0]['output'][0][0] + 1) % 10
    with pytest.raises(hf.ConversionError, match='differ'):  # the zip and the Parquet grids disagree
        hf.convert(dataset(arc={**arc, first: json.dumps(changed, separators=(',', ':'))}), tmp_path / 'c', log=quiet)
    lifecycle = deepcopy(sample['lifecycle'])
    record = json.loads(lifecycle[0]['json']); record['registry_entry']['arc_sha256'] = '0' * 64
    lifecycle[0]['json'] = json.dumps(record)
    with pytest.raises(hf.ConversionError, match='original arc_sha256'):
        hf.convert(dataset(lifecycle=lifecycle), tmp_path / 'd', log=quiet)
    assert not (tmp_path / 'c').exists() and not (tmp_path / 'd').exists() and not list(tmp_path.glob('.*.converting-*'))
    missing = {k: v for k, v in arc.items() if k != first}  # no ARC file in the zip: the task comes from the Parquet grids
    summary = hf.convert(dataset(arc=missing), tmp_path / 'e', log=quiet)
    assert summary['arc_files'] == {'from_zip': 9, 'from_parquet_grids': 1}
    assert (tmp_path / 'e/tasks' / f'{first}.json').read_bytes() == (tmp_path / 'empty/tasks' / f'{first}.json').read_bytes()


def test_a_run_commits_into_a_converted_bank(dataset, tmp_path, monkeypatch):
    root = tmp_path / 'root'; bank = root / 'committed_tasks'
    hf.convert(dataset(), bank, log=quiet)
    before = {e['registry_task_id']: (e['arc_sha256'], e['dossier_sha256'])
              for e in json.loads((bank / 'registry.json').read_text())['entries']}
    fx.make_run(root, 'b1', 3, homes=commit.HOMES, seed=3)
    (root / 'commit_code.py').write_text('# commit code\n')
    # no forked workers: the dataset fixture may have loaded pyarrow (and its threads) into this process
    monkeypatch.setenv('ARCGEN_COMMIT_WORKERS', '1'); monkeypatch.setenv('ARCGEN_COMMIT_VERIFY_PROCESSES', '1')
    monkeypatch.setattr(commit, 'ROOT', root); monkeypatch.setattr(commit, 'BANK', bank)
    monkeypatch.setattr(commit, '__file__', str(root / 'commit_code.py'))
    monkeypatch.setattr(commit.Commit, 'verify_candidate', fx.light_verify_candidate)
    c = commit.Commit(root / 'runs/b1', 'b1-task-commit-x', 'ok', committed_on='2026-11-09', then_install=True)
    result = c.commit(c.prepare())
    assert result['committed'] == 3 and result['historical_bank'] == 13 and result['derived_artifacts']['valid']
    registry = load_committed_task_registry(bank / 'registry.json')
    assert {e['registry_task_id']: (e['arc_sha256'], e['dossier_sha256']) for e in registry['entries'][:10]} == before
    assert registry['commit_sets'][-1]['commit_set_id'] == 'b1-task-commit-x' and registry['source_dataset']['tasks'] == 10
    hf.verify(bank, processes=1)
