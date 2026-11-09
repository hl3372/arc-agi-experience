"""Build a task bank folder from the published ARC Task Bank dataset, so batches start from its tasks, not an empty bank.

  python -m arcgen.batch.bank_from_hf --repo                  download the dataset files it reads, then convert
  python -m arcgen.batch.bank_from_hf --local <folder>        convert a downloaded copy of the dataset

  --out <folder>              the bank to create (default: ARCGEN_BANK, else <workspace>/bank); refused unless it is
                              missing or an empty folder
  --limit N                   N tasks spread evenly over the bank's commit order (default: every task)
  --exclude-eval-near-twins   leave out the tasks flagged as a twin or near-twin of a public ARC-AGI-2 evaluation task
  --exclude-flag FLAG         also leave out the tasks with this provenance flag (repeatable; see FLAGS)
  --with-lifecycle            put each task's lifecycle record in its dossier (about 1.3 GB for the full bank)
  --revision R, --cache-dir D with --repo: a dataset revision, and the huggingface_hub cache folder

The dataset (https://huggingface.co/datasets/hl3372/arc-agi-task-bank, CC BY 4.0) holds the bank in three forms, all
read here: `data/tasks/*.parquet` (one row per task: grids, rule texts, design summary, flags),
`data/records/lifecycle-*.parquet` (each task's original registry entry and review record) and `arc/arc_tasks.zip`
(the ARC files). The folder written is a bank as `bank_inputs.prepare` and `bin/arcgen commit` expect it:

- `tasks/<id>.json`: each ARC file of the zip (the Parquet grids if the zip lacks it), written in the bank's own format,
  so its bytes reproduce the original bank's: its SHA-256 must equal the original `arc_sha256` of the lifecycle record,
  and its grids the Parquet row's;
- `registry.json`, `active_registry.json`: one entry per task, the original registry entry as published, with the fields
  the dataset keeps elsewhere restored (search mode, editorial mode and quality grade, boardless disposition, project
  stage, primary arm) and the dataset's `eval_near_twin` and `excluded_from_training` flags added; the commit sets with
  their dates (their original notes and authorizations are not published);
- `dossiers/<id>/`: minimal dossiers: the task, its `tasks` row without the grids, provenance and the hash manifest (and
  the lifecycle record with --with-lifecycle). The full records are in the dataset's `records` config;
- empty quarantine and basic-valid registries, one commit manifest per commit set, the catalog, checksums, a README and
  the dataset's licence.

The bank is built in a sibling folder and moved into place only after it verifies as the commit verifies a bank: the
registry (fields, task hashes, ARC validity, every dossier hash), the active partition, the basic-valid registry, the
catalog and the checksums. Nothing here makes a model or embedding call, and no render is drawn: the first preparation
of a batch on the bank embeds its reference texts once (OpenRouter by default; cached in the workspace), and the first
commit into it draws one render per bank task.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile

DATASET = 'hl3372/arc-agi-task-bank'
DATASET_URL = 'https://huggingface.co/datasets/' + DATASET
LICENSE = 'CC-BY-4.0'
# What a conversion reads (the `records` config's model-call file is not needed).
PATTERNS = ('data/tasks/*.parquet', 'data/records/lifecycle-*.parquet', 'arc/arc_tasks.zip', 'README.md', 'LICENSE')
# Provenance flags of the dataset card that can exclude a task: column -> the values that count as flagged.
FLAGS = {'eval_near_twin': ('twin', 'near'), 'excluded_from_training': (True,), 'admitted_via_right_twice': (True,),
         'owner_override': (True,), 'overruled_credible_rival': (True,)}
# Registry fields the dataset keeps in the `tasks` row (or the review record) rather than in the published registry entry.
RESTORED = ('search_mode', 'editorial_mode', 'quality_grade', 'project_stage', 'primary_arm')
GRIDS = ('train', 'test')
PROGRESS = 2000
RECORDS_NOTE = ('Minimal dossier built from the published ARC Task Bank dataset. The full design records are in its '
                '`records` config: the lifecycle record `{record}` and the model calls whose record_id is listed in '
                '`call_record_ids`.')


class ConversionError(ValueError):
    pass


def task_bytes(task):
    """An ARC task as the bank writes it (arcgen.lib.util.atomic_write_json): the bytes its arc_sha256 binds."""
    return (json.dumps(task, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def file_sha256(path):
    digest = sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, compact=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, separators=(',', ':')) if compact else json.dumps(value, ensure_ascii=False, indent=2)
    path.write_text(text + '\n', encoding='utf-8')


# ------------------------------------------------------------------ dataset files
def download(repo=DATASET, revision=None, cache_dir=None):
    """The dataset files a conversion reads, fetched into huggingface_hub's cache: (folder, resolved revision)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("--repo needs huggingface_hub: pip install 'huggingface_hub==2.4.0' (or pip install '.[hf]')")
    folder = Path(snapshot_download(repo_id=repo, repo_type='dataset', revision=revision, allow_patterns=list(PATTERNS),
                                    cache_dir=str(cache_dir) if cache_dir else None))
    return folder, (folder.name if re.fullmatch('[0-9a-f]{40}', folder.name) else revision)


def dataset_files(folder):
    folder = Path(folder)
    files = {'tasks': sorted(folder.glob('data/tasks/*.parquet')),
             'lifecycle': sorted(folder.glob('data/records/lifecycle-*.parquet')), 'zip': folder / 'arc/arc_tasks.zip'}
    if not files['tasks'] or not files['lifecycle']:
        raise ConversionError(f'{folder} is not a copy of the dataset: it needs data/tasks/*.parquet and '
                              'data/records/lifecycle-*.parquet')
    return files


def read_rows(path, columns=None):
    """The rows of one Parquet file as dicts, one row group at a time (the whole bank is never in memory at once)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("reading the dataset needs pyarrow: pip install 'pyarrow==26.0.1' (or pip install '.[hf]')")
    handle = pq.ParquetFile(path)
    for group in range(handle.metadata.num_row_groups):
        yield from handle.read_row_group(group, columns=columns).to_pylist()


def rows(paths, columns=None):
    for path in paths:
        yield from read_rows(path, columns)


# ------------------------------------------------------------------ conversion
def select(files, exclude=(), limit=None):
    """The task IDs to convert, in the dataset's order (commit date, then ID): every task without an excluded flag, or
    `limit` of them spread evenly over that order. Returns (ids, {flag: tasks left out}, tasks in the dataset)."""
    exclude = sorted(set(exclude))
    kept, dropped, seen = [], {flag: 0 for flag in exclude}, set()
    for row in rows(files['tasks'], ['task_id'] + exclude):
        if row['task_id'] in seen:
            raise ConversionError('duplicate task_id in the tasks config: ' + row['task_id'])
        seen.add(row['task_id'])
        hits = [flag for flag in exclude if row[flag] in FLAGS[flag]]
        for flag in hits:
            dropped[flag] += 1
        if not hits:
            kept.append(row['task_id'])
    if limit is not None and limit < len(kept):
        kept = [kept[i * len(kept) // limit] for i in range(limit)]
    return kept, dropped, len(seen)


def lifecycle(files, wanted, stage=None):
    """{task id: what the conversion needs from its lifecycle record}; with stage, each record is also written to its
    dossier as lifecycle.json."""
    found = {}
    for row in rows(files['lifecycle'], ['record_id', 'json']):
        record_id = row['record_id'] or ''
        tid = record_id[len('lifecycle-'):]
        if not record_id.startswith('lifecycle-') or tid not in wanted:
            continue
        value = json.loads(row['json'])
        entry, review = value.get('registry_entry') or {}, value.get('review') or {}
        if value.get('task_id') != tid or entry.get('registry_task_id') != tid or tid in found:
            raise ConversionError('lifecycle record does not match its task: ' + record_id)
        found[tid] = {'entry': entry, 'source_candidate_id': review.get('source_candidate_id') or review.get('candidate_id'),
                      'boardless_disposition': review.get('boardless_disposition')}
        if stage is not None:
            path = stage / 'dossiers' / tid / 'lifecycle.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(row['json'] + '\n', encoding='utf-8')
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ConversionError(f'{len(missing)} tasks have no lifecycle record (the original arc_sha256), e.g. {missing[0]}')
    return found


def registry_entry(row, life, dossier_path, dossier_sha256):
    """The bank entry: the published original, with the fields kept elsewhere restored and the dataset's flags added."""
    tid, entry = row['task_id'], deepcopy(life['entry'])
    for field in ('title', 'concept_summary', 'intended_rule', 'commit_set_id'):
        if entry.get(field) != row[field]:
            raise ConversionError(f'{tid}: {field} differs between the tasks row and the lifecycle record')
    for field in RESTORED:
        if field not in entry and row.get(field) is not None:
            entry[field] = row[field]
    if not entry.get('source_candidate_id'):
        entry['source_candidate_id'] = life['source_candidate_id'] or tid
    if entry.get('intrinsic_status') == 'boardless_verified' and 'boardless_disposition' not in entry:
        entry['boardless_disposition'] = life['boardless_disposition'] or 'verified_candidate'
    entry.update(task_path=f'tasks/{tid}.json', eval_near_twin=row['eval_near_twin'],
                 excluded_from_training=bool(row['excluded_from_training']), dataset_record_id='lifecycle-' + tid,
                 dossier_path=dossier_path, dossier_sha256=dossier_sha256)
    return entry


def write_dossier(stage, row, task, source):
    """A minimal dossier the commit verifies (authoring, review and provenance records plus the hash manifest)."""
    from arcgen.lib.committed_dossiers import _manifest_files
    tid = row['task_id']; folder = stage / 'dossiers' / tid
    note = RECORDS_NOTE.format(record='lifecycle-' + tid)
    write_json(folder / 'authoring.json', {'task': task, 'note': note}, compact=True)  # grids: one line, not one per cell
    write_json(folder / 'review.json', {'task_id': tid, 'complete_task': task, 'note': note,
                                        'dataset_tasks_row': {k: v for k, v in row.items() if k not in GRIDS}}, compact=True)
    write_json(folder / 'provenance.json', {
        'schema_version': 1, 'visibility': 'author_only_contains_hidden_test_output_and_rule', 'source': source,
        'records': {'lifecycle_record_id': 'lifecycle-' + tid, 'call_record_ids': row.get('call_record_ids') or [],
                    'lifecycle_record_in_dossier': (folder / 'lifecycle.json').is_file()},
        'task_payload': {'path': f'tasks/{tid}.json', 'sha256': sha256(task_bytes(task)).hexdigest()}})
    manifest = {'schema_version': 1, 'registry_task_id': tid, 'visibility': 'author_only', 'contains_hidden_test_output': True,
                'contains_intended_rule': True, 'source_run': row.get('batch'), 'arc_sha256': sha256(task_bytes(task)).hexdigest(),
                'artifact_policy': {'dossier': note}, 'files': _manifest_files(folder)}
    write_json(folder / 'manifest.json', manifest)
    return f'dossiers/{tid}/manifest.json', file_sha256(folder / 'manifest.json')


def bank_readme(meta):
    filters = []
    if meta['filters']['exclude']:
        filters.append('without the tasks flagged ' + ', '.join(f'`{f}`' for f in meta['filters']['exclude']))
    if meta['filters']['limit'] is not None:
        filters.append(f'limited to {meta["filters"]["limit"]} tasks spread over the commit order')
    revision = f' (revision `{meta["revision"]}`)' if meta.get('revision') else ''
    return '\n'.join([
        '# Task bank: ARC Task Bank', '',
        f'{meta["tasks"]:,} of the {meta["dataset_tasks"]:,} tasks of the ARC Task Bank dataset ({meta["url"]}){revision}'
        + (', ' + '; '.join(filters) if filters else '') + f', converted by `python -m arcgen.batch.bank_from_hf` on '
        f'{meta["converted_on"]}. The dataset is licensed CC BY 4.0 (see LICENSE): credit it when you share this bank or '
        'tasks from it.', '',
        '- `tasks/<id>.json`: the ARC tasks, byte for byte as in the original bank (`arc_sha256` in the registry).',
        '- `registry.json`, `active_registry.json`: one entry per task (the original registry entry, plus the dataset\'s '
        '`eval_near_twin` and `excluded_from_training` flags).',
        '- `dossiers/<id>/`: minimal dossiers: the task, its `tasks` row without the grids, and provenance'
        + (', and its lifecycle record' if meta['dossiers'] == 'minimal+lifecycle' else '') + '. The full design records '
        'are in the dataset\'s `records` config (`lifecycle-<id>` and the model calls listed in `call_record_ids`).',
        '- `eval_near_twin`: `twin` or `near` marks a task found to be a twin or near-twin of a public ARC-AGI-2 '
        'evaluation task; tasks committed after mid-October are `unchecked`. Report public-evaluation scores of a system '
        'trained on this bank split by this flag.', '',
        '`bin/arcgen commit` adds verified tasks to the bank. The first batch prepared on it embeds the bank\'s reference '
        'texts once (cached in the workspace); the first commit draws one render per task.', ''])


def convert(folder, out, *, exclude=(), limit=None, with_lifecycle=False, source=None, log=print):
    """Write a verified bank at `out` from a copy of the dataset in `folder`; returns a summary."""
    from arcgen.batch.bank_inputs import empty_registry, quarantine_registry, basic_valid_registry
    from arcgen.lib.active_bank import filter_active_registry
    from arcgen.lib.authorized_commits import _write_catalog, _write_checksums
    from arcgen.lib.util import atomic_write_json
    from arcgen.lib.validation import validate_arc_task
    started = time.time()
    folder, out = Path(folder), Path(out).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ConversionError(f'{out} exists and is not empty; refusing to overwrite a bank (choose another --out, or move '
                              'the folder away first)')
    if limit is not None and limit < 1:
        raise ConversionError('--limit must be at least 1')
    files = dataset_files(folder)
    ids, dropped, total = select(files, exclude, limit)
    if not ids:
        raise ConversionError('no task is left to convert')
    log(f'Converting {len(ids):,} of {total:,} dataset tasks into {out} ...')
    source = {'dataset': DATASET, 'url': DATASET_URL, 'license': LICENSE, 'downloaded_from_hub': False, 'revision': None,
              **(source or {})}
    meta = {**source, 'converter': 'arcgen.batch.bank_from_hf', 'converted_on': time.strftime('%Y-%m-%d'),
            'dataset_tasks': total, 'tasks': len(ids), 'filters': {'exclude': sorted(set(exclude)), 'limit': limit},
            'excluded': dropped, 'dossiers': 'minimal+lifecycle' if with_lifecycle else 'minimal',
            'files': {p.relative_to(folder).as_posix(): file_sha256(p) for p in [*files['tasks'], *files['lifecycle'], files['zip']]
                      if p.is_file()},
            'full_records': "the dataset's `records` config: lifecycle-<task id> and the model calls in call_record_ids"}
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{out.name}.converting-', dir=out.parent))
    umask = os.umask(0); os.umask(umask); stage.chmod(0o777 & ~umask)
    try:
        wanted = set(ids)
        life = lifecycle(files, wanted, stage if with_lifecycle else None)
        log(f'Read {len(life):,} lifecycle records ({time.time() - started:.0f} s); writing tasks and dossiers ...')
        archive = zipfile.ZipFile(files['zip']) if files['zip'].is_file() else None
        members = set(archive.namelist()) if archive else set()
        entries, dates, from_parquet = {}, {}, 0
        dossier_source = {'dataset': source['dataset'], 'url': source['url'], 'revision': source['revision'],
                          'license': LICENSE, 'converter': meta['converter']}
        for row in rows(files['tasks']):
            tid = row['task_id']
            if tid not in wanted:
                continue
            grids = {s: [{'input': pair['input'], 'output': pair['output']} for pair in row[s]] for s in GRIDS}
            if tid + '.json' in members:
                task = json.loads(archive.read(tid + '.json'))
                if task != grids:
                    raise ConversionError(f'{tid}: the ARC file in arc/arc_tasks.zip and the grids of its tasks row differ')
            else:
                task, from_parquet = grids, from_parquet + 1
            data = task_bytes(task); digest = sha256(data).hexdigest()
            original = life[tid]['entry'].get('arc_sha256')
            if digest != original:
                raise ConversionError(f'{tid}: the task bytes hash to {digest}, not to the original arc_sha256 {original}')
            check = validate_arc_task(task, require_test_outputs=True)
            if not check.valid:
                raise ConversionError(f'{tid}: not a valid ARC task: ' + '; '.join(check.errors))
            (stage / 'tasks').mkdir(exist_ok=True)
            (stage / 'tasks' / f'{tid}.json').write_bytes(data)
            entries[tid] = registry_entry(row, life[tid], *write_dossier(stage, row, task, dossier_source))
            dates.setdefault(row['commit_set_id'], row['commit_date'])
            if len(entries) % PROGRESS == 0:
                log(f'  {len(entries):,}/{len(ids):,} tasks ({time.time() - started:.0f} s)')
        if archive:
            archive.close()
        if set(entries) != wanted:
            raise ConversionError(f'{len(wanted - set(entries))} selected tasks are missing from the tasks config')
        meta['arc_files'] = {'from_zip': len(entries) - from_parquet, 'from_parquet_grids': from_parquet}
        # commit sets in the order they first appear (the dataset is ordered by commit date), their tasks in dataset order
        order = {}
        for tid in ids:
            order.setdefault(entries[tid]['commit_set_id'], []).append(entries[tid])
        registry = empty_registry(source_dataset=meta)
        for set_id, in_set in order.items():
            stages = {e.get('project_stage') for e in in_set}
            registry['commit_sets'].append({
                'commit_set_id': set_id, 'committed_on': dates[set_id], 'entry_count': len(in_set),
                'notes': f'Commit set of the published ARC Task Bank ({DATASET}), imported by arcgen.batch.bank_from_hf.',
                **({'project_stage': stages.pop()} if len(stages) == 1 and None not in stages else {})})
            registry['entries'].extend(in_set)
        atomic_write_json(stage / 'registry.json', registry)
        atomic_write_json(stage / 'active_registry.json', filter_active_registry(registry, bank=stage))
        atomic_write_json(stage / 'quarantine_registry.json', quarantine_registry(len(entries)))
        atomic_write_json(stage / 'basic_valid_registry.json', basic_valid_registry())
        for item in registry['commit_sets']:
            atomic_write_json(stage / 'commit_manifests' / f'{item["commit_set_id"]}.json',
                              {'schema_version': 1, **item, 'source_dataset': DATASET, 'new_model_calls': 0})
        _write_catalog(stage, registry); _write_checksums(stage, registry)
        (stage / 'README.md').write_text(bank_readme(meta), encoding='utf-8')
        if (Path(folder) / 'LICENSE').is_file():
            shutil.copyfile(Path(folder) / 'LICENSE', stage / 'LICENSE')
        log(f'Verifying the bank ({time.time() - started:.0f} s) ...')
        verify(stage, processes=1)  # no forked workers: pyarrow's threads are alive in this process
        if out.exists():
            out.rmdir()
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    summary = {'bank': str(out), 'tasks': len(entries), 'dataset_tasks': total, 'excluded': dropped,
               'commit_sets': len(registry['commit_sets']), 'arc_files': meta['arc_files'], 'dossiers': meta['dossiers'],
               'seconds': round(time.time() - started, 1)}
    log(json.dumps(summary))
    return summary


def verify(bank, processes=None):
    """The checks the commit makes of a bank before and after adding to it, on the new bank: the registry loaded with
    every entry checked (fields, task hash and ARC validity, dossier manifest and every dossier file hash), the active
    partition, the basic-valid registry, and the catalog and checksums exact. processes: dossier-checking worker
    processes (commit_bank.verify_processes() by default; 1 checks them in this process)."""
    from arcgen.batch import commit_bank
    from arcgen.lib import e18_commit
    from arcgen.lib.active_bank import filter_active_registry
    from arcgen.lib.util import read_json
    bank = Path(bank)
    registry = commit_bank.load_registry(bank / 'registry.json', deep=True, processes=processes)
    e18_commit.load_basic_valid_registry(bank / 'basic_valid_registry.json')
    if read_json(bank / 'active_registry.json') != filter_active_registry(read_json(bank / 'registry.json'), bank=bank):
        raise ConversionError('the active registry is not the active partition of the registry')
    with commit_bank.patched(e18_commit, load_committed_task_registry=lambda path, **kwargs: registry):
        derived = e18_commit._verify_main_derived_artifacts({'_paths': {'historical_registry': str(bank / 'registry.json')}})
    if not (derived['catalog_exact'] and derived['checksums_exact']):
        raise ConversionError('the catalog or checksums do not match the registry')
    return registry


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m arcgen.batch.bank_from_hf', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument('--repo', nargs='?', const=DATASET, metavar='ORG/NAME',
                       help=f'download from the Hugging Face Hub (default {DATASET}); needs huggingface_hub')
    where.add_argument('--local', type=Path, metavar='FOLDER', help='a downloaded copy of the dataset')
    p.add_argument('--revision', help='with --repo: the dataset revision (branch, tag or commit)')
    p.add_argument('--cache-dir', type=Path, help='with --repo: the huggingface_hub cache folder')
    p.add_argument('--out', type=Path, help='the bank folder to create (default: ARCGEN_BANK, else <workspace>/bank)')
    p.add_argument('--limit', type=int, metavar='N', help='N tasks spread evenly over the commit order')
    p.add_argument('--exclude-eval-near-twins', action='store_true',
                   help='leave out tasks flagged as a twin or near-twin of a public ARC-AGI-2 evaluation task')
    p.add_argument('--exclude-flag', action='append', default=[], choices=sorted(FLAGS), metavar='FLAG',
                   help='leave out tasks with this provenance flag: ' + ', '.join(sorted(FLAGS)) + ' (repeatable)')
    p.add_argument('--with-lifecycle', action='store_true', help="add each task's lifecycle record to its dossier")
    x = p.parse_args(argv)
    if x.local and (x.revision or x.cache_dir):
        p.error('--revision and --cache-dir go with --repo')
    from arcgen.paths import BANK
    out = (x.out or BANK).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):  # before any download
        p.error(f'{out} exists and is not empty; refusing to overwrite a bank')
    if x.repo:
        print(f'Downloading {x.repo} (tasks, lifecycle records and ARC files, about 260 MB) ...', flush=True)
        folder, revision = download(x.repo, x.revision, x.cache_dir)
        source = {'dataset': x.repo, 'url': 'https://huggingface.co/datasets/' + x.repo, 'downloaded_from_hub': True,
                  'revision': revision}
    else:
        folder, source = x.local.expanduser().resolve(), None
    exclude = list(x.exclude_flag) + (['eval_near_twin'] if x.exclude_eval_near_twins else [])
    try:
        convert(folder, out, exclude=exclude, limit=x.limit, with_lifecycle=x.with_lifecycle, source=source,
                log=lambda m: print(m, flush=True))
    except ConversionError as exc:
        raise SystemExit(f'conversion failed, nothing written: {exc}')
    from arcgen.paths import BANK as default
    print(f'Done. The bank is at {out}' + ('' if out == default else f'; use it with: export ARCGEN_BANK={out}'))


if __name__ == '__main__':
    main()
