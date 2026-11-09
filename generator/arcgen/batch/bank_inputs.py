"""Freeze a run's view of the task bank: task copies, retrieval references, author rows and the palette context.

A bank is a folder (arcgen.paths.BANK) with `active_registry.json`, the active tasks, and `tasks/<id>.json` files named
by each entry's `task_path`. `ensure_bank` creates an empty one. Preparation copies (or hard-links) every active task
into `<run>/inputs/bank_tasks/`, so a run never reads the live bank after it is prepared; the bank registry files are
recorded in `<run>/inputs/protected.private.json` and a resumed run stops if they changed (commit a finished batch,
not one still running).

The reference corpus for novelty retrieval is the accepted bank tasks only: `committed:<registry id>` with the task's
concept summary and rule. Selector memory starts empty; the batch's own selected ideas join it per reference wave
(recent.py, waves.py).
"""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import time
from arcgen.lib.util import content_hash
from arcgen.lib.v4_reservoir import concept_hash
from arcgen.lib.v41_palette import palette_equivalent_hash
from arcgen.paths import BANK, CACHE, rel
from .common import IntegrityError, read, ref, sha, frozen, atomic, digest, verify_refs, isolated_root
from .config import SCIENCE

REVISION = 'arcgen-bank-inputs-v1'
REGISTRY_FILES = ('registry.json', 'active_registry.json', 'quarantine_registry.json', 'basic_valid_registry.json')
CONTEXTS = ('c01', 'c02', 'c03', 'c04')  # idea-window contexts named by allocation specs; each keeps its own selector memory
# Default embedding provider for novelty retrieval (science.embedding overrides it). The key comes from the environment.
EMBEDDING = {'provider': 'openrouter', 'model': 'openai/text-embedding-3-large', 'api_key_env': 'OPENROUTER_API_KEY',
             'base_url': 'https://openrouter.ai/api/v1', 'batch_size': 64, 'timeout_seconds': 120, 'retries': 2,
             'app_name': 'arcgen-v4-control'}
ELIGIBLE_STATUSES = ('core_quality', 'reserve_quality', 'boardless_verified')
DOSSIER_POLICY = {'schema_version': 1, 'required_for_every_committed_task': True,
                  'visibility': 'author_only_contains_hidden_answers',
                  'raw_artifact_rule': 'Relevant candidate model-call directories are copied verbatim into each dossier.',
                  'solver_export_rule': 'Dossier content must never enter solver-facing ARC exports.',
                  'authorization': 'explicit user-authorized commits; each authorization is retained in its registry commit set'}
COMMIT_POLICY = {'included_status': 'Explicitly authorized core/reserve tasks and ungraded boardless-verified tasks',
                 'novelty_bank_rule': 'A generated concept enters novelty comparison only through an explicitly authorized commit.',
                 'run_log_rule': 'Uncommitted kernels, rejections and interrupted work stay in run folders; they are not bank entries.',
                 'transient_rule': 'Candidates of one batch may be compared with one another without becoming bank entries.',
                 'boardless_rule': 'boardless_verified records evidence clearance under editorial bypass; it is not a quality grade.'}


def embedding_config():
    return deepcopy(SCIENCE.get('embedding') or EMBEDDING)


def empty_registry(registry_id='arcgen-committed-good-tasks-v1', **extra):
    return {'schema_version': 1, 'registry_id': registry_id, 'frozen': True, 'automatic_writes_allowed': False,
            'requires_explicit_user_authorization': True, 'commit_policy': deepcopy(COMMIT_POLICY), 'commit_sets': [],
            'entries': [], 'dossier_policy': deepcopy(DOSSIER_POLICY), **extra}


def quarantine_registry(entry_count=0):
    """An empty quarantine registry beside a bank of entry_count tasks, all active."""
    return {'schema_version': 1, 'registry_id': 'arcgen-evidence-quarantine-bank', 'status': 'active_quarantine',
            'purpose': 'Accepted tasks with unresolved answer-evidence concerns, kept out of the active bank.',
            'historical_registry_path': 'registry.json', 'historical_registry_entry_count': entry_count,
            'active_registry_path': 'active_registry.json', 'active_registry_entry_count': entry_count, 'quarantine_entry_count': 0,
            'automatic_reactivation_allowed': False, 'historical_artifacts_mutated': False, 'entries': []}


def basic_valid_registry():
    """An empty basic-valid (curriculum) registry."""
    return {'schema_version': 1, 'registry_id': 'arcgen-committed-basic-valid-v1', 'frozen': True, 'automatic_writes_allowed': False,
            'requires_explicit_user_authorization': True, 'excluded_from_positive_generation_context': True,
            'curriculum_status_is_not_intrinsic_quality': True, 'dossier_policy': deepcopy(DOSSIER_POLICY), 'commit_sets': [],
            'entries': []}


def ensure_bank(bank=None):
    """Create an empty bank (registries only) when the folder has none; an existing bank is never touched. A bank with
    tasks can be built from the published dataset instead (bank_from_hf.py)."""
    bank = Path(bank or BANK)
    if (bank / 'active_registry.json').is_file():
        return bank
    if any((bank / name).exists() for name in REGISTRY_FILES):
        raise IntegrityError(f'{bank} has some bank registries but no active_registry.json; refusing to overwrite')
    bank.mkdir(parents=True, exist_ok=True)
    (bank / 'tasks').mkdir(exist_ok=True)
    atomic(bank / 'registry.json', empty_registry())
    atomic(bank / 'active_registry.json', empty_registry(active_bank_filter={
        'historical_registry_entry_count': 0, 'active_entry_count': 0, 'excluded_quarantined_registry_task_ids': [],
        'historical_registry_mutated': False}))
    atomic(bank / 'quarantine_registry.json', quarantine_registry())
    atomic(bank / 'basic_valid_registry.json', basic_valid_registry())
    (bank / 'README.md').write_text('# Task bank\n\nCreated empty by arcgen on ' + time.strftime('%Y-%m-%d') +
                                    '. `bin/arcgen commit` adds verified tasks; see the repository README.\n')
    return bank


def _place(source, target):
    """Hard-link a bank task into the run when the filesystem allows it, else copy it (same bytes either way)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != sha(source):
            raise IntegrityError('frozen bank copy changed: ' + str(target))
        return
    try:
        os.link(source, target)
        target.chmod(0o444)
    except OSError:
        shutil.copyfile(source, target)


def empty_memory():
    """The selector memory structure of one context, with nothing in it."""
    return {'schema_version': 1, 'payload': {'in_flight_reservations': [],
            'rejected_memory_digest': {'hard_duplicate': [], 'soft_mechanism': [], 'failure_mode': [], 'recoverable_work': []}}}


def prepare(root, bank=None):
    """Freeze the bank view for a new run (idempotent on resume)."""
    from arcgen.astra_usage_diversity_20260913.palette import freeze_inputs as freeze_palette, Allocator
    root = isolated_root(root)
    path = root / 'inputs/revision_inputs.private.json'
    if path.exists():
        value = read(path); verify_refs(value)
        Allocator(root)
        return value
    bank = ensure_bank(bank)
    registry_path = bank / 'active_registry.json'
    registry = read(registry_path)
    inputs = root / 'inputs'
    files = []
    production_registry = inputs / 'production/committed_tasks/active_registry.json'
    _place(registry_path, production_registry)
    files.append(ref(production_registry))
    order = {r['commit_set_id']: i for i, r in enumerate(registry.get('commit_sets', []))}
    entries = sorted(enumerate(registry['entries']), key=lambda v: (order.get(v[1].get('commit_set_id'), -1), v[0]))
    rows, metadata, identities, facts, provenance, references, exact = [], {}, [], [], [], [], set()
    for ordinal, (_, entry) in enumerate(entries):
        rid = entry['registry_task_id']
        source = bank / entry['task_path']
        if not source.resolve().is_relative_to(bank.resolve()) or sha(source) != entry['arc_sha256']:
            raise IntegrityError('bank task path or bytes differ from the registry: ' + rid)
        copy = inputs / 'bank_tasks' / (rid + '.json')
        _place(source, copy)
        files.append(ref(copy))
        task = read(copy); task_hash = content_hash(task)
        if entry.get('source_episode_hash') and task_hash != entry['source_episode_hash']:
            raise IntegrityError('committed final episode hash changed: ' + rid)
        key = 'a' + content_hash({'e16': 'accepted', 'registry_task_id': rid})[:13]
        rows.append({'accepted_concept_id': key, 'title': entry['title'], 'concept_summary': entry['concept_summary'],
                     'rule_summary': entry['intended_rule']})
        study = entry.get('study') or {}
        meta = {'registry_task_id': rid, 'known_lineage_key': 'committed:' + rid,
                'eligible': entry.get('intrinsic_status') in ELIGIBLE_STATUSES, 'relevance': [],
                'admission_ordinal': ordinal, 'admission_set': entry.get('commit_set_id'),
                'source_lineage': {'source_candidate_id': entry.get('source_candidate_id'), 'source_run': entry.get('source_run'),
                                   'source_kernel_id': entry.get('source_kernel_id')}}
        arm = entry.get('primary_arm') or study.get('primary_arm')
        if arm:
            meta['source_labels'] = {'arm': arm, 'invitation': entry.get('invitation', study.get('invitation')),
                                     'evidence': {'registry_task_id': rid, 'episode_hash': task_hash,
                                                  'status': 'committed_generation_provenance_not_mode_confirmation'}}
        metadata[key] = meta
        facts.append({'entry_id': key, 'training_pairs': len(task['train']), 'test_pairs': len(task['test'])})
        identities.append({'registry_task_id': rid, 'exact_hash': task_hash, 'palette_hash': palette_equivalent_hash(task),
                           'task': ref(copy)})
        provenance.append({'source_path': rel(source), 'source_sha256': entry['arc_sha256'], 'copy': ref(copy)})
        references.append({'reference_id': 'committed:' + rid, 'source': 'accepted_generated_task',
                           'archive_tier': 'hard_duplicate_memory',
                           'central_relation': entry['concept_summary'] + '\n' + entry['intended_rule'],
                           'summary': entry['intended_rule']})
        exact.update(concept_hash(entry[k]) for k in ('concept_summary', 'intended_rule'))
    portfolio = {'bank_snapshot_id': digest(rows), 'committed_generated_task_count': len(rows),
                 'concepts': [{k: e[k] for k in ('registry_task_id', 'title', 'concept_summary')} for e in registry['entries']]}
    production = {'portfolio_digest': portfolio, 'paths': {'committed_tasks/active_registry.json': str(production_registry)},
                  'files': [ref(production_registry)], 'provenance': [{'source_path': rel(registry_path),
                  'source_sha256': sha(registry_path), 'copy': ref(production_registry)}], 'active_bank_requested': True}
    atomic(inputs / 'production_context.private.json', production)
    contexts = {}
    for key in CONTEXTS:
        frozen(inputs / key / 'selector_memory.private.json', empty_memory())
        contexts[key] = {'description': {'id': key, 'search_mode': 'v1_loop'},
                         'selector_memory': str(inputs / key / 'selector_memory.private.json')}
        files.append(ref(inputs / key / 'selector_memory.private.json'))
    selector = {'accepted_bank_digest': portfolio, 'accepted_concept_hashes': sorted(exact), 'references': references,
                'selector_memory': {'schema_version': 1, 'active_reservations': [], 'tiered_rejection_memory': [],
                                    'quarantine_failure_mode_warnings': []},
                'embedding': embedding_config()}
    selector_path = inputs / 'selector_context.private.json'
    author_path, meta_path = inputs / 'author_bank.json', inputs / 'author_metadata.private.json'
    identity_path = inputs / 'bank_identities.private.json'
    frozen(selector_path, selector); frozen(author_path, rows)
    frozen(meta_path, {'metadata': metadata, 'structural_facts': facts, 'mode_label_inference_calls': 0})
    frozen(identity_path, {'registry': ref(production_registry), 'tasks': identities, 'provenance': provenance})
    palette = freeze_palette(root, registry_path=production_registry, task_root=bank)
    Allocator(root)  # initialize SQLite/WAL once before parallel runtime construction
    for p in (selector_path, author_path, meta_path, identity_path, inputs / 'production_context.private.json',
              inputs / 'palette_context.private.json'):
        files.append(ref(p))
    protected = frozen(inputs / 'protected.private.json', [ref(bank / name) for name in REGISTRY_FILES if (bank / name).is_file()])
    result = {'revision': REVISION, 'bank_mode': 'fresh_active', 'bank': str(bank), 'active_tasks': len(rows),
              'contexts': contexts, 'selector_context': str(selector_path), 'author_bank': str(author_path),
              'author_metadata': str(meta_path), 'bank_identities': str(identity_path),
              'historical_cache': str(CACHE), 'production': production, 'palette': palette, 'protected': protected,
              'files': files, 'new_text_calls': 0, 'launch_authorized': False}
    verify_refs(result)
    return frozen(path, result)
