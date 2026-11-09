"""Synthetic committed bank and verified runs for the commit tests (test_commit_transactions.py); no model calls.

A run built here passes the commit's own lineage checks unchanged: per-call request/prompt/schema/routing provenance,
frozen idea requests, the two-turn idea loop, the window's single selector call, source rows, the idea-loop kernel and
the terminal-artifact folder. Only the final-evidence gate (`Commit.verify_candidate`, which reads the full evidence DAG
of a real run) is replaced by `light_verify_candidate`.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from hashlib import sha256
from pathlib import Path

from arcgen.lib.active_bank import filter_active_registry
from arcgen.lib.authorized_commits import _write_catalog, _write_checksums
from arcgen.lib.committed_dossiers import _manifest_files
from arcgen.lib.committed_tasks import load_committed_task_registry
from arcgen.lib.e18_commit import _new_basic_registry
from arcgen.lib.util import content_hash
from arcgen.lib.validation import validate_arc_task

COHORT = 'arcgen-synthetic-commit'
MODEL, EFFORT = 'gpt-6.1-sol', 'xhigh'


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) if not isinstance(value, str) else value)
    return path


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def arc_task(rng):
    def grid(h, w):
        return [[rng.randrange(10) for _ in range(w)] for _ in range(h)]
    while True:
        pairs = []
        for _ in range(rng.randint(3, 4)):
            g = grid(rng.randint(3, 11), rng.randint(3, 11))
            pairs.append({'input': g, 'output': [list(r) for r in zip(*g)]})
        task = {'train': pairs[:-1], 'test': pairs[-1:]}
        if validate_arc_task(task, require_test_outputs=True).valid:
            return task


def _entry(rid, task_path, arc, commit_set, rule, summary):
    return {'registry_task_id': rid, 'title': summary[:60] + f' [{rid[-8:]}]', 'task_path': task_path, 'arc_sha256': arc,
            'source_pipeline': 'synthetic-seed', 'source_run': 'runs/synthetic-seed', 'source_candidate_id': rid.split('__')[-1],
            'search_mode': 'v1_loop', 'intrinsic_status': 'boardless_verified', 'source_intrinsic_status': None,
            'quality_grade': 'unassigned', 'editorial_mode': 'bypass', 'boardless_disposition': 'verified_candidate',
            'answer_mode': 'single_answer', 'portfolio_status': 'include_now', 'commit_set_id': commit_set,
            'concept_summary': summary, 'intended_rule': rule, 'project_stage': 'stage_3'}


def make_bank(root, n, *, seed=0, blob=1500, render=True, quarantine=1):
    """A frozen registry of n hash-bound tasks with dossiers, catalog, checksums, active registry and (as the old commit
    renders them) the full render folder and contact sheet."""
    from arcgen.batch import commit as committer
    root = Path(root); bank = root / 'committed_tasks'; rng = random.Random(seed)
    sets = [('seed-set-a', n // 2), ('seed-set-b', n - n // 2)]
    entries = []
    for set_id, count in sets:
        for _ in range(count):
            i = len(entries); rid = f'e18-production__bank{seed:02d}{i:05d}'
            task = arc_task(rng)
            write(bank / 'tasks' / (rid + '.json'), json.dumps(task))
            summary = f'Seed concept {i}: transpose the {rng.choice(["grid", "board", "scene"])} along its main diagonal'
            entry = _entry(rid, f'tasks/{rid}.json', sha(bank / 'tasks' / (rid + '.json')), set_id,
                           f'Transpose every cell (rule {i}) ' + ' '.join(rng.choice(['keep', 'the', 'colours', 'as', 'they', 'are']) for _ in range(rng.randint(5, 40))), summary)
            d = bank / 'dossiers' / rid
            write(d / 'authoring.json', {'task': task}); write(d / 'review.json', {'complete_task': task, 'i': i})
            write(d / 'provenance.json', {'synthetic': True}); (d / 'model_calls/x').mkdir(parents=True)
            (d / 'model_calls/x/raw_response.txt').write_bytes(os.urandom(blob))
            manifest = {'schema_version': 1, 'registry_task_id': rid, 'visibility': 'author_only', 'contains_hidden_test_output': True,
                        'contains_intended_rule': True, 'arc_sha256': entry['arc_sha256'], 'files': _manifest_files(d)}
            write(d / 'manifest.json', manifest)
            entry['dossier_path'] = f'dossiers/{rid}/manifest.json'; entry['dossier_sha256'] = sha(d / 'manifest.json')
            entries.append(entry)
    registry = {'schema_version': 1, 'registry_id': 'arcgen-committed-good-tasks-v1', 'frozen': True, 'automatic_writes_allowed': False,
                'requires_explicit_user_authorization': True, 'commit_policy': {'explicit_user_authorization': True},
                'commit_sets': [{'commit_set_id': s, 'committed_on': '2026-10-01', 'entry_count': c, 'notes': f'Seed set {s}.'} for s, c in sets],
                'dossier_policy': {'schema_version': 1, 'required_for_every_committed_task': True,
                                   'visibility': 'author_only_contains_hidden_answers'},
                'entries': entries}
    write(bank / 'registry.json', json.dumps(registry, ensure_ascii=False, indent=2) + '\n')
    for k in range(quarantine):
        write(bank / 'audit_flags' / f'flag{k}.json', {'registry_task_id': entries[1 + k]['registry_task_id'], 'status': 'quarantine_synthetic'})
    write(bank / 'active_registry.json', json.dumps(filter_active_registry(registry, bank=bank), ensure_ascii=False, indent=2) + '\n')
    write(bank / 'basic_valid_registry.json', json.dumps(_new_basic_registry(), indent=2) + '\n')
    write(bank / 'quarantine_registry.json', {'entries': []}); write(root / 'reservoirs/v4_control.json', {'reservoir': []})
    write(bank / 'README.md', '# Synthetic bank\n')
    for s, c in sets:
        write(bank / 'commit_manifests' / f'{s}.json', {'commit_set_id': s, 'entry_count': c})
    _write_catalog(bank, registry); _write_checksums(bank, registry)
    if render:
        committer.render_bank_bounded(bank, load_committed_task_registry(bank / 'registry.json'))
    return bank


class _Calls:
    """Model-call folders with the provenance the commit checks (request, prompt, schema, routing, response hashes)."""

    def __init__(self, run, homes, rng, blob):
        self.run, self.homes, self.rng, self.blob = run, homes, rng, blob

    def make(self, namespace, stage, call_id, *, role, purpose, parent_ids, account, routed=True, prompt=None, big=0):
        log = namespace / 'model_calls' / stage / call_id
        prompt = prompt if prompt is not None else f'Prompt for {call_id} ({stage}). ' + 'x' * self.rng.randint(10, 200)
        schema = {'type': 'object', 'title': stage}
        write(namespace / 'schemas' / (content_hash(schema)[:16] + '.json'), schema)
        response = {'call': call_id, 'answer': self.rng.random()}
        raw = json.dumps(response)
        write(log / 'prompt.md', prompt)
        write(log / 'request.json', {'prompt_hash': content_hash(prompt), 'model': {'name': MODEL, 'reasoning_effort': EFFORT},
                                     'schema_hash': content_hash(schema)})
        write(log / 'parsed_response.json', response); write(log / 'raw_response.txt', raw)
        write(log / 'result.json', {'parsed_response_hash': content_hash(response), 'raw_response_hash': content_hash(raw)})
        write(log / 'program_transport_identity.json', {'verified': True, 'account': account})
        (log / 'transport.stderr.log').write_bytes(os.urandom(self.blob + big))
        (log / 'physical_attempts/attempt-01').mkdir(parents=True)
        write(log / 'physical_attempts/attempt-01/attempt-1.log', f'attempt of {call_id}\n')
        if routed:
            write(namespace / 'routing' / stage / (call_id + '.json'), {'model': MODEL, 'reasoning_effort': EFFORT, 'call_id': call_id,
                                                                       'account': account})
        return {'call_id': call_id, 'stage': stage, 'role': role, 'purpose': purpose, 'parent_ids': list(parent_ids),
                'log_dir': str(log), 'model': MODEL, 'reasoning_effort': EFFORT, 'account': account,
                'codex_home': self.homes[account], 'status': 'complete', 'response_content_hash': content_hash(response)}


def make_run(root, name, n_tasks, *, homes, seed=1, per_window=3, blob=4000, unusable=1, prefix=None):
    """A verified_terminal run of n_tasks usable tasks (plus `unusable` excluded rows) laid out as a v1 run."""
    root = Path(root); run = root / 'runs' / name; rng = random.Random(seed)
    account = sorted(homes)[0]; prefix = prefix or f'{seed:02d}'
    calls = _Calls(run, homes, rng, blob)
    write(run / 'config.json', {'revision': COHORT, 'default_run': f'runs/{name}'})
    write(run / 'STATUS.json', {'status': 'verified_terminal'})
    for f in ('terminal_verification.json', 'completion.json', 'prepared_manifest.private.json'):
        write(run / f, {'file': f, 'run': name})
    write(run / 'launch_manifest.private.json', {'run': name, 'padding': 'p' * 20000})
    (run / 'budget.sqlite3').write_bytes(os.urandom(4096))
    windows = [f'one-supply-{w + 1:02d}' for w in range(-(-(n_tasks + unusable) // per_window))]
    write(run / 'design/batch.private.json', {'windows': [{'id': w, 'kind': 'fresh_supply'} for w in windows]})
    rows, research, k = [], {}, 0
    for w in windows:
        wdir = run / 'accounts' / account / w
        write(wdir / 'window.private.json', {'window': {'id': w, 'cohort': COHORT, 'kind': 'fresh_supply'}})
        write(wdir / 'selector_inputs.private.json', {'inputs': 'i' * 3000, 'window': w})
        write(wdir / 'eligible_queue.private.json', {'queue': [w]})
        sel = calls.make(wdir / 'selector', 'arm_bank_selector_v1', f's{prefix}{w[-2:]}sel', role='selector', purpose='selection',
                         parent_ids=[], account=account, big=6000)
        write(wdir / 'selector/state.json', {'call_ledger': {sel['call_id']: sel}})
        base = f'p{prefix}{w[-2:]}idea'
        sketch_prompt = f'Sketch prompt for {base}'
        write(run / 'design/windows' / w / 'ideas.private.json',
              {base: {'private': {'idea_model': MODEL, 'effort': EFFORT, 'account': account}, 'prompt': sketch_prompt}})
        up = wdir / 'upstream-01'
        sk = calls.make(up, 'v1_idea_sketch', base + '-sketch', role='idea', purpose='idea', parent_ids=[], account=account,
                        routed=False, prompt=sketch_prompt)
        cm = calls.make(up, 'v1_idea_commit', base + '-commit', role='idea', purpose='idea', parent_ids=[sk['call_id']],
                        account=account, routed=False)
        write(up / 'state.json', {'call_ledger': {sk['call_id']: sk, cm['call_id']: cm}})
        members = [f'{prefix}{w[-2:]}t{j}{rng.randrange(16**6):06x}' for j in range(per_window)]
        kernels = {tid: {'candidate_id': 'k-' + tid, 'central_relation': f'Concept of {tid}: transpose the scene ' +
                         ' '.join(rng.choice(['so', 'that', 'rows', 'become', 'columns']) for _ in range(rng.randint(3, 20)))}
                   for tid in members}
        write(wdir / 'idea_loop' / (base + '.private.json'), {'draft': base})
        write(wdir / 'idea_loop' / (base + '.committed.private.json'), {'ideas': {'ideas': list(kernels.values())}})
        write(wdir / 'source_rows.private.json', {kern['candidate_id']: {'kernel': kern, 'private_source_call_id': base,
                                                                         'private_effort': EFFORT} for kern in kernels.values()})
        for j, tid in enumerate(members):
            if k >= n_tasks + unusable:
                break
            usable = k < n_tasks; k += 1
            dev = run / 'accounts' / account / f'{w}-develop-{j + 1:02d}'; ns = dev / 'downstream'
            write(dev / 'window.private.json', {'window': {'cohort': COHORT, 'source_account': account, 'source_window': w}})
            write(dev / 'source_rows.private.json', {'rows': [tid]}); write(dev / 'allocation.private.json', {'allocation': tid})
            task = arc_task(rng)
            write(ns / 'tasks' / (tid + '.arc.json'), json.dumps(task)); write(ns / 'tasks' / (tid + '.authoring.json'), {'task': task})
            ledger = {}
            for stage, cid, role, purpose, big in (('astra_concrete_plan_v3', tid + '-concrete-plan', 'concrete_planner', 'planning', 0),
                                                   ('weakness_v1_task_realizer', tid + '-realization', 'realizer', 'realization', 3 * blob),
                                                   ('astra_technical_retention_v4', tid + '-technical', 'technical_verifier', 'technical', 0),
                                                   ('v4_blind_solver', tid + '-blind-1', 'blind_solver', 'blind', blob)):
                c = calls.make(ns, stage, cid, role=role, purpose=purpose, parent_ids=[tid], account=account, big=big)
                ledger[cid] = c
            cand = dev / 'candidates' / tid
            write(cand / 'palette.r0.json', {'palette': rng.random(), 'pad': 'q' * 2000})
            write(cand / 'realization.r0.json', {'realization': tid})
            write(cand / 'pre_repair_history' / (tid + '.authoring.json'), {'task': task})
            write(cand / 'artifact_manifest.json', {'schema_version': 2, 'task_id': tid})
            record = {'task_id': tid, 'complete_task': task, 'kernel': kernels[tid], 'source_candidate_id': kernels[tid]['candidate_id'],
                      'private_source_call_id': base, 'search_mode': 'v1_loop', 'private_route': 'ordinary_current',
                      'study': {'primary_arm': 'SEE', 'invitation': None, 'bank_setting': 'loop', 'window': w,
                                'original_selector_window': w, 'planner': 'sol', 'overlay': None, 'colour_roles': 'v1'},
                      'concrete_plan': {'semantic_core': {'central_rule': f'Transpose every grid of {tid}.'}},
                      'terminal_artifact': {'artifact_manifest': str(cand / 'artifact_manifest.json')},
                      'intrinsic_quality': None, 'boardless_disposition': 'verified_candidate'}
            write(ns / 'editorial_bypass.private.json', {'policy': 'bypass', 'decisions': {tid: 'verified_candidate'}})
            write(ns / 'state.json', {'semantic_records': {tid: record}, 'call_ledger': ledger})
            rows.append({'task_id': tid, 'status': 'verified' if usable else 'rejected', 'evidence_clear': usable, 'arm': 'SEE',
                         'review_reason': None if usable else 'blind solver failed',
                         'usable_after_exact_palette_union': usable, 'verified_before_union': usable, 'window': dev.name,
                         'state': {'path': str(ns / 'state.json'), 'sha256': sha(ns / 'state.json')},
                         'blind_performance': {'any_all_test_success': True, 'solved': usable},
                         'generation_model': MODEL, 'effort': EFFORT, 'cohort': COHORT})
            research[tid] = {'research_status': 'disabled_by_design'}
    write(run / 'analysis/terminal_report.private.json', {'status': 'verified_terminal', 'tasks': rows,
                                                         'windows': [{'window': r['window'], 'terminal': True} for r in rows]})
    write(run / 'analysis/research_index.private.json', research)
    return run


def light_verify_candidate(self, row):
    """The final-evidence gate replaced for synthetic runs (same proof shape); every other check runs unchanged."""
    namespace = Path(row['state']['path']).parent
    record = self.state(namespace)['semantic_records'][row['task_id']]
    if sha((namespace / 'state.json')) != row['state']['sha256']:
        raise ValueError('Reviewed source state changed')
    self.track(namespace / 'editorial_bypass.private.json')
    if not validate_arc_task(record['complete_task'], require_test_outputs=True).valid:
        raise ValueError('Candidate does not meet unchanged final evidence gates: ' + row['task_id'])
    self.research_disabled()
    return record, {'task_id': row['task_id'], 'final_evidence': {'status': 'evidence_clear'}, 'terminal': {'synthetic': True},
                    'original_record_hash': content_hash(record), 'research_status': 'disabled_by_design',
                    'editorial_mode': 'bypass', 'quality_grade': 'unassigned', 'research_is_not_production_evidence': True}


def snapshot(folder, *, skip=('.e18-commit.lock',)):
    """{relative path: sha256} of every file under folder (symlinks not followed), for byte-for-byte state comparison.
    The bank's lock file (created empty by the first commit's lock, old and new code alike) is left out."""
    folder = Path(folder); out = {}
    for path in sorted(folder.rglob('*')):
        rel = path.relative_to(folder).as_posix()
        if any(rel == s or rel.startswith(s + '/') for s in skip):
            continue
        if path.is_symlink():
            out[rel] = 'symlink:' + os.readlink(path)
        elif path.is_file():
            out[rel] = sha(path)
    return out


def clone(source, target):
    """A copy of a synthetic root (hard links are not preserved: each test root owns its files)."""
    shutil.copytree(source, target, symlinks=True)
    return Path(target)
