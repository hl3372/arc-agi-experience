"""Explicitly authorized bank commitment of a finished v1 arcgen batch's verified tasks.

Adapted from the earlier commit code for v1 runs: the same bank
transaction (staging, backup, rollback, full re-verification), per-task evidence gates, exact/palette duplicate
check against the bank, complete dossiers with call lineage, and a project_stage flag on every entry and on the
commit set. v1 lineage: the two-turn idea call (draft, then commit after the bank check) checked against the frozen
design request and the saved loop record; every downstream call's model/effort checked against the pipeline's own
per-call routing record; the selector call of the original idea window.

Committed: every task usable after the batch's exact/palette union audit. Nothing else changes; no model calls.

  python -m arcgen.batch.commit --root runs/<run> --name <commit-set-id> --authorization "<user message>" [--partial] [--commit]

--partial takes a stopped run that will not resume: its finished windows are reported into this commit's folder (from
copies of the run's ledgers; the run is only read), and only verified tasks of windows with a window terminal are taken.

Without --commit it only prepares and verifies the staging area (resumable); with --commit it installs and
verifies. A completed commit re-run is an idempotent verification.

Several runs as one bank transaction (bin/arcgen commit A B C; 2026-10-24): each run is prepared on its own, in its own
process and config, stacked on the previous run's staged bank (--after; --defer-sheet for all but the last), then

  python -m arcgen.batch.commit --install <commit-set-id> [<commit-set-id> ...] --commit

installs them under one lock and journal with one backup, writes the bank-wide files and renders once and verifies the
final bank once, leaving one commit set, manifest and report per run, as separate commits would.

Speed-up of 2026-10-24: only the new tasks are drawn (the
renders kept are checked against committed_tasks/RENDER_SHA256SUMS; the first commit without it draws the whole bank
once), renders are installed file by file instead of replacing the folder, a dossier's model-call and terminal-evidence
records are hard links (read-only) of the run's files, and the bank's dossiers are hash-verified in full once per
transaction, on the final bank after the install (a dry run verifies them at prepare instead).
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import fcntl, shutil, struct, tempfile, time, zlib

from arcgen.lib import committed_dossiers, e18_commit
from arcgen.lib.active_bank import filter_active_registry
from arcgen.lib.authorized_commits import _concept_summary, _intended_rule, _render_bank, _write_catalog, _write_checksums
from arcgen.lib.committed_dossiers import _build_one, _manifest_files, CONTEXT_FILES
from arcgen.lib.committed_tasks import load_committed_task_registry
from arcgen.lib.e18_artifacts import strict_technical_clear, verify_terminal_artifact_bundle
from arcgen.lib.e18_commit import (_backup_manifest, _committed_bank_lock, _rollback, _staging_manifest,
                                   _verify_backup, _verify_main_derived_artifacts, _verify_staging, load_basic_valid_registry)
from arcgen.lib.factory_config import DEFAULT_SOURCE_PATHS
from arcgen.lib.factory_evidence import project_final_evidence
from arcgen.lib.util import atomic_write_json, atomic_write_text, content_hash, read_json
from arcgen.lib.v41_palette import palette_equivalent_hash
from arcgen.lib.v47_baseline_core_commit import _title
from arcgen.lib.validation import validate_arc_task
from . import commit_bank as fast
from .common import ROOT, HOMES
from .downstream import verify_bypass
from arcgen.paths import BANK


def rel(path):
    """A path as recorded by the commit: relative to the workspace (ROOT) when inside it, else absolute."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
def bank_paths():
    """The bank files the commit transaction reads and writes (arcgen.lib.e18_commit config['_paths'])."""
    return {k: str(BANK / Path(v).relative_to('committed_tasks')) if v.startswith('committed_tasks/') else str(BANK / Path(v).name)
            for k, v in DEFAULT_SOURCE_PATHS.items()}


IDEA_STAGES = ('v1_idea_sketch', 'v1_idea_explore', 'v1_idea_commit', 'v1_idea_commit_kept')
FORMAT = 2  # prepared.json / transaction.json of this code (the previous code wrote neither field)
BANK_METADATA = ('registry.json', 'active_registry.json', 'CATALOG.md', 'SHA256SUMS', fast.RENDER_INDEX)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def idle(run):
    """A stopped run nobody is writing: no coordinator holds its lock and no ledger has a write-ahead log."""
    lock = Path(run) / 'runner.lock'
    if any(Path(run).glob('*-wal')):
        return False
    if lock.exists():
        with lock.open() as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return True


def planner_call(ledger, tid):
    """The task's planning call. A rung task may hold two (its plan and the plan-stage re-plan, rungs.py); both run the
    task's frozen planner on its development account, so they bind the same role (2026-10-13: until then a re-planned task
    made the whole commit fail)."""
    plans = [c for c in ledger if c['purpose'] == 'planning' and tid in c['parent_ids']]
    if not plans or len({(c['model'], c['reasoning_effort'], c['account']) for c in plans}) != 1:
        raise ValueError('Planning lineage incomplete or inconsistent: ' + tid)
    return next((c for c in plans if c['call_id'] == tid + '-concrete-plan'), plans[0])


def read_holds(path):
    """Tasks held back from a commit by the owner (2026-10-13): a JSON object {task_id: reason} or a list of task IDs or
    of {"task_id", "reason"} objects, for example readers' human-"no" verdicts or reused answers. They are listed as
    excluded with their reason and stay in the run."""
    value = read_json(path)
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    return {(x if isinstance(x, str) else x['task_id']): ('held by the owner' if isinstance(x, str) else x.get('reason') or 'held by the owner')
            for x in value}


class _LinkingShutil:
    """shutil as committed_dossiers._build_one sees it during packaging: its copytree of the task's model-call folders
    hard-links the run's records (Commit.link_record) instead of copying them; everything else is shutil itself."""

    def __init__(self, commit):
        self._commit = commit

    def __getattr__(self, name):
        return getattr(shutil, name)

    def copytree(self, src, dst, **kwargs):
        return shutil.copytree(src, dst, copy_function=self._commit.link_record, **kwargs)


class Commit:
    def __init__(self, run, name, authorization, stage_flag='stage_3', committed_on=None, partial=False, hold=None,
                 release_judge=None, after=None, defer_sheet=False, hold_subset=False, then_install=False):
        self.run, self.name, self.flag = Path(run).resolve(), name, stage_flag
        self.hold_path = Path(hold).resolve() if hold else None
        self.release_path = Path(release_judge).resolve() if release_judge else None
        self.out = ROOT / 'runs' / name
        self.stage = self.out / 'staged_bank'
        self.partial, self.partial_record, self.unfinished = partial, {}, set()
        self.analysis = self.out / 'partial_source/analysis' if partial else self.run / 'analysis'
        self.user_message = authorization
        self.committed_on = committed_on or time.strftime('%Y-%m-%d')
        self.sources = {}
        self.config = read_json(self.run / 'config.json')
        self.cohort = self.config['revision']
        # transaction of several runs: the prepared commit this one stacks on, and whether the last one makes the sheet
        self.after, self.defer_sheet, self.hold_subset = after or None, bool(defer_sheet), bool(hold_subset)
        # an install follows in this transaction and hash-verifies every dossier of the final bank; a dry run does it here
        self.then_install = bool(then_install)
        self.linked, self.link_stats = set(), {}

    @classmethod
    def from_prepared(cls, name):
        """A prepared commit, rebuilt from its folder for the install of its transaction: (Commit, prepared)."""
        out = ROOT / 'runs' / name
        prepared = read_json(out / 'prepared.json')
        if prepared.get('format') != FORMAT:
            raise ValueError(f'runs/{name} was prepared by the previous commit code; delete it and commit again')
        member = cls(ROOT / prepared['run'], name, read_json(out / 'authorization.json')['user_message'], prepared['stage_flag'],
                     prepared['committed_on'], prepared['partial'], after=prepared['transaction']['after'],
                     defer_sheet=prepared['transaction']['defer_sheet'])
        return member, prepared

    # ---------------------------------------------------------------- tracked reads
    def track(self, path):
        path = Path(path).resolve(); key = rel(path)
        if key not in self.sources:
            self.sources[key] = sha256(path.read_bytes()).hexdigest()
        return path

    def read(self, path):
        return read_json(self.track(path))

    def copy(self, source, target):
        source = self.track(source); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)

    def copy_tree(self, source, target):
        for p in sorted(Path(source).rglob('*')):
            if p.is_file():
                self.copy(p, target / p.relative_to(source))

    def link_record(self, source, target):
        """A run's immutable record (a model-call or terminal-evidence file) put into the staged dossier: tracked (hashed
        into the source manifest) like every copy, then hard-linked (read-only) when it is a file of a run on the same
        filesystem, else copied as before. Same bytes either way."""
        source = self.track(source); target = Path(target)
        linked = fast.link_or_copy(source, target, link=fast.linkable(source, (ROOT / 'runs').resolve()), stats=self.link_stats)
        relative = target.relative_to(self.stage).as_posix()
        (self.linked.add if linked else self.linked.discard)(relative)
        return target

    def link_tree(self, source, target):
        for p in sorted(Path(source).rglob('*')):
            if p.is_file():
                self.link_record(p, target / p.relative_to(source))

    def cfg(self):
        return {'_repository_root': str(ROOT), '_output_directory': str(self.out),
                '_paths': bank_paths(), 'run': {'id': self.name},
                'commit': {'commit_set_id': self.name, 'committed_on': self.committed_on, 'authorization_record': self.authorization()}}

    def authorization(self):
        scope = (f'partial commit of the stopped run {rel(self.run)}: every task of its finished windows usable after its '
                 'exact/palette union audit, tasks of unfinished windows excluded' if self.partial else
                 f'every task of {rel(self.run)} usable after its exact/palette union audit')
        return (self.user_message + f' Scope: {scope}; flagged project_stage {self.flag}. Boardless-verified, editorial quality unassigned; original model/'
                'account/effort, planner and colour-instruction provenance and saved evidence preserved. Research disabled '
                'by design. Zero new model or embedding calls; no resume or new batch launch.')

    # ---------------------------------------------------------------- saved state
    @lru_cache(None)
    def state(self, namespace):
        return self.read(Path(namespace) / 'state.json')

    @lru_cache(None)
    def schemas(self, namespace):
        return {content_hash(self.read(p)): p for p in (Path(namespace) / 'schemas').glob('*.json')}

    @lru_cache(None)
    def window_spec(self, window):
        spec = self.read(Path(window) / 'window.private.json')['window']
        if spec['cohort'] != self.cohort:
            raise ValueError('Unexpected source cohort: ' + str(window))
        return spec

    def source_window(self, namespace):
        spec = self.window_spec(Path(namespace).parent)
        return self.run / 'accounts' / spec['source_account'] / spec['source_window']

    @lru_cache(None)
    def idea_calls(self, run=None):
        out = {}
        for p in sorted((run or self.run).glob('accounts/*/*/upstream*/state.json')):
            for c in self.state(p.parent)['call_ledger'].values():
                if c['stage'] in IDEA_STAGES:
                    out[c['call_id']] = c
        return out

    @lru_cache(None)
    def frozen_ideas(self, run=None):
        frozen = {}
        for path in sorted(((run or self.run) / 'design/windows').glob('*/ideas.private.json')):
            for call_id, request in self.read(path).items():
                if call_id in frozen:
                    raise ValueError('Repeated frozen generation identity')
                frozen[call_id] = request
        return frozen

    @lru_cache(None)
    def moved_runs(self):
        """{original root: current root} for saved source runs moved aside after their calls ran. Their records keep the
        original absolute call paths, where another run may now live (v1-b17-t1 -> v1-b17-t1.crashed-20261016, then a
        fresh v1-b17-t1 with the same call ids)."""
        moved = {}
        for run in self.saved_runs():
            original = read_json(run / 'config.json').get('default_run')
            if original and (ROOT / original).resolve() != run:
                moved[(ROOT / original).resolve()] = run
        return moved

    def local(self, log_dir):
        """A call's log folder where it lives now."""
        path = Path(log_dir)
        if path.is_relative_to(self.run):
            return path
        for original, run in self.moved_runs().items():
            if path.is_relative_to(original):
                return run / path.relative_to(original)
        return path

    def routed(self, call):
        """The pipeline's own routing record for a downstream call (model and effort bound at dispatch)."""
        path = self.local(call['log_dir']).parents[2] / 'routing' / call['stage'] / (call['call_id'] + '.json')
        return self.read(path)

    def saved_source_run(self, namespace):
        """For a task re-developed from a saved queue: the run whose idea and selector calls produced it."""
        spec = self.window_spec(Path(namespace).parent)
        saved = self.window_spec(self.run / 'accounts' / spec['source_account'] / spec['source_window'])
        source = self.read(ROOT / saved['saved_queue'])['provenance']['source_run']
        return (ROOT / source).resolve()

    def saved_runs(self):
        """Source runs of this run's saved queues and saved ideas (their idea calls, and a saved queue's selector)."""
        return sorted({(ROOT / self.read(ROOT / w[w['kind']])['provenance']['source_run']).resolve()
                       for w in self.read(self.run / 'design/batch.private.json')['windows'] if w.get('kind') in ('saved_queue', 'saved_ideas')})

    def upstream(self, namespace, record):
        """(run with the idea calls, window with the selector call, window with the idea loop) for one task."""
        origin = record['study'].get('saved_queue_origin')
        if origin:  # re-developed from a saved queue: its idea and selector calls are in the queue's source run
            upstream = self.saved_source_run(namespace)
            original_window, = upstream.glob(f"accounts/*/{origin['window']}")
            return upstream, original_window, original_window
        original_window = self.source_window(namespace)
        spec = self.window_spec(original_window)
        if spec.get('kind') != 'saved_ideas':
            return self.run, original_window, original_window
        # selected here from saved ideas: the selector call is in this run, the idea calls in the ideas' source run
        if sha256((ROOT / spec['saved_ideas']).read_bytes()).hexdigest() != spec['saved_ideas_sha256']:
            raise ValueError('Saved ideas changed after they were bound: ' + spec['saved_ideas'])
        saved = self.read(ROOT / spec['saved_ideas'])['provenance']
        upstream = (ROOT / saved['source_run']).resolve()
        idea_window, = upstream.glob(f"accounts/*/{saved['source_window']}")
        if record['study']['window'] != saved['source_window'] or record['source_candidate_id'] not in spec['saved_kernel_ids']:
            raise ValueError('Saved idea provenance changed: ' + record['source_candidate_id'])
        return upstream, original_window, idea_window

    def run_of(self, path):
        """The run a call's files live in: this run, or the source run of a saved queue or saved ideas."""
        for run in [self.run] + self.saved_runs():
            if Path(path).is_relative_to(run):
                return run
        raise ValueError('Call outside this run and its saved sources: ' + str(path))

    def call_assets(self, call):
        path = self.local(call['log_dir'])
        request = self.read(path / 'request.json')
        prompt = self.track(path / 'prompt.md').read_text()
        schema = self.schemas(path.parents[2])[request['schema_hash']]
        run = self.run_of(path)
        account = path.relative_to(run / 'accounts').parts[0]
        if call['stage'] in IDEA_STAGES:
            frozen = self.frozen_ideas(run)[call['call_id'].rsplit('-', 1)[0]]['private']
            expected = (frozen['idea_model'], frozen['effort'])
            if call['stage'] == 'v1_idea_sketch' and self.frozen_ideas(run)[call['call_id'].rsplit('-', 1)[0]]['prompt'] != prompt:
                raise ValueError('Frozen idea request changed: ' + str(path))
            if frozen['account'] != account:
                raise ValueError('Idea call ran on an unexpected account: ' + str(path))
        else:
            route = self.routed(call); expected = (route['model'], route['reasoning_effort'])
            if route['call_id'] != call['call_id'] or route['account'] != account:
                raise ValueError('Routing record mismatch: ' + str(path))
        if (request['prompt_hash'] != content_hash(prompt) or request['model']['name'] != call['model']
                or request['model']['reasoning_effort'] != call['reasoning_effort'] or (call['model'], call['reasoning_effort']) != expected
                or account not in HOMES or call['account'] != account or call['codex_home'] != HOMES[account]):
            raise ValueError('Source call provenance changed: ' + str(path))
        if call['status'] == 'complete':
            response = self.read(path / 'parsed_response.json'); marker = self.read(path / 'result.json')
            raw = self.track(path / 'raw_response.txt').read_text()
            if (content_hash(response) != call['response_content_hash'] or marker['parsed_response_hash'] != content_hash(response)
                    or marker['raw_response_hash'] != content_hash(raw)):
                raise ValueError('Source response changed: ' + str(path))
            identity = self.read(path / 'program_transport_identity.json')
            if not identity['verified'] or identity['account'] != account:
                raise ValueError('Unverified physical route: ' + str(path))
        return path, schema

    # ---------------------------------------------------------------- membership and evidence
    def holds(self):
        if not self.hold_path:
            return {}
        self.track(self.hold_path)
        return read_holds(self.hold_path)

    def weakness_holds(self, report):
        """Weakness run (2026-10-20, DESIGN §6.9): JUDGE decoys are planted negatives and never committed; JUDGE tasks still
        pending their offline consensus reads are held unless listed in --release-judge (the tasks that passed). Empty for
        every other run."""
        released = set()
        if self.release_path:
            self.track(self.release_path)
            released = set(read_holds(self.release_path))
        out = {}
        for r in report['tasks']:
            weakness = (r.get('blind_performance') or {}).get('weakness') or {}
            if weakness.get('judge_decoy'):
                out[r['task_id']] = 'a JUDGE decoy (planted negative for the consensus procedure)'
            elif weakness.get('line_tag') == 'pending_consensus' and r['task_id'] not in released:
                out[r['task_id']] = 'a JUDGE task awaiting its offline consensus reads'
        return out

    def select_rows(self, report):
        holds = self.holds()
        if self.hold_subset:  # one hold file for a transaction of several runs: this run's part (the cli checks the rest)
            holds = {k: v for k, v in holds.items() if k in {r['task_id'] for r in report['tasks']}}
        unknown = set(holds) - {r['task_id'] for r in report['tasks']}
        if unknown:
            raise ValueError('Held tasks are not in this run: ' + ', '.join(sorted(unknown)))
        holds = {**self.weakness_holds(report), **holds}
        rows = sorted((r for r in report['tasks'] if r['usable_after_exact_palette_union'] and r['task_id'] not in holds),
                      key=lambda r: r['task_id'])
        if len({r['task_id'] for r in report['tasks']}) != len(report['tasks']) or not rows:
            raise ValueError('Terminal report membership is inconsistent')
        if self.partial:
            finished = {w['window'] for w in report['windows'] if w['terminal']}
            terminals = [Path(r['state']['path']).parent / 'window_terminal.private.json' for r in rows]
            if (report['status'] != 'stopped_partial' or not all(r['verified_before_union'] and r['window'] in finished for r in rows)
                    or not all(p.is_file() for p in terminals)):
                raise ValueError('A partial commit takes only verified tasks of finished windows')
            for path in terminals:
                self.track(path)
        return rows

    @lru_cache(None)
    def research_disabled(self):
        index = self.read(self.analysis / 'research_index.private.json')
        if any(r.get('research_status') != 'disabled_by_design' or r.get('research') for r in index.values()):
            raise ValueError('Disabled research policy changed')
        return True

    @lru_cache(None)
    def verify_window(self, namespace):
        self.track(Path(namespace) / 'editorial_bypass.private.json')
        return verify_bypass(SimpleNamespace(state=self.state(namespace), run_dir=Path(namespace)))

    def verify_candidate(self, row):
        namespace = Path(row['state']['path']).parent
        saved = self.state(namespace); record = saved['semantic_records'][row['task_id']]
        if sha256((namespace / 'state.json').read_bytes()).hexdigest() != row['state']['sha256']:
            raise ValueError('Reviewed source state changed')
        self.verify_window(namespace)
        validation = validate_arc_task(record['complete_task'], require_test_outputs=True)
        final = project_final_evidence(row['task_id'], record, list(saved['call_ledger'].values()))
        if (not validation.valid or not strict_technical_clear(record) or not record['deterministic_diagnostics']['hard_valid']
                or not record['plan_fidelity']['passed'] or final['status'] not in ['evidence_clear', 'evidence_clear_after_review']
                or record['boardless_disposition'] != 'verified_candidate' or record.get('intrinsic_quality') is not None
                or not record.get('blind_results') or record.get('pipeline_usable_before_union_audit') is not True):
            raise ValueError('Candidate does not meet unchanged final evidence gates: ' + row['task_id'])
        terminal = verify_terminal_artifact_bundle({'_output_directory': str(namespace.parent)}, record['terminal_artifact'], record)
        self.research_disabled()
        return record, {'task_id': row['task_id'], 'final_evidence': final, 'terminal': terminal,
                        'original_record_hash': content_hash(record), 'research_status': 'disabled_by_design',
                        'editorial_mode': 'bypass', 'quality_grade': 'unassigned', 'research_is_not_production_evidence': True}

    # ---------------------------------------------------------------- dossier
    def package(self, row, record, analysis):
        tid = row['task_id']; namespace = Path(row['state']['path']).parent; window = namespace.parent
        ledger = self.state(namespace)['call_ledger'].values()
        planner = planner_call(ledger, tid)
        realizer = [c for c in ledger if c['purpose'] == 'realization' and tid in c['parent_ids']]
        entry = {'registry_task_id': 'e18-production__' + tid, 'title': _title(record, tid),
                 'source_run': rel(namespace), 'source_candidate_id': tid,
                 'source_pipeline': 'arcgen-v1-authorized-commit', 'search_mode': record['search_mode'],
                 'intrinsic_status': 'boardless_verified', 'source_intrinsic_status': None, 'quality_grade': 'unassigned',
                 'editorial_mode': 'bypass', 'boardless_disposition': 'verified_candidate', 'answer_mode': 'single_answer',
                 'portfolio_status': 'include_now',
                 'bank_grade_basis': 'Explicit user commitment of final evidence-clear boardless task; no editorial grade.',
                 'commit_set_id': self.name, 'concept_summary': _concept_summary(record), 'intended_rule': _intended_rule(record),
                 'study': deepcopy(record['study']), 'primary_arm': record['study']['primary_arm'],
                 'invitation': record['study']['invitation'], 'bank_setting': record['study']['bank_setting'],
                 'source_episode_hash': content_hash(record['complete_task']),
                 'role_bindings': {'planner': {'model': planner['model'], 'effort': planner['reasoning_effort'], 'account': planner['account'],
                                               'planner': record['study'].get('planner')},
                                   'selector': None,
                                   'realizer': {'model': realizer[0]['model'], 'effort': realizer[0]['reasoning_effort']} if realizer else None},
                 'project_stage': self.flag,
                 'route': {'route': record.get('private_route'), 'overlay': record['study'].get('overlay'),
                           'colour_instruction': record['study'].get('colour_roles')},
                 'difficulty_retention_policy': 'grounded-difficulty-retention-v1',
                 'blind_performance': deepcopy(row['blind_performance']), 'generation_model': row['generation_model'],
                 'generation_effort': row['effort'], 'cohort': row['cohort'],
                 'admission_watch': bool(row.get('evidence_consistency_watch'))}
        rid = entry['registry_task_id']; entry['task_path'] = 'tasks/' + rid + '.json'
        self.copy(namespace / 'tasks' / (tid + '.arc.json'), self.stage / entry['task_path'])
        entry['arc_sha256'] = sha256((self.stage / entry['task_path']).read_bytes()).hexdigest()
        entry['dossier_path'] = 'dossiers/' + rid + '/manifest.json'
        destination = self.stage / 'dossiers' / rid
        for path in [namespace / 'tasks' / (tid + '.authoring.json'), namespace / 'report.json', *[namespace / f for f in CONTEXT_FILES]]:
            if path.is_file():
                self.track(path)
        with fast.patched(committed_dossiers, shutil=_LinkingShutil(self)):  # its model-call folders: linked records
            manifest = _build_one(repository_root=ROOT, committed_root=self.stage, entry=entry, destination=destination)
        calls = {c['log_dir']: c for c in ledger if tid in c.get('parent_ids', [])}
        source_call = record['private_source_call_id']
        upstream, original_window, idea_window = self.upstream(namespace, record)
        ideas = self.idea_calls(upstream)
        idea = [ideas.get(source_call + '-sketch'), ideas.get(source_call + '-commit')]
        if None in idea or idea[1]['parent_ids'] != [idea[0]['call_id']]:
            raise ValueError('Idea loop lineage incomplete: ' + tid)
        explore = ideas.get(source_call + '-explore')  # explore calls have a middle turn, also from the sketch
        if explore is not None:
            if explore['parent_ids'] != [idea[0]['call_id']]:
                raise ValueError('Idea loop lineage incomplete: ' + tid)
            idea.append(explore)
        for c in idea:
            calls[c['log_dir']] = c
        selectors = list(self.state(original_window / 'selector')['call_ledger'].values())
        if len(selectors) != 1 or selectors[0]['stage'] != 'arm_bank_selector_v1':
            raise ValueError('Selector lineage changed')
        calls[selectors[0]['log_dir']] = selectors[0]
        entry['role_bindings']['selector'] = {'model': selectors[0]['model'], 'effort': selectors[0]['reasoning_effort']}
        source_row = self.read(original_window / 'source_rows.private.json')[record['source_candidate_id']]
        loop_file = idea_window / 'idea_loop' / (source_call + '.committed.private.json')
        committed = self.read(loop_file)['ideas']
        kernel, = [k for k in committed['ideas'] if k['candidate_id'] == record['source_candidate_id']]
        if (source_row['kernel'] != record['kernel'] or kernel != record['kernel'] or source_row['private_source_call_id'] != source_call
                or idea[0]['reasoning_effort'] != source_row['private_effort'] or idea[0]['model'] != row['generation_model']
                or original_window.name != record['study']['original_selector_window']):
            raise ValueError('Original generation provenance changed: ' + tid)
        inventory = []
        for call in calls.values():
            path, schema = self.call_assets(call)
            if path.is_relative_to(namespace / 'model_calls'):
                target = destination / 'model_calls' / path.relative_to(namespace / 'model_calls')
            else:
                target = destination / 'model_calls' / path.parent.name / (content_hash(str(path))[:12] + '__' + path.name)
            self.link_tree(path, target)
            self.copy(schema, target / 'schema.snapshot.json')
            atomic_write_json(target / 'call_ledger.snapshot.json', call)
            atomic_write_json(target / 'commit_binding.json', {'task_id': tid, 'original_source_path': rel(path),
                                                               'author_only_post_run_annex': True, 'model_input_or_response_changed': False})
            inventory.append({'path': rel(path), 'stored_path': str(target.relative_to(destination)), 'stage': call['stage'],
                              'role': call['role'], 'account': call['account'], 'model': call['model'], 'effort': call['reasoning_effort']})
        stages = Counter(c['stage'] for c in calls.values())
        if (stages['v1_idea_sketch'] != 1 or stages['v1_idea_commit'] + stages['v1_idea_commit_kept'] != 1 or stages['arm_bank_selector_v1'] != 1
                or any(c['purpose'] == 'editorial_board' for c in calls.values())
                or not any(c['role'] == 'concrete_planner' for c in calls.values()) or not any('realizer' in s for s in stages)
                or not any(c['role'] == 'technical_verifier' for c in calls.values()) or not stages['v4_blind_solver']
                or any(c['purpose'].startswith('completed_research') for c in calls.values())):
            raise ValueError('Incomplete boardless call lineage: ' + tid)
        context = destination / 'run_context'
        for filename in ['source_rows.private.json', 'allocation.private.json', 'window.private.json']:
            self.copy(window / filename, context / 'window' / filename)
        for filename in ['source_rows.private.json', 'selector_inputs.private.json', 'eligible_queue.private.json', 'window.private.json']:
            self.copy(original_window / filename, context / 'original_selector_window' / filename)
        for name in (source_call + '.private.json', source_call + '.committed.private.json'):
            self.copy(idea_window / 'idea_loop' / name, context / 'idea_loop' / name)
        self.copy(namespace / 'editorial_bypass.private.json', context / 'editorial_bypass.private.json')
        self.link_tree(Path(record['terminal_artifact']['artifact_manifest']).parent, context / 'terminal')
        terminal = ([self.out / 'partial_source/partial_receipt.json', self.run / 'STATUS.json'] if self.partial
                    else [self.run / 'terminal_verification.json'])
        for source in [self.run / 'launch_manifest.private.json', self.run / 'config.json', *terminal, self.run / 'prepared_manifest.private.json']:
            self.copy(source, context / 'run_provenance' / source.name)
        if upstream != self.run:  # generated in an earlier run (saved queue or saved ideas): that run's records too
            for source in [upstream / 'launch_manifest.private.json', upstream / 'config.json']:
                self.copy(source, context / 'source_run_provenance' / source.name)
            selector_spec = self.read(original_window / 'window.private.json')['window']
            if selector_spec.get('kind') == 'saved_ideas':
                self.copy(ROOT / selector_spec['saved_ideas'], context / 'saved_ideas.json')
        atomic_write_json(context / 'analysis.private.json', analysis)
        atomic_write_json(context / 'commit_authorization.json', {
            'user_message': self.user_message, 'authorization': self.authorization(), 'quality_grade': 'unassigned',
            'source_record_hash': content_hash(record), 'original_record_and_judgments_unchanged': True,
            'production_calls_added': 0, 'research_is_not_production_evidence': True})
        provenance = read_json(destination / 'provenance.json')
        provenance['commit_supplement'] = {'call_inventory': inventory, 'authorization': 'run_context/commit_authorization.json',
                                           'saved_artifact_analysis': 'run_context/analysis.private.json',
                                           'source_state_sha256': row['state']['sha256'], 'original_review_verbatim': True}
        atomic_write_json(destination / 'provenance.json', provenance)
        manifest['files'] = _manifest_files(destination)
        manifest['artifact_policy']['additional_context'] = ('Exact saved calls, schemas, ledgers, idea loop (draft, nearest-bank '
                                                             'check, commit), selector, bypass and run receipts; author-only. Research disabled by design.')
        atomic_write_json(destination / 'manifest.json', manifest)
        entry['research_status'] = 'disabled_by_design'
        entry['dossier_sha256'] = sha256((destination / 'manifest.json').read_bytes()).hexdigest()
        return entry

    # ---------------------------------------------------------------- staging
    def verify_sources(self):
        files = read_json(self.out / 'source_manifest.json')['files']
        for (relative, expected), actual in zip(files.items(), fast.hash_each([ROOT / r for r in files])):
            if isinstance(actual, BaseException):
                raise actual
            if actual != expected:
                raise ValueError('Frozen source changed: ' + relative)

    def partial_source(self, status):
        """A stopped run: its finished windows reported into this commit's folder from ledger copies; the run is only read."""
        if not idle(self.run):
            raise ValueError('The run may be alive: its runner lock is held or a ledger has a write-ahead log')
        source = self.out / 'partial_source'
        if source.exists():
            shutil.rmtree(source)
        for path in [Path(__file__), self.run / 'STATUS.json', self.run / 'prepared_manifest.private.json',
                     self.run / 'launch_manifest.private.json', *[p for p in [self.run / 'STOP_DISPATCH.json'] if p.exists()]]:
            self.track(path)
        from .batch_report import report as batch_report
        value = batch_report(self.run, out=source)
        for row in value['ledger_snapshot']:
            if self.sources[rel(self.track(row['source']['path']))] != row['source']['sha256']:
                raise ValueError('A ledger changed while the partial report was built')
        finished = {w['window'] for w in value['windows'] if w['terminal']}
        self.unfinished = {t['task_id'] for t in value['tasks'] if t['window'] not in finished}
        self.partial_record = {'partial': True, 'run_status': status['status'], 'stop_reason': status.get('reason'),
                               'windows': {'total': len(value['windows']), 'finished': len(finished),
                                           'unfinished': len(value['windows']) - len(finished)},
                               'unfinished_task_ids': sorted(self.unfinished)}
        atomic_write_json(source / 'partial_receipt.json', {
            **self.partial_record, 'unfinished_windows': sorted(w['window'] for w in value['windows'] if not w['terminal']),
            'terminal_report_sha256': sha256((source / 'analysis/terminal_report.private.json').read_bytes()).hexdigest(),
            'ledger_snapshot': value['ledger_snapshot'], 'run_written': False, 'new_model_calls': 0})
        self.track(source / 'partial_receipt.json')
        return self.read(source / 'analysis/terminal_report.private.json'), self.read(source / 'analysis/research_index.private.json')

    def protected_files(self):
        """Bank and reservoir files a commit must leave untouched: audit flags, quarantine, reservoir, basic registry."""
        protected = {rel(p): sha256(p.read_bytes()).hexdigest() for p in (BANK / 'audit_flags').glob('*.json')}
        for k in ['quarantine_registry', 'basic_valid_registry']:
            path = Path(self.cfg()['_paths'][k]); protected[rel(path)] = sha256(path.read_bytes()).hexdigest()
        return protected

    def base(self):
        return StagedBase(self) if self.after else BankBase(self)

    def legacy_prepared(self, prepared):
        """A commit folder of the previous commit code: an interrupted install is rolled back exactly as that code would
        before retrying; then it must be staged again (cli does so when nothing was installed)."""
        journal_path = self.out / 'transaction.json'
        journal = read_json(journal_path) if journal_path.exists() else None
        if journal and journal.get('status') == 'installing':
            with _committed_bank_lock(self.cfg()):
                journal = read_json(journal_path)
                if journal.get('status') == 'installing':
                    _verify_backup(self.out / 'backup', journal['backup_manifest_sha256'])
                    _rollback(self.cfg(), self.out / 'backup', prepared)
                    journal.update(status='rolled_back', error='install of the previous commit code was interrupted; rolled back')
                    atomic_write_json(journal_path, journal)
            raise ValueError(f'runs/{self.name}: an interrupted install of the previous commit code was rolled back; '
                             f'delete runs/{self.name} and commit again')
        if journal:
            raise ValueError(f'runs/{self.name} was committed by the previous commit code (transaction {journal.get("status")})')
        raise ValueError(f'runs/{self.name} was prepared by the previous commit code; delete it to stage again')

    def prepare(self):
        if (self.out / 'prepared.json').exists():
            prepared = read_json(self.out / 'prepared.json')
            if prepared.get('format') != FORMAT:
                self.legacy_prepared(prepared)
            if prepared['transaction']['after'] != self.after:
                raise ValueError(f'runs/{self.name} was prepared in another commit transaction; delete it to stage again')
            if self.after and prepared['transaction']['base_prepared_sha256'] != digest(ROOT / 'runs' / self.after / 'prepared.json'):
                raise ValueError(f'The prepared commit {self.after} changed after {self.name} was stacked on it; stage both again')
            self.verify_sources(); _verify_staging(self.stage, self.name)
            return prepared
        self.out.mkdir(parents=True, exist_ok=True)
        if self.stage.exists():
            raise ValueError('Incomplete staging exists; inspect before rebuilding')
        status = self.read(self.run / 'STATUS.json')
        if status['status'] != ('stopped' if self.partial else 'verified_terminal'):
            raise ValueError('The run is not ' + ('stopped' if self.partial else 'verified_terminal'))
        if self.partial:
            report, analysis = self.partial_source(status)
        else:
            for path in [Path(__file__), self.run / 'budget.sqlite3', self.run / 'terminal_verification.json',
                         self.run / 'completion.json', self.run / 'prepared_manifest.private.json', self.run / 'launch_manifest.private.json']:
                self.track(path)
            report = self.read(self.run / 'analysis/terminal_report.private.json')
            analysis = self.read(self.run / 'analysis/research_index.private.json')
        selected = self.select_rows(report)
        chosen = {r['task_id'] for r in selected}
        holds = {**self.weakness_holds(report), **self.holds()}
        excluded = [{**{k: r[k] for k in ['task_id', 'status', 'evidence_clear', 'arm']},
                     'reason': ('held: ' + holds[r['task_id']]) if r['task_id'] in holds else
                     'window_unfinished' if r['task_id'] in self.unfinished else r.get('review_reason') or r['status']}
                    for r in report['tasks'] if r['task_id'] not in chosen]
        print(f'Verifying the bank and saved evidence for {len(selected)} candidates...', flush=True)
        base = self.base()
        raw, baseline, protected, palettes = base.raw, base.json_hashes, base.protected, base.palettes()
        atomic_write_json(self.out / 'authorization.json', {'user_message': self.user_message, 'authorization': self.authorization(),
                                                            'task_ids': sorted(chosen), 'excluded': excluded})
        atomic_write_json(self.out / 'excluded.private.json', excluded)
        entries, proof = [], []
        for i, row in enumerate(selected, 1):
            record, verification = self.verify_candidate(row)
            palette = palette_equivalent_hash(record['complete_task'])
            if palette in palettes:
                raise ValueError('Exact/palette duplicate: ' + row['task_id'] + ' / ' + palettes[palette])
            palettes[palette] = row['task_id']
            entries.append(self.package(row, record, analysis[row['task_id']]))
            proof.append({**verification, 'exact_palette_duplicate': False})
            if i % 10 == 0 or i == len(selected):
                print(f'Packaged {i}/{len(selected)} complete dossiers', flush=True)
        for entry in entries:  # the index the next commit of a transaction checks against: registry IDs, as in the bank
            palettes[palette_equivalent_hash(read_json(self.stage / entry['task_path']))] = entry['registry_task_id']
        commit_set = {'commit_set_id': self.name, 'committed_on': self.committed_on, 'project_stage': self.flag,
                      'authorization': self.authorization(), 'entry_count': len(entries), 'notes': self.authorization(), **self.partial_record}
        new = deepcopy(raw); new['entries'] = entries; new['commit_sets'] = [commit_set]; new.pop('active_bank_filter', None)
        atomic_write_json(self.stage / 'new_registry.private.json', new)
        verified_new = load_committed_task_registry(self.stage / 'new_registry.private.json')
        merged = deepcopy(raw); merged['entries'].extend(entries); merged['commit_sets'].append(commit_set)
        atomic_write_json(self.stage / 'registry.json', merged)
        atomic_write_json(self.stage / 'active_registry.json', filter_active_registry(merged, bank=BANK))
        _write_catalog(self.stage, merged); _write_checksums(self.stage, merged)
        renders = self.stage_renders(base, merged, base.view_entries + verified_new['entries'], len(entries))
        manifest = {'schema_version': 1, **commit_set, 'authorization_record': self.authorization(),
                    'source_run': rel(self.run), 'entries': entries, 'excluded': excluded,
                    'new_model_calls': 0, 'source_pipeline_changed': False, 'quality_grade': 'unassigned', 'project_stage': self.flag}
        atomic_write_json(self.stage / 'commit_manifests' / (self.name + '.json'), manifest)
        atomic_write_json(self.out / 'proof.private.json', proof)
        atomic_write_json(self.out / 'palette_index.private.json', palettes)
        atomic_write_json(self.out / 'dossier_links.json', {'linked': sorted(self.linked), 'counts': dict(sorted(self.link_stats.items()))})
        atomic_write_json(self.out / 'source_manifest.json', {'files': self.sources})
        atomic_write_json(self.stage / 'staging_manifest.json', _staging_manifest(self.stage, self.name))
        prepared = {'main_task_ids': [e['source_candidate_id'] for e in entries], 'basic_task_ids': [],
                    'old_bank_json_hashes': baseline, 'protected': protected,
                    'staging_manifest_sha256': sha256((self.stage / 'staging_manifest.json').read_bytes()).hexdigest(),
                    'source_manifest_sha256': sha256((self.out / 'source_manifest.json').read_bytes()).hexdigest(),
                    'authorization_sha256': sha256((self.out / 'authorization.json').read_bytes()).hexdigest(),
                    'before_historical': len(raw['entries']), 'after_historical': len(merged['entries']),
                    'after_active': len(read_json(self.stage / 'active_registry.json')['entries']),
                    'format': FORMAT, 'run': rel(self.run), 'stage_flag': self.flag, 'partial': self.partial,
                    'committed_on': self.committed_on,
                    'transaction': {'after': self.after, 'chain': base.chain, 'base_prepared_sha256': base.prepared_sha256,
                                    'defer_sheet': self.defer_sheet},
                    'renders': renders,
                    'palette_index_sha256': digest(self.out / 'palette_index.private.json'),
                    'dossier_links_sha256': digest(self.out / 'dossier_links.json'),
                    'dossier_links': dict(sorted(self.link_stats.items())),
                    'bank_dossiers_hashed_at_prepare': base.head and not self.then_install}
        if not self.then_install:  # an install follows: it re-verifies sources and staging under the bank lock first
            self.verify_sources(); _verify_staging(self.stage, self.name)
        atomic_write_json(self.out / 'prepared.json', prepared)
        print({'status': 'prepared', 'tasks': len(entries), 'excluded': len(excluded), 'project_stage': self.flag,
               'renders': renders['mode'], 'drawn': renders['drawn'], 'dossier_links': self.link_stats.get('linked', 0)}, flush=True)
        return prepared

    # ---------------------------------------------------------------- renders
    def render_mode(self, old_draws):
        """Draw only the new tasks when the bank's renders are exactly those its render index lists (every kept render
        hash-checked) and the renderer still draws a spread sample of them byte for byte; else draw the whole bank."""
        index = fast.read_render_index(BANK)
        if index is None:
            return 'full', 'the bank has no render index yet (first commit with incremental renders)'
        folder = BANK / 'renders'
        expected = {d['name'] for d in old_draws} | set(fast.SHARED_RENDER_FILES)
        actual = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
        if set(index) != expected or actual != expected:
            return 'full', 'the render index or render folder does not match the registry'
        hashes = fast.hash_files([folder / d['name'] for d in old_draws])
        changed = [p.name for p, h in hashes.items() if h != index[p.name]]
        if changed:
            return 'full', f'{len(changed)} renders differ from the render index (first {changed[0]})'
        n = len(old_draws); sample = [old_draws[i] for i in sorted({round(k * (n - 1) / 5) for k in range(6)})] if n else []
        with tempfile.TemporaryDirectory(dir=self.out) as scratch:
            fast.draw_renders(sample, scratch, processes=1)
            redrawn = [d['name'] for d in sample if (Path(scratch) / d['name']).read_bytes() != (folder / d['name']).read_bytes()]
        if redrawn:
            return 'full', f'the renderer now draws {redrawn[0]} differently'
        return 'incremental', None

    def stage_renders(self, base, merged, view_entries, new_count):
        """The renders of this commit: the canonical _render_bank's manifest and README (the last commit of a transaction
        only), the drawings of the new tasks (all tasks when the bank's renders cannot be kept), and for the last commit
        the contact sheet and the render index of the final bank."""
        folder = self.stage / 'renders'
        plan_root = self.stage if not self.defer_sheet else Path(tempfile.mkdtemp(prefix='.render-plan.', dir=self.out))
        draws, images = fast.plan_renders(plan_root, {**merged, 'entries': view_entries})
        if self.defer_sheet:
            shutil.rmtree(plan_root)
        old_draws, new_draws = draws[:len(draws) - new_count], draws[len(draws) - new_count:]
        if base.head:
            mode, reason = self.render_mode(old_draws)
            basis = {'bank_index_sha256': digest(BANK / fast.RENDER_INDEX) if (BANK / fast.RENDER_INDEX).is_file() else None,
                     'bank_render_files': sorted(p.name for p in (BANK / 'renders').iterdir()) if (BANK / 'renders').is_dir() else []}
            atomic_write_json(self.out / 'render_basis.json', basis)
            record = {'mode': mode, 'reason': reason, 'head': self.name, 'basis_sha256': digest(self.out / 'render_basis.json'),
                      'bank_index_sha256': basis['bank_index_sha256']}
        else:
            record = {k: base.renders[k] for k in ('mode', 'reason', 'head', 'basis_sha256', 'bank_index_sha256')}
        jobs = draws if record['mode'] == 'full' and base.head else new_draws
        print(f"Drawing {len(jobs)} renders ({record['mode']}{': ' + record['reason'] if record['reason'] else ''})...", flush=True)
        fast.draw_renders(jobs, folder)
        if not self.defer_sheet:
            folders = [folder] + [stage / 'renders' for stage in reversed(base.stages)]
            staged_names = {f: {p.name for p in f.iterdir()} if f.is_dir() else set() for f in folders}

            def where(name):  # a render drawn in this transaction (newest stage first), else the bank's own
                return next((f / name for f in folders if name in staged_names[f]), BANK / 'renders' / name)
            print('Making the contact sheet of the bank...', flush=True)
            fast.contact_sheet([(label, where(name)) for label, name in images], folder / 'contact_sheet.png')
            staged_names[folder].add('contact_sheet.png')
            expected = {n: where(n) for n in [d['name'] for d in draws] + list(fast.SHARED_RENDER_FILES)}
            kept = fast.read_render_index(BANK) if record['mode'] == 'incremental' else {}
            staged = fast.hash_files([p for p in expected.values() if p.parent != BANK / 'renders'])
            index = {n: staged[p] if p in staged else kept[n] for n, p in expected.items()}
            atomic_write_text(self.stage / fast.RENDER_INDEX, fast.format_render_index(index))
        return {**record, 'drawn': len(jobs), 'final': not self.defer_sheet}

    # ---------------------------------------------------------------- install
    def verify_committed(self, prepared):
        return Transaction([self]).verify([prepared])[0]

    def commit(self, prepared):
        return Transaction([self]).run([prepared])[0]


class BankBase:
    """The bank as the starting point of a commit (the first, or only, commit of a transaction)."""
    head, chain, prepared_sha256, stages, renders = True, [], None, [], None

    def __init__(self, commit):
        deep = not commit.then_install  # a dry run hash-verifies every dossier now; an install does it on the final bank
        old = fast.load_registry(BANK / 'registry.json', deep=deep)
        self.basic = load_basic_valid_registry(BANK / 'basic_valid_registry.json')
        self.raw = read_json(BANK / 'registry.json')
        self.json_hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in BANK.glob('*.json')}
        self.protected = commit.protected_files()
        self.view_entries = old['entries']

    def palettes(self):
        return {palette_equivalent_hash(read_json(Path(e['absolute_task_path']))): e['registry_task_id']
                for e in self.view_entries + self.basic['entries']}


class StagedBase:
    """A prepared, not yet installed commit of the same transaction as the starting point of the next: its staged
    registry, its palette index (bank and every earlier commit of the transaction) and the transaction's render basis.
    The bank itself must still be what the transaction's first commit verified."""
    head = False

    def __init__(self, commit):
        before = ROOT / 'runs' / commit.after
        prepared = read_json(before / 'prepared.json')
        if prepared.get('format') != FORMAT or prepared['transaction']['defer_sheet'] is not True:
            raise ValueError(f'{commit.after} is not a prepared commit to stack on (prepare it with --defer-sheet)')
        if (before / 'transaction.json').exists():
            raise ValueError(f'{commit.after} has already started installing; it cannot take more runs')
        self.prepared_sha256 = digest(before / 'prepared.json')
        self.chain = prepared['transaction']['chain'] + [commit.after]
        first = read_json(ROOT / 'runs' / self.chain[0] / 'prepared.json')
        if {p.name: sha256(p.read_bytes()).hexdigest() for p in BANK.glob('*.json')} != first['old_bank_json_hashes']:
            raise ValueError('The bank changed since the first commit of this transaction was prepared; stage them again')
        self.protected = commit.protected_files()
        if self.protected != first['protected']:
            raise ValueError('Protected bank/reservoir state changed since the first commit of this transaction was prepared')
        stage = before / 'staged_bank'
        self.raw = read_json(stage / 'registry.json')
        self.json_hashes = {**prepared['old_bank_json_hashes'], 'registry.json': digest(stage / 'registry.json'),
                            'active_registry.json': digest(stage / 'active_registry.json')}
        if digest(before / 'palette_index.private.json') != prepared['palette_index_sha256']:
            raise ValueError('Palette index of ' + commit.after + ' changed')
        self.index = read_json(before / 'palette_index.private.json')
        self.stages = [ROOT / 'runs' / name / 'staged_bank' for name in self.chain]
        homes = {name: ROOT / 'runs' / name / 'staged_bank' for name in self.chain}
        self.view_entries = [{**e, 'absolute_task_path': str((homes.get(e['commit_set_id'], BANK) / e['task_path']).resolve())}
                             for e in self.raw['entries']]
        self.renders = prepared['renders']

    def palettes(self):
        return dict(self.index)


class Transaction:
    """Install one or more prepared commits as one bank transaction: one lock, one journal (in the first commit's
    folder), one backup of exactly what the install replaces, the bank-wide files and renders written once, one full
    verification of the final bank (every dossier hashed), and per run the commit set, manifest and report that separate
    commits would leave. If anything fails, the bank is restored to its state before the transaction."""

    def __init__(self, members):
        self.members = list(members)
        self.head, self.last = self.members[0], self.members[-1]
        self.journal_path = self.head.out / 'transaction.json'
        self.names = [c.name for c in self.members]

    def cfg(self):
        return self.head.cfg()

    def write_journal(self, journal):
        atomic_write_json(self.journal_path, journal)
        for c in self.members[1:]:  # each run's folder says how its commit ended (cli prunes complete ones)
            atomic_write_json(c.out / 'transaction.json', {'format': FORMAT, 'status': journal['status'], 'master': self.head.name,
                                                          **({'error': journal['error']} if journal.get('error') else {})})

    def check(self, prepared):
        if len(prepared) != len(self.members) or len(set(self.names)) != len(self.names):
            raise ValueError('Prepared transaction membership changed')
        for i, (c, p) in enumerate(zip(self.members, prepared)):
            if p.get('format') != FORMAT:
                raise ValueError(f'runs/{c.name} was prepared by the previous commit code; delete it and commit again')
            t = p['transaction']
            if (t['after'] != (self.names[i - 1] if i else None) or t['chain'] != self.names[:i]
                    or t['defer_sheet'] != (c is not self.last)):
                raise ValueError('Prepared transaction order changed: ' + c.name)
            if i and t['base_prepared_sha256'] != digest(self.members[i - 1].out / 'prepared.json'):
                raise ValueError('A prepared commit this one stacks on changed: ' + c.name)
            entries = read_json(c.stage / 'new_registry.private.json')['entries']
            if p['main_task_ids'] != [e['source_candidate_id'] for e in entries] or p['basic_task_ids']:
                raise ValueError('Prepared task membership changed')
        if any(p['renders']['head'] != self.head.name for p in prepared) or prepared[-1]['renders']['final'] is not True:
            raise ValueError('Prepared render plan changed')

    def check_identity(self, prepared):
        for c, p in zip(self.members, prepared):
            if (digest(c.stage / 'staging_manifest.json') != p['staging_manifest_sha256']
                    or digest(c.out / 'source_manifest.json') != p['source_manifest_sha256']
                    or digest(c.out / 'authorization.json') != p['authorization_sha256']
                    or digest(c.out / 'palette_index.private.json') != p['palette_index_sha256']
                    or digest(c.out / 'dossier_links.json') != p['dossier_links_sha256']):
                raise ValueError('Prepared commit identity changed')
            c.verify_sources(); _verify_staging(c.stage, c.name)

    def check_bank(self, prepared):
        """The bank is still what the first commit was prepared on; with incremental renders, the renders it keeps are
        still the files its render index lists, byte for byte."""
        for name, expected in prepared[0]['old_bank_json_hashes'].items():
            if not (BANK / name).is_file() or digest(BANK / name) != expected:
                raise ValueError('Bank changed since preparation: ' + name)
        if {p.name for p in BANK.glob('*.json')} != set(prepared[0]['old_bank_json_hashes']):
            raise ValueError('Bank changed since preparation: its registry files')
        renders = prepared[0]['renders']; basis_path = self.head.out / 'render_basis.json'
        if digest(basis_path) != renders['basis_sha256']:
            raise ValueError('Render basis changed')
        index_file = BANK / fast.RENDER_INDEX
        if (digest(index_file) if index_file.is_file() else None) != renders['bank_index_sha256']:
            raise ValueError('Bank changed since preparation: ' + fast.RENDER_INDEX)
        if renders['mode'] == 'incremental':
            basis, index = read_json(basis_path), fast.read_render_index(BANK)
            folder = BANK / 'renders'
            if sorted(p.name for p in folder.iterdir()) != basis['bank_render_files']:
                raise ValueError('Bank renders changed since preparation')
            kept = [folder / n for n in basis['bank_render_files'] if n not in fast.SHARED_RENDER_FILES]
            for path, value in fast.hash_files(kept).items():
                if value != index[path.name]:
                    raise ValueError('Bank render changed since preparation: ' + path.name)

    def staged_renders(self):
        files = {}
        for c in self.members:  # later commits' files win; only the last stages the manifest, README and sheet
            folder = c.stage / 'renders'
            if folder.is_dir():
                files.update({p.name: p for p in folder.iterdir() if p.is_file()})
        return files

    def new_registry_ids(self):
        return [e['registry_task_id'] for c in self.members for e in read_json(c.stage / 'new_registry.private.json')['entries']]

    def make_backup(self, backup):
        folder = BANK / 'renders'
        before = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
        staged = set(self.staged_renders())
        final = set(fast.read_render_index(self.last.stage))
        if not staged <= final:
            raise ValueError('Staged renders outside the final render index')
        if backup.exists():
            shutil.rmtree(backup)
        temporary = Path(tempfile.mkdtemp(prefix='.backup.building.', dir=self.head.out))
        try:
            fast.backup(BANK, temporary, manifests=['commit_manifests/' + n + '.json' for n in self.names],
                        replaced_renders=sorted((before & staged) | (before - final)), added_renders=sorted(staged - before),
                        new_registry_ids=self.new_registry_ids())
            atomic_write_json(temporary / 'backup_manifest.json', _backup_manifest(temporary))
            temporary.replace(backup)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def install(self):
        for c in self.members:
            links = set(read_json(c.out / 'dossier_links.json')['linked'])
            for entry in read_json(c.stage / 'new_registry.private.json')['entries']:
                task = BANK / entry['task_path']; dossier = BANK / 'dossiers' / entry['registry_task_id']
                if task.exists() or dossier.exists():
                    raise ValueError('Commit destination already exists')
                shutil.copy2(c.stage / entry['task_path'], task)
                prefix = 'dossiers/' + entry['registry_task_id'] + '/'
                fast.install_tree(c.stage / 'dossiers' / entry['registry_task_id'], dossier,
                                  {p[len(prefix):] for p in links if p.startswith(prefix)})
        for name in BANK_METADATA:
            fast.place(self.last.stage / name, BANK / name, link=False)
        for c in self.members:
            name = 'commit_manifests/' + c.name + '.json'
            fast.place(c.stage / name, BANK / name, link=False)
        final = set(fast.read_render_index(BANK))
        for name, source in sorted(self.staged_renders().items()):
            fast.place(source, BANK / 'renders' / name, link=True)
        for path in sorted((BANK / 'renders').iterdir()):
            if path.name not in final:
                path.unlink()

    def verify(self, prepared):
        """The installed bank: metadata as staged, active partition, protected files, every dossier and task of the final
        registry hash-verified (once), catalog/checksums/renders exact, every render file as the render index says, and
        per run its membership, project stage, commit set, original review records and task bytes."""
        for c in self.members:
            c.verify_sources()
        for name in BANK_METADATA:
            if digest(BANK / name) != digest(self.last.stage / name):
                raise ValueError('Installed bank metadata differs: ' + name)
        for c in self.members:
            name = 'commit_manifests/' + c.name + '.json'
            if digest(BANK / name) != digest(c.stage / name):
                raise ValueError('Installed bank metadata differs: ' + name)
        historical = read_json(BANK / 'registry.json'); active = read_json(BANK / 'active_registry.json')
        if active != filter_active_registry(historical, bank=BANK):
            raise ValueError('Active bank differs from the verified historical partition')
        for relative, expected in prepared[0]['protected'].items():
            if sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Protected bank/reservoir state changed: ' + relative)
        print('Verifying every task and dossier of the bank...', flush=True)
        registry = fast.load_registry(BANK / 'registry.json', deep=True)
        loader, recorded = e18_commit.load_committed_task_registry, {}

        def same_registry(path, **kwargs):
            return registry if Path(path).resolve() == (BANK / 'registry.json').resolve() and not kwargs else loader(path, **kwargs)

        def recording_sha(path):
            recorded[Path(path)] = value = fast.file_sha256(path)
            return value
        with fast.patched(e18_commit, load_committed_task_registry=same_registry, _sha=recording_sha):
            derived = _verify_main_derived_artifacts(self.cfg())
        if not derived['valid']:
            raise ValueError('Bank catalog/checksums/renders are incomplete')
        for name, value in fast.read_render_index(BANK).items():
            if recorded.get(BANK / 'renders' / name) != value:
                raise ValueError('Installed render differs from staged render')
        results = []
        for c, p in zip(self.members, prepared):
            new = [e for e in historical['entries'] if e['commit_set_id'] == c.name]
            if ({e['source_candidate_id'] for e in new} != set(p['main_task_ids']) or len(new) != len(p['main_task_ids'])
                    or any(e.get('project_stage') != c.flag for e in new)):
                raise ValueError('Installed membership or project stage differs')
            commit_sets = [s for s in historical['commit_sets'] if s['commit_set_id'] == c.name]
            if len(commit_sets) != 1 or commit_sets[0].get('project_stage') != c.flag:
                raise ValueError('Commit set project stage missing')
            for entry in new:
                source = c.state(ROOT / entry['source_run'])['semantic_records'][entry['source_candidate_id']]
                dossier = (BANK / entry['dossier_path']).parent
                if read_json(dossier / 'review.json') != source or source.get('intrinsic_quality') is not None:
                    raise ValueError('Original review record or ungraded status changed')
                if digest(BANK / entry['task_path']) != digest(c.stage / entry['task_path']):
                    raise ValueError('Task bytes changed during commitment')
            result = {'status': 'committed_and_verified', 'committed': len(new), 'boardless_verified': len(new),
                      'quality_grade': 'unassigned', 'project_stage': c.flag, 'historical_bank': len(historical['entries']),
                      'active_bank': len(active['entries']), 'source_files_verified': len(read_json(c.out / 'source_manifest.json')['files']),
                      'original_reviews_unchanged': True, 'task_bytes_preserved': True, 'derived_artifacts': derived,
                      'new_model_calls': 0, 'run_resumed': False}
            if len(self.members) > 1:
                result['transaction'] = self.names
            results.append(result)
        return results

    def report(self, c, result):
        atomic_write_json(c.out / 'report.json', result)
        together = (f' in one transaction with {", ".join(n for n in self.names if n != c.name)}' if len(self.names) > 1 else '')
        count = len(read_json(c.stage / 'new_registry.private.json')['entries'])
        atomic_write_text(c.out / 'report.md', f'# {c.name} (project {c.flag})\n\n'
            f'Committed **{count} tasks** from {rel(c.run)}{together}, tagged `boardless_verified` with editorial '
            f'quality unassigned and `project_stage: {c.flag}` on every entry and on the commit set. Original task bytes, '
            'production technical/blind evidence, idea-loop lineage, source model/effort, planner, route, overlay and colour-'
            'instruction provenance are preserved; research was disabled.\n\n'
            f'Active bank: **{result["active_bank"]}**; historical registry: {result["historical_bank"]}. Final evidence, '
            'exact/palette checks, dossiers, catalogs and renders verified. No new model or embedding calls.\n')

    def run(self, prepared):
        with _committed_bank_lock(self.cfg()):
            self.check(prepared)
            journal = read_json(self.journal_path) if self.journal_path.exists() else None
            if journal is not None:
                if journal.get('master'):
                    raise ValueError(f'{self.head.name} belongs to the commit transaction of {journal["master"]}; run that one again')
                if (journal.get('format') != FORMAT or journal.get('members') != self.names
                        or journal.get('prepared_sha256') != {c.name: digest(c.out / 'prepared.json') for c in self.members}):
                    raise ValueError('Prepared transaction identity changed')
            if journal is not None and journal['status'] == 'complete':
                for c in self.members:
                    if journal['report_sha256'][c.name] != digest(c.out / 'report.json'):
                        raise ValueError('Terminal commit receipt changed')
                results = self.verify(prepared)
                for c, result in zip(self.members, results):
                    atomic_write_json(c.out / 'resume_verification.json', {'status': 'passed', 'idempotent': True, 'tasks_added': 0,
                                                                           'new_model_calls': 0, 'verification': result})
                return results
            self.check_identity(prepared)
            backup = self.head.out / 'backup'
            if journal is not None:
                if journal['status'] != 'installing':
                    raise ValueError('Inspect the previous failed transaction before retrying')
                _verify_backup(backup, journal['backup_manifest_sha256']); fast.rollback(BANK, backup)
            self.check_bank(prepared)
            if journal is None:
                self.make_backup(backup)
            journal = {'format': FORMAT, 'status': 'installing', 'members': self.names,
                       'backup_manifest_sha256': digest(backup / 'backup_manifest.json'),
                       'prepared_sha256': {c.name: digest(c.out / 'prepared.json') for c in self.members}}
            self.write_journal(journal)
            try:
                self.install()
                print(f'Installed {sum(len(p["main_task_ids"]) for p in prepared)} tasks of {len(self.members)} run(s); '
                      'verifying the full bank and dossiers...', flush=True)
                results = self.verify(prepared)
                for c, result in zip(self.members, results):
                    self.report(c, result)
                journal['status'] = 'complete'; journal['report_sha256'] = {c.name: digest(c.out / 'report.json') for c in self.members}
                self.write_journal(journal)
                return results
            except BaseException as exc:
                _verify_backup(backup, journal['backup_manifest_sha256']); fast.rollback(BANK, backup)
                journal.update(status='rolled_back', error=str(exc)); self.write_journal(journal)
                raise


def bounded_contact_sheet(images, destination, *, columns=None):
    """Same contact-sheet pixels, with one row in memory instead of the full bank (as in the r23 commit)."""
    from PIL import Image, ImageDraw
    from arcgen.lib.lean_v3_rendering import _font
    if not images:
        return None
    if columns is None:
        columns = 2 if len(images) > 1 else 1
    columns = min(columns, len(images))
    sizes = []
    for _, path in images:
        with Image.open(path) as source:
            thumb = source.convert('RGB'); thumb.thumbnail((760, 720), Image.Resampling.LANCZOS); sizes.append(thumb.size); thumb.close()
    tile_width = max(w for w, _ in sizes)
    row_heights = [max(h for _, h in sizes[i:i + columns]) for i in range(0, len(sizes), columns)]
    width = 48 + columns * tile_width + (columns - 1) * 24
    height = 48 + sum(row_heights) + len(row_heights) * 30 + (len(row_heights) - 1) * 24
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    compressor = zlib.compressobj()
    with destination.open('wb') as handle:
        def chunk(kind, payload):
            handle.write(struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff))

        def band(image):
            pixels = image.tobytes(); stride = width * 3
            for offset in range(0, len(pixels), stride):
                compressed = compressor.compress(b'\x00' + pixels[offset:offset + stride])
                if compressed:
                    chunk(b'IDAT', compressed)
            image.close()
        handle.write(b'\x89PNG\r\n\x1a\n')
        chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
        band(Image.new('RGB', (width, 24), '#EFEFED'))
        for row, row_height in enumerate(row_heights):
            strip = Image.new('RGB', (width, 30 + row_height + 24), '#EFEFED'); draw = ImageDraw.Draw(strip)
            for column in range(columns):
                index = row * columns + column
                if index >= len(images):
                    break
                label, path = images[index]; x = 24 + column * (tile_width + 24)
                draw.text((x, 0), label, fill='#111111', font=_font(17, bold=True))
                with Image.open(path) as source:
                    thumb = source.convert('RGB'); thumb.thumbnail((760, 720), Image.Resampling.LANCZOS)
                    strip.paste(thumb, (x, 30)); thumb.close()
            band(strip)
        chunk(b'IDAT', compressor.flush()); chunk(b'IEND', b'')
    return destination


def render_bank_bounded(destination, registry):
    from arcgen.lib import authorized_commits
    original = authorized_commits.build_contact_sheet
    try:
        authorized_commits.build_contact_sheet = bounded_contact_sheet
        _render_bank(destination, registry)
    finally:
        authorized_commits.build_contact_sheet = original


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path)
    p.add_argument('--name')
    p.add_argument('--authorization', help='the exact user instruction authorizing the commit')
    p.add_argument('--stage-flag', default='stage_3')
    p.add_argument('--committed-on')
    p.add_argument('--partial', action='store_true', help='a stopped run: commit only verified tasks of its finished windows')
    p.add_argument('--hold', type=Path, help='JSON of tasks to hold back ({task_id: reason} or a list); they are listed as excluded')
    p.add_argument('--release-judge', type=Path, help='weakness runs: JSON list of JUDGE task IDs that passed the offline '
                   'consensus reads; other JUDGE tasks pending consensus, and every JUDGE decoy, are held')
    p.add_argument('--after', help='transaction of several runs: stack this commit on the prepared commit AFTER (its staged bank)')
    p.add_argument('--defer-sheet', action='store_true', help='transaction of several runs: not the last commit (the last makes '
                   'the contact sheet and render index of the final bank)')
    p.add_argument('--hold-subset', action='store_true', help='the --hold file may list tasks of other runs of the transaction')
    p.add_argument('--then-install', action='store_true', help='an --install of this transaction follows; it hash-verifies every '
                   'dossier of the final bank, so prepare checks the bank\'s dossier manifests only')
    p.add_argument('--install', nargs='+', metavar='NAME', help='install these prepared commits (in order) as one transaction')
    p.add_argument('--commit', action='store_true')
    a = p.parse_args()
    if a.install:
        if not a.commit or a.root or a.name:
            p.error('--install takes only the prepared commit names and --commit')
        members = [Commit.from_prepared(name) for name in a.install]
        for result in Transaction([m for m, _ in members]).run([prepared for _, prepared in members]):
            print(result, flush=True)
        return
    if not (a.root and a.name and a.authorization):
        p.error('--root, --name and --authorization are required')
    if a.commit and (a.after or a.defer_sheet):
        p.error('a commit stacked in a transaction is installed with --install')
    c = Commit(a.root, a.name, a.authorization, a.stage_flag, a.committed_on, a.partial, a.hold, a.release_judge,
               after=a.after, defer_sheet=a.defer_sheet, hold_subset=a.hold_subset, then_install=a.commit or a.then_install)
    prepared = c.prepare()
    if a.commit:
        print(c.commit(prepared), flush=True)


if __name__ == '__main__':
    main()
