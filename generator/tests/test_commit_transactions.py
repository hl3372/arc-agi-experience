"""Bank commits on synthetic banks and runs (no model calls).

(a) a first commit bootstraps the render index and a second draws only its new tasks; (b) several runs committed as one
transaction leave the same bank as separate commits; (c) tampered staging is caught and a failure mid-install restores
the exact prior bank; (d) an interrupted install resumes and a completed one re-runs as an idempotent verification.
Also: linked read-only dossier records and the copy fallback, the parallel contact sheet, the fall-back to drawing the
whole bank, and the cli's several-run commit.
"""
import errno, json, os, shutil, stat, time
from pathlib import Path
from types import SimpleNamespace

import pytest

from arcgen.batch import cli, commit, commit_bank
from support import commit_fixtures as fx

RUNS = (('b1', 5, 3), ('b2', 4, 4))


class World:
    """A synthetic repository root (bank of 24 tasks, runs b1 and b2) restored from a pristine copy between commits. Run
    records hold absolute paths, so every commit runs at the same root path; results are moved aside to compare."""

    def __init__(self, base, monkeypatch, bank=24, runs=RUNS):
        self.base, self.mp = Path(base), monkeypatch
        self.root, self.pristine = self.base / 'root', self.base / 'pristine'
        monkeypatch.setenv('ARCGEN_COMMIT_WORKERS', '2')
        fx.make_bank(self.root, bank)
        for name, n, seed in runs:
            fx.make_run(self.root, name, n, homes=commit.HOMES, seed=seed)
        (self.root / 'commit_code.py').write_text('# commit code\n')
        shutil.copytree(self.root, self.pristine, symlinks=True)
        for mod in (commit,):
            monkeypatch.setattr(mod, 'ROOT', self.root); monkeypatch.setattr(mod, 'BANK', self.root / 'committed_tasks')
            monkeypatch.setattr(mod, '__file__', str(self.root / 'commit_code.py'))
            monkeypatch.setattr(mod.Commit, 'verify_candidate', fx.light_verify_candidate)
        assert str(commit.BANK).startswith(str(self.base))

    @property
    def bank(self):
        return self.root / 'committed_tasks'

    def reset(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        shutil.copytree(self.pristine, self.root, symlinks=True)

    def keep(self, label):
        target = self.base / label
        self.root.rename(target)
        return target / 'committed_tasks'

    def new(self, run, **kw):
        return commit.Commit(self.root / 'runs' / run, f'{run}-task-commit-x', 'ok', committed_on='2026-10-24', **kw)

    def commit_new(self, run):
        c = self.new(run, then_install=True)
        return c, c.commit(c.prepare())

    def transaction(self, runs):
        names = []
        for i, run in enumerate(runs):
            c = self.new(run, after=names[-1] if names else None, defer_sheet=i < len(runs) - 1, then_install=True)
            c.prepare(); names.append(c.name)
        members = [commit.Commit.from_prepared(n) for n in names]
        return commit.Transaction([m for m, _ in members]), [p for _, p in members]


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def without_index(snap):
    return {k: v for k, v in snap.items() if k != commit_bank.RENDER_INDEX}


def index_matches_renders(bank):
    index = commit_bank.read_render_index(bank)
    return index == {p.name: fx.sha(p) for p in (bank / 'renders').iterdir()}


# ---------------------------------------------------------------- (a) incremental renders
def test_a_second_commit_draws_only_its_new_tasks(world):
    modes = []
    for run, n, _ in RUNS:
        c, result = world.commit_new(run)
        prepared = json.loads((c.out / 'prepared.json').read_text())
        modes.append((prepared['renders']['mode'], prepared['renders']['drawn']))
        assert result['derived_artifacts']['valid'] and result['committed'] == n
    assert modes == [('full', 24 + 5), ('incremental', 4)]  # the first commit bootstraps the render index; the next draws 4
    assert index_matches_renders(world.bank)


# ---------------------------------------------------------------- (b) one transaction == separate commits
def test_a_transaction_of_two_runs_leaves_the_same_bank_as_two_commits(world):
    for run, _, _ in RUNS:
        world.commit_new(run)
    sequential = fx.snapshot(world.keep('sequential'))
    world.reset()
    transaction, prepared = world.transaction(['b1', 'b2'])
    assert prepared[0]['transaction']['defer_sheet'] and prepared[1]['transaction']['after'] == 'b1-task-commit-x'
    results = transaction.run(prepared)
    assert [r['committed'] for r in results] == [5, 4] and all(r['historical_bank'] == 33 for r in results)
    out = world.root / 'runs'
    head, member = (json.loads((out / f'{r}-task-commit-x/transaction.json').read_text()) for r in ('b1', 'b2'))
    assert head['status'] == 'complete' and head['members'] == ['b1-task-commit-x', 'b2-task-commit-x']
    assert member == {'format': 2, 'status': 'complete', 'master': 'b1-task-commit-x'}
    assert all((out / f'{r}-task-commit-x/report.md').is_file() for r in ('b1', 'b2'))
    assert fx.snapshot(world.keep('transaction')) == sequential


def test_a_transaction_refuses_a_changed_member(world):
    transaction, prepared = world.transaction(['b1', 'b2'])
    path = world.root / 'runs/b1-task-commit-x/prepared.json'
    path.write_text(path.read_text().replace('"partial": false', '"partial": false ', 1))
    before = fx.snapshot(world.bank)
    with pytest.raises(ValueError, match='stacks on changed'):
        transaction.run([json.loads(path.read_text()), prepared[1]])
    assert fx.snapshot(world.bank) == before


# ---------------------------------------------------------------- (c) tampering and rollback
@pytest.mark.parametrize('target', ['task', 'render', 'dossier', 'record'])
def test_tampered_staging_is_caught_before_the_bank_changes(world, target):
    world.commit_new('b1')
    c = world.new('b2', then_install=True); prepared = c.prepare()
    before = fx.snapshot(world.bank)
    stage = c.stage
    if target == 'task':
        path = next((stage / 'tasks').iterdir())
    elif target == 'render':
        path = next(p for p in (stage / 'renders').iterdir() if p.suffix == '.png' and p.name != 'contact_sheet.png')
    elif target == 'dossier':
        path = next((stage / 'dossiers').iterdir()) / 'provenance.json'
    else:  # a linked model-call record: the run's own file
        path = next(p for p in (stage / 'dossiers').rglob('raw_response.txt'))
        assert os.stat(path).st_nlink > 1
        os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises((ValueError, RuntimeError), match='changed'):
        c.commit(prepared)
    assert fx.snapshot(world.bank) == before


@pytest.mark.parametrize('when', ['first', 'second', 'transaction'])
def test_a_failure_mid_install_restores_the_exact_bank(world, when):
    if when == 'second':
        world.commit_new('b1')
    before = fx.snapshot(world.bank)
    real, calls = commit_bank.place, []

    def failing(source, target, **kw):  # dies after the tasks, dossiers, metadata and a few renders are in
        calls.append(target)
        if len([t for t in calls if Path(t).parent.name == 'renders']) == 3:
            raise OSError('disk full')
        return real(source, target, **kw)
    world.mp.setattr(commit_bank, 'place', failing)
    if when == 'transaction':
        transaction, prepared = world.transaction(['b1', 'b2']); folder = world.root / 'runs/b1-task-commit-x'
        with pytest.raises(OSError, match='disk full'):
            transaction.run(prepared)
    else:
        c = world.new('b2' if when == 'second' else 'b1', then_install=True); prepared = c.prepare(); folder = c.out
        with pytest.raises(OSError, match='disk full'):
            c.commit(prepared)
    assert fx.snapshot(world.bank) == before
    assert json.loads((folder / 'transaction.json').read_text())['status'] == 'rolled_back'


def test_a_damaged_old_dossier_still_stops_the_commit_and_the_bank_stays_as_it_was(world):
    """Every dossier is hash-verified once per transaction, on the final bank after the install: damage to an old dossier
    is caught there and the install is undone (the old code caught it at prepare). A dry run still checks at prepare."""
    victim = next((world.bank / 'dossiers').iterdir()) / 'model_calls/x/raw_response.txt'
    victim.write_bytes(b'bit rot')
    before = fx.snapshot(world.bank)
    with pytest.raises(ValueError, match='invalid committed dossier'):
        world.new('b1').prepare()  # dry run: deep check at prepare, nothing staged
    assert not (world.root / 'runs/b1-task-commit-x/staged_bank').exists()
    world.reset(); victim.write_bytes(b'bit rot')
    c = world.new('b1', then_install=True); prepared = c.prepare()  # the install follows: manifests only here
    with pytest.raises(ValueError, match='invalid committed dossier'):
        c.commit(prepared)
    assert fx.snapshot(world.bank) == before
    assert json.loads((c.out / 'transaction.json').read_text())['status'] == 'rolled_back'


# ---------------------------------------------------------------- (d) resume and idempotency
def test_an_interrupted_install_resumes_and_a_finished_one_reverifies(world):
    world.commit_new('b1')
    c = world.new('b2', then_install=True); c.commit(c.prepare())
    reference = fx.snapshot(world.keep('reference'))
    world.reset(); world.commit_new('b1')
    c = world.new('b2', then_install=True); prepared = c.prepare()
    real_verify, real_rollback = commit.Transaction.verify, commit_bank.rollback

    def crash(*args, **kwargs):  # the process dies during the install: no rollback runs, the journal says installing
        raise SystemExit('killed')
    world.mp.setattr(commit.Transaction, 'verify', lambda self, p: (_ for _ in ()).throw(OSError('power cut')))
    world.mp.setattr(commit_bank, 'rollback', crash)
    with pytest.raises(SystemExit):
        c.commit(prepared)
    assert json.loads((c.out / 'transaction.json').read_text())['status'] == 'installing'
    assert any(p.name.startswith('e18-production__04') for p in (world.bank / 'tasks').iterdir())  # half installed
    world.mp.setattr(commit.Transaction, 'verify', real_verify); world.mp.setattr(commit_bank, 'rollback', real_rollback)
    again = world.new('b2', then_install=True)
    assert again.prepare() == prepared  # a prepared commit resumes after re-verifying its sources and staging
    result = again.commit(prepared)
    assert result['committed'] == 4 and json.loads((c.out / 'transaction.json').read_text())['status'] == 'complete'
    snap = fx.snapshot(world.bank)
    repeat = world.new('b2', then_install=True).commit(prepared)  # a completed commit re-runs as a verification
    assert repeat['committed'] == 4 and json.loads((c.out / 'resume_verification.json').read_text())['idempotent'] is True
    assert fx.snapshot(world.bank) == snap
    assert fx.snapshot(world.keep('resumed')) == reference


# ---------------------------------------------------------------- dossier records as read-only links
def test_dossier_records_are_read_only_links_of_the_run_and_copies_otherwise(world):
    c, _ = world.commit_new('b1')
    linked = json.loads((c.out / 'dossier_links.json').read_text())['linked']
    assert linked and all('/model_calls/' in p or '/run_context/terminal/' in p for p in linked)
    rel = next(p for p in linked if '/astra_concrete_plan_v3/' in p and p.endswith('raw_response.txt'))
    bank_file = world.bank / rel
    call_id = bank_file.parent.name
    source = next(p for p in (world.root / 'runs/b1').rglob('raw_response.txt') if p.parent.name == call_id)
    assert os.stat(bank_file).st_ino == os.stat(source).st_ino and not stat.S_IMODE(os.stat(source).st_mode) & 0o222
    dossier = bank_file.parents[3]
    for generated in ('manifest.json', 'provenance.json', 'review.json'):
        assert os.stat(dossier / generated).st_nlink == 1  # commit-written records are copies
    assert not any(os.stat(p).st_nlink > 1 for p in (dossier / 'run_context/run_provenance').iterdir())  # not call/evidence
    linked_bank = fx.snapshot(world.keep('linked'))
    world.reset()

    def no_links(*args, **kwargs):
        raise OSError(errno.EXDEV, 'cross-device link')
    world.mp.setattr(commit_bank.os, 'link', no_links)
    c, _ = world.commit_new('b1')
    counts = json.loads((c.out / 'dossier_links.json').read_text())
    assert counts['linked'] == [] and counts['counts'].get('link_fallback_EXDEV', 0) > 0
    assert fx.snapshot(world.bank) == linked_bank  # same bytes, copied


# ---------------------------------------------------------------- renders
def test_the_parallel_contact_sheet_is_the_bounded_contact_sheet_byte_for_byte(tmp_path):
    from PIL import Image
    images = []
    for i, (w, h) in enumerate([(300, 200), (1500, 700), (700, 1600), (761, 721), (760, 720), (1200, 1199), (64, 2000)] * 4 + [(900, 450)]):
        path = tmp_path / f'{i:02d}.png'
        Image.new('RGB', (w, h), ((i * 37) % 256, (i * 91) % 256, 200)).save(path)
        images.append((f'task-{i}', path))
        with Image.open(path) as source:
            thumb = source.convert('RGB'); thumb.thumbnail((760, 720), Image.Resampling.LANCZOS)
            assert commit_bank.thumbnail_size(w, h) == thumb.size
    commit.bounded_contact_sheet(images, tmp_path / 'old.png')
    commit_bank.contact_sheet(images, tmp_path / 'new.png', processes=2, in_flight=5)
    assert (tmp_path / 'old.png').read_bytes() == (tmp_path / 'new.png').read_bytes()


@pytest.mark.parametrize('damage', ['render', 'index'])
def test_renders_that_cannot_be_kept_are_drawn_again(world, damage):
    world.commit_new('b1')
    if damage == 'render':
        victim = next(p for p in (world.bank / 'renders').iterdir() if p.name.startswith('03__'))
        victim.write_bytes(victim.read_bytes()[:-10])
    else:
        (world.bank / commit_bank.RENDER_INDEX).unlink()
    c, _ = world.commit_new('b2')
    renders = json.loads((c.out / 'prepared.json').read_text())['renders']
    assert renders['mode'] == 'full' and renders['drawn'] == 33 and renders['reason']
    bank = world.keep('healed'); healed = fx.snapshot(bank)
    world.reset()
    for run, _, _ in RUNS:
        world.commit_new(run)
    assert healed == fx.snapshot(world.bank) and index_matches_renders(bank)


def test_registry_loading_matches_the_canonical_loader(world):
    from arcgen.lib.committed_tasks import load_committed_task_registry
    canonical = load_committed_task_registry(world.bank / 'registry.json')
    assert commit_bank.load_registry(world.bank / 'registry.json', deep=True, processes=3) == canonical
    assert commit_bank.load_registry(world.bank / 'registry.json', deep=False) == canonical
    entry = canonical['entries'][7]
    artifact = Path(entry['absolute_dossier_path']).parent / 'model_calls/x/raw_response.txt'
    artifact.write_bytes(b'tampered')
    with pytest.raises(ValueError) as deep_error:
        commit_bank.load_registry(world.bank / 'registry.json', deep=True, processes=3)
    with pytest.raises(ValueError) as canonical_error:
        load_committed_task_registry(world.bank / 'registry.json')
    assert str(deep_error.value) == str(canonical_error.value)
    commit_bank.load_registry(world.bank / 'registry.json', deep=False)  # the manifest itself is intact


# ---------------------------------------------------------------- cli
def test_cli_commits_several_runs_as_one_transaction(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path); monkeypatch.setattr(cli, 'alive', lambda name: False)
    monkeypatch.setattr(cli, 'run_pids', lambda root: [])
    calls = []
    monkeypatch.setattr(cli.subprocess, 'run', lambda args, **kw: calls.append((args, kw['env'])) or SimpleNamespace(returncode=0))
    for name in ('a', 'b', 'c'):
        fx.write(tmp_path / 'runs' / name / 'STATUS.json', {'status': 'verified_terminal'})
    (tmp_path / 'committed_tasks').mkdir()
    cli.commit(['a', 'b', 'c'])
    prepares, (install, _) = calls[:3], calls[3]
    names = [args[args.index('--name') + 1] for args, _ in prepares]
    assert [n.split('-task-commit-')[0] for n in names] == ['a', 'b', 'c']
    for i, (args, env) in enumerate(prepares):
        assert '--commit' not in args and '--then-install' in args and env['ARCGEN_CONFIG'].endswith(f'runs/{"abc"[i]}/config.json')
        assert ('--after' in args) == (i > 0) and ('--defer-sheet' in args) == (i < 2)
        if i:
            assert args[args.index('--after') + 1] == names[i - 1]
    assert install[install.index('--install') + 1:] == names + ['--commit']
    calls.clear()
    cli.commit(['a'], dry_run=True)  # one name: the usual single commit
    (args, _), = calls
    assert '--after' not in args and '--install' not in args and '--then-install' not in args and '--commit' not in args
    calls.clear()
    for name in ('a', 'b'):  # what the (mocked) prepares would have recorded: nothing held
        fx.write(tmp_path / 'runs' / f"{name}-task-commit-{time.strftime('%Y%m%d')}" / 'authorization.json', {'excluded': []})
    fx.write(tmp_path / 'held.json', ['t-nowhere'])
    with pytest.raises(SystemExit, match='in none of these runs'):
        cli.commit(['a', 'b'], hold=str(tmp_path / 'held.json'))
    assert len(calls) == 2 and all('--hold-subset' in args for args, _ in calls)
