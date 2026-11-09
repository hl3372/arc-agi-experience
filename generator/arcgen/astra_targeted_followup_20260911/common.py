"""Revision identity, immutable inputs and independently supplied finite budgets."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
from arcgen.astra_broad_arm_program_20260908.common import (
    ROOT, SOURCE, HOMES as BASE_HOMES, IntegrityError, BudgetError, canonical, digest, sha,
    ref, read, frozen, atomic, verify_refs, component, Budget as BaseBudget,
    config_for as base_config,
)

HERE = Path(__file__).resolve().parent
REVISION = 'astra-targeted-followup-r3-20260911'
CARDS = HERE / 'cards'
SETTINGS = read(HERE / 'settings.json')
HOMES = dict(BASE_HOMES)


def isolated_root(root):
    from arcgen.paths import isolated
    return isolated(root, IntegrityError)


def code_identity():
    paths = [p for p in HERE.rglob('*') if p.is_file()
             and p.suffix in ('.py', '.json', '.md')
             and not any(x in p.relative_to(HERE).parts for x in ('evidence', 'test_work', '__pycache__', '.pytest_cache'))]
    return digest([{'path': str(p.relative_to(HERE)), 'sha256': sha(p)} for p in sorted(paths)])


def config_for(root, account=None, *, seed=2026091040, mock=False):
    root = isolated_root(root)
    account = account or next(iter(HOMES))
    if account not in HOMES:
        raise IntegrityError('only the configured Codex account is bound')
    value = base_config(root, account, seed=seed, mock=mock)
    value['_paths']['prompts'] = HERE / 'prompts'
    value['paths']['prompts'] = str(HERE / 'prompts')
    value['run']['replicate'] = REVISION
    value['_run_identity']['replicate'] = REVISION
    value['_experiment_profile'] = {'id': REVISION, 'settings_sha256': sha(HERE / 'settings.json'),
                                    'launch_authorized_by_implementation': False}
    value['_config_sha256'] = digest({'base': value['_config_sha256'],
                                     'run': value['run'], 'revision': value['_experiment_profile'],
                                     'prompt_root': str(HERE / 'prompts')})
    value['_run_identity']['config_sha256'] = value['_config_sha256']
    return value


class Budget(BaseBudget):
    """Reuse native atomic admission/two-attempt transport; no inherited spend."""
    def __init__(self, root, limits=None):
        self.root = isolated_root(root)
        path = self.root / 'budget_limits.private.json'
        if limits is None:
            if not path.exists():
                raise BudgetError('a new experiment must supply its own finite limits')
            limits = read(path)
        self.limits = deepcopy(limits)
        required = ('logical_cap', 'physical_attempt_cap', 'maximum_developments')
        if any(type(limits.get(k)) is not int or limits[k] < 0 for k in required):
            raise BudgetError('explicit nonnegative program limits required')
        if limits['physical_attempt_cap'] > 2 * limits['logical_cap']:
            raise BudgetError('physical allowance exceeds two attempts per call')
        names = []
        for p in limits.get('phases', []):
            names.append(p['phase'])
            if p.get('fable_logical_cap') != 0 or p.get('astra_logical_cap') != p['logical_cap']:
                raise BudgetError('only the existing Astra account routes are allowed')
            if any(type(p.get(k)) is not int or p[k] < 0 for k in required):
                raise BudgetError('explicit finite phase limits required')
            if p['physical_attempt_cap'] > 2 * p['logical_cap'] or any(
                    type(v) is not int or v < 0 for v in p['categories'].values()):
                raise BudgetError('invalid category or attempt allowance')
        if not names or len(names) != len(set(names)):
            raise BudgetError('unique explicit phases required')
        self.root.mkdir(parents=True, exist_ok=True)
        frozen(path, self.limits)
        self.path = self.root / 'budget.sqlite3'
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, phase TEXT, category TEXT, provider TEXT, account TEXT, window TEXT, fingerprint TEXT, admitted REAL, status TEXT, outcome TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (key TEXT, ordinal INTEGER, started REAL, ended REAL, status TEXT, metadata TEXT, PRIMARY KEY(key, ordinal))')
            db.execute('CREATE TABLE IF NOT EXISTS developments (key TEXT PRIMARY KEY, phase TEXT, window TEXT, fingerprint TEXT)')
            identity = canonical({'revision': REVISION, 'limits': self.limits})
            prior = db.execute("SELECT value FROM metadata WHERE key='revision_limits'").fetchone()
            if prior and prior[0] != identity:
                raise IntegrityError('revision budget changed on resume')
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('revision_limits', ?)", (identity,))

    def admit(self, key, **kwargs):
        if kwargs['provider'] != 'astra' or kwargs['account'] not in HOMES:
            raise IntegrityError('unbound provider/account')
        return super().admit(key, **kwargs)

    def reserve_development(self, key, phase, window, fingerprint, window_cap):
        p = self.phase(phase)
        with self.transaction() as db:
            prior = db.execute('SELECT phase,window,fingerprint FROM developments WHERE key=?', (key,)).fetchone()
            if prior:
                if prior != (phase, window, fingerprint):
                    raise IntegrityError('development changed on resume')
                return
            checks = [('SELECT count(*) FROM developments', (), self.limits['maximum_developments']),
                      ('SELECT count(*) FROM developments WHERE phase=?', (phase,), p['maximum_developments']),
                      ('SELECT count(*) FROM developments WHERE phase=? AND window=?', (phase, window), window_cap)]
            for query, args, cap in checks:
                if db.execute(query, args).fetchone()[0] >= cap:
                    raise BudgetError('development ceiling')
            db.execute('INSERT INTO developments VALUES (?,?,?,?)', (key, phase, window, fingerprint))
