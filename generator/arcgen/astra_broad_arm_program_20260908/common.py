"""Frozen inputs, isolated account profiles and atomic finite call admission."""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib, json, sqlite3, time

from arcgen.paths import WORKSPACE as ROOT  # the data root: runs and the bank
from arcgen.batch.config import OPERATIONS as _OPERATIONS

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent  # arcgen/: prompts/, config/, lib/
CARDS = HERE / 'inputs/astra_broad_arm_program_codesign_20260908'
HOMES = dict(_OPERATIONS['accounts'])  # {account: Codex home}: the one configured account

class ProgramError(RuntimeError): pass
class IntegrityError(ProgramError): pass
class BudgetError(ProgramError): pass

def read(path): return json.loads(Path(path).read_text())
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}

def component(path):
    """Plain invitation cards have byte identities, not invented prompt text."""
    from arcgen.lib.util import load_prompt,content_hash
    text=Path(path).read_text()
    if '<!-- prompt-version:' in text:return load_prompt(Path(path))
    return text,'frozen-card-'+sha(path)[:16],content_hash(text)

def atomic(path, value):
    from arcgen.lib.util import atomic_write_json
    atomic_write_json(Path(path), value)

def frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value: raise IntegrityError(f'frozen artifact changed: {path}')
    else: atomic(path, value)
    return value

def verify_refs(value):
    n = 0
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            p = Path(value['path']); p = p if p.is_absolute() else ROOT/p
            if not p.is_file() or sha(p) != value['sha256']: raise IntegrityError(f'source changed: {p}')
            n += 1
        for v in value.values(): n += verify_refs(v)
    elif isinstance(value, list):
        for v in value: n += verify_refs(v)
    return n


def config_for(root, account, *, seed=2026090840, mock=False):
    from arcgen.lib.factory_config import load_factory_config
    from arcgen.lib.util import content_hash
    base = load_factory_config(SOURCE/'config/astra_reduced_release_r4_20260908.yaml')
    c = deepcopy(base); root = Path(root).resolve()
    if account not in HOMES: raise IntegrityError('only the configured Codex account is bound')
    c['seed'] = seed
    c['run'].update(id=root.name, account=account, replicate='broad-arm-r1', mock=mock)
    c['transport']['codex_home'] = HOMES[account]
    c['_output_directory'] = str(root)
    c['_paths'].update(output_directory=root, cache=root/'cache', prompts=SOURCE/'prompts')
    c['paths'].update(output_directory=str(root), cache=str(root/'cache'), prompts=str(SOURCE/'prompts'))
    c['_run_identity'].update(run_id=root.name, account=account, replicate='broad-arm-r1')
    c['_experiment_profile'] = {'id':'broad-arm-r1',
        'baseline_config_sha256':base['_config_sha256'], 'q2_policy':True, 'stationary_windows':True}
    c['_config_sha256'] = content_hash({'baseline':base['_config_sha256'], 'run':c['run'],
        'seed':seed, 'transport':c['transport'], 'paths':c['paths'], 'experiment':c['_experiment_profile']})
    c['_run_identity']['config_sha256'] = c['_config_sha256']
    return c

class Budget:
    """SQLite transactions make reservations atomic across threads and resumes.

    Logical reservations and physical attempts are never reclaimed. The native
    E18 journal remains Astra's sole retry owner; Fable uses the same two-attempt
    envelope. This ledger is admission/accounting, not another retry wrapper.
    """
    def __init__(self, root, limits):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'budget.sqlite3'
        self.limits = limits
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, phase TEXT, category TEXT, provider TEXT, account TEXT, window TEXT, fingerprint TEXT, admitted REAL, status TEXT, outcome TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (key TEXT, ordinal INTEGER, started REAL, ended REAL, status TEXT, metadata TEXT, PRIMARY KEY(key, ordinal))')
            db.execute('CREATE TABLE IF NOT EXISTS developments (key TEXT PRIMARY KEY, phase TEXT, window TEXT, fingerprint TEXT)')
            old = db.execute("SELECT value FROM metadata WHERE key='limits'").fetchone()
            if old and old[0] != canonical(self.limits): raise IntegrityError('budget changed on resume')
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('limits', ?)", (canonical(self.limits),))

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            db.execute('PRAGMA journal_mode=WAL'); db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback(); raise
        finally: db.close()

    def phase(self, phase):
        return next(p for p in self.limits['phases'] if p['phase'] == phase)

    def admit(self, key, *, phase, category, provider, account, window, fingerprint):
        p = self.phase(phase)
        if category not in p['categories']: raise BudgetError('unbudgeted category')
        if provider not in ('astra','fable') or account not in (set(HOMES) if provider=='astra' else {'fable'}):
            raise IntegrityError('unbound provider/account')
        fields = (phase, category, provider, account, window, fingerprint)
        with self.transaction() as db:
            old = db.execute('SELECT phase,category,provider,account,window,fingerprint FROM calls WHERE key=?',(key,)).fetchone()
            if old:
                if old != fields: raise IntegrityError('resumed logical request changed')
                return
            checks = [("SELECT count(*) FROM calls", (), self.limits['logical_cap']),
                ("SELECT count(*) FROM calls WHERE phase=?", (phase,), p['logical_cap']),
                ("SELECT count(*) FROM calls WHERE phase=? AND category=?", (phase, category), p['categories'][category]),
                ("SELECT count(*) FROM calls WHERE phase=? AND provider=?", (phase,provider), p[provider+'_logical_cap'])]
            if window and category in ('conditional_evidence_repair','actual_boards'):
                checks.append(("SELECT count(*) FROM calls WHERE phase=? AND category=? AND window=?", (phase,category,window), 5 if category=='conditional_evidence_repair' else 3))
            for query, args, cap in checks:
                if db.execute(query,args).fetchone()[0] >= cap: raise BudgetError(f'allowance exhausted: {phase}/{category}')
            db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)',(key,*fields,time.time(),'admitted',None))

    def start_attempt(self, key):
        with self.transaction() as db:
            row = db.execute('SELECT phase FROM calls WHERE key=?',(key,)).fetchone()
            if not row: raise BudgetError('attempt has no logical reservation')
            n = db.execute('SELECT count(*) FROM attempts WHERE key=?',(key,)).fetchone()[0]
            if n >= 2: raise BudgetError('two physical attempts exhausted')
            if db.execute('SELECT count(*) FROM attempts').fetchone()[0] >= self.limits['physical_attempt_cap']:
                raise BudgetError('physical program ceiling')
            p = self.phase(row[0])
            if db.execute('SELECT count(*) FROM attempts a JOIN calls c ON a.key=c.key WHERE c.phase=?',row).fetchone()[0] >= p['physical_attempt_cap']:
                raise BudgetError('physical phase ceiling')
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?)',(key,n+1,time.time(),None,'started',None))
            return n+1

    def end_attempt(self, key, ordinal, status, **metadata):
        with self.transaction() as db:
            db.execute('UPDATE attempts SET ended=?,status=?,metadata=? WHERE key=? AND ordinal=?',
                (time.time(),status,canonical(metadata),key,ordinal))

    def outcome(self, key, status, **value):
        with self.transaction() as db:
            db.execute('UPDATE calls SET status=?,outcome=? WHERE key=?',(status,canonical(value),key))

    def reserve_development(self, key, phase, window, fingerprint, window_cap):
        p = self.phase(phase)
        with self.transaction() as db:
            old = db.execute('SELECT phase,window,fingerprint FROM developments WHERE key=?',(key,)).fetchone()
            if old:
                if old != (phase,window,fingerprint): raise IntegrityError('development changed on resume')
                return
            for query,args,cap in [('SELECT count(*) FROM developments',(),336),
                ('SELECT count(*) FROM developments WHERE phase=?',(phase,),p['maximum_developments']),
                ('SELECT count(*) FROM developments WHERE phase=? AND window=?',(phase,window),window_cap)]:
                if db.execute(query,args).fetchone()[0] >= cap: raise BudgetError('development ceiling')
            db.execute('INSERT INTO developments VALUES (?,?,?,?)',(key,phase,window,fingerprint))

    def summary(self):
        with self.transaction() as db:
            db.row_factory = sqlite3.Row
            calls = [dict(r) for r in db.execute('SELECT * FROM calls ORDER BY key')]
            attempts = [dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY key,ordinal')]
            developments = [dict(r) for r in db.execute('SELECT * FROM developments ORDER BY key')]
        return {'logical_calls':len(calls),'physical_attempts':len(attempts),'developments':len(developments),
            'by_phase_category':dict(Counter(r['phase']+'/'+r['category'] for r in calls)),
            'by_provider_account':dict(Counter(r['provider']+'/'+r['account'] for r in calls)),
            'calls':calls,'attempts':attempts,'development_rows':developments,'limits':self.limits}

def halt(root, reason, **details):
    atomic(Path(root)/'STOP_DISPATCH.json', {'reason':reason,'at_epoch':time.time(),**details})
    raise IntegrityError(reason)

def log(event, **fields):
    print(json.dumps({'at_epoch':round(time.time(),3),'event':event,**fields},sort_keys=True),flush=True)
