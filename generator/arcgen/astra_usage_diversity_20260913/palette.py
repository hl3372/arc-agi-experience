"""One process-safe palette ledger per batch; native semantic safeguards retained."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sqlite3
import json

from .common import ROOT, SETTINGS, IntegrityError, isolated_root, read, ref, sha, digest, frozen, atomic

POLICY = SETTINGS['palette_policy']
WEIGHTS = {int(c): w for c, w in POLICY['target_percentages'].items()}
PREFERRED = frozenset(POLICY['preferred_colors'])


def freeze_inputs(root, *, registry_path=None, task_root=None, seed=20260913):
    """Call during preparation with the current frozen production-bank snapshot.

    Historical measurements are context/reporting only. They never create a
    corrective wave in the next batch. Existing inputs are reused on resume.
    """
    from arcgen.lib.v41_palette import measure_background_distribution
    root = isolated_root(root)
    path = root/'inputs/palette_context.private.json'
    if path.exists():
        value = read(path)
        if value.get('policy') != POLICY or value.get('seed') != seed:
            raise IntegrityError('palette preparation policy or seed changed on resume')
        if registry_path is not None and value['bank_registry']['path'] != str(Path(registry_path).resolve()):
            raise IntegrityError('palette bank snapshot changed on resume')
        return value
    registry_path = Path(registry_path or ROOT/'committed_tasks/active_registry.json').resolve()
    task_root = Path(task_root or registry_path.parent).resolve()
    registry = read(registry_path); tasks = []; hashes = []
    for row in registry['entries']:
        p = (task_root/row['task_path']).resolve()
        if not p.is_relative_to(task_root) or sha(p) != row['arc_sha256']:
            raise IntegrityError('palette bank task path or content differs from its registry')
        tasks.append((row['registry_task_id'], read(p)))
        hashes.append((row['registry_task_id'], row['arc_sha256']))
    measurement = measure_background_distribution(tasks, modal_fraction=.5, require_consistent_input_mode=True)
    measurement.pop('task_ids_by_background')
    return frozen(path, {'schema_version': 2, 'policy': deepcopy(POLICY), 'seed': seed,
        'bank_registry': ref(registry_path), 'bank_task_set_hash': digest(sorted(hashes)),
        'bank_measurement': measurement, 'bank_counts_used_for_allocation': False})


def choose(counts, seed):
    """Fill individual color deficits within the preferred group's upper bound.

    Unsafe originals count too. Allowing the combined share to rise toward 70%
    prevents an unavoidable excess of black from crowding out orange and cyan.
    An immutable background can exceed the bound; eligible tasks never add to
    that excess. Historical bank deficits do not enter this decision.
    """
    n = sum(counts.values()) + 1
    allowed = set(WEIGHTS)
    upper = POLICY['preferred_combined_desired_range_percent'][1]
    if 100*(sum(counts.get(c, 0) for c in PREFERRED)+1) > upper*n:
        allowed -= PREFERRED
    return min(allowed, key=lambda c: (
        -(n*WEIGHTS[c]-100*counts.get(c, 0)), digest({'seed': seed, 'n': n, 'color': c}), c))


BUSY_SECONDS = 600  # writers wait their turn; a final audit serializes thousands of write_report calls


class Allocator:
    def __init__(self, root):
        self.root = isolated_root(root)
        self.context = read(self.root/'inputs/palette_context.private.json')
        if self.context.get('schema_version') != 2 or self.context.get('policy') != POLICY:
            raise IntegrityError('new palette policy requires newly frozen preparation inputs')
        if set(WEIGHTS) != set(range(10)) or sum(WEIGHTS.values()) != 100:
            raise IntegrityError('palette weights must cover all ten colors and total 100')
        if sum(WEIGHTS[c] for c in PREFERRED) != POLICY['preferred_combined_target_percent']:
            raise IntegrityError('preferred group target differs from its component weights')
        self.context_hash = digest(self.context)
        self.path = self.root/'palette.sqlite3'
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS candidates (task_id TEXT PRIMARY KEY, account TEXT NOT NULL, window TEXT NOT NULL, admission_hash TEXT NOT NULL, request_hash TEXT, receipt TEXT)')
            previous = db.execute("SELECT value FROM metadata WHERE key='context_hash'").fetchone()
            if previous and previous[0] != self.context_hash:
                raise IntegrityError('batch palette context changed on resume')
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('context_hash', ?)", (self.context_hash,))

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=BUSY_SECONDS)
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def register(self, task_id, account, window, admission_hash):
        from .common import HOMES
        if account not in HOMES:
            raise IntegrityError('palette registration has an unsupported account')
        identity = (account, window, admission_hash)
        with self.transaction() as db:
            previous = db.execute('SELECT account,window,admission_hash FROM candidates WHERE task_id=?', (task_id,)).fetchone()
            if previous and previous != identity:
                raise IntegrityError('palette candidate attribution changed on resume')
            db.execute('INSERT OR IGNORE INTO candidates(task_id,account,window,admission_hash) VALUES(?,?,?,?)', (task_id, *identity))

    def materialize(self, native, plan, realization, target_background, palette_config):
        if target_background is not None:
            raise IntegrityError('batch palette allocation must be deferred until realization')
        allowed = palette_config.get('allowed_background_colors', list(range(10)))
        if any(type(c) is not int for c in allowed) or set(allowed) != set(WEIGHTS):
            raise IntegrityError('runtime background colors differ from the frozen mixture')
        tid = plan['candidate_id']
        if realization['candidate_id'] != tid:
            raise IntegrityError('palette plan and realization identities differ')
        overview = plan['public_contract']['episode_overview']
        source = next((r['color'] for r in overview['functional_color_roles']
                       if r['role'] == overview['palette_contract']['background_role']), None)
        # Validate using the unchanged native materializer, initially without a
        # color change. This is local computation, never another model call.
        probe = native(plan, realization, target_background=source, palette_config=palette_config)
        request_hash = digest({'plan': plan, 'realization': realization, 'palette_config': palette_config})
        with self.transaction() as db:
            row = db.execute('SELECT account,window,request_hash,receipt FROM candidates WHERE task_id=?', (tid,)).fetchone()
            if row is None:
                raise IntegrityError('palette realization was not admitted to this batch')
            account, window, previous_hash, saved = row
            if saved:
                if previous_hash != request_hash:
                    raise IntegrityError('palette source realization changed on resume')
                receipt = json.loads(saved)
                binding = {k:v for k,v in receipt.items() if k != 'result_hash'}
            else:
                receipts = [json.loads(r[0]) for r in db.execute('SELECT receipt FROM candidates WHERE receipt IS NOT NULL')]
                counts = Counter(r['actual_background'] for r in receipts if r['actual_background'] is not None)
                target = choose(counts, self.context['seed']) if probe['eligible'] else source
                binding = {'policy_id': POLICY['id'], 'context_hash': self.context_hash,
                    'task_id': tid, 'account': account, 'window': window,
                    'ordinal': len(receipts)+1, 'request_hash': request_hash,
                    'eligible': probe['eligible'], 'source_background': source, 'target_background': target,
                    'actual_background': (target if probe['eligible'] else source) if probe['complete_task'] else None,
                    'has_grid': probe['complete_task'] is not None,
                    'reason': 'balanced_new_batch' if probe['eligible'] else 'preserved_native_opt_out'}
            target = binding['target_background']
            result = probe if target == source else native(plan, realization, target_background=target, palette_config=palette_config)
            result['allocation_binding'] = deepcopy(binding)
            if saved:
                if digest(result) != receipt['result_hash']:
                    raise IntegrityError('cached palette materialization no longer matches its receipt')
            else:
                receipt = {**binding, 'result_hash': digest(result)}
                db.execute('UPDATE candidates SET request_hash=?,receipt=? WHERE task_id=?',
                           (request_hash, json.dumps(receipt, sort_keys=True), tid))
        return result

    def summary(self, *, window=None):
        # A plain read: under WAL one SELECT sees a committed snapshot without the write lock, so a run's thousands of
        # terminal window checks never queue behind materialize/write_report (v1-weak-c2's 'database is locked', 2026-10-28).
        db = sqlite3.connect(self.path, timeout=BUSY_SECONDS)
        try:
            rows = db.execute('SELECT task_id,account,window,receipt FROM candidates ORDER BY task_id').fetchall()
        finally:
            db.close()
        return self._summarize(rows, window=window)

    def write_report(self):
        # Serialize snapshot+write too: a slower group must not overwrite a
        # newer batch report with an older snapshot after both have finished.
        with self.transaction() as db:
            rows = db.execute('SELECT task_id,account,window,receipt FROM candidates ORDER BY task_id').fetchall()
            value = self._summarize(rows)
            atomic(self.root/'palette_report.private.json', value)
        return value

    def _summarize(self, rows, *, window=None):
        if window is not None: rows = [r for r in rows if r[2] == window]
        receipts = [json.loads(r[3]) for r in rows if r[3]]
        counts = Counter(r['actual_background'] for r in receipts if r['actual_background'] is not None)
        n = sum(counts.values()); preferred = sum(counts[c] for c in PREFERRED)
        return {'policy': deepcopy(POLICY), 'context_hash': self.context_hash,
            'registered': len(rows), 'materialized': len(receipts),
            'unmaterialized_task_ids': [r[0] for r in rows if not r[3]],
            'known_background_count': n, 'actual_background_counts': {str(c):counts[c] for c in range(10)},
            'preferred_combined_fraction': preferred/n if n else None,
            'preferred_range_met': (POLICY['preferred_combined_desired_range_percent'][0]/100 <= preferred/n
                <= POLICY['preferred_combined_desired_range_percent'][1]/100) if n else None,
            'unsafe_preserved': sum(r['has_grid'] and not r['eligible'] for r in receipts),
            'unresolved_backgrounds': sum(r['has_grid'] and r['actual_background'] is None for r in receipts),
            'without_grids': sum(not r['has_grid'] for r in receipts),
            'receipts': receipts}
