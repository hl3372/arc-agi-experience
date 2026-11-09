"""Account ceilings and simple completion reservations in the existing ledger."""
import time, json, threading, sqlite3
from contextlib import contextmanager
from .common import Budget as NativeBudget, BudgetError, IntegrityError, HOMES, CALL_HOMES, canonical


def primary_allowance(spec, account, owner, tasks):
    return spec.get('primary_calls_per_account', {}).get(account,
        spec.get('primary_call_allowance', 4*tasks) if account == owner else 0)


class Budget(NativeBudget):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS execution_groups (window TEXT PRIMARY KEY, account TEXT, tasks INTEGER, identity TEXT, development_done INTEGER DEFAULT 0, research_done INTEGER DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS review_packages (key TEXT PRIMARY KEY, phase TEXT, account TEXT, window TEXT, task_id TEXT, requested INTEGER, remaining INTEGER, closed INTEGER DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS review_call_packages (call_key TEXT PRIMARY KEY, package_key TEXT)')
            db.execute('CREATE INDEX IF NOT EXISTS calls_by_account_window_category ON calls(account, window, category)')

    @contextmanager
    def transaction(self):
        """A write transaction that waits out a busy lock (retried with backoff) instead of failing the run."""
        for attempt in range(6):
            db = sqlite3.connect(self.path, timeout=60)
            try:
                db.execute('PRAGMA journal_mode=WAL'); db.execute('BEGIN IMMEDIATE')
            except sqlite3.OperationalError as exc:
                db.close()
                if 'locked' not in str(exc) or attempt == 5: raise
                time.sleep(5 * (attempt + 1)); continue
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback(); raise
            finally:
                db.close()
            return

    @contextmanager
    def reading(self):
        """A consistent read snapshot (WAL) without taking the write lock."""
        db = sqlite3.connect(self.path, timeout=60)
        try:
            db.execute('BEGIN'); yield db
        finally:
            db.close()

    def summary(self):
        from collections import Counter
        with self.reading() as db:
            db.row_factory = sqlite3.Row
            calls = [dict(r) for r in db.execute('SELECT * FROM calls ORDER BY key')]
            attempts = [dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY key,ordinal')]
            developments = [dict(r) for r in db.execute('SELECT * FROM developments ORDER BY key')]
        return {'logical_calls':len(calls),'physical_attempts':len(attempts),'developments':len(developments),
            'by_phase_category':dict(Counter(r['phase']+'/'+r['category'] for r in calls)),
            'by_provider_account':dict(Counter(r['provider']+'/'+r['account'] for r in calls)),
            'calls':calls,'attempts':attempts,'development_rows':developments,'limits':self.limits}

    def reserve_review(self, *, window, account, phase, key, count):
        if account not in HOMES:raise IntegrityError('review account is not authorized')
        if not isinstance(count,int) or not 1<=count<=3:raise IntegrityError('review package must reserve one to three calls')
        identity=window+'/'+key;task_id=key.split(':',1)[0]
        with self.transaction() as db:
            old=db.execute('SELECT phase,account,window,task_id,requested FROM review_packages WHERE key=?',(identity,)).fetchone()
            fields=(phase,account,window,task_id,count)
            if old:
                if old!=fields:raise IntegrityError('review package identity changed on resume')
                return True
            for condition,args,cap in [
                ('phase=?',(phase,),self.phase(phase)['categories']['conditional_evidence_repair']),
                ('account=?',(account,),self.limits.get('accounts',{}).get(account,{}).get('categories',self.phase(phase)['categories'])['conditional_evidence_repair'])]:
                spent=db.execute('SELECT count(*) FROM calls WHERE category=? AND '+condition,('conditional_evidence_repair',*args)).fetchone()[0]
                pending=db.execute('SELECT coalesce(sum(remaining),0) FROM review_packages WHERE closed=0 AND '+condition,args).fetchone()[0]
                if spent+pending+count>cap:return False
            db.execute('INSERT INTO review_packages VALUES(?,?,?,?,?,?,?,0)',(identity,*fields,count))
            return True

    def release_reviews(self,window,task_id):
        """Release only undispatched logical slots; never erase calls or attempts."""
        with self.transaction() as db:
            db.execute('UPDATE review_packages SET remaining=0,closed=1 WHERE window=? AND task_id=?',(window,task_id))

    def review_snapshot(self,window):
        with self.reading() as db:
            return [dict(zip(('key','requested','remaining','closed','charged'),r)) for r in db.execute(
                'SELECT p.key,p.requested,p.remaining,p.closed,(SELECT count(*) FROM review_call_packages c WHERE c.package_key=p.key) FROM review_packages p WHERE window=? ORDER BY p.rowid',(window,))]

    def _spent(self,db,account,window,category):
        return db.execute('SELECT count(DISTINCT c.key) FROM calls c JOIN attempts a ON a.key=c.key WHERE c.account=? AND c.window=? AND c.category=?',(account,window,category)).fetchone()[0]

    def _future(self,db,account):
        """One physical attempt for every still-possible reserved logical call.

        A second attempt may consume only the explicit spare attempt budget.
        Completion releases unused conditional slots; category allowances never transfer.
        """
        if 'accounts' not in self.limits:return 0
        # Two grouped queries instead of two per execution group: this runs inside the write lock on every call, and
        # per-group queries held the lock long enough (GIL convoy under load) for other threads to fail "locked".
        by_cat=dict(db.execute('SELECT c.category,count(DISTINCT c.key) FROM calls c JOIN attempts a ON a.key=c.key '
                               'WHERE c.account=? GROUP BY c.category',(account,)).fetchall())
        by_win={(w,c):n for w,c,n in db.execute('SELECT c.window,c.category,count(DISTINCT c.key) FROM calls c JOIN attempts a '
            'ON a.key=c.key WHERE c.account=? AND c.category IN (?,?) GROUP BY c.window,c.category',
            (account,'primary_downstream','completed_research')).fetchall()}
        needed=0
        for category in ('transport_preflight','ideas','selectors'):
            needed+=max(0,self.limits['accounts'][account]['categories'][category]-by_cat.get(category,0))
        for window,owner,tasks,identity,dev_done,research_done in db.execute('SELECT * FROM execution_groups').fetchall():
            spec=json.loads(identity)
            for cat,cap in [('primary_downstream',primary_allowance(spec,account,owner,tasks) if not dev_done else 0),
                            ('completed_research',spec.get('research_calls_per_account', {}).get(account, spec.get('research_call_allowance',4)) if not research_done else 0)]:
                needed+=max(0,cap-by_win.get((window,cat),0))
        # One shared allowance replaces independent five-call group envelopes.
        cap=self.limits['accounts'][account]['categories']['conditional_evidence_repair']
        needed+=max(0,cap-by_cat.get('conditional_evidence_repair',0))
        return needed

    def reserve_group(self,window):
        wid,account,n=window['id'],window['account'],window['maximum_developments'];identity=canonical(window)
        if account not in HOMES:raise IntegrityError('development account is not authorized')
        if not 1<=n<=6:raise IntegrityError('execution group must contain one to six tasks')
        with self.transaction() as db:
            old=db.execute('SELECT account,tasks,identity FROM execution_groups WHERE window=?',(wid,)).fetchone()
            if old:
                if old!=(account,n,identity):raise IntegrityError('reserved group changed on resume')
                return True
            per=self.limits['accounts'][account]
            count,total=db.execute('SELECT count(*),coalesce(sum(tasks),0) FROM execution_groups WHERE account=?',(account,)).fetchone()
            if count>=per['maximum_execution_groups'] or total+n>per['maximum_developments']:return False
            if db.execute('SELECT count(*) FROM execution_groups').fetchone()[0]>=self.limits['maximum_execution_groups']:return False
            for acct in self.limits['accounts']:
                used=db.execute('SELECT count(*) FROM attempts a JOIN calls c ON a.key=c.key WHERE c.account=?',(acct,)).fetchone()[0]
                extra=primary_allowance(window,acct,account,n)+window.get('research_calls_per_account', {}).get(acct, window.get('research_call_allowance',0))
                if used+self._future(db,acct)+extra>self.limits['accounts'][acct]['physical_attempt_cap']:return False
            db.execute('INSERT INTO execution_groups(window,account,tasks,identity) VALUES(?,?,?,?)',(wid,account,n,identity))
            return True

    def release_stage(self,window,stage):
        if stage not in ('development','research'):raise IntegrityError('unknown completion stage')
        with self.transaction() as db:db.execute(f'UPDATE execution_groups SET {stage}_done=1 WHERE window=?',(window,))

    def admit(self,key,*,phase,category,provider,account,window,fingerprint):
        if provider!='astra' or account not in CALL_HOMES:raise IntegrityError('only the configured Codex account is authorized')
        if 'accounts' not in self.limits:return super().admit(key,phase=phase,category=category,provider=provider,account=account,window=window,fingerprint=fingerprint)
        p=self.phase(phase);fields=(phase,category,provider,account,window,fingerprint)
        if provider!='astra' or account not in CALL_HOMES:raise IntegrityError('unbound route')
        if category not in p['categories']:raise BudgetError('unbudgeted category')
        with self.transaction() as db:
            old=db.execute('SELECT phase,category,provider,account,window,fingerprint FROM calls WHERE key=?',(key,)).fetchone()
            if old:
                if old!=fields:raise IntegrityError('resumed logical request changed')
                return
            if category in ('primary_downstream','conditional_evidence_repair','completed_research'):
                group=db.execute('SELECT account,tasks,development_done,research_done,identity FROM execution_groups WHERE window=?',(window,)).fetchone()
                if not group:raise BudgetError('downstream requires its completion reservation')
                if category == 'conditional_evidence_repair' and group[0] != account:
                    raise BudgetError('reviews require their development owner')
                if (group[3] if category=='completed_research' else group[2]):raise BudgetError('completed stage cannot admit a new logical call')
                spec=json.loads(group[4])
                cap=spec.get('research_calls_per_account', {}).get(account, 0) if category=='completed_research' else self.limits['accounts'][account]['categories']['conditional_evidence_repair'] if category=='conditional_evidence_repair' else primary_allowance(spec,account,group[0],group[1])
                if db.execute('SELECT count(*) FROM calls WHERE account=? AND window=? AND category=?',(account,window,category)).fetchone()[0]>=cap:raise BudgetError('reserved group category ceiling')
            checks=[('SELECT count(*) FROM calls',(),self.limits['logical_cap']),
                    ('SELECT count(*) FROM calls WHERE phase=?',(phase,),p['logical_cap']),
                    ('SELECT count(*) FROM calls WHERE phase=? AND category=?',(phase,category),p['categories'][category]),
                    ('SELECT count(*) FROM calls WHERE account=?',(account,),self.limits['accounts'][account]['logical_cap']),
                    ('SELECT count(*) FROM calls WHERE account=? AND category=?',(account,category),self.limits['accounts'][account]['categories'][category])]
            for query,args,cap in checks:
                if db.execute(query,args).fetchone()[0]>=cap:raise BudgetError('logical/category/account ceiling')
            if category=='conditional_evidence_repair':
                task_id=key.rsplit('/',1)[-1].split('-',1)[0]
                package=db.execute('SELECT key FROM review_packages WHERE phase=? AND account=? AND window=? AND task_id=? AND closed=0 AND remaining>0 ORDER BY rowid LIMIT 1',(phase,account,window,task_id)).fetchone()
                if not package:raise BudgetError('conditional call lacks a complete-package reservation')
                db.execute('UPDATE review_packages SET remaining=remaining-1 WHERE key=?',package)
                db.execute('INSERT INTO review_call_packages VALUES(?,?)',(key,package[0]))
            db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?,?,?)',(key,*fields,time.time(),'admitted',None))

    def start_attempt(self,key):
        if 'accounts' not in self.limits:return super().start_attempt(key)
        with self.transaction() as db:
            row=db.execute('SELECT phase,account,window,category FROM calls WHERE key=?',(key,)).fetchone()
            if not row:raise BudgetError('attempt lacks logical admission')
            phase,account,window,category=row
            n=db.execute('SELECT count(*) FROM attempts WHERE key=?',(key,)).fetchone()[0]
            total=db.execute('SELECT count(*) FROM attempts').fetchone()[0]
            used=db.execute('SELECT count(*) FROM attempts a JOIN calls c ON a.key=c.key WHERE c.account=?',(account,)).fetchone()[0]
            if n>=2 or total>=self.limits['physical_attempt_cap'] or used>=self.limits['accounts'][account]['physical_attempt_cap']:raise BudgetError('physical attempt ceiling')
            # Insert inside the transaction before calculating obligations; a first
            # attempt consumes one of its own reserved slots, a retry does not.
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?)',(key,n+1,time.time(),None,'started',None))
            if used+1+self._future(db,account)>self.limits['accounts'][account]['physical_attempt_cap']:raise BudgetError('attempt would consume completion reservations')
            return n+1
