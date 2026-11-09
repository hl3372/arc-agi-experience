"""Frozen terminal/accounting receipts; no admission or generation in reporting."""
import time
from .common import *
from .budget import Budget
from .batch_report import report
from .routing import POLICY


def verify_routes(root):
    n=0
    for path in (root/'accounts').glob('*/*/*/state.json'):
        account = path.parents[2].name
        if account not in HOMES: raise IntegrityError('unknown account directory')
        for c in read(path).get('call_ledger',{}).values():
            route=read(path.parent/'routing'/c['stage']/(c['call_id']+'.json'))
            expected_account=account
            if route['account']!=expected_account or c['account']!=expected_account or any(c[k]!=route[k] for k in ('model','reasoning_effort')):
                raise IntegrityError('native route differs from the frozen policy')
            if c['purpose']=='planning':
                from .rungs import task_of_plan_call
                study=read(path)['semantic_records'][task_of_plan_call(c['call_id'])]['study']
                if route.get('planner')!=study['planner'] or {k:route[k] for k in ('model','reasoning_effort')}!=POLICY['planners'][study['planner']]:
                    raise IntegrityError('planning call differs from its frozen planner assignment')
            if c['status']=='complete':
                directory=Path(c['log_dir']); request=read(directory/'request.json')
                identity=read(directory/'program_transport_identity.json')
                if not identity['verified'] or identity['account']!=expected_account: raise IntegrityError('unverified physical route')
                if request['model']['name']!=route['model'] or request['model']['reasoning_effort']!=route['reasoning_effort']:
                    raise IntegrityError('physical request model/effort changed')
            n+=1
    return n


def finish(root):
    result=report(root)
    if result['status']!='complete_pending_semantic_synthesis': raise IntegrityError('nonterminal pilot cannot finish')
    checked=verify_sources(root); calls=verify_routes(root)
    frozen(root/'terminal_artifacts.private.json',[ref(p) for sub in ('accounts','research')
        for p in sorted((root/sub).rglob('*')) if p.is_file() and p.suffix not in ('.sqlite3','-wal','-shm')])
    frozen(root/'completion.json',{'status':'complete_pending_semantic_synthesis','completed_at_epoch':time.time(),
        'report_hash':digest(result),'admissions':0,'new_historical_case_reviews':0,
        'source_files_checked':checked,'route_records_checked':calls})
    return verify_terminal(root)


def verify_terminal(root):
    budget=Budget(root); require_launch(root,budget); verify_sources(root)
    report_value=read(root/'analysis/terminal_report.private.json')
    if digest(report_value)!=read(root/'completion.json')['report_hash']: raise IntegrityError('terminal report changed')
    verify_refs(read(root/'terminal_artifacts.private.json')); verify_routes(root)
    s=budget.summary()
    if any(c['account'] not in CALL_HOMES for c in s['calls']) or any(a['ordinal']>2 for a in s['attempts']):
        raise IntegrityError('account or two-attempt limit exceeded')
    if any(s[k]>budget.limits[cap] for k,cap in [('logical_calls','logical_cap'),('physical_attempts','physical_attempt_cap'),('developments','maximum_developments')]):
        raise IntegrityError('pilot budget exceeded')
    value=dict(status='verified_terminal',counts=report_value['counts'],logical_calls=s['logical_calls'],
        physical_attempts=s['physical_attempts'],developments=s['developments'],new_calls_on_terminal_resume=0,
        accounts=sorted(CALL_HOMES),admissions=0,semantic_synthesis='pending',research_coverage=report_value['research_coverage'])
    atomic(root/'terminal_verification.json',value); atomic(root/'STATUS.json',value)
    return value
