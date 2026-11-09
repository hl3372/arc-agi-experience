"""Launch preflight: other Codex activity on the account, and the route probes run first at launch."""
from pathlib import Path
import os
import time
from .common import ROOT, HOMES, CALL_HOMES, IntegrityError, atomic


def capacity(root, *, require_idle=True):
    """Codex calls already running on the configured account (same CODEX_HOME), and other pipeline coordinators in
    this workspace. Only process command lines and CODEX_HOME are read; never a credential file."""
    homes = {str(Path(h).resolve()) for h in CALL_HOMES.values()}
    default_home = str(Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex').expanduser().resolve())
    active, other, coordinators = [], [], []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            argv = (p / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not argv or not argv[0]:
                continue
            if any(Path(a).name == 'codex' for a in argv[:3]) and 'exec' in argv:
                found = [v.split(b'=', 1)[1].decode(errors='replace') for v in
                         (p / 'environ').read_bytes().split(b'\0') if v.startswith(b'CODEX_HOME=')]
                home = str(Path(found[0]).resolve()) if len(found) == 1 else default_home
                (active if home in homes else other).append({'pid': int(p.name), 'kind': 'codex_exec', 'same_account': home in homes})
            if Path(argv[0]).name.startswith('python') and any(a.endswith('/entry.py') for a in argv[1:]) and 'run' in argv:
                if (p / 'cwd').resolve().is_relative_to(ROOT) or any(str(ROOT) in a for a in argv):
                    coordinators.append({'pid': int(p.name), 'kind': 'potential_pipeline_coordinator'})
        except (OSError, PermissionError, ProcessLookupError):
            continue
    value = {'checked_at_epoch': time.time(), 'accounts': sorted(CALL_HOMES),
             'active_text_consumers': active, 'other_account_consumers': other,
             'potential_coordinators': coordinators,
             'capacity_clear': not active and not coordinators,
             'new_text_calls': 0, 'must_recheck_at_launch': True, 'require_idle': require_idle}
    atomic(Path(root) / 'launch_process_preflight.json', value)
    if require_idle and not value['capacity_clear']:
        raise IntegrityError('other Codex calls or another pipeline run are active on this account; wait for them, or set '
                             'operations.allow_concurrent_activity')
    return value


def model_probes(root, budget, manifest, production, *, mock=False):
    """Twelve bounded routing probes on the pinned release; completed receipts are reused on resume."""
    import itertools
    from arcgen.lib.e18_pre_e20_pool import DedicatedCallPool
    from .allocation import design, PHASE
    from .common import IDEA_MODELS, digest, frozen, ref, read, verify_refs, account_pool
    from .pipeline import window_config
    from .runtime import make_runtime
    path = Path(root)/'model_transport_qualification.private.json'
    if path.exists():
        value = read(path); verify_refs(value)
        if value['status'] != 'passed': raise IntegrityError('model routing qualification failed')
        return value
    rows = []; windows = design()['windows']
    # Serial tiny probes avoid spending task calls on an unavailable route.
    for account in HOMES:
        pool = account_pool(Path(root)/('preflight_pool-'+account+'.json'))
        window = next((w for w in windows if w['account'] == account), None)
        if window is None:  # saved-queue designs may have no window on this account; probe under a synthetic identity
            base = next(w for w in windows if w['account'])
            window = {**base, 'id': account + '-routing-probe', 'account': account}
        try:
            for model, effort in itertools.product(IDEA_MODELS, ('high', 'xhigh', 'max')):
                conf = window_config(Path(root), window, production, mock=mock)
                conf['role_profile']['transport_probe'].update(model=model, reasoning_effort=effort)
                conf['_profile_hash'] = digest(conf['role_profile'])
                conf['_config_sha256'] = digest({'base': conf['_config_sha256'], 'probe_model': model, 'effort': effort})
                conf['_run_identity']['config_sha256'] = conf['_config_sha256']
                rt = make_runtime(conf, manifest, pool, budget, PHASE, window['id'], 'routing-probe-'+model+'-'+effort,
                                  'selection_codex', category='transport_preflight')
                cid = account+'-'+model+'-'+effort+'-routing-probe'
                prompt = 'Return a JSON object with probe_id equal to '+cid+' and ok equal to true. This is a transport check.'
                result = rt.invoke_rendered(purpose='transport_probe', stage='factory_transport_probe', call_id=cid, prompt=prompt,
                    prompt_template_version='routing-probe-v1', prompt_template_hash=digest(prompt), parent_ids=[])
                if result.value != {'probe_id': cid, 'ok': True}: raise IntegrityError('transport probe returned incorrect identity')
                receipt = read(result.log_dir/'program_transport_identity.json')
                if not receipt['verified'] or receipt['account'] != account: raise IntegrityError('probe route identity failed')
                rows.append({'effort': effort, 'model': model, 'account': account,
                    'identity_receipt': ref(result.log_dir/'program_transport_identity.json'),
                    'ledger': ref(rt.run_dir/'state.json'), 'logical_calls': 1, 'physical_attempts': result.attempts})
        finally: pool.finish()
    return frozen(path, {'status': 'passed', 'probes': rows, 'new_logical_calls': len(rows),
        'scope': 'Routing and structured transport only; no task-quality claim.'})
