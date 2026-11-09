"""Prepare, qualify, authorize and run one batch. Preparing or qualifying never makes a model call.

  prepare    freeze the config, the design (allocation), the budget, the bank view (bank_inputs.py), the reference
             embeddings, wave A's idea requests and the Codex launcher
  qualify    check the frozen reference index covers the corpus and every prepared idea prompt fits its context
  ready      prepare + qualify + the offline test gate for the current code; writes the launch manifest draft
  authorize  record the launch instruction (the command line) and a usage reading
  run        launch (or resume) under that authorization; ends with the terminal report
"""
from pathlib import Path
import os
import shutil
from .common import *
from .budget import Budget
from .allocation import DEFAULT_RUN, design
from . import bank_inputs, supply, recent
from .preflight import capacity


def _seed_shared_cache(provider_dir):
    """Copy embeddings a bank ships (<bank>/embeddings/<config hash>/) into the workspace cache; never overwrite."""
    from arcgen.paths import BANK, CACHE
    source = BANK / 'embeddings' / provider_dir.name
    if not source.is_dir():
        return 0
    copied = 0
    for p in source.rglob('*.json'):
        dest = CACHE / 'embeddings' / provider_dir.name / p.relative_to(source)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest); copied += 1
    return copied


def _share_new_vectors(root):
    """Add the run's newly computed reference vectors to the workspace cache, so the next run need not embed them."""
    from arcgen.paths import CACHE
    added = 0
    for p in sorted((root / 'retrieval_cache').rglob('*.json')):
        dest = CACHE / p.relative_to(root / 'retrieval_cache')
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest); added += 1
    return added


def cached_index(root, inputs):
    """The run's reference index over the frozen corpus. Missing reference vectors are embedded here, at preparation
    (the only embedding requests before launch); candidate queries are embedded during the run (retrieval.py)."""
    from arcgen.lib.embeddings import build_embedding_provider
    from .retrieval import ReferenceIndex, Retrieval
    context = read(inputs['selector_context'])
    probe = build_embedding_provider(context['embedding'], root / 'retrieval_cache')
    _seed_shared_cache(probe.cache_dir)
    native = Retrieval(root, inputs)  # copies cached vectors of the corpus from the workspace cache
    texts = list(dict.fromkeys(r['central_relation'] for r in context['references']))
    missing = [t for t in texts if not native.provider._path(native.provider._key(t)).exists()]
    if missing:
        native.embed(missing)
    _share_new_vectors(root)

    class Cached:
        """Read-only view of the run cache: a cache miss never falls through to the network here."""
        def embed(self, values):
            out = []
            for text in values:
                key = native.provider._key(text); path = native.provider._path(key)
                if not path.exists():
                    raise IntegrityError('missing cached reference vector; no network fallback during preparation')
                out.append(native.provider._read(path, key=key, text=text))
            return out
    return ReferenceIndex(context['references'], Cached(), provider_identity={
        'provider_config_hash': native.provider._config_hash, 'mode': 'read_only_run_cache'})


def prepare_wave(root, wave, inputs, *, index):
    """Freeze one reference wave's idea requests (v1 loop calls carry no bank pack, calibration or pins)."""
    path = root / 'design/waves' / wave / 'prepared.private.json'
    if path.exists():
        value = read(path); verify_refs(value)
        return value
    plan = read(root / 'design/batch.private.json')
    recent.for_wave(root, wave, inputs)
    requests = {}
    for w in plan['windows']:
        if w['wave'] != wave: continue
        calls = {}
        for call in w['calls']:
            if call['bank_setting'] != 'loop':
                raise IntegrityError('this release runs the v1 idea loop only (bank_setting "loop"): ' + call['opaque_id'])
            calls[call['opaque_id']] = supply.idea_request(call, inputs['contexts'][w['context']]['description'], inputs)
        dest = root / 'design/windows' / w['id'] / 'ideas.private.json'; frozen(dest, calls); requests[w['id']] = ref(dest)
    return frozen(path, dict(wave=wave, inputs=ref(root / 'design/waves' / wave / 'inputs.private.json'),
                             recent_snapshot=ref(root / 'design/recent' / f'{wave}.private.json'), packs=[],
                             idea_requests=requests, new_text_calls=0))


def prepare(root=DEFAULT_RUN):
    root = isolated_root(root)
    from .config import CONFIG
    import json as _json
    frozen(root / 'config.json', _json.loads(Path(CONFIG['_source']).read_text()))
    plan = frozen(root / 'design/batch.private.json', design()); Budget(root, plan['limits'])
    inputs = bank_inputs.prepare(root)
    prepared = root / 'prepared_manifest.private.json'
    if prepared.exists():
        value = read(prepared); verify_refs(value); verify_sources(root); return value
    index = cached_index(root, inputs)
    recent.freeze_snapshot(root, 'A', inputs)
    prepare_wave(root, 'A', inputs, index=index)
    from arcgen.lib.factory_runtime import _runtime_config, _bind_account_launcher
    launchers = {}
    for account in HOMES:
        conf = config_for(root / 'accounts' / account / 'preparation', account)
        launcher = _bind_account_launcher(_runtime_config(conf), Path(conf['_output_directory']) / 'policy')
        launchers[account] = {'launcher': launcher, 'executable': ref(launcher['resolved_executable']),
                              'stage_bindings': conf['role_profile']}
    transport = frozen(root / 'transport_preflight.json', {'astra': launchers, 'new_text_calls': 0})
    for account in HOMES:
        if transport['astra'][account]['executable'] != executable():
            raise IntegrityError('launcher does not resolve to the configured Codex executable')
    verify_sources(root); capacity(root, require_idle=False)
    value = dict(status='prepared_not_launched', revision=REVISION, design=ref(root / 'design/batch.private.json'),
                 inputs=ref(root / 'inputs/revision_inputs.private.json'), wave_a=ref(root / 'design/waves/A/prepared.private.json'),
                 transport=ref(root / 'transport_preflight.json'), research_status='disabled_by_design', new_text_calls=0)
    frozen(prepared, value); atomic(root / 'STATUS.json', value); return value


def qualify(root=DEFAULT_RUN):
    prepare(root)
    from .qualification import qualify as qualify_selector
    inputs = read(root / 'inputs/revision_inputs.private.json')
    return qualify_selector(root, inputs=inputs, index=cached_index(root, inputs))


def ready(root=DEFAULT_RUN):
    prepare(root)
    from arcgen.paths import GATE
    q = qualify(root); gate = read(GATE / 'offline_gate.json') if (GATE / 'offline_gate.json').exists() else {}
    if q['status'] != 'passed' or q['code_hash'] != code_identity():
        raise IntegrityError('current source requires passing retrieval and prompt qualification')
    gated = gate.get('status') == 'passed' and gate.get('code_hash') == code_identity()
    if not gated and os.environ.get('ARCGEN_SKIP_TEST_GATE') != '1':
        raise IntegrityError('the offline test gate has not passed for this code: run `bin/arcgen test` '
                             '(or `bin/arcgen start --skip-tests`, which records that the gate was skipped)')
    from .retrieval import Retrieval
    inputs = read(root / 'inputs/revision_inputs.private.json'); native = Retrieval(root, inputs)
    frozen(root / 'reference_cache_manifest.private.json', {'vectors': [ref(native.provider._path(native.provider._key(r['central_relation'])))
        for r in read(inputs['selector_context'])['references']]})
    capacity(root, require_idle=False); verify_sources(root)
    plan = design(); budget = Budget(root)
    from .config import CONFIG, science_hash
    return frozen(root / 'prepared_launch_manifest.private.json', dict(revision=REVISION, status='ready_not_launched', launch_authorized=False,
        accounts=sorted(CALL_HOMES), config=ref(root / 'config.json'), config_science_hash=science_hash(CONFIG),
        code_hash=code_identity(), code=code_manifest(), runtime_policy=POLICY, budget_hash=digest(budget.limits), budget=budget.limits,
        window_ids=[w['id'] for w in plan['windows'] + plan['potential_groups']],
        reference_snapshot_hash=q['reference_snapshot_hash'], palette_context_hash=digest(read(root / 'inputs/palette_context.private.json')),
        prepared=ref(root / 'prepared_manifest.private.json'), selector_qualification=ref(root / 'selector_qualification.private.json'),
        research_status='disabled_by_design', planner_comparison=plan['planner_comparison'], executable=executable(),
        preparation_text_calls={'logical': 0, 'physical_attempts': 0, 'batch_generation_calls': 0,
            'launch_route_probes': 'transport_preflight calls at launch, before any idea call'},
        test_gate='passed' if gated else 'skipped_by_owner',
        reference_cache=ref(root / 'reference_cache_manifest.private.json'),
        authorization='Prepared design. Record the current user launch instruction with authorize before dispatch.',
        quota_reset_never_refills_call_budget=True,
        conditional_allocation='Shared hard category ceiling with complete-package reservations; release unused packages after definitive stage completion.',
        automatic_commit=False, automatic_extension=False))


def run(root=DEFAULT_RUN, *, mock=False):
    """Launch or resume. mock=True runs every model call through FactoryModelInvoker._mock_response (tests only)."""
    from .quota import Quota, UsagePaused
    from .coordinator import run as coordinate
    from .preflight import model_probes
    root = isolated_root(root); budget = Budget(root); manifest = require_launch(root, budget)
    verify_refs(manifest); verify_sources(root); capacity(root, require_idle=not OPERATIONS.get('allow_concurrent_activity', False))
    if (root / 'STOP_DISPATCH.json').exists(): raise IntegrityError('operational stop needs explicit resolution')
    if (root / 'completion.json').exists():
        from .report import verify_terminal
        return verify_terminal(root)
    transport = read(root / 'transport_preflight.json')['astra']
    budget.quotas = {a: Quota(root, transport[a]['executable']['path'], account=a, mock=mock) for a in HOMES}
    try:
        for quota in budget.quotas.values(): quota.start()
        inputs = read(root / 'inputs/revision_inputs.private.json')
        model_probes(root, budget, manifest, inputs['production'], mock=mock)
        result = coordinate(root, budget=budget, manifest=manifest, inputs=inputs, mock=mock)
        before = budget.summary(); resumed = coordinate(root, budget=budget, manifest=manifest, inputs=inputs, mock=mock); after = budget.summary()
        if result != resumed or any(before[k] != after[k] for k in ('logical_calls', 'physical_attempts', 'developments')):
            raise IntegrityError('terminal replay consumed new calls')
        frozen(root / 'terminal_resume_verification.private.json', {'status': 'verified', 'terminal_hash': digest(result),
            'new_text_calls_on_resume': 0, 'logical_calls': after['logical_calls'], 'physical_attempts': after['physical_attempts']})
        from .report import finish
        return finish(root)
    except UsagePaused as exc:
        value = {'status': 'paused_usage', 'reason': str(exc), 'accounts': sorted(HOMES), 'progress_preserved': True,
                 'resume_command': 'bin/arcgen resume', 'budgets_refilled': False}
        atomic(root / 'STATUS.json', value); return value


def observe_quota(root=DEFAULT_RUN, *, mock=False):
    from .quota import Quota
    transport = read(root / 'transport_preflight.json')['astra']
    values = {a: Quota(root, transport[a]['executable']['path'], account=a, mock=mock).observe(force=True) for a in HOMES}
    result = dict(status='observed', accounts=values, new_text_calls=0, must_recheck_at_launch=True)
    atomic(root / 'capacity_quota_preflight.json', result)
    return result


def authorize(root=DEFAULT_RUN, *, authorization, mock=False):
    if not authorization or not authorization.strip(): raise IntegrityError('explicit launch instruction required')
    # Preparation can overlap unrelated work; dispatch needs an idle account unless the run allows concurrent activity.
    capacity(root, require_idle=not OPERATIONS.get('allow_concurrent_activity', False))
    value = ready(root)
    observation = observe_quota(root, mock=mock)
    if not OPERATIONS.get('usage_credits', False) and any(r['weekly_remaining_percent'] <= 0 for r in observation['accounts'].values()):
        raise IntegrityError('account usage unavailable before launch')
    return frozen(root / 'launch_manifest.private.json', {**value, 'status': 'authorized',
        'launch_authorized': True, 'authorization': authorization})
