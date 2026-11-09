"""Bind the new supply/selector to the unchanged native downstream DAG and pools."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from copy import deepcopy
from contextlib import nullcontext
from arcgen.lib.factory_effort import runtime_profile
from arcgen.lib.factory_window import window_bindings
from arcgen.lib.model import ModelInvocationError
from arcgen.astra_broad_arm_program_20260908.runtime import Runtime as NativeRuntime
from arcgen.astra_broad_arm_program_20260908.program import invoke
from arcgen.astra_broad_arm_program_20260908 import reviews
from .routing import SOL
from .common import (REVISION, IntegrityError, isolated_root, config_for, read, frozen,
                     digest, code_identity, require_launch, verify_refs)
from arcgen.astra_usage_diversity_20260913 import selection
from . import downstream
from . import supply
from . import idea_v1


from .runtime import Runtime, make_runtime
from arcgen.astra_targeted_followup_20260911.pipeline import DispatchPaused


def window_config(root, window, production=None, *, mock=False):
    root = isolated_root(root)
    conf = config_for(root / 'accounts' / window['account'] / window['id'],
                      window['account'], seed=window['seed'], mock=mock)
    mode = window.get('editorial_mode', 'enabled')
    if mode not in ('enabled', 'bypass'):
        raise IntegrityError('editorial_mode must be enabled or bypass')
    conf['_experiment_profile']['editorial_mode'] = mode
    conf['_config_sha256'] = digest({'base': conf['_config_sha256'], 'editorial_mode': mode})
    conf['_run_identity']['config_sha256'] = conf['_config_sha256']
    if production:
        conf['_paths'].update(active_registry=root / 'inputs/production/committed_tasks/active_registry.json')
    return conf


def prepare_window(root, window, requests, inputs, index, *, pool, budget, production,
                   manifest, mock=False, invoke_call=invoke, on_ready=None, events=None, on_failure=None):
    """Freeze supply and allocation independently of downstream availability."""
    root = isolated_root(root)
    if (root / 'STOP_DISPATCH.json').exists(): raise IntegrityError('revision dispatch paused')
    verify_refs(inputs)
    if production != inputs['production']:
        raise IntegrityError('downstream production context differs from the prepared inputs')
    if not mock:
        require_launch(root, budget, window['id'], reference_snapshot_hash=digest(
            sorted(read(inputs['selector_context'])['references'], key=lambda r: r['reference_id'])))
    if window.get('kind') == 'eligible_group':
        from .common import ref
        if requests or window['calls']:
            raise IntegrityError('eligible groups cannot generate or select again')
        parent = root/'accounts'/(window.get('source_account') or window['account'])/window['source_window']
        queue_path = parent/'eligible_queue.private.json'
        queue = read(queue_path)
        allocation = selection.group_allocation(queue, window['group_ordinal'], group_size=window['maximum_developments'])
        if len(allocation['rows']) > window['maximum_developments']:
            raise IntegrityError('eligible group exceeds its frozen source capacity')
        conf = window_config(root, window, production, mock=mock)
        directory = Path(conf['_output_directory'])
        source = read(parent/'source_rows.private.json')
        frozen(directory/'source_queue.private.json', {'queue':ref(queue_path),
            'source':ref(parent/'source_rows.private.json'), 'group':window['group_ordinal']})
        frozen(directory/'window.private.json', {'window':window,'requests':{},
            'source_revision':code_identity(),'limits_hash':digest(budget.limits)})
        frozen(directory/'source_rows.private.json', {r['source_candidate_id']:source[r['source_candidate_id']]
                                                    for r in allocation['rows']})
        frozen(directory/'allocation.private.json', allocation)
        if on_ready and allocation['rows']: on_ready(window['id'])
        return allocation
    if window.get('kind') == 'saved_queue':
        from .saved_queue import materialize
        if requests or window['calls']:
            raise IntegrityError('saved queues cannot generate ideas or run a selector')
        conf = window_config(root, window, production, mock=mock)
        directory = Path(conf['_output_directory'])
        frozen(directory/'window.private.json', {'window': window, 'requests': {},
            'source_revision': code_identity(), 'limits_hash': digest(budget.limits)})
        allocation = materialize(window, directory)
        if on_ready and allocation['rows']: on_ready(window['id'])
        return allocation
    if window.get('kind') == 'saved_ideas':
        from .saved_ideas import load
        if requests or window['calls']:
            raise IntegrityError('saved ideas cannot generate ideas')
        conf = window_config(root, window, production, mock=mock)
        directory = Path(conf['_output_directory'])
        frozen(directory/'window.private.json', {'window': window, 'requests': {},
            'source_revision': code_identity(), 'limits_hash': digest(budget.limits)})
        source = load(window)['source_rows']
        frozen(directory/'source_rows.private.json', source)
        measure = lambda stage, **fields: events.span(stage, window=window['id'], account=window['account'], **fields) if events else nullcontext()
        return _select(root, window, source, inputs, index, conf, directory, pool=pool, budget=budget, manifest=manifest,
                       invoke_call=invoke_call, on_ready=on_ready, measure=measure)
    ids = [call['opaque_id'] for call in window['calls']]
    if len(ids) != len(set(ids)) or set(ids) != set(requests):
        raise IntegrityError('frozen idea request membership changed')
    kernels = [key for call in window['calls'] for key in call['opaque_kernel_ids']]
    if (len(kernels) > 48 or len(kernels) != len(set(kernels))
            or any(len(call['opaque_kernel_ids']) != (6 if call.get('commit') in ('kept', 'explore') else 4) for call in window['calls'])
            or type(window['maximum_developments']) is not int or not 0 <= window['maximum_developments'] <= 12
            or not 1 <= window['minimum_selector_eligible_kernels'] <= 48):
        raise IntegrityError('window exceeds the fixed supply/development contract')
    for call in window['calls']:
        req = requests[call['opaque_id']]
        private = req['private']
        if (call['context'] != window['context'] or call['window'] != window['id']
                or req['call_id'] != call['opaque_id'] + ('-sketch' if req['stage'] == idea_v1.STAGE_SKETCH else '')
                or any(private[k] != call[k] for k in ('recipe_id', 'effort', 'bank_setting', 'account', 'idea_model'))
                or private['account'] != window['account']):
            raise IntegrityError('idea/window recipe, account or effort mismatch')
    conf = window_config(root, window, production, mock=mock)
    directory = Path(conf['_output_directory'])
    frozen(directory / 'window.private.json', {'window': window,
           'requests': {k: {f: v for f, v in r.items() if f != 'response_validator'} for k, r in requests.items()},
           'source_revision': code_identity(), 'limits_hash': digest(budget.limits)})
    runtimes, responses = {}, {}
    measure = lambda stage, **fields: events.span(stage, window=window['id'], account=window['account'], **fields) if events else nullcontext()
    with measure('ideas'), ThreadPoolExecutor(max_workers=12) as executor:
        jobs = {}
        try:
            for call in window['calls']:
                req = requests[call['opaque_id']]
                if req['private']['account'] != window['account'] or call['account'] != window['account']:
                    raise IntegrityError('idea/window account mismatch')
                key = (req['purpose'], call['effort'], call['idea_model'])
                if key not in runtimes:
                    runtimes[key] = make_runtime(conf, manifest, pool, budget, window['phase'], window['id'],
                        'upstream-' + '-'.join(key), 'selection_codex', effort=key[:2], idea_model=call['idea_model'],
                        extra_bindings=idea_v1.stage_bindings() if req['stage'] == idea_v1.STAGE_SKETCH else None)
                rt = runtimes[key]
                if rt.role_profile[key[0]]['reasoning_effort'] != call['effort'] or rt.role_profile[key[0]]['model'] != call['idea_model']:
                    raise IntegrityError('actual idea model/effort differs from its frozen recipe')
                if req['stage'] == idea_v1.STAGE_SKETCH:
                    jobs[executor.submit(idea_v1.run_loop, rt, call, req, index, invoke_call, directory)] = call['opaque_id']
                    continue
                kwargs = {k: v for k, v in req.items() if k != 'private'}
                kwargs['response_validator'] = supply.idea_validator(call, req)
                jobs[executor.submit(invoke_call, rt, kwargs)] = call['opaque_id']
            for future in as_completed(jobs): responses[jobs[future]] = future.result()
        except BaseException as exc:
            if on_failure and not isinstance(exc, DispatchPaused): on_failure(exc)
            for future in jobs: future.cancel()
            raise  # Publish failure before executor shutdown drains running siblings.
    source = supply.source_rows(window, requests, responses)
    frozen(directory / 'source_rows.private.json', source)
    return _select(root, window, source, inputs, index, conf, directory, pool=pool, budget=budget, manifest=manifest,
                   invoke_call=invoke_call, on_ready=on_ready, measure=measure)


def _select(root, window, source, inputs, index, conf, directory, *, pool, budget, manifest, invoke_call, on_ready, measure):
    """Selector inputs, the selector call and the frozen allocation, for an idea window or a saved-ideas window."""
    packed_path = directory / 'selector_inputs.private.json'
    with measure('selector_retrieval', reused=packed_path.exists()):
        packed = read(packed_path) if packed_path.exists() else supply.selector_inputs(window, source, inputs, index)
        supply.verify_selector_inputs(window, source, inputs, packed, getattr(index, 'batch_references', ()), index)
        frozen(packed_path, packed)
    if packed['selection_eligible']:
        rt = make_runtime(conf, manifest, pool, budget, window['phase'], window['id'],
                          'selector', 'selection_codex', packed=packed)
        if rt.role_profile['full_window_selection']['reasoning_effort'] != 'xhigh' or rt.role_profile['full_window_selection']['model'] != SOL:
            raise IntegrityError('incumbent selector effort changed')
        with measure('selector'):
            response = invoke_call(rt, supply.selector_request(window, packed))
    else:
        response = None
    allocation = selection.allocate(window, source, response, conf['selection'], protected=window.get('protected', True)) if response else {
        'rows': [], 'targets': [], 'retained_ids': [], 'real_ranks': {},
        'reason': 'no_exact_eligible_candidates' if not packed['selection_eligible'] else 'selector_operational_failure',
        'eligibility_undetermined_ids': packed['payload']['proposal_ids'] if packed['selection_eligible'] else []}
    if window.get('supply_only'):
        from .common import ref
        from . import pairing
        if type(window.get('pairing_index')) is not int:
            raise IntegrityError('supply window lacks its frozen pairing index')
        allocation = pairing.expand(window, window['pairing_index'], allocation, seed=window['seed'])
        frozen(directory/'planner_pairing.private.json', allocation['planner_pairing'])
        frozen(directory/'eligible_queue.private.json', allocation)
        allocation = {**allocation, 'rows':[], 'source_queue':ref(directory/'eligible_queue.private.json'),
                      'supply_only':True, 'maximum_developments':0}
    frozen(directory / 'allocation.private.json', allocation)
    if on_ready and allocation['rows']:
        on_ready(window['id'])
    return allocation


def develop_window(root, window, allocation, *, pool, budget, production, manifest,
                   mock=False, invoke_call=invoke, events=None, on_failure=None):
    conf = window_config(root, window, production, mock=mock)
    rt = make_runtime(conf, manifest, pool, budget, window['phase'], window['id'], 'downstream', 'downstream')
    rt.on_failure = on_failure
    span = events.span('development', window=window['id'], account=window['account']) if events else nullcontext()
    def guarded_invoke(runtime, request):
        try:
            return invoke_call(runtime, request)
        except BaseException as exc:
            if on_failure and not isinstance(exc, DispatchPaused): on_failure(exc)
            raise
    with span:
        result = downstream.run(rt, window, allocation, production['portfolio_digest'], guarded_invoke)
    return {'runtime': rt, 'result': result}


def execute_window(root, window, requests, inputs, index, *, pool, budget, production,
                   manifest, mock=False, invoke_call=invoke):
    """Compatibility wrapper; the coordinator shares priority across windows."""
    from .scheduler import DownstreamPriority
    priority = DownstreamPriority.for_pool(pool)
    common = dict(pool=pool, budget=budget, production=production, manifest=manifest,
                  mock=mock, invoke_call=invoke_call)
    try:
        allocation = prepare_window(root, window, requests, inputs, index, on_ready=priority.add, **common)
        return develop_window(root, window, allocation, **common)['result']
    finally:
        priority.remove(window['id'])
